#if canImport(AVFoundation)
import AVFoundation
import os

/// Lock-free SPSC ring buffer for audio samples.
/// Producer (TTS thread) writes, consumer (audio render thread) reads.
public final class AudioSampleRingBuffer: @unchecked Sendable {
    private let buffer: UnsafeMutableBufferPointer<Float>
    private let capacity: Int
    private var writePos: Int = 0  // only written by producer
    private var readPos: Int = 0   // only written by consumer

    public init(capacity: Int) {
        self.capacity = capacity
        let ptr = UnsafeMutablePointer<Float>.allocate(capacity: capacity)
        ptr.initialize(repeating: 0, count: capacity)
        self.buffer = UnsafeMutableBufferPointer(start: ptr, count: capacity)
    }

    deinit {
        buffer.baseAddress?.deinitialize(count: capacity)
        buffer.baseAddress?.deallocate()
    }

    /// Number of samples available to read.
    public var availableToRead: Int {
        let w = writePos
        let r = readPos
        return w >= r ? w - r : capacity - r + w
    }

    /// Number of free slots for writing.
    public var availableToWrite: Int {
        return capacity - availableToRead - 1
    }

    /// Write samples into the buffer. Returns number actually written.
    @discardableResult
    public func write(_ samples: [Float]) -> Int {
        let count = min(samples.count, availableToWrite)
        guard count > 0 else { return 0 }

        samples.withUnsafeBufferPointer { src in
            let w = writePos
            let firstChunk = min(count, capacity - w)
            buffer.baseAddress!.advanced(by: w).update(from: src.baseAddress!, count: firstChunk)
            if firstChunk < count {
                buffer.baseAddress!.update(from: src.baseAddress!.advanced(by: firstChunk), count: count - firstChunk)
            }
        }
        writePos = (writePos + count) % capacity
        return count
    }

    /// Read samples from the buffer into dst. Returns number actually read.
    @discardableResult
    public func read(into dst: UnsafeMutablePointer<Float>, count: Int) -> Int {
        let available = min(count, availableToRead)
        guard available > 0 else { return 0 }

        let r = readPos
        let firstChunk = min(available, capacity - r)
        dst.update(from: buffer.baseAddress!.advanced(by: r), count: firstChunk)
        if firstChunk < available {
            dst.advanced(by: firstChunk).update(from: buffer.baseAddress!, count: available - firstChunk)
        }
        readPos = (readPos + available) % capacity
        return available
    }

    /// Reset both pointers (call when not actively reading/writing).
    public func reset() {
        readPos = 0
        writePos = 0
    }
}

/// Streams TTS audio via AVAudioEngine using an event-driven render callback.
///
/// Architecture:
/// ```
/// TTS (producer) → [Ring Buffer] → AVAudioSourceNode render callback (consumer)
///                   pre-fill N sec    hardware pulls when it needs data
/// ```
///
/// The render thread calls our callback when it needs audio. We read from the
/// ring buffer. If the buffer is empty (underflow), we output silence.
///
/// `preBufferDuration` controls how much audio must accumulate before playback
/// starts. This is the latency-quality tradeoff:
/// - Higher = more resilient to TTS jitter, but more latency
/// - Lower = less latency, but risk of underflow gaps
///
/// Typical values:
/// - 0s: single-pass TTS (Kokoro) where all audio arrives at once
/// - 2s: streaming TTS (Qwen3-TTS, RTF ~0.53)
public final class StreamingAudioPlayer: @unchecked Sendable {
    private var engine: AVAudioEngine?
    private var sourceNode: AVAudioSourceNode?
    private var format: AVAudioFormat?
    private let lock = NSLock()

    private var ringBuffer: AudioSampleRingBuffer?
    private var playbackStarted = false
    private var generationComplete = false
    private var isFirstChunk = true
    private var upsampler: AVAudioConverter?
    private var preBufferSamples: Int = 0
    public private(set) var totalWritten: Int = 0
    /// Number of samples written for external diagnostics.
    public var totalWrittenSamples: Int { totalWritten }
    private var totalRead: Int = 0

    public private(set) var isPlaying = false
    private var playbackFinishedFired = false

    /// Pre-buffer duration in seconds. Playback starts after this much audio accumulates.
    /// Default 1.0s — sufficient for streaming TTS at RTF < 0.6.
    public var preBufferDuration: Double = 1.0

    /// Callback when all audio has finished playing.
    public var onPlaybackFinished: (() -> Void)?

    /// Ring buffer capacity in seconds. Default 30s — enough for any TTS response.
    public var ringBufferDuration: Double = 30

    public init() {}

    // MARK: - Standalone mode

    /// Start playback engine at the given sample rate.
    public func start(sampleRate: Double = 24000) throws {
        stop()
        let eng = AVAudioEngine()
        guard let fmt = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else { return }

        setupSourceNode(engine: eng, format: fmt)
        try eng.start()
        self.engine = eng
        self.format = fmt
    }

    /// Create a standalone engine at the hardware's native sample rate.
    public func ensureStandaloneEngine() {
        guard sourceNode == nil else { return }
        let eng = AVAudioEngine()
        let mixerFormat = eng.mainMixerNode.outputFormat(forBus: 0)
        guard let monoFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: mixerFormat.sampleRate,
            channels: 1,
            interleaved: false
        ) else { return }
        setupSourceNode(engine: eng, format: monoFormat)
        do {
            try eng.start()
            self.engine = eng
            self.format = monoFormat
        } catch {}
    }

    // MARK: - Attached mode

    /// Attach to an existing AVAudioEngine.
    public func attach(to engine: AVAudioEngine, format: AVAudioFormat) {
        setupSourceNode(engine: engine, format: format)
        self.format = format
    }

    /// Start the source node (for use when attaching before engine.start()).
    public func startPlayback() {
        // Source node is always running once attached — no-op
    }

    /// Detach from an external engine.
    public func detach(from engine: AVAudioEngine) {
        if let node = sourceNode {
            engine.disconnectNodeOutput(node)
            engine.detach(node)
        }
        sourceNode = nil
        format = nil
        upsampler = nil
        ringBuffer?.reset()
    }

    // MARK: - Audio Scheduling

    /// Write a chunk of audio samples into the ring buffer.
    /// If pre-buffer threshold is reached, playback begins automatically.
    public func scheduleChunk(_ samples: [Float]) {
        guard !samples.isEmpty else { return }

        var output = samples

        // Drop near-silent warmup chunks at start of generation
        if isFirstChunk {
            var sumSq: Float = 0
            for s in samples { sumSq += s * s }
            let rms = sqrt(sumSq / Float(samples.count))
            if rms < 0.005 { return }  // Only drop near-silence (codec init noise)
            isFirstChunk = false
            // 5ms fade-in to prevent pop
            if let fmt = format {
                let fadeFrames = min(samples.count, Int(fmt.sampleRate * 0.005))
                for i in 0..<fadeFrames {
                    output[i] *= Float(i) / Float(fadeFrames)
                }
            }
        }

        lock.lock()
        ringBuffer?.write(output)
        totalWritten += output.count
        isPlaying = true

        if !playbackStarted && preBufferSamples > 0 {
            if (ringBuffer?.availableToRead ?? 0) >= preBufferSamples {
                playbackStarted = true
            }
        } else if preBufferSamples == 0 {
            playbackStarted = true
        }
        lock.unlock()
    }

    /// Write samples with resampling from sourceSampleRate to the player's rate.
    public func play(samples: [Float], sampleRate: Int) throws {
        guard let fmt = format else { return }
        if Double(sampleRate) == fmt.sampleRate {
            scheduleChunk(samples)
        } else {
            guard let srcFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false) else { return }
            if upsampler == nil || upsampler?.inputFormat.sampleRate != Double(sampleRate) {
                upsampler = AVAudioConverter(from: srcFmt, to: fmt)
            }
            guard let converter = upsampler else { return }
            guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: srcFmt, frameCapacity: AVAudioFrameCount(samples.count)) else { return }
            inputBuffer.frameLength = AVAudioFrameCount(samples.count)
            samples.withUnsafeBufferPointer { ptr in
                inputBuffer.floatChannelData![0].update(from: ptr.baseAddress!, count: samples.count)
            }
            let outFrameCount = AVAudioFrameCount(Double(samples.count) * fmt.sampleRate / Double(sampleRate))
            guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: outFrameCount) else { return }
            var consumed = false
            var error: NSError?
            converter.convert(to: outputBuffer, error: &error) { _, outStatus in
                if consumed { outStatus.pointee = .noDataNow; return nil }
                consumed = true
                outStatus.pointee = .haveData
                return inputBuffer
            }
            let count = Int(outputBuffer.frameLength)
            guard count > 0, let data = outputBuffer.floatChannelData else { return }
            let resampled = Array(UnsafeBufferPointer(start: data[0], count: count))
            scheduleChunk(resampled)
        }
    }

    // MARK: - Completion

    /// Signal that TTS generation is complete — no more chunks will arrive.
    /// The render callback will drain remaining samples, then fire onPlaybackFinished.
    public func markGenerationComplete() {
        lock.lock()
        generationComplete = true
        playbackStarted = true
        let hasEngine = sourceNode != nil
        let empty = (ringBuffer?.availableToRead ?? 0) == 0
        lock.unlock()

        // No engine or nothing was written — fire immediately
        if !hasEngine || (empty && totalWritten == 0) {
            guard !playbackFinishedFired else { return }
            playbackFinishedFired = true
            isPlaying = false
            onPlaybackFinished?()
        }
    }

    /// Reset for a new generation cycle.
    public func resetGeneration() {
        lock.lock()
        generationComplete = false
        playbackFinishedFired = false
        playbackStarted = false
        isFirstChunk = true
        totalWritten = 0
        totalRead = 0
        ringBuffer?.reset()
        lock.unlock()
    }

    /// Wait until all audio has finished playing.
    public func waitForCompletion() async {
        while isPlaying {
            try? await Task.sleep(nanoseconds: 50_000_000) // 50ms poll
        }
    }

    /// Stop immediately.
    public func fadeOutAndStop() {
        lock.lock()
        generationComplete = false
        playbackStarted = false
        isFirstChunk = true
        totalWritten = 0
        totalRead = 0
        ringBuffer?.reset()
        lock.unlock()
        isPlaying = false
    }

    /// Stop and release resources.
    public func stop() {
        if let eng = engine, let node = sourceNode {
            eng.disconnectNodeOutput(node)
            eng.detach(node)
        }
        engine?.stop()
        engine = nil
        sourceNode = nil
        format = nil
        upsampler = nil
        lock.lock()
        generationComplete = false
        playbackStarted = false
        isFirstChunk = true
        totalWritten = 0
        totalRead = 0
        ringBuffer?.reset()
        lock.unlock()
        isPlaying = false
    }

    // MARK: - Private

    private func setupSourceNode(engine: AVAudioEngine, format: AVAudioFormat) {
        let bufferCapacity = Int(format.sampleRate * ringBufferDuration)
        let rb = AudioSampleRingBuffer(capacity: bufferCapacity)
        self.ringBuffer = rb
        self.preBufferSamples = Int(format.sampleRate * preBufferDuration)

        let node = AVAudioSourceNode(format: format) { [weak self] _, _, frameCount, bufferList -> OSStatus in
            guard let self else { return noErr }

            let ablPointer = UnsafeMutableAudioBufferListPointer(bufferList)
            guard let dst = ablPointer[0].mData?.assumingMemoryBound(to: Float.self) else {
                return noErr
            }
            let frames = Int(frameCount)

            self.lock.lock()
            let started = self.playbackStarted
            let complete = self.generationComplete
            let available = rb.availableToRead
            self.lock.unlock()

            if !started {
                // Pre-buffer not full yet — output silence
                dst.update(repeating: 0, count: frames)
                return noErr
            }

            if available > 0 {
                let read = rb.read(into: dst, count: min(frames, available))
                // Zero-fill remainder if not enough
                if read < frames {
                    dst.advanced(by: read).update(repeating: 0, count: frames - read)
                }
                self.lock.lock()
                self.totalRead += read
                self.lock.unlock()
            } else if complete && !self.playbackFinishedFired {
                // Buffer empty + generation done = playback finished (fire once)
                self.playbackFinishedFired = true
                dst.update(repeating: 0, count: frames)
                DispatchQueue.main.async {
                    self.isPlaying = false
                    self.onPlaybackFinished?()
                }
            } else {
                // Underflow — output silence, keep waiting for more data
                dst.update(repeating: 0, count: frames)
            }

            return noErr
        }

        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
        self.sourceNode = node
    }

    /// Compress long silent gaps to at most `maxSilence` samples.
    /// TTS models produce long pauses between sentences (500ms+).
    /// This shortens them while keeping a natural brief pause.
    static func compressSilence(_ samples: [Float], maxSilence: Int, threshold: Float) -> [Float] {
        guard samples.count > maxSilence else { return samples }

        var result = [Float]()
        result.reserveCapacity(samples.count)
        var silenceRun = 0

        // Process in small frames (240 samples = 10ms at 24kHz)
        let frameSize = 240
        var offset = 0

        while offset < samples.count {
            let end = min(offset + frameSize, samples.count)
            let frame = samples[offset..<end]

            // Compute frame RMS
            var sumSq: Float = 0
            for s in frame { sumSq += s * s }
            let rms = sqrt(sumSq / Float(frame.count))

            if rms < threshold {
                silenceRun += frame.count
                // Only keep silence up to maxSilence
                if silenceRun <= maxSilence {
                    result.append(contentsOf: frame)
                }
                // Else: drop this frame (compress the silence)
            } else {
                silenceRun = 0
                result.append(contentsOf: frame)
            }
            offset = end
        }

        return result
    }
}
#endif
