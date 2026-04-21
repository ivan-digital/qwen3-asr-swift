import Foundation
import MLX

/// Content-addressed cache for voice-cloning reference audio artifacts.
///
/// Stores the ECAPA-TDNN speaker embedding and (optionally) the ICL codec-encoder
/// tokens produced from a reference waveform, keyed by the content hash of the
/// raw samples and the associated sample rate. Subsequent synthesize calls using
/// the same reference audio reuse the cached artifacts instead of re-running
/// mel extraction, the speaker encoder, or the codec encoder.
///
/// The cache is bounded (LRU) to keep memory predictable when many distinct
/// references are used in sequence.
final class ReferenceAudioCache {

    private struct Entry {
        let key: UInt64
        let sampleRate: Int
        var speakerEmbed: MLXArray?
        var codecRefCodes: MLXArray?
    }

    private var entries: [Entry] = []
    private let capacity: Int

    init(capacity: Int = 4) {
        precondition(capacity > 0, "capacity must be positive")
        self.capacity = capacity
    }

    // MARK: - Lookup

    func speakerEmbed(for audio: [Float], sampleRate: Int) -> MLXArray? {
        entry(for: audio, sampleRate: sampleRate)?.speakerEmbed
    }

    func codecRefCodes(for audio: [Float], sampleRate: Int) -> MLXArray? {
        entry(for: audio, sampleRate: sampleRate)?.codecRefCodes
    }

    // MARK: - Store

    func storeSpeakerEmbed(_ embed: MLXArray, audio: [Float], sampleRate: Int) {
        upsert(audio: audio, sampleRate: sampleRate) { $0.speakerEmbed = embed }
    }

    func storeCodecRefCodes(_ codes: MLXArray, audio: [Float], sampleRate: Int) {
        upsert(audio: audio, sampleRate: sampleRate) { $0.codecRefCodes = codes }
    }

    // MARK: - Management

    func clear() {
        entries.removeAll()
    }

    var count: Int { entries.count }

    // MARK: - Internals

    private func entry(for audio: [Float], sampleRate: Int) -> Entry? {
        let k = Self.hash(audio)
        guard let idx = entries.firstIndex(where: { $0.key == k && $0.sampleRate == sampleRate })
        else { return nil }
        let e = entries.remove(at: idx)
        entries.append(e)
        return e
    }

    private func upsert(audio: [Float], sampleRate: Int, update: (inout Entry) -> Void) {
        let k = Self.hash(audio)
        if let idx = entries.firstIndex(where: { $0.key == k && $0.sampleRate == sampleRate }) {
            var e = entries.remove(at: idx)
            update(&e)
            entries.append(e)
            return
        }
        var e = Entry(key: k, sampleRate: sampleRate, speakerEmbed: nil, codecRefCodes: nil)
        update(&e)
        if entries.count >= capacity { entries.removeFirst() }
        entries.append(e)
    }

    static func hash(_ audio: [Float]) -> UInt64 {
        var hasher = Hasher()
        hasher.combine(audio.count)
        for v in audio { hasher.combine(v.bitPattern) }
        return UInt64(bitPattern: Int64(hasher.finalize()))
    }
}
