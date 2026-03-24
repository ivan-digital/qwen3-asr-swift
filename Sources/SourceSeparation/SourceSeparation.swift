import MLX
import MLXNN
import Foundation
import AudioCommon

/// Music source separation: split a stereo mix into stems (vocals, drums, bass, other).
///
/// Uses Open-Unmix (UMX-HQ) — 4 independent BiLSTM models, one per target.
///
/// ```swift
/// let separator = try await SourceSeparator.fromPretrained()
/// let stems = separator.separate(audio: stereoAudio, sampleRate: 44100)
/// // stems.vocals, stems.drums, stems.bass, stems.other — each [2, N]
/// ```
public final class SourceSeparator {

    public static let defaultModelId = "aufklarer/OpenUnmix-HQ-MLX"

    public let config: OpenUnmixConfig
    private let models: [SeparationTarget: OpenUnmixStemModel]
    private let stft: STFTProcessor

    public init(config: OpenUnmixConfig, models: [SeparationTarget: OpenUnmixStemModel]) {
        self.config = config
        self.models = models
        self.stft = STFTProcessor(nFFT: config.nFFT, nHop: config.nHop)
    }

    /// Separate a stereo audio mix into stems.
    ///
    /// - Parameters:
    ///   - audio: Stereo audio as `[[Float]]` — `audio[0]` = left, `audio[1]` = right
    ///   - sampleRate: Input sample rate (resampled to 44100 if different)
    ///   - targets: Which stems to extract (default: all 4)
    /// - Returns: Dictionary of target → stereo audio `[[Float]]`
    public func separate(
        audio: [[Float]],
        sampleRate: Int = 44100,
        targets: [SeparationTarget] = SeparationTarget.allCases
    ) -> [SeparationTarget: [[Float]]] {
        guard audio.count == 2 else {
            // Mono → duplicate to stereo
            let mono = audio.first ?? []
            return separate(audio: [mono, mono], sampleRate: sampleRate, targets: targets)
        }

        let left = audio[0]
        let right = audio[1]
        let length = left.count

        // STFT each channel
        let (leftReal, leftImag) = stft.forward(left)
        let (rightReal, rightImag) = stft.forward(right)

        let T = leftReal.count
        let leftMag = stft.magnitude(real: leftReal, imag: leftImag)
        let rightMag = stft.magnitude(real: rightReal, imag: rightImag)

        // Stack into [T, 2, 2049] for the model
        var inputMag = [[Float]](repeating: [Float](repeating: 0, count: 2 * stft.nBins), count: T)
        for t in 0..<T {
            for f in 0..<stft.nBins {
                inputMag[t][f] = leftMag[t][f]
                inputMag[t][stft.nBins + f] = rightMag[t][f]
            }
        }

        var results: [SeparationTarget: [[Float]]] = [:]

        for target in targets {
            guard let model = models[target] else { continue }

            // Convert to MLXArray [T, 2, 2049]
            let flatInput = inputMag.flatMap { $0 }
            let mlxInput = MLXArray(flatInput, [T, 2, stft.nBins])

            // Forward pass
            let output = model(mlxInput)
            eval(output)

            // Extract [T, 2, 2049] back to per-channel magnitude
            let outputFlat = output.asArray(Float.self)
            var leftOutMag = [[Float]](repeating: [Float](repeating: 0, count: stft.nBins), count: T)
            var rightOutMag = [[Float]](repeating: [Float](repeating: 0, count: stft.nBins), count: T)

            for t in 0..<T {
                for f in 0..<stft.nBins {
                    leftOutMag[t][f] = outputFlat[t * 2 * stft.nBins + f]
                    rightOutMag[t][f] = outputFlat[t * 2 * stft.nBins + stft.nBins + f]
                }
            }

            // Apply phase from original mix and iSTFT
            let leftStem = stft.applyMaskAndInvert(
                maskedMag: leftOutMag, origReal: leftReal, origImag: leftImag, length: length)
            let rightStem = stft.applyMaskAndInvert(
                maskedMag: rightOutMag, origReal: rightReal, origImag: rightImag, length: length)

            results[target] = [leftStem, rightStem]
        }

        return results
    }

    /// Load pretrained Open-Unmix model from HuggingFace.
    public static func fromPretrained(
        modelId: String = defaultModelId,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> SourceSeparator {
        let config = OpenUnmixConfig.umxhq

        let cacheDir: URL
        do {
            cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId, reason: "Cache directory error", underlying: error)
        }

        // Download weights
        progressHandler?(0.0, "Downloading model...")
        do {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: modelId,
                to: cacheDir,
                additionalFiles: [
                    "vocals.safetensors",
                    "drums.safetensors",
                    "bass.safetensors",
                    "other.safetensors",
                    "config.json",
                ]
            ) { fraction in
                progressHandler?(fraction * 0.7, "Downloading model...")
            }
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId, reason: "Download failed", underlying: error)
        }

        // Load each target
        var models: [SeparationTarget: OpenUnmixStemModel] = [:]
        let targets: [SeparationTarget] = [.vocals, .drums, .bass, .other]

        for (i, target) in targets.enumerated() {
            let progress = 0.7 + Double(i) * 0.075
            progressHandler?(progress, "Loading \(target.rawValue)...")

            let weightsURL = cacheDir.appendingPathComponent("\(target.rawValue).safetensors")
            guard FileManager.default.fileExists(atPath: weightsURL.path) else {
                throw AudioModelError.modelLoadFailed(
                    modelId: modelId, reason: "\(target.rawValue).safetensors not found")
            }

            let model = OpenUnmixStemModel(hiddenSize: config.hiddenSize)
            let weights = try MLX.loadArrays(url: weightsURL)
            let params = ModuleParameters.unflattened(weights)
            try model.update(parameters: params)
            models[target] = model
        }

        progressHandler?(1.0, "Ready")
        return SourceSeparator(config: config, models: models)
    }
}
