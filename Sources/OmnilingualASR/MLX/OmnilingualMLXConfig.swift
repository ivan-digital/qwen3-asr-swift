import Foundation

/// Variant + quantization config for the MLX backend of Omnilingual ASR.
///
/// All published MLX repos share an identical fairseq2 wav2vec2 architecture
/// (CNN feature extractor → weight-normed conv positional encoder → N pre-norm
/// transformer encoder layers → final layer norm → linear CTC head). Only the
/// model dimension, FFN dimension, head count, and layer count vary across
/// 300M / 1B / 3B / 7B variants. Quantization is mlx-swift `QuantizedLinear`
/// format with `groupSize = 64` and `bits ∈ {4, 8}`.
public struct OmnilingualMLXConfig: Sendable, Equatable {
    /// Identifying variant size.
    public enum Variant: String, Sendable, CaseIterable {
        case m300 = "300M"
        case b1 = "1B"
        case b3 = "3B"
        case b7 = "7B"
    }

    public let variant: Variant

    // Encoder dims
    public let modelDim: Int
    public let numLayers: Int
    public let numHeads: Int
    public let ffnDim: Int

    // Frontend (always the same across variants — fairseq2 wav2vec2 7-layer conv stack)
    public let featureDim: Int          // 512
    public let convKernels: [Int]       // [10, 3, 3, 3, 3, 2, 2]
    public let convStrides: [Int]       // [5, 2, 2, 2, 2, 2, 2]

    // Position encoder
    public let posEncoderKernel: Int    // 128
    public let posEncoderGroups: Int    // 16

    // CTC head
    public let vocabSize: Int           // 10288 (v2 tokenizer)

    // Quantization
    public let groupSize: Int
    public let bits: Int

    // Audio
    public let sampleRate: Int          // 16000
    public let layerNormEps: Float      // 1e-5

    public init(
        variant: Variant,
        modelDim: Int,
        numLayers: Int,
        numHeads: Int,
        ffnDim: Int,
        groupSize: Int = 64,
        bits: Int = 4,
        featureDim: Int = 512,
        convKernels: [Int] = [10, 3, 3, 3, 3, 2, 2],
        convStrides: [Int] = [5, 2, 2, 2, 2, 2, 2],
        posEncoderKernel: Int = 128,
        posEncoderGroups: Int = 16,
        vocabSize: Int = 10288,
        sampleRate: Int = 16000,
        layerNormEps: Float = 1e-5
    ) {
        self.variant = variant
        self.modelDim = modelDim
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.ffnDim = ffnDim
        self.groupSize = groupSize
        self.bits = bits
        self.featureDim = featureDim
        self.convKernels = convKernels
        self.convStrides = convStrides
        self.posEncoderKernel = posEncoderKernel
        self.posEncoderGroups = posEncoderGroups
        self.vocabSize = vocabSize
        self.sampleRate = sampleRate
        self.layerNormEps = layerNormEps
    }

    public var headDim: Int { modelDim / numHeads }

    /// Encoder output frame stride relative to input samples (= product of conv strides).
    /// Always 320 for the standard wav2vec2 frontend → 50 Hz frames at 16 kHz.
    public var encoderStride: Int { convStrides.reduce(1, *) }

    public static func variant(_ v: Variant, bits: Int = 4) -> OmnilingualMLXConfig {
        switch v {
        case .m300:
            return OmnilingualMLXConfig(
                variant: .m300, modelDim: 1024, numLayers: 24, numHeads: 16, ffnDim: 4096, bits: bits)
        case .b1:
            return OmnilingualMLXConfig(
                variant: .b1, modelDim: 1280, numLayers: 48, numHeads: 20, ffnDim: 5120, bits: bits)
        case .b3:
            return OmnilingualMLXConfig(
                variant: .b3, modelDim: 2048, numLayers: 60, numHeads: 32, ffnDim: 8192, bits: bits)
        case .b7:
            return OmnilingualMLXConfig(
                variant: .b7, modelDim: 2048, numLayers: 128, numHeads: 32, ffnDim: 8192, bits: bits)
        }
    }

    /// Resolve a HuggingFace model id from a variant + bits.
    public static func defaultModelId(variant: Variant, bits: Int) -> String {
        let bitsStr = bits == 4 ? "4bit" : "8bit"
        return "aufklarer/Omnilingual-ASR-CTC-\(variant.rawValue)-MLX-\(bitsStr)"
    }
}
