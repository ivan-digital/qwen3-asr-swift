import Foundation
import MLX
import MLXNN

/// Wav2Vec2 feature extractor: 7 strided convolutions over raw audio (channel
/// last `[B, T, C]`). Each layer is `Conv1d → LayerNorm(C) → GELU`. Total
/// downsample is 320× — at 16 kHz this produces a 50 Hz frame rate.
///
/// This matches fairseq2's `Wav2Vec2FeatureExtractor` in `layer_norm_features`
/// mode (weight names `feature_extractor.layers.{i}.conv.{weight,bias}` and
/// `feature_extractor.layers.{i}.layer_norm.{weight,bias}`).
public class Wav2Vec2FeatureExtractor: Module {
    @ModuleInfo public var layers: [Wav2Vec2ConvLayer]

    public let kernels: [Int]
    public let strides: [Int]

    public init(featureDim: Int, kernels: [Int], strides: [Int]) {
        precondition(kernels.count == strides.count, "kernels and strides must have the same count")
        self.kernels = kernels
        self.strides = strides

        var built: [Wav2Vec2ConvLayer] = []
        var inChannels = 1
        for i in 0..<kernels.count {
            built.append(Wav2Vec2ConvLayer(
                inputChannels: inChannels,
                outputChannels: featureDim,
                kernelSize: kernels[i],
                stride: strides[i]))
            inChannels = featureDim
        }
        self._layers.wrappedValue = built
        super.init()
    }

    /// Input: `[B, T, 1]` (channel last). Output: `[B, T', C]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h)
        }
        return h
    }

    /// Encoder output frames for a given input sample count, applying each
    /// strided conv's `floor((L - K) / S) + 1` rule.
    public func outputLength(for inputLength: Int) -> Int {
        var L = inputLength
        for i in 0..<kernels.count {
            L = ((L - kernels[i]) / strides[i]) + 1
            if L <= 0 { return 0 }
        }
        return L
    }
}

/// One feature-extractor layer: 1-D conv + per-channel LayerNorm + GELU.
public class Wav2Vec2ConvLayer: Module {
    @ModuleInfo public var conv: Conv1d
    @ModuleInfo(key: "layer_norm") public var layerNorm: LayerNorm

    public init(inputChannels: Int, outputChannels: Int, kernelSize: Int, stride: Int) {
        self._conv.wrappedValue = Conv1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            bias: true)
        self._layerNorm.wrappedValue = LayerNorm(dimensions: outputChannels)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv(x)
        h = layerNorm(h)
        h = gelu(h)
        return h
    }
}

/// Convolutional positional encoder used by fairseq2 wav2vec2: a single
/// grouped 1-D convolution with kernel 128 / groups 16, weight-normalised in
/// the original PyTorch model. The Swift loader fuses `weight_g`/`weight_v`
/// into a plain `weight` tensor at load time, so this module exposes a normal
/// `Conv1d`.
///
/// fairseq2 trims the last frame when the kernel is even (which it is — 128),
/// then applies GELU and a residual connection back to the input.
public class Wav2Vec2PositionEncoder: Module {
    @ModuleInfo public var conv: Conv1d
    public let kernel: Int

    public init(modelDim: Int, kernel: Int, groups: Int) {
        self.kernel = kernel
        // Padding = kernel/2 means the conv produces L+1 frames for an even
        // kernel; we trim the trailing one in `callAsFunction` to recover L.
        self._conv.wrappedValue = Conv1d(
            inputChannels: modelDim,
            outputChannels: modelDim,
            kernelSize: kernel,
            stride: 1,
            padding: kernel / 2,
            groups: groups,
            bias: true)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let length = x.dim(1)
        var h = conv(x)
        if kernel % 2 == 0 {
            h = h[0..., 0..<length, 0...]
        }
        return gelu(h) + x
    }
}

/// Full wav2vec2 frontend: feature extractor → post-extract LayerNorm →
/// `model_dim_proj` (Linear 512 → modelDim) → positional encoder → output
/// `[B, T', modelDim]` ready for the transformer encoder stack.
public class Wav2Vec2Frontend: Module {
    @ModuleInfo(key: "feature_extractor") public var featureExtractor: Wav2Vec2FeatureExtractor
    @ModuleInfo(key: "post_extract_layer_norm") public var postExtractLayerNorm: LayerNorm
    @ModuleInfo(key: "model_dim_proj") public var modelDimProj: Linear
    @ModuleInfo(key: "pos_encoder") public var posEncoder: Wav2Vec2PositionEncoder

    public let config: OmnilingualMLXConfig

    public init(config: OmnilingualMLXConfig) {
        self.config = config
        self._featureExtractor.wrappedValue = Wav2Vec2FeatureExtractor(
            featureDim: config.featureDim,
            kernels: config.convKernels,
            strides: config.convStrides)
        self._postExtractLayerNorm.wrappedValue = LayerNorm(dimensions: config.featureDim)
        self._modelDimProj.wrappedValue = Linear(config.featureDim, config.modelDim, bias: true)
        self._posEncoder.wrappedValue = Wav2Vec2PositionEncoder(
            modelDim: config.modelDim,
            kernel: config.posEncoderKernel,
            groups: config.posEncoderGroups)
        super.init()
    }

    /// Input: `[B, T, 1]` raw audio (already layer-normalised by the caller).
    /// Output: `[B, T', modelDim]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = featureExtractor(x)
        h = postExtractLayerNorm(h)
        h = modelDimProj(h)
        h = posEncoder(h)
        return h
    }
}
