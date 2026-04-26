import Foundation
import MLX
import MLXNN

/// Encoder-only semantic tokenizer. The semantic encoder mirrors the acoustic
/// encoder's `TokenizerEncoder` (same conv stack, same depths/ratios), only
/// the output dim differs (`vae_dim=128` for semantic, `vae_dim=64` for
/// acoustic). There is no decoder and no VAE sampling head — the semantic
/// stream is a raw deterministic representation, used solely for conditioning.
public class VibeVoiceSemanticTokenizer: Module {
    public let config: AcousticTokenizerConfiguration
    @ModuleInfo(key: "encoder") public var encoder: TokenizerEncoder

    public init(_ config: AcousticTokenizerConfiguration) {
        self.config = config
        _encoder.wrappedValue = TokenizerEncoder(config)
        super.init()
    }

    /// Encode audio to semantic latents. Output shape: `[B, T, vae_dim]`.
    public func encode(_ audio: MLXArray) -> MLXArray {
        var x = audio
        if x.ndim == 2 {
            x = expandedDimensions(x, axis: 1)
        }
        let latents = encoder(x)
        return latents.transposed(0, 2, 1)
    }
}
