import Foundation
import MLX
import MLXFast
import MLXNN

public class Qwen2Model: Module {
    public let config: Qwen2Configuration

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    public let layers: [Qwen2TransformerBlock]
    public let norm: RMSNorm

    public init(_ config: Qwen2Configuration) {
        self.config = config

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )

        var layersList: [Qwen2TransformerBlock] = []
        for _ in 0..<config.hiddenLayers {
            layersList.append(Qwen2TransformerBlock(config))
        }
        self.layers = layersList

        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    public func callAsFunction(
        _ inputIds: MLXArray,
        cache: [KVCacheSimple]? = nil
    ) -> MLXArray {
        var h = embedTokens(inputIds)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        if let cache = cache {
            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache[i])
            }
        } else {
            for layer in layers {
                h = layer(h, mask: mask, cache: nil)
            }
        }

        return norm(h)
    }

    public func forwardWithEmbeddings(
        _ embeddings: MLXArray,
        cache: [KVCacheSimple]? = nil,
        applyFinalNorm: Bool = true
    ) -> MLXArray {
        var h = embeddings

        let mask = createAttentionMask(h: h, cache: cache?.first)

        if let cache = cache {
            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache[i])
            }
        } else {
            for layer in layers {
                h = layer(h, mask: mask, cache: nil)
            }
        }

        if applyFinalNorm {
            return norm(h)
        } else {
            return h
        }
    }

    private func createAttentionMask(h: MLXArray, cache: KVCacheSimple?) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let n = h.dim(1)

        if n == 1 {
            return .none
        }

        let offset = cache?.offset ?? 0

        if offset == 0 {
            return .causal
        }

        let mask = createCausalMask(n: n, offset: offset)
        return .array(mask)
    }

    private func createCausalMask(n: Int, offset: Int) -> MLXArray {
        var rinds = MLXArray(Int32(0) ..< Int32(offset + n))
        var linds = MLXArray(Int32(offset) ..< Int32(offset + n))
        linds = linds[0..., .newAxis]
        rinds = rinds[.newAxis]
        let mask = linds .>= rinds
        return mask
    }

    public func newCache() -> [KVCacheSimple] {
        (0..<config.hiddenLayers).map { _ in KVCacheSimple() }
    }
}

public class Qwen2ForCausalLM: Module {
    public let config: Qwen2Configuration
    public let model: Qwen2Model

    @ModuleInfo(key: "lm_head") public var lmHead: Linear?

    public init(_ config: Qwen2Configuration) {
        self.config = config
        self.model = Qwen2Model(config)

        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }

        super.init()
    }

    public func callAsFunction(_ inputIds: MLXArray, cache: [KVCacheSimple]? = nil) -> MLXArray {
        var out = model(inputIds, cache: cache)

        if let lmHead = lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }

        return out
    }

    public func newCache() -> [KVCacheSimple] {
        model.newCache()
    }
}

extension Embedding {
    func asLinear(_ x: MLXArray) -> MLXArray {
        matmul(x, weight.T)
    }
}
