import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - CSMModel

public final class CSMModel: Module {
    public let args: CSMModelArgs

    @ModuleInfo public var backbone: CSMLlamaModel
    @ModuleInfo public var decoder: CSMLlamaModel

    @ModuleInfo public var text_embeddings: Embedding
    @ModuleInfo public var audio_embeddings: Embedding
    @ModuleInfo public var projection: Module // backbone_dim → decoder_dim
    @ModuleInfo public var codebook0_head: Module // logits for codebook 0
    public var audio_head: MLXArray // [nq-1, decoder_dim, audio_vocab]

    public var backboneCache: [KVCacheSimple]? = nil
    public var decoderCache: [KVCacheSimple]? = nil
    public var cachesEnabled: Bool = false

    public init(config: CSMModelArgs, groupSize: Int? = nil, bits: Int? = nil) {
        self.args = config

        let backCfg = config.backboneConfiguration()
        let decCfg = config.decoderConfiguration()

        self._backbone = ModuleInfo(wrappedValue: CSMLlamaModel(backCfg, groupSize: groupSize, bits: bits))
        self._decoder = ModuleInfo(wrappedValue: CSMLlamaModel(decCfg, groupSize: groupSize, bits: bits))

        let backboneDim = backCfg.hiddenSize
        let decoderDim = decCfg.hiddenSize

        if let gs = groupSize, let b = bits {
            self._text_embeddings = ModuleInfo(wrappedValue: QuantizedEmbedding(embeddingCount: args.textVocabSize, dimensions: backboneDim, groupSize: gs, bits: b))
            let audioVocabCombined = args.audioVocabSize * args.audioNumCodebooks
            self._audio_embeddings = ModuleInfo(wrappedValue: QuantizedEmbedding(embeddingCount: audioVocabCombined, dimensions: backboneDim, groupSize: gs, bits: b))
        } else {
            self._text_embeddings = ModuleInfo(wrappedValue: Embedding(embeddingCount: args.textVocabSize, dimensions: backboneDim))
            let audioVocabCombined = args.audioVocabSize * args.audioNumCodebooks
            self._audio_embeddings = ModuleInfo(wrappedValue: Embedding(embeddingCount: audioVocabCombined, dimensions: backboneDim))
        }

        self._projection = ModuleInfo(wrappedValue: makeLinear(backboneDim, decoderDim, bias: false, groupSize: groupSize, bits: bits) as Module)
        self._codebook0_head = ModuleInfo(wrappedValue: makeLinear(backboneDim, args.audioVocabSize, bias: false, groupSize: groupSize, bits: bits) as Module)

        let restCodebooks = max(args.audioNumCodebooks - 1, 0)
        self.audio_head = MLXArray.zeros([restCodebooks, decoderDim, args.audioVocabSize])

        self.backboneCache = nil
        self.decoderCache = nil
        self.cachesEnabled = false
    }

    public func resetCaches() {
        backboneCache = backbone.makeCache()
        decoderCache = decoder.makeCache()
        cachesEnabled = true
    }

    public func generateFrame(
        maxCodebooks: Int?,
        tokens: MLXArray,
        tokensMask: MLXArray,
        inputPos: MLXArray,
        sampler: (MLXArray) -> MLXArray
    ) -> MLXArray {
        precondition(cachesEnabled, "backbone caches are not enabled")

        let embeds = _embedTokens(tokens) // [B, T, Cb+1, D]
        let masked = embeds * tokensMask.expandedDimensions(axis: -1)
        var h = sum(masked, axis: 2) // [B, T, D]

        h = backbone(h, cache: backboneCache) // [B, T, D]

        let B = h.shape[0]
        let dBackbone = h.shape[2]
        let lastT = h.shape[1] - 1
        let split1 = split(h, indices: [lastT], axis: 1)
        let lastSlice = split(split1[1], indices: [1], axis: 1)[0]
        let lastH = lastSlice.reshaped([B, dBackbone])

        let c0Logits = applyLinear(codebook0_head, lastH) // [B, vocab_audio]
        let c0SampleVec = sampler(c0Logits) // [B]
        let c0Sample = c0SampleVec.expandedDimensions(axis: -1) // [B, 1]
        let c0Embed = _embedAudio(codebook: 0, tokens: c0Sample) // [B, 1, D_backbone]

        let lastH3 = expandedDimensions(lastH, axis: 1) // [B, 1, D_backbone]
        var currH = concatenated([lastH3, c0Embed], axis: 1) // [B, 2, D_backbone]
        var currSample = c0Sample // [B, 1]
        var currPos = repeated(MLXArray(Array(0..<2)).reshaped([1, 2]), count: B, axis: 0)

        decoderCache = decoder.makeCache()

        let Cb = maxCodebooks != nil ? min(args.audioNumCodebooks, maxCodebooks!) : args.audioNumCodebooks
        if Cb > 1 {
            for i in 1..<Cb {
                let decH = decoder(applyLinear(projection, currH), cache: decoderCache) // [B, Tcur, D_dec]

                let dDec = decH.shape[2]
                let lastSplit1 = split(decH, indices: [decH.shape[1] - 1], axis: 1)
                let lastDec = split(lastSplit1[1], indices: [1], axis: 1)[0].reshaped([B, dDec])

                let Wi = take2DHead(audio_head, index: i - 1) // [decoder_dim, audio_vocab]
                let ciLogits = matmul(lastDec, Wi)

                let ciSampleVec = sampler(ciLogits)
                let ciSample = expandedDimensions(ciSampleVec, axis: -1)
                let ciEmbed = _embedAudio(codebook: i, tokens: ciSample)

                currH = ciEmbed
                currSample = concatenated([currSample, ciSample], axis: 1)
                currPos = split(currPos, indices: [1], axis: 1)[1] + MLXArray(1)
            }
        }

        return currSample // [B, Cb]
    }

    private func _embedAudio(codebook: Int, tokens: MLXArray) -> MLXArray {
        let offset = codebook * args.audioVocabSize
        let shifted = tokens + MLXArray(Int32(offset))
        return audio_embeddings(shifted)
    }

    private func _embedTokens(_ tokens: MLXArray) -> MLXArray {
        let B = tokens.shape[0], T = tokens.shape[1], CbPlus = tokens.shape[2]
        let Cb = CbPlus - 1

        let split1 = split(tokens, indices: [Cb], axis: 2)
        let audioIds = split1[0] // [B, T, Cb]
        let textIds = split(split1[1], indices: [1], axis: 2)[0].reshaped([B, T])

        var textEmb = text_embeddings(textIds) // [B, T, D]
        textEmb = expandedDimensions(textEmb, axis: -2) // [B, T, 1, D]

        let cbIdx = MLXArray(Array(0..<Cb).map { Int32($0) })
        let cbOffsets = (cbIdx * MLXArray(Int32(args.audioVocabSize))).reshaped([1, 1, Cb])
        let shiftedAudioIds = audioIds + cbOffsets

        let flat = shiftedAudioIds.flattened()
        let audioFlatEmb = audio_embeddings(flat)
        let D = audioFlatEmb.shape[1]
        let audioEmb = audioFlatEmb.reshaped([B, T, Cb, D])

        return concatenated([audioEmb, textEmb], axis: 2)
    }

    private func take2DHead(_ W: MLXArray, index i: Int) -> MLXArray {
        if W.ndim == 3 {
            let left = split(W, indices: [i], axis: 0)
            let tail = split(left[1], indices: [1], axis: 0)
            return tail[0].reshaped([W.shape[1], W.shape[2]])
        }
        return W
    }
}
