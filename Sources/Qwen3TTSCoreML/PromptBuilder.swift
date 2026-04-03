#if canImport(CoreML)
import CoreML
import Foundation
import AudioCommon

/// Builds the non-streaming prefill embedding sequence using CoreML embedder models.
///
/// All text tokens are in the prefill. Decode loop feeds codec_sum + tts_pad only.
///
/// Prefill layout:
/// ```
/// [0:3]     role:    TextProjector(im_start, assistant, \n)
/// [3:7]     ctrl:    tts_pad + CodeEmbedder(think, think_bos, lang, think_eos)
/// [7]       speaker: tts_pad + speaker_embedding
/// [8]       ctrl:    tts_bos + CodeEmbedder(codec_pad)
/// [9:+N]    text:    TextProjector(token) + CodeEmbedder(codec_pad)
/// [+N+1]    eos:     tts_eos + CodeEmbedder(codec_pad)
/// [+N+2]    final:   tts_pad + CodeEmbedder(codec_bos)
/// ```
struct PromptBuilder {

    static let imStartId = 151644
    static let imEndId = 151645
    static let newlineId = 198
    static let assistantId = 77091

    static let languageIds: [String: Int] = [
        "chinese": 2055, "english": 2050, "german": 2053,
        "italian": 2070, "portuguese": 2071, "spanish": 2054,
        "japanese": 2058, "korean": 2064, "french": 2061, "russian": 2069,
    ]

    /// Build prefill embeddings using CoreML models.
    static func build(
        text: String,
        language: String,
        tokenizer: Qwen3Tokenizer,
        textProjector: TextProjectorModel,
        codeEmbedder: CodeEmbedderModel,
        ttsPadEmbed: MLMultiArray,
        ttsBosEmbed: MLMultiArray,
        ttsEosEmbed: MLMultiArray,
        speakerEmbedding: MLMultiArray?
    ) throws -> [MLMultiArray] {
        let textTokens = prepareTextTokens(text: text, tokenizer: tokenizer)
        let roleIds = Array(textTokens[0..<3])
        let textIds = Array(textTokens[3..<(textTokens.count - 5)])
        let langId = languageIds[language.lowercased()] ?? languageIds["english"]!

        var prefill = [MLMultiArray]()

        // [0:3] Role: TextProjector only
        for tid in roleIds {
            prefill.append(ensureNCHW(try textProjector.embed(tid), channels: 1024))
        }

        // [3:7] Control: tts_pad + CodeEmbedder(think tokens)
        for ctok in [2154, 2156, langId, 2157] {
            let ce = ensureNCHW(try codeEmbedder.embed(ctok), channels: 1024)
            prefill.append(addMLMultiArrays(ttsPadEmbed, ce))
        }

        // [7] Speaker embedding: tts_pad + speaker_embed
        if let spk = speakerEmbedding {
            prefill.append(addMLMultiArrays(ttsPadEmbed, spk))
        }

        // Control: tts_bos + CodeEmbedder(codec_pad)
        let codecPadEmbed = ensureNCHW(try codeEmbedder.embed(2148), channels: 1024)
        prefill.append(addMLMultiArrays(ttsBosEmbed, codecPadEmbed))

        // Text: TextProjector(token) + CodeEmbedder(codec_pad)
        for tid in textIds {
            let tp = ensureNCHW(try textProjector.embed(tid), channels: 1024)
            prefill.append(addMLMultiArrays(tp, codecPadEmbed))
        }

        // EOS: tts_eos + codec_pad
        prefill.append(addMLMultiArrays(ttsEosEmbed, codecPadEmbed))

        // Final: tts_pad + CodeEmbedder(codec_bos)
        let codecBosEmbed = ensureNCHW(try codeEmbedder.embed(2149), channels: 1024)
        prefill.append(addMLMultiArrays(ttsPadEmbed, codecBosEmbed))

        return prefill
    }

    private static func prepareTextTokens(text: String, tokenizer: Qwen3Tokenizer) -> [Int] {
        var tokens = [Int]()
        tokens.append(contentsOf: [imStartId, assistantId, newlineId])
        let encoded = tokenizer.encode(text)
        tokens.append(contentsOf: encoded)
        tokens.append(contentsOf: [imEndId, newlineId, imStartId, assistantId, newlineId])
        return tokens
    }
}
#endif
