#if canImport(CoreML)
import CoreML
import Foundation
import AudioCommon

/// Builds the prefill embedding sequence for Qwen3-TTS CoreML inference.
///
/// Constructs the combined text+codec embeddings following the ChatML prompt format:
/// ```
/// [role_embed(3)] [text_overlay+codec_prefix(codecLen-1)] [first_text+codec_bos(1)]
/// ```
/// Trailing text embeddings are returned separately for per-step decode.
struct PromptBuilder {

    // TTS special text token IDs
    static let imStartId = 151644
    static let imEndId = 151645
    static let newlineId = 198
    static let assistantId = 77091
    static let ttsPadId = 151671
    static let ttsBosId = 151672
    static let ttsEosId = 151673

    // Language → codec token ID
    static let languageIds: [String: Int] = [
        "chinese": 2055, "english": 2050, "german": 2053,
        "italian": 2070, "portuguese": 2071, "spanish": 2054,
        "japanese": 2058, "korean": 2064, "french": 2061, "russian": 2069,
    ]

    /// Build all embeddings needed for TTS generation.
    ///
    /// - Returns:
    ///   - prefillEmbeds: MLMultiArray [1, prefillLen, hiddenSize] for Talker prefill
    ///   - trailingTextEmbeds: Array of per-step text embeddings for decode phase
    ///   - ttsPadEmbed: [hiddenSize] padding embedding for when text runs out
    static func build(
        text: String,
        language: String,
        tokenizer: Qwen3Tokenizer,
        embeddings: EmbeddingManager
    ) -> (prefillEmbeds: MLMultiArray, trailingTextEmbeds: [[Float16]], ttsPadEmbed: [Float16], codecPrefix: [Int32]) {
        let hiddenSize = embeddings.hiddenSize

        // 1. Tokenize text in ChatML format
        let textTokens = prepareTextTokens(text: text, tokenizer: tokenizer)

        // 2. Build codec prefix: [think, think_bos, lang_id, think_eos, pad, bos]
        let langId = languageIds[language.lowercased()] ?? languageIds["english"]!
        let codecPrefix = buildCodecPrefix(languageId: langId)
        let codecLen = codecPrefix.count  // 6

        // 3. Compute all text embeddings (with projection)
        let textEmbeds = textTokens.map { embeddings.textEmbed($0) }

        // 4. Compute codec prefix embeddings
        let codecEmbeds = codecPrefix.map { embeddings.codecEmbed(Int($0)) }

        // 5. TTS special embeddings
        let ttsPadEmbed = embeddings.textEmbed(ttsPadId)
        let ttsBosEmbed = embeddings.textEmbed(ttsBosId)
        let ttsEosEmbed = embeddings.textEmbed(ttsEosId)

        // 6. Build prefill sequence
        // Role: first 3 text tokens (<|im_start|>assistant\n)
        let roleEmbeds = Array(textEmbeds[0..<3])

        // Text overlay for codec prefix: [pad, pad, pad, pad, tts_bos]
        let padCount = codecLen - 2
        var overlayEmbeds = [[Float16]]()
        for _ in 0..<padCount { overlayEmbeds.append(ttsPadEmbed) }
        overlayEmbeds.append(ttsBosEmbed)

        // Combined: overlay + codec[:-1] element-wise
        var combinedEmbeds = [[Float16]]()
        for i in 0..<(codecLen - 1) {
            var combined = [Float16](repeating: 0, count: hiddenSize)
            for j in 0..<hiddenSize {
                combined[j] = Float16(Float(overlayEmbeds[i][j]) + Float(codecEmbeds[i][j]))
            }
            combinedEmbeds.append(combined)
        }

        // First text token (index 3) + last codec token (bos)
        var firstTextPlusCodec = [Float16](repeating: 0, count: hiddenSize)
        let firstTextEmbed = textEmbeds[3]
        let lastCodecEmbed = codecEmbeds[codecLen - 1]
        for j in 0..<hiddenSize {
            firstTextPlusCodec[j] = Float16(Float(firstTextEmbed[j]) + Float(lastCodecEmbed[j]))
        }

        // Assemble prefill: [role(3), combined(codecLen-1), first_text+codec(1)]
        var prefillSequence = [[Float16]]()
        prefillSequence.append(contentsOf: roleEmbeds)
        prefillSequence.append(contentsOf: combinedEmbeds)
        prefillSequence.append(firstTextPlusCodec)

        let prefillLen = prefillSequence.count

        // Convert to MLMultiArray [1, prefillLen, hiddenSize]
        let prefillArray = try! MLMultiArray(
            shape: [1, NSNumber(value: prefillLen), NSNumber(value: hiddenSize)],
            dataType: .float16)
        let ptr = prefillArray.dataPointer.assumingMemoryBound(to: Float16.self)
        for t in 0..<prefillLen {
            for j in 0..<hiddenSize {
                ptr[t * hiddenSize + j] = prefillSequence[t][j]
            }
        }

        // 7. Trailing text: tokens[4..<(count-5)] + ttsEos
        // These are fed one per decode step after the prefill
        var trailing = [[Float16]]()
        if textTokens.count > 9 {
            for i in 4..<(textTokens.count - 5) {
                trailing.append(textEmbeds[i])
            }
        }
        trailing.append(ttsEosEmbed)

        return (prefillArray, trailing, ttsPadEmbed, codecPrefix)
    }

    // MARK: - Helpers

    private static func prepareTextTokens(text: String, tokenizer: Qwen3Tokenizer) -> [Int] {
        var tokens: [Int] = []
        // <|im_start|>assistant\n
        tokens.append(contentsOf: [imStartId, assistantId, newlineId])
        // Encoded text
        tokens.append(contentsOf: tokenizer.encode(text))
        // <|im_end|>\n<|im_start|>assistant\n
        tokens.append(contentsOf: [imEndId, newlineId, imStartId, assistantId, newlineId])
        return tokens
    }

    private static func buildCodecPrefix(languageId: Int) -> [Int32] {
        [
            Int32(2154),  // codec_think
            Int32(2156),  // codec_think_bos
            Int32(languageId),
            Int32(2157),  // codec_think_eos
            Int32(2148),  // codec_pad
            Int32(2149),  // codec_bos
        ]
    }
}
#endif
