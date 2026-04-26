import Foundation
import Tokenizers

/// Tokenizer wrapper for MADLAD-400.
///
/// Wraps a HuggingFace `Tokenizer` (loaded from `tokenizer.json`) and adds
/// the MADLAD-specific language-token convention:
///
/// ```
/// encode(text, target: "es")  →  [<2es>, ...text pieces..., </s>]
/// ```
///
/// MADLAD's SentencePiece vocab includes 400+ user-defined `<2xx>` language
/// tokens (one per supported target language). Prepending `<2{lang}>` is how
/// you tell the encoder which language to translate *into*. Source language
/// is auto-detected from the input text — there is no source-language token.
public final class MADLADTokenizer: @unchecked Sendable {
    public let tokenizer: Tokenizer
    public let eosTokenId: Int
    public let padTokenId: Int

    /// Cached `<2xx>` token ids, keyed by language code (e.g. `"es"`, `"zh"`).
    private var languageTokenCache: [String: Int] = [:]
    private let cacheLock = NSLock()

    public init(tokenizer: Tokenizer, eosTokenId: Int = 1, padTokenId: Int = 0) {
        self.tokenizer = tokenizer
        self.eosTokenId = eosTokenId
        self.padTokenId = padTokenId
    }

    /// Load tokenizer from a directory containing `tokenizer.json`.
    ///
    /// Pass `eosTokenId` / `padTokenId` from the model's `config.json` so the
    /// wrapper strips/appends the same termination ids the model was trained
    /// to produce. MADLAD uses `</s>` = 2 and `<pad>` = 1 (NOT the T5 default
    /// `</s>` = 1, `<pad>` = 0), so leaving these on defaults silently mangles
    /// the encoded source sequence.
    public static func load(
        from directory: URL,
        eosTokenId: Int = 2,
        padTokenId: Int = 1
    ) async throws -> MADLADTokenizer {
        let tk: Tokenizer
        do {
            tk = try await AutoTokenizer.from(modelFolder: directory)
        } catch {
            throw MADLADTranslationError.tokenizerLoadFailed(
                "AutoTokenizer.from failed at \(directory.path): \(error)")
        }
        return MADLADTokenizer(tokenizer: tk, eosTokenId: eosTokenId, padTokenId: padTokenId)
    }

    /// Encode source text for translation into `targetLanguage`.
    ///
    /// Returns: `[▁, <2{targetLanguage}>, ...text pieces..., </s>]`.
    ///
    /// Matches HF T5Tokenizer behavior on `f"<2{lang}> {text}"`: SentencePiece
    /// prepends a sequence-start `▁` (U+2581) token, then the language token
    /// (no `▁` prefix because it's user-defined), then the regular encoded
    /// text (where the first piece already carries its own `▁`), then `</s>`.
    /// MADLAD is trained to expect this exact prefix — without the leading
    /// `▁` the model produces degenerate / repetitive output.
    public func encode(text: String, targetLanguage: String) throws -> [Int] {
        let langTokenId = try languageTokenId(for: targetLanguage)
        let underscoreId = tokenizer.convertTokenToId("\u{2581}")  // ▁
        var ids = tokenizer.encode(text: text)
        // HF tokenizer for T5 typically already appends </s>. Strip it if so —
        // we control EOS placement explicitly to keep the format predictable.
        if ids.last == eosTokenId { ids.removeLast() }
        var prefix: [Int] = []
        if let u = underscoreId { prefix.append(u) }
        prefix.append(langTokenId)
        return prefix + ids + [eosTokenId]
    }

    /// Decode generated token ids back to text.
    ///
    /// Strips `<pad>`, `</s>`, and any leading `<2xx>` tokens.
    public func decode(_ ids: [Int]) -> String {
        let filtered = ids.filter { $0 != eosTokenId && $0 != padTokenId }
        return tokenizer.decode(tokens: filtered)
    }

    /// Decode a single token (for streaming output).
    public func decodeToken(_ id: Int) -> String? {
        if id == eosTokenId || id == padTokenId { return nil }
        return tokenizer.decode(tokens: [id])
    }

    /// Look up the `<2{language}>` user-defined token id, with caching.
    ///
    /// MADLAD's `<2xx>` tokens are stored in the Unigram vocab with score 0.0
    /// but are NOT registered in `added_tokens`, so calling `tokenizer.encode`
    /// on the literal string runs the Unigram algorithm and splits the token
    /// into sub-pieces (because zero-score loses to the combined score of
    /// `<`, `2`, `es`, `>`). Direct vocab lookup via `convertTokenToId`
    /// bypasses the tokenization algorithm and gets the correct id.
    ///
    /// `convertTokenToId` may also silently return the unknown-token id for
    /// strings not in the vocab, so we round-trip back through `convertIdToToken`
    /// and reject any mismatch as `unsupportedLanguage`.
    public func languageTokenId(for language: String) throws -> Int {
        cacheLock.lock()
        if let cached = languageTokenCache[language] {
            cacheLock.unlock()
            return cached
        }
        cacheLock.unlock()

        let tokenString = "<2\(language)>"
        guard let id = tokenizer.convertTokenToId(tokenString),
              tokenizer.convertIdToToken(id) == tokenString else {
            throw MADLADTranslationError.unsupportedLanguage(language)
        }

        cacheLock.lock()
        languageTokenCache[language] = id
        cacheLock.unlock()
        return id
    }
}
