import Foundation
import NaturalLanguage
import AudioCommon

/// Result of preprocessing text for forced alignment
public struct SlottedText: Sendable {
    /// Token IDs with timestamp tokens inserted around each word
    public let tokenIds: [Int]
    /// Indices within tokenIds that are timestamp tokens
    public let timestampPositions: [Int]
    /// The original words (one per timestamp pair)
    public let words: [String]
}

/// Language-specific text preprocessing for forced alignment.
///
/// Three paths:
/// - Japanese → morpheme-level segmentation
/// - Korean   → word-level segmentation
/// - Other    → whitespace split + per-Han-ideograph break
///
/// Tokens are filtered to keep only Unicode Letters / Numbers and the ASCII
/// apostrophe — punctuation, symbols, and marks are stripped before
/// timestamp slots are inserted. Han-ideograph splitting is restricted to
/// CJK Unified + Extensions A–E + Compatibility; hiragana, katakana, and
/// Hangul are NOT split per character (the model emits timestamp slots
/// between morphemes for those scripts, not between every kana or jamo).
public enum TextPreprocessor {

    /// Split text into words and insert timestamp slots for alignment.
    ///
    /// For each word, inserts `<timestamp><timestamp>` pairs so the model
    /// can predict start/end timestamps at those positions.
    public static func prepareForAlignment(
        text: String,
        tokenizer: Qwen3Tokenizer,
        language: String = "English"
    ) -> SlottedText {
        let words = splitIntoWords(text, language: language)
        let tsId = Qwen3ASRTokens.timestampTokenId

        var tokenIds: [Int] = []
        var timestampPositions: [Int] = []
        var validWords: [String] = []

        for word in words {
            let wordTokens = tokenizer.encode(word)
            guard !wordTokens.isEmpty else { continue }

            timestampPositions.append(tokenIds.count)
            tokenIds.append(tsId)

            tokenIds.append(contentsOf: wordTokens)

            timestampPositions.append(tokenIds.count)
            tokenIds.append(tsId)

            validWords.append(word)
        }

        return SlottedText(
            tokenIds: tokenIds,
            timestampPositions: timestampPositions,
            words: validWords
        )
    }

    /// Split text into words using the language-appropriate strategy.
    static func splitIntoWords(_ text: String, language: String) -> [String] {
        let lang = language.lowercased()

        if lang.contains("japanese") || lang == "ja" {
            return tokenizeJapanese(text)
        }
        if lang.contains("korean") || lang == "ko" {
            return tokenizeKorean(text)
        }
        // Scripts without word-level whitespace where Apple's NLTokenizer
        // provides native segmentation. Without these dispatches the
        // default whitespace path collapses each sentence to one token.
        if let nlLang = nlLanguageForUnspaced(lang) {
            return nlTokenize(text, language: nlLang)
        }
        return tokenizeSpaceLang(text)
    }

    private static func nlLanguageForUnspaced(_ lang: String) -> NLLanguage? {
        if lang.contains("thai")     || lang == "th" { return .thai }
        if lang.contains("lao")      || lang == "lo" { return .lao }
        if lang.contains("khmer")    || lang == "km" { return .khmer }
        if lang.contains("burmese")  || lang.contains("myanmar") || lang == "my" { return .burmese }
        if lang.contains("tibetan")  || lang == "bo" { return .tibetan }
        return nil
    }

    // MARK: - Japanese

    /// Morpheme-level segmentation via Apple's `NLTokenizer`. Produces
    /// tokens like `["今日", "は", "いい", "天気", "です", "ね"]`.
    static func tokenizeJapanese(_ text: String) -> [String] {
        return nlTokenize(text, language: .japanese)
    }

    // MARK: - Korean

    /// Word-level segmentation via Apple's `NLTokenizer`. Avoids the
    /// per-jamo character split that would be wrong for Korean.
    static func tokenizeKorean(_ text: String) -> [String] {
        return nlTokenize(text, language: .korean)
    }

    private static func nlTokenize(_ text: String, language: NLLanguage) -> [String] {
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.setLanguage(language)
        tokenizer.string = text
        var tokens: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let cleaned = cleanToken(String(text[range]))
            if !cleaned.isEmpty { tokens.append(cleaned) }
            return true
        }
        return tokens
    }

    // MARK: - Default path (whitespace + per-Han break)

    /// Whitespace split, then per-Han-ideograph break inside each segment.
    /// Default path for languages with reliable word-level whitespace —
    /// English, European languages, Hindi, Arabic, Vietnamese, Mongolian,
    /// Indonesian, etc. Chinese works because most Chinese text has no
    /// whitespace, so each ideograph becomes its own token via the per-Han
    /// break inside the (single) whitespace-bounded segment.
    static func tokenizeSpaceLang(_ text: String) -> [String] {
        var tokens: [String] = []
        for raw in text.split(whereSeparator: \.isWhitespace) {
            let cleaned = cleanToken(String(raw))
            guard !cleaned.isEmpty else { continue }
            tokens.append(contentsOf: splitSegmentWithChinese(cleaned))
        }
        return tokens
    }

    /// Inside a non-empty whitespace-bounded segment, peel each Han ideograph
    /// out as its own token while leaving non-Han runs grouped.
    private static func splitSegmentWithChinese(_ seg: String) -> [String] {
        var tokens: [String] = []
        var buf = ""
        for scalar in seg.unicodeScalars {
            if isHanIdeograph(scalar) {
                if !buf.isEmpty { tokens.append(buf); buf = "" }
                tokens.append(String(scalar))
            } else {
                buf.append(Character(scalar))
            }
        }
        if !buf.isEmpty { tokens.append(buf) }
        return tokens
    }

    // MARK: - Cleaning + classification

    /// Keep only Unicode Letters (`L*`), Numbers (`N*`), and ASCII apostrophe.
    /// Strips punctuation (e.g. the full-width period `。`), symbols,
    /// separators, and marks.
    static func cleanToken(_ token: String) -> String {
        var out = ""
        for scalar in token.unicodeScalars {
            if isKeptScalar(scalar) {
                out.unicodeScalars.append(scalar)
            }
        }
        return out
    }

    private static func isKeptScalar(_ scalar: Unicode.Scalar) -> Bool {
        if scalar == "'" { return true }
        let cat = scalar.properties.generalCategory
        switch cat {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter,
             .modifierLetter, .otherLetter,
             .decimalNumber, .letterNumber, .otherNumber,
             // Combining marks are essential for scripts like Thai, Lao,
             // Khmer, Burmese, Tibetan, Devanagari, Bengali, Arabic harakat
             // — stripping them mangles the word (e.g. "สวัสดี" → "สวสด").
             .nonspacingMark, .spacingMark, .enclosingMark:
            return true
        default:
            return false
        }
    }

    /// Han ideograph ranges only. Notably **excludes** hiragana
    /// (0x3040–0x309F), katakana (0x30A0–0x30FF), and Hangul syllables
    /// (0xAC00–0xD7AF), which are handled by the language-specific
    /// tokenizers (Japanese / Korean) instead.
    static func isHanIdeograph(_ scalar: Unicode.Scalar) -> Bool {
        let v = scalar.value
        if v >= 0x4E00 && v <= 0x9FFF   { return true }  // CJK Unified
        if v >= 0x3400 && v <= 0x4DBF   { return true }  // Extension A
        if v >= 0x20000 && v <= 0x2A6DF { return true }  // Extension B
        if v >= 0x2A700 && v <= 0x2B73F { return true }  // Extension C
        if v >= 0x2B740 && v <= 0x2B81F { return true }  // Extension D
        if v >= 0x2B820 && v <= 0x2CEAF { return true }  // Extension E
        if v >= 0xF900 && v <= 0xFAFF   { return true }  // Compatibility
        return false
    }
}
