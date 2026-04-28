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
/// Mirrors the upstream reference `Qwen3ForceAlignProcessor.encode_timestamp`
/// from `QwenLM/Qwen3-ASR/qwen_asr/inference/qwen3_forced_aligner.py`:
///
/// ```python
/// if language == "japanese": tokenize_japanese(text)   # nagisa morphemes
/// elif language == "korean": tokenize_korean(text)     # soynlp LTokenizer
/// else:                      tokenize_space_lang(text) # whitespace + per-Han break
/// ```
///
/// Universal `clean_token` strips anything that isn't a Unicode Letter / Number
/// (or an ASCII apostrophe). `is_cjk_char` is Han-ideographs **only** —
/// hiragana, katakana, and Hangul are NOT split per character because the
/// model was trained to emit `<timestamp>` slots between morphemes for those
/// scripts (Japanese via nagisa, Korean via soynlp), not between every kana
/// or jamo.
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

    /// Split text into words using the language-appropriate strategy that
    /// matches the upstream Python preprocessing exactly.
    static func splitIntoWords(_ text: String, language: String) -> [String] {
        let lang = language.lowercased()

        if lang.contains("japanese") || lang == "ja" {
            return tokenizeJapanese(text)
        }
        if lang.contains("korean") || lang == "ko" {
            return tokenizeKorean(text)
        }
        return tokenizeSpaceLang(text)
    }

    // MARK: - Japanese (nagisa-equivalent via Apple NLTokenizer)

    /// Morpheme-level segmentation, matching upstream `tokenize_japanese`
    /// (which uses `nagisa.tagging`). Apple's `NLTokenizer(.word)` for
    /// Japanese performs morphological word segmentation backed by the
    /// platform's built-in Japanese tokenizer (similar accuracy to MeCab
    /// IPADic), producing morphemes like `["今日", "は", "いい", "天気",
    /// "です", "ね"]` — comparable granularity to nagisa.
    static func tokenizeJapanese(_ text: String) -> [String] {
        return nlTokenize(text, language: .japanese)
    }

    // MARK: - Korean (soynlp-equivalent via Apple NLTokenizer)

    /// Morpheme-ish segmentation, matching upstream `tokenize_korean`
    /// (`soynlp.LTokenizer`). NLTokenizer for Korean splits on word
    /// boundaries; this produces coarser granularity than soynlp's L-part
    /// extraction but is the closest natively-available segmenter and avoids
    /// the per-jamo character split that would be wrong here.
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
    /// Matches upstream `tokenize_space_lang` + `split_segment_with_chinese`.
    /// Used for **everything** except Japanese and Korean — including
    /// English, Chinese, European languages, Hindi, Arabic, etc. Chinese
    /// works because most Chinese text has no whitespace, so each ideograph
    /// becomes its own token via the per-Han break.
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
    /// Mirrors upstream `split_segment_with_chinese`.
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
    /// Mirrors upstream `is_kept_char` + `clean_token`. Strips punctuation
    /// (e.g. the full-width period `。`), symbols, separators, and marks.
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
             .decimalNumber, .letterNumber, .otherNumber:
            return true
        default:
            return false
        }
    }

    /// Han ideograph ranges only — matches upstream `is_cjk_char` exactly.
    /// Notably **excludes** hiragana (0x3040–0x309F), katakana (0x30A0–0x30FF),
    /// and Hangul syllables (0xAC00–0xD7AF), which are handled by the
    /// language-specific tokenizers (Japanese / Korean) instead.
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
