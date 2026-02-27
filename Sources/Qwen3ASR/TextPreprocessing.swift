import Foundation
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

/// Language-specific text preprocessing for forced alignment
public enum TextPreprocessor {

    /// Split text into words and insert timestamp slots for alignment.
    ///
    /// For each word, inserts `<timestamp><timestamp>` pairs so the model
    /// can predict start/end timestamps at those positions.
    ///
    /// - Parameters:
    ///   - text: Input text to align
    ///   - tokenizer: Tokenizer for encoding word tokens
    ///   - language: Language hint for word splitting strategy
    /// - Returns: SlottedText with token IDs, timestamp positions, and words
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

            // Insert <timestamp> before word (start marker)
            timestampPositions.append(tokenIds.count)
            tokenIds.append(tsId)

            // Word tokens
            tokenIds.append(contentsOf: wordTokens)

            // Insert <timestamp> after word (end marker)
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

    /// Split text into words using language-appropriate strategy
    static func splitIntoWords(_ text: String, language: String) -> [String] {
        let lang = language.lowercased()

        if isCJKLanguage(lang) {
            return splitCJK(text)
        } else {
            return splitWhitespace(text)
        }
    }

    /// Split on whitespace and punctuation boundaries (English, European languages)
    private static func splitWhitespace(_ text: String) -> [String] {
        // Split on whitespace, filter empty
        let raw = text.components(separatedBy: .whitespaces)
        return raw.filter { !$0.isEmpty }
    }

    /// Character-level splitting for CJK languages
    private static func splitCJK(_ text: String) -> [String] {
        var words: [String] = []
        var currentNonCJK = ""

        for scalar in text.unicodeScalars {
            if isCJKScalar(scalar) {
                // Flush any accumulated non-CJK text
                if !currentNonCJK.isEmpty {
                    let trimmed = currentNonCJK.trimmingCharacters(in: .whitespaces)
                    if !trimmed.isEmpty {
                        words.append(trimmed)
                    }
                    currentNonCJK = ""
                }
                words.append(String(scalar))
            } else {
                currentNonCJK.append(Character(scalar))
            }
        }

        // Flush remaining non-CJK text
        if !currentNonCJK.isEmpty {
            let trimmed = currentNonCJK.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty {
                words.append(trimmed)
            }
        }

        return words
    }

    private static func isCJKLanguage(_ lang: String) -> Bool {
        return lang.contains("chinese") || lang.contains("japanese")
            || lang.contains("korean") || lang == "zh" || lang == "ja"
            || lang == "ko" || lang == "cjk"
    }

    private static func isCJKScalar(_ scalar: Unicode.Scalar) -> Bool {
        let v = scalar.value
        // CJK Unified Ideographs
        if v >= 0x4E00 && v <= 0x9FFF { return true }
        // CJK Extension A
        if v >= 0x3400 && v <= 0x4DBF { return true }
        // CJK Extension B+
        if v >= 0x20000 && v <= 0x2A6DF { return true }
        // CJK Compatibility Ideographs
        if v >= 0xF900 && v <= 0xFAFF { return true }
        // Hiragana
        if v >= 0x3040 && v <= 0x309F { return true }
        // Katakana
        if v >= 0x30A0 && v <= 0x30FF { return true }
        // Hangul Syllables
        if v >= 0xAC00 && v <= 0xD7AF { return true }
        return false
    }
}
