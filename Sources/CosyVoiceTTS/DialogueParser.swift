import Foundation

/// A parsed segment of dialogue text with optional speaker and emotion tags.
public struct DialogueSegment: Sendable, Equatable {
    /// Speaker identifier (e.g. "S1", "S2"), nil for untagged text
    public let speaker: String?
    /// Emotion or style tag (e.g. "happy", "whispers"), nil if none
    public let emotion: String?
    /// Cleaned text to synthesize
    public let text: String
}

/// Parses multi-speaker dialogue text with inline emotion/style tags.
///
/// Supports two tag types:
/// - Speaker tags: `[S1]`, `[Bob]`, `[speaker_1]`
/// - Emotion tags: `(happy)`, `(whispers)`, `(Speak like a pirate)`
///
/// Example: `[S1] (excited) Hello! [S2] (calm) Hey there.`
public enum DialogueParser {

    // MARK: - Emotion dictionary

    private static let emotionInstructions: [String: String] = [
        "happy": "Speak happily and with excitement.",
        "excited": "Speak happily and with excitement.",
        "sad": "Speak sadly with a melancholic tone.",
        "angry": "Speak with anger and intensity.",
        "whispers": "Speak in a soft, gentle whisper.",
        "whispering": "Speak in a soft, gentle whisper.",
        "laughs": "Speak while laughing.",
        "laughing": "Speak while laughing.",
        "calm": "Speak calmly and peacefully.",
        "surprised": "Speak with surprise and amazement.",
        "serious": "Speak in a serious, formal tone.",
    ]

    /// Convert an emotion tag to an instruction string for CosyVoice3.
    ///
    /// Known tags (happy, sad, angry, etc.) map to curated instructions.
    /// Unknown tags pass through as-is, allowing freeform instructions like `(Speak like a pirate)`.
    public static func emotionToInstruction(_ emotion: String) -> String {
        let key = emotion.lowercased().trimmingCharacters(in: .whitespaces)
        return emotionInstructions[key] ?? emotion
    }

    // MARK: - Parser

    /// Parse text into dialogue segments.
    ///
    /// Rules:
    /// - Speaker tag `[Name]` starts a new segment with that speaker
    /// - Emotion tag `(tag)` at segment start sets the emotion
    /// - Text between tags is the content to synthesize
    /// - Empty segments are skipped
    /// - Untagged text at the start becomes a segment with nil speaker
    public static func parse(_ text: String) -> [DialogueSegment] {
        let input = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !input.isEmpty else { return [] }

        // Split on speaker tags, keeping the tags as separators
        let speakerPattern = try! NSRegularExpression(pattern: #"\[([A-Za-z0-9_]+)\]"#)
        let matches = speakerPattern.matches(in: input, range: NSRange(input.startIndex..., in: input))

        // No speaker tags — parse as single segment (possibly with emotion)
        if matches.isEmpty {
            return parseEmotionSegments(input, speaker: nil)
        }

        var segments: [DialogueSegment] = []

        // Text before first speaker tag (if any)
        if let firstMatch = matches.first, firstMatch.range.location > 0 {
            let beforeIdx = input.index(input.startIndex, offsetBy: firstMatch.range.location)
            let beforeText = String(input[input.startIndex..<beforeIdx])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !beforeText.isEmpty {
                segments.append(contentsOf: parseEmotionSegments(beforeText, speaker: nil))
            }
        }

        // Process each speaker-tagged section
        for (i, match) in matches.enumerated() {
            let speakerRange = Range(match.range(at: 1), in: input)!
            let speaker = String(input[speakerRange])

            // Text after this speaker tag until next speaker tag or end
            let afterTagStart = input.index(input.startIndex, offsetBy: match.range.location + match.range.length)
            let afterTagEnd: String.Index
            if i + 1 < matches.count {
                afterTagEnd = input.index(input.startIndex, offsetBy: matches[i + 1].range.location)
            } else {
                afterTagEnd = input.endIndex
            }

            let sectionText = String(input[afterTagStart..<afterTagEnd])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !sectionText.isEmpty {
                segments.append(contentsOf: parseEmotionSegments(sectionText, speaker: speaker))
            }
        }

        return segments
    }

    /// Parse a text section for emotion tags, producing one or more segments.
    /// Emotion tags `(tag)` split the text into segments, each with an optional emotion.
    private static func parseEmotionSegments(_ text: String, speaker: String?) -> [DialogueSegment] {
        let emotionPattern = try! NSRegularExpression(pattern: #"\(([^)]+)\)"#)
        let matches = emotionPattern.matches(in: text, range: NSRange(text.startIndex..., in: text))

        // No emotion tags — single plain segment
        if matches.isEmpty {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return [] }
            return [DialogueSegment(speaker: speaker, emotion: nil, text: trimmed)]
        }

        var segments: [DialogueSegment] = []

        // Text before first emotion tag
        if let firstMatch = matches.first, firstMatch.range.location > 0 {
            let beforeIdx = text.index(text.startIndex, offsetBy: firstMatch.range.location)
            let beforeText = String(text[text.startIndex..<beforeIdx])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !beforeText.isEmpty {
                segments.append(DialogueSegment(speaker: speaker, emotion: nil, text: beforeText))
            }
        }

        // Each emotion tag starts a new segment with the text that follows
        for (i, match) in matches.enumerated() {
            let emotionRange = Range(match.range(at: 1), in: text)!
            let emotion = String(text[emotionRange])

            let afterTagStart = text.index(text.startIndex, offsetBy: match.range.location + match.range.length)
            let afterTagEnd: String.Index
            if i + 1 < matches.count {
                afterTagEnd = text.index(text.startIndex, offsetBy: matches[i + 1].range.location)
            } else {
                afterTagEnd = text.endIndex
            }

            let sectionText = String(text[afterTagStart..<afterTagEnd])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !sectionText.isEmpty {
                segments.append(DialogueSegment(speaker: speaker, emotion: emotion, text: sectionText))
            }
        }

        return segments
    }
}
