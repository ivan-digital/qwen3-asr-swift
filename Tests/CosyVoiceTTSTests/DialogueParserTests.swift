import XCTest
@testable import CosyVoiceTTS

final class DialogueParserTests: XCTestCase {

    // MARK: - Plain text (no tags)

    func testPlainText() {
        let segments = DialogueParser.parse("Hello world")
        XCTAssertEqual(segments.count, 1)
        XCTAssertNil(segments[0].speaker)
        XCTAssertNil(segments[0].emotion)
        XCTAssertEqual(segments[0].text, "Hello world")
    }

    func testEmptyText() {
        XCTAssertEqual(DialogueParser.parse(""), [])
        XCTAssertEqual(DialogueParser.parse("   "), [])
        XCTAssertEqual(DialogueParser.parse("\n\t"), [])
    }

    // MARK: - Speaker tags

    func testSpeakerTags() {
        let segments = DialogueParser.parse("[S1] Hello [S2] World")
        XCTAssertEqual(segments.count, 2)

        XCTAssertEqual(segments[0].speaker, "S1")
        XCTAssertNil(segments[0].emotion)
        XCTAssertEqual(segments[0].text, "Hello")

        XCTAssertEqual(segments[1].speaker, "S2")
        XCTAssertNil(segments[1].emotion)
        XCTAssertEqual(segments[1].text, "World")
    }

    func testSpeakerTagsWithUnderscore() {
        let segments = DialogueParser.parse("[speaker_1] Hi [speaker_2] Hey")
        XCTAssertEqual(segments.count, 2)
        XCTAssertEqual(segments[0].speaker, "speaker_1")
        XCTAssertEqual(segments[1].speaker, "speaker_2")
    }

    func testSingleSpeakerTag() {
        let segments = DialogueParser.parse("[Alice] Good morning everyone!")
        XCTAssertEqual(segments.count, 1)
        XCTAssertEqual(segments[0].speaker, "Alice")
        XCTAssertEqual(segments[0].text, "Good morning everyone!")
    }

    func testTextBeforeSpeakerTag() {
        let segments = DialogueParser.parse("Intro text [S1] Hello")
        XCTAssertEqual(segments.count, 2)
        XCTAssertNil(segments[0].speaker)
        XCTAssertEqual(segments[0].text, "Intro text")
        XCTAssertEqual(segments[1].speaker, "S1")
        XCTAssertEqual(segments[1].text, "Hello")
    }

    // MARK: - Emotion tags

    func testEmotionTags() {
        let segments = DialogueParser.parse("(happy) Hello! (sad) Goodbye.")
        XCTAssertEqual(segments.count, 2)

        XCTAssertNil(segments[0].speaker)
        XCTAssertEqual(segments[0].emotion, "happy")
        XCTAssertEqual(segments[0].text, "Hello!")

        XCTAssertNil(segments[1].speaker)
        XCTAssertEqual(segments[1].emotion, "sad")
        XCTAssertEqual(segments[1].text, "Goodbye.")
    }

    func testSingleEmotion() {
        let segments = DialogueParser.parse("(whispers) Be very quiet.")
        XCTAssertEqual(segments.count, 1)
        XCTAssertEqual(segments[0].emotion, "whispers")
        XCTAssertEqual(segments[0].text, "Be very quiet.")
    }

    // MARK: - Combined speaker + emotion

    func testCombinedSpeakerAndEmotion() {
        let segments = DialogueParser.parse("[S1] (excited) Great news! [S2] (calm) Tell me more.")
        XCTAssertEqual(segments.count, 2)

        XCTAssertEqual(segments[0].speaker, "S1")
        XCTAssertEqual(segments[0].emotion, "excited")
        XCTAssertEqual(segments[0].text, "Great news!")

        XCTAssertEqual(segments[1].speaker, "S2")
        XCTAssertEqual(segments[1].emotion, "calm")
        XCTAssertEqual(segments[1].text, "Tell me more.")
    }

    func testMixedTaggedAndUntagged() {
        let segments = DialogueParser.parse("[S1] (happy) Hi! [S2] Just normal here.")
        XCTAssertEqual(segments.count, 2)

        XCTAssertEqual(segments[0].emotion, "happy")
        XCTAssertEqual(segments[0].text, "Hi!")

        XCTAssertEqual(segments[1].speaker, "S2")
        XCTAssertNil(segments[1].emotion)
        XCTAssertEqual(segments[1].text, "Just normal here.")
    }

    // MARK: - Emotion to instruction mapping

    func testKnownEmotions() {
        XCTAssertEqual(
            DialogueParser.emotionToInstruction("happy"),
            "Speak happily and with excitement.")
        XCTAssertEqual(
            DialogueParser.emotionToInstruction("sad"),
            "Speak sadly with a melancholic tone.")
        XCTAssertEqual(
            DialogueParser.emotionToInstruction("angry"),
            "Speak with anger and intensity.")
        XCTAssertEqual(
            DialogueParser.emotionToInstruction("whispers"),
            "Speak in a soft, gentle whisper.")
        XCTAssertEqual(
            DialogueParser.emotionToInstruction("calm"),
            "Speak calmly and peacefully.")
        XCTAssertEqual(
            DialogueParser.emotionToInstruction("surprised"),
            "Speak with surprise and amazement.")
        XCTAssertEqual(
            DialogueParser.emotionToInstruction("serious"),
            "Speak in a serious, formal tone.")
    }

    func testEmotionCaseInsensitive() {
        XCTAssertEqual(
            DialogueParser.emotionToInstruction("HAPPY"),
            "Speak happily and with excitement.")
        XCTAssertEqual(
            DialogueParser.emotionToInstruction("Happy"),
            "Speak happily and with excitement.")
    }

    func testUnknownEmotionPassthrough() {
        XCTAssertEqual(
            DialogueParser.emotionToInstruction("Speak like a pirate"),
            "Speak like a pirate")
    }

    func testEmotionAliases() {
        // excited = happy
        XCTAssertEqual(
            DialogueParser.emotionToInstruction("excited"),
            DialogueParser.emotionToInstruction("happy"))
        // whispering = whispers
        XCTAssertEqual(
            DialogueParser.emotionToInstruction("whispering"),
            DialogueParser.emotionToInstruction("whispers"))
        // laughing = laughs
        XCTAssertEqual(
            DialogueParser.emotionToInstruction("laughing"),
            DialogueParser.emotionToInstruction("laughs"))
    }

    // MARK: - Edge cases

    func testThreeSpeakers() {
        let segments = DialogueParser.parse("[A] One [B] Two [C] Three")
        XCTAssertEqual(segments.count, 3)
        XCTAssertEqual(segments[0].speaker, "A")
        XCTAssertEqual(segments[1].speaker, "B")
        XCTAssertEqual(segments[2].speaker, "C")
    }

    func testWhitespaceHandling() {
        let segments = DialogueParser.parse("  [S1]   Hello   [S2]   World  ")
        XCTAssertEqual(segments.count, 2)
        XCTAssertEqual(segments[0].text, "Hello")
        XCTAssertEqual(segments[1].text, "World")
    }

    func testMultipleEmotionsPerSpeaker() {
        let segments = DialogueParser.parse("[S1] (happy) Good news! (sad) But also bad news.")
        XCTAssertEqual(segments.count, 2)
        XCTAssertEqual(segments[0].speaker, "S1")
        XCTAssertEqual(segments[0].emotion, "happy")
        XCTAssertEqual(segments[0].text, "Good news!")
        XCTAssertEqual(segments[1].speaker, "S1")
        XCTAssertEqual(segments[1].emotion, "sad")
        XCTAssertEqual(segments[1].text, "But also bad news.")
    }

    func testFreeformInstruction() {
        let segments = DialogueParser.parse("(Speak in a slow, dramatic voice) The end is near.")
        XCTAssertEqual(segments.count, 1)
        XCTAssertEqual(segments[0].emotion, "Speak in a slow, dramatic voice")
        XCTAssertEqual(segments[0].text, "The end is near.")
    }
}
