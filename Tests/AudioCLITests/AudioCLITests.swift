import XCTest
import ArgumentParser
@testable import AudioCLILib

// MARK: - Root Command

final class AudioCLIRootTests: XCTestCase {

    func testHelpParsesSuccessfully() throws {
        // --help may throw CleanExit or return the command depending on AP version
        do {
            _ = try AudioCLI.parseAsRoot(["--help"])
        } catch {
            XCTAssertEqual(AudioCLI.exitCode(for: error), .success)
        }
    }

    func testHelpContainsAllSubcommands() {
        let help = AudioCLI.helpMessage()
        XCTAssertTrue(help.contains("transcribe"))
        XCTAssertTrue(help.contains("align"))
        XCTAssertTrue(help.contains("speak"))
        XCTAssertTrue(help.contains("respond"))
    }

    func testHelpContainsAbstract() {
        let help = AudioCLI.helpMessage()
        XCTAssertTrue(help.contains("AI speech models"))
    }

    func testUnknownSubcommandFails() {
        XCTAssertThrowsError(try AudioCLI.parseAsRoot(["synthesize"])) { error in
            XCTAssertEqual(AudioCLI.exitCode(for: error), .validationFailure)
        }
    }

    func testNoSubcommandParsesRootCommand() throws {
        // No arguments — returns root command (which prints help on run())
        do {
            let cmd = try AudioCLI.parseAsRoot([])
            XCTAssertTrue(cmd is AudioCLI)
        } catch {
            // Some AP versions throw; that's also acceptable
        }
    }
}

// MARK: - TranscribeCommand

final class TranscribeCommandTests: XCTestCase {

    func testParsesAudioFile() throws {
        let cmd = try AudioCLI.parseAsRoot(["transcribe", "/path/to/audio.wav"])
        let transcribe = try XCTUnwrap(cmd as? TranscribeCommand)
        XCTAssertEqual(transcribe.audioFile, "/path/to/audio.wav")
    }

    func testDefaultModel() throws {
        let cmd = try AudioCLI.parseAsRoot(["transcribe", "audio.wav"])
        let transcribe = try XCTUnwrap(cmd as? TranscribeCommand)
        XCTAssertEqual(transcribe.model, "0.6B")
    }

    func testDefaultLanguageIsNil() throws {
        let cmd = try AudioCLI.parseAsRoot(["transcribe", "audio.wav"])
        let transcribe = try XCTUnwrap(cmd as? TranscribeCommand)
        XCTAssertNil(transcribe.language)
    }

    func testParsesModelOption() throws {
        let cmd = try AudioCLI.parseAsRoot(["transcribe", "audio.wav", "--model", "1.7B"])
        let transcribe = try XCTUnwrap(cmd as? TranscribeCommand)
        XCTAssertEqual(transcribe.model, "1.7B")
    }

    func testParsesModelShortFlag() throws {
        let cmd = try AudioCLI.parseAsRoot(["transcribe", "audio.wav", "-m", "1.7B"])
        let transcribe = try XCTUnwrap(cmd as? TranscribeCommand)
        XCTAssertEqual(transcribe.model, "1.7B")
    }

    func testParsesLanguage() throws {
        let cmd = try AudioCLI.parseAsRoot(["transcribe", "audio.wav", "--language", "zh"])
        let transcribe = try XCTUnwrap(cmd as? TranscribeCommand)
        XCTAssertEqual(transcribe.language, "zh")
    }

    func testParsesFullModelId() throws {
        let cmd = try AudioCLI.parseAsRoot(["transcribe", "audio.wav", "-m", "org/my-custom-model"])
        let transcribe = try XCTUnwrap(cmd as? TranscribeCommand)
        XCTAssertEqual(transcribe.model, "org/my-custom-model")
    }

    func testMissingAudioFileFails() {
        XCTAssertThrowsError(try AudioCLI.parseAsRoot(["transcribe"])) { error in
            XCTAssertEqual(AudioCLI.exitCode(for: error), .validationFailure)
        }
    }

    func testHelpParsesSuccessfully() throws {
        do {
            _ = try AudioCLI.parseAsRoot(["transcribe", "--help"])
        } catch {
            XCTAssertEqual(AudioCLI.exitCode(for: error), .success)
        }
    }
}

// MARK: - AlignCommand

final class AlignCommandTests: XCTestCase {

    func testParsesAudioFile() throws {
        let cmd = try AudioCLI.parseAsRoot(["align", "/path/audio.wav"])
        let align = try XCTUnwrap(cmd as? AlignCommand)
        XCTAssertEqual(align.audioFile, "/path/audio.wav")
    }

    func testDefaultTextIsNil() throws {
        let cmd = try AudioCLI.parseAsRoot(["align", "audio.wav"])
        let align = try XCTUnwrap(cmd as? AlignCommand)
        XCTAssertNil(align.text)
    }

    func testDefaultModel() throws {
        let cmd = try AudioCLI.parseAsRoot(["align", "audio.wav"])
        let align = try XCTUnwrap(cmd as? AlignCommand)
        XCTAssertEqual(align.model, "0.6B")
    }

    func testDefaultAlignerModel() throws {
        let cmd = try AudioCLI.parseAsRoot(["align", "audio.wav"])
        let align = try XCTUnwrap(cmd as? AlignCommand)
        XCTAssertEqual(align.alignerModel, "aufklarer/Qwen3-ForcedAligner-0.6B-4bit")
    }

    func testParsesTextOption() throws {
        let cmd = try AudioCLI.parseAsRoot(["align", "audio.wav", "--text", "Hello world"])
        let align = try XCTUnwrap(cmd as? AlignCommand)
        XCTAssertEqual(align.text, "Hello world")
    }

    func testParsesTextShortFlag() throws {
        let cmd = try AudioCLI.parseAsRoot(["align", "audio.wav", "-t", "Hello world"])
        let align = try XCTUnwrap(cmd as? AlignCommand)
        XCTAssertEqual(align.text, "Hello world")
    }

    func testParsesAlignerModel() throws {
        let cmd = try AudioCLI.parseAsRoot(["align", "audio.wav", "--aligner-model", "org/custom-aligner"])
        let align = try XCTUnwrap(cmd as? AlignCommand)
        XCTAssertEqual(align.alignerModel, "org/custom-aligner")
    }

    func testParsesLanguage() throws {
        let cmd = try AudioCLI.parseAsRoot(["align", "audio.wav", "--language", "de"])
        let align = try XCTUnwrap(cmd as? AlignCommand)
        XCTAssertEqual(align.language, "de")
    }

    func testMissingAudioFileFails() {
        XCTAssertThrowsError(try AudioCLI.parseAsRoot(["align"])) { error in
            XCTAssertEqual(AudioCLI.exitCode(for: error), .validationFailure)
        }
    }

    func testAllOptionsCombined() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "align", "audio.wav",
            "-t", "Test text",
            "-m", "1.7B",
            "--aligner-model", "org/aligner",
            "--language", "fr"
        ])
        let align = try XCTUnwrap(cmd as? AlignCommand)
        XCTAssertEqual(align.audioFile, "audio.wav")
        XCTAssertEqual(align.text, "Test text")
        XCTAssertEqual(align.model, "1.7B")
        XCTAssertEqual(align.alignerModel, "org/aligner")
        XCTAssertEqual(align.language, "fr")
    }
}

// MARK: - SpeakCommand

final class SpeakCommandTests: XCTestCase {

    // MARK: Defaults

    func testDefaultEngine() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "Hello"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.engine, "qwen3")
    }

    func testDefaultOutput() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "Hello"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.output, "output.wav")
    }

    func testDefaultLanguage() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "Hello"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.language, "english")
    }

    func testDefaultSamplingParams() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "Hello"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.temperature, 0.9, accuracy: 0.001)
        XCTAssertEqual(speak.topK, 50)
        XCTAssertEqual(speak.maxTokens, 500)
    }

    func testDefaultFlags() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "Hello"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertFalse(speak.stream)
        XCTAssertFalse(speak.listSpeakers)
        XCTAssertFalse(speak.verbose)
    }

    func testDefaultQwen3Options() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "Hello"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.model, "base")
        XCTAssertNil(speak.speaker)
        XCTAssertNil(speak.instruct)
        XCTAssertNil(speak.batchFile)
        XCTAssertEqual(speak.batchSize, 4)
        XCTAssertEqual(speak.firstChunkFrames, 3)
        XCTAssertEqual(speak.chunkFrames, 25)
    }

    func testDefaultCosyVoiceModelId() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "Hello"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.modelId, "aufklarer/CosyVoice3-0.5B-MLX-4bit")
    }

    // MARK: Engine selection

    func testCosyVoiceEngine() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "--engine", "cosyvoice", "Hello"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.engine, "cosyvoice")
    }

    func testInvalidEngineFails() {
        XCTAssertThrowsError(try AudioCLI.parseAsRoot(["speak", "--engine", "whisper", "hi"])) { error in
            XCTAssertEqual(AudioCLI.exitCode(for: error), .validationFailure)
        }
    }

    // MARK: Validation

    func testNoTextNoFlagsNoFileFails() {
        XCTAssertThrowsError(try AudioCLI.parseAsRoot(["speak"])) { error in
            XCTAssertEqual(AudioCLI.exitCode(for: error), .validationFailure)
        }
    }

    func testListSpeakersSatisfiesValidation() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "--list-speakers"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertTrue(speak.listSpeakers)
        XCTAssertNil(speak.text)
    }

    func testBatchFileSatisfiesValidation() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "--batch-file", "texts.txt"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.batchFile, "texts.txt")
        XCTAssertNil(speak.text)
    }

    // MARK: Options parsing

    func testOutputOption() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "hi", "--output", "out.wav"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.output, "out.wav")
    }

    func testOutputShortFlag() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "hi", "-o", "short.wav"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.output, "short.wav")
    }

    func testStreamFlag() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "hi", "--stream"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertTrue(speak.stream)
    }

    func testSamplingOptions() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "speak", "hi",
            "--temperature", "0.7",
            "--top-k", "25",
            "--max-tokens", "200"
        ])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.temperature, 0.7, accuracy: 0.001)
        XCTAssertEqual(speak.topK, 25)
        XCTAssertEqual(speak.maxTokens, 200)
    }

    func testSpeakerOption() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "hi", "--speaker", "vivian"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.speaker, "vivian")
    }

    func testInstructOption() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "hi", "--instruct", "Speak cheerfully"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.instruct, "Speak cheerfully")
    }

    func testModelOption() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "hi", "--model", "customVoice"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.model, "customVoice")
    }

    func testChunkFramesOptions() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "speak", "hi", "--stream",
            "--first-chunk-frames", "1",
            "--chunk-frames", "10"
        ])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.firstChunkFrames, 1)
        XCTAssertEqual(speak.chunkFrames, 10)
    }

    func testCosyVoiceModelIdOption() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "speak", "hi", "--engine", "cosyvoice",
            "--model-id", "org/my-cosyvoice"
        ])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.modelId, "org/my-cosyvoice")
    }

    func testVerboseFlag() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "hi", "--verbose"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertTrue(speak.verbose)
    }

    func testLanguageOption() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "Hallo", "--language", "german"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.language, "german")
    }

    func testBatchSizeOption() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "--batch-file", "f.txt", "--batch-size", "8"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.batchSize, 8)
    }
}

// MARK: - RespondCommand

final class RespondCommandTests: XCTestCase {

    // MARK: Defaults

    func testDefaultValues() throws {
        let cmd = try AudioCLI.parseAsRoot(["respond", "--input", "user.wav"])
        let respond = try XCTUnwrap(cmd as? RespondCommand)
        XCTAssertEqual(respond.input, "user.wav")
        XCTAssertEqual(respond.output, "response.wav")
        XCTAssertEqual(respond.voice, "NATM0")
        XCTAssertEqual(respond.systemPrompt, "assistant")
        XCTAssertEqual(respond.maxSteps, 500)
        XCTAssertEqual(respond.modelId, "aufklarer/PersonaPlex-7B-MLX-4bit")
        XCTAssertFalse(respond.stream)
        XCTAssertEqual(respond.chunkFrames, 25)
        XCTAssertFalse(respond.compile)
        XCTAssertFalse(respond.listVoices)
        XCTAssertFalse(respond.listPrompts)
        XCTAssertFalse(respond.verbose)
    }

    // MARK: Input parsing

    func testInputShortFlag() throws {
        let cmd = try AudioCLI.parseAsRoot(["respond", "-i", "user.wav"])
        let respond = try XCTUnwrap(cmd as? RespondCommand)
        XCTAssertEqual(respond.input, "user.wav")
    }

    func testOutputShortFlag() throws {
        let cmd = try AudioCLI.parseAsRoot(["respond", "-i", "user.wav", "-o", "out.wav"])
        let respond = try XCTUnwrap(cmd as? RespondCommand)
        XCTAssertEqual(respond.output, "out.wav")
    }

    // MARK: All options

    func testAllOptions() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "respond",
            "--input", "user.wav",
            "--output", "out.wav",
            "--voice", "NATF1",
            "--system-prompt", "teacher",
            "--max-steps", "250",
            "--model-id", "my/custom-model",
            "--stream",
            "--chunk-frames", "50",
            "--compile",
            "--verbose"
        ])
        let respond = try XCTUnwrap(cmd as? RespondCommand)
        XCTAssertEqual(respond.input, "user.wav")
        XCTAssertEqual(respond.output, "out.wav")
        XCTAssertEqual(respond.voice, "NATF1")
        XCTAssertEqual(respond.systemPrompt, "teacher")
        XCTAssertEqual(respond.maxSteps, 250)
        XCTAssertEqual(respond.modelId, "my/custom-model")
        XCTAssertTrue(respond.stream)
        XCTAssertEqual(respond.chunkFrames, 50)
        XCTAssertTrue(respond.compile)
        XCTAssertTrue(respond.verbose)
    }

    // MARK: List flags

    func testListVoices() throws {
        let cmd = try AudioCLI.parseAsRoot(["respond", "--list-voices"])
        let respond = try XCTUnwrap(cmd as? RespondCommand)
        XCTAssertTrue(respond.listVoices)
    }

    func testListPrompts() throws {
        let cmd = try AudioCLI.parseAsRoot(["respond", "--list-prompts"])
        let respond = try XCTUnwrap(cmd as? RespondCommand)
        XCTAssertTrue(respond.listPrompts)
    }

    // MARK: Voice presets

    func testVoiceOption() throws {
        let cmd = try AudioCLI.parseAsRoot(["respond", "-i", "a.wav", "--voice", "VARF0"])
        let respond = try XCTUnwrap(cmd as? RespondCommand)
        XCTAssertEqual(respond.voice, "VARF0")
    }

    func testSystemPromptOption() throws {
        let cmd = try AudioCLI.parseAsRoot(["respond", "-i", "a.wav", "--system-prompt", "customer-service"])
        let respond = try XCTUnwrap(cmd as? RespondCommand)
        XCTAssertEqual(respond.systemPrompt, "customer-service")
    }
}

// MARK: - Utility Functions

final class UtilityTests: XCTestCase {

    func testResolveASRModelId_06B() {
        XCTAssertTrue(resolveASRModelId("0.6b").contains("0.6B"))
        XCTAssertTrue(resolveASRModelId("0.6B").contains("0.6B"))
        XCTAssertTrue(resolveASRModelId("small").contains("0.6B"))
    }

    func testResolveASRModelId_17B() {
        XCTAssertTrue(resolveASRModelId("1.7b").contains("1.7B"))
        XCTAssertTrue(resolveASRModelId("1.7B").contains("1.7B"))
        XCTAssertTrue(resolveASRModelId("large").contains("1.7B"))
    }

    func testResolveASRModelId_passthrough() {
        XCTAssertEqual(resolveASRModelId("org/custom-model"), "org/custom-model")
        XCTAssertEqual(resolveASRModelId("aufklarer/Qwen3-ASR-0.6B-MLX-4bit"), "aufklarer/Qwen3-ASR-0.6B-MLX-4bit")
    }

    func testFormatDuration() {
        XCTAssertEqual(formatDuration(24000), "1.00")
        XCTAssertEqual(formatDuration(48000), "2.00")
        XCTAssertEqual(formatDuration(12000), "0.50")
        XCTAssertEqual(formatDuration(0), "0.00")
    }

    func testFormatDurationCustomSampleRate() {
        XCTAssertEqual(formatDuration(16000, sampleRate: 16000), "1.00")
        XCTAssertEqual(formatDuration(44100, sampleRate: 44100), "1.00")
    }
}
