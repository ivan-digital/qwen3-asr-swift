import XCTest
@testable import SpeechCore

final class MemoryTierTests: XCTestCase {

    // MARK: - Tier Detection

    func testDetectReturnsValidTier() {
        let tier = MemoryTier.detect()
        // Should always return a valid tier
        let allTiers: [MemoryTier] = [.full, .standard, .constrained, .minimal]
        XCTAssertTrue(allTiers.contains(tier), "detect() should return a known tier")
    }

    func testDetectOnMacOS() {
        #if os(macOS)
        let totalGB = Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024)
        let tier = MemoryTier.detect()
        if totalGB >= 8 {
            XCTAssertEqual(tier, .full, "macOS with 8GB+ should be .full")
        } else if totalGB >= 4 {
            XCTAssertEqual(tier, .standard, "macOS with 4-8GB should be .standard")
        }
        #else
        throw XCTSkip("macOS-only test")
        #endif
    }

    // MARK: - Tier Properties: Feature Flags

    func testFullTierEnablesEverything() {
        XCTAssertTrue(MemoryTier.full.useCoreMLASR)
        XCTAssertTrue(MemoryTier.full.useCoreMLTTS)
        XCTAssertTrue(MemoryTier.full.useLLM)
        XCTAssertFalse(MemoryTier.full.autoUnload,
                       "Full tier should NOT auto-unload — enough memory for all models")
    }

    func testStandardTierEnablesCoreMLWithAutoUnload() {
        XCTAssertTrue(MemoryTier.standard.useCoreMLASR)
        XCTAssertTrue(MemoryTier.standard.useCoreMLTTS)
        XCTAssertTrue(MemoryTier.standard.useLLM)
        XCTAssertTrue(MemoryTier.standard.autoUnload,
                       "Standard tier should auto-unload between phases")
    }

    func testConstrainedTierFallsBackOnASR() {
        XCTAssertFalse(MemoryTier.constrained.useCoreMLASR,
                       "Constrained should use Apple Speech, not CoreML ASR")
        XCTAssertTrue(MemoryTier.constrained.useCoreMLTTS,
                      "Constrained should still use CoreML TTS (Kokoro is small)")
        XCTAssertTrue(MemoryTier.constrained.useLLM)
        XCTAssertTrue(MemoryTier.constrained.autoUnload)
    }

    func testMinimalTierUsesOnlySystemTTS() {
        XCTAssertFalse(MemoryTier.minimal.useCoreMLASR)
        XCTAssertFalse(MemoryTier.minimal.useCoreMLTTS,
                       "Minimal should use system TTS, not CoreML")
        XCTAssertFalse(MemoryTier.minimal.useLLM,
                       "Minimal should not load LLM")
        XCTAssertTrue(MemoryTier.minimal.autoUnload)
    }

    // MARK: - Tier Ordering: Higher Tiers Include Lower Tier Capabilities

    func testTierCapabilityOrdering() {
        let tiers: [MemoryTier] = [.full, .standard, .constrained, .minimal]

        // useCoreMLASR: full and standard only
        XCTAssertTrue(tiers[0].useCoreMLASR)
        XCTAssertTrue(tiers[1].useCoreMLASR)
        XCTAssertFalse(tiers[2].useCoreMLASR)
        XCTAssertFalse(tiers[3].useCoreMLASR)

        // useLLM: all except minimal
        XCTAssertTrue(tiers[0].useLLM)
        XCTAssertTrue(tiers[1].useLLM)
        XCTAssertTrue(tiers[2].useLLM)
        XCTAssertFalse(tiers[3].useLLM)

        // useCoreMLTTS: all except minimal
        XCTAssertTrue(tiers[0].useCoreMLTTS)
        XCTAssertTrue(tiers[1].useCoreMLTTS)
        XCTAssertTrue(tiers[2].useCoreMLTTS)
        XCTAssertFalse(tiers[3].useCoreMLTTS)
    }

    // MARK: - Description

    func testDescriptionNotEmpty() {
        for tier in [MemoryTier.full, .standard, .constrained, .minimal] {
            XCTAssertFalse(tier.description.isEmpty,
                           "\(tier.rawValue) description should not be empty")
        }
    }

    func testDescriptionContainsTierName() {
        XCTAssertTrue(MemoryTier.full.description.contains("full"))
        XCTAssertTrue(MemoryTier.standard.description.contains("standard"))
        XCTAssertTrue(MemoryTier.constrained.description.contains("constrained"))
        XCTAssertTrue(MemoryTier.minimal.description.contains("minimal"))
    }

    // MARK: - Raw Value Roundtrip

    func testRawValueRoundtrip() {
        for tier in [MemoryTier.full, .standard, .constrained, .minimal] {
            let raw = tier.rawValue
            let reconstructed = MemoryTier(rawValue: raw)
            XCTAssertEqual(reconstructed, tier, "rawValue roundtrip failed for \(raw)")
        }
    }
}

// MARK: - Pipeline Config Tests

final class PipelineConfigTests: XCTestCase {

    func testDefaultConfig() {
        let config = PipelineConfig()
        XCTAssertEqual(config.mode, .echo)
        XCTAssertEqual(config.vadOnset, 0.5, accuracy: 0.01)
        XCTAssertEqual(config.vadOffset, 0.35, accuracy: 0.01)
        XCTAssertTrue(config.allowInterruptions)
        XCTAssertEqual(config.maxUtteranceDuration, 15.0, accuracy: 0.1)
        XCTAssertFalse(config.autoUnloadModels, "Default should NOT auto-unload")
        XCTAssertTrue(config.warmupSTT)
        XCTAssertTrue(config.eagerSTT)
    }

    func testAutoUnloadConfig() {
        var config = PipelineConfig()
        config.autoUnloadModels = true
        XCTAssertTrue(config.autoUnloadModels)
    }

    func testVoicePipelineMode() {
        var config = PipelineConfig()
        config.mode = .voicePipeline
        XCTAssertEqual(config.mode, .voicePipeline)
        XCTAssertEqual(config.mode.rawValue, 0)
    }

    func testTranscribeOnlyMode() {
        var config = PipelineConfig()
        config.mode = .transcribeOnly
        XCTAssertEqual(config.mode, .transcribeOnly)
        XCTAssertEqual(config.mode.rawValue, 1)
    }

    func testEchoMode() {
        XCTAssertEqual(PipelineConfig.default.mode, .echo)
        XCTAssertEqual(PipelineMode.echo.rawValue, 2)
    }
}

final class PipelineStateTests: XCTestCase {

    func testAllStatesHaveRawValues() {
        XCTAssertEqual(PipelineState.idle.rawValue, 0)
        XCTAssertEqual(PipelineState.listening.rawValue, 1)
        XCTAssertEqual(PipelineState.transcribing.rawValue, 2)
        XCTAssertEqual(PipelineState.thinking.rawValue, 3)
        XCTAssertEqual(PipelineState.speaking.rawValue, 4)
    }

    func testStateRoundtrip() {
        for raw in 0...4 {
            let state = PipelineState(rawValue: raw)
            XCTAssertNotNil(state, "Raw value \(raw) should produce a valid state")
        }
    }

    func testInvalidStateReturnsNil() {
        XCTAssertNil(PipelineState(rawValue: 99))
    }
}
