import XCTest
import MLX
@testable import Qwen3TTS

final class ReferenceAudioCacheTests: XCTestCase {

    // MARK: - Hashing

    func testHashIsStableForSameAudio() {
        let audio: [Float] = [0.1, -0.2, 0.3, -0.4, 0.5]
        XCTAssertEqual(ReferenceAudioCache.hash(audio), ReferenceAudioCache.hash(audio))
    }

    func testHashDiffersOnContentChange() {
        let a: [Float] = [0.1, 0.2, 0.3]
        let b: [Float] = [0.1, 0.2, 0.4]
        XCTAssertNotEqual(ReferenceAudioCache.hash(a), ReferenceAudioCache.hash(b))
    }

    func testHashDiffersOnLengthChange() {
        let a: [Float] = [0.1, 0.2, 0.3]
        let b: [Float] = [0.1, 0.2, 0.3, 0.0]
        XCTAssertNotEqual(ReferenceAudioCache.hash(a), ReferenceAudioCache.hash(b))
    }

    // MARK: - Round-trip

    func testStoreAndRetrieveSpeakerEmbedding() {
        let cache = ReferenceAudioCache()
        let audio: [Float] = [0.1, 0.2, 0.3]
        let embed = MLXArray.zeros([1, 1024])

        XCTAssertNil(cache.speakerEmbed(for: audio, sampleRate: 16000))

        cache.storeSpeakerEmbed(embed, audio: audio, sampleRate: 16000)

        let cached = cache.speakerEmbed(for: audio, sampleRate: 16000)
        XCTAssertNotNil(cached)
        XCTAssertEqual(cached?.shape, [1, 1024])
    }

    func testStoreAndRetrieveCodecRefCodes() {
        let cache = ReferenceAudioCache()
        let audio: [Float] = [0.1, 0.2, 0.3]
        let codes = MLXArray.zeros([1, 16, 32])

        cache.storeCodecRefCodes(codes, audio: audio, sampleRate: 24000)

        let cached = cache.codecRefCodes(for: audio, sampleRate: 24000)
        XCTAssertNotNil(cached)
        XCTAssertEqual(cached?.shape, [1, 16, 32])
    }

    func testSpeakerAndCodecShareSameEntry() {
        let cache = ReferenceAudioCache()
        let audio: [Float] = [0.1, 0.2, 0.3]
        let embed = MLXArray.zeros([1, 1024])
        let codes = MLXArray.zeros([1, 16, 32])

        cache.storeSpeakerEmbed(embed, audio: audio, sampleRate: 24000)
        cache.storeCodecRefCodes(codes, audio: audio, sampleRate: 24000)

        XCTAssertEqual(cache.count, 1, "Same audio + sample rate should collapse into one entry")
        XCTAssertNotNil(cache.speakerEmbed(for: audio, sampleRate: 24000))
        XCTAssertNotNil(cache.codecRefCodes(for: audio, sampleRate: 24000))
    }

    // MARK: - Keying

    func testDifferentSampleRateMissesCache() {
        let cache = ReferenceAudioCache()
        let audio: [Float] = [0.1, 0.2, 0.3]
        let embed = MLXArray.zeros([1, 1024])

        cache.storeSpeakerEmbed(embed, audio: audio, sampleRate: 16000)
        XCTAssertNotNil(cache.speakerEmbed(for: audio, sampleRate: 16000))
        XCTAssertNil(cache.speakerEmbed(for: audio, sampleRate: 24000))
    }

    func testDifferentAudioMissesCache() {
        let cache = ReferenceAudioCache()
        let embed = MLXArray.zeros([1, 1024])

        cache.storeSpeakerEmbed(embed, audio: [0.1, 0.2, 0.3], sampleRate: 24000)
        XCTAssertNil(cache.speakerEmbed(for: [0.1, 0.2, 0.4], sampleRate: 24000))
    }

    // MARK: - LRU eviction

    func testLRUEvictionAtCapacity() {
        let cache = ReferenceAudioCache(capacity: 2)
        let embed = MLXArray.zeros([1, 4])

        cache.storeSpeakerEmbed(embed, audio: [1.0], sampleRate: 24000)
        cache.storeSpeakerEmbed(embed, audio: [2.0], sampleRate: 24000)
        XCTAssertEqual(cache.count, 2)

        cache.storeSpeakerEmbed(embed, audio: [3.0], sampleRate: 24000)
        XCTAssertEqual(cache.count, 2)
        XCTAssertNil(cache.speakerEmbed(for: [1.0], sampleRate: 24000), "Oldest entry should evict")
        XCTAssertNotNil(cache.speakerEmbed(for: [2.0], sampleRate: 24000))
        XCTAssertNotNil(cache.speakerEmbed(for: [3.0], sampleRate: 24000))
    }

    func testLRUTouchOnLookupPreservesRecentEntry() {
        let cache = ReferenceAudioCache(capacity: 2)
        let embed = MLXArray.zeros([1, 4])

        cache.storeSpeakerEmbed(embed, audio: [1.0], sampleRate: 24000)
        cache.storeSpeakerEmbed(embed, audio: [2.0], sampleRate: 24000)

        // Touch [1.0] via lookup — it should now be the most recent
        _ = cache.speakerEmbed(for: [1.0], sampleRate: 24000)

        cache.storeSpeakerEmbed(embed, audio: [3.0], sampleRate: 24000)
        XCTAssertNotNil(cache.speakerEmbed(for: [1.0], sampleRate: 24000), "Touched entry should survive")
        XCTAssertNil(cache.speakerEmbed(for: [2.0], sampleRate: 24000), "Untouched entry should evict")
    }

    // MARK: - Clear

    func testClearRemovesAllEntries() {
        let cache = ReferenceAudioCache()
        let embed = MLXArray.zeros([1, 4])

        cache.storeSpeakerEmbed(embed, audio: [1.0], sampleRate: 24000)
        cache.storeSpeakerEmbed(embed, audio: [2.0], sampleRate: 24000)
        XCTAssertEqual(cache.count, 2)

        cache.clear()
        XCTAssertEqual(cache.count, 0)
        XCTAssertNil(cache.speakerEmbed(for: [1.0], sampleRate: 24000))
    }
}
