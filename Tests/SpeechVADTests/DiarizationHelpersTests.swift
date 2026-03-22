import XCTest
@testable import SpeechVAD
import AudioCommon

final class DiarizationHelpersTests: XCTestCase {

    // MARK: - mergeSegments

    func testMergeSegmentsEmpty() {
        let result = DiarizationHelpers.mergeSegments([], minSilence: 0.15)
        XCTAssertTrue(result.isEmpty)
    }

    func testMergeSegmentsSingleSegment() {
        let segs = [DiarizedSegment(startTime: 1.0, endTime: 2.0, speakerId: 0)]
        let result = DiarizationHelpers.mergeSegments(segs, minSilence: 0.15)
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].startTime, 1.0, accuracy: 0.001)
        XCTAssertEqual(result[0].endTime, 2.0, accuracy: 0.001)
    }

    func testMergeSegmentsAdjacentSameSpeaker() {
        // Two segments from same speaker with small gap → should merge
        let segs = [
            DiarizedSegment(startTime: 1.0, endTime: 2.0, speakerId: 0),
            DiarizedSegment(startTime: 2.1, endTime: 3.0, speakerId: 0),
        ]
        let result = DiarizationHelpers.mergeSegments(segs, minSilence: 0.15)
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].startTime, 1.0, accuracy: 0.001)
        XCTAssertEqual(result[0].endTime, 3.0, accuracy: 0.001)
    }

    func testMergeSegmentsLargeGapNotMerged() {
        // Two segments from same speaker with large gap → should NOT merge
        let segs = [
            DiarizedSegment(startTime: 1.0, endTime: 2.0, speakerId: 0),
            DiarizedSegment(startTime: 3.0, endTime: 4.0, speakerId: 0),
        ]
        let result = DiarizationHelpers.mergeSegments(segs, minSilence: 0.15)
        XCTAssertEqual(result.count, 2)
    }

    func testMergeSegmentsDifferentSpeakers() {
        // Adjacent segments from different speakers → should NOT merge
        let segs = [
            DiarizedSegment(startTime: 1.0, endTime: 2.0, speakerId: 0),
            DiarizedSegment(startTime: 2.05, endTime: 3.0, speakerId: 1),
        ]
        let result = DiarizationHelpers.mergeSegments(segs, minSilence: 0.15)
        XCTAssertEqual(result.count, 2)
    }

    func testMergeSegmentsMultipleSpeakersInterleaved() {
        // Interleaved speakers with small gaps within each speaker
        let segs = [
            DiarizedSegment(startTime: 0.0, endTime: 1.0, speakerId: 0),
            DiarizedSegment(startTime: 1.0, endTime: 2.0, speakerId: 1),
            DiarizedSegment(startTime: 2.0, endTime: 3.0, speakerId: 0),
            DiarizedSegment(startTime: 3.0, endTime: 4.0, speakerId: 1),
        ]
        let result = DiarizationHelpers.mergeSegments(segs, minSilence: 0.15)
        // Speaker 0 has gap of 1.0s between segments → not merged
        // Speaker 1 has gap of 1.0s between segments → not merged
        XCTAssertEqual(result.count, 4)
        // Should be sorted by start time
        for i in 1..<result.count {
            XCTAssertGreaterThanOrEqual(result[i].startTime, result[i-1].startTime)
        }
    }

    // MARK: - compactSpeakerIds

    func testCompactSpeakerIdsEmpty() {
        let result = DiarizationHelpers.compactSpeakerIds([])
        XCTAssertTrue(result.isEmpty)
    }

    func testCompactSpeakerIdsAlreadyContiguous() {
        let segs = [
            DiarizedSegment(startTime: 0, endTime: 1, speakerId: 0),
            DiarizedSegment(startTime: 1, endTime: 2, speakerId: 1),
        ]
        let result = DiarizationHelpers.compactSpeakerIds(segs)
        XCTAssertEqual(result[0].speakerId, 0)
        XCTAssertEqual(result[1].speakerId, 1)
    }

    func testCompactSpeakerIdsWithGaps() {
        // Speaker IDs 2, 5 → should become 0, 1
        let segs = [
            DiarizedSegment(startTime: 0, endTime: 1, speakerId: 2),
            DiarizedSegment(startTime: 1, endTime: 2, speakerId: 5),
            DiarizedSegment(startTime: 2, endTime: 3, speakerId: 2),
        ]
        let result = DiarizationHelpers.compactSpeakerIds(segs)
        XCTAssertEqual(result[0].speakerId, 0)
        XCTAssertEqual(result[1].speakerId, 1)
        XCTAssertEqual(result[2].speakerId, 0)
    }

    func testCompactSpeakerIdsPreservesTimestamps() {
        let segs = [
            DiarizedSegment(startTime: 1.5, endTime: 3.7, speakerId: 10),
        ]
        let result = DiarizationHelpers.compactSpeakerIds(segs)
        XCTAssertEqual(result[0].startTime, 1.5, accuracy: 0.001)
        XCTAssertEqual(result[0].endTime, 3.7, accuracy: 0.001)
        XCTAssertEqual(result[0].speakerId, 0)
    }

    // MARK: - resample

    func testResampleSameRate() {
        let audio: [Float] = [1.0, 2.0, 3.0, 4.0]
        let result = DiarizationHelpers.resample(audio, from: 16000, to: 16000)
        XCTAssertEqual(result, audio)
    }

    func testResampleUpsample() {
        // Use longer signal for AVAudioConverter (short signals may have different frame counts)
        let audio: [Float] = (0..<100).map { Float($0) / 100.0 }
        let result = DiarizationHelpers.resample(audio, from: 8000, to: 16000)
        // 100 samples at 8kHz → ~200 samples at 16kHz
        XCTAssertGreaterThan(result.count, 150)
        XCTAssertLessThan(result.count, 250)
    }

    func testResampleDownsample() {
        let audio: [Float] = (0..<200).map { Float($0) / 200.0 }
        let result = DiarizationHelpers.resample(audio, from: 16000, to: 8000)
        // 200 samples at 16kHz → ~100 samples at 8kHz
        XCTAssertGreaterThan(result.count, 75)
        XCTAssertLessThan(result.count, 125)
    }

    func testResampleEmpty() {
        let result = DiarizationHelpers.resample([], from: 16000, to: 8000)
        XCTAssertTrue(result.isEmpty)
    }

    // MARK: - DiarizationConfig

    func testDiarizationConfigDefaultValues() {
        let config = DiarizationConfig.default
        XCTAssertEqual(config.onset, 0.5, accuracy: 0.001)
        XCTAssertEqual(config.offset, 0.3, accuracy: 0.001)
        XCTAssertEqual(config.minSpeechDuration, 0.3, accuracy: 0.001)
        XCTAssertEqual(config.minSilenceDuration, 0.15, accuracy: 0.001)
        XCTAssertEqual(config.clusteringThreshold, 0.715, accuracy: 0.001)
    }

    func testDiarizationConfigCustomValues() {
        let config = DiarizationConfig(
            onset: 0.7, offset: 0.4,
            minSpeechDuration: 0.5, minSilenceDuration: 0.2,
            clusteringThreshold: 0.5
        )
        XCTAssertEqual(config.onset, 0.7, accuracy: 0.001)
        XCTAssertEqual(config.offset, 0.4, accuracy: 0.001)
        XCTAssertEqual(config.minSpeechDuration, 0.5, accuracy: 0.001)
        XCTAssertEqual(config.minSilenceDuration, 0.2, accuracy: 0.001)
        XCTAssertEqual(config.clusteringThreshold, 0.5, accuracy: 0.001)
    }

    // MARK: - cosineDistance

    func testCosineDistanceSameVector() {
        let a: [Float] = [1, 0, 0]
        let dist = DiarizationHelpers.cosineDistance(a, a)
        XCTAssertEqual(dist, 0.0, accuracy: 0.001)
    }

    func testCosineDistanceOpposite() {
        let a: [Float] = [1, 0, 0]
        let b: [Float] = [-1, 0, 0]
        let dist = DiarizationHelpers.cosineDistance(a, b)
        XCTAssertEqual(dist, 2.0, accuracy: 0.001)
    }

    func testCosineDistanceOrthogonal() {
        let a: [Float] = [1, 0, 0]
        let b: [Float] = [0, 1, 0]
        let dist = DiarizationHelpers.cosineDistance(a, b)
        XCTAssertEqual(dist, 1.0, accuracy: 0.001)
    }

    func testCosineDistanceZeroVector() {
        let a: [Float] = [0, 0, 0]
        let b: [Float] = [1, 0, 0]
        let dist = DiarizationHelpers.cosineDistance(a, b)
        XCTAssertEqual(dist, 2.0, accuracy: 0.001)  // degenerate
    }

    // MARK: - constrainedAgglomerativeClustering

    func testClusteringEmpty() {
        let (assignment, centroids) = DiarizationHelpers.constrainedAgglomerativeClustering(
            items: [], threshold: 0.5)
        XCTAssertTrue(assignment.isEmpty)
        XCTAssertTrue(centroids.isEmpty)
    }

    func testClusteringSingleItem() {
        let items = [DiarizationHelpers.ClusterItem(
            windowIndex: 0, localSpeakerId: 0, embedding: [1, 0, 0])]
        let (assignment, centroids) = DiarizationHelpers.constrainedAgglomerativeClustering(
            items: items, threshold: 0.5)
        XCTAssertEqual(assignment, [0])
        XCTAssertEqual(centroids.count, 1)
    }

    func testClusteringSimilarEmbeddingsDifferentWindows() {
        // Two similar embeddings from different windows → should merge
        let items = [
            DiarizationHelpers.ClusterItem(windowIndex: 0, localSpeakerId: 0,
                                           embedding: [1, 0, 0, 0]),
            DiarizationHelpers.ClusterItem(windowIndex: 1, localSpeakerId: 0,
                                           embedding: [0.99, 0.1, 0, 0]),
        ]
        let (assignment, centroids) = DiarizationHelpers.constrainedAgglomerativeClustering(
            items: items, threshold: 0.5)

        // Should be merged into same cluster
        XCTAssertEqual(assignment[0], assignment[1])
        XCTAssertEqual(centroids.count, 1)
    }

    func testClusteringSameWindowNeverMerge() {
        // Two similar embeddings from SAME window → constraint prevents merge
        let items = [
            DiarizationHelpers.ClusterItem(windowIndex: 0, localSpeakerId: 0,
                                           embedding: [1, 0, 0, 0]),
            DiarizationHelpers.ClusterItem(windowIndex: 0, localSpeakerId: 1,
                                           embedding: [0.99, 0.1, 0, 0]),
        ]
        let (assignment, centroids) = DiarizationHelpers.constrainedAgglomerativeClustering(
            items: items, threshold: 2.0)  // Very permissive threshold

        // Must NOT merge despite being similar
        XCTAssertNotEqual(assignment[0], assignment[1])
        XCTAssertEqual(centroids.count, 2)
    }

    func testClusteringThresholdRespected() {
        // Two embeddings with cosine distance > threshold → should NOT merge
        let items = [
            DiarizationHelpers.ClusterItem(windowIndex: 0, localSpeakerId: 0,
                                           embedding: [1, 0, 0, 0]),
            DiarizationHelpers.ClusterItem(windowIndex: 1, localSpeakerId: 0,
                                           embedding: [0, 1, 0, 0]),
        ]
        // Cosine distance = 1.0, threshold = 0.5 → should NOT merge
        let (assignment, centroids) = DiarizationHelpers.constrainedAgglomerativeClustering(
            items: items, threshold: 0.5)

        XCTAssertNotEqual(assignment[0], assignment[1])
        XCTAssertEqual(centroids.count, 2)
    }

    func testClusteringCentroidLinkage() {
        // After merging, centroid should be the weighted average
        let items = [
            DiarizationHelpers.ClusterItem(windowIndex: 0, localSpeakerId: 0,
                                           embedding: [1, 0]),
            DiarizationHelpers.ClusterItem(windowIndex: 1, localSpeakerId: 0,
                                           embedding: [0.9, 0.1]),
        ]
        let (assignment, centroids) = DiarizationHelpers.constrainedAgglomerativeClustering(
            items: items, threshold: 1.0)

        XCTAssertEqual(assignment[0], assignment[1])
        XCTAssertEqual(centroids.count, 1)
        // Centroid should be average: [(1+0.9)/2, (0+0.1)/2] = [0.95, 0.05]
        XCTAssertEqual(centroids[0][0], 0.95, accuracy: 0.001)
        XCTAssertEqual(centroids[0][1], 0.05, accuracy: 0.001)
    }

    func testClusteringTransitiveConstraints() {
        // Three items: A(win0), B(win1), C(win0)
        // A and B are similar (closest pair) → merge. Merged {A,B} inherits win0 from A,
        // so {A,B} cannot merge with C (both have win0).
        let items = [
            DiarizationHelpers.ClusterItem(windowIndex: 0, localSpeakerId: 0,
                                           embedding: [1, 0, 0, 0]),  // A
            DiarizationHelpers.ClusterItem(windowIndex: 1, localSpeakerId: 0,
                                           embedding: [0.99, 0.01, 0, 0]),  // B (close to A)
            DiarizationHelpers.ClusterItem(windowIndex: 0, localSpeakerId: 1,
                                           embedding: [0, 0, 1, 0]),  // C (far from A and B)
        ]
        let (assignment, centroids) = DiarizationHelpers.constrainedAgglomerativeClustering(
            items: items, threshold: 2.0)

        // A and B should merge (closest, different windows)
        XCTAssertEqual(assignment[0], assignment[1], "A and B should be in same cluster")
        // C cannot merge with {A,B} — both have window 0
        XCTAssertNotEqual(assignment[0], assignment[2], "C should not merge with {A,B}")
        XCTAssertEqual(centroids.count, 2)
    }

    func testClusteringMultipleSpeakers() {
        // 2 speakers across 3 windows — speaker embeddings are clearly separated
        let spk0: [Float] = [1, 0, 0, 0]
        let spk1: [Float] = [0, 0, 1, 0]
        let items = [
            DiarizationHelpers.ClusterItem(windowIndex: 0, localSpeakerId: 0, embedding: spk0),
            DiarizationHelpers.ClusterItem(windowIndex: 0, localSpeakerId: 1, embedding: spk1),
            DiarizationHelpers.ClusterItem(windowIndex: 1, localSpeakerId: 0, embedding: spk0),
            DiarizationHelpers.ClusterItem(windowIndex: 1, localSpeakerId: 1, embedding: spk1),
            DiarizationHelpers.ClusterItem(windowIndex: 2, localSpeakerId: 0, embedding: spk0),
            DiarizationHelpers.ClusterItem(windowIndex: 2, localSpeakerId: 1, embedding: spk1),
        ]
        let (assignment, centroids) = DiarizationHelpers.constrainedAgglomerativeClustering(
            items: items, threshold: 0.5)

        // Should produce exactly 2 clusters
        XCTAssertEqual(centroids.count, 2)

        // All spk0 items should be in same cluster
        XCTAssertEqual(assignment[0], assignment[2])
        XCTAssertEqual(assignment[0], assignment[4])

        // All spk1 items should be in same cluster
        XCTAssertEqual(assignment[1], assignment[3])
        XCTAssertEqual(assignment[1], assignment[5])

        // spk0 and spk1 should be in different clusters
        XCTAssertNotEqual(assignment[0], assignment[1])
    }

    // MARK: - DiarizationResult

    func testDiarizationResultEmpty() {
        let result = DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
        XCTAssertTrue(result.segments.isEmpty)
        XCTAssertEqual(result.numSpeakers, 0)
        XCTAssertTrue(result.speakerEmbeddings.isEmpty)
    }

    // MARK: - Protocol Conformance

    func testPyannoteDiarizationPipelineTypealias() {
        // Verify the typealias compiles — DiarizationPipeline should be PyannoteDiarizationPipeline
        let _: DiarizationPipeline.Type = PyannoteDiarizationPipeline.self
    }

    func testSpeakerExtractionCapableConformance() {
        // PyannoteDiarizationPipeline conforms to SpeakerExtractionCapable
        XCTAssertTrue((PyannoteDiarizationPipeline.self as Any) is SpeakerExtractionCapable.Type)
    }

    func testSpeakerDiarizationModelConformance() {
        // PyannoteDiarizationPipeline conforms to SpeakerDiarizationModel
        XCTAssertTrue((PyannoteDiarizationPipeline.self as Any) is SpeakerDiarizationModel.Type)
    }

    #if canImport(CoreML)
    func testSortformerDiarizationModelConformance() {
        // SortformerDiarizer conforms to SpeakerDiarizationModel but NOT SpeakerExtractionCapable
        XCTAssertTrue((SortformerDiarizer.self as Any) is SpeakerDiarizationModel.Type)
        XCTAssertFalse((SortformerDiarizer.self as Any) is SpeakerExtractionCapable.Type)
    }
    #endif
}
