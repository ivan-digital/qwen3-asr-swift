import Foundation
import os

/// Loaded model set — holds references to all loaded models.
public struct ModelSet: Sendable {
    public let vad: (any StreamingVADProvider)?
    public let stt: (any SpeechRecognitionModel)?
    public let tts: (any SpeechGenerationModel)?

    public init(
        vad: (any StreamingVADProvider)? = nil,
        stt: (any SpeechRecognitionModel)? = nil,
        tts: (any SpeechGenerationModel)? = nil
    ) {
        self.vad = vad
        self.stt = stt
        self.tts = tts
    }
}

/// A model to load, with its factory closure and progress weight.
public struct ModelSpec: Sendable {
    let name: String
    let weight: Double
    let group: Int  // 0 = parallel group 1, 1 = sequential group 2
    let loader: @Sendable (_ progress: @escaping (Double, String) -> Void) async throws -> any Sendable

    /// VAD model spec.
    public static func vad(
        _ factory: @escaping @Sendable (_ progress: @escaping (Double, String) -> Void) async throws -> any StreamingVADProvider
    ) -> ModelSpec {
        ModelSpec(name: "VAD", weight: 1, group: 0, loader: { progress in
            try await factory(progress) as any Sendable
        })
    }

    /// Speech-to-text model spec.
    public static func stt(
        _ factory: @escaping @Sendable (_ progress: @escaping (Double, String) -> Void) async throws -> any SpeechRecognitionModel
    ) -> ModelSpec {
        ModelSpec(name: "ASR", weight: 15, group: 0, loader: { progress in
            try await factory(progress) as any Sendable
        })
    }

    /// Text-to-speech model spec.
    public static func tts(
        _ factory: @escaping @Sendable (_ progress: @escaping (Double, String) -> Void) async throws -> any SpeechGenerationModel
    ) -> ModelSpec {
        ModelSpec(name: "TTS", weight: 20, group: 1, loader: { progress in
            try await factory(progress) as any Sendable
        })
    }
}

/// Unified model loading orchestrator with aggregated progress.
///
/// Loads multiple speech models with coordinated progress reporting.
/// Group 0 models (VAD, ASR) load in parallel; Group 1 (TTS) loads after
/// to reduce peak memory.
///
/// ```swift
/// let models = try await ModelLoader.load([
///     .vad { p in try await SileroVADModel.fromPretrained(engine: .coreml, progressHandler: p) },
///     .stt { p in try await ParakeetASRModel.fromPretrained(progressHandler: p) },
///     .tts { p in try await KokoroTTSModel.fromPretrained(progressHandler: p) },
/// ], onProgress: { progress, stage in
///     self.loadProgress = progress
///     self.loadingStatus = stage
/// })
/// // models.vad, models.stt, models.tts are ready
/// ```
public enum ModelLoader {

    private static let log = Logger(subsystem: "audio.soniqo", category: "ModelLoader")

    /// Load the requested models with aggregated progress reporting.
    public static func load(
        _ specs: [ModelSpec],
        onProgress: @escaping @Sendable (_ progress: Double, _ stage: String) -> Void = { _, _ in }
    ) async throws -> ModelSet {
        let totalWeight = specs.reduce(0.0) { $0 + $1.weight }
        guard totalWeight > 0 else { return ModelSet() }

        let state = LoadState(totalWeight: totalWeight)

        // Group 0: parallel (VAD + ASR)
        let group0 = specs.filter { $0.group == 0 }
        // Group 1: sequential after group 0 (TTS — heavy, reduce peak memory)
        let group1 = specs.filter { $0.group != 0 }

        var results: [(String, any Sendable)] = []

        // Load group 0 in parallel
        if !group0.isEmpty {
            try await withThrowingTaskGroup(of: (String, any Sendable).self) { group in
                for spec in group0 {
                    group.addTask {
                        let model = try await loadSpec(spec, state: state, onProgress: onProgress)
                        return (spec.name, model)
                    }
                }
                for try await result in group {
                    results.append(result)
                }
            }
        }

        // Load group 1 sequentially
        for spec in group1 {
            let model = try await loadSpec(spec, state: state, onProgress: onProgress)
            results.append((spec.name, model))
        }

        onProgress(1.0, "Ready")
        log.info("All models loaded")

        // Build ModelSet from results
        var vad: (any StreamingVADProvider)?
        var stt: (any SpeechRecognitionModel)?
        var tts: (any SpeechGenerationModel)?

        for (name, model) in results {
            if let m = model as? any StreamingVADProvider { vad = m }
            if let m = model as? any SpeechRecognitionModel { stt = m }
            if let m = model as? any SpeechGenerationModel { tts = m }
        }

        return ModelSet(vad: vad, stt: stt, tts: tts)
    }

    // MARK: - Internal

    private final class LoadState: @unchecked Sendable {
        let totalWeight: Double
        private var completed: Double = 0
        private let lock = NSLock()

        init(totalWeight: Double) { self.totalWeight = totalWeight }

        func addCompleted(_ w: Double) {
            lock.lock(); completed += w; lock.unlock()
        }

        var completedFraction: Double {
            lock.lock(); defer { lock.unlock() }
            return completed / totalWeight
        }

        func overallProgress(specWeight: Double, localFraction: Double) -> Double {
            lock.lock(); defer { lock.unlock() }
            return (completed + localFraction * specWeight) / totalWeight
        }
    }

    private static func loadSpec(
        _ spec: ModelSpec,
        state: LoadState,
        onProgress: @escaping @Sendable (Double, String) -> Void
    ) async throws -> any Sendable {
        log.info("Loading \(spec.name)...")
        onProgress(state.completedFraction, "\(spec.name)...")

        let adapter: @Sendable (Double, String) -> Void = { fraction, status in
            let overall = state.overallProgress(specWeight: spec.weight, localFraction: fraction)
            let stage = status.isEmpty ? spec.name : "\(spec.name): \(status)"
            onProgress(overall, stage)
        }

        let model = try await spec.loader(adapter)
        state.addCompleted(spec.weight)
        log.info("\(spec.name) loaded")
        return model
    }
}
