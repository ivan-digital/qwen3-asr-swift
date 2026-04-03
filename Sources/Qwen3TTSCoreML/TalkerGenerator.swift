#if canImport(CoreML)
import CoreML
import Foundation

/// CoreML CodeDecoder with scatter-write KV cache.
///
/// Supports both stateful (MLState) and stateless (explicit I/O)
/// KV cache management. Falls back to stateless automatically.
final class TalkerGenerator {

    private let model: MLModel
    private let maxSeqLen: Int
    private let hiddenSize: Int
    private let isStateful: Bool
    private let kvDim: Int

    /// MLState for stateful models.
    private var state: MLState?
    /// Explicit KV cache for stateless models.
    private var keyCache: MLMultiArray?
    private var valueCache: MLMultiArray?
    private var currentPos: Int = 0

    /// Last hidden state from the most recent forward pass.
    private(set) var lastHiddenState: MLMultiArray?

    init(model: MLModel, maxSeqLen: Int = 256, hiddenSize: Int = 1024) {
        self.model = model
        self.maxSeqLen = maxSeqLen
        self.hiddenSize = hiddenSize
        self.isStateful = !model.modelDescription.stateDescriptionsByName.isEmpty

        // Detect KV dim from model input spec
        if let kcDesc = model.modelDescription.inputDescriptionsByName["key_cache"],
           let constraint = kcDesc.multiArrayConstraint {
            self.kvDim = constraint.shape[1].intValue
        } else {
            // Default: 28 layers * 8 heads * 128 dim = 28672
            self.kvDim = 28672
        }
    }

    func resetCache() {
        if isStateful {
            state = model.makeState()
        } else {
            let shape = [1, NSNumber(value: kvDim), 1, NSNumber(value: maxSeqLen)] as [NSNumber]
            keyCache = try! MLMultiArray(shape: shape, dataType: .float16)
            memset(keyCache!.dataPointer, 0, kvDim * maxSeqLen * 2)
            valueCache = try! MLMultiArray(shape: shape, dataType: .float16)
            memset(valueCache!.dataPointer, 0, kvDim * maxSeqLen * 2)
        }
        currentPos = 0
    }

    func forward(embedArray: MLMultiArray) throws -> (logits: [Float], hidden: [Float16]) {
        return try forwardInternal(inputEmbeds: ensureNCHW(embedArray, channels: hiddenSize))
    }

    func forward(embed: [Float16]) throws -> (logits: [Float], hidden: [Float16]) {
        let inputEmbeds = try MLMultiArray(
            shape: [1, NSNumber(value: hiddenSize), 1, 1], dataType: .float16)
        let embPtr = inputEmbeds.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<hiddenSize { embPtr[i] = embed[i] }
        return try forwardInternal(inputEmbeds: inputEmbeds)
    }

    private func forwardInternal(inputEmbeds: MLMultiArray) throws -> (logits: [Float], hidden: [Float16]) {
        guard currentPos < maxSeqLen else { throw TalkerError.cacheFull }

        let cacheLength = try MLMultiArray(shape: [1], dataType: .int32)
        cacheLength.dataPointer.assumingMemoryBound(to: Int32.self)[0] = Int32(currentPos)

        let keyPaddingMask = try MLMultiArray(
            shape: [1, NSNumber(value: maxSeqLen)], dataType: .float16)
        let maskPtr = keyPaddingMask.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<maxSeqLen { maskPtr[i] = i <= currentPos ? Float16(0) : Float16(-1e4) }

        let updateMask = try MLMultiArray(
            shape: [1, NSNumber(value: maxSeqLen)], dataType: .float16)
        memset(updateMask.dataPointer, 0, maxSeqLen * 2)
        updateMask.dataPointer.assumingMemoryBound(to: Float16.self)[currentPos] = Float16(1.0)

        var inputs: [String: MLFeatureValue] = [
            "input_embeds": MLFeatureValue(multiArray: inputEmbeds),
            "cache_length": MLFeatureValue(multiArray: cacheLength),
            "key_padding_mask": MLFeatureValue(multiArray: keyPaddingMask),
            "kv_cache_update_mask": MLFeatureValue(multiArray: updateMask),
        ]

        // Add explicit KV cache for stateless models
        if !isStateful, let kc = keyCache, let vc = valueCache {
            inputs["key_cache"] = MLFeatureValue(multiArray: kc)
            inputs["value_cache"] = MLFeatureValue(multiArray: vc)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
        let result: MLFeatureProvider

        if isStateful, let mlState = state {
            result = try model.prediction(from: provider, using: mlState)
        } else {
            result = try model.prediction(from: provider)
        }

        let logitsArray = result.featureValue(for: "logits")!.multiArrayValue!
        let hiddenArray = result.featureValue(for: "hidden_states")!.multiArrayValue!

        // Update explicit KV cache from outputs
        if !isStateful {
            if let newKC = result.featureValue(for: "new_key_cache")?.multiArrayValue {
                keyCache = newKC
            }
            if let newVC = result.featureValue(for: "new_value_cache")?.multiArrayValue {
                valueCache = newVC
            }
        }

        currentPos += 1
        lastHiddenState = ensureNCHW(hiddenArray, channels: hiddenSize)

        // Extract using subscript for stride safety
        var logits = [Float](repeating: 0, count: 3072)
        let ndim = logitsArray.shape.count
        for i in 0..<3072 {
            var idx = [NSNumber](repeating: 0, count: ndim)
            idx[ndim - 1] = i as NSNumber
            logits[i] = logitsArray[idx].floatValue
        }

        var hidden = [Float16](repeating: 0, count: hiddenSize)
        let hndim = hiddenArray.shape.count
        for i in 0..<hiddenSize {
            var idx = [NSNumber](repeating: 0, count: hndim)
            if hndim >= 4 { idx[1] = i as NSNumber } else { idx[0] = i as NSNumber }
            hidden[i] = Float16(hiddenArray[idx].floatValue)
        }

        return (logits, hidden)
    }

    func prefill(embeds: [[Float16]]) throws -> (logits: [Float], hidden: [Float16]) {
        var lastLogits = [Float]()
        var lastHidden = [Float16]()
        for embed in embeds {
            (lastLogits, lastHidden) = try forward(embed: embed)
        }
        return (lastLogits, lastHidden)
    }

    enum TalkerError: Error {
        case cacheFull
    }
}
#endif
