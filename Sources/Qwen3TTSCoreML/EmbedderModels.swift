#if canImport(CoreML)
import CoreML
import Foundation

/// CoreML text token → embedding [1, 1024, 1, 1].
/// Replaces Swift-side text_embedding + FC1→SiLU→FC2 projection.
final class TextProjectorModel {
    private let model: MLModel
    init(model: MLModel) { self.model = model }

    func embed(_ tokenId: Int) throws -> MLMultiArray {
        let input = try MLMultiArray(shape: [1], dataType: .int32)
        input.dataPointer.assumingMemoryBound(to: Int32.self)[0] = Int32(tokenId)
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: input)])
        let result = try model.prediction(from: provider)
        return result.featureValue(for: "input_embeds")!.multiArrayValue!
    }
}

/// CoreML codec token → embedding [1, 1024, 1, 1].
/// Replaces Swift-side codec_embedding table lookup.
final class CodeEmbedderModel {
    private let model: MLModel
    init(model: MLModel) { self.model = model }

    func embed(_ tokenId: Int) throws -> MLMultiArray {
        let input = try MLMultiArray(shape: [1], dataType: .int32)
        input.dataPointer.assumingMemoryBound(to: Int32.self)[0] = Int32(tokenId)
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: input)])
        let result = try model.prediction(from: provider)
        return result.featureValue(for: "input_embeds")!.multiArrayValue!
    }
}

/// CoreML linearized CB1-15 token → embedding [1, 1024, 1, 1].
/// Replaces Swift-side cpCodecEmbeddings table lookup + sum.
final class MultiCodeEmbedderModel {
    private let model: MLModel
    private let vocabSize = 2048
    init(model: MLModel) { self.model = model }

    /// Embed a single codebook token using linearized index: codebookIdx * 2048 + tokenId.
    func embed(codebookIdx: Int, tokenId: Int) throws -> MLMultiArray {
        let linearIdx = codebookIdx * vocabSize + tokenId
        let input = try MLMultiArray(shape: [1], dataType: .int32)
        input.dataPointer.assumingMemoryBound(to: Int32.self)[0] = Int32(linearIdx)
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: input)])
        let result = try model.prediction(from: provider)
        return result.featureValue(for: "input_embeds")!.multiArrayValue!
    }
}

// MARK: - MLMultiArray helpers

/// Add two MLMultiArrays element-wise. Accumulates in FP32 internally, stores as FP16.
/// Python coremltools returns FP32 arrays, so additions happen in FP32 naturally.
/// Swift CoreML returns FP16, so we must explicitly upcast for correct accumulation.
/// Read a Float value from an MLMultiArray at logical index, respecting strides.
func readMLFloat(_ arr: MLMultiArray, _ logicalIdx: Int) -> Float {
    // For [1, C, 1, 1] or [C, 1, 1] — the first dimension is the channel index
    let stride = arr.strides[0].intValue
    let physIdx = logicalIdx * stride
    if arr.dataType == .float16 {
        return Float(arr.dataPointer.assumingMemoryBound(to: Float16.self)[physIdx])
    } else {
        return arr.dataPointer.assumingMemoryBound(to: Float.self)[physIdx]
    }
}

/// Add two MLMultiArrays element-wise. Both should be [1, C, 1, 1] contiguous FP16.
/// Use ensureNCHW() on inputs first to make them contiguous.
func addMLMultiArrays(_ a: MLMultiArray, _ b: MLMultiArray) -> MLMultiArray {
    let channels = a.shape[1].intValue  // [1, C, 1, 1]
    let result = try! MLMultiArray(shape: [1, NSNumber(value: channels), 1, 1], dataType: .float16)
    let rp = result.dataPointer.assumingMemoryBound(to: Float16.self)

    // Read each input respecting its data type
    func read(_ arr: MLMultiArray, _ i: Int) -> Float {
        if arr.dataType == .float32 {
            return arr.dataPointer.assumingMemoryBound(to: Float.self)[i]
        } else {
            return Float(arr.dataPointer.assumingMemoryBound(to: Float16.self)[i])
        }
    }

    for i in 0..<channels {
        rp[i] = Float16(read(a, i) + read(b, i))
    }
    return result
}

/// Ensure MLMultiArray is contiguous rank 4 [1, C, 1, 1].
/// CoreML may use non-contiguous strides (e.g., stride=32 for SIMD alignment).
/// This copies data respecting strides into a contiguous buffer.
func ensureNCHW(_ array: MLMultiArray, channels: Int) -> MLMultiArray {
    let result = try! MLMultiArray(shape: [1, NSNumber(value: channels), 1, 1], dataType: .float16)
    let dst = result.dataPointer.assumingMemoryBound(to: Float16.self)

    // Check if strides indicate non-contiguous layout
    let strides = array.strides.map { $0.intValue }
    let isContiguous = strides.last == 1 && (strides.count < 2 || strides[strides.count - 2] <= 1
        || (strides.count >= 1 && strides[0] == 1))

    // CoreML outputs may have non-unit strides (SIMD alignment).
    // Use MLMultiArray subscript for correct strided access.
    let shape = array.shape.map { $0.intValue }
    let ndim = shape.count
    for i in 0..<channels {
        var idx = [NSNumber](repeating: 0, count: ndim)
        // Channel is always the first non-batch dimension
        if ndim == 3 { idx[0] = i as NSNumber }     // [C, 1, 1]
        else if ndim == 4 { idx[1] = i as NSNumber } // [1, C, 1, 1]
        else { idx[0] = i as NSNumber }
        let val = array[idx]
        dst[i] = Float16(val.floatValue)
    }
    return result
}
#endif
