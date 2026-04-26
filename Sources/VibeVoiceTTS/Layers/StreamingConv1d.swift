import Foundation
import MLX
import MLXNN

public typealias ConvCacheKey = Int

public class StreamingConvCache {
    private var cache: [ConvCacheKey: MLXArray] = [:]

    public init() {}

    public func get(_ layerId: ConvCacheKey) -> MLXArray? {
        cache[layerId]
    }

    public func set(_ layerId: ConvCacheKey, _ value: MLXArray) {
        cache[layerId] = value
    }

    public func clear() {
        cache.removeAll()
    }
}

func getExtraPaddingForConv1d(length: Int, kernelSize: Int, stride: Int, paddingTotal: Int) -> Int {
    let nFrames = Float(length - kernelSize + paddingTotal) / Float(stride) + 1
    let idealLength = (Int(ceil(nFrames)) - 1) * stride + (kernelSize - paddingTotal)
    return idealLength - length
}

public class SConv1d: Module {
    public let conv: Conv1d
    public let causal: Bool
    public let padMode: String

    public let kernelSize: Int
    public let dilation: Int
    public let stride: Int
    public let inChannels: Int
    public let outChannels: Int
    public let contextSize: Int
    public let paddingTotal: Int

    private var layerId: ConvCacheKey?

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true,
        causal: Bool = false,
        padMode: String = "reflect"
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.dilation = dilation
        self.causal = causal
        self.padMode = padMode

        self.contextSize = (kernelSize - 1) * dilation - (stride - 1)
        self.paddingTotal = (kernelSize - 1) * dilation - (stride - 1)

        self.conv = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            dilation: dilation,
            groups: groups,
            bias: bias
        )

        super.init()
    }

    public func getLayerId() -> ConvCacheKey {
        if layerId == nil {
            layerId = ObjectIdentifier(self).hashValue
        }
        return layerId!
    }

    public func callAsFunction(
        _ x: MLXArray,
        cache: StreamingConvCache? = nil,
        useCache: Bool = false
    ) -> MLXArray {
        guard useCache, let cache = cache else {
            return forwardNonStreaming(x)
        }

        precondition(causal, "Streaming mode only supported for causal convolutions")
        return forwardStreaming(x, cache: cache)
    }

    private func forwardNonStreaming(_ x: MLXArray) -> MLXArray {
        let length = x.dim(2)
        let extraPadding = getExtraPaddingForConv1d(
            length: length,
            kernelSize: kernelSize,
            stride: stride,
            paddingTotal: paddingTotal
        )

        var padded: MLXArray
        if causal {
            padded = padNCL(x, left: paddingTotal, right: extraPadding)
        } else {
            let paddingRight = paddingTotal / 2
            let paddingLeft = paddingTotal - paddingRight
            padded = padNCL(x, left: paddingLeft, right: paddingRight + extraPadding)
        }

        let nlc = padded.transposed(0, 2, 1)
        let outputNLC = conv(nlc)
        return outputNLC.transposed(0, 2, 1)
    }

    private func forwardStreaming(_ x: MLXArray, cache: StreamingConvCache) -> MLXArray {
        let layerId = getLayerId()

        var cachedStates = cache.get(layerId)

        if cachedStates == nil && contextSize > 0 {
            cachedStates = MLXArray.zeros([x.dim(0), inChannels, contextSize])
        }

        var inputWithContext: MLXArray
        if let cached = cachedStates, cached.dim(2) > 0 {
            inputWithContext = concatenated([cached, x], axis: 2)
        } else {
            inputWithContext = x
        }

        let nlc = inputWithContext.transposed(0, 2, 1)
        let outputNLC = conv(nlc)
        let output = outputNLC.transposed(0, 2, 1)

        if contextSize > 0 {
            let totalLength = inputWithContext.dim(2)
            if totalLength >= contextSize {
                let newCacheStart = totalLength - contextSize
                let newCache = inputWithContext[0..., 0..., newCacheStart...]
                cache.set(layerId, newCache)
            } else {
                cache.set(layerId, inputWithContext)
            }
        }

        return output
    }

    private func padNCL(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
        if left == 0 && right == 0 {
            return x
        }

        if padMode == "reflect" {
            return reflectPadNCL(x, left: left, right: right)
        } else {
            return zeroPadNCL(x, left: left, right: right)
        }
    }

    private func zeroPadNCL(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
        let batch = x.dim(0)
        let channels = x.dim(1)

        var result = x
        if left > 0 {
            let leftPad = MLXArray.zeros([batch, channels, left])
            result = concatenated([leftPad, result], axis: 2)
        }
        if right > 0 {
            let rightPad = MLXArray.zeros([batch, channels, right])
            result = concatenated([result, rightPad], axis: 2)
        }

        return result
    }

    private func reflectPadNCL(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
        let length = x.dim(2)

        if length <= 1 {
            return zeroPadNCL(x, left: left, right: right)
        }

        var result = x

        if left > 0 {
            let leftPad = buildReflectPadLeft(x, padSize: left)
            result = concatenated([leftPad, result], axis: 2)
        }

        if right > 0 {
            let rightPad = buildReflectPadRight(x, padSize: right)
            result = concatenated([result, rightPad], axis: 2)
        }

        return result
    }

    private func buildReflectPadLeft(_ x: MLXArray, padSize: Int) -> MLXArray {
        let length = x.dim(2)
        let maxReflect = length - 1

        if padSize <= maxReflect {
            let slice = x[0..., 0..., 1..<(padSize + 1)]
            let indices = MLXArray((0..<padSize).reversed().map { Int32($0) })
            return slice.take(indices, axis: 2)
        } else {
            var parts: [MLXArray] = []
            let numParts = (padSize + maxReflect - 1) / maxReflect
            parts.reserveCapacity(numParts)
            var remaining = padSize

            while remaining > 0 {
                let take = min(remaining, maxReflect)
                let slice = x[0..., 0..., 1..<(take + 1)]
                let indices = MLXArray((0..<take).reversed().map { Int32($0) })
                let reversed = slice.take(indices, axis: 2)
                parts.append(reversed)
                remaining -= take
            }

            return concatenated(parts.reversed(), axis: 2)
        }
    }

    private func buildReflectPadRight(_ x: MLXArray, padSize: Int) -> MLXArray {
        let length = x.dim(2)
        let maxReflect = length - 1

        if padSize <= maxReflect {
            let startIdx = length - padSize - 1
            let endIdx = length - 1
            let slice = x[0..., 0..., startIdx..<endIdx]
            let indices = MLXArray((0..<padSize).reversed().map { Int32($0) })
            return slice.take(indices, axis: 2)
        } else {
            var parts: [MLXArray] = []
            let numParts = (padSize + maxReflect - 1) / maxReflect
            parts.reserveCapacity(numParts)
            var remaining = padSize

            while remaining > 0 {
                let take = min(remaining, maxReflect)
                let startIdx = length - take - 1
                let endIdx = length - 1
                let slice = x[0..., 0..., startIdx..<endIdx]
                let indices = MLXArray((0..<take).reversed().map { Int32($0) })
                let reversed = slice.take(indices, axis: 2)
                parts.append(reversed)
                remaining -= take
            }

            return concatenated(parts, axis: 2)
        }
    }
}

public class SConvTranspose1d: Module {
    public let convtr: ConvTransposed1d
    public let causal: Bool
    public let trimRightRatio: Float

    public let kernelSize: Int
    public let stride: Int
    public let inChannels: Int
    public let outChannels: Int
    public let paddingTotal: Int
    public let contextSize: Int

    private var layerId: ConvCacheKey?

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        bias: Bool = true,
        causal: Bool = false,
        trimRightRatio: Float = 1.0
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.causal = causal
        self.trimRightRatio = trimRightRatio

        self.paddingTotal = kernelSize - stride
        self.contextSize = kernelSize - 1

        self.convtr = ConvTransposed1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            bias: bias
        )

        super.init()
    }

    public func getLayerId() -> ConvCacheKey {
        if layerId == nil {
            layerId = ObjectIdentifier(self).hashValue
        }
        return layerId!
    }

    public func callAsFunction(
        _ x: MLXArray,
        cache: StreamingConvCache? = nil,
        useCache: Bool = false
    ) -> MLXArray {
        guard useCache, let cache = cache else {
            return forwardNonStreaming(x)
        }

        return forwardStreaming(x, cache: cache)
    }

    private func forwardNonStreaming(_ x: MLXArray) -> MLXArray {
        let nlc = x.transposed(0, 2, 1)
        let outputNLC = convtr(nlc)
        var output = outputNLC.transposed(0, 2, 1)

        let paddingRight: Int
        let paddingLeft: Int
        if causal {
            paddingRight = Int(ceil(Float(paddingTotal) * trimRightRatio))
            paddingLeft = paddingTotal - paddingRight
        } else {
            paddingRight = paddingTotal / 2
            paddingLeft = paddingTotal - paddingRight
        }

        if paddingLeft + paddingRight > 0 {
            let end = output.dim(2) - paddingRight
            output = output[0..., 0..., paddingLeft..<end]
        }

        return output
    }

    private func forwardStreaming(_ x: MLXArray, cache: StreamingConvCache) -> MLXArray {
        let layerId = getLayerId()
        let T = x.dim(2)

        let cachedInput = cache.get(layerId) ?? MLXArray.zeros([x.dim(0), inChannels, 0])

        let fullInput = concatenated([cachedInput, x], axis: 2)

        let nlc = fullInput.transposed(0, 2, 1)
        let fullOutputNLC = convtr(nlc)
        var fullOutput = fullOutputNLC.transposed(0, 2, 1)

        let paddingRight: Int
        let paddingLeft: Int
        if causal {
            paddingRight = Int(ceil(Float(paddingTotal) * trimRightRatio))
            paddingLeft = paddingTotal - paddingRight
        } else {
            paddingRight = paddingTotal / 2
            paddingLeft = paddingTotal - paddingRight
        }

        if paddingLeft + paddingRight > 0 {
            let end = fullOutput.dim(2) - paddingRight
            fullOutput = fullOutput[0..., 0..., paddingLeft..<end]
        }

        let output: MLXArray
        if cachedInput.dim(2) == 0 {
            output = fullOutput
        } else {
            let expectedNewOutput = T * stride
            if fullOutput.dim(2) >= expectedNewOutput {
                let start = fullOutput.dim(2) - expectedNewOutput
                output = fullOutput[0..., 0..., start...]
            } else {
                output = fullOutput
            }
        }

        if fullInput.dim(2) > contextSize {
            let start = fullInput.dim(2) - contextSize
            cache.set(layerId, fullInput[0..., 0..., start...])
        } else {
            cache.set(layerId, fullInput)
        }

        return output
    }
}
