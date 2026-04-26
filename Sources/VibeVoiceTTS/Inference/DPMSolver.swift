import Foundation
import MLX
import MLXRandom

public enum BetaSchedule: String, Codable, Sendable {
    case cosine
    case squaredcosCaptV2 = "squaredcos_cap_v2"
    case linear
}

public enum PredictionType: String, Codable, Sendable {
    case epsilon
    case vPrediction = "v_prediction"
    case sample
}

public enum AlgorithmType: String, Codable, Sendable {
    case dpmsolverPlusPlus = "dpmsolver++"
}

public enum AlphaTransformType: String, Sendable {
    case cosine
    case exp
}

public enum FinalSigmasType: String, Codable, Sendable {
    case zero
    case sigma_min
}

public func betasForAlphaBar(numDiffusionTimesteps: Int, maxBeta: Float = 0.999, alphaTransformType: AlphaTransformType = .cosine) throws -> MLXArray {
    var betas: [Float] = []

    for i in 0..<numDiffusionTimesteps {
        let t1 = Float(i) / Float(numDiffusionTimesteps)
        let t2 = Float(i + 1) / Float(numDiffusionTimesteps)

        let alphaBar1: Float
        let alphaBar2: Float

        switch alphaTransformType {
        case .cosine:
            let value1 = (t1 + 0.008) / 1.008 * Float.pi / 2
            let value2 = (t2 + 0.008) / 1.008 * Float.pi / 2
            alphaBar1 = pow(cos(value1), 2)
            alphaBar2 = pow(cos(value2), 2)
        case .exp:
            alphaBar1 = exp(t1 * -12.0)
            alphaBar2 = exp(t2 * -12.0)
        }

        let beta = min(1 - alphaBar2 / alphaBar1, maxBeta)
        betas.append(beta)
    }

    return MLXArray(betas)
}

public class DPMSolverMultistepScheduler {
    public let numTrainTimesteps: Int
    public let betaSchedule: BetaSchedule
    public let predictionType: PredictionType
    public let solverOrder: Int
    public let algorithmType: AlgorithmType
    public let lowerOrderFinal: Bool
    public let finalSigmasType: FinalSigmasType

    public let betas: MLXArray
    public let alphas: MLXArray
    public let alphasCumprod: MLXArray
    public let alphaT: MLXArray
    public let sigmaT: MLXArray
    public let lambdaT: MLXArray
    public let sigmas: MLXArray

    public private(set) var numInferenceSteps: Int = 0
    public private(set) var timesteps: MLXArray = MLXArray([])
    public private(set) var modelOutputs: [MLXArray?] = []
    public private(set) var lowerOrderNums: Int = 0
    public private(set) var stepIndex: Int = 0

    public private(set) var inferenceSigmas: MLXArray = MLXArray([])

    private var precomputedAlphaT: [Float] = []
    private var precomputedSigmaT: [Float] = []
    private var precomputedLambda: [Float] = []

    private var cachedInferenceSigmas: [Float] = []

    private var cachedInferenceAlphaT: [Float] = []
    private var cachedInferenceSigmaT: [Float] = []
    private var cachedInferenceLambda: [Float] = []

    public init(
        numTrainTimesteps: Int = 1000,
        betaSchedule: BetaSchedule = .cosine,
        predictionType: PredictionType = .vPrediction,
        solverOrder: Int = 2,
        algorithmType: AlgorithmType = .dpmsolverPlusPlus,
        lowerOrderFinal: Bool = true,
        finalSigmasType: FinalSigmasType = .zero
    ) throws {
        self.numTrainTimesteps = numTrainTimesteps
        self.betaSchedule = betaSchedule
        self.predictionType = predictionType
        self.solverOrder = solverOrder
        self.algorithmType = algorithmType
        self.lowerOrderFinal = lowerOrderFinal
        self.finalSigmasType = finalSigmasType

        switch betaSchedule {
        case .cosine, .squaredcosCaptV2:
            self.betas = try betasForAlphaBar(numDiffusionTimesteps: numTrainTimesteps, alphaTransformType: .cosine)
        case .linear:
            self.betas = MLXArray(stride(from: Float(0.0001), through: Float(0.02), by: Float(0.02 - 0.0001) / Float(numTrainTimesteps - 1)).map { $0 })
        }

        self.alphas = 1.0 - betas
        self.alphasCumprod = cumprod(alphas, axis: 0)
        self.alphaT = sqrt(alphasCumprod)
        self.sigmaT = sqrt(1 - alphasCumprod)
        self.lambdaT = log(alphaT) - log(sigmaT)
        self.sigmas = sqrt((1 - alphasCumprod) / alphasCumprod)

        self.modelOutputs = Array(repeating: nil, count: solverOrder)

        let alphasCumprodArray = self.alphasCumprod.asArray(Float.self)
        for i in 0..<numTrainTimesteps {
            let alpha = alphasCumprodArray[i]
            let sigma = sqrt((1 - alpha) / alpha)
            let alphaT = 1.0 / sqrt(sigma * sigma + 1.0)
            let sigmaT = sigma * alphaT
            let lambda = log(alphaT) - log(sigmaT)
            precomputedAlphaT.append(alphaT)
            precomputedSigmaT.append(sigmaT)
            precomputedLambda.append(lambda)
        }
    }

    public func setTimesteps(numInferenceSteps: Int) {
        self.numInferenceSteps = numInferenceSteps

        var timestepValues: [Int] = []
        for i in 0..<numInferenceSteps {
            let t = Float(numTrainTimesteps - 1) * (1.0 - Float(i) / Float(numInferenceSteps))
            timestepValues.append(Int(t.rounded()))
        }

        self.timesteps = MLXArray(timestepValues.map { Int32($0) })

        var inferSigmaValues: [Float] = []
        var inferAlphaTValues: [Float] = []
        var inferSigmaTValues: [Float] = []
        var inferLambdaValues: [Float] = []

        for t in timestepValues {
            let sigma = sqrt((1 - precomputedAlphaT[t] * precomputedAlphaT[t]) / (precomputedAlphaT[t] * precomputedAlphaT[t]))
            inferSigmaValues.append(sigma)

            let alphaT = 1.0 / sqrt(sigma * sigma + 1.0)
            let sigmaT = sigma * alphaT
            let lambda = log(alphaT) - log(sigmaT)
            inferAlphaTValues.append(alphaT)
            inferSigmaTValues.append(sigmaT)
            inferLambdaValues.append(lambda)
        }
        inferSigmaValues.append(0.0)
        inferAlphaTValues.append(1.0)
        inferSigmaTValues.append(0.0)
        inferLambdaValues.append(Float.infinity)

        self.inferenceSigmas = MLXArray(inferSigmaValues)
        self.cachedInferenceSigmas = inferSigmaValues
        self.cachedInferenceAlphaT = inferAlphaTValues
        self.cachedInferenceSigmaT = inferSigmaTValues
        self.cachedInferenceLambda = inferLambdaValues

        self.modelOutputs = Array(repeating: nil, count: solverOrder)
        self.lowerOrderNums = 0
        self.stepIndex = 0
    }

    private func sigmaToAlphaSigmaT(_ sigma: Float) -> (alphaT: Float, sigmaT: Float) {
        let alphaT = 1.0 / sqrt(sigma * sigma + 1.0)
        let sigmaT = sigma * alphaT
        return (alphaT, sigmaT)
    }

    public func convertModelOutputGPU(modelOutput: MLXArray, sample: MLXArray, sigmaIdx: Int) throws -> MLXArray {
        let sigma = cachedInferenceSigmas[sigmaIdx]
        let (alphaT, sigmaT) = sigmaToAlphaSigmaT(sigma)

        switch predictionType {
        case .epsilon:
            return (sample - sigmaT * modelOutput) / alphaT

        case .vPrediction:
            return alphaT * sample - sigmaT * modelOutput

        case .sample:
            return modelOutput
        }
    }

    public func stepGPU(
        modelOutput: MLXArray,
        stepIdx: Int,
        sample: MLXArray,
        prevX0: MLXArray?
    ) throws -> (sample: MLXArray, x0Pred: MLXArray) {
        let x0Pred = try convertModelOutputGPU(modelOutput: modelOutput, sample: sample, sigmaIdx: stepIdx)

        let alphaTVal = cachedInferenceAlphaT[stepIdx + 1]
        let sigmaTConv = cachedInferenceSigmaT[stepIdx + 1]
        let sigmaSConv = cachedInferenceSigmaT[stepIdx]

        let lambdaT = cachedInferenceLambda[stepIdx + 1]
        let lambdaS = cachedInferenceLambda[stepIdx]
        let h = lambdaT - lambdaS

        let lowerOrderFinalFlag = (stepIdx == numInferenceSteps - 1) &&
            ((lowerOrderFinal && numInferenceSteps < 15) || finalSigmasType == .zero)

        let useSecondOrder = !lowerOrderFinalFlag && prevX0 != nil && stepIdx > 0

        let prevSample: MLXArray
        if useSecondOrder, let prev = prevX0 {
            let lambdaS0 = cachedInferenceLambda[stepIdx]
            let lambdaS1 = cachedInferenceLambda[stepIdx - 1]

            let h0 = lambdaS0 - lambdaS1
            let r0 = h0 / h

            let D0 = x0Pred
            let D1 = (1.0 / r0) * (x0Pred - prev)

            let sigmaRatio = sigmaTConv / sigmaSConv
            let expNegH = exp(-h)
            prevSample = sigmaRatio * sample - alphaTVal * (expNegH - 1.0) * D0 -
                   0.5 * alphaTVal * (expNegH - 1.0) * D1
        } else {
            let sigmaRatio = sigmaTConv / sigmaSConv
            let expNegH = exp(-h)
            prevSample = sigmaRatio * sample - alphaTVal * (expNegH - 1.0) * x0Pred
        }

        return (prevSample, x0Pred)
    }

    public func step(
        modelOutput: MLXArray,
        timestep: Int,
        sample: MLXArray
    ) throws -> MLXArray {
        let alphaT = cachedInferenceAlphaT[stepIndex]
        let sigmaT = cachedInferenceSigmaT[stepIndex]

        let x0Pred: MLXArray
        switch predictionType {
        case .epsilon:
            x0Pred = (sample - sigmaT * modelOutput) / alphaT
        case .vPrediction:
            x0Pred = alphaT * sample - sigmaT * modelOutput
        case .sample:
            x0Pred = modelOutput
        }

        for i in stride(from: solverOrder - 1, through: 1, by: -1) {
            modelOutputs[i] = modelOutputs[i - 1]
        }
        modelOutputs[0] = x0Pred

        let lowerOrderFinalFlag = (stepIndex == numInferenceSteps - 1) &&
            ((lowerOrderFinal && numInferenceSteps < 15) || finalSigmasType == .zero)

        let order: Int
        if lowerOrderNums < 1 || lowerOrderFinalFlag {
            order = 1
        } else if solverOrder == 2 || lowerOrderNums < 2 {
            order = 2
        } else {
            order = solverOrder
        }

        let prevSample: MLXArray
        if order == 1 {
            prevSample = try dpmSolverFirstOrderUpdate(
                modelOutput: x0Pred,
                sample: sample
            )
        } else if order == 2 {
            guard let m1 = modelOutputs[1], let m0 = modelOutputs[0] else {
                throw VibeVoiceError.modelNotInitialized(component: "DPM solver model outputs")
            }
            prevSample = try dpmSolverSecondOrderUpdate(
                modelOutputList: [m1, m0],
                sample: sample
            )
        } else {
            throw VibeVoiceError.unsupportedSchedulerConfig("solver order > 2 not implemented")
        }

        if lowerOrderNums < solverOrder - 1 {
            lowerOrderNums += 1
        }
        stepIndex += 1

        return prevSample
    }

    private func dpmSolverFirstOrderUpdate(
        modelOutput: MLXArray,
        sample: MLXArray
    ) throws -> MLXArray {
        let x0 = modelOutput

        let alphaTVal = cachedInferenceAlphaT[stepIndex + 1]
        let sigmaTConv = cachedInferenceSigmaT[stepIndex + 1]
        let sigmaSConv = cachedInferenceSigmaT[stepIndex]

        let lambdaT = cachedInferenceLambda[stepIndex + 1]
        let lambdaS = cachedInferenceLambda[stepIndex]
        let h = lambdaT - lambdaS

        let sigmaRatio = sigmaTConv / sigmaSConv
        let expNegH = exp(-h)
        let result = sigmaRatio * sample - alphaTVal * (expNegH - 1.0) * x0

        return result
    }

    private func dpmSolverSecondOrderUpdate(
        modelOutputList: [MLXArray],
        sample: MLXArray
    ) throws -> MLXArray {
        let alphaTVal = cachedInferenceAlphaT[stepIndex + 1]
        let sigmaTConv = cachedInferenceSigmaT[stepIndex + 1]
        let sigmaS0Conv = cachedInferenceSigmaT[stepIndex]

        let lambdaT = cachedInferenceLambda[stepIndex + 1]
        let lambdaS0 = cachedInferenceLambda[stepIndex]
        let lambdaS1 = cachedInferenceLambda[stepIndex - 1]

        let h = lambdaT - lambdaS0
        let h0 = lambdaS0 - lambdaS1
        let r0 = h0 / h

        let D0 = modelOutputList[1]
        let D1 = (1.0 / r0) * (modelOutputList[1] - modelOutputList[0])

        let sigmaRatio = sigmaTConv / sigmaS0Conv
        let expNegH = exp(-h)
        let result = sigmaRatio * sample - alphaTVal * (expNegH - 1.0) * D0 -
               0.5 * alphaTVal * (expNegH - 1.0) * D1
        return result
    }

    public func reset() {
        modelOutputs = Array(repeating: nil, count: solverOrder)
        lowerOrderNums = 0
        stepIndex = 0
    }

    public func addNoise(originalSamples: MLXArray, noise: MLXArray, timesteps: MLXArray) -> MLXArray {
        let alphaT = self.alphaT.take(timesteps, axis: 0)
        let sigmaT = self.sigmaT.take(timesteps, axis: 0)

        let alphaExpanded = expandedDimensions(expandedDimensions(alphaT, axis: -1), axis: -1)
        let sigmaExpanded = expandedDimensions(expandedDimensions(sigmaT, axis: -1), axis: -1)

        return alphaExpanded * originalSamples + sigmaExpanded * noise
    }
}
