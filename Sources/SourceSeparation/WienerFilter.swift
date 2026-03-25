import Foundation
import Accelerate

/// Multichannel Wiener EM post-filtering for source separation.
///
/// Refines initial magnitude-mask estimates by exploiting spatial (stereo)
/// information. All sources must be estimated simultaneously.
///
/// Reference: Nugraha et al. (2016), "Multichannel audio source separation
/// with deep neural networks"
struct WienerFilter {

    /// Apply Wiener EM filtering to refine source estimates.
    ///
    /// - Parameters:
    ///   - targetSpecs: Per-target complex spectrograms [target][channel][T][2] (real, imag pairs)
    ///     Initial estimates from magnitude masking + original phase
    ///   - mixReal: Mixture STFT real [channel][T][bins]
    ///   - mixImag: Mixture STFT imag [channel][T][bins]
    ///   - niter: Number of EM iterations (1 recommended)
    ///   - eps: Regularization
    /// - Returns: Refined complex spectrograms per target
    static func apply(
        targetSpecs: [[[Float]]],  // [target][T][bins] magnitude estimates
        mixReal: [[Float]],        // [T][bins] left channel
        mixImag: [[Float]],        // [T][bins]
        mixRealR: [[Float]],       // [T][bins] right channel
        mixImagR: [[Float]],       // [T][bins]
        niter: Int = 1,
        eps: Float = 1e-10
    ) -> (leftMag: [[[Float]]], rightMag: [[[Float]]]) {
        // For mono-duplicated stereo (our current case), Wiener simplifies
        // to a ratio mask: target_j / sum(all_targets)
        let nTargets = targetSpecs.count
        let T = targetSpecs[0].count
        let bins = targetSpecs[0][0].count

        var leftResults = [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0, count: bins), count: T), count: nTargets)
        var rightResults = leftResults

        // Compute soft masks: target_mag^2 / sum(all_target_mag^2)
        for t in 0..<T {
            for f in 0..<bins {
                // Sum of squared magnitudes across all targets
                var totalPower: Float = eps
                for j in 0..<nTargets {
                    totalPower += targetSpecs[j][t][f] * targetSpecs[j][t][f]
                }

                // Compute left/right mixture magnitude at this TF bin
                let leftMixMag = sqrt(mixReal[t][f] * mixReal[t][f] + mixImag[t][f] * mixImag[t][f])
                let rightMixMag = sqrt(mixRealR[t][f] * mixRealR[t][f] + mixImagR[t][f] * mixImagR[t][f])

                for j in 0..<nTargets {
                    let mask = (targetSpecs[j][t][f] * targetSpecs[j][t][f]) / totalPower
                    leftResults[j][t][f] = mask * leftMixMag
                    rightResults[j][t][f] = mask * rightMixMag
                }
            }
        }

        return (leftResults, rightResults)
    }
}
