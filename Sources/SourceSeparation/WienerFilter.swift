import Foundation

/// Multichannel Wiener EM post-filtering for stereo source separation.
///
/// Processes in windows (default 300 frames) to capture time-varying spatial
/// structure. Estimates per-source spatial covariance matrices and applies
/// 2x2 complex Wiener gain to the mixture STFT.
///
/// Reference: Open-Unmix filtering.py + model.Separator.forward (wiener_win_len)
struct WienerFilter {

    /// Apply windowed multichannel Wiener EM filtering.
    static func apply(
        targetMagsL: [[[Float]]],
        targetMagsR: [[[Float]]],
        mixRealL: [[Float]],
        mixImagL: [[Float]],
        mixRealR: [[Float]],
        mixImagR: [[Float]],
        iterations: Int = 1,
        windowLen: Int = 300
    ) -> [(realL: [[Float]], imagL: [[Float]], realR: [[Float]], imagR: [[Float]])] {
        let nSources = targetMagsL.count
        let T = targetMagsL[0].count
        let nBins = targetMagsL[0][0].count

        // Allocate output
        var outRL = [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0, count: nBins), count: T), count: nSources)
        var outIL = outRL, outRR = outRL, outIR = outRL

        // Process in windows
        var pos = 0
        while pos < T {
            let end = min(T, pos + windowLen)
            let wT = end - pos

            // Slice window
            var wTargetsL = [[[Float]]]()
            var wTargetsR = [[[Float]]]()
            for j in 0..<nSources {
                wTargetsL.append(Array(targetMagsL[j][pos..<end]))
                wTargetsR.append(Array(targetMagsR[j][pos..<end]))
            }
            let wMixRL = Array(mixRealL[pos..<end])
            let wMixIL = Array(mixImagL[pos..<end])
            let wMixRR = Array(mixRealR[pos..<end])
            let wMixIR = Array(mixImagR[pos..<end])

            // Run EM on this window
            let windowResult = emWiener(
                targetMagsL: wTargetsL, targetMagsR: wTargetsR,
                mixRealL: wMixRL, mixImagL: wMixIL,
                mixRealR: wMixRR, mixImagR: wMixIR,
                T: wT, nBins: nBins, nSources: nSources, iterations: iterations)

            // Copy results back
            for j in 0..<nSources {
                for t in 0..<wT {
                    outRL[j][pos + t] = windowResult[j].realL[t]
                    outIL[j][pos + t] = windowResult[j].imagL[t]
                    outRR[j][pos + t] = windowResult[j].realR[t]
                    outIR[j][pos + t] = windowResult[j].imagR[t]
                }
            }
            pos = end
        }

        return (0..<nSources).map { j in (outRL[j], outIL[j], outRR[j], outIR[j]) }
    }

    // MARK: - Single-window EM

    private static func emWiener(
        targetMagsL: [[[Float]]], targetMagsR: [[[Float]]],
        mixRealL: [[Float]], mixImagL: [[Float]],
        mixRealR: [[Float]], mixImagR: [[Float]],
        T: Int, nBins: Int, nSources: Int, iterations: Int
    ) -> [(realL: [[Float]], imagL: [[Float]], realR: [[Float]], imagR: [[Float]])] {
        let eps: Float = 1e-10

        // Numerical scaling
        var maxMag: Float = 1.0
        for t in 0..<T {
            for f in 0..<nBins {
                let mL = sqrt(mixRealL[t][f] * mixRealL[t][f] + mixImagL[t][f] * mixImagL[t][f])
                let mR = sqrt(mixRealR[t][f] * mixRealR[t][f] + mixImagR[t][f] * mixImagR[t][f])
                if mL > maxMag { maxMag = mL }
                if mR > maxMag { maxMag = mR }
            }
        }
        let scaleDiv = max(Float(1.0), maxMag / 10.0)
        let invScale = 1.0 / scaleDiv

        // Initial complex estimates: magnitude * exp(i * angle(mix)), scaled
        var y = [Float](repeating: 0, count: nSources * T * nBins * 4)

        for j in 0..<nSources {
            for t in 0..<T {
                for f in 0..<nBins {
                    let mL = sqrt(mixRealL[t][f] * mixRealL[t][f] + mixImagL[t][f] * mixImagL[t][f])
                    let mR = sqrt(mixRealR[t][f] * mixRealR[t][f] + mixImagR[t][f] * mixImagR[t][f])
                    let cosL = mL > eps ? mixRealL[t][f] / mL : 1.0
                    let sinL = mL > eps ? mixImagL[t][f] / mL : 0.0
                    let cosR = mR > eps ? mixRealR[t][f] / mR : 1.0
                    let sinR = mR > eps ? mixImagR[t][f] / mR : 0.0

                    let idx = ((j * T + t) * nBins + f) * 4
                    y[idx + 0] = targetMagsL[j][t][f] * invScale * cosL
                    y[idx + 1] = targetMagsL[j][t][f] * invScale * sinL
                    y[idx + 2] = targetMagsR[j][t][f] * invScale * cosR
                    y[idx + 3] = targetMagsR[j][t][f] * invScale * sinR
                }
            }
        }

        // EM iterations
        var v = [Float](repeating: 0, count: nSources * T * nBins)
        var R = [Float](repeating: 0, count: nSources * nBins * 8)

        for _ in 0..<iterations {
            // E-step: PSD
            for j in 0..<nSources {
                for t in 0..<T {
                    for f in 0..<nBins {
                        let idx = ((j * T + t) * nBins + f) * 4
                        let rL = y[idx], iL = y[idx + 1], rR = y[idx + 2], iR = y[idx + 3]
                        v[j * T * nBins + t * nBins + f] = (rL * rL + iL * iL + rR * rR + iR * iR) * 0.5
                    }
                }
            }

            // E-step: SCM
            for j in 0..<nSources {
                for f in 0..<nBins {
                    var sumV: Float = eps
                    var r00: Float = 0, r01re: Float = 0, r01im: Float = 0
                    var r10re: Float = 0, r10im: Float = 0, r11: Float = 0

                    for t in 0..<T {
                        sumV += v[j * T * nBins + t * nBins + f]
                        let idx = ((j * T + t) * nBins + f) * 4
                        let aRe = y[idx], aIm = y[idx + 1]
                        let bRe = y[idx + 2], bIm = y[idx + 3]

                        r00 += aRe * aRe + aIm * aIm
                        r01re += aRe * bRe + aIm * bIm
                        r01im += aIm * bRe - aRe * bIm
                        r10re += bRe * aRe + bIm * aIm
                        r10im += bIm * aRe - bRe * aIm
                        r11 += bRe * bRe + bIm * bIm
                    }

                    let inv = 1.0 / sumV
                    let rIdx = (j * nBins + f) * 8
                    R[rIdx] = r00 * inv; R[rIdx+1] = 0
                    R[rIdx+2] = r01re * inv; R[rIdx+3] = r01im * inv
                    R[rIdx+4] = r10re * inv; R[rIdx+5] = r10im * inv
                    R[rIdx+6] = r11 * inv; R[rIdx+7] = 0
                }
            }

            // M-step: Wiener gain
            for t in 0..<T {
                for f in 0..<nBins {
                    var c0r: Float = eps, c0i: Float = 0
                    var c1r: Float = 0, c1i: Float = 0
                    var c2r: Float = 0, c2i: Float = 0
                    var c3r: Float = eps, c3i: Float = 0

                    for j in 0..<nSources {
                        let vv = v[j * T * nBins + t * nBins + f]
                        let ri = (j * nBins + f) * 8
                        c0r += vv * R[ri]; c0i += vv * R[ri+1]
                        c1r += vv * R[ri+2]; c1i += vv * R[ri+3]
                        c2r += vv * R[ri+4]; c2i += vv * R[ri+5]
                        c3r += vv * R[ri+6]; c3i += vv * R[ri+7]
                    }

                    // 2x2 complex inverse
                    let dR = (c0r*c3r - c0i*c3i) - (c1r*c2r - c1i*c2i)
                    let dI = (c0r*c3i + c0i*c3r) - (c1r*c2i + c1i*c2r)
                    let dM2 = dR*dR + dI*dI
                    guard dM2 > eps * eps else { continue }
                    let idR = dR / dM2, idI = -dI / dM2

                    let i0r = c3r*idR - c3i*idI, i0i = c3r*idI + c3i*idR
                    let i1r = -(c1r*idR - c1i*idI), i1i = -(c1r*idI + c1i*idR)
                    let i2r = -(c2r*idR - c2i*idI), i2i = -(c2r*idI + c2i*idR)
                    let i3r = c0r*idR - c0i*idI, i3i = c0r*idI + c0i*idR

                    let xLR = mixRealL[t][f] * invScale, xLI = mixImagL[t][f] * invScale
                    let xRR = mixRealR[t][f] * invScale, xRI = mixImagR[t][f] * invScale

                    for j in 0..<nSources {
                        let vv = v[j * T * nBins + t * nBins + f]
                        let ri = (j * nBins + f) * 8
                        let g0r = vv*R[ri], g0i = vv*R[ri+1]
                        let g1r = vv*R[ri+2], g1i = vv*R[ri+3]
                        let g2r = vv*R[ri+4], g2i = vv*R[ri+5]
                        let g3r = vv*R[ri+6], g3i = vv*R[ri+7]

                        // W = G @ inv(Cxx)
                        let w0r = g0r*i0r - g0i*i0i + g1r*i2r - g1i*i2i
                        let w0i = g0r*i0i + g0i*i0r + g1r*i2i + g1i*i2r
                        let w1r = g0r*i1r - g0i*i1i + g1r*i3r - g1i*i3i
                        let w1i = g0r*i1i + g0i*i1r + g1r*i3i + g1i*i3r
                        let w2r = g2r*i0r - g2i*i0i + g3r*i2r - g3i*i2i
                        let w2i = g2r*i0i + g2i*i0r + g3r*i2i + g3i*i2r
                        let w3r = g2r*i1r - g2i*i1i + g3r*i3r - g3i*i3i
                        let w3i = g2r*i1i + g2i*i1r + g3r*i3i + g3i*i3r

                        let yi = ((j * T + t) * nBins + f) * 4
                        y[yi]   = w0r*xLR - w0i*xLI + w1r*xRR - w1i*xRI
                        y[yi+1] = w0r*xLI + w0i*xLR + w1r*xRI + w1i*xRR
                        y[yi+2] = w2r*xLR - w2i*xLI + w3r*xRR - w3i*xRI
                        y[yi+3] = w2r*xLI + w2i*xLR + w3r*xRI + w3i*xRR
                    }
                }
            }
        }

        // Extract results (scale back)
        var results = [(realL: [[Float]], imagL: [[Float]], realR: [[Float]], imagR: [[Float]])]()
        for j in 0..<nSources {
            var rL = [[Float]](repeating: [Float](repeating: 0, count: nBins), count: T)
            var iL = rL, rR = rL, iR = rL
            for t in 0..<T {
                for f in 0..<nBins {
                    let idx = ((j * T + t) * nBins + f) * 4
                    rL[t][f] = y[idx] * scaleDiv
                    iL[t][f] = y[idx+1] * scaleDiv
                    rR[t][f] = y[idx+2] * scaleDiv
                    iR[t][f] = y[idx+3] * scaleDiv
                }
            }
            results.append((rL, iL, rR, iR))
        }
        return results
    }
}
