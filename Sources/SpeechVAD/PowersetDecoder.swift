import MLX

/// Decodes pyannote's 7-class powerset output to per-speaker probabilities.
///
/// The 7 powerset classes represent all possible speaker combinations
/// for up to 3 speakers:
/// - 0: non-speech
/// - 1: speaker 1 alone
/// - 2: speaker 2 alone
/// - 3: speaker 3 alone
/// - 4: speakers 1+2 overlap
/// - 5: speakers 1+3 overlap
/// - 6: speakers 2+3 overlap
enum PowersetDecoder {

    /// Convert 7-class powerset posteriors to per-speaker probabilities.
    ///
    /// Each speaker's probability is the sum of all classes where that speaker
    /// is active (alone or in overlap).
    ///
    /// - Parameter posteriors: `[batch, frames, 7]` softmax probabilities
    /// - Returns: `[batch, frames, 3]` per-speaker probabilities
    static func speakerProbabilities(from posteriors: MLXArray) -> MLXArray {
        // spk1: alone(1) + with_spk2(4) + with_spk3(5)
        let spk1 = posteriors[0..., 0..., 1] + posteriors[0..., 0..., 4] + posteriors[0..., 0..., 5]
        // spk2: alone(2) + with_spk1(4) + with_spk3(6)
        let spk2 = posteriors[0..., 0..., 2] + posteriors[0..., 0..., 4] + posteriors[0..., 0..., 6]
        // spk3: alone(3) + with_spk1(5) + with_spk2(6)
        let spk3 = posteriors[0..., 0..., 3] + posteriors[0..., 0..., 5] + posteriors[0..., 0..., 6]
        return stacked([spk1, spk2, spk3], axis: -1)
    }

    /// Apply hysteresis binarization to per-speaker probabilities.
    ///
    /// - Parameters:
    ///   - probs: per-frame probabilities for one speaker `[frames]`
    ///   - onset: threshold to start a segment
    ///   - offset: threshold to end a segment
    ///   - frameDuration: duration of one frame in seconds
    /// - Returns: array of (startTime, endTime) tuples
    static func binarize(
        probs: [Float],
        onset: Float,
        offset: Float,
        frameDuration: Float
    ) -> [(startTime: Float, endTime: Float)] {
        var segments = [(startTime: Float, endTime: Float)]()
        var inSpeech = false
        var speechStart: Float = 0

        for (i, prob) in probs.enumerated() {
            let time = Float(i) * frameDuration

            if !inSpeech && prob >= onset {
                inSpeech = true
                speechStart = time
            } else if inSpeech && prob < offset {
                inSpeech = false
                segments.append((speechStart, time))
            }
        }

        if inSpeech {
            let endTime = Float(probs.count) * frameDuration
            segments.append((speechStart, endTime))
        }

        return segments
    }
}
