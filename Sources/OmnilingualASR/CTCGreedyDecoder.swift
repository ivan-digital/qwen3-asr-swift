import Foundation

/// Greedy CTC decoder matching Meta's `ASRInferencePipeline` pipeline.py:
///
/// ```python
/// pred_ids = torch.argmax(logits, dim=-1)           # [B, T]
/// seq = pred_ids[i][: seq_lens[i]]
/// mask[1:] = seq[1:] != seq[:-1]                    # collapse consecutive duplicates
/// decoded_ids = seq[mask]
/// text = tokenizer.create_decoder(skip_special_tokens=True)(decoded_ids)
/// ```
///
/// There is intentionally **no explicit blank-token filtering**. The CTC blank
/// id in Omnilingual is tied to the SentencePiece `pad` id (1), and it is
/// stripped by the vocabulary's `skip_special_tokens` behavior at detokenize
/// time — not here.
public struct CTCGreedyDecoder {

    /// Run greedy CTC over logits `[T, V]` and return the collapsed token id
    /// sequence. The caller is responsible for final detokenization and
    /// special-token skipping.
    ///
    /// - Parameters:
    ///   - logits: Row-major `[T, V]` float logits. The argmax over `V` is
    ///     taken per frame.
    ///   - validFrames: Optional prefix length — only the first `validFrames`
    ///     rows are considered. Pass nil to decode all rows.
    public static func decode(logits: [Float], timeSteps T: Int, vocabSize V: Int, validFrames: Int? = nil) -> [Int] {
        precondition(logits.count == T * V, "logits.count=\(logits.count) does not match T*V=\(T*V)")
        let frames = min(validFrames ?? T, T)
        guard frames > 0 else { return [] }

        var collapsed: [Int] = []
        collapsed.reserveCapacity(frames)
        var previous: Int = -1

        for t in 0..<frames {
            let base = t * V
            var bestIndex = 0
            var bestValue = logits[base]
            for v in 1..<V {
                let value = logits[base + v]
                if value > bestValue {
                    bestValue = value
                    bestIndex = v
                }
            }
            // Collapse consecutive duplicates.
            if bestIndex != previous {
                collapsed.append(bestIndex)
                previous = bestIndex
            }
        }
        return collapsed
    }
}
