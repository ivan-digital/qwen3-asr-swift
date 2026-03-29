#if canImport(CoreML)
import CoreML
import Foundation

/// Manages embedding table lookups for Qwen3-TTS CoreML inference.
///
/// The Talker CoreML model takes pre-computed embeddings as input (not token IDs).
/// This class loads the embedding tables from safetensors and provides lookup functions.
final class EmbeddingManager {

    // Embedding tables (Float16)
    private let codecEmbedding: [Float16]      // [3072, 1024]
    private let textEmbedding: [Float16]       // [151936, 2048]
    private let textProjW1: [Float16]          // [2048, 2048]
    private let textProjB1: [Float16]          // [2048]
    private let textProjW2: [Float16]          // [1024, 2048]
    private let textProjB2: [Float16]          // [1024]
    private(set) var cpCodecEmbeddings: [[Float16]] // [16][2048, 1024]
    private let cpGroupEmbedding: [Float16]?   // [16, 1024]
    private(set) var cpLMHeadWeights: [[Float16]] = [] // [15][2048, 1024]

    let hiddenSize: Int
    let textHiddenSize: Int
    let codecVocabSize: Int

    init(embeddingsURL: URL, hiddenSize: Int = 1024, textHiddenSize: Int = 2048) throws {
        self.hiddenSize = hiddenSize
        self.textHiddenSize = textHiddenSize
        self.codecVocabSize = 3072

        // Load safetensors
        let data = try Data(contentsOf: embeddingsURL)
        let tensors = try Self.loadSafetensors(data)

        codecEmbedding = tensors["codec_embedding"]!
        textEmbedding = tensors["text_embedding"]!
        textProjW1 = tensors["text_projection.linear_fc1.weight"]!
        textProjB1 = tensors["text_projection.linear_fc1.bias"]!
        textProjW2 = tensors["text_projection.linear_fc2.weight"]!
        textProjB2 = tensors["text_projection.linear_fc2.bias"]!

        var cpEmbeds: [[Float16]] = []
        for i in 0..<16 {
            if let e = tensors["cp_codec_embedding.\(i)"] {
                cpEmbeds.append(e)
            }
        }
        cpCodecEmbeddings = cpEmbeds
        cpGroupEmbedding = tensors["cp_group_embedding"]

        // CP lm_head weights for autoregressive prediction
        var lmHeads: [[Float16]] = []
        for i in 0..<15 {
            if let w = tensors["cp_lm_head.\(i)"] {
                lmHeads.append(w)
            }
        }
        cpLMHeadWeights = lmHeads
    }

    // MARK: - Embedding Lookups

    /// Look up codec embedding for a token ID.
    /// Returns [hiddenSize] Float16 values.
    func codecEmbed(_ tokenId: Int) -> [Float16] {
        let offset = tokenId * hiddenSize
        return Array(codecEmbedding[offset..<(offset + hiddenSize)])
    }

    /// Compute text embedding + projection for a text token ID.
    /// text_embedding[id] → FC1 → SiLU → FC2 → [hiddenSize]
    func textEmbed(_ tokenId: Int) -> [Float16] {
        // Lookup: [textHiddenSize]
        let offset = tokenId * textHiddenSize
        let embed = Array(textEmbedding[offset..<(offset + textHiddenSize)])

        // FC1: [2048, 2048] @ [2048] + bias
        var h = matmulFloat16(embed, textProjW1, rows: textHiddenSize, cols: textHiddenSize)
        for i in 0..<textHiddenSize { h[i] = Float16(Float(h[i]) + Float(textProjB1[i])) }

        // SiLU
        for i in 0..<textHiddenSize {
            let x = Float(h[i])
            h[i] = Float16(x / (1.0 + exp(-x)))
        }

        // FC2: [1024, 2048] @ [2048] + bias
        var out = matmulFloat16(h, textProjW2, rows: hiddenSize, cols: textHiddenSize)
        for i in 0..<hiddenSize { out[i] = Float16(Float(out[i]) + Float(textProjB2[i])) }

        return out
    }

    /// Sum CP codec embeddings for all 15 groups given their token IDs.
    /// This is the `batchEmbedAllGroups` equivalent from the MLX version.
    /// Returns [hiddenSize] Float16.
    func cpGroupEmbedSum(_ cpTokens: [Int32]) -> [Float16] {
        var sum = [Float](repeating: 0, count: hiddenSize)
        for (groupIdx, token) in cpTokens.enumerated() {
            guard groupIdx < cpCodecEmbeddings.count else { break }
            let offset = Int(token) * hiddenSize
            guard offset + hiddenSize <= cpCodecEmbeddings[groupIdx].count else { continue }
            for j in 0..<hiddenSize {
                sum[j] += Float(cpCodecEmbeddings[groupIdx][offset + j])
            }
        }
        return sum.map { Float16($0) }
    }

    // MARK: - Private Helpers

    private func matmulFloat16(_ x: [Float16], _ w: [Float16], rows: Int, cols: Int) -> [Float16] {
        // y = W @ x where W is [rows, cols] and x is [cols]
        var result = [Float16](repeating: 0, count: rows)
        for r in 0..<rows {
            var sum: Float = 0
            for c in 0..<cols {
                sum += Float(w[r * cols + c]) * Float(x[c])
            }
            result[r] = Float16(sum)
        }
        return result
    }

    // MARK: - Safetensors Parser

    private static func loadSafetensors(_ data: Data) throws -> [String: [Float16]] {
        // Read header length (first 8 bytes, little-endian uint64)
        guard data.count >= 8 else { throw EmbeddingError.invalidFormat }
        let headerLen = data.withUnsafeBytes { $0.load(as: UInt64.self) }
        let headerEnd = 8 + Int(headerLen)
        guard headerEnd <= data.count else { throw EmbeddingError.invalidFormat }

        // Parse JSON header
        let headerData = data[8..<headerEnd]
        guard let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw EmbeddingError.invalidFormat
        }

        var result: [String: [Float16]] = [:]

        for (key, value) in header {
            guard key != "__metadata__",
                  let info = value as? [String: Any],
                  let dtype = info["dtype"] as? String,
                  let offsets = info["data_offsets"] as? [Int],
                  offsets.count == 2 else { continue }

            let dataStart = headerEnd + offsets[0]
            let dataEnd = headerEnd + offsets[1]
            guard dataEnd <= data.count else { continue }

            let tensorData = data[dataStart..<dataEnd]

            if dtype == "F16" {
                let count = tensorData.count / 2
                var floats = [Float16](repeating: 0, count: count)
                tensorData.withUnsafeBytes { ptr in
                    let src = ptr.bindMemory(to: Float16.self)
                    for i in 0..<count { floats[i] = src[i] }
                }
                result[key] = floats
            }
        }

        return result
    }

    enum EmbeddingError: Error {
        case invalidFormat
        case missingKey(String)
    }
}
#endif
