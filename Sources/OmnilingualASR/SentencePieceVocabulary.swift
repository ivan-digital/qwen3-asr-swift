import Foundation
import AudioCommon

/// Decoder wrapper around `AudioCommon.SentencePieceModel` that mirrors
/// fairseq2's `tokenizer.create_decoder(skip_special_tokens=True)` used by
/// Meta's `ASRInferencePipeline`. Strips control / unknown / unused / byte
/// pieces and the configured `bos`/`pad`/`eos`/`unk` ids, then maps the
/// SentencePiece word boundary marker `▁` (U+2581) to a regular space.
public struct OmnilingualVocabulary: Sendable {
    private let model: SentencePieceModel
    private let specialIds: Set<Int>

    public var count: Int { model.count }

    public static func load(
        from url: URL,
        tokenizer: OmnilingualConfig.Tokenizer
    ) throws -> OmnilingualVocabulary {
        let model: SentencePieceModel
        do {
            model = try SentencePieceModel(contentsOf: url)
        } catch SentencePieceModelError.emptyModel {
            throw OmnilingualVocabularyError.emptyVocabulary(url: url)
        }
        let specials: Set<Int> = [
            tokenizer.padIdx, tokenizer.bosIdx, tokenizer.eosIdx, tokenizer.unkIdx,
        ]
        return OmnilingualVocabulary(model: model, specialIds: specials)
    }

    private init(model: SentencePieceModel, specialIds: Set<Int>) {
        self.model = model
        self.specialIds = specialIds
    }

    /// Decode token ids to text, dropping special / control pieces.
    public func decode(_ ids: [Int]) -> String {
        var result = ""
        for id in ids {
            guard let piece = model[id] else { continue }
            if isSpecial(id) { continue }
            result += piece.text
        }
        return result
            .replacingOccurrences(of: "\u{2581}", with: " ")
            .trimmingCharacters(in: .whitespaces)
    }

    public func isSpecial(_ id: Int) -> Bool {
        if specialIds.contains(id) { return true }
        guard let piece = model[id] else { return false }
        return piece.isControlOrUnknown
    }
}

public enum OmnilingualVocabularyError: Error, CustomStringConvertible {
    case emptyVocabulary(url: URL)

    public var description: String {
        switch self {
        case .emptyVocabulary(let url):
            return "SentencePiece model at \(url.path) contained no pieces"
        }
    }
}
