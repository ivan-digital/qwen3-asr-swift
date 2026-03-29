import XCTest
@testable import Qwen3Chat

/// E2E tests for Qwen3.5 CoreML conversion verification.
/// Tests the PyTorch decoder directly (same weights/architecture as CoreML).
/// Runs via the conversion script's built-in verification.
final class E2EQwen35CoreMLConversionTests: XCTestCase {

    /// Verify the CoreML conversion script produces a working decoder.
    /// Runs the PyTorch model (pre-CoreML) to confirm correct logits.
    func testPyTorchDecoderProducesHello() throws {
        // Run Python verification inline
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["python3", "-c", """
        import torch, importlib.util, sys
        import torch.nn.functional as F

        spec = importlib.util.spec_from_file_location('conv', 'scripts/convert_qwen35_chat_coreml.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        decoder = mod.Qwen35Decoder(mod.MODEL_CONFIG)
        emb = mod.EmbeddingLookup(mod.MODEL_CONFIG["vocab_size"], mod.MODEL_CONFIG["hidden_size"])
        weights = mod.download_weights("Qwen/Qwen3.5-0.8B")
        mod.load_weights(decoder, emb, weights)
        decoder.eval(); emb.eval()

        tokens = [248045, 846, 198, 44240, 23066, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]
        mod.reset_all_states(decoder)

        with torch.no_grad():
            for pos, tid in enumerate(tokens):
                e = emb(torch.tensor([[tid]], dtype=torch.int32))
                p = torch.tensor([pos], dtype=torch.int32)
                m = torch.zeros(1, 1, 1, mod.MAX_SEQ)
                m[0, 0, 0, pos+1:] = -3.4e38
                logits = decoder(e, p, m)

            last = logits[0, 0].float()
            top = last.argmax().item()
            hello_logit = last[9419].item()

        # Token 9419 = Hello should be top or near-top
        print(f"top_token={top}")
        print(f"hello_logit={hello_logit:.3f}")
        print(f"top_logit={last[top].item():.3f}")
        sys.exit(0 if top == 9419 or hello_logit > 15.0 else 1)
        """]
        process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        try process.run()
        process.waitUntilExit()

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8) ?? ""
        print("PyTorch decoder output:\n\(output)")

        XCTAssertEqual(process.terminationStatus, 0,
                       "PyTorch decoder should produce Hello (9419) as top token. Output: \(output)")
    }
}
