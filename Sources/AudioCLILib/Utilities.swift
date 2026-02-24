import Foundation
import ArgumentParser

/// Run an async block from a synchronous ArgumentParser `run()` method.
public func runAsync(_ block: @escaping () async throws -> Void) throws {
    let semaphore = DispatchSemaphore(value: 0)
    var exitCode: Int32 = 0

    Task {
        do {
            try await block()
        } catch {
            print("Error: \(error)")
            exitCode = 1
        }
        semaphore.signal()
    }

    semaphore.wait()
    if exitCode != 0 {
        throw ExitCode(exitCode)
    }
}

/// Print model loading progress in a consistent format.
public func reportProgress(_ progress: Double, _ status: String) {
    print("  [\(Int(progress * 100))%] \(status)")
}

/// Format audio duration from sample count.
public func formatDuration(_ samples: Int, sampleRate: Int = 24000) -> String {
    String(format: "%.2f", Double(samples) / Double(sampleRate))
}
