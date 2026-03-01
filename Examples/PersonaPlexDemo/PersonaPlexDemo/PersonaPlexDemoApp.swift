import SwiftUI
import PersonaPlex

@main
struct PersonaPlexDemoApp: App {
    init() {
        // If --test-warmup flag is passed, test warmUp and exit
        if CommandLine.arguments.contains("--test-warmup") {
            Task {
                do {
                    print("Loading model...")
                    let m = try await PersonaPlexModel.fromPretrained { p, s in
                        print("  \(s) (\(Int(p * 100))%)")
                    }
                    print("Calling warmUp()...")
                    m.warmUp()
                    print("warmUp() succeeded!")
                } catch {
                    print("Error: \(error)")
                }
                exit(0)
            }
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 300))
        }
    }

    var body: some Scene {
        WindowGroup {
            PersonaPlexView()
        }
        .defaultSize(width: 600, height: 700)
    }
}
