import SwiftUI

@main
struct SpeechDemoApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    var body: some View {
        TabView {
            DictateView()
                .tabItem { Label("Dictate", systemImage: "mic.fill") }
            SpeakView()
                .tabItem { Label("Speak", systemImage: "speaker.wave.2.fill") }
            #if os(macOS)
            EchoView()
                .tabItem { Label("Echo", systemImage: "waveform.path.ecg") }
            #endif
        }
        #if os(macOS)
        .frame(minWidth: 500, minHeight: 400)
        #endif
    }
}
