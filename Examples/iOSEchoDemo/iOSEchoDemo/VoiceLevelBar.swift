import SwiftUI

struct VoiceLevelBar: View {
    let level: Float

    var body: some View {
        GeometryReader { geo in
            let width = geo.size.width * CGFloat(min(level * 5, 1.0))
            RoundedRectangle(cornerRadius: 2)
                .fill(barColor)
                .frame(width: max(width, 2))
                .animation(.easeOut(duration: 0.1), value: level)
        }
    }

    private var barColor: Color {
        if level > 0.15 { return .red }
        if level > 0.05 { return .orange }
        return .green
    }
}

struct TypingIndicator: View {
    @State private var dot = 0

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3) { i in
                Circle()
                    .fill(Color.secondary)
                    .frame(width: 8, height: 8)
                    .opacity(dot == i ? 1 : 0.3)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .background(Color.gray.opacity(0.2))
        .clipShape(RoundedRectangle(cornerRadius: 18))
        .onAppear {
            Timer.scheduledTimer(withTimeInterval: 0.4, repeats: true) { _ in
                withAnimation { dot = (dot + 1) % 3 }
            }
        }
    }
}
