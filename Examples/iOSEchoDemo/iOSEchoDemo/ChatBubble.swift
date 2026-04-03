import SwiftUI

struct ChatBubble: View {
    let message: ChatBubbleMessage

    var body: some View {
        if message.role == .system {
            // System info message — centered, muted
            Text(message.text)
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
                .padding(.vertical, 4)
        } else {
            let isUser = message.role == .user
            HStack {
                if isUser { Spacer(minLength: 60) }

                Text(message.text)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(isUser ? Color.accentColor : Color.gray.opacity(0.2))
                    .foregroundStyle(isUser ? .white : .primary)
                    .clipShape(RoundedRectangle(cornerRadius: 18))

                if !isUser { Spacer(minLength: 60) }
            }
            .padding(.horizontal)
        }
    }
}
