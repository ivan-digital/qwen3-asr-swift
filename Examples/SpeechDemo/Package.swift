// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SpeechDemo",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "SpeechDemo",
            dependencies: [
                .product(name: "ParakeetASR", package: "qwen3-asr-swift"),
                .product(name: "Qwen3ASR", package: "qwen3-asr-swift"),
                .product(name: "Qwen3TTS", package: "qwen3-asr-swift"),
                .product(name: "AudioCommon", package: "qwen3-asr-swift"),
            ],
            path: "SpeechDemo",
            exclude: ["SpeechDemo.entitlements", "Info.plist"]
        ),
    ]
)
