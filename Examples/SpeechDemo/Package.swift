// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SpeechDemo",
    platforms: [.macOS(.v14), .iOS(.v17)],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "SpeechDemo",
            dependencies: [
                .product(name: "ParakeetASR", package: "speech-swift"),
                .product(name: "Qwen3ASR", package: "speech-swift"),
                .product(name: "Qwen3TTS", package: "speech-swift"),
                .product(name: "SpeechVAD", package: "speech-swift"),
                .product(name: "SpeechCore", package: "speech-swift"),
                .product(name: "AudioCommon", package: "speech-swift"),
            ],
            path: "SpeechDemo",
            exclude: ["SpeechDemo.entitlements", "Info.plist"]
        ),
    ]
)
