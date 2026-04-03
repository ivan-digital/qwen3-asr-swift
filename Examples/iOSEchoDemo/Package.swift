// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "iOSEchoDemo",
    platforms: [.iOS("18.0"), .macOS("15.0")],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "iOSEchoDemo",
            dependencies: [
                .product(name: "KokoroTTS", package: "qwen3-asr-swift"),
                .product(name: "Qwen3TTSCoreML", package: "qwen3-asr-swift"),
                .product(name: "ParakeetASR", package: "qwen3-asr-swift"),
                .product(name: "SpeechVAD", package: "qwen3-asr-swift"),
                .product(name: "SpeechCore", package: "qwen3-asr-swift"),
                .product(name: "AudioCommon", package: "qwen3-asr-swift"),
            ],
            path: "iOSEchoDemo",
            exclude: ["Info.plist"]
        ),
    ]
)
