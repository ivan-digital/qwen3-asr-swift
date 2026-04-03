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
                .product(name: "KokoroTTS", package: "Qwen3Speech"),
                .product(name: "ParakeetASR", package: "Qwen3Speech"),
                .product(name: "SpeechVAD", package: "Qwen3Speech"),
                .product(name: "SpeechCore", package: "Qwen3Speech"),
                .product(name: "AudioCommon", package: "Qwen3Speech"),
            ],
            path: "iOSEchoDemo",
            exclude: ["Info.plist"]
        ),
    ]
)
