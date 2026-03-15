// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "TamagotchiDemo",
    platforms: [.iOS(.v17), .macOS(.v14)],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "TamagotchiDemo",
            dependencies: [
                .product(name: "Qwen3Chat", package: "speech-swift"),
                .product(name: "KokoroTTS", package: "speech-swift"),
                .product(name: "ParakeetASR", package: "speech-swift"),
                .product(name: "SpeechVAD", package: "speech-swift"),
                .product(name: "SpeechCore", package: "speech-swift"),
                .product(name: "AudioCommon", package: "speech-swift"),
            ],
            path: "TamagotchiDemo",
            exclude: ["Info.plist"]
        ),
    ]
)
