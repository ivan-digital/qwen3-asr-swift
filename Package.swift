// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "Qwen3Speech",
    platforms: [
        .macOS("15.0"),
        .iOS("18.0")
    ],
    products: [
        .library(
            name: "Qwen3ASR",
            targets: ["Qwen3ASR"]
        ),
        .library(
            name: "Qwen3TTS",
            targets: ["Qwen3TTS"]
        ),
        .library(
            name: "AudioCommon",
            targets: ["AudioCommon"]
        ),
        .library(
            name: "CosyVoiceTTS",
            targets: ["CosyVoiceTTS"]
        ),
        .library(
            name: "PersonaPlex",
            targets: ["PersonaPlex"]
        ),
        .library(
            name: "SpeechVAD",
            targets: ["SpeechVAD"]
        ),
        .library(
            name: "SpeechEnhancement",
            targets: ["SpeechEnhancement"]
        ),
        .library(
            name: "ParakeetASR",
            targets: ["ParakeetASR"]
        ),
        .library(
            name: "ParakeetStreamingASR",
            targets: ["ParakeetStreamingASR"]
        ),
        .library(
            name: "SpeechCore",
            targets: ["SpeechCore"]
        ),
        .library(
            name: "KokoroTTS",
            targets: ["KokoroTTS"]
        ),
        .library(
            name: "Qwen3TTSCoreML",
            targets: ["Qwen3TTSCoreML"]
        ),
        .library(
            name: "Qwen3Chat",
            targets: ["Qwen3Chat"]
        ),
        .library(
            name: "SpeechUI",
            targets: ["SpeechUI"]
        ),
        .executable(
            name: "audio",
            targets: ["AudioCLI"]
        ),
        .executable(
            name: "audio-server",
            targets: ["AudioServerCLI"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", "2.5.0"..<"2.17.0"),
        .package(url: "https://github.com/hummingbird-project/hummingbird-websocket.git", from: "2.6.0")
    ],
    targets: [
        .target(
            name: "AudioCommon",
            dependencies: [
                .product(name: "Hub", package: "swift-transformers")
            ]
        ),
        .target(
            name: "MLXCommon",
            dependencies: [
                "AudioCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "Qwen3ASR",
            dependencies: [
                "AudioCommon",
                "MLXCommon",
                "SpeechVAD",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift")
            ]
        ),
        .target(
            name: "Qwen3TTS",
            dependencies: [
                "AudioCommon",
                "MLXCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift")
            ]
        ),
        .target(
            name: "Qwen3TTSCoreML",
            dependencies: [
                "AudioCommon",
            ]
        ),
        .target(
            name: "CosyVoiceTTS",
            dependencies: [
                "AudioCommon",
                "MLXCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift")
            ]
        ),
        .target(
            name: "PersonaPlex",
            dependencies: [
                "AudioCommon",
                "MLXCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift")
            ]
        ),
        .target(
            name: "SpeechVAD",
            dependencies: [
                "AudioCommon",
                "MLXCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "SpeechEnhancement",
            dependencies: [
                "AudioCommon",
                "MLXCommon",
            ]
        ),
        .target(
            name: "ParakeetASR",
            dependencies: [
                "AudioCommon",
            ]
        ),
        .target(
            name: "ParakeetStreamingASR",
            dependencies: [
                "AudioCommon",
            ]
        ),
        .binaryTarget(
            name: "CSpeechCore",
            url: "https://github.com/soniqo/speech-core/releases/download/v0.0.5/SpeechCore.xcframework.zip",
            checksum: "ec87ae9191875390e19cd2aa74bfdbd8f314c4a0e83dfe012eed4d1ca30c4a5d"
        ),
        .target(
            name: "SpeechCore",
            dependencies: [
                "CSpeechCore",
                "AudioCommon",
            ]
        ),
        .target(
            name: "KokoroTTS",
            dependencies: [
                "AudioCommon",
            ],
            resources: [
                .copy("Resources"),
            ]
        ),
        .target(
            name: "Qwen3Chat",
            dependencies: [
                "AudioCommon",
                "MLXCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "SpeechUI",
            dependencies: []
        ),
        .target(
            name: "AudioCLILib",
            dependencies: [
                "Qwen3ASR",
                "Qwen3TTS",
                "CosyVoiceTTS",
                "Qwen3TTSCoreML",
                "PersonaPlex",
                "SpeechVAD",
                "SpeechEnhancement",
                "ParakeetASR",
                "ParakeetStreamingASR",
                "KokoroTTS",
                "AudioCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]
        ),
        .executableTarget(
            name: "AudioCLI",
            dependencies: ["AudioCLILib"]
        ),
        .target(
            name: "AudioServer",
            dependencies: [
                "Qwen3ASR",
                "Qwen3TTS",
                "CosyVoiceTTS",
                "PersonaPlex",
                "SpeechEnhancement",
                "AudioCommon",
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "HummingbirdWebSocket", package: "hummingbird-websocket")
            ]
        ),
        .executableTarget(
            name: "AudioServerCLI",
            dependencies: [
                "AudioServer",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]
        ),
        .testTarget(
            name: "PersonaPlexTests",
            dependencies: ["PersonaPlex", "AudioCommon", "Qwen3ASR"]
        ),
        .testTarget(
            name: "Qwen3ASRTests",
            dependencies: ["Qwen3ASR", "SpeechVAD", "AudioCommon"],
            resources: [
                .copy("Resources/test_audio.wav")
            ]
        ),
        .testTarget(
            name: "Qwen3TTSTests",
            dependencies: ["Qwen3TTS", "Qwen3ASR", "AudioCommon"]
        ),
        .testTarget(
            name: "Qwen3TTSCoreMLTests",
            dependencies: ["Qwen3TTSCoreML", "Qwen3ASR", "AudioCommon"]
        ),
        .testTarget(
            name: "CosyVoiceTTSTests",
            dependencies: ["CosyVoiceTTS", "AudioCommon"]
        ),
        .testTarget(
            name: "SpeechVADTests",
            dependencies: [
                "SpeechVAD",
                "AudioCommon",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "ParakeetASRTests",
            dependencies: ["ParakeetASR", "AudioCommon"],
            resources: [
                .copy("Resources/test_audio.wav"),
                .copy("Resources/test_audio_german.wav")
            ]
        ),
        .testTarget(
            name: "ParakeetStreamingASRTests",
            dependencies: ["ParakeetStreamingASR", "AudioCommon"],
            resources: [
                .copy("Resources/test_audio.wav")
            ]
        ),
        .testTarget(
            name: "AudioCommonTests",
            dependencies: [
                "AudioCommon",
            ]
        ),
        .testTarget(
            name: "KokoroTTSTests",
            dependencies: [
                "KokoroTTS",
                "AudioCommon",
                "Qwen3ASR",
            ]
        ),
        .testTarget(
            name: "SpeechEnhancementTests",
            dependencies: [
                "SpeechEnhancement",
                "AudioCommon",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "Qwen3ChatTests",
            dependencies: [
                "Qwen3Chat",
                "AudioCommon",
            ]
        ),
        .testTarget(
            name: "AudioCLITests",
            dependencies: [
                "AudioCLILib",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]
        ),
        .testTarget(
            name: "AudioServerTests",
            dependencies: [
                "AudioServer"
            ]
        ),
        .testTarget(
            name: "SpeechCoreTests",
            dependencies: [
                "SpeechCore",
                "AudioCommon",
                "SpeechVAD",
                "KokoroTTS",
                "ParakeetASR"
            ]
        ),
        .testTarget(
            name: "SpeechUITests",
            dependencies: [
                "SpeechUI",
                "ParakeetStreamingASR",
                "AudioCommon"
            ],
            resources: [
                .copy("Resources/test_audio.wav")
            ]
        )
    ]
)
