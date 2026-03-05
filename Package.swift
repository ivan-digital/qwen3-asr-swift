// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Qwen3Speech",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
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
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.5.0")
    ],
    targets: [
        .target(
            name: "AudioCommon",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers")
            ]
        ),
        .target(
            name: "Qwen3ASR",
            dependencies: [
                "AudioCommon",
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
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift")
            ]
        ),
        .target(
            name: "CosyVoiceTTS",
            dependencies: [
                "AudioCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift")
            ]
        ),
        .target(
            name: "PersonaPlex",
            dependencies: [
                "AudioCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift")
            ]
        ),
        .target(
            name: "SpeechVAD",
            dependencies: [
                "AudioCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "SpeechEnhancement",
            dependencies: [
                "AudioCommon",
            ]
        ),
        .target(
            name: "ParakeetASR",
            dependencies: [
                "AudioCommon",
            ]
        ),
        .target(
            name: "AudioCLILib",
            dependencies: [
                "Qwen3ASR",
                "Qwen3TTS",
                "CosyVoiceTTS",
                "PersonaPlex",
                "SpeechVAD",
                "SpeechEnhancement",
                "ParakeetASR",
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
                .product(name: "Hummingbird", package: "hummingbird")
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
                .copy("Resources/test_audio.wav")
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
            name: "AudioCLITests",
            dependencies: [
                "AudioCLILib",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]
        )
    ]
)
