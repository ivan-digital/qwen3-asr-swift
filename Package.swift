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
            name: "Qwen3Common",
            targets: ["Qwen3Common"]
        ),
        .library(
            name: "CosyVoiceTTS",
            targets: ["CosyVoiceTTS"]
        ),
        .executable(
            name: "qwen3-asr-cli",
            targets: ["Qwen3ASRCLI"]
        ),
        .executable(
            name: "qwen3-tts-cli",
            targets: ["Qwen3TTSCLI"]
        ),
        .executable(
            name: "cosyvoice-tts-cli",
            targets: ["CosyVoiceTTSCLI"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6")
    ],
    targets: [
        .target(
            name: "Qwen3Common",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift")
            ]
        ),
        .target(
            name: "Qwen3ASR",
            dependencies: [
                "Qwen3Common",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift")
            ]
        ),
        .target(
            name: "Qwen3TTS",
            dependencies: [
                "Qwen3Common",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift")
            ]
        ),
        .target(
            name: "CosyVoiceTTS",
            dependencies: [
                "Qwen3Common",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers")
            ]
        ),
        .executableTarget(
            name: "Qwen3ASRCLI",
            dependencies: [
                "Qwen3ASR",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]
        ),
        .executableTarget(
            name: "Qwen3TTSCLI",
            dependencies: [
                "Qwen3TTS",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]
        ),
        .executableTarget(
            name: "CosyVoiceTTSCLI",
            dependencies: [
                "CosyVoiceTTS",
                "Qwen3Common",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]
        ),
        .testTarget(
            name: "Qwen3ASRTests",
            dependencies: ["Qwen3ASR", "Qwen3Common"],
            resources: [
                .copy("Resources/test_audio.wav")
            ]
        ),
        .testTarget(
            name: "Qwen3TTSTests",
            dependencies: ["Qwen3TTS", "Qwen3ASR", "Qwen3Common"]
        ),
        .testTarget(
            name: "CosyVoiceTTSTests",
            dependencies: ["CosyVoiceTTS", "Qwen3Common"]
        )
    ]
)
