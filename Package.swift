// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Qwen3ASR",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    products: [
        .library(
            name: "Qwen3ASR",
            targets: ["Qwen3ASR"]
        ),
        .executable(
            name: "qwen3-asr-cli",
            targets: ["Qwen3ASRCLI"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0")
    ],
    targets: [
        .target(
            name: "Qwen3ASR",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift")
            ]
        ),
        .executableTarget(
            name: "Qwen3ASRCLI",
            dependencies: [
                "Qwen3ASR",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]
        ),
        .testTarget(
            name: "Qwen3ASRTests",
            dependencies: ["Qwen3ASR"],
            resources: [
                .copy("Resources/test_audio.wav")
            ]
        )
    ]
)
