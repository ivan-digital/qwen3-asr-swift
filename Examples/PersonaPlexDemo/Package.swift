// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "PersonaPlexDemo",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "PersonaPlexDemo",
            dependencies: [
                .product(name: "PersonaPlex", package: "qwen3-asr-swift"),
                .product(name: "Qwen3ASR", package: "qwen3-asr-swift"),
                .product(name: "AudioCommon", package: "qwen3-asr-swift"),
            ],
            path: "PersonaPlexDemo",
            exclude: ["PersonaPlexDemo.entitlements", "Info.plist"]
        ),
    ]
)
