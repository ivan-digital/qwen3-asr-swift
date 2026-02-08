#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== Qwen3-ASR Integration Tests ==="

# Step 1: Build the package
echo "[1/4] Building package..."
swift build 2>&1

# Step 2: Build MLX metallib (required for Metal GPU operations)
echo "[2/4] Building MLX metallib..."
"$ROOT/scripts/build_mlx_metallib.sh" debug

# Step 3: Copy metallib to test binary location
echo "[3/4] Copying metallib to test bundle..."
ARCH="$(uname -m)-apple-macosx"
TEST_BUNDLE_DIR="$ROOT/.build/$ARCH/debug/Qwen3ASRPackageTests.xctest/Contents/MacOS"

# Build tests first to ensure the bundle exists
swift build --build-tests 2>&1

if [[ -d "$TEST_BUNDLE_DIR" ]]; then
  cp "$ROOT/.build/debug/mlx.metallib" "$TEST_BUNDLE_DIR/mlx.metallib"
  echo "  Copied to $TEST_BUNDLE_DIR"
else
  echo "  Warning: test bundle not found at $TEST_BUNDLE_DIR"
  echo "  Tests requiring MLX will fail"
fi

# Step 4: Run tests
echo "[4/4] Running tests..."
swift test 2>&1

echo ""
echo "=== All tests passed ==="
