#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/build_mlx_metallib.sh [debug|release]

Builds MLX's Metal shader library (mlx.metallib) and places it next to the
SwiftPM-built executable output (e.g. .build/release/mlx.metallib).

If you see: "missing Metal Toolchain", run:
  xcodebuild -downloadComponent MetalToolchain
EOF
}

CONFIG="${1:-release}"
if [[ "$CONFIG" != "release" && "$CONFIG" != "debug" ]]; then
  usage
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/.build"

if [[ ! -d "$BUILD_DIR" ]]; then
  echo "error: $BUILD_DIR not found (run swift build first)" >&2
  exit 1
fi

OUT_DIR="$BUILD_DIR/$CONFIG"
if [[ ! -d "$OUT_DIR" ]]; then
  # Fallback for non-symlink layouts
  OUT_DIR="$(find "$BUILD_DIR" -maxdepth 3 -type d -path "*/$CONFIG" | head -n 1 || true)"
fi
if [[ -z "${OUT_DIR:-}" || ! -d "$OUT_DIR" ]]; then
  echo "error: failed to locate SwiftPM output dir for config=$CONFIG under $BUILD_DIR" >&2
  exit 1
fi

MLX_SWIFT_DIR="$BUILD_DIR/checkouts/mlx-swift"
KERNELS_DIR="$MLX_SWIFT_DIR/Source/Cmlx/mlx/mlx/backend/metal/kernels"

if [[ ! -d "$KERNELS_DIR" ]]; then
  echo "error: MLX kernels dir not found at $KERNELS_DIR" >&2
  echo "hint: ensure dependencies are fetched (swift build) and mlx-swift checkout exists" >&2
  exit 1
fi

METAL_SRCS=()
while IFS= read -r line; do
  METAL_SRCS+=("$line")
done < <(find "$KERNELS_DIR" -type f -name '*.metal' | LC_ALL=C sort)
if [[ "${#METAL_SRCS[@]}" -eq 0 ]]; then
  echo "error: no .metal sources found under $KERNELS_DIR" >&2
  exit 1
fi

TMPDIR_ROOT="${TMPDIR:-/tmp}"
TMP="$(mktemp -d "$TMPDIR_ROOT/mlx-metallib.XXXXXX")"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

AIR_FILES=()
METAL_FLAGS=(
  -x metal
  -Wall
  -Wextra
  -fno-fast-math
  -Wno-c++17-extensions
  -Wno-c++20-extensions
)

echo "Compiling ${#METAL_SRCS[@]} Metal sources..."
for SRC in "${METAL_SRCS[@]}"; do
  REL="${SRC#"$KERNELS_DIR/"}"
  KEY="$(printf '%s' "$REL" | shasum -a 256 | awk '{print $1}' | cut -c1-16)"
  OUT_AIR="$TMP/$KEY.air"

  if ! xcrun -sdk macosx metal "${METAL_FLAGS[@]}" -c "$SRC" -I"$KERNELS_DIR" -o "$OUT_AIR" 2>"$TMP/metal.err"; then
    if grep -q "missing Metal Toolchain" "$TMP/metal.err" 2>/dev/null; then
      echo "error: Xcode Metal Toolchain is missing." >&2
      echo "run: xcodebuild -downloadComponent MetalToolchain" >&2
    fi
    cat "$TMP/metal.err" >&2
    exit 1
  fi
  AIR_FILES+=("$OUT_AIR")
done

OUT_METALLIB="$OUT_DIR/mlx.metallib"
echo "Linking mlx.metallib -> $OUT_METALLIB"
xcrun -sdk macosx metallib "${AIR_FILES[@]}" -o "$OUT_METALLIB"

echo "OK: wrote $OUT_METALLIB"
