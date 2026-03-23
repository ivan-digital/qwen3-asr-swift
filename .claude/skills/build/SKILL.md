---
name: build
description: Build the speech-swift package (release or debug). Use when preparing for testing, benchmarking, or running demos.
disable-model-invocation: true
argument-hint: [release|debug]
allowed-tools: Bash
---

# Build

Build the package in the specified configuration. Default: release.

```bash
config="${ARGUMENTS:-release}"
if [ "$config" = "debug" ]; then
  make debug
else
  make build
fi
```

The metallib step compiles MLX Metal shaders. Without it, inference runs ~5x slower due to JIT compilation. `make build` handles this automatically.
