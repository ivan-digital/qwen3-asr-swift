.PHONY: build run clean

REPO_ROOT := $(shell cd ../.. && pwd)

build:
	swift build -c release --disable-sandbox
	BUILD_DIR=$$(pwd)/.build $(REPO_ROOT)/scripts/build_mlx_metallib.sh release

run: build
	.build/release/SpeechDemo

clean:
	swift package clean
