.PHONY: build debug test clean

CONFIG ?= release

build:
	swift build -c release --disable-sandbox
	./scripts/build_mlx_metallib.sh release

debug:
	swift build -c debug --disable-sandbox
	./scripts/build_mlx_metallib.sh debug

test: debug
	swift test --filter "WAVParsingSecurityTests|DownloadSecurityTests|MetallibScriptTests|DERScoringTests|SpectralClusteringTests|Qwen3TTSConfigTests|CosyVoiceTTSConfigTests|SamplingTests|PersonaPlexTests|ForcedAlignerTests/testText|ForcedAlignerTests/testTimestamp|ForcedAlignerTests/testLIS|SileroVADTests/testSilero|SileroVADTests/testReflection|SileroVADTests/testProcess|SileroVADTests/testReset|SileroVADTests/testDetect|SileroVADTests/testStreaming|SileroVADTests/testVADEvent"

clean:
	swift package clean
