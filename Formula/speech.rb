class Speech < Formula
  desc "AI speech models for Apple Silicon — ASR, TTS, speech-to-speech"
  homepage "https://github.com/ivan-digital/qwen3-asr-swift"
  url "https://github.com/ivan-digital/qwen3-asr-swift/releases/download/v0.0.0/audio-macos-arm64.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "Apache-2.0"

  depends_on arch: :arm64
  depends_on :macos

  def install
    libexec.install "audio", "mlx.metallib"
    bin.write_exec_script libexec/"audio"
  end

  test do
    assert_match "AI speech models", shell_output("#{bin}/audio --help")
  end
end
