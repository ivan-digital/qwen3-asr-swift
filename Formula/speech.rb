class Speech < Formula
  desc "AI speech models for Apple Silicon — ASR, TTS, speech-to-speech"
  homepage "https://github.com/ivan-digital/qwen3-asr-swift"
  url "https://github.com/ivan-digital/qwen3-asr-swift/releases/download/v0.0.2/audio-macos-arm64.tar.gz"
  sha256 "2b125ce18f607a30f9d1d5723b6afaf64f033f4ed1ff6d459971c15ff612fe14"
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
