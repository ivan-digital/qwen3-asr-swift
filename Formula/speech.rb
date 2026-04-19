class Speech < Formula
  desc "AI speech models for Apple Silicon — ASR, TTS, speech-to-speech"
  homepage "https://github.com/soniqo/speech-swift"
  url "https://github.com/soniqo/speech-swift/releases/download/v0.0.10/audio-macos-arm64.tar.gz"
  sha256 "774f0f8713299aeb70d252e18cc25bb3d4922f613e7eccdedc5404366dc4e7ed"
  license "Apache-2.0"

  depends_on arch: :arm64
  depends_on :macos

  def install
    libexec.install "audio", "audio-server", "mlx.metallib"
    libexec.install "Qwen3Speech_KokoroTTS.bundle"
    bin.write_exec_script libexec/"audio"
    bin.write_exec_script libexec/"audio-server"
  end

  test do
    assert_match "AI speech models", shell_output("#{bin}/audio --help")
    assert_match "HTTP API server", shell_output("#{bin}/audio-server --help")
  end
end
