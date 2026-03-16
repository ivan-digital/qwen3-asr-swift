class Speech < Formula
  desc "AI speech models for Apple Silicon — ASR, TTS, speech-to-speech"
  homepage "https://github.com/soniqo/speech-swift"
  url "https://github.com/soniqo/speech-swift/releases/download/v0.0.5/audio-macos-arm64.tar.gz"
  sha256 "104f3bf9c0b033194a76f71acbf9a2762a567165c7ac4dde1dc904261cbf6bc5"
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
