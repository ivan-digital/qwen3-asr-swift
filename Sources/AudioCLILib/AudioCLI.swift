import ArgumentParser

public struct AudioCLI: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "audio",
        abstract: "AI speech models for Apple Silicon",
        subcommands: [
            TranscribeCommand.self,
            AlignCommand.self,
            SpeakCommand.self,
            RespondCommand.self,
            VadCommand.self,
            VadStreamCommand.self,
            DiarizeCommand.self,
            EmbedSpeakerCommand.self,
        ]
    )

    public init() {}
}
