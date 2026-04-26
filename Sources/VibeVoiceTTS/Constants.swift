import Foundation

public enum AudioConstants {
    public static let sampleRate: Int = 24000
}

public enum TTSConstants {
    public static let textWindowSize: Int = 5
    public static let speechWindowSize: Int = 6
    public static let lmBaseLayers: Int = 4
}

public enum TokenConstants {
    public static let speechStartId: Int = 151652
    public static let speechEndId: Int = 151653
    public static let speechDiffusionId: Int = 151654
    public static let negativeTextId: Int = 151655
    public static let eosTokenId: Int = 151643
}
