import SwiftUI

public struct PlayerArguments {
    public var isCompareMode: Bool = false
    public var inputPath: String? = nil
    public var bitrate: Int = 1000
    public var profile: UInt8 = 1
    
    public static func parse() -> PlayerArguments {
        var args = PlayerArguments()
        let argv = CommandLine.arguments
        
        var i = 1
        while i < argv.count {
            let arg = argv[i]
            if arg == "-compare" || arg == "--compare" {
                args.isCompareMode = true
            } else if arg == "-y4m" || arg == "--y4m" || arg == "-i" || arg == "--input" {
                if i + 1 < argv.count {
                    i += 1
                    args.inputPath = argv[i]
                }
            } else if arg == "-bitrate" || arg == "--bitrate" || arg == "-b" {
                if i + 1 < argv.count, let br = Int(argv[i]) {
                    i += 1
                    args.bitrate = br
                }
            } else if arg == "-profile" || arg == "--profile" || arg == "-p" {
                if i + 1 < argv.count, let p = UInt8(argv[i]) {
                    i += 1
                    args.profile = p
                }
            } else if !arg.starts(with: "-") {
                args.inputPath = arg
            }
            i += 1
        }
        return args
    }
}

@main
struct VEVCMacOSApp: App {
    private let args = PlayerArguments.parse()
    
    var body: some Scene {
        WindowGroup {
            ContentView(args: args)
        }
    }
}
