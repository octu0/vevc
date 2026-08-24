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
            switch arg {
            case "-compare", "--compare":
                args.isCompareMode = true
            case "-y4m", "--y4m", "-i", "--input":
                if i + 1 < argv.count {
                    args.inputPath = argv[i + 1]
                    i += 1
                }
            case "-b", "--bitrate", "-bitrate":
                if i + 1 < argv.count {
                    if let br = Int(argv[i + 1]) {
                        args.bitrate = br
                    }
                    i += 1
                }
            case "-p", "--profile", "-profile":
                if i + 1 < argv.count {
                    if let p = UInt8(argv[i + 1]) {
                        args.profile = p
                    }
                    i += 1
                }
            default:
                if arg.starts(with: "-") != true {
                    args.inputPath = arg
                }
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
