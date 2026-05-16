import Foundation

func printUsage() {
    print("usage: vevc-splitter -i <input.vevc> -o <output.vevc> [-maxLayer 0-2]")
}

var inputPath: String? = nil
var outputPath: String? = nil
var maxLayer: Int = 1

var i = 1
while i < CommandLine.arguments.count {
    let arg = CommandLine.arguments[i]
    if arg == "-i", i + 1 < CommandLine.arguments.count {
        inputPath = CommandLine.arguments[i + 1]
        i += 2
    } else if arg == "-o", i + 1 < CommandLine.arguments.count {
        outputPath = CommandLine.arguments[i + 1]
        i += 2
    } else if arg == "-maxLayer", i + 1 < CommandLine.arguments.count {
        if let val = Int(CommandLine.arguments[i + 1]) {
            maxLayer = val
        }
        i += 2
    } else {
        i += 1
    }
}

guard let input = inputPath, let output = outputPath else {
    printUsage()
    exit(1)
}

do {
    try runSplitter(input: input, output: output, maxLayer: maxLayer)
} catch {
    print("Error: \(error)")
    exit(1)
}
