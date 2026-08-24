import Foundation
import vevc

let args = CommandLine.arguments
var inputPath = ""
var outDir = ".out/"
var maxLayer = 2
var maxFrames = 4

var i = 1
while i < args.count {
    let arg = args[i]
    switch arg {
    case "-i":
        if (i + 1) < args.count {
            inputPath = args[i + 1]
            i += 1
        }
    case "-o":
        if (i + 1) < args.count {
            outDir = args[i + 1]
            i += 1
        }
    case "-max-layer", "-maxLayer":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { maxLayer = v }
            i += 1
        }
    case "-max-frames", "-maxFrames":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { maxFrames = v }
            i += 1
        }
    default:
        ()
    }
    i += 1
}

if inputPath.isEmpty {
    print("Usage: example-dec -i <input.vevc> [-o <output_dir>] [-max-layer 0-2] [-max-frames 1|2|4]")
    exit(1)
}

guard let inputData = try? Data(contentsOf: URL(fileURLWithPath: inputPath)) else {
    print("Failed to read \(inputPath)")
    exit(1)
}

do {
    let startTime = Date()
    let images: [YCbCrImage]
    let decoder = Decoder(maxLayer: maxLayer)
    images = try await decoder.decode(data: Array(inputData))
    let elapsed = Date().timeIntervalSince(startTime)
    print(String(
        format: "Decoded %d frames in %.4fms (%.4fms/frame)",
        images.count,
        elapsed * 1000,
        elapsed * 1000 / Double(images.count)
    ))
    
    let outputURL: URL = URL(fileURLWithPath: outDir).standardized

    try? FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)
    
    let outputFile = outputURL.appendingPathComponent("output.y4m")
    FileManager.default.createFile(atPath: outputFile.path, contents: nil)
    if let fileHandle = FileHandle(forWritingAtPath: outputFile.path), let first = images.first {
        let writer = try Y4MWriter(fileHandle: fileHandle, width: first.width, height: first.height, fpsHeader: "F60:1")
        for img in images {
            try writer.writeFrame(img)
        }
        fileHandle.closeFile()
        print("Saved \(outputFile.path)")
    }
} catch {
    print("Failed to decode: \(error)")
    exit(1)
}