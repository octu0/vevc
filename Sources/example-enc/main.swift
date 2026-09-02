import Foundation
import vevc

let args = CommandLine.arguments
var bitrate = 500
var positionalArgs: [String] = []
var outPath = "a.vevc"
var zeroThreshold = 0
var keyint = 30
var sceneThreshold = 8
var isOne = false

var i = 1
while i < args.count {
    let arg = args[i]
    switch arg {
    case "-b", "--bitrate", "-bitrate":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { bitrate = v }
            i += 1
        }
    case "-o":
        if (i + 1) < args.count {
            outPath = args[i + 1]
            i += 1
        }
    case "-zero-threshold", "-zeroThreshold":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { zeroThreshold = v }
            i += 1
        }
    case "-keyint":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { keyint = v }
            i += 1
        }
    case "-scene-threshold", "-sceneThreshold":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { sceneThreshold = v }
            i += 1
        }
    default:
        positionalArgs.append(arg)
    }
    i += 1
}

if positionalArgs.isEmpty {
    print("Usage: example-enc -o <output.vevc> [-b <kbits> | --bitrate <kbits>] [-zero-threshold <threshold>] [-keyint <frames>] [-scene-threshold <sad>] <input1.y4m> [input2.y4m ...]")
    exit(1)
}

var images: [YCbCrImage] = []
for p in positionalArgs {
    let fileHandle: FileHandle
    if p == "-" {
        fileHandle = FileHandle.standardInput
    } else {
        guard let f = FileHandle(forReadingAtPath: p) else {
            print("Failed to open \(p)")
            continue
        }
        fileHandle = f
    }
    
    guard let reader = try? Y4MReader(fileHandle: fileHandle) else {
        print("Failed to read y4m header from \(p)")
        if p != "-" { fileHandle.closeFile() }
        continue
    }
    
    while let img = try? reader.readFrame() {
        images.append(img)
    }
    
    if p != "-" {
        fileHandle.closeFile()
    }
}

do {
    let startTime = Date()
    guard let first = images.first else {
        print("No images to encode")
        exit(1)
    }
    let encoder = VEVCEncoder(width: first.width, height: first.height, profile: 0x01)
    encoder.maxbitrate = bitrate * 1000
    encoder.zeroThreshold = zeroThreshold
    encoder.keyint = keyint
    encoder.sceneChangeThreshold = sceneThreshold
    let out = try await encoder.encodeToData(images: images)
    let elapsed = Date().timeIntervalSince(startTime)
    
    let dataSize: Int
    if images.isEmpty != true {
        let first = images[0]
        dataSize = images.count * first.width * first.height * 3
    } else {
        dataSize = 0
    }
    
    print(String(
        format:"elapse= %.4fms (%.4fms/frame) %3.2fKB -> %3.2fKB compressed %3.2f%%",
        elapsed * 1000,
        elapsed * 1000 / Double(images.count),
        Double(dataSize) / 1024.0,
        Double(out.count) / 1024.0,
        Double(out.count) / Double(dataSize) * 100.0
    ))
    
    if FileManager.default.createFile(atPath: outPath, contents: Data(out)) {
        print("Successfully encoded \(images.count) frames to \(outPath)")
    } else {
        print("Failed to write \(outPath)")
        exit(1)
    }
} catch {
    print("Failed to encode: \(error)")
    exit(1)
}