import Foundation
import vevc

var inputPath = ""
var outPath = ""
var bitrate = 500
var zeroThreshold = 3
var gopSize = 15
var sceneThreshold = 8
var isOne = false

let args = CommandLine.arguments
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
            outPath = args[i + 1]
            i += 1
        }
    case "-b":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { bitrate = v }
            i += 1
        }
    case "-I":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { gopSize = v }
            i += 1
        }
    case "-zeroThreshold":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { zeroThreshold = v }
            i += 1
        }
    case "-sceneThreshold":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { sceneThreshold = v }
            i += 1
        }
    case "-one":
        isOne = true
    default:
        ()
    }
    i += 1
}

if inputPath.isEmpty || outPath.isEmpty {
    fputs("Usage: vevc-enc [-one] -i </path/to/input.y4m | -> -o </path/to/output.vevc | -> [-b <kilobit>] [-I <keyint>] [-zeroThreshold <threshold>] [-sceneThreshold <sad>]\n", stderr)
    exit(1)
}

do {
    let inFileHandle: FileHandle
    if inputPath == "-" {
        inFileHandle = FileHandle.standardInput
    } else {
        guard let f = FileHandle(forReadingAtPath: inputPath) else {
            fputs("Failed to read \(inputPath)\n", stderr)
            exit(1)
        }
        inFileHandle = f
    }

    let outFileHandle: FileHandle
    if outPath == "-" {
        outFileHandle = FileHandle.standardOutput
    } else {
        FileManager.default.createFile(atPath: outPath, contents: nil, attributes: nil)
        guard let f = FileHandle(forWritingAtPath: outPath) else {
            fputs("Failed to write to \(outPath)\n", stderr)
            exit(1)
        }
        outFileHandle = f
    }

    let y4mReader = try Y4MReader(fileHandle: inFileHandle)
    let encoder = vevc.Encoder(
        width: y4mReader.width,
        height: y4mReader.height,
        maxbitrate: bitrate * 1000,
        zeroThreshold: zeroThreshold,
        gopSize: gopSize,
        sceneChangeThreshold: sceneThreshold,
        isOne: isOne
    )

    var frameCount = 0
    let startTime = Date()
    
    // Write VEVH header with fpsHeader
    let fpsStr = y4mReader.fpsHeader
    if let fpsData = fpsStr.data(using: .utf8) {
        var vevh = Data([0x56, 0x45, 0x56, 0x48])
        let len = UInt32(fpsData.count)
        vevh.append(UInt8((len >> 24) & 0xFF))
        vevh.append(UInt8((len >> 16) & 0xFF))
        vevh.append(UInt8((len >> 8) & 0xFF))
        vevh.append(UInt8(len & 0xFF))
        vevh.append(fpsData)
        outFileHandle.write(vevh)
    }

    while let image = try y4mReader.readFrame() {
        let chunk = try await encoder.encode(image: image)
        outFileHandle.write(Data(chunk))
        frameCount += 1
    }

    let elapsed = Date().timeIntervalSince(startTime)
    if outPath != "-" {
        let msPerFrame = frameCount > 0 ? (elapsed * 1000 / Double(frameCount)) : 0
        let logMsg = String(format: "Encoded %d frames in %.4fms (%.4fms/frame)\n", frameCount, elapsed * 1000, msPerFrame)
        fputs(logMsg, stderr)
    }

    inFileHandle.closeFile()
    outFileHandle.closeFile()
} catch {
    fputs("Failed to encode: \(error)\n", stderr)
    exit(1)
}
