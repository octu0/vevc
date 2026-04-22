import Foundation
import vevc

var inputPath = ""
var outPath = ""
var bitrate = 500
var zeroThreshold = 3
var keyint = 60
var sceneThreshold = 32

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
    case "-keyint":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { keyint = v }
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
    default:
        ()
    }
    i += 1
}

if inputPath.isEmpty || outPath.isEmpty {
    fputs("Usage: vevc-enc -i </path/to/input.y4m | -> -o </path/to/output.vevc | -> [-b <kilobit>] [-keyint <keyint>] [-zeroThreshold <threshold>] [-sceneThreshold <sad>]\n", stderr)
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
    
    var fps = 30
    if y4mReader.fpsHeader.starts(with: "F") {
        let parts = y4mReader.fpsHeader.dropFirst().split(separator: ":")
        if parts.count == 2, let num = Int(parts[0]), let den = Int(parts[1]), den > 0 {
            fps = num / den
            if fps == 0 { fps = 30 }
        }
    }
    
    let encoder = vevc.VEVCEncoder(
        width: y4mReader.width,
        height: y4mReader.height,
        maxbitrate: bitrate * 1000,
        framerate: Int(fps),
        zeroThreshold: zeroThreshold,
        keyint: keyint,
        sceneChangeThreshold: sceneThreshold
    )

    var frameCount = 0
    let startTime = Date()

    let frameStream = AsyncStream<YCbCrImage> {
        do {
            return try y4mReader.readFrame()
        } catch {
            fputs("Failed to read frame: \(error)\n", stderr)
            return nil
        }
    }

    let chunkStream = encoder.encode(stream: frameStream)
    frameCount = try await Task(priority: .userInitiated) {
        var count = 0
        for try await chunk in chunkStream {
            outFileHandle.write(Data(chunk))
            count += 1
        }
        return count
    }.value

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
