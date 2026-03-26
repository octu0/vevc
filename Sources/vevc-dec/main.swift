import Foundation
import vevc

var inputPath = ""
var outPath = ""
var maxLayer = 2
var maxFrames = 4
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
    case "-maxLayer":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { maxLayer = v }
            i += 1
        }
    case "-maxFrames":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { maxFrames = v }
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
    fputs("Usage: vevc-dec [-one] -i </path/to/input.vevc | -> -o </path/to/output.y4m | -> [-maxLayer 0-2] [-maxFrames 1|2|4]\n", stderr)
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

    let vevcReader = VEVCReader(fileHandle: inFileHandle)
    
    let layerToUse = isOne ? 0 : maxLayer
    let decoder = vevc.Decoder(maxLayer: layerToUse)
    
    var y4mWriter: Y4MWriter? = nil

    var frameCount = 0
    let startTime = Date()

    let encodedStream = AsyncStream<[UInt8]> { continuation in
        Task {
            do {
                while let chunk = try vevcReader.readFrameChunk() {
                    continuation.yield(chunk)
                }
                continuation.finish()
            } catch {
                fputs("Failed to read chunk: \(error)\n", stderr)
                continuation.finish()
            }
        }
    }

    let imageStream = decoder.decode(stream: encodedStream)
    for try await image in imageStream {
        if y4mWriter == nil {
            let fpsHeader = "F\(vevcReader.fps):1"
            y4mWriter = try Y4MWriter(fileHandle: outFileHandle, width: image.width, height: image.height, fpsHeader: fpsHeader)
        }
        
        try y4mWriter?.writeFrame(image)
        frameCount += 1
    }

    let elapsed = Date().timeIntervalSince(startTime)
    if outPath != "-" {
        let msPerFrame = frameCount > 0 ? (elapsed * 1000 / Double(frameCount)) : 0
        let logMsg = String(format: "Decoded %d frames in %.4fms (%.4fms/frame)\n", frameCount, elapsed * 1000, msPerFrame)
        fputs(logMsg, stderr)
    }

    inFileHandle.closeFile()
    outFileHandle.closeFile()
} catch let error as vevc.DecodeError {
    fputs("Failed to decode: DecodeError \(error)\n", stderr)
    exit(1)
} catch let error as vevc.VEVCReaderError {
    fputs("Failed to decode: VEVCReaderError \(error)\n", stderr)
    exit(1)
} catch {
    fputs("Failed to decode: \(error.localizedDescription) (\(error))\n", stderr)
    exit(1)
}
