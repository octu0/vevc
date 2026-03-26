import Foundation
import vevc

var inputPath = ""
var outPath = ""
var bitrate = 500
var zeroThreshold = 3
var keyint = 60
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
    case "-one":
        isOne = true
    default:
        ()
    }
    i += 1
}

if inputPath.isEmpty || outPath.isEmpty {
    fputs("Usage: vevc-enc [-one] -i </path/to/input.y4m | -> -o </path/to/output.vevc | -> [-b <kilobit>] [-keyint <keyint>] [-zeroThreshold <threshold>] [-sceneThreshold <sad>]\n", stderr)
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
    
    let encoder = vevc.Encoder(
        width: y4mReader.width,
        height: y4mReader.height,
        maxbitrate: bitrate * 1000,
        framerate: fps,
        zeroThreshold: zeroThreshold,
        keyint: keyint,
        sceneChangeThreshold: sceneThreshold,
        isOne: isOne
    )

    var frameCount = 0
    let startTime = Date()
    
    // Write VEVC file header: magic(4B) + dataSize(4B) = 8 bytes
    // dataSize is placeholder 0, updated after encoding completes
    var vevcHeader = Data([0x56, 0x45, 0x56, 0x43]) // VEVC
    vevcHeader.append(contentsOf: [0x00, 0x00, 0x00, 0x00]) // DataSize placeholder
    outFileHandle.write(vevcHeader)
    
    // Write Metadata: metadataSize(2B) + profile(1B) + width(2B) + height(2B) + colorGamut(1B) + fps(2B) + timescale(1B)
    var metadata = Data()
    let metadataPayloadSize: UInt16 = 9 // Profile(1) + Width(2) + Height(2) + ColorGamut(1) + FPS(2) + Timescale(1)
    metadata.append(UInt8((metadataPayloadSize >> 8) & 0xFF))
    metadata.append(UInt8(metadataPayloadSize & 0xFF))
    metadata.append(0x01) // Profile 1
    let w = UInt16(y4mReader.width)
    metadata.append(UInt8((w >> 8) & 0xFF))
    metadata.append(UInt8(w & 0xFF))
    let h = UInt16(y4mReader.height)
    metadata.append(UInt8((h >> 8) & 0xFF))
    metadata.append(UInt8(h & 0xFF))
    metadata.append(0x01) // ColorGamut: BT.709
    let fpsValue = UInt16(fps)
    metadata.append(UInt8((fpsValue >> 8) & 0xFF))
    metadata.append(UInt8(fpsValue & 0xFF))
    metadata.append(0x00) // Timescale: 0=1000ms
    outFileHandle.write(metadata)

    let frameStream = AsyncStream<YCbCrImage> { continuation in
        Task {
            do {
                while let image = try y4mReader.readFrame() {
                    continuation.yield(image)
                }
                continuation.finish()
            } catch {
                fputs("Failed to read frame: \(error)\n", stderr)
                continuation.finish()
            }
        }
    }

    let chunkStream = encoder.encode(stream: frameStream)
    for try await chunk in chunkStream {
        outFileHandle.write(Data(chunk))
        frameCount += 1
    }
    
    // Update DataSize in file header (seek back to offset 4)
    if outPath != "-" {
        let fileSize = outFileHandle.offsetInFile
        let dataSize = UInt32(fileSize - 8) // everything after the 8-byte file header
        outFileHandle.seek(toFileOffset: 4)
        var dataSizeBytes = Data()
        dataSizeBytes.append(UInt8((dataSize >> 24) & 0xFF))
        dataSizeBytes.append(UInt8((dataSize >> 16) & 0xFF))
        dataSizeBytes.append(UInt8((dataSize >> 8) & 0xFF))
        dataSizeBytes.append(UInt8(dataSize & 0xFF))
        outFileHandle.write(dataSizeBytes)
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
