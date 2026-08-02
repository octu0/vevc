import Foundation
import vevc

var inputPath = ""
var outPath = ""
var bitrate = 500
var zeroThreshold = 3
var keyint = 30
var sceneThreshold = 10
var qstep: Int? = nil
var profile: UInt8 = 0x01
var skipThreshold: Int = 2
var reconThresholdScale: Int = 1
var inFpsOpt: Int? = nil
var outFpsOpt: Int? = nil

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
    case "-qstep", "--qstep":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { qstep = v }
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
    case "-profile":
        if (i + 1) < args.count {
            if let v = UInt8(args[i + 1]) { profile = v }
            i += 1
        }
    case "-skip-thresh":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { skipThreshold = v }
            i += 1
        }
    case "-reconThresholdScale", "--recon-threshold-scale":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { reconThresholdScale = v }
            i += 1
        }
    case "-framerate":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { outFpsOpt = v }
            i += 1
        }
    case "-in-fps":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { inFpsOpt = v }
            i += 1
        }
    default:
        ()
    }
    i += 1
}

if inputPath.isEmpty || outPath.isEmpty {
    fputs("Usage: vevc-enc -i </path/to/input.y4m | -> -o </path/to/output.vevc | -> [-b <kilobit>] [-qstep <val>] [-framerate <out_fps>] [-in-fps <in_fps>] [-keyint <keyint>] [-zeroThreshold <threshold>] [-sceneThreshold <sad>] [-profile <profile>] [-reconThresholdScale <scale>]\n", stderr)
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
        if parts.count == 2, let num = Int(parts[0]), let den = Int(parts[1]), 0 < den {
            fps = num / den
            if fps == 0 { fps = 30 }
        }
    }
    
    let sourceFps = inFpsOpt ?? fps
    let targetFps = outFpsOpt ?? sourceFps
    
    var converter: vevc.FrameRateConverter? = nil
    if sourceFps != targetFps {
        converter = vevc.FrameRateConverter(inFps: sourceFps, outFps: targetFps)
    }
    
    let encoder: vevc.VEVCEncoder
    if let qstep = qstep {
        encoder = vevc.VEVCEncoder(
            width: y4mReader.width,
            height: y4mReader.height,
            qstep: qstep,
            framerate: targetFps,
            zeroThreshold: zeroThreshold,
            keyint: keyint,
            sceneChangeThreshold: sceneThreshold,
            profile: profile,
            skipThreshold: skipThreshold,
            reconThresholdScale: reconThresholdScale
        )
    } else {
        encoder = vevc.VEVCEncoder(
            width: y4mReader.width,
            height: y4mReader.height,
            maxbitrate: bitrate * 1000,
            framerate: targetFps,
            zeroThreshold: zeroThreshold,
            keyint: keyint,
            sceneChangeThreshold: sceneThreshold,
            profile: profile,
            skipThreshold: skipThreshold,
            reconThresholdScale: reconThresholdScale
        )
    }

    var frameCount = 0
    var totalEncodeTime: TimeInterval = 0

    while let image = try y4mReader.readFrame() {
        var converterCount: Int
        if converter != nil {
            converterCount = converter!.repeatCount()
        } else {
            converterCount = 1
        }
        
        for _ in 0..<converterCount {
            let encStart = Date()
            let chunk = try await encoder.encode(image: image)
            totalEncodeTime += Date().timeIntervalSince(encStart)
            
            outFileHandle.write(Data(chunk))
            frameCount += 1
        }
    }

    if outPath != "-" {
        let msPerFrame = if 0 < frameCount { (totalEncodeTime * 1000 / Double(frameCount)) } else { 0.0 }
        let logMsg = String(format: "Encoded %d frames in %.4fms (%.4fms/frame)\n", frameCount, totalEncodeTime * 1000, msPerFrame)
        fputs(logMsg, stderr)
    }

    inFileHandle.closeFile()
    outFileHandle.closeFile()
} catch {
    fputs("Failed to encode: \(error)\n", stderr)
    exit(1)
}
