import Foundation
import vevc

var inputPath = ""
var outPath = ""
var bitrate = 500
var zeroThreshold = 4
var keyint = 30
/// Whether the user named -keyint / -iq-floor on the command line. The
/// profile 0x02 defaults below only fill in the ones that were left out.
var keyintExplicit = false
var iqFloorExplicit = false
var sceneThreshold = 500
var qstep: Int? = nil
var profile: UInt8 = 0x01
var skipThreshold: Int = 2
var reconThresholdScale: Int = 1
var gop: Int = 12
var l2Cadence: Int = 0
var l1Cadence: Int = 0
var l0Cadence: Int = 1
var inFpsOpt: Int? = nil
var outFpsOpt: Int? = nil
var motionMaskingPx: Int = 2
var smooth: Int = 1
var temporalLayers: Int = 1
var skipModel: Int = 1
var skipRefresh: Int = 0
var skipRefreshPhase: Int = 0
var iqFloor: Int = 0

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
    case "-b", "--bitrate":
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
            if let v = Int(args[i + 1]) {
                keyint = v
                keyintExplicit = true
            }
            i += 1
        }
    case "-zero-threshold", "-zeroThreshold":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { zeroThreshold = v }
            i += 1
        }
    case "-scene-threshold", "-sceneThreshold":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { sceneThreshold = v }
            i += 1
        }
    case "-profile":
        if (i + 1) < args.count {
            let str = args[i + 1]
            if str.hasPrefix("0x") {
                if let v = UInt8(str.dropFirst(2), radix: 16) { profile = v }
            } else {
                if let v = UInt8(str) { profile = v }
            }
            i += 1
        }
    case "-gop":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { gop = v }
            i += 1
        }
    case "-skip-threshold", "-skip-thresh":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { skipThreshold = v }
            i += 1
        }
    case "-recon-threshold-scale", "-reconThresholdScale", "--recon-threshold-scale":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { reconThresholdScale = v }
            i += 1
        }
    case "-l2-cadence", "-l2cadence", "--l2cadence":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { l2Cadence = v }
            i += 1
        }
    case "-l1-cadence", "-l1cadence", "--l1cadence":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { l1Cadence = v }
            i += 1
        }
    case "-l0-cadence", "-l0cadence", "--l0cadence":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { l0Cadence = v }
            i += 1
        }
    case "-mvt", "--mvt":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { motionMaskingPx = v }
            i += 1
        }
    case "-smooth", "--smooth":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { smooth = v }
            i += 1
        }
    case "-temporal-layers", "--temporal-layers":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { temporalLayers = v }
            i += 1
        }
    case "-skip-model", "--skip-model":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { skipModel = v }
            i += 1
        }
    case "-skip-refresh", "--skip-refresh":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { skipRefresh = v }
            i += 1
        }
    case "-skip-refresh-phase", "--skip-refresh-phase":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { skipRefreshPhase = v }
            i += 1
        }
    case "-iq-floor", "--iq-floor":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) {
                iqFloor = v
                iqFloorExplicit = true
            }
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

// Profile 0x02 defaults. The quality floor places the I frames and `keyint`
// only caps the GOP, so the pair ships together: keyint 120 as the upper bound
// and alpha 2.5 as the floor. Resolved after the whole argument list is read,
// because -profile can appear anywhere in it. Profile 0x01 is untouched — it
// has no floor and keeps its fixed keyint 30 period.
if profile == 0x02 {
    if keyintExplicit != true { keyint = 120 }
    if iqFloorExplicit != true { iqFloor = 250 }
}

if inputPath.isEmpty || outPath.isEmpty {
    fputs("Usage: vevc-enc -i </path/to/input.y4m | -> -o </path/to/output.vevc | -> [-b <kilobit> | --bitrate <kilobit>] [-qstep <val>] [-framerate <out_fps>] [-in-fps <in_fps>] [-keyint <keyint>] [-zero-threshold <threshold>] [-scene-threshold <sad>] [-profile <profile>] [-gop <gop>] [-l2-cadence <n>] [-l1-cadence <n>] [-l0-cadence <n>] [-skip-threshold <threshold>] [-recon-threshold-scale <scale>] [-mvt <px>] [-smooth <0|1>] [-temporal-layers <1|2>] [-skip-model <0|1>] [-skip-refresh <frames>] [-iq-floor <alphax100>]\n  -iq-floor <alphax100>: Quality floor for early I frames; codes an I once a P frame's luma MSE exceeds alpha x the GOP's I-frame MSE, making -keyint an upper bound (default: 250 on profile 2, 0 = off on profile 1)\n  -keyint <keyint>: Maximum GOP size (default: 120 on profile 2, 30 on profile 1)\n  -skip-refresh <frames>: Periodic skip refresh; a block skipped this many frames in a row is coded as inter again (default: 0 = off, profile 2 only)\n  -mvt <px>: Motion masking threshold in px/frame; drops full-resolution detail on high-motion blocks (motion masking); active only during saturation (default: 2, 0 disables)\n  -smooth <0|1>: P-frame residual plane smoothing (default: 1, 0 disables)\n  -temporal-layers <1|2>: Number of temporal layers (default: 1, 2 for T0/T1)\n  -skip-model <0|1>: Learned skip-safety decider on profile 0x02 P-frames (default: 1, 0 disables; no effect on profile 0x01)\n", stderr)
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
            reconThresholdScale: reconThresholdScale,
            gop: gop,
            l2Cadence: l2Cadence,
            l1Cadence: l1Cadence,
            l0Cadence: l0Cadence,
            motionMaskingPx: motionMaskingPx,
            smooth: smooth,
            temporalLayers: temporalLayers,
            skipModel: skipModel,
            skipRefresh: skipRefresh,
            skipRefreshPhase: skipRefreshPhase,
            iqFloor: iqFloor
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
            reconThresholdScale: reconThresholdScale,
            gop: gop,
            l2Cadence: l2Cadence,
            l1Cadence: l1Cadence,
            l0Cadence: l0Cadence,
            motionMaskingPx: motionMaskingPx,
            smooth: smooth,
            temporalLayers: temporalLayers,
            skipModel: skipModel,
            skipRefresh: skipRefresh,
            skipRefreshPhase: skipRefreshPhase,
            iqFloor: iqFloor
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

    if 0 < iqFloor || 0 < keyint {
        let census = await encoder.frameCensus()
        fputs("FrameCensus: I forced/periodic=\(census.iForced) scene=\(census.iScene) floor=\(census.iFloor) copyFrame=\(census.copy)\n", stderr)
        if 0 < iqFloor {
            fputs("IQFloor alpha=\(iqFloor)/100 firings=\(census.firings.count)\n", stderr)
            for f in census.firings {
                fputs("IQFloorFire frame=\(f.frame) k=\(f.k) dist=\(f.dist) frameMSE=\(f.frameMSE) iMSE=\(f.iMSE) ratioQ8=\(0 < f.iMSE ? (f.frameMSE * 256) / f.iMSE : -1)\n", stderr)
            }
        }
    }

    if 0 < skipRefresh {
        let stats = await encoder.skipRefreshStats()
        let permyriad = 0 < stats.examined ? (stats.forced * 10000) / stats.examined : 0
        let ratioStr = "\(permyriad / 100).\(permyriad % 100 / 10)\(permyriad % 10)"
        fputs("SkipRefresh r=\(skipRefresh): forced-inter blocks \(stats.forced) / P-frame blocks \(stats.examined) (\(ratioStr)%)\n", stderr)
    }

    inFileHandle.closeFile()
    outFileHandle.closeFile()
} catch {
    fputs("Failed to encode: \(error)\n", stderr)
    exit(1)
}
