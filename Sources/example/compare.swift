import Foundation
import VideoToolbox
import CoreMedia
import AppKit
import CryptoKit
import vevc

/// Measurement only: decode an existing bitstream and score it against the
/// source, reporting the same statistics the benchmark path reports plus the
/// optional per-frame CSV. Nothing is encoded here, so the numbers describe
/// exactly the file that was handed in.
func scoreExistingStream(streamPath: String, y4mPath: String, config: Config) async throws {
    guard let streamHandle = FileHandle(forReadingAtPath: streamPath) else {
        print("cannot open stream \(streamPath)")
        exit(1)
    }
    defer { streamHandle.closeFile() }
    let attrs = try? FileManager.default.attributesOfItem(atPath: streamPath)
    let byteCount = (attrs?[.size] as? Int) ?? 0
    let iter = try Y4MIterator(path: y4mPath, config: config)

    // Same entry point vevc-dec uses, so the frame splitting matches the
    // production decode path exactly.
    let decoder = Decoder(maxLayer: config.maxLayer)

    var mets = [QualityMetrics]()
    var perFrameCSV = "frame,psnr,ssim,ssimY\n"
    var frameIdx = 0
    let decodeStart = Date()
    for try await decodedImg in decoder.decodeFile(fileHandle: streamHandle) {
        guard let orig = try iter.next() else { break }
        let psnr = calculatePSNR(img1: orig.vevcImage, img2: decodedImg)
        let ssim = calculateSSIM(img1: orig.vevcImage, img2: decodedImg)
        let ssimY = calculateSSIMLuma(img1: orig.vevcImage, img2: decodedImg)
        mets.append(QualityMetrics(psnr: psnr, ssim: ssim))
        perFrameCSV += "\(frameIdx),\(psnr),\(ssim),\(ssimY)\n"
        frameIdx += 1
    }
    let decodeTime = Date().timeIntervalSince(decodeStart)

    print("[SCORE] stream=\(streamPath)")
    print("  Bytes  : \(byteCount)")
    print("  Frames : \(frameIdx)")
    print(String(format: "  Decode : %7.2f ms (%.4f ms/frame)", decodeTime * 1000, 0 < frameIdx ? decodeTime * 1000 / Double(frameIdx) : 0))
    if let stats = calculateQualityStats(metrics: mets) {
        print(String(format: "  PSNR   : Avg: %5.2f | Min: %5.2f | Max: %5.2f", stats.avgPSNR, stats.minPSNR, stats.maxPSNR))
        print(String(format: "  SSIM   : Avg: %7.5f | Min: %7.5f | Max: %7.5f", stats.avgSSIM, stats.minSSIM, stats.maxSSIM))
    }
    if let dumpPath = config.dumpFrameMetrics {
        try? perFrameCSV.write(toFile: dumpPath, atomically: true, encoding: .utf8)
        print("  per-frame metrics written to \(dumpPath)")
    }
}

struct Config {
    var bitrate: Int = 500
    var framerate: Int = 60
    var inFps: Int? = nil
    var zeroThreshold: Int = 4
    var keyint: Int = 30
    var sceneThreshold: Int = 500
    var maxLayer: Int = 2
    var quality: Bool = false
    var outputGraph: Bool = false
    var outputVersus: Bool = false
    var outputBitrates: Bool = false
    var vevcOnly: Bool = false
    var dumpHash: Bool = false
    var qstep: Int? = nil
    var skipThreshold: Int = 2
    var profile: UInt8 = 0x01
    var gop: Int = 12
    var l2Cadence: Int = 0
    var l1Cadence: Int = 0
    var l0Cadence: Int = 1
    var motionMaskingPx: Int = 2
    var smooth: Int = 1
    var skipModel: Int = 1
    var iqFloor: Int = 0
    /// Whether the user named -keyint / -iq-floor. runVEVC fills in the
    /// profile 0x02 defaults for the ones that were left out; it resolves them
    /// rather than the parser because -output-graph clones this config and
    /// flips `profile` afterwards.
    var keyintExplicit: Bool = false
    var iqFloorExplicit: Bool = false
    /// Measurement only: path for a per-frame "frame,psnr,ssim,ssimY" CSV of
    /// the VEVC layer-2 decode. nil disables the dump.
    var dumpFrameMetrics: String? = nil
    /// Measurement only: score this existing .vevc bitstream against -y4m
    /// instead of encoding one. nil runs the normal benchmark.
    var scoreStream: String? = nil
    var temporalLayers: Int = 1
    var maxFrames: Int? = nil
}

struct ImageInput {
    let vevcImage: YCbCrImage
    let width: Int
    let height: Int
}

func createPixelBuffer(from img: YCbCrImage) -> CVPixelBuffer? {
    let width = img.width
    let height = img.height
    
    let attrs = [
        kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
        kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue,
        kCVPixelBufferMetalCompatibilityKey: kCFBooleanTrue,
    ] as CFDictionary
    
    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, attrs, &pixelBuffer)
    guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }
    
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    
    if let yDest = CVPixelBufferGetBaseAddressOfPlane(buffer, 0) {
        let destStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
        img.yPlane.withUnsafeBufferPointer { ySrc in
            guard let srcBase = ySrc.baseAddress else { return }
            for y in 0..<height {
                let destRow = yDest.advanced(by: y * destStride)
                let srcRow = srcBase.advanced(by: y * width)
                memcpy(destRow, srcRow, width)
            }
        }
    }
    
    if let uvDest = CVPixelBufferGetBaseAddressOfPlane(buffer, 1) {
        let destStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
        let cWidth = (width + 1) / 2
        let cHeight = (height + 1) / 2
        
        img.cbPlane.withUnsafeBufferPointer { cbSrc in
            img.crPlane.withUnsafeBufferPointer { crSrc in
                guard let cbBase = cbSrc.baseAddress, let crBase = crSrc.baseAddress else { return }
                for y in 0..<cHeight {
                    let destRow = uvDest.advanced(by: y * destStride).assumingMemoryBound(to: UInt8.self)
                    let cbRow = cbBase.advanced(by: y * cWidth)
                    let crRow = crBase.advanced(by: y * cWidth)
                    for x in 0..<cWidth {
                        destRow[x * 2 + 0] = cbRow[x]
                        destRow[x * 2 + 1] = crRow[x]
                    }
                }
            }
        }
    }
    return buffer
}

func saveVersusImage(idx: Int, orig: ImageInput, vevcF: YCbCrImage?, h264F: YCbCrImage?, hevcF: YCbCrImage?, prefix: String) {
    let w = orig.width
    let h = orig.height
    
    let cropW = 400
    let cropH = 400
    let cx = max(0, w / 2 - cropW / 2)
    let cy = max(0, h / 2 - cropH / 2)
    let cropRect = CGRect(x: cx, y: cy, width: cropW, height: cropH)
    
    func createCroppedPNG(from ycbcr: YCbCrImage?, outPath: String) {
        guard let f = ycbcr else { return }
        var rgba = vevc.ycbcrToRGBA(img: f)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * f.width
        
        for i in stride(from: 3, to: rgba.count, by: 4) { rgba[i] = 255 }
        
        rgba.withUnsafeMutableBytes { ptr in
            guard let baseAddress = ptr.baseAddress else { return }
            guard let context = CGContext(data: baseAddress, width: f.width, height: f.height, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else { return }
            guard let cgImage = context.makeImage() else { return }
            guard let croppedCGImage = cgImage.cropping(to: cropRect) else { return }
            let rep = NSBitmapImageRep(cgImage: croppedCGImage)
            guard let pngData = rep.representation(using: .png, properties: [:]) else { return }
            let url = URL(fileURLWithPath: outPath)
            do { try pngData.write(to: url) } catch {}
        }
    }
    
    createCroppedPNG(from: orig.vevcImage, outPath: "docs/versus_\(prefix)_frame\(idx)_orig.png")
    createCroppedPNG(from: vevcF, outPath: "docs/versus_\(prefix)_frame\(idx)_vevc.png")
    createCroppedPNG(from: h264F, outPath: "docs/versus_\(prefix)_frame\(idx)_h264.png")
    createCroppedPNG(from: hevcF, outPath: "docs/versus_\(prefix)_frame\(idx)_hevc.png")
}

@main
struct CompareApp {
    static func main() async throws {
        try await Task(priority: .userInitiated) {
            try await _main()
        }.value
    }

    static func _main() async throws {
        let args = CommandLine.arguments

        var config = Config()
        var positionalArgs: [String] = []
        var y4mPath: String? = nil

        var i = 1
        while i < args.count {
            let arg = args[i]
            switch arg {
            case "-b", "--bitrate", "-bitrate":
                if (i + 1) < args.count {
                    if let v = Int(args[i + 1]) { config.bitrate = v }
                    i += 1
                }
            case "-framerate":
                if (i + 1) < args.count {
                    if let v = Int(args[i + 1]) { config.framerate = v }
                    i += 1
                }
            case "-in-fps":
                if (i + 1) < args.count {
                    if let v = Int(args[i + 1]) { config.inFps = v }
                    i += 1
                }
            case "-zero-threshold", "-zeroThreshold":
                if (i + 1) < args.count {
                    if let v = Int(args[i + 1]) { config.zeroThreshold = v }
                    i += 1
                }
            case "-keyint":
                if (i + 1) < args.count {
                    if let v = Int(args[i + 1]) {
                        config.keyint = v
                        config.keyintExplicit = true
                    }
                    i += 1
                }
            case "-scene-threshold", "-sceneThreshold":
                if (i + 1) < args.count {
                    if let v = Int(args[i + 1]) { config.sceneThreshold = v }
                    i += 1
                }
            case "-max-layer", "-maxLayer":
                if (i + 1) < args.count {
                    if let v = Int(args[i + 1]) { config.maxLayer = v }
                    i += 1
                }
            case "-quality":
                config.quality = true
            case "-output-graph", "--output-graph":
                config.outputGraph = true
            case "-output-versus", "--output-versus":
                config.outputVersus = true
            case "-output-bitrates", "--output-bitrates":
                config.outputBitrates = true
                config.quality = true
            case "-vevc-only", "--vevc-only":
                config.vevcOnly = true
            case "-dump-hash", "--dump-hash":
                config.dumpHash = true
            case "-qstep", "--qstep":
                if (i + 1) < args.count {
                    if let v = Int(args[i + 1]) { config.qstep = v }
                    i += 1
                }
            case "-skip-threshold", "-skip-thresh":
                if (i + 1) < args.count {
                    if let v = Int(args[i + 1]) { config.skipThreshold = v }
                    i += 1
                }
            case "-profile":
                if (i + 1) < args.count {
                    let str = args[i + 1]
                    if str.hasPrefix("0x") {
                        if let v = UInt8(str.dropFirst(2), radix: 16) { config.profile = v }
                    } else {
                        if let v = UInt8(str) { config.profile = v }
                    }
                    i += 1
                }
            case "-gop":
                if (i + 1) < args.count {
                    if let v = Int(args[i + 1]) { config.gop = v }
                    i += 1
                }
            case "-l2-cadence", "-l2cadence", "--l2cadence":
                if (i + 1) < args.count {
                    if let v = Int(args[i + 1]) { config.l2Cadence = v }
                    i += 1
                }
            case "-l1-cadence", "-l1cadence", "--l1cadence":
                if (i + 1) < args.count {
                    if let v = Int(args[i + 1]) { config.l1Cadence = v }
                    i += 1
                }
            case "-l0-cadence", "-l0cadence", "--l0cadence":
                if (i + 1) < args.count {
                    if let v = Int(args[i + 1]) { config.l0Cadence = v }
                    i += 1
                }
            case "-mvt", "--mvt":
                if (i + 1) < args.count {
                    if let v = Int(args[i + 1]) { config.motionMaskingPx = v }
                    i += 1
                }
            case "-smooth", "--smooth":
                if (i + 1) < args.count {
                    if let v = Int(args[i + 1]) { config.smooth = v }
                    i += 1
                }
            case "-skip-model", "--skip-model":
                if (i + 1) < args.count {
                    if let v = Int(args[i + 1]) { config.skipModel = v }
                    i += 1
                }
            case "-iq-floor", "--iq-floor":
                if (i + 1) < args.count {
                    if let v = Int(args[i + 1]) {
                        config.iqFloor = v
                        config.iqFloorExplicit = true
                    }
                    i += 1
                }
            case "-dump-frame-metrics", "--dump-frame-metrics":
                if (i + 1) < args.count {
                    config.dumpFrameMetrics = args[i + 1]
                    config.quality = true
                    i += 1
                }
            case "-score", "--score":
                if (i + 1) < args.count {
                    config.scoreStream = args[i + 1]
                    config.quality = true
                    i += 1
                }
            case "-temporal-layers", "--temporal-layers":
                if (i + 1) < args.count {
                    if let v = Int(args[i + 1]) { config.temporalLayers = v }
                    i += 1
                }
            case "-frames", "--frames", "-max-frames", "-maxFrames":
                if (i + 1) < args.count {
                    if let v = Int(args[i + 1]) { config.maxFrames = v }
                    i += 1
                }
            case "-y4m":
                if (i + 1) < args.count {
                    y4mPath = args[i + 1]
                    i += 1
                }
            default:
                positionalArgs.append(arg)
            }
            i += 1
        }

        if positionalArgs.isEmpty && y4mPath == nil {
            print("Usage: compare [-y4m <input.y4m>] [-b <kbits> | --bitrate <kbits>] [-qstep <val>] [-framerate <fps>] [-in-fps <in_fps>] [-zero-threshold <threshold>] [-keyint <frames>] [-scene-threshold <sad>] [-max-layer <0-2>] [-profile <0x01|0x02>] [-gop <frames>] [-l2-cadence <n>] [-l1-cadence <n>] [-l0-cadence <n>] [-skip-threshold <threshold>] [-mvt <px>] [-smooth <0|1>] [-skip-model <0|1>] [-temporal-layers <1|2>] [-quality] [-output-graph] [-output-versus] [-output-bitrates] [-vevc-only] [-dump-hash] [-frames <n>]")
            exit(1)
        }

        // Measurement mode: score an existing bitstream against the source,
        // instead of encoding one here. Reporting size and quality from the
        // same bytes removes any chance of the two disagreeing about the
        // encoder configuration.
        if let streamPath = config.scoreStream {
            try await scoreExistingStream(streamPath: streamPath, y4mPath: y4mPath!, config: config)
            return
        }

        var finalY4mPath = y4mPath!
        var tempPath: String? = nil
        if finalY4mPath == "-" && (config.quality || config.outputVersus || config.outputBitrates) {
            let tmp = NSTemporaryDirectory() + UUID().uuidString + ".y4m"
            let data = try FileHandle.standardInput.readToEnd() ?? Data()
            try data.write(to: URL(fileURLWithPath: tmp))
            finalY4mPath = tmp
            tempPath = tmp
        }
        defer {
            if let t = tempPath { try? FileManager.default.removeItem(atPath: t) }
        }

        var width = 0
        var height = 0
        var frameCount = 0
        if let iter = try? Y4MIterator(path: finalY4mPath, config: config) {
            if let first = try? iter.next() {
                width = first.width
                height = first.height
                frameCount = 1
                while let _ = try? iter.next() { frameCount += 1 }
            }
        }

        if width == 0 {
            print("No valid input images found.")
            exit(1)
        }

        print("--- Settings ---")
        print("Input frames   : \(frameCount)")
        print("Resolution     : \(width)x\(height)")
        print("Target Bitrate : \(config.bitrate) kbps")
        print("Target FPS     : \(config.framerate)")
        print("Quality Check  : \(config.quality)")
        print("Profile        : \(String(format: "0x%02X", config.profile))")
        print("----------------")
        
        let localConfig = config
        let localWidth = width
        let localHeight = height
        let localY4mPath = finalY4mPath
        
        let rawTotalSizeKB = Double(frameCount * localWidth * localHeight * 3) / 1024.0
        
        do {
            print("Running vevc (layers)...")
            let vevcResult = try await runVEVC(y4mPath: localY4mPath, config: localConfig)
            
            typealias CodecResult = (encTime: Double, decTime: Double, compSize: Int, metrics: [QualityMetrics]?, bitstream: [VTFrameData])
            var h264Result: CodecResult? = nil
            var h264SwResult: CodecResult? = nil
            var hevcResult: CodecResult? = nil
            var hevcSwResult: CodecResult? = nil
            
            if localConfig.vevcOnly != true {
                print("Running H.264 (VideoToolbox HWA)...")
                h264Result = try await runH264(y4mPath: localY4mPath, config: localConfig, width: localWidth, height: localHeight)
                
                print("Running H.264 (VideoToolbox SW)...")
                h264SwResult = try await runH264(y4mPath: localY4mPath, config: localConfig, width: localWidth, height: localHeight, disableHWA: true)
                
                print("Running HEVC (VideoToolbox HWA)...")
                hevcResult = try await runHEVC(y4mPath: localY4mPath, config: localConfig, width: localWidth, height: localHeight)
                
                print("Running HEVC (VideoToolbox SW)...")
                hevcSwResult = try await runHEVC(y4mPath: localY4mPath, config: localConfig, width: localWidth, height: localHeight, disableHWA: true)
            }
            
            func printStats(name: String, encTime: Double, decTime: Double, compSize: Int, metrics: [QualityMetrics]?, count: Int, rawSizeKB: Double, skips: (prev: Int, ltr: Int, inter: Int, copy: Int, total: Int)? = nil) -> CodecBenchmarkResult {
                let encMs = encTime * 1000
                let decMs = decTime * 1000
                let encFps = count > 0 ? Double(count) / encTime : 0
                let decFps = count > 0 ? Double(count) / decTime : 0
                let sizeKB = Double(compSize) / 1024.0
                
                print("[\(name)]")
                print(String(format: "  Encode : %7.2f ms (%.2f fps) - %.2f ms / frame", encMs, encFps, count > 0 ? encMs / Double(count) : 0))
                print(String(format: "  Decode : %7.2f ms (%.2f fps) - %.2f ms / frame", decMs, decFps, count > 0 ? decMs / Double(count) : 0))
                print(String(format: "  Size   : %7.2f KB (%.2f%% of raw %.2f KB)", sizeKB, rawSizeKB > 0 ? (sizeKB / rawSizeKB) * 100.0 : 0, rawSizeKB))
                if let s = skips, s.total > 0 {
                    let nonCopyTotal = s.total - s.copy
                    let copyRatio = Double(s.copy) / Double(s.total) * 100.0
                    if nonCopyTotal > 0 {
                        let prevRatio = Double(s.prev) / Double(nonCopyTotal) * 100.0
                        let ltrRatio = Double(s.ltr) / Double(nonCopyTotal) * 100.0
                        let interRatio = Double(s.inter) / Double(nonCopyTotal) * 100.0
                        print(String(format: "  Skips  : CopyFrame: %5.2f%% | Non-Copy Skips: Prev: %5.2f%% | LTR: %5.2f%% | Inter: %5.2f%%", copyRatio, prevRatio, ltrRatio, interRatio))
                    } else {
                        print(String(format: "  Skips  : CopyFrame: %5.2f%% | Non-Copy Skips: N/A", copyRatio))
                    }
                }
                
                var statsOut: QualityStats? = nil
                if let stats = calculateQualityStats(metrics: metrics ?? []) {
                    statsOut = stats
                    print(String(format: "  PSNR   : Avg: %5.2f | Min: %5.2f | Max: %5.2f | 50%%: %5.2f | 90%%: %5.2f | SD: %5.2f", 
                                stats.avgPSNR, stats.minPSNR, stats.maxPSNR, stats.p50PSNR, stats.p90PSNR, stats.stddevPSNR))
                    print(String(format: "  SSIM   : Avg: %5.4f | Min: %5.4f | Max: %5.4f | 50%%: %5.4f | 90%%: %5.4f | SD: %5.4f",
                                stats.avgSSIM, stats.minSSIM, stats.maxSSIM, stats.p50SSIM, stats.p90SSIM, stats.stddevSSIM))
                    // Worst-frame indices (decoded frame numbers): the min
                    // values gate quality work, so name the frames they come
                    // from.
                    if let m = metrics, m.isEmpty != true {
                        let minSsimIdx = m.indices.min(by: { m[$0].ssim < m[$1].ssim }) ?? 0
                        let minPsnrIdx = m.indices.min(by: { m[$0].psnr < m[$1].psnr }) ?? 0
                        print("  Worst  : SSIM min at decoded frame \(minSsimIdx) | PSNR min at decoded frame \(minPsnrIdx)")
                    }
                }
                
                return CodecBenchmarkResult(name: name, encTimeMs: count > 0 ? encMs / Double(count) : 0, decTimeMs: count > 0 ? decMs / Double(count) : 0, sizeKB: sizeKB, stats: statsOut)
            }
            
            print("\n--- Results ---")
            var chartResults: [CodecBenchmarkResult] = []
            chartResults.append(printStats(name: "VEVC (Layer 2)", encTime: vevcResult.encTime, decTime: vevcResult.l2Dec.time, compSize: vevcResult.sizes.l2, metrics: vevcResult.l2Dec.metrics, count: frameCount, rawSizeKB: rawTotalSizeKB, skips: vevcResult.skips))
            chartResults.append(printStats(name: "VEVC (Layer 1)", encTime: vevcResult.encTime, decTime: vevcResult.l1Dec.time, compSize: vevcResult.sizes.l1, metrics: vevcResult.l1Dec.metrics, count: frameCount, rawSizeKB: rawTotalSizeKB, skips: vevcResult.skips))
            chartResults.append(printStats(name: "VEVC (Layer 0)", encTime: vevcResult.encTime, decTime: vevcResult.l0Dec.time, compSize: vevcResult.sizes.l0, metrics: vevcResult.l0Dec.metrics, count: frameCount, rawSizeKB: rawTotalSizeKB, skips: vevcResult.skips))
            
            if localConfig.vevcOnly != true {
                if let h264Sw = h264SwResult {
                    chartResults.append(printStats(name: "H.264 (SW)", encTime: h264Sw.encTime, decTime: h264Sw.decTime, compSize: h264Sw.compSize, metrics: h264Sw.metrics, count: frameCount, rawSizeKB: rawTotalSizeKB))
                }
                if let hevcSw = hevcSwResult {
                    chartResults.append(printStats(name: "HEVC (SW)", encTime: hevcSw.encTime, decTime: hevcSw.decTime, compSize: hevcSw.compSize, metrics: hevcSw.metrics, count: frameCount, rawSizeKB: rawTotalSizeKB))
                }
                if let h264 = h264Result {
                    chartResults.append(printStats(name: "H.264 (HWA)", encTime: h264.encTime, decTime: h264.decTime, compSize: h264.compSize, metrics: h264.metrics, count: frameCount, rawSizeKB: rawTotalSizeKB))
                }
                if let hevc = hevcResult {
                    chartResults.append(printStats(name: "HEVC (HWA)", encTime: hevc.encTime, decTime: hevc.decTime, compSize: hevc.compSize, metrics: hevc.metrics, count: frameCount, rawSizeKB: rawTotalSizeKB))
                }
            }
            print("---------------")
            
            if localConfig.outputGraph {
                print("\n--- Generating Speed & Size Graphs ---")
                var p1Config = localConfig
                p1Config.profile = 1
                var p2Config = localConfig
                p2Config.profile = 2
                
                print(">> Running VEVC Profile 1...")
                let p1Res = try await runVEVC(y4mPath: localY4mPath, config: p1Config)
                print(">> Running VEVC Profile 2...")
                let p2Res = try await runVEVC(y4mPath: localY4mPath, config: p2Config)
                
                var speedSizeResults: [CodecBenchmarkResult] = []
                speedSizeResults.append(CodecBenchmarkResult(name: "VEVC (p1)", encTimeMs: p1Res.encTime * 1000 / Double(frameCount), decTimeMs: p1Res.l2Dec.time * 1000 / Double(frameCount), sizeKB: Double(p1Res.sizes.l2) / 1024.0, stats: calculateQualityStats(metrics: p1Res.l2Dec.metrics ?? [])))
                speedSizeResults.append(CodecBenchmarkResult(name: "VEVC (p2)", encTimeMs: p2Res.encTime * 1000 / Double(frameCount), decTimeMs: p2Res.l2Dec.time * 1000 / Double(frameCount), sizeKB: Double(p2Res.sizes.l2) / 1024.0, stats: calculateQualityStats(metrics: p2Res.l2Dec.metrics ?? [])))
                
                if localConfig.vevcOnly != true {
                    if let h264 = h264SwResult {
                        speedSizeResults.append(CodecBenchmarkResult(name: "H.264 (SW)", encTimeMs: h264.encTime * 1000 / Double(frameCount), decTimeMs: h264.decTime * 1000 / Double(frameCount), sizeKB: Double(h264.compSize) / 1024.0, stats: calculateQualityStats(metrics: h264.metrics ?? [])))
                    }
                    if let hevc = hevcSwResult {
                        speedSizeResults.append(CodecBenchmarkResult(name: "HEVC (SW)", encTimeMs: hevc.encTime * 1000 / Double(frameCount), decTimeMs: hevc.decTime * 1000 / Double(frameCount), sizeKB: Double(hevc.compSize) / 1024.0, stats: calculateQualityStats(metrics: hevc.metrics ?? [])))
                    }
                    if let h264Hwa = h264Result {
                        speedSizeResults.append(CodecBenchmarkResult(name: "H.264 (HWA)", encTimeMs: h264Hwa.encTime * 1000 / Double(frameCount), decTimeMs: h264Hwa.decTime * 1000 / Double(frameCount), sizeKB: Double(h264Hwa.compSize) / 1024.0, stats: calculateQualityStats(metrics: h264Hwa.metrics ?? [])))
                    }
                    if let hevcHwa = hevcResult {
                        speedSizeResults.append(CodecBenchmarkResult(name: "HEVC (HWA)", encTimeMs: hevcHwa.encTime * 1000 / Double(frameCount), decTimeMs: hevcHwa.decTime * 1000 / Double(frameCount), sizeKB: Double(hevcHwa.compSize) / 1024.0, stats: calculateQualityStats(metrics: hevcHwa.metrics ?? [])))
                    }
                }
                
                var p1Layers: [CodecBenchmarkResult] = []
                p1Layers.append(CodecBenchmarkResult(name: "Layer 2", encTimeMs: 0, decTimeMs: p1Res.l2Dec.time * 1000 / Double(frameCount), sizeKB: Double(p1Res.sizes.l2) / 1024.0, stats: nil))
                p1Layers.append(CodecBenchmarkResult(name: "Layer 1", encTimeMs: 0, decTimeMs: p1Res.l1Dec.time * 1000 / Double(frameCount), sizeKB: Double(p1Res.sizes.l1) / 1024.0, stats: nil))
                p1Layers.append(CodecBenchmarkResult(name: "Layer 0", encTimeMs: 0, decTimeMs: p1Res.l0Dec.time * 1000 / Double(frameCount), sizeKB: Double(p1Res.sizes.l0) / 1024.0, stats: nil))
                
                var p2Layers: [CodecBenchmarkResult] = []
                p2Layers.append(CodecBenchmarkResult(name: "Layer 2", encTimeMs: 0, decTimeMs: p2Res.l2Dec.time * 1000 / Double(frameCount), sizeKB: Double(p2Res.sizes.l2) / 1024.0, stats: nil))
                p2Layers.append(CodecBenchmarkResult(name: "Layer 1", encTimeMs: 0, decTimeMs: p2Res.l1Dec.time * 1000 / Double(frameCount), sizeKB: Double(p2Res.sizes.l1) / 1024.0, stats: nil))
                p2Layers.append(CodecBenchmarkResult(name: "Layer 0", encTimeMs: 0, decTimeMs: p2Res.l0Dec.time * 1000 / Double(frameCount), sizeKB: Double(p2Res.sizes.l0) / 1024.0, stats: nil))

                await MainActor.run {
                    generateAndSaveSpeedSizeChart(results: speedSizeResults, outPath: "docs/speed.png")
                    generateAndSaveSpeedSizeChart(results: p1Layers, outPath: "docs/speed_p1.png", showEncode: false)
                    generateAndSaveSpeedSizeChart(results: p2Layers, outPath: "docs/speed_p2.png", showEncode: false)
                    generateAndSaveQualityCharts(results: speedSizeResults, outDir: "docs", suffix: "")
                }
            }
            
            if localConfig.outputVersus {
                print("\n--- Output Versus Images ---")
                let vevcMinIdx = vevcResult.l2Dec.metrics?.enumerated().min(by: { $0.element.ssim < $1.element.ssim })?.offset ?? 0
                let h264MinIdx = h264SwResult?.metrics?.enumerated().min(by: { $0.element.ssim < $1.element.ssim })?.offset ?? 0
                let hevcMinIdx = hevcSwResult?.metrics?.enumerated().min(by: { $0.element.ssim < $1.element.ssim })?.offset ?? 0
                let sec14Idx = min(14 * localConfig.framerate, vevcResult.skips.total - 1)
                
                let targetIndices: Set<Int> = [vevcMinIdx, h264MinIdx, hevcMinIdx, sec14Idx]
                print("Target Indices: VEVC (\(vevcMinIdx)), H264 (\(h264MinIdx)), HEVC (\(hevcMinIdx)), 14s (\(sec14Idx))")
                
                print("Extracting VEVC frames...")
                let vevcExtracted = try await extractVEVCFrames(bitstream: vevcResult.bitstream, config: localConfig, indices: targetIndices)
                
                var h264Extracted: [Int: YCbCrImage] = [:]
                if let res = h264SwResult {
                    print("Extracting H.264 frames...")
                    h264Extracted = try extractVTFrames(bitstream: res.bitstream, disableHWA: true, indices: targetIndices)
                }
                
                var hevcExtracted: [Int: YCbCrImage] = [:]
                if let res = hevcSwResult {
                    print("Extracting HEVC frames...")
                    hevcExtracted = try extractVTFrames(bitstream: res.bitstream, disableHWA: true, indices: targetIndices)
                }
                
                let versusY4M = try Y4MIterator(path: localY4mPath, config: localConfig)
                var origFrames: [Int: ImageInput] = [:]
                var idx = 0
                while let img = try versusY4M.next() {
                    if targetIndices.contains(idx) {
                        origFrames[idx] = img
                    }
                    idx += 1
                }
                
                let pairs: [(name: String, idx: Int)] = [
                    ("vevc_min", vevcMinIdx),
                    ("h264_min", h264MinIdx),
                    ("hevc_min", hevcMinIdx),
                    ("14s", sec14Idx)
                ]
                
                for p in pairs {
                    guard let origFrame = origFrames[p.idx] else { continue }
                    saveVersusImage(idx: p.idx, orig: origFrame, vevcF: vevcExtracted[p.idx], h264F: h264Extracted[p.idx], hevcF: hevcExtracted[p.idx], prefix: p.name)
                }
                print("Versus images written successfully.")
            }
        } catch {
            print("Error: \(error)")
            exit(1)
        }
        exit(0)
    }
}
