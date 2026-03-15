import Foundation
import VideoToolbox
import CoreMedia
import PNG
import vevc

// MARK: - CommandLine argument parsing
let args = CommandLine.arguments
struct Config {
    var bitrate: Int = 500
    var framerate: Int = 60
    var zeroThreshold: Int = 0
    var gopSize: Int = 15
    var sceneThreshold: Int = 8
    var maxLayer: Int = 2
}

var config = Config()
var positionalArgs: [String] = []

var i = 1
while i < args.count {
    let arg = args[i]
    switch arg {
    case "-bitrate":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { config.bitrate = v }
            i += 1
        }
    case "-framerate":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { config.framerate = v }
            i += 1
        }
    case "-zeroThreshold":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { config.zeroThreshold = v }
            i += 1
        }
    case "-gopSize":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { config.gopSize = v }
            i += 1
        }
    case "-sceneThreshold":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { config.sceneThreshold = v }
            i += 1
        }
    case "-maxLayer":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { config.maxLayer = v }
            i += 1
        }
    default:
        positionalArgs.append(arg)
    }
    i += 1
}

if positionalArgs.isEmpty {
    print("Usage: compare [-bitrate <kbits>] [-framerate <fps>] [-zeroThreshold <threshold>] [-gopSize <frames>] [-sceneThreshold <sad>] [-maxLayer <0-2>] <input1.png> [input2.png ...]")
    exit(1)
}

// MARK: - Utilities
func readPNG(path: String) -> (image: YCbCrImage, rawData: [PNG.RGBA<UInt8>], width: Int, height: Int)? {
    guard let image: PNG.Image = try? .decompress(path: path) else { return nil }
    let rgba: [PNG.RGBA<UInt8>] = image.unpack(as: PNG.RGBA<UInt8>.self)
    var data = [UInt8](repeating: 0, count: rgba.count * 4)
    for j in 0..<rgba.count {
        let offset = j * 4
        data[offset + 0] = rgba[j].r
        data[offset + 1] = rgba[j].g
        data[offset + 2] = rgba[j].b
        data[offset + 3] = rgba[j].a
    }
    return (vevc.rgbaToYCbCr(data: data, width: image.size.x, height: image.size.y), rgba, image.size.x, image.size.y)
}

struct ImageInput {
    let vevcImage: YCbCrImage
    let rgbaFrames: [PNG.RGBA<UInt8>]
    let width: Int
    let height: Int
}

var images: [ImageInput] = []
for p in positionalArgs {
    if let imgData = readPNG(path: p) {
        images.append(ImageInput(vevcImage: imgData.image, rgbaFrames: imgData.rawData, width: imgData.width, height: imgData.height))
    } else {
        print("Failed to read \(p)")
    }
}

if images.isEmpty {
    print("No valid input images found.")
    exit(1)
}

let width = images[0].width
let height = images[0].height
let rawFrameSize = width * height * 4 // RGBA 8bit

print("--- Settings ---")
print("Input frames   : \(images.count)")
print("Resolution     : \(width)x\(height)")
print("Target Bitrate : \(config.bitrate) kbps")
print("Target FPS     : \(config.framerate)")
print("----------------")

// MARK: - VEVC Encode / Decode
func runVEVC(images: [ImageInput], config: Config) async throws -> (encTime: Double, decTime: Double, compSize: Int) {
    let vevcImages = images.map { $0.vevcImage }
    
    // Encode
    let encStart = Date()
    let outBytes: [UInt8] = try await vevc.encode(images: vevcImages, maxbitrate: config.bitrate * 1000, zeroThreshold: config.zeroThreshold, gopSize: config.gopSize, sceneChangeThreshold: config.sceneThreshold)
    let encTime = Date().timeIntervalSince(encStart)
    
    // Decode
    let opts = vevc.DecodeOptions(maxLayer: config.maxLayer, maxFrames: 4)
    let decStart = Date()
    let _ = try await vevc.decode(data: outBytes, opts: opts)
    let decTime = Date().timeIntervalSince(decStart)
    
    return (encTime, decTime, outBytes.count)
}

func runVEVCOne(images: [ImageInput], config: Config) async throws -> (encTime: Double, decTime: Double, compSize: Int) {
    let vevcImages = images.map { $0.vevcImage }
    
    // Encode
    let encStart = Date()
    let outBytes: [UInt8] = try await vevc.encodeOne(images: vevcImages, maxbitrate: config.bitrate * 1000, zeroThreshold: config.zeroThreshold, gopSize: config.gopSize, sceneChangeThreshold: config.sceneThreshold)
    let encTime = Date().timeIntervalSince(encStart)
    
    // Decode
    let decStart = Date()
    let _ = try await vevc.decodeOne(data: outBytes)
    let decTime = Date().timeIntervalSince(decStart)
    
    return (encTime, decTime, outBytes.count)
}

// MARK: - H264 Encode / Decode (VideoToolbox)
func createPixelBuffer(from rgba: [PNG.RGBA<UInt8>], width: Int, height: Int) -> CVPixelBuffer? {
    let attrs = [
        kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
        kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
    ] as CFDictionary
    
    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
    guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }
    
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    
    guard let context = CGContext(
        data: CVPixelBufferGetBaseAddress(buffer),
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue,
    ) else { return nil }
    
    // ARGB: skip first byte for Alpha/Padding, then R, G, B
    let ptr = context.data!.bindMemory(to: UInt8.self, capacity: width * height * 4)
    for i in 0..<(width * height) {
        let rgbaPixel = rgba[i]
        ptr[i * 4 + 0] = 255 // A (Ignored)
        ptr[i * 4 + 1] = rgbaPixel.r
        ptr[i * 4 + 2] = rgbaPixel.g
        ptr[i * 4 + 3] = rgbaPixel.b
    }
    
    return buffer
}

func runH264(images: [ImageInput], config: Config, width: Int, height: Int) async throws -> (encTime: Double, decTime: Double, compSize: Int) {
    var encTime: Double = 0
    var compSize: Int = 0
    
    // We must use a class to capture it safely in the C callback without escaping unsafe pointers.
    class FrameBox {
        var frames: [CMSampleBuffer] = []
    }
    let frameBox = FrameBox()
    
    // 1. Setup Compression Session
    var compressionSessionOut: VTCompressionSession?
    let status = VTCompressionSessionCreate(
        allocator: kCFAllocatorDefault,
        width: Int32(width),
        height: Int32(height),
        codecType: kCMVideoCodecType_H264,
        encoderSpecification: nil,
        imageBufferAttributes: nil,
        compressedDataAllocator: nil,
        outputCallback: { (outputCallbackRefCon, _, status, infoFlags, sampleBuffer) in
            guard status == noErr, let sampleBuffer = sampleBuffer else { return }
            
            let box = Unmanaged<FrameBox>.fromOpaque(outputCallbackRefCon!).takeUnretainedValue()
            box.frames.append(sampleBuffer)
        },
        refcon: Unmanaged.passUnretained(frameBox).toOpaque(),
        compressionSessionOut: &compressionSessionOut,)
    
    guard status == noErr, let compressionSession = compressionSessionOut else {
        throw NSError(domain: "VTCompressionSessionCreate", code: Int(status), userInfo: nil)
    }
    
    // Properties
    let bitRateBps = config.bitrate * 1000
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_AverageBitRate, value: NSNumber(value: bitRateBps))
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_DataRateLimits, value: [bitRateBps / 8 * 2, 1] as CFArray)
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_ExpectedFrameRate, value: NSNumber(value: config.framerate))
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_RealTime, value: kCFBooleanTrue)
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_ProfileLevel, value: kVTProfileLevel_H264_Baseline_AutoLevel)
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_AllowFrameReordering, value: kCFBooleanTrue)

    VTCompressionSessionPrepareToEncodeFrames(compressionSession)
    
    // Encode loop
    let encStart = Date()
    for (idx, imgInput) in images.enumerated() {
        guard let pixelBuffer = createPixelBuffer(from: imgInput.rgbaFrames, width: width, height: height) else { continue }
        
        // Define frame time
        let presentationTimeStamp = CMTime(value: CMTimeValue(idx), timescale: CMTimeScale(config.framerate))
        
        var flags: VTEncodeInfoFlags = []
        VTCompressionSessionEncodeFrame(
            compressionSession,
            imageBuffer: pixelBuffer,
            presentationTimeStamp: presentationTimeStamp,
            duration: .invalid,
            frameProperties: nil,
            sourceFrameRefcon: nil,
            infoFlagsOut: &flags,
        )
    }
    
    VTCompressionSessionCompleteFrames(compressionSession, untilPresentationTimeStamp: .invalid)
    encTime = Date().timeIntervalSince(encStart)
    
    // Calculate size
    for sample in frameBox.frames {
        if let dataBuffer = CMSampleBufferGetDataBuffer(sample) {
            compSize += CMBlockBufferGetDataLength(dataBuffer)
        }
    }
    
    // 2. Setup Decompression Session
    var decTime: Double = 0
    guard !frameBox.frames.isEmpty else { return (encTime, decTime, compSize) }
    
    // Need format desc for decompression
    guard let formatDesc = CMSampleBufferGetFormatDescription(frameBox.frames[0]) else {
        throw NSError(domain: "CMSampleBufferGetFormatDescription", code: -1, userInfo: nil)
    }
    
    let destPixelBufferAttributes: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferMetalCompatibilityKey as String: true
    ]
    
    var decompressionSessionOut: VTDecompressionSession?
    let decStatus = VTDecompressionSessionCreate(
        allocator: kCFAllocatorDefault,
        formatDescription: formatDesc,
        decoderSpecification: nil,
        imageBufferAttributes: destPixelBufferAttributes as CFDictionary,
        outputCallback: nil,
        decompressionSessionOut: &decompressionSessionOut,
    )
                                                
    guard decStatus == noErr, let decompressionSession = decompressionSessionOut else {
        throw NSError(domain: "VTDecompressionSessionCreate", code: Int(decStatus), userInfo: nil)
    }

    // Decode loop
    let decStart = Date()
    for sample in frameBox.frames {
        var flags: VTDecodeInfoFlags = []
        VTDecompressionSessionDecodeFrame(
            decompressionSession,
            sampleBuffer: sample,
            flags: [],
            infoFlagsOut: &flags,
            outputHandler: { (status, infoFlags, imageBuffer, presentationTimeStamp, presentationDuration) in
                // nop
            },
        )
    }
    VTDecompressionSessionWaitForAsynchronousFrames(decompressionSession)
    decTime = Date().timeIntervalSince(decStart)

    return (encTime, decTime, compSize)
}

// MARK: - HEVC Encode / Decode (VideoToolbox)
func runHEVC(images: [ImageInput], config: Config, width: Int, height: Int) async throws -> (encTime: Double, decTime: Double, compSize: Int) {
    var encTime: Double = 0
    var compSize: Int = 0
    
    class FrameBox {
        var frames: [CMSampleBuffer] = []
    }
    let frameBox = FrameBox()
    
    // 1. Setup Compression Session
    var compressionSessionOut: VTCompressionSession?
    let status = VTCompressionSessionCreate(
        allocator: kCFAllocatorDefault,
        width: Int32(width),
        height: Int32(height),
        codecType: kCMVideoCodecType_HEVC,
        encoderSpecification: nil,
        imageBufferAttributes: nil,
        compressedDataAllocator: nil,
        outputCallback: { (outputCallbackRefCon, _, status, infoFlags, sampleBuffer) in
            guard status == noErr, let sampleBuffer = sampleBuffer else { return }
            
            let box = Unmanaged<FrameBox>.fromOpaque(outputCallbackRefCon!).takeUnretainedValue()
            box.frames.append(sampleBuffer)
        },
        refcon: Unmanaged.passUnretained(frameBox).toOpaque(),
        compressionSessionOut: &compressionSessionOut,
    )
    
    guard status == noErr, let compressionSession = compressionSessionOut else {
        throw NSError(domain: "VTCompressionSessionCreate (HEVC)", code: Int(status), userInfo: nil)
    }
    
    // Properties
    let bitRateBps = config.bitrate * 1000
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_AverageBitRate, value: NSNumber(value: bitRateBps))
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_DataRateLimits, value: [bitRateBps / 8 * 2, 1] as CFArray)
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_ExpectedFrameRate, value: NSNumber(value: config.framerate))
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_RealTime, value: kCFBooleanTrue)
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_ProfileLevel, value: kVTProfileLevel_HEVC_Main_AutoLevel)
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_AllowFrameReordering, value: kCFBooleanTrue)

    VTCompressionSessionPrepareToEncodeFrames(compressionSession)
    
    // Encode loop
    let encStart = Date()
    for (idx, imgInput) in images.enumerated() {
        guard let pixelBuffer = createPixelBuffer(from: imgInput.rgbaFrames, width: width, height: height) else { continue }
        
        let presentationTimeStamp = CMTime(value: CMTimeValue(idx), timescale: CMTimeScale(config.framerate))
        var flags: VTEncodeInfoFlags = []
        VTCompressionSessionEncodeFrame(
            compressionSession,
            imageBuffer: pixelBuffer,
            presentationTimeStamp: presentationTimeStamp,
            duration: .invalid,
            frameProperties: nil,
            sourceFrameRefcon: nil,
            infoFlagsOut: &flags,
        )
    }
    
    VTCompressionSessionCompleteFrames(compressionSession, untilPresentationTimeStamp: .invalid)
    encTime = Date().timeIntervalSince(encStart)
    
    // Calculate size
    for sample in frameBox.frames {
        if let dataBuffer = CMSampleBufferGetDataBuffer(sample) {
            compSize += CMBlockBufferGetDataLength(dataBuffer)
        }
    }
    
    // 2. Setup Decompression Session
    var decTime: Double = 0
    guard !frameBox.frames.isEmpty else { return (encTime, decTime, compSize) }
    
    guard let formatDesc = CMSampleBufferGetFormatDescription(frameBox.frames[0]) else {
        throw NSError(domain: "CMSampleBufferGetFormatDescription (HEVC)", code: -1, userInfo: nil)
    }
    
    let destPixelBufferAttributes: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferMetalCompatibilityKey as String: true
    ]
    
    var decompressionSessionOut: VTDecompressionSession?
    let decStatus = VTDecompressionSessionCreate(
        allocator: kCFAllocatorDefault,
        formatDescription: formatDesc,
        decoderSpecification: nil,
        imageBufferAttributes: destPixelBufferAttributes as CFDictionary,
        outputCallback: nil,
        decompressionSessionOut: &decompressionSessionOut,
    )
                                                
    guard decStatus == noErr, let decompressionSession = decompressionSessionOut else {
        throw NSError(domain: "VTDecompressionSessionCreate (HEVC)", code: Int(decStatus), userInfo: nil)
    }

    // Decode loop
    let decStart = Date()
    for sample in frameBox.frames {
        var flags: VTDecodeInfoFlags = []
        VTDecompressionSessionDecodeFrame(
            decompressionSession,
            sampleBuffer: sample,
            flags: [],
            infoFlagsOut: &flags,
            outputHandler: { (status, infoFlags, imageBuffer, presentationTimeStamp, presentationDuration) in
                // nop
            },
        )
    }
    VTDecompressionSessionWaitForAsynchronousFrames(decompressionSession)
    decTime = Date().timeIntervalSince(decStart)

    return (encTime, decTime, compSize)
}

// MARK: - MJPEG Encode / Decode (VideoToolbox)
func runMJPEG(images: [ImageInput], config: Config, width: Int, height: Int) async throws -> (encTime: Double, decTime: Double, compSize: Int) {
    var encTime: Double = 0
    var compSize: Int = 0
    
    class FrameBox {
        var frames: [CMSampleBuffer] = []
    }
    let frameBox = FrameBox()
    
    // 1. Setup Compression Session
    var compressionSessionOut: VTCompressionSession?
    let status = VTCompressionSessionCreate(
        allocator: kCFAllocatorDefault,
        width: Int32(width),
        height: Int32(height),
        codecType: kCMVideoCodecType_JPEG,
        encoderSpecification: nil,
        imageBufferAttributes: nil,
        compressedDataAllocator: nil,
        outputCallback: { (outputCallbackRefCon, _, status, infoFlags, sampleBuffer) in
            guard status == noErr, let sampleBuffer = sampleBuffer else { return }
            
            let box = Unmanaged<FrameBox>.fromOpaque(outputCallbackRefCon!).takeUnretainedValue()
            box.frames.append(sampleBuffer)
        },
        refcon: Unmanaged.passUnretained(frameBox).toOpaque(),
        compressionSessionOut: &compressionSessionOut,
    )
    
    guard status == noErr, let compressionSession = compressionSessionOut else {
        throw NSError(domain: "VTCompressionSessionCreate (MJPEG)", code: Int(status), userInfo: nil)
    }
    
    // Properties
    let bitRateBps = config.bitrate * 1000
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_AverageBitRate, value: NSNumber(value: bitRateBps))
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_DataRateLimits, value: [bitRateBps / 8 * 2, 1] as CFArray)
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_ExpectedFrameRate, value: NSNumber(value: config.framerate))
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_RealTime, value: kCFBooleanTrue)

    VTCompressionSessionPrepareToEncodeFrames(compressionSession)
    
    // Encode loop
    let encStart = Date()
    for (idx, imgInput) in images.enumerated() {
        guard let pixelBuffer = createPixelBuffer(from: imgInput.rgbaFrames, width: width, height: height) else { continue }
        
        let presentationTimeStamp = CMTime(value: CMTimeValue(idx), timescale: CMTimeScale(config.framerate))
        var flags: VTEncodeInfoFlags = []
        VTCompressionSessionEncodeFrame(
            compressionSession,
            imageBuffer: pixelBuffer,
            presentationTimeStamp: presentationTimeStamp,
            duration: .invalid,
            frameProperties: nil,
            sourceFrameRefcon: nil,
            infoFlagsOut: &flags,
        )
    }
    
    VTCompressionSessionCompleteFrames(compressionSession, untilPresentationTimeStamp: .invalid)
    encTime = Date().timeIntervalSince(encStart)
    
    // Calculate size
    for sample in frameBox.frames {
        if let dataBuffer = CMSampleBufferGetDataBuffer(sample) {
            compSize += CMBlockBufferGetDataLength(dataBuffer)
        }
    }
    
    // 2. Setup Decompression Session
    var decTime: Double = 0
    guard !frameBox.frames.isEmpty else { return (encTime, decTime, compSize) }
    
    guard let formatDesc = CMSampleBufferGetFormatDescription(frameBox.frames[0]) else {
        throw NSError(domain: "CMSampleBufferGetFormatDescription (MJPEG)", code: -1, userInfo: nil)
    }
    
    let destPixelBufferAttributes: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferMetalCompatibilityKey as String: true
    ]
    
    var decompressionSessionOut: VTDecompressionSession?
    let decStatus = VTDecompressionSessionCreate(
        allocator: kCFAllocatorDefault,
        formatDescription: formatDesc,
        decoderSpecification: nil,
        imageBufferAttributes: destPixelBufferAttributes as CFDictionary,
        outputCallback: nil,
        decompressionSessionOut: &decompressionSessionOut,
    )
                                                
    guard decStatus == noErr, let decompressionSession = decompressionSessionOut else {
        throw NSError(domain: "VTDecompressionSessionCreate (MJPEG)", code: Int(decStatus), userInfo: nil)
    }

    // Decode loop
    let decStart = Date()
    for sample in frameBox.frames {
        var flags: VTDecodeInfoFlags = []
        VTDecompressionSessionDecodeFrame(
            decompressionSession,
            sampleBuffer: sample,
            flags: [],
            infoFlagsOut: &flags,
            outputHandler: { (status, infoFlags, imageBuffer, presentationTimeStamp, presentationDuration) in
                // nop
            },
        )
    }
    VTDecompressionSessionWaitForAsynchronousFrames(decompressionSession)
    decTime = Date().timeIntervalSince(decStart)

    return (encTime, decTime, compSize)
}

// MARK: - Main Execution
Task {
    // Top-level variables captured inside Task locally to avoid isolation errors
    let localImages = images
    let localConfig = config
    let localWidth = width
    let localHeight = height
    let rawTotalSizeKB = Double(localImages.count * localWidth * localHeight * 3) / 1024.0 // Assuming YCbCr size calculation standard. H264 is YUV 4:2:0 mostly.
    
    do {
        print("Running vevc...")
        let vevcResult = try await runVEVC(images: localImages, config: localConfig)
        
        print("Running vevc (One)...")
        let vevcOneResult = try await runVEVCOne(images: localImages, config: localConfig)
        
        print("Running H.264 (VideoToolbox)...")
        let h264Result = try await runH264(images: localImages, config: localConfig, width: localWidth, height: localHeight)
        
        print("Running HEVC (VideoToolbox)...")
        let hevcResult = try await runHEVC(images: localImages, config: localConfig, width: localWidth, height: localHeight)
        
        print("Running MJPEG (VideoToolbox)...")
        let mjpegResult = try await runMJPEG(images: localImages, config: localConfig, width: localWidth, height: localHeight)
        
        func printStats(name: String, result: (encTime: Double, decTime: Double, compSize: Int), count: Int, rawSizeKB: Double) {
            let encMs = result.encTime * 1000
            let decMs = result.decTime * 1000
            let encFps = Double(count) / result.encTime
            let decFps = Double(count) / result.decTime
            let sizeKB = Double(result.compSize) / 1024.0
            
            print("[\(name)]")
            print(String(format: "  Encode : %7.2f ms (%.2f fps) - %.2f ms / frame", encMs, encFps, encMs / Double(count)))
            print(String(format: "  Decode : %7.2f ms (%.2f fps) - %.2f ms / frame", decMs, decFps, decMs / Double(count)))
            print(String(format: "  Size   : %7.2f KB (%.2f%% of raw %.2f KB)", sizeKB, (sizeKB / rawSizeKB) * 100.0, rawSizeKB))
        }
        
        print("\n--- Results ---")
        printStats(name: "VEVC", result: vevcResult, count: localImages.count, rawSizeKB: rawTotalSizeKB)
        printStats(name: "VEVC (One)", result: vevcOneResult, count: localImages.count, rawSizeKB: rawTotalSizeKB)
        printStats(name: "H.264", result: h264Result, count: localImages.count, rawSizeKB: rawTotalSizeKB)
        printStats(name: "HEVC", result: hevcResult, count: localImages.count, rawSizeKB: rawTotalSizeKB)
        printStats(name: "MJPEG", result: mjpegResult, count: localImages.count, rawSizeKB: rawTotalSizeKB)
        print("---------------")
        
    } catch {
        print("Error: \(error)")
        exit(1)
    }
    exit(0)
}

RunLoop.main.run()
