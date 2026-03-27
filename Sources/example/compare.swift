import Foundation
import VideoToolbox
import CoreMedia
import PNG
import vevc

struct Config {
    var bitrate: Int = 500
    var framerate: Int = 60
    var zeroThreshold: Int = 0
    var keyint: Int = 60
    var sceneThreshold: Int = 32
    var maxLayer: Int = 2
    var quality: Bool = false
    var outputGraph: Bool = false
}

struct ImageInput {
    let vevcImage: YCbCrImage
    let rgbaFrames: [PNG.RGBA<UInt8>]
    let width: Int
    let height: Int
}

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

func readY4M(path: String) -> [ImageInput]? {
    let fileHandle: FileHandle
    if path == "-" {
        fileHandle = FileHandle.standardInput
    } else {
        guard let f = FileHandle(forReadingAtPath: path) else { return nil }
        fileHandle = f
    }
    defer { if path != "-" { fileHandle.closeFile() } }
    
    guard let reader = try? Y4MReader(fileHandle: fileHandle) else { return nil }
    var inputs: [ImageInput] = []
    
    while let img = try? reader.readFrame() {
        let width = reader.width
        let height = reader.height
        
        let rawData = vevc.ycbcrToRGBA(img: img)
        let pixelCount = width * height
        var rgbaFrames = [PNG.RGBA<UInt8>](repeating: .init(0, 0, 0, 0), count: pixelCount)
        
        rawData.withUnsafeBufferPointer { rawPtr in
            rgbaFrames.withUnsafeMutableBufferPointer { rgbaPtr in
                guard let rawBase = rawPtr.baseAddress, let rgbaBase = rgbaPtr.baseAddress else { return }
                for i in 0..<pixelCount {
                    let offset = i * 4
                    rgbaBase[i] = PNG.RGBA<UInt8>(rawBase[offset], rawBase[offset + 1], rawBase[offset + 2], rawBase[offset + 3])
                }
            }
        }
        
        inputs.append(ImageInput(vevcImage: img, rgbaFrames: rgbaFrames, width: width, height: height))
    }
    return inputs
}

// MARK: - VEVC Encode / Decode
func runVEVC(images: [ImageInput], config: Config) async throws -> (encTime: Double, decTime: Double, compSize: Int, metrics: [QualityMetrics]?) {
    let vevcImages = images.map { $0.vevcImage }
    
    // Encode
    print("  -> runVEVC Encoding...")
    let encStart = Date()
    let outBytes: [UInt8] = try await vevc.encode(images: vevcImages, maxbitrate: config.bitrate * 1000, framerate: config.framerate, zeroThreshold: config.zeroThreshold, keyint: config.keyint, sceneChangeThreshold: config.sceneThreshold)
    let encTime = Date().timeIntervalSince(encStart)
    print("  -> runVEVC Encoded \(outBytes.count) bytes")
    
    // Decode
    print("  -> runVEVC Decoding...")
    let opts = vevc.DecodeOptions(maxLayer: config.maxLayer, maxFrames: 4)
    let decStart = Date()
    let outFrames = try await vevc.decode(data: outBytes, opts: opts)
    let decTime = Date().timeIntervalSince(decStart)
    print("  -> runVEVC Decoded \(outFrames.count) frames")
    
    var metrics: [QualityMetrics]? = nil
    if config.quality {
        var mets = [QualityMetrics]()
        for i in 0..<min(images.count, outFrames.count) {
            let psnr = calculatePSNR(img1: images[i].vevcImage, img2: outFrames[i])
            let ssim = calculateSSIM(img1: images[i].vevcImage, img2: outFrames[i])
            mets.append(QualityMetrics(psnr: psnr, ssim: ssim))
        }
        metrics = mets
    }
    
    return (encTime, decTime, outBytes.count, metrics)
}

func runVEVCOne(images: [ImageInput], config: Config) async throws -> (encTime: Double, decTime: Double, compSize: Int, metrics: [QualityMetrics]?) {
    let vevcImages = images.map { $0.vevcImage }
    
    // Encode
    let encStart = Date()
    let outBytes: [UInt8] = try await vevc.encode(images: vevcImages, maxbitrate: config.bitrate * 1000, framerate: config.framerate, zeroThreshold: config.zeroThreshold, keyint: config.keyint, sceneChangeThreshold: config.sceneThreshold)
    let encTime = Date().timeIntervalSince(encStart)
    
    // Decode
    let decStart = Date()
    let opts = vevc.DecodeOptions(maxLayer: 2, maxFrames: 4)
    let outFrames = try await vevc.decode(data: outBytes, opts: opts)
    let decTime = Date().timeIntervalSince(decStart)
    
    var metrics: [QualityMetrics]? = nil
    if config.quality {
        var mets = [QualityMetrics]()
        for i in 0..<min(images.count, outFrames.count) {
            let psnr = calculatePSNR(img1: images[i].vevcImage, img2: outFrames[i])
            let ssim = calculateSSIM(img1: images[i].vevcImage, img2: outFrames[i])
            mets.append(QualityMetrics(psnr: psnr, ssim: ssim))
        }
        metrics = mets
    }
    
    return (encTime, decTime, outBytes.count, metrics)
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
    
    // Y Plane
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
    
    // UV Plane (BiPlanar)
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

func runH264(images: [ImageInput], config: Config, width: Int, height: Int, disableHWA: Bool = false) async throws -> (encTime: Double, decTime: Double, compSize: Int, metrics: [QualityMetrics]?) {
    var encTime: Double = 0
    var compSize: Int = 0
    
    // We must use a class to capture it safely in the C callback without escaping unsafe pointers.
    class FrameBox: @unchecked Sendable {
        var frames: [CMSampleBuffer] = []
        var decodedBuffers: [Int: CVPixelBuffer] = [:]
        let lock = NSLock()
    }
    let frameBox = FrameBox()
    
    // Encoder/decoder spec to disable HWA
    let encoderSpec: CFDictionary? = disableHWA ? ([
        kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder: false,
        kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder: false
    ] as CFDictionary) : nil
    
    // 1. Setup Compression Session
    var compressionSessionOut: VTCompressionSession?
    let status = VTCompressionSessionCreate(
        allocator: kCFAllocatorDefault,
        width: Int32(width),
        height: Int32(height),
        codecType: kCMVideoCodecType_H264,
        encoderSpecification: encoderSpec,
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
    
    // Pre-create pixel buffers for fair encoding speed comparison
    var encodeBuffers: [CVPixelBuffer] = []
    for imgInput in images {
        if let pb = createPixelBuffer(from: imgInput.vevcImage) {
            encodeBuffers.append(pb)
        }
    }
    
    // Encode loop
    let encStart = Date()
    for (idx, pixelBuffer) in encodeBuffers.enumerated() {
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
    guard !frameBox.frames.isEmpty else { return (encTime, decTime, compSize, nil) }
    
    // Need format desc for decompression
    guard let formatDesc = CMSampleBufferGetFormatDescription(frameBox.frames[0]) else {
        throw NSError(domain: "CMSampleBufferGetFormatDescription", code: -1, userInfo: nil)
    }
    
    let destPixelBufferAttributes: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        kCVPixelBufferMetalCompatibilityKey as String: true
    ]
    
    // Decoder spec to disable HWA
    let decoderSpec: CFDictionary? = disableHWA ? ([
        kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder: false,
        kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder: false
    ] as CFDictionary) : nil
    
    var decompressionSessionOut: VTDecompressionSession?
    let decStatus = VTDecompressionSessionCreate(
        allocator: kCFAllocatorDefault,
        formatDescription: formatDesc,
        decoderSpecification: decoderSpec,
        imageBufferAttributes: destPixelBufferAttributes as CFDictionary,
        outputCallback: nil,
        decompressionSessionOut: &decompressionSessionOut,
    )
                                                
    guard decStatus == noErr, let decompressionSession = decompressionSessionOut else {
        throw NSError(domain: "VTDecompressionSessionCreate", code: Int(decStatus), userInfo: nil)
    }

    // Decode loop (Speed Pass)
    let decStart = Date()
    for sample in frameBox.frames {
        var flags: VTDecodeInfoFlags = []
        VTDecompressionSessionDecodeFrame(
            decompressionSession,
            sampleBuffer: sample,
            flags: [],
            infoFlagsOut: &flags,
            outputHandler: { (_, _, _, _, _) in }
        )
    }
    VTDecompressionSessionWaitForAsynchronousFrames(decompressionSession)
    decTime = Date().timeIntervalSince(decStart)

    var metrics: [QualityMetrics]? = nil
    if config.quality {
        var qualitySessionOut: VTDecompressionSession?
        let qualityStatus = VTDecompressionSessionCreate(
            allocator: kCFAllocatorDefault,
            formatDescription: formatDesc,
            decoderSpecification: decoderSpec,
            imageBufferAttributes: destPixelBufferAttributes as CFDictionary,
            outputCallback: nil,
            decompressionSessionOut: &qualitySessionOut
        )
        guard qualityStatus == noErr, let qualitySession = qualitySessionOut else {
            throw NSError(domain: "VTDecompressionSessionCreate (H264 Quality)", code: Int(qualityStatus), userInfo: nil)
        }
        
        for sample in frameBox.frames {
            var flags: VTDecodeInfoFlags = []
            VTDecompressionSessionDecodeFrame(
                qualitySession,
                sampleBuffer: sample,
                flags: [],
                infoFlagsOut: &flags,
                outputHandler: { (status, infoFlags, imageBuffer, presentationTimeStamp, presentationDuration) in
                    if let buf = imageBuffer {
                        let idx = Int(presentationTimeStamp.value)
                        frameBox.lock.lock()
                        frameBox.decodedBuffers[idx] = buf
                        frameBox.lock.unlock()
                    }
                }
            )
        }
        VTDecompressionSessionWaitForAsynchronousFrames(qualitySession)

        var mets = [QualityMetrics]()
        for i in 0..<images.count {
            if let buf = frameBox.decodedBuffers[i] {
                let psnr = calculatePSNR(img1: images[i].vevcImage, bgraBuffer: buf)
                let ssim = calculateSSIM(img1: images[i].vevcImage, bgraBuffer: buf)
                mets.append(QualityMetrics(psnr: psnr, ssim: ssim))
            }
        }
        metrics = mets
    }

    return (encTime, decTime, compSize, metrics)
}

// MARK: - HEVC Encode / Decode (VideoToolbox)
func runHEVC(images: [ImageInput], config: Config, width: Int, height: Int, disableHWA: Bool = false) async throws -> (encTime: Double, decTime: Double, compSize: Int, metrics: [QualityMetrics]?) {
    var encTime: Double = 0
    var compSize: Int = 0
    
    class FrameBox: @unchecked Sendable {
        var frames: [CMSampleBuffer] = []
        var decodedBuffers: [Int: CVPixelBuffer] = [:]
        let lock = NSLock()
    }
    let frameBox = FrameBox()
    
    // Encoder/decoder spec to disable HWA
    let encoderSpec: CFDictionary? = disableHWA ? ([
        kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder: false,
        kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder: false
    ] as CFDictionary) : nil
    
    // 1. Setup Compression Session
    var compressionSessionOut: VTCompressionSession?
    let status = VTCompressionSessionCreate(
        allocator: kCFAllocatorDefault,
        width: Int32(width),
        height: Int32(height),
        codecType: kCMVideoCodecType_HEVC,
        encoderSpecification: encoderSpec,
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
    
    // Pre-create pixel buffers for fair encoding speed comparison
    var encodeBuffers: [CVPixelBuffer] = []
    for imgInput in images {
        if let pb = createPixelBuffer(from: imgInput.vevcImage) {
            encodeBuffers.append(pb)
        }
    }
    
    // Encode loop
    let encStart = Date()
    for (idx, pixelBuffer) in encodeBuffers.enumerated() {
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
    guard !frameBox.frames.isEmpty else { return (encTime, decTime, compSize, nil) }
    
    guard let formatDesc = CMSampleBufferGetFormatDescription(frameBox.frames[0]) else {
        throw NSError(domain: "CMSampleBufferGetFormatDescription (HEVC)", code: -1, userInfo: nil)
    }
    
    let destPixelBufferAttributes: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        kCVPixelBufferMetalCompatibilityKey as String: true
    ]
    
    // Decoder spec to disable HWA
    let decoderSpec: CFDictionary? = disableHWA ? ([
        kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder: false,
        kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder: false
    ] as CFDictionary) : nil
    
    var decompressionSessionOut: VTDecompressionSession?
    let decStatus = VTDecompressionSessionCreate(
        allocator: kCFAllocatorDefault,
        formatDescription: formatDesc,
        decoderSpecification: decoderSpec,
        imageBufferAttributes: destPixelBufferAttributes as CFDictionary,
        outputCallback: nil,
        decompressionSessionOut: &decompressionSessionOut,
    )
                                                
    guard decStatus == noErr, let decompressionSession = decompressionSessionOut else {
        throw NSError(domain: "VTDecompressionSessionCreate (HEVC)", code: Int(decStatus), userInfo: nil)
    }

    // Decode loop (Speed Pass)
    let decStart = Date()
    for sample in frameBox.frames {
        var flags: VTDecodeInfoFlags = []
        VTDecompressionSessionDecodeFrame(
            decompressionSession,
            sampleBuffer: sample,
            flags: [],
            infoFlagsOut: &flags,
            outputHandler: { (_, _, _, _, _) in }
        )
    }
    VTDecompressionSessionWaitForAsynchronousFrames(decompressionSession)
    decTime = Date().timeIntervalSince(decStart)

    var metrics: [QualityMetrics]? = nil
    if config.quality {
        var qualitySessionOut: VTDecompressionSession?
        let qualityStatus = VTDecompressionSessionCreate(
            allocator: kCFAllocatorDefault,
            formatDescription: formatDesc,
            decoderSpecification: decoderSpec,
            imageBufferAttributes: destPixelBufferAttributes as CFDictionary,
            outputCallback: nil,
            decompressionSessionOut: &qualitySessionOut
        )
        guard qualityStatus == noErr, let qualitySession = qualitySessionOut else {
            throw NSError(domain: "VTDecompressionSessionCreate (HEVC Quality)", code: Int(qualityStatus), userInfo: nil)
        }
        
        for sample in frameBox.frames {
            var flags: VTDecodeInfoFlags = []
            VTDecompressionSessionDecodeFrame(
                qualitySession,
                sampleBuffer: sample,
                flags: [],
                infoFlagsOut: &flags,
                outputHandler: { (status, infoFlags, imageBuffer, presentationTimeStamp, presentationDuration) in
                    if let buf = imageBuffer {
                        let idx = Int(presentationTimeStamp.value)
                        frameBox.lock.lock()
                        frameBox.decodedBuffers[idx] = buf
                        frameBox.lock.unlock()
                    }
                }
            )
        }
        VTDecompressionSessionWaitForAsynchronousFrames(qualitySession)
        
        var mets = [QualityMetrics]()
        for i in 0..<images.count {
            if let buf = frameBox.decodedBuffers[i] {
                let psnr = calculatePSNR(img1: images[i].vevcImage, bgraBuffer: buf)
                let ssim = calculateSSIM(img1: images[i].vevcImage, bgraBuffer: buf)
                mets.append(QualityMetrics(psnr: psnr, ssim: ssim))
            }
        }
        metrics = mets
    }

    return (encTime, decTime, compSize, metrics)
}

// MARK: - MJPEG Encode / Decode (VideoToolbox)
func runMJPEG(images: [ImageInput], config: Config, width: Int, height: Int) async throws -> (encTime: Double, decTime: Double, compSize: Int, metrics: [QualityMetrics]?) {
    var encTime: Double = 0
    var compSize: Int = 0
    
    class FrameBox: @unchecked Sendable {
        var frames: [CMSampleBuffer] = []
        var decodedBuffers: [Int: CVPixelBuffer] = [:]
        let lock = NSLock()
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
    
    // Pre-create pixel buffers for fair encoding speed comparison
    var encodeBuffers: [CVPixelBuffer] = []
    for imgInput in images {
        if let pb = createPixelBuffer(from: imgInput.vevcImage) {
            encodeBuffers.append(pb)
        }
    }
    
    // Encode loop
    let encStart = Date()
    for (idx, pixelBuffer) in encodeBuffers.enumerated() {
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
    guard !frameBox.frames.isEmpty else { return (encTime, decTime, compSize, nil) }
    
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

    // Decode loop (Speed Pass)
    let decStart = Date()
    for sample in frameBox.frames {
        var flags: VTDecodeInfoFlags = []
        VTDecompressionSessionDecodeFrame(
            decompressionSession,
            sampleBuffer: sample,
            flags: [],
            infoFlagsOut: &flags,
            outputHandler: { (_, _, _, _, _) in }
        )
    }
    VTDecompressionSessionWaitForAsynchronousFrames(decompressionSession)
    decTime = Date().timeIntervalSince(decStart)

    var metrics: [QualityMetrics]? = nil
    if config.quality {
        var qualitySessionOut: VTDecompressionSession?
        let qualityStatus = VTDecompressionSessionCreate(
            allocator: kCFAllocatorDefault,
            formatDescription: formatDesc,
            decoderSpecification: nil,
            imageBufferAttributes: destPixelBufferAttributes as CFDictionary,
            outputCallback: nil,
            decompressionSessionOut: &qualitySessionOut
        )
        guard qualityStatus == noErr, let qualitySession = qualitySessionOut else {
            throw NSError(domain: "VTDecompressionSessionCreate (MJPEG Quality)", code: Int(qualityStatus), userInfo: nil)
        }
        
        for sample in frameBox.frames {
            var flags: VTDecodeInfoFlags = []
            VTDecompressionSessionDecodeFrame(
                qualitySession,
                sampleBuffer: sample,
                flags: [],
                infoFlagsOut: &flags,
                outputHandler: { (status, infoFlags, imageBuffer, presentationTimeStamp, presentationDuration) in
                    if let buf = imageBuffer {
                        let idx = Int(presentationTimeStamp.value)
                        frameBox.lock.lock()
                        frameBox.decodedBuffers[idx] = buf
                        frameBox.lock.unlock()
                    }
                }
            )
        }
        VTDecompressionSessionWaitForAsynchronousFrames(qualitySession)
        
        var mets = [QualityMetrics]()
        for i in 0..<images.count {
            if let buf = frameBox.decodedBuffers[i] {
                let psnr = calculatePSNR(img1: images[i].vevcImage, bgraBuffer: buf)
                let ssim = calculateSSIM(img1: images[i].vevcImage, bgraBuffer: buf)
                mets.append(QualityMetrics(psnr: psnr, ssim: ssim))
            }
        }
        metrics = mets
    }

    return (encTime, decTime, compSize, metrics)
}


@main
struct CompareApp {
    static func main() async throws {
    let args = CommandLine.arguments

    var config = Config()
    var positionalArgs: [String] = []
    var y4mPath: String? = nil

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
        case "-keyint":
            if (i + 1) < args.count {
                if let v = Int(args[i + 1]) { config.keyint = v }
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
        case "-quality":
            config.quality = true
        case "-output-graph":
            config.outputGraph = true
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
        print("Usage: compare [-y4m <input.y4m>] [-bitrate <kbits>] [-framerate <fps>] [-zeroThreshold <threshold>] [-keyint <frames>] [-sceneThreshold <sad>] [-maxLayer <0-2>] [-quality] [-output-graph] [<input1.png> input2.png ...]")
        exit(1)
    }

    var images: [ImageInput] = []
    if let y4m = y4mPath {
        if let y4mImages = readY4M(path: y4m) {
            images = y4mImages
        } else {
            print("Failed to read y4m: \(y4m)")
        }
    } else {
        for p in positionalArgs {
            if let imgData = readPNG(path: p) {
                images.append(ImageInput(vevcImage: imgData.image, rgbaFrames: imgData.rawData, width: imgData.width, height: imgData.height))
            } else {
                print("Failed to read \(p)")
            }
        }
    }

    if images.isEmpty {
        print("No valid input images found.")
        exit(1)
    }

    let width = images[0].width
    let height = images[0].height

    print("--- Settings ---")
    print("Input frames   : \(images.count)")
    print("Resolution     : \(width)x\(height)")
    print("Target Bitrate : \(config.bitrate) kbps")
    print("Target FPS     : \(config.framerate)")
    print("Quality Check  : \(config.quality)")
    print("----------------")
        // Top-level variables captured inside Task locally to avoid isolation errors
        let localImages = images
        let localConfig = config
        let localWidth = width
        let localHeight = height
        let rawTotalSizeKB = Double(localImages.count * localWidth * localHeight * 3) / 1024.0 // Assuming YCbCr size calculation standard. H264 is YUV 4:2:0 mostly.
        
        do {
            // Warmup: dummy run for up to 5 frames to warm up CPU/code cache
            let warmupCount = min(5, localImages.count)
            let warmupImages = Array(localImages[0..<warmupCount])
            print("Warming up (\(warmupCount) frames)...")
            let _ = try await runVEVC(images: warmupImages, config: localConfig)
            let _ = try await runVEVCOne(images: warmupImages, config: localConfig)
            let _ = try await runH264(images: warmupImages, config: localConfig, width: localWidth, height: localHeight)
            let _ = try await runH264(images: warmupImages, config: localConfig, width: localWidth, height: localHeight, disableHWA: true)
            let _ = try await runHEVC(images: warmupImages, config: localConfig, width: localWidth, height: localHeight)
            let _ = try await runHEVC(images: warmupImages, config: localConfig, width: localWidth, height: localHeight, disableHWA: true)
            let _ = try await runMJPEG(images: warmupImages, config: localConfig, width: localWidth, height: localHeight)
            print("Warmup complete.\n")

            print("Running vevc (layers)...")
            let vevcResult = try await runVEVC(images: localImages, config: localConfig)
            
            print("Running vevc (One)...")
            let vevcOneResult = try await runVEVCOne(images: localImages, config: localConfig)
            
            print("Running H.264 (VideoToolbox HWA)...")
            let h264Result = try await runH264(images: localImages, config: localConfig, width: localWidth, height: localHeight)
            
            print("Running H.264 (VideoToolbox SW)...")
            let h264SwResult = try await runH264(images: localImages, config: localConfig, width: localWidth, height: localHeight, disableHWA: true)
            
            print("Running HEVC (VideoToolbox HWA)...")
            let hevcResult = try await runHEVC(images: localImages, config: localConfig, width: localWidth, height: localHeight)
            
            print("Running HEVC (VideoToolbox SW)...")
            let hevcSwResult = try await runHEVC(images: localImages, config: localConfig, width: localWidth, height: localHeight, disableHWA: true)
            
            print("Running MJPEG (VideoToolbox)...")
            let mjpegResult = try await runMJPEG(images: localImages, config: localConfig, width: localWidth, height: localHeight)
            
            func printStats(name: String, result: (encTime: Double, decTime: Double, compSize: Int, metrics: [QualityMetrics]?), count: Int, rawSizeKB: Double) -> CodecBenchmarkResult {
                let encMs = result.encTime * 1000
                let decMs = result.decTime * 1000
                let encFps = Double(count) / result.encTime
                let decFps = Double(count) / result.decTime
                let sizeKB = Double(result.compSize) / 1024.0
                
                print("[\(name)]")
                print(String(format: "  Encode : %7.2f ms (%.2f fps) - %.2f ms / frame", encMs, encFps, encMs / Double(count)))
                print(String(format: "  Decode : %7.2f ms (%.2f fps) - %.2f ms / frame", decMs, decFps, decMs / Double(count)))
                print(String(format: "  Size   : %7.2f KB (%.2f%% of raw %.2f KB)", sizeKB, (sizeKB / rawSizeKB) * 100.0, rawSizeKB))
                
                var avgPsnr: Double? = nil
                var avgSsim: Double? = nil
                if let stats = calculateQualityStats(metrics: result.metrics ?? []) {
                    avgPsnr = stats.avgPSNR
                    avgSsim = stats.avgSSIM
                    print(String(format: "  PSNR   : Avg: %5.2f | Min: %5.2f | Max: %5.2f | 50%%: %5.2f | 90%%: %5.2f | SD: %5.2f", 
                                stats.avgPSNR, stats.minPSNR, stats.maxPSNR, stats.p50PSNR, stats.p90PSNR, stats.stddevPSNR))
                    print(String(format: "  SSIM   : Avg: %5.4f | Min: %5.4f | Max: %5.4f | 50%%: %5.4f | 90%%: %5.4f | SD: %5.4f", 
                                stats.avgSSIM, stats.minSSIM, stats.maxSSIM, stats.p50SSIM, stats.p90SSIM, stats.stddevSSIM))
                }
                
                return CodecBenchmarkResult(name: name, encTimeMs: encMs / Double(count), decTimeMs: decMs / Double(count), sizeKB: sizeKB, avgPSNR: avgPsnr, avgSSIM: avgSsim)
            }
            
            print("\n--- Results ---")
            var chartResults: [CodecBenchmarkResult] = []
            chartResults.append(printStats(name: "VEVC (Layers)", result: vevcResult, count: localImages.count, rawSizeKB: rawTotalSizeKB))
            chartResults.append(printStats(name: "VEVC (One)", result: vevcOneResult, count: localImages.count, rawSizeKB: rawTotalSizeKB))
            
            chartResults.append(printStats(name: "H.264 (SW)", result: h264SwResult, count: localImages.count, rawSizeKB: rawTotalSizeKB))
            chartResults.append(printStats(name: "HEVC (SW)", result: hevcSwResult, count: localImages.count, rawSizeKB: rawTotalSizeKB))
            
            chartResults.append(printStats(name: "H.264 (HWA)", result: h264Result, count: localImages.count, rawSizeKB: rawTotalSizeKB))
            chartResults.append(printStats(name: "HEVC (HWA)", result: hevcResult, count: localImages.count, rawSizeKB: rawTotalSizeKB))
            
            chartResults.append(printStats(name: "MJPEG", result: mjpegResult, count: localImages.count, rawSizeKB: rawTotalSizeKB))
            print("---------------")
            
            if localConfig.outputGraph {
                await MainActor.run {
                    generateAndSaveCharts(results: chartResults)
                }
            }
            
        } catch {
            print("Error: \(error)")
            exit(1)
        }
        exit(0)
    }
}
