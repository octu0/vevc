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
    var outputVersus: Bool = false
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
func runVEVC(images: [ImageInput], config: Config) async throws -> (encTime: Double, decTime: Double, compSize: Int, metrics: [QualityMetrics]?, bitstream: [UInt8]) {
    let vevcImages = images.map { $0.vevcImage }
    
    // Encode
    print("  -> runVEVC Encoding...")
    let encStart = Date()
    guard let first = vevcImages.first else { return (0, 0, 0, nil, []) }
    let vevcEncoder = VEVCEncoder(
        width: first.width,
        height: first.height,
        maxbitrate: config.bitrate * 1000,
        framerate: config.framerate,
        zeroThreshold: config.zeroThreshold,
        keyint: config.keyint,
        sceneChangeThreshold: config.sceneThreshold
    )
    let outBytes = try await vevcEncoder.encodeToData(images: vevcImages)
    let encTime = Date().timeIntervalSince(encStart)
    print("  -> runVEVC Encoded \(outBytes.count) bytes")
    
    // Decode
    print("  -> runVEVC Decoding...")
    let vevcDecoder = Decoder(maxLayer: config.maxLayer)
    let decStart = Date()
    let outFrames = try await vevcDecoder.decode(data: outBytes)
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
    
    return (encTime, decTime, outBytes.count, metrics, outBytes)
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

func runH264(images: [ImageInput], config: Config, width: Int, height: Int, disableHWA: Bool = false) async throws -> (encTime: Double, decTime: Double, compSize: Int, metrics: [QualityMetrics]?, bitstream: [CMSampleBuffer]) {
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
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_AllowFrameReordering, value: kCFBooleanFalse)

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
    guard !frameBox.frames.isEmpty else { return (encTime, decTime, compSize, nil, frameBox.frames) }
    
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
        frameBox.decodedBuffers.removeAll() // Clear memory!
    }

    return (encTime, decTime, compSize, metrics, frameBox.frames)
}

// MARK: - HEVC Encode / Decode (VideoToolbox)
func runHEVC(images: [ImageInput], config: Config, width: Int, height: Int, disableHWA: Bool = false) async throws -> (encTime: Double, decTime: Double, compSize: Int, metrics: [QualityMetrics]?, bitstream: [CMSampleBuffer]) {
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
    guard !frameBox.frames.isEmpty else { return (encTime, decTime, compSize, nil, frameBox.frames) }
    
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
        frameBox.decodedBuffers.removeAll() // Clear memory!
    }

    return (encTime, decTime, compSize, metrics, frameBox.frames)
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
        frameBox.decodedBuffers.removeAll() // Clear memory!
    }

    return (encTime, decTime, compSize, metrics)
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
        case "-output-versus":
            config.outputVersus = true
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
            let _ = try await runH264(images: warmupImages, config: localConfig, width: localWidth, height: localHeight)
            let _ = try await runH264(images: warmupImages, config: localConfig, width: localWidth, height: localHeight, disableHWA: true)
            let _ = try await runHEVC(images: warmupImages, config: localConfig, width: localWidth, height: localHeight)
            let _ = try await runHEVC(images: warmupImages, config: localConfig, width: localWidth, height: localHeight, disableHWA: true)
            let _ = try await runMJPEG(images: warmupImages, config: localConfig, width: localWidth, height: localHeight)
            print("Warmup complete.\n")

            print("Running vevc (layers)...")
            let vevcResult = try await runVEVC(images: localImages, config: localConfig)
            
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
            
            func printStats(name: String, encTime: Double, decTime: Double, compSize: Int, metrics: [QualityMetrics]?, count: Int, rawSizeKB: Double) -> CodecBenchmarkResult {
                let encMs = encTime * 1000
                let decMs = decTime * 1000
                let encFps = Double(count) / encTime
                let decFps = Double(count) / decTime
                let sizeKB = Double(compSize) / 1024.0
                
                print("[\(name)]")
                print(String(format: "  Encode : %7.2f ms (%.2f fps) - %.2f ms / frame", encMs, encFps, encMs / Double(count)))
                print(String(format: "  Decode : %7.2f ms (%.2f fps) - %.2f ms / frame", decMs, decFps, decMs / Double(count)))
                print(String(format: "  Size   : %7.2f KB (%.2f%% of raw %.2f KB)", sizeKB, (sizeKB / rawSizeKB) * 100.0, rawSizeKB))
                
                var avgPsnr: Double? = nil
                var avgSsim: Double? = nil
                if let stats = calculateQualityStats(metrics: metrics ?? []) {
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
            chartResults.append(printStats(name: "VEVC (Layers)", encTime: vevcResult.encTime, decTime: vevcResult.decTime, compSize: vevcResult.compSize, metrics: vevcResult.metrics, count: localImages.count, rawSizeKB: rawTotalSizeKB))
            
            chartResults.append(printStats(name: "H.264 (SW)", encTime: h264SwResult.encTime, decTime: h264SwResult.decTime, compSize: h264SwResult.compSize, metrics: h264SwResult.metrics, count: localImages.count, rawSizeKB: rawTotalSizeKB))
            chartResults.append(printStats(name: "HEVC (SW)", encTime: hevcSwResult.encTime, decTime: hevcSwResult.decTime, compSize: hevcSwResult.compSize, metrics: hevcSwResult.metrics, count: localImages.count, rawSizeKB: rawTotalSizeKB))
            
            chartResults.append(printStats(name: "H.264 (HWA)", encTime: h264Result.encTime, decTime: h264Result.decTime, compSize: h264Result.compSize, metrics: h264Result.metrics, count: localImages.count, rawSizeKB: rawTotalSizeKB))
            chartResults.append(printStats(name: "HEVC (HWA)", encTime: hevcResult.encTime, decTime: hevcResult.decTime, compSize: hevcResult.compSize, metrics: hevcResult.metrics, count: localImages.count, rawSizeKB: rawTotalSizeKB))
            
            chartResults.append(printStats(name: "MJPEG", encTime: mjpegResult.encTime, decTime: mjpegResult.decTime, compSize: mjpegResult.compSize, metrics: mjpegResult.metrics, count: localImages.count, rawSizeKB: rawTotalSizeKB))
            print("---------------")
            
            if localConfig.outputGraph {
                await MainActor.run {
                    generateAndSaveCharts(results: chartResults)
                }
            }
            
            if localConfig.outputVersus {
                print("\n--- Output Versus Images ---")
                
                let vevcMinIdx = vevcResult.metrics?.enumerated().min(by: { $0.element.ssim < $1.element.ssim })?.offset ?? 0
                let h264MinIdx = h264SwResult.metrics?.enumerated().min(by: { $0.element.ssim < $1.element.ssim })?.offset ?? 0
                let hevcMinIdx = hevcSwResult.metrics?.enumerated().min(by: { $0.element.ssim < $1.element.ssim })?.offset ?? 0
                let sec14Idx = min(14 * localConfig.framerate, localImages.count - 1)
                
                let targetIndices: Set<Int> = [vevcMinIdx, h264MinIdx, hevcMinIdx, sec14Idx]
                print("Target Indices: VEVC Min SSIM (\(vevcMinIdx)), H264 Min SSIM (\(h264MinIdx)), HEVC Min SSIM (\(hevcMinIdx)), 14s (\(sec14Idx))")
                
                print("Extracting VEVC frames...")
                let vevcExtracted = try await extractVEVCFrames(bitstream: vevcResult.bitstream, config: localConfig, indices: targetIndices)
                
                print("Extracting H.264 frames...")
                let h264Extracted = try extractVTFrames(bitstream: h264SwResult.bitstream, disableHWA: false, indices: targetIndices)
                
                print("Extracting HEVC frames...")
                let hevcExtracted = try extractVTFrames(bitstream: hevcSwResult.bitstream, disableHWA: false, indices: targetIndices)
                
                let pairs: [(name: String, idx: Int)] = [
                    ("vevc_min", vevcMinIdx),
                    ("h264_min", h264MinIdx),
                    ("hevc_min", hevcMinIdx),
                    ("14s", sec14Idx)
                ]
                
                for p in pairs {
                    let origFrame = localImages[p.idx]
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

// MARK: - Frame Extraction Helpers

func extractVTFrames(bitstream: [CMSampleBuffer], disableHWA: Bool, indices: Set<Int>) throws -> [Int: YCbCrImage] {
    guard !bitstream.isEmpty else { return [:] }
    guard let formatDesc = CMSampleBufferGetFormatDescription(bitstream[0]) else {
        throw NSError(domain: "CMSampleBufferGetFormatDescription (Extract)", code: -1, userInfo: nil)
    }
    
    class ExtractBox: @unchecked Sendable {
        var extracted: [Int: YCbCrImage] = [:]
        let targetIndices: Set<Int>
        let lock = NSLock()
        init(indices: Set<Int>) { self.targetIndices = indices }
    }
    let extractBox = ExtractBox(indices: indices)
    
    let destPixelBufferAttributes: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        kCVPixelBufferMetalCompatibilityKey as String: true
    ]
    
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
        decompressionSessionOut: &decompressionSessionOut
    )
    
    guard decStatus == noErr, let session = decompressionSessionOut else {
        throw NSError(domain: "VTDecompressionSessionCreate (Extract)", code: Int(decStatus), userInfo: nil)
    }
    
    for sample in bitstream {
        var flags: VTDecodeInfoFlags = []
        VTDecompressionSessionDecodeFrame(
            session,
            sampleBuffer: sample,
            flags: [],
            infoFlagsOut: &flags,
            outputHandler: { (status, infoFlags, imageBuffer, presentationTimeStamp, _) in
                if let buf = imageBuffer {
                    let idx = Int(presentationTimeStamp.value)
                    if extractBox.targetIndices.contains(idx) {
                        let w = CVPixelBufferGetWidth(buf)
                        let h = CVPixelBufferGetHeight(buf)
                        let ycbcr = createYCbCrImage(from: buf, width: w, height: h)
                        extractBox.lock.lock()
                        extractBox.extracted[idx] = ycbcr
                        extractBox.lock.unlock()
                    }
                }
            }
        )
    }
    VTDecompressionSessionWaitForAsynchronousFrames(session)
    return extractBox.extracted
}

func extractVEVCFrames(bitstream: [UInt8], config: Config, indices: Set<Int>) async throws -> [Int: YCbCrImage] {
    let vevcDecoder = Decoder(maxLayer: config.maxLayer)
    let outFrames = try await vevcDecoder.decode(data: bitstream)
    var extracted: [Int: YCbCrImage] = [:]
    for i in 0..<outFrames.count {
        if indices.contains(i) {
            extracted[i] = outFrames[i]
        }
    }
    return extracted
}

func saveVersusImage(idx: Int, orig: ImageInput, vevcF: YCbCrImage?, h264F: YCbCrImage?, hevcF: YCbCrImage?, prefix: String) {
    let w = orig.width
    let h = orig.height
    
    // We will crop 400x400 from the center
    let cropW = 400
    let cropH = 400
    let cx = max(0, w / 2 - cropW / 2)
    let cy = max(0, h / 2 - cropH / 2)
    
    func doCrop(rgba: [PNG.RGBA<UInt8>], width: Int, height: Int) -> [PNG.RGBA<UInt8>] {
        var out = [PNG.RGBA<UInt8>](repeating: .init(0,0,0,255), count: cropW * cropH)
        for y in 0..<cropH {
            let sy = cy + y
            if sy < 0 || sy >= height { continue }
            for x in 0..<cropW {
                let sx = cx + x
                if sx < 0 || sx >= width { continue }
                out[y * cropW + x] = rgba[sy * width + sx]
                out[y * cropW + x].a = 255 // Force opaque
            }
        }
        return out
    }
    
    // Convert to RGBA and crop
    let origRGBA = orig.rgbaFrames
    
    // Helper to convert UInt8 array to PNG.RGBA
    func toPNGRGBA(_ data: [UInt8]) -> [PNG.RGBA<UInt8>] {
        let count = data.count / 4
        var arr = [PNG.RGBA<UInt8>](repeating: .init(0,0,0,255), count: count)
        data.withUnsafeBufferPointer { src in
            arr.withUnsafeMutableBufferPointer { dst in
                guard let s = src.baseAddress, let d = dst.baseAddress else { return }
                for i in 0..<count {
                    let off = i * 4
                    d[i] = PNG.RGBA<UInt8>(s[off], s[off+1], s[off+2], 255)
                }
            }
        }
        return arr
    }
    
    let vevcRGBA = vevcF != nil ? toPNGRGBA(vevc.ycbcrToRGBA(img: vevcF!)) : nil
    let h264RGBA = h264F != nil ? toPNGRGBA(vevc.ycbcrToRGBA(img: h264F!)) : nil
    let hevcRGBA = hevcF != nil ? toPNGRGBA(vevc.ycbcrToRGBA(img: hevcF!)) : nil
    
    let crops: [(name: String, data: [PNG.RGBA<UInt8>]?)] = [
        ("orig", doCrop(rgba: origRGBA, width: w, height: h)),
        ("vevc", vevcRGBA != nil ? doCrop(rgba: vevcRGBA!, width: w, height: h) : nil),
        ("h264", h264RGBA != nil ? doCrop(rgba: h264RGBA!, width: w, height: h) : nil),
        ("hevc", hevcRGBA != nil ? doCrop(rgba: hevcRGBA!, width: w, height: h) : nil),
    ]
    
    for c in crops {
        guard let data = c.data else { continue }
        let filename = "docs/versus_\(prefix)_frame\(idx)_\(c.name).png"
        let image = PNG.Image(
            packing: data,
            size: (x: cropW, y: cropH),
            layout: .init(format: .rgba8(palette: [], fill: nil))
        )
        if let _ = try? image.compress(path: filename, level: 6) {
            print("  -> Saved \(filename)")
        } else {
            print("  -> Failed to save \(filename)")
        }
    }
}
