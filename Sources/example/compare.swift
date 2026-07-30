import Foundation
import VideoToolbox
import CoreMedia
import AppKit
import CryptoKit
import vevc

struct Config {
    var bitrate: Int = 500
    var framerate: Int = 60
    var zeroThreshold: Int = 5
    var keyint: Int = 30
    var sceneThreshold: Int = 32
    var maxLayer: Int = 2
    var quality: Bool = false
    var outputGraph: Bool = false
    var outputVersus: Bool = false
    var outputBitrates: Bool = false
    var vevcOnly: Bool = false
    var dumpHash: Bool = false
    var qstep: Int? = nil
    var profile: UInt8 = 0x01
    var maxFrames: Int? = nil
}

struct ImageInput {
    let vevcImage: YCbCrImage
    let width: Int
    let height: Int
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
        
        inputs.append(ImageInput(vevcImage: img, width: width, height: height))
    }
    return inputs
}

func parseVEVCLayerSizes(bitstream: [UInt8], profile: UInt8, width: Int, height: Int) -> (l0: Int, l1: Int, l2: Int, skips: (prev: Int, ltr: Int, inter: Int, copy: Int, total: Int)) {
    var offset = 0
    var headerAndBase = 0
    var l0 = 0
    var l1 = 0
    var l2 = 0
    
    var skipPrev = 0
    var skipLtr = 0
    var inter = 0
    var copyPrev = 0
    var totalBlocks = 0
    
    let bw = (width + 31) / 32
    let bh = (height + 31) / 32
    let blockCount = bw * bh
    
    while offset < bitstream.count {
        if offset + 4 <= bitstream.count && bitstream[offset] == 0x56 && bitstream[offset+1] == 0x45 && bitstream[offset+2] == 0x56 && bitstream[offset+3] == 0x43 {
            let start = offset
            if let _ = try? VEVCFileHeader.deserialize(from: bitstream, offset: &offset) {
                headerAndBase += (offset - start)
            } else {
                break
            }
        } else {
            let start = offset
            if let fh = try? VEVCFrameHeader.deserialize(from: bitstream, offset: &offset, profile: profile) {
                let headerSize = offset - start
                headerAndBase += headerSize
                if !fh.isCopyFrame {
                    if profile == 0x02 && fh.skipMapSize > 0 {
                        let smData = Array(bitstream[offset..<(offset + fh.skipMapSize)])
                        if let map = try? decodeSkipMap(data: smData, count: blockCount) {
                            for m in map {
                                if m == .skip_prev { skipPrev += 1 }
                                else if m == .skip_ltr { skipLtr += 1 }
                                else { inter += 1 }
                                totalBlocks += 1
                            }
                        }
                    } else if profile == 0x01 && !fh.isIFrame {
                        inter += blockCount
                        totalBlocks += blockCount
                    } else if fh.isIFrame {
                        inter += blockCount
                        totalBlocks += blockCount
                    }
                    
                    headerAndBase += fh.skipMapSize + fh.mvsSize + fh.refDirSize
                    l0 += fh.layer0Size
                    l1 += fh.layer1Size
                    l2 += fh.layer2Size
                    offset += fh.payloadSize
                } else {
                    copyPrev += blockCount
                    totalBlocks += blockCount
                }
            } else {
                break
            }
        }
    }
    
    let layer0Total = headerAndBase + l0
    let layer1Total = layer0Total + l1
    let layer2Total = layer1Total + l2
    
    return (layer0Total, layer1Total, layer2Total, (skipPrev, skipLtr, inter, copyPrev, totalBlocks))
}

// MARK: - VEVC Encode / Decode
func runVEVC(images: [ImageInput], config: Config) async throws -> (
    encTime: Double,
    bitstream: [UInt8],
    sizes: (l0: Int, l1: Int, l2: Int),
    skips: (prev: Int, ltr: Int, inter: Int, copy: Int, total: Int),
    l0Dec: (time: Double, metrics: [QualityMetrics]?),
    l1Dec: (time: Double, metrics: [QualityMetrics]?),
    l2Dec: (time: Double, metrics: [QualityMetrics]?)
) {
    let vevcImages = images.map { $0.vevcImage }
    
    // Encode
    print("  -> runVEVC Encoding...")
    let encStart = Date()
    guard let first = vevcImages.first else { return (0, [], (0,0,0), (0, 0, 0, 0, 0), (0, nil), (0, nil), (0, nil)) }
    let vevcEncoder: VEVCEncoder
    if let qstep = config.qstep {
        vevcEncoder = VEVCEncoder(
            width: first.width,
            height: first.height,
            qstep: qstep,
            framerate: config.framerate,
            zeroThreshold: config.zeroThreshold,
            keyint: config.keyint,
            sceneChangeThreshold: config.sceneThreshold,
            profile: config.profile
        )
    } else {
        vevcEncoder = VEVCEncoder(
            width: first.width,
            height: first.height,
            maxbitrate: config.bitrate * 1000,
            framerate: config.framerate,
            zeroThreshold: config.zeroThreshold,
            keyint: config.keyint,
            sceneChangeThreshold: config.sceneThreshold,
            profile: config.profile
        )
    }
    let outBytes = try await vevcEncoder.encodeToData(images: vevcImages)
    let encTime = Date().timeIntervalSince(encStart)
    print("  -> runVEVC Encoded \(outBytes.count) bytes")
    
    let res = parseVEVCLayerSizes(bitstream: outBytes, profile: config.profile, width: first.width, height: first.height)
    let sizes = (l0: res.l0, l1: res.l1, l2: res.l2)
    let skips = res.skips
    
    func decodeAndMetrics(maxLayer: Int) async throws -> (time: Double, metrics: [QualityMetrics]?) {
        print("  -> runVEVC Decoding Layer \(maxLayer)...")
        let vevcDecoder = Decoder(maxLayer: maxLayer)
        let decStart = Date()
        let outFrames = try await vevcDecoder.decode(data: outBytes)
        let decTime = Date().timeIntervalSince(decStart)
        print("  -> runVEVC Decoded Layer \(maxLayer) \(outFrames.count) frames")
        
        if config.dumpHash && maxLayer == 2 {
            let encHash = SHA256.hash(data: outBytes).map { String(format: "%02x", $0) }.joined()
            var hasher = SHA256()
            for frame in outFrames {
                frame.yPlane.withUnsafeBufferPointer { hasher.update(bufferPointer: UnsafeRawBufferPointer($0)) }
                frame.cbPlane.withUnsafeBufferPointer { hasher.update(bufferPointer: UnsafeRawBufferPointer($0)) }
                frame.crPlane.withUnsafeBufferPointer { hasher.update(bufferPointer: UnsafeRawBufferPointer($0)) }
            }
            let decHash = hasher.finalize().map { String(format: "%02x", $0) }.joined()
            print("[FIXTURE_HASH] Encoded SHA-256: \(encHash)")
            print("[FIXTURE_HASH] Decoded SHA-256: \(decHash)")
        }
        
        var metrics: [QualityMetrics]? = nil
        if config.quality && maxLayer == 2 {
            var mets = [QualityMetrics]()
            for i in 0..<min(images.count, outFrames.count) {
                let psnr = calculatePSNR(img1: images[i].vevcImage, img2: outFrames[i])
                let ssim = calculateSSIM(img1: images[i].vevcImage, img2: outFrames[i])
                mets.append(QualityMetrics(psnr: psnr, ssim: ssim))
            }
            metrics = mets
        }
        return (decTime, metrics)
    }
    
    let l2Dec = try await decodeAndMetrics(maxLayer: 2)
    let l1Dec = try await decodeAndMetrics(maxLayer: 1)
    let l0Dec = try await decodeAndMetrics(maxLayer: 0)
    
    return (encTime, outBytes, sizes, skips, l0Dec, l1Dec, l2Dec)
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
    var encoderSpec: CFDictionary? = nil
    if disableHWA {
        encoderSpec = [
            kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder: false,
            kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder: false
        ] as CFDictionary
    }
    

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
    

    for sample in frameBox.frames {
        if let dataBuffer = CMSampleBufferGetDataBuffer(sample) {
            compSize += CMBlockBufferGetDataLength(dataBuffer)
        }
    }
    

    var decTime: Double = 0
    guard frameBox.frames.isEmpty != true else { return (encTime, decTime, compSize, nil, frameBox.frames) }
    
    // Need format desc for decompression
    guard let formatDesc = CMSampleBufferGetFormatDescription(frameBox.frames[0]) else {
        throw NSError(domain: "CMSampleBufferGetFormatDescription", code: -1, userInfo: nil)
    }
    
    let destPixelBufferAttributes: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        kCVPixelBufferMetalCompatibilityKey as String: true
    ]
    
    // Decoder spec to disable HWA
    var decoderSpec: CFDictionary? = nil
    if disableHWA {
        decoderSpec = [
            kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder: false,
            kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder: false
        ] as CFDictionary
    }
    
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
        frameBox.decodedBuffers.removeAll()
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
    var encoderSpec: CFDictionary? = nil
    if disableHWA {
        encoderSpec = [
            kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder: false,
            kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder: false
        ] as CFDictionary
    }
    

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
    
    for sample in frameBox.frames {
        if let dataBuffer = CMSampleBufferGetDataBuffer(sample) {
            compSize += CMBlockBufferGetDataLength(dataBuffer)
        }
    }

    var decTime: Double = 0
    guard frameBox.frames.isEmpty != true else { return (encTime, decTime, compSize, nil, frameBox.frames) }
    
    guard let formatDesc = CMSampleBufferGetFormatDescription(frameBox.frames[0]) else {
        throw NSError(domain: "CMSampleBufferGetFormatDescription (HEVC)", code: -1, userInfo: nil)
    }
    
    let destPixelBufferAttributes: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        kCVPixelBufferMetalCompatibilityKey as String: true
    ]
    
    // Decoder spec to disable HWA
    var decoderSpec: CFDictionary? = nil
    if disableHWA {
        decoderSpec = [
            kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder: false,
            kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder: false
        ] as CFDictionary
    }
    
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
        frameBox.decodedBuffers.removeAll()
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
    

    for sample in frameBox.frames {
        if let dataBuffer = CMSampleBufferGetDataBuffer(sample) {
            compSize += CMBlockBufferGetDataLength(dataBuffer)
        }
    }
    
    var decTime: Double = 0
    guard frameBox.frames.isEmpty != true else { return (encTime, decTime, compSize, nil) }
    
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
        frameBox.decodedBuffers.removeAll()
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
        case "-profile":
            if (i + 1) < args.count {
                if let v = UInt8(args[i + 1]) { config.profile = v }
                i += 1
            }
        case "-frames", "--frames":
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
        print("Usage: compare [-y4m <input.y4m>] [-bitrate <kbits>] [-qstep <val>] [-framerate <fps>] [-zeroThreshold <threshold>] [-keyint <frames>] [-sceneThreshold <sad>] [-maxLayer <0-2>] [-profile <0x01|0x02>] [-quality] [-output-graph] [-vevc-only]")
        exit(1)
    }

    var images: [ImageInput] = []
    if let y4m = y4mPath {
        if let y4mImages = readY4M(path: y4m) {
            images = y4mImages
        } else {
            print("Failed to read y4m: \(y4m)")
        }
    }

    if let maxF = config.maxFrames, maxF < images.count {
        images = Array(images[0..<maxF])
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
    print("Profile        : \(String(format: "0x%02X", config.profile))")
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
            _ = try await runVEVC(images: warmupImages, config: localConfig)
            if localConfig.vevcOnly != true {
                _ = try await runH264(images: warmupImages, config: localConfig, width: localWidth, height: localHeight)
                _ = try await runH264(images: warmupImages, config: localConfig, width: localWidth, height: localHeight, disableHWA: true)
                _ = try await runHEVC(images: warmupImages, config: localConfig, width: localWidth, height: localHeight)
                _ = try await runHEVC(images: warmupImages, config: localConfig, width: localWidth, height: localHeight, disableHWA: true)
                _ = try await runMJPEG(images: warmupImages, config: localConfig, width: localWidth, height: localHeight)
            }
            print("Warmup complete.\n")

            print("Running vevc (layers)...")
            let vevcResult = try await runVEVC(images: localImages, config: localConfig)
            
            typealias CodecResult = (encTime: Double, decTime: Double, compSize: Int, metrics: [QualityMetrics]?, bitstream: [CMSampleBuffer])
            var h264Result: CodecResult? = nil
            var h264SwResult: CodecResult? = nil
            var hevcResult: CodecResult? = nil
            var hevcSwResult: CodecResult? = nil
            var mjpegResult: (encTime: Double, decTime: Double, compSize: Int, metrics: [QualityMetrics]?)? = nil
            
            if localConfig.vevcOnly != true {
                print("Running H.264 (VideoToolbox HWA)...")
                h264Result = try await runH264(images: localImages, config: localConfig, width: localWidth, height: localHeight)
                
                print("Running H.264 (VideoToolbox SW)...")
                h264SwResult = try await runH264(images: localImages, config: localConfig, width: localWidth, height: localHeight, disableHWA: true)
                
                print("Running HEVC (VideoToolbox HWA)...")
                hevcResult = try await runHEVC(images: localImages, config: localConfig, width: localWidth, height: localHeight)
                
                print("Running HEVC (VideoToolbox SW)...")
                hevcSwResult = try await runHEVC(images: localImages, config: localConfig, width: localWidth, height: localHeight, disableHWA: true)
                
                print("Running MJPEG (VideoToolbox)...")
                mjpegResult = try await runMJPEG(images: localImages, config: localConfig, width: localWidth, height: localHeight)
            }
            
            func printStats(name: String, encTime: Double, decTime: Double, compSize: Int, metrics: [QualityMetrics]?, count: Int, rawSizeKB: Double, skips: (prev: Int, ltr: Int, inter: Int, copy: Int, total: Int)? = nil) -> CodecBenchmarkResult {
                let encMs = encTime * 1000
                let decMs = decTime * 1000
                let encFps = Double(count) / encTime
                let decFps = Double(count) / decTime
                let sizeKB = Double(compSize) / 1024.0
                
                print("[\(name)]")
                print(String(format: "  Encode : %7.2f ms (%.2f fps) - %.2f ms / frame", encMs, encFps, encMs / Double(count)))
                print(String(format: "  Decode : %7.2f ms (%.2f fps) - %.2f ms / frame", decMs, decFps, decMs / Double(count)))
                print(String(format: "  Size   : %7.2f KB (%.2f%% of raw %.2f KB)", sizeKB, (sizeKB / rawSizeKB) * 100.0, rawSizeKB))
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
                }
                
                return CodecBenchmarkResult(name: name, encTimeMs: encMs / Double(count), decTimeMs: decMs / Double(count), sizeKB: sizeKB, stats: statsOut)
            }
            
            print("\n--- Results ---")
            var chartResults: [CodecBenchmarkResult] = []
            chartResults.append(printStats(name: "VEVC (Layer 2)", encTime: vevcResult.encTime, decTime: vevcResult.l2Dec.time, compSize: vevcResult.sizes.l2, metrics: vevcResult.l2Dec.metrics, count: localImages.count, rawSizeKB: rawTotalSizeKB, skips: vevcResult.skips))
            chartResults.append(printStats(name: "VEVC (Layer 1)", encTime: vevcResult.encTime, decTime: vevcResult.l1Dec.time, compSize: vevcResult.sizes.l1, metrics: vevcResult.l1Dec.metrics, count: localImages.count, rawSizeKB: rawTotalSizeKB, skips: vevcResult.skips))
            chartResults.append(printStats(name: "VEVC (Layer 0)", encTime: vevcResult.encTime, decTime: vevcResult.l0Dec.time, compSize: vevcResult.sizes.l0, metrics: vevcResult.l0Dec.metrics, count: localImages.count, rawSizeKB: rawTotalSizeKB, skips: vevcResult.skips))
            
            if localConfig.vevcOnly != true {
                if let h264Sw = h264SwResult {
                    chartResults.append(printStats(name: "H.264 (SW)", encTime: h264Sw.encTime, decTime: h264Sw.decTime, compSize: h264Sw.compSize, metrics: h264Sw.metrics, count: localImages.count, rawSizeKB: rawTotalSizeKB))
                }
                if let hevcSw = hevcSwResult {
                    chartResults.append(printStats(name: "HEVC (SW)", encTime: hevcSw.encTime, decTime: hevcSw.decTime, compSize: hevcSw.compSize, metrics: hevcSw.metrics, count: localImages.count, rawSizeKB: rawTotalSizeKB))
                }
                if let h264 = h264Result {
                    chartResults.append(printStats(name: "H.264 (HWA)", encTime: h264.encTime, decTime: h264.decTime, compSize: h264.compSize, metrics: h264.metrics, count: localImages.count, rawSizeKB: rawTotalSizeKB))
                }
                if let hevc = hevcResult {
                    chartResults.append(printStats(name: "HEVC (HWA)", encTime: hevc.encTime, decTime: hevc.decTime, compSize: hevc.compSize, metrics: hevc.metrics, count: localImages.count, rawSizeKB: rawTotalSizeKB))
                }
                if let mjpeg = mjpegResult {
                    chartResults.append(printStats(name: "MJPEG", encTime: mjpeg.encTime, decTime: mjpeg.decTime, compSize: mjpeg.compSize, metrics: mjpeg.metrics, count: localImages.count, rawSizeKB: rawTotalSizeKB))
                }
            }
            print("---------------")
            
            if localConfig.outputGraph {
                await MainActor.run {
                    generateAndSaveCharts(results: chartResults)
                }
            }
            
            if localConfig.outputBitrates {
                print("\n--- Running Bitrate Sweep (300 - 5000) ---")
                var chartPoints: [BitrateSsimPoint] = []
                let bitrates = [300, 500, 800, 1000, 1200, 1500, 1800, 2500, 3000]
                
                for br in bitrates {
                    var sweepConfig = localConfig
                    sweepConfig.bitrate = br
                    print(">> Bitrate: \(br) kbps")
                    
                    let vevcRes = try await runVEVC(images: localImages, config: sweepConfig)
                    if let stats = calculateQualityStats(metrics: vevcRes.l2Dec.metrics ?? []) {
                        chartPoints.append(.init(codec: "VEVC (Layer 2)", bitrate: br, ssimAvg: stats.avgSSIM, ssimMin: stats.minSSIM, ssimMax: stats.maxSSIM, sizeKB: Double(vevcRes.sizes.l2) / 1024.0))
                    }
                    chartPoints.append(.init(codec: "VEVC (Layer 1)", bitrate: br, ssimAvg: 0, ssimMin: 0, ssimMax: 0, sizeKB: Double(vevcRes.sizes.l1) / 1024.0))
                    chartPoints.append(.init(codec: "VEVC (Layer 0)", bitrate: br, ssimAvg: 0, ssimMin: 0, ssimMax: 0, sizeKB: Double(vevcRes.sizes.l0) / 1024.0))
                    
                    if localConfig.vevcOnly != true {
                        let h264SwRes = try await runH264(images: localImages, config: sweepConfig, width: localWidth, height: localHeight, disableHWA: true)
                        if let stats = calculateQualityStats(metrics: h264SwRes.metrics ?? []) {
                            chartPoints.append(.init(codec: "H.264 (SW)", bitrate: br, ssimAvg: stats.avgSSIM, ssimMin: stats.minSSIM, ssimMax: stats.maxSSIM, sizeKB: Double(h264SwRes.compSize) / 1024.0))
                        }
                        
                        let hevcSwRes = try await runHEVC(images: localImages, config: sweepConfig, width: localWidth, height: localHeight, disableHWA: true)
                        if let stats = calculateQualityStats(metrics: hevcSwRes.metrics ?? []) {
                            chartPoints.append(.init(codec: "HEVC (SW)", bitrate: br, ssimAvg: stats.avgSSIM, ssimMin: stats.minSSIM, ssimMax: stats.maxSSIM, sizeKB: Double(hevcSwRes.compSize) / 1024.0))
                        }
                    }
                }
                
                await MainActor.run {
                    generateAndSaveBitrateCharts(points: chartPoints)
                }
            }
            
            if localConfig.outputVersus {
                print("\n--- Output Versus Images ---")
                
                let vevcMinIdx = vevcResult.l2Dec.metrics?.enumerated().min(by: { $0.element.ssim < $1.element.ssim })?.offset ?? 0
                let h264MinIdx = h264SwResult?.metrics?.enumerated().min(by: { $0.element.ssim < $1.element.ssim })?.offset ?? 0
                let hevcMinIdx = hevcSwResult?.metrics?.enumerated().min(by: { $0.element.ssim < $1.element.ssim })?.offset ?? 0
                let sec14Idx = min(14 * localConfig.framerate, localImages.count - 1)
                
                let targetIndices: Set<Int> = [vevcMinIdx, h264MinIdx, hevcMinIdx, sec14Idx]
                print("Target Indices: VEVC Min SSIM (\(vevcMinIdx)), H264 Min SSIM (\(h264MinIdx)), HEVC Min SSIM (\(hevcMinIdx)), 14s (\(sec14Idx))")
                
                print("Extracting VEVC frames...")
                let vevcExtracted = try await extractVEVCFrames(bitstream: vevcResult.bitstream, config: localConfig, indices: targetIndices)
                
                var h264Extracted: [Int: YCbCrImage] = [:]
                if let res = h264SwResult {
                    print("Extracting H.264 frames...")
                    h264Extracted = try extractVTFrames(bitstream: res.bitstream, disableHWA: false, indices: targetIndices)
                }
                
                var hevcExtracted: [Int: YCbCrImage] = [:]
                if let res = hevcSwResult {
                    print("Extracting HEVC frames...")
                    hevcExtracted = try extractVTFrames(bitstream: res.bitstream, disableHWA: false, indices: targetIndices)
                }
                
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
    guard bitstream.isEmpty != true else { return [:] }
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
    
    var decoderSpec: CFDictionary? = nil
    if disableHWA {
        decoderSpec = [
            kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder: false,
            kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder: false
        ] as CFDictionary
    }
    
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
    
    // Crop center 400x400
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
        
        // Force opaque alpha
        for i in stride(from: 3, to: rgba.count, by: 4) {
            rgba[i] = 255
        }
        
        rgba.withUnsafeMutableBytes { ptr in
            guard let baseAddress = ptr.baseAddress else { return }
            guard let context = CGContext(
                data: baseAddress,
                width: f.width,
                height: f.height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
            ) else { return }
            
            guard let cgImage = context.makeImage() else { return }
            guard let croppedCGImage = cgImage.cropping(to: cropRect) else { return }
            
            let rep = NSBitmapImageRep(cgImage: croppedCGImage)
            guard let pngData = rep.representation(using: .png, properties: [:]) else { return }
            
            let url = URL(fileURLWithPath: outPath)
            do {
                try pngData.write(to: url)
                print("  -> Saved \(outPath)")
            } catch {
                print("  -> Failed to save \(outPath)")
            }
        }
    }
    
    createCroppedPNG(from: orig.vevcImage, outPath: "docs/versus_\(prefix)_frame\(idx)_orig.png")
    createCroppedPNG(from: vevcF, outPath: "docs/versus_\(prefix)_frame\(idx)_vevc.png")
    createCroppedPNG(from: h264F, outPath: "docs/versus_\(prefix)_frame\(idx)_h264.png")
    createCroppedPNG(from: hevcF, outPath: "docs/versus_\(prefix)_frame\(idx)_hevc.png")
}
