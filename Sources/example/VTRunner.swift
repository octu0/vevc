import Foundation
import VideoToolbox
import CoreMedia
import CryptoKit
import vevc

public struct VTFrameData: @unchecked Sendable {
    public let pts: CMTime
    public let formatDesc: CMVideoFormatDescription?
    public let data: [UInt8]
}

class QualityBox: @unchecked Sendable {
    var metrics: [Int: QualityMetrics] = [:]
    var y4mFrames: [Int: YCbCrImage] = [:]
    var currentY4MIdx = 0
    let qY4M: Y4MIterator
    let lock = NSLock()
    
    init(qY4M: Y4MIterator) {
        self.qY4M = qY4M
    }
    
    func getOrigFrame(idx: Int) throws -> YCbCrImage? {
        lock.lock()
        defer { lock.unlock() }
        
        while currentY4MIdx <= idx {
            if let img = try qY4M.next() {
                y4mFrames[currentY4MIdx] = img.vevcImage
                currentY4MIdx += 1
            } else {
                break
            }
        }
        return y4mFrames[idx]
    }
    
    func removeOrigFrame(idx: Int) {
        lock.lock()
        y4mFrames.removeValue(forKey: idx)
        lock.unlock()
    }
}

func runH264(y4mPath: String, config: Config, width: Int, height: Int) async throws -> (encTime: Double, decTime: Double, compSize: Int, metrics: [QualityMetrics]?, bitstream: [VTFrameData]) {
    return try await runH264(y4mPath: y4mPath, config: config, width: width, height: height, disableHWA: false)
}

func runH264(y4mPath: String, config: Config, width: Int, height: Int, disableHWA: Bool) async throws -> (encTime: Double, decTime: Double, compSize: Int, metrics: [QualityMetrics]?, bitstream: [VTFrameData]) {
    var encTime: Double = 0
    var compSize: Int = 0

    class FrameBox: @unchecked Sendable {
        var frames: [VTFrameData] = []
        let lock = NSLock()
        let dumpHandle: FileHandle?
        init(dumpEnv: String) {
            if let p = ProcessInfo.processInfo.environment[dumpEnv] {
                FileManager.default.createFile(atPath: p, contents: nil)
                dumpHandle = FileHandle(forWritingAtPath: p)
            } else { dumpHandle = nil }
        }
    }
    let frameBox = FrameBox(dumpEnv: "VEVC_DUMP_H264")
    
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
            
            guard let dataBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { return }
            let length = CMBlockBufferGetDataLength(dataBuffer)
            var data = [UInt8](repeating: 0, count: length)
            CMBlockBufferCopyDataBytes(dataBuffer, atOffset: 0, dataLength: length, destination: &data)
            let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            let formatDesc = CMSampleBufferGetFormatDescription(sampleBuffer)
            
            box.lock.lock()
            box.frames.append(VTFrameData(pts: pts, formatDesc: formatDesc, data: data))
            box.dumpHandle?.write(Data(data))
            box.lock.unlock()
        },
        refcon: Unmanaged.passUnretained(frameBox).toOpaque(),
        compressionSessionOut: &compressionSessionOut)

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
    
    let encY4M = try Y4MIterator(path: y4mPath, config: config)
    let encStart = Date()
    var idx = 0
    while let imgInput = try encY4M.next() {
        autoreleasepool {
            if let pixelBuffer = createPixelBuffer(from: imgInput.vevcImage) {
                let presentationTimeStamp = CMTime(value: CMTimeValue(idx), timescale: CMTimeScale(config.framerate))
                var flags: VTEncodeInfoFlags = []
                VTCompressionSessionEncodeFrame(
                    compressionSession,
                    imageBuffer: pixelBuffer,
                    presentationTimeStamp: presentationTimeStamp,
                    duration: .invalid,
                    frameProperties: nil,
                    sourceFrameRefcon: nil,
                    infoFlagsOut: &flags
                )
                idx += 1
            }
        }
    }
    VTCompressionSessionCompleteFrames(compressionSession, untilPresentationTimeStamp: .invalid)
    encTime = Date().timeIntervalSince(encStart)
    
    for sample in frameBox.frames {
        compSize += sample.data.count
    }
    
    var decTime: Double = 0
    guard !frameBox.frames.isEmpty else { return (encTime, decTime, compSize, nil, frameBox.frames) }
    
    guard let formatDesc = frameBox.frames[0].formatDesc else {
        throw NSError(domain: "CMSampleBufferGetFormatDescription", code: -1, userInfo: nil)
    }
    
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
    guard decStatus == noErr, let decompressionSession = decompressionSessionOut else {
        throw NSError(domain: "VTDecompressionSessionCreate", code: Int(decStatus), userInfo: nil)
    }
    
    let decStart = Date()
    for sample in frameBox.frames {
        guard let sb = recreateCMSampleBuffer(from: sample) else { continue }
        var flags: VTDecodeInfoFlags = []
        VTDecompressionSessionDecodeFrame(
            decompressionSession,
            sampleBuffer: sb,
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
        
        let qY4M = try Y4MIterator(path: y4mPath, config: config)
        let qBox = QualityBox(qY4M: qY4M)
        
        struct SendableSession: @unchecked Sendable {
            let session: VTDecompressionSession
        }
        let safeQualitySession = SendableSession(session: qualitySession)
        
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            DispatchQueue.global().async {
                let sem = DispatchSemaphore(value: 300)
                let qualitySession = safeQualitySession.session
                for sample in frameBox.frames {
                    guard let sb = recreateCMSampleBuffer(from: sample) else { continue }
                    sem.wait()
                    var flags: VTDecodeInfoFlags = []
                    VTDecompressionSessionDecodeFrame(
                        qualitySession,
                        sampleBuffer: sb,
                        flags: [],
                        infoFlagsOut: &flags,
                        outputHandler: { (status, infoFlags, imageBuffer, presentationTimeStamp, presentationDuration) in
                            if let buf = imageBuffer {
                                let idx = Int(presentationTimeStamp.value)
                                if let orig = try? qBox.getOrigFrame(idx: idx) {
                                    let psnr = calculatePSNR(img1: orig, bgraBuffer: buf)
                                    let ssim = calculateSSIM(img1: orig, bgraBuffer: buf)
                                    qBox.lock.lock()
                                    qBox.metrics[idx] = QualityMetrics(psnr: psnr, ssim: ssim)
                                    qBox.lock.unlock()
                                    qBox.removeOrigFrame(idx: idx)
                                }
                            }
                            sem.signal()
                        }
                    )
                }
                VTDecompressionSessionWaitForAsynchronousFrames(qualitySession)
                continuation.resume()
            }
        }
        
        var mets = [QualityMetrics]()
        for i in 0..<idx {
            if let m = qBox.metrics[i] { mets.append(m) }
        }
        metrics = mets
    }
    
    return (encTime, decTime, compSize, metrics, frameBox.frames)
}

func runHEVC(y4mPath: String, config: Config, width: Int, height: Int) async throws -> (encTime: Double, decTime: Double, compSize: Int, metrics: [QualityMetrics]?, bitstream: [VTFrameData]) {
    return try await runHEVC(y4mPath: y4mPath, config: config, width: width, height: height, disableHWA: false)
}

func runHEVC(y4mPath: String, config: Config, width: Int, height: Int, disableHWA: Bool) async throws -> (encTime: Double, decTime: Double, compSize: Int, metrics: [QualityMetrics]?, bitstream: [VTFrameData]) {
    var encTime: Double = 0
    var compSize: Int = 0
    
    class FrameBox: @unchecked Sendable {
        var frames: [VTFrameData] = []
        let lock = NSLock()
        let dumpHandle: FileHandle?
        init(dumpEnv: String) {
            if let p = ProcessInfo.processInfo.environment[dumpEnv] {
                FileManager.default.createFile(atPath: p, contents: nil)
                dumpHandle = FileHandle(forWritingAtPath: p)
            } else { dumpHandle = nil }
        }
    }
    let frameBox = FrameBox(dumpEnv: "VEVC_DUMP_HEVC")

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
            
            guard let dataBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { return }
            let length = CMBlockBufferGetDataLength(dataBuffer)
            var data = [UInt8](repeating: 0, count: length)
            CMBlockBufferCopyDataBytes(dataBuffer, atOffset: 0, dataLength: length, destination: &data)
            let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            let formatDesc = CMSampleBufferGetFormatDescription(sampleBuffer)
            
            box.lock.lock()
            box.frames.append(VTFrameData(pts: pts, formatDesc: formatDesc, data: data))
            box.dumpHandle?.write(Data(data))
            box.lock.unlock()
        },
        refcon: Unmanaged.passUnretained(frameBox).toOpaque(),
        compressionSessionOut: &compressionSessionOut
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
    
    let encY4M = try Y4MIterator(path: y4mPath, config: config)
    let encStart = Date()
    var idx = 0
    while let imgInput = try encY4M.next() {
        autoreleasepool {
            if let pixelBuffer = createPixelBuffer(from: imgInput.vevcImage) {
                let presentationTimeStamp = CMTime(value: CMTimeValue(idx), timescale: CMTimeScale(config.framerate))
                var flags: VTEncodeInfoFlags = []
                VTCompressionSessionEncodeFrame(
                    compressionSession,
                    imageBuffer: pixelBuffer,
                    presentationTimeStamp: presentationTimeStamp,
                    duration: .invalid,
                    frameProperties: nil,
                    sourceFrameRefcon: nil,
                    infoFlagsOut: &flags
                )
                idx += 1
            }
        }
    }
    VTCompressionSessionCompleteFrames(compressionSession, untilPresentationTimeStamp: .invalid)
    encTime = Date().timeIntervalSince(encStart)
    
    for sample in frameBox.frames {
        compSize += sample.data.count
    }
    
    var decTime: Double = 0
    guard !frameBox.frames.isEmpty else { return (encTime, decTime, compSize, nil, frameBox.frames) }
    
    guard let formatDesc = frameBox.frames[0].formatDesc else {
        throw NSError(domain: "CMSampleBufferGetFormatDescription (HEVC)", code: -1, userInfo: nil)
    }
    
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
    guard decStatus == noErr, let decompressionSession = decompressionSessionOut else {
        throw NSError(domain: "VTDecompressionSessionCreate (HEVC)", code: Int(decStatus), userInfo: nil)
    }
    
    let decStart = Date()
    for sample in frameBox.frames {
        guard let sb = recreateCMSampleBuffer(from: sample) else { continue }
        var flags: VTDecodeInfoFlags = []
        VTDecompressionSessionDecodeFrame(
            decompressionSession,
            sampleBuffer: sb,
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
        
        let qY4M = try Y4MIterator(path: y4mPath, config: config)
        let qBox = QualityBox(qY4M: qY4M)
        
        struct SendableSession: @unchecked Sendable {
            let session: VTDecompressionSession
        }
        let safeQualitySession = SendableSession(session: qualitySession)
        
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            DispatchQueue.global().async {
                let sem = DispatchSemaphore(value: 300)
                let qualitySession = safeQualitySession.session
                for sample in frameBox.frames {
                    guard let sb = recreateCMSampleBuffer(from: sample) else { continue }
                    sem.wait()
                    var flags: VTDecodeInfoFlags = []
                    VTDecompressionSessionDecodeFrame(
                        qualitySession,
                        sampleBuffer: sb,
                        flags: [],
                        infoFlagsOut: &flags,
                        outputHandler: { (status, infoFlags, imageBuffer, presentationTimeStamp, presentationDuration) in
                            if let buf = imageBuffer {
                                let idx = Int(presentationTimeStamp.value)
                                if let orig = try? qBox.getOrigFrame(idx: idx) {
                                    let psnr = calculatePSNR(img1: orig, bgraBuffer: buf)
                                    let ssim = calculateSSIM(img1: orig, bgraBuffer: buf)
                                    qBox.lock.lock()
                                    qBox.metrics[idx] = QualityMetrics(psnr: psnr, ssim: ssim)
                                    qBox.lock.unlock()
                                    qBox.removeOrigFrame(idx: idx)
                                }
                            }
                            sem.signal()
                        }
                    )
                }
                VTDecompressionSessionWaitForAsynchronousFrames(qualitySession)
                continuation.resume()
            }
        }
        
        var mets = [QualityMetrics]()
        for i in 0..<idx {
            if let m = qBox.metrics[i] { mets.append(m) }
        }
        metrics = mets
    }
    
    return (encTime, decTime, compSize, metrics, frameBox.frames)
}

func recreateCMSampleBuffer(from frame: VTFrameData) -> CMSampleBuffer? {
    var blockBuffer: CMBlockBuffer?
    frame.data.withUnsafeBytes { ptr in
        guard let base = ptr.baseAddress else { return }
        CMBlockBufferCreateWithMemoryBlock(
            allocator: kCFAllocatorDefault,
            memoryBlock: UnsafeMutableRawPointer(mutating: base),
            blockLength: frame.data.count,
            blockAllocator: kCFAllocatorNull,
            customBlockSource: nil,
            offsetToData: 0,
            dataLength: frame.data.count,
            flags: 0,
            blockBufferOut: &blockBuffer
        )
    }
    guard let bb = blockBuffer else { return nil }
    
    var timing = CMSampleTimingInfo(duration: .invalid, presentationTimeStamp: frame.pts, decodeTimeStamp: .invalid)
    var sampleSize = frame.data.count
    var sampleBuffer: CMSampleBuffer?
    
    CMSampleBufferCreateReady(
        allocator: kCFAllocatorDefault,
        dataBuffer: bb,
        formatDescription: frame.formatDesc,
        sampleCount: 1,
        sampleTimingEntryCount: 1,
        sampleTimingArray: &timing,
        sampleSizeEntryCount: 1,
        sampleSizeArray: &sampleSize,
        sampleBufferOut: &sampleBuffer
    )
    
    return sampleBuffer
}

func extractVTFrames(bitstream: [VTFrameData], disableHWA: Bool, indices: Set<Int>) throws -> [Int: YCbCrImage] {
    guard bitstream.isEmpty != true else { return [:] }
    guard let formatDesc = bitstream[0].formatDesc else {
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
        guard let sb = recreateCMSampleBuffer(from: sample) else { continue }
        var flags: VTDecodeInfoFlags = []
        VTDecompressionSessionDecodeFrame(
            session,
            sampleBuffer: sb,
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