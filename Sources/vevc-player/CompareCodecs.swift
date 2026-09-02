import Foundation
import VideoToolbox
import CoreMedia
import vevc

public struct CodecFrameResult: @unchecked Sendable {
    public let image: YCbCrImage
    public let cgImage: CGImage?
    public let psnr: Double
    public let ssim: Double
    public let frameSizeBytes: Int
}

public struct CodecResultData: @unchecked Sendable {
    public let codecName: String
    public let frames: [CodecFrameResult]
    public let totalSizeBytes: Int
    public let avgBitrateKbps: Double
    public let avgPSNR: Double
    public let avgSSIM: Double
}

public struct VTFramePacket: @unchecked Sendable {
    public let pts: CMTime
    public let formatDesc: CMVideoFormatDescription?
    public let data: [UInt8]
}

// MARK: - Quality Calculation (PSNR / SSIM)

@inline(__always)
public func calculatePSNR(img1: YCbCrImage, img2: YCbCrImage) -> Double {
    let w = min(img1.width, img2.width)
    let h = min(img1.height, img2.height)
    
    let psnrY = calcPlanePSNR(p1: img1.yPlane, p2: img2.yPlane, w: w, h: h, stride1: img1.width, stride2: img2.width)
    let cw = min((img1.width + 1) / 2, (img2.width + 1) / 2)
    let ch = min((img1.height + 1) / 2, (img2.height + 1) / 2)
    let psnrU = calcPlanePSNR(p1: img1.cbPlane, p2: img2.cbPlane, w: cw, h: ch, stride1: (img1.width + 1) / 2, stride2: (img2.width + 1) / 2)
    let psnrV = calcPlanePSNR(p1: img1.crPlane, p2: img2.crPlane, w: cw, h: ch, stride1: (img1.width + 1) / 2, stride2: (img2.width + 1) / 2)
    
    return (4.0 * psnrY + psnrU + psnrV) / 6.0
}

@inline(__always)
private func calcPlanePSNR(p1: [UInt8], p2: [UInt8], w: Int, h: Int, stride1: Int, stride2: Int) -> Double {
    var ssd = 0
    let count = w * h
    if count == 0 { return 100.0 }
    
    p1.withUnsafeBufferPointer { ptr1 in
        p2.withUnsafeBufferPointer { ptr2 in
            guard let b1 = ptr1.baseAddress, let b2 = ptr2.baseAddress else { return }
            for y in 0..<h {
                let r1 = b1.advanced(by: y * stride1)
                let r2 = b2.advanced(by: y * stride2)
                for x in 0..<w {
                    let diff = Int(r1[x]) - Int(r2[x])
                    ssd += diff * diff
                }
            }
        }
    }
    
    if ssd == 0 { return 100.0 }
    let mse = Double(ssd) / Double(count)
    return 10.0 * log10((255.0 * 255.0) / mse)
}

@inline(__always)
public func calculateSSIM(img1: YCbCrImage, img2: YCbCrImage) -> Double {
    let w = min(img1.width, img2.width)
    let h = min(img1.height, img2.height)
    
    let ssimY = calcPlaneSSIM(p1: img1.yPlane, p2: img2.yPlane, w: w, h: h, stride1: img1.width, stride2: img2.width)
    let cw = min((img1.width + 1) / 2, (img2.width + 1) / 2)
    let ch = min((img1.height + 1) / 2, (img2.height + 1) / 2)
    let ssimU = calcPlaneSSIM(p1: img1.cbPlane, p2: img2.cbPlane, w: cw, h: ch, stride1: (img1.width + 1) / 2, stride2: (img2.width + 1) / 2)
    let ssimV = calcPlaneSSIM(p1: img1.crPlane, p2: img2.crPlane, w: cw, h: ch, stride1: (img1.width + 1) / 2, stride2: (img2.width + 1) / 2)
    
    return (4.0 * ssimY + ssimU + ssimV) / 6.0
}

@inline(__always)
private func calcPlaneSSIM(p1: [UInt8], p2: [UInt8], w: Int, h: Int, stride1: Int, stride2: Int) -> Double {
    var ssimSum: Double = 0
    var blocks = 0
    let C1: Double = 6.5025
    let C2: Double = 58.5225
    
    p1.withUnsafeBufferPointer { ptr1 in
        p2.withUnsafeBufferPointer { ptr2 in
            guard let b1 = ptr1.baseAddress, let b2 = ptr2.baseAddress else { return }
            for y in stride(from: 0, to: h - 7, by: 8) {
                for x in stride(from: 0, to: w - 7, by: 8) {
                    var sum1 = 0
                    var sum2 = 0
                    var sum1sq = 0
                    var sum2sq = 0
                    var sum12 = 0
                    
                    for dy in 0..<8 {
                        let r1 = b1.advanced(by: (y + dy) * stride1 + x)
                        let r2 = b2.advanced(by: (y + dy) * stride2 + x)
                        for dx in 0..<8 {
                            let v1 = Int(r1[dx])
                            let v2 = Int(r2[dx])
                            sum1 += v1
                            sum2 += v2
                            sum1sq += v1 * v1
                            sum2sq += v2 * v2
                            sum12 += v1 * v2
                        }
                    }
                    
                    let n = 64.0
                    let mu1 = Double(sum1) / n
                    let mu2 = Double(sum2) / n
                    let mu1sq = mu1 * mu1
                    let mu2sq = mu2 * mu2
                    let mu12 = mu1 * mu2
                    
                    let sigma1sq = (Double(sum1sq) / n) - mu1sq
                    let sigma2sq = (Double(sum2sq) / n) - mu2sq
                    let sigma12 = (Double(sum12) / n) - mu12
                    
                    let num = (2.0 * mu12 + C1) * (2.0 * sigma12 + C2)
                    let den = (mu1sq + mu2sq + C1) * (sigma1sq + sigma2sq + C2)
                    ssimSum += num / den
                    blocks += 1
                }
            }
        }
    }
    if blocks == 0 { return 1.0 }
    return ssimSum / Double(blocks)
}

// MARK: - VEVC Pipeline

public func runVEVCPipeline(images: [YCbCrImage], bitrate: Int, fps: Int, profile: UInt8, onProgress: @Sendable (String) -> Void) async throws -> CodecResultData {
    guard images.isEmpty != true else {
        return CodecResultData(codecName: "VEVC (Layer 2)", frames: [], totalSizeBytes: 0, avgBitrateKbps: 0, avgPSNR: 0, avgSSIM: 0)
    }
    
    onProgress("Encoding VEVC (Layer 2)...")
    let width = images[0].width
    let height = images[0].height
    
    let encoder = VEVCEncoder(
        width: width,
        height: height,
        maxbitrate: bitrate * 1000,
        framerate: fps,
        zeroThreshold: 3,
        keyint: 30,
        sceneChangeThreshold: 10,
        profile: profile
    )
    
    var chunks: [[UInt8]] = []
    var frameSizes: [Int] = []
    var totalBytes = 0
    
    for (i, img) in images.enumerated() {
        try Task.checkCancellation()
        let chunk = try await encoder.encode(image: img)
        chunks.append(chunk)
        frameSizes.append(chunk.count)
        totalBytes += chunk.count
        if i % 10 == 0 || i == images.count - 1 {
            onProgress(String(format: "Encoding VEVC: %d/%d frames", i + 1, images.count))
        }
    }
    
    onProgress("Decoding VEVC (Layer 2)...")
    let decoder = Decoder(maxLayer: 2)
    let stream = AsyncStream<[UInt8]> { continuation in
        for c in chunks { continuation.yield(c) }
        continuation.finish()
    }
    
    var decodedFrames: [YCbCrImage] = []
    var frameIdx = 0
    for try await frame in decoder.decodeStream(stream: stream) {
        try Task.checkCancellation()
        decodedFrames.append(frame)
        frameIdx += 1
        if frameIdx % 10 == 0 || frameIdx == images.count {
            onProgress(String(format: "Decoding VEVC: %d/%d frames", frameIdx, images.count))
        }
    }
    
    var results: [CodecFrameResult] = []
    var sumPsnr = 0.0
    var sumSsim = 0.0
    let count = min(images.count, decodedFrames.count)
    
    for i in 0..<count {
        let orig = images[i]
        let dec = decodedFrames[i]
        let psnr = calculatePSNR(img1: orig, img2: dec)
        let ssim = calculateSSIM(img1: orig, img2: dec)
        let cg = try? createCGImage(from: dec)
        sumPsnr += psnr
        sumSsim += ssim
        let sz = i < frameSizes.count ? frameSizes[i] : 0
        results.append(CodecFrameResult(image: dec, cgImage: cg, psnr: psnr, ssim: ssim, frameSizeBytes: sz))
    }
    
    let durationSec = Double(count) / Double(max(fps, 1))
    let avgKbps = durationSec > 0 ? (Double(totalBytes * 8) / 1000.0) / durationSec : Double(bitrate)
    let avgPsnr = count > 0 ? sumPsnr / Double(count) : 0.0
    let avgSsim = count > 0 ? sumSsim / Double(count) : 0.0
    
    return CodecResultData(
        codecName: "VEVC (Layer 2)",
        frames: results,
        totalSizeBytes: totalBytes,
        avgBitrateKbps: avgKbps,
        avgPSNR: avgPsnr,
        avgSSIM: avgSsim
    )
}

// MARK: - H.264 Pipeline (Baseline, CABAC, no B-frame)

public func runH264Pipeline(images: [YCbCrImage], bitrate: Int, fps: Int, onProgress: @Sendable (String) -> Void) async throws -> CodecResultData {
    guard images.isEmpty != true else {
        return CodecResultData(codecName: "H.264 (Baseline, CABAC)", frames: [], totalSizeBytes: 0, avgBitrateKbps: 0, avgPSNR: 0, avgSSIM: 0)
    }
    
    onProgress("Encoding H.264 (Baseline, CABAC, no B-frame)...")
    let width = images[0].width
    let height = images[0].height
    
    final class H264Box: @unchecked Sendable {
        var packets: [VTFramePacket] = []
        let lock = NSLock()
    }
    let frameBox = H264Box()
    
    var compressionSessionOut: VTCompressionSession?
    let status = VTCompressionSessionCreate(
        allocator: kCFAllocatorDefault,
        width: Int32(width),
        height: Int32(height),
        codecType: kCMVideoCodecType_H264,
        encoderSpecification: nil,
        imageBufferAttributes: nil,
        compressedDataAllocator: nil,
        outputCallback: { (outputCallbackRefCon, _, status, _, sampleBuffer) in
            guard status == noErr, let sampleBuffer = sampleBuffer else { return }
            let box = Unmanaged<H264Box>.fromOpaque(outputCallbackRefCon!).takeUnretainedValue()
            
            guard let dataBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { return }
            let length = CMBlockBufferGetDataLength(dataBuffer)
            var data = [UInt8](repeating: 0, count: length)
            CMBlockBufferCopyDataBytes(dataBuffer, atOffset: 0, dataLength: length, destination: &data)
            let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            let formatDesc = CMSampleBufferGetFormatDescription(sampleBuffer)
            
            box.lock.lock()
            box.packets.append(VTFramePacket(pts: pts, formatDesc: formatDesc, data: data))
            box.lock.unlock()
        },
        refcon: Unmanaged.passUnretained(frameBox).toOpaque(),
        compressionSessionOut: &compressionSessionOut
    )
    
    guard status == noErr, let compressionSession = compressionSessionOut else {
        throw NSError(domain: "VTCompressionSessionCreate", code: Int(status), userInfo: [NSLocalizedDescriptionKey: "Failed to create H.264 compression session"])
    }
    
    let bitRateBps = bitrate * 1000
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_AverageBitRate, value: NSNumber(value: bitRateBps))
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_DataRateLimits, value: [bitRateBps / 8 * 2, 1] as CFArray)
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_ExpectedFrameRate, value: NSNumber(value: fps))
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_RealTime, value: kCFBooleanFalse)
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_ProfileLevel, value: kVTProfileLevel_H264_Baseline_AutoLevel)
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_H264EntropyMode, value: kVTH264EntropyMode_CABAC)
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_AllowFrameReordering, value: kCFBooleanFalse)
    
    VTCompressionSessionPrepareToEncodeFrames(compressionSession)
    
    for (idx, img) in images.enumerated() {
        try Task.checkCancellation()
        autoreleasepool {
            if let pixelBuffer = createPixelBuffer(from: img) {
                let pts = CMTime(value: CMTimeValue(idx), timescale: CMTimeScale(fps))
                var flags: VTEncodeInfoFlags = []
                VTCompressionSessionEncodeFrame(
                    compressionSession,
                    imageBuffer: pixelBuffer,
                    presentationTimeStamp: pts,
                    duration: .invalid,
                    frameProperties: nil,
                    sourceFrameRefcon: nil,
                    infoFlagsOut: &flags
                )
            }
        }
        if idx % 10 == 0 || idx == images.count - 1 {
            onProgress(String(format: "Encoding H.264: %d/%d frames", idx + 1, images.count))
        }
    }
    VTCompressionSessionCompleteFrames(compressionSession, untilPresentationTimeStamp: .invalid)
    
    let packets = frameBox.packets
    var totalBytes = 0
    for p in packets { totalBytes += p.data.count }
    
    guard packets.isEmpty != true, let formatDesc = packets[0].formatDesc else {
        return CodecResultData(codecName: "H.264 (Baseline, CABAC)", frames: [], totalSizeBytes: 0, avgBitrateKbps: 0, avgPSNR: 0, avgSSIM: 0)
    }
    
    onProgress("Decoding H.264...")
    let destPixelBufferAttributes: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        kCVPixelBufferMetalCompatibilityKey as String: true
    ]
    
    var decompressionSessionOut: VTDecompressionSession?
    let decStatus = VTDecompressionSessionCreate(
        allocator: kCFAllocatorDefault,
        formatDescription: formatDesc,
        decoderSpecification: nil,
        imageBufferAttributes: destPixelBufferAttributes as CFDictionary,
        outputCallback: nil,
        decompressionSessionOut: &decompressionSessionOut
    )
    guard decStatus == noErr, let decompressionSession = decompressionSessionOut else {
        throw NSError(domain: "VTDecompressionSessionCreate", code: Int(decStatus), userInfo: [NSLocalizedDescriptionKey: "Failed to create H.264 decompression session"])
    }
    
    final class H264DecBox: @unchecked Sendable {
        var decoded: [Int: YCbCrImage] = [:]
        let lock = NSLock()
    }
    let decBox = H264DecBox()
    
    struct SendableSession: @unchecked Sendable {
        let session: VTDecompressionSession
    }
    let safeSession = SendableSession(session: decompressionSession)
    
    await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
        DispatchQueue.global().async {
            let session = safeSession.session
            for sample in packets {
                guard let sb = recreateCMSampleBuffer(from: sample) else { continue }
                var flags: VTDecodeInfoFlags = []
                VTDecompressionSessionDecodeFrame(
                    session,
                    sampleBuffer: sb,
                    flags: [],
                    infoFlagsOut: &flags,
                    outputHandler: { (_, _, imageBuffer, pts, _) in
                        if let buf = imageBuffer {
                            let idx = Int(pts.value)
                            let w = CVPixelBufferGetWidth(buf)
                            let h = CVPixelBufferGetHeight(buf)
                            let ycbcr = createYCbCrImage(from: buf, width: w, height: h)
                            decBox.lock.lock()
                            decBox.decoded[idx] = ycbcr
                            decBox.lock.unlock()
                        }
                    }
                )
            }
            VTDecompressionSessionWaitForAsynchronousFrames(session)
            continuation.resume()
        }
    }
    
    var results: [CodecFrameResult] = []
    var sumPsnr = 0.0
    var sumSsim = 0.0
    
    for i in 0..<images.count {
        guard let dec = decBox.decoded[i] else { continue }
        let orig = images[i]
        let psnr = calculatePSNR(img1: orig, img2: dec)
        let ssim = calculateSSIM(img1: orig, img2: dec)
        let cg = try? createCGImage(from: dec)
        sumPsnr += psnr
        sumSsim += ssim
        let sz = i < packets.count ? packets[i].data.count : 0
        results.append(CodecFrameResult(image: dec, cgImage: cg, psnr: psnr, ssim: ssim, frameSizeBytes: sz))
    }
    
    let durationSec = Double(results.count) / Double(max(fps, 1))
    let avgKbps = durationSec > 0 ? (Double(totalBytes * 8) / 1000.0) / durationSec : Double(bitrate)
    let avgPsnr = results.count > 0 ? sumPsnr / Double(results.count) : 0.0
    let avgSsim = results.count > 0 ? sumSsim / Double(results.count) : 0.0
    
    return CodecResultData(
        codecName: "H.264 (Baseline, CABAC)",
        frames: results,
        totalSizeBytes: totalBytes,
        avgBitrateKbps: avgKbps,
        avgPSNR: avgPsnr,
        avgSSIM: avgSsim
    )
}

// MARK: - H.265 (HEVC) Pipeline

public func runHEVCPipeline(images: [YCbCrImage], bitrate: Int, fps: Int, onProgress: @Sendable (String) -> Void) async throws -> CodecResultData {
    guard images.isEmpty != true else {
        return CodecResultData(codecName: "H.265 (HEVC)", frames: [], totalSizeBytes: 0, avgBitrateKbps: 0, avgPSNR: 0, avgSSIM: 0)
    }
    
    onProgress("Encoding H.265 (HEVC)...")
    let width = images[0].width
    let height = images[0].height
    
    final class HEVCBox: @unchecked Sendable {
        var packets: [VTFramePacket] = []
        let lock = NSLock()
    }
    let frameBox = HEVCBox()
    
    var compressionSessionOut: VTCompressionSession?
    let status = VTCompressionSessionCreate(
        allocator: kCFAllocatorDefault,
        width: Int32(width),
        height: Int32(height),
        codecType: kCMVideoCodecType_HEVC,
        encoderSpecification: nil,
        imageBufferAttributes: nil,
        compressedDataAllocator: nil,
        outputCallback: { (outputCallbackRefCon, _, status, _, sampleBuffer) in
            guard status == noErr, let sampleBuffer = sampleBuffer else { return }
            let box = Unmanaged<HEVCBox>.fromOpaque(outputCallbackRefCon!).takeUnretainedValue()
            
            guard let dataBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { return }
            let length = CMBlockBufferGetDataLength(dataBuffer)
            var data = [UInt8](repeating: 0, count: length)
            CMBlockBufferCopyDataBytes(dataBuffer, atOffset: 0, dataLength: length, destination: &data)
            let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            let formatDesc = CMSampleBufferGetFormatDescription(sampleBuffer)
            
            box.lock.lock()
            box.packets.append(VTFramePacket(pts: pts, formatDesc: formatDesc, data: data))
            box.lock.unlock()
        },
        refcon: Unmanaged.passUnretained(frameBox).toOpaque(),
        compressionSessionOut: &compressionSessionOut
    )
    
    guard status == noErr, let compressionSession = compressionSessionOut else {
        throw NSError(domain: "VTCompressionSessionCreate", code: Int(status), userInfo: [NSLocalizedDescriptionKey: "Failed to create HEVC compression session"])
    }
    
    let bitRateBps = bitrate * 1000
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_AverageBitRate, value: NSNumber(value: bitRateBps))
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_DataRateLimits, value: [bitRateBps / 8 * 2, 1] as CFArray)
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_ExpectedFrameRate, value: NSNumber(value: fps))
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_RealTime, value: kCFBooleanFalse)
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_ProfileLevel, value: kVTProfileLevel_HEVC_Main_AutoLevel)
    VTSessionSetProperty(compressionSession, key: kVTCompressionPropertyKey_AllowFrameReordering, value: kCFBooleanFalse)
    
    VTCompressionSessionPrepareToEncodeFrames(compressionSession)
    
    for (idx, img) in images.enumerated() {
        try Task.checkCancellation()
        autoreleasepool {
            if let pixelBuffer = createPixelBuffer(from: img) {
                let pts = CMTime(value: CMTimeValue(idx), timescale: CMTimeScale(fps))
                var flags: VTEncodeInfoFlags = []
                VTCompressionSessionEncodeFrame(
                    compressionSession,
                    imageBuffer: pixelBuffer,
                    presentationTimeStamp: pts,
                    duration: .invalid,
                    frameProperties: nil,
                    sourceFrameRefcon: nil,
                    infoFlagsOut: &flags
                )
            }
        }
        if idx % 10 == 0 || idx == images.count - 1 {
            onProgress(String(format: "Encoding HEVC: %d/%d frames", idx + 1, images.count))
        }
    }
    VTCompressionSessionCompleteFrames(compressionSession, untilPresentationTimeStamp: .invalid)
    
    let packets = frameBox.packets
    var totalBytes = 0
    for p in packets { totalBytes += p.data.count }
    
    guard packets.isEmpty != true, let formatDesc = packets[0].formatDesc else {
        return CodecResultData(codecName: "H.265 (HEVC)", frames: [], totalSizeBytes: 0, avgBitrateKbps: 0, avgPSNR: 0, avgSSIM: 0)
    }
    
    onProgress("Decoding H.265 (HEVC)...")
    let destPixelBufferAttributes: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        kCVPixelBufferMetalCompatibilityKey as String: true
    ]
    
    var decompressionSessionOut: VTDecompressionSession?
    let decStatus = VTDecompressionSessionCreate(
        allocator: kCFAllocatorDefault,
        formatDescription: formatDesc,
        decoderSpecification: nil,
        imageBufferAttributes: destPixelBufferAttributes as CFDictionary,
        outputCallback: nil,
        decompressionSessionOut: &decompressionSessionOut
    )
    guard decStatus == noErr, let decompressionSession = decompressionSessionOut else {
        throw NSError(domain: "VTDecompressionSessionCreate", code: Int(decStatus), userInfo: [NSLocalizedDescriptionKey: "Failed to create HEVC decompression session"])
    }
    
    final class HEVCDecBox: @unchecked Sendable {
        var decoded: [Int: YCbCrImage] = [:]
        let lock = NSLock()
    }
    let decBox = HEVCDecBox()
    
    struct SendableSession: @unchecked Sendable {
        let session: VTDecompressionSession
    }
    let safeSession = SendableSession(session: decompressionSession)
    
    await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
        DispatchQueue.global().async {
            let session = safeSession.session
            for sample in packets {
                guard let sb = recreateCMSampleBuffer(from: sample) else { continue }
                var flags: VTDecodeInfoFlags = []
                VTDecompressionSessionDecodeFrame(
                    session,
                    sampleBuffer: sb,
                    flags: [],
                    infoFlagsOut: &flags,
                    outputHandler: { (_, _, imageBuffer, pts, _) in
                        if let buf = imageBuffer {
                            let idx = Int(pts.value)
                            let w = CVPixelBufferGetWidth(buf)
                            let h = CVPixelBufferGetHeight(buf)
                            let ycbcr = createYCbCrImage(from: buf, width: w, height: h)
                            decBox.lock.lock()
                            decBox.decoded[idx] = ycbcr
                            decBox.lock.unlock()
                        }
                    }
                )
            }
            VTDecompressionSessionWaitForAsynchronousFrames(session)
            continuation.resume()
        }
    }
    
    var results: [CodecFrameResult] = []
    var sumPsnr = 0.0
    var sumSsim = 0.0
    
    for i in 0..<images.count {
        guard let dec = decBox.decoded[i] else { continue }
        let orig = images[i]
        let psnr = calculatePSNR(img1: orig, img2: dec)
        let ssim = calculateSSIM(img1: orig, img2: dec)
        let cg = try? createCGImage(from: dec)
        sumPsnr += psnr
        sumSsim += ssim
        let sz = i < packets.count ? packets[i].data.count : 0
        results.append(CodecFrameResult(image: dec, cgImage: cg, psnr: psnr, ssim: ssim, frameSizeBytes: sz))
    }
    
    let durationSec = Double(results.count) / Double(max(fps, 1))
    let avgKbps = durationSec > 0 ? (Double(totalBytes * 8) / 1000.0) / durationSec : Double(bitrate)
    let avgPsnr = results.count > 0 ? sumPsnr / Double(results.count) : 0.0
    let avgSsim = results.count > 0 ? sumSsim / Double(results.count) : 0.0
    
    return CodecResultData(
        codecName: "H.265 (HEVC)",
        frames: results,
        totalSizeBytes: totalBytes,
        avgBitrateKbps: avgKbps,
        avgPSNR: avgPsnr,
        avgSSIM: avgSsim
    )
}

// MARK: - Helper to Recreate CMSampleBuffer

public func recreateCMSampleBuffer(from frame: VTFramePacket) -> CMSampleBuffer? {
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
