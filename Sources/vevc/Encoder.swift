import Foundation



struct EncoderUtils {
    @inline(__always)
    static func buildVEVCHeader(width: Int, height: Int, framerate: Int) -> [UInt8] {
        var header: [UInt8] = [0x56, 0x45, 0x56, 0x43] // Magic 'VEVC'
        let metadataPayloadSize: UInt16 = 9
        appendUInt16BE(&header, metadataPayloadSize)
        header.append(0x01) // Profile 1
        appendUInt16BE(&header, UInt16(width))
        appendUInt16BE(&header, UInt16(height))
        header.append(0x01) // ColorGamut: BT.709
        appendUInt16BE(&header, UInt16(framerate))
        header.append(0x00) // Timescale: 0=1000ms
        return header
    }
}

// MARK: - LayersEncoder & LayersCoreEncoder (Temporal DWT, Mode=0x00)

public class VEVCEncoder {
    public let width: Int
    public let height: Int
    public let maxbitrate: Int
    public let framerate: Int
    public let zeroThreshold: Int
    public let keyint: Int
    public let sceneChangeThreshold: Int
    public let maxConcurrency: Int
    
    public init(width: Int, height: Int, maxbitrate: Int, framerate: Int = 30, zeroThreshold: Int = 3, keyint: Int = 60, sceneChangeThreshold: Int = 32, maxConcurrency: Int = 4) {
        self.width = width
        self.height = height
        self.maxbitrate = maxbitrate
        self.framerate = framerate
        self.zeroThreshold = zeroThreshold
        self.keyint = keyint
        self.sceneChangeThreshold = sceneChangeThreshold
        self.maxConcurrency = maxConcurrency
    }
    
    public func encode(images: [YCbCrImage]) async throws -> [[UInt8]] {
        let stream = AsyncStream<YCbCrImage> { continuation in
            for img in images {
                continuation.yield(img)
            }
            continuation.finish()
        }
        var chunks: [[UInt8]] = []
        for try await chunk in self.encode(stream: stream) {
            chunks.append(chunk)
        }
        return chunks
    }
    
    public func encode(stream: AsyncStream<YCbCrImage>) -> AsyncThrowingStream<[UInt8], Error> {
        let width = self.width
        let height = self.height
        let maxbitrate = self.maxbitrate
        let framerate = self.framerate
        let zeroThreshold = self.zeroThreshold
        let keyint = self.keyint
        let sceneChangeThreshold = self.sceneChangeThreshold
        
        return AsyncThrowingStream { continuation in
            Task {
                var iterator = stream.makeAsyncIterator()
                let coreEncoder = LayersEncodeActor(
                    width: width,
                    height: height,
                    maxbitrate: maxbitrate,
                    framerate: framerate,
                    zeroThreshold: zeroThreshold,
                    keyint: keyint,
                    sceneChangeThreshold: sceneChangeThreshold
                )
                
                do {
                    continuation.yield(EncoderUtils.buildVEVCHeader(width: width, height: height, framerate: framerate))
                    
                    var buffer: [YCbCrImage] = []
                    while let img = await iterator.next() {
                        buffer.append(img)
                        if buffer.count == 4 {
                            let gopBytes = try await coreEncoder.encodeTemporalGOP4(images: buffer)
                            continuation.yield(gopBytes)
                            buffer.removeAll(keepingCapacity: true)
                        }
                    }
                    for img in buffer {
                        let chunk = try await coreEncoder.encodeSingleFrame(image: img)
                        continuation.yield(chunk)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}

actor LayersEncodeActor {
    let width: Int
    let height: Int
    let maxbitrate: Int
    let framerate: Int
    let zeroThreshold: Int
    let keyint: Int
    let sceneChangeThreshold: Int
    
    private var rateController: RateController
    private var framesSinceKeyframe = 0
    private var frameIndex = 0
    private var qt: QuantizationTable?
    
    init(width: Int, height: Int, maxbitrate: Int, framerate: Int, zeroThreshold: Int, keyint: Int, sceneChangeThreshold: Int) {
        self.width = width
        self.height = height
        self.maxbitrate = maxbitrate
        self.framerate = framerate
        self.zeroThreshold = zeroThreshold
        self.keyint = keyint
        self.sceneChangeThreshold = sceneChangeThreshold
        self.rateController = RateController(maxbitrate: maxbitrate, framerate: framerate, keyint: keyint)
    }
    
    func encodeTemporalGOP4(images: [YCbCrImage]) async throws -> [UInt8] {
        guard !images.isEmpty else {
            throw NSError(domain: "vevc.Encoder", code: 2, userInfo: [NSLocalizedDescriptionKey: "TemporalGOP4 requires at least 1 frame"])
        }
        
        // Rate control
        var baseQt: QuantizationTable
        if keyint <= framesSinceKeyframe || frameIndex == 0 {
            let targetBitsPerGOP = rateController.beginGOP()
            baseQt = estimateQuantization(img: images[0], targetBits: targetBitsPerGOP / images.count)
            self.qt = baseQt
            framesSinceKeyframe = 0
        } else {
            baseQt = self.qt!
        }
        framesSinceKeyframe += images.count
        frameIndex += images.count
        
        let planes = toPlaneData420(images: images)
        let baseStep = Int(baseQt.step)
        
        let localMaxbitrate = maxbitrate
        let localZeroThreshold = zeroThreshold
        
        var encodedFrames = [[UInt8]]()
        var previousReconstructed: PlaneData420? = nil
        var previousInputPlane: PlaneData420? = nil
        var isFirstEncoded = true
        
        for plane in planes {
            // Duplicate frame detection: if current frame pixels are identical
            // to the previous frame, emit a copy frame (empty data, FrameLen=0)
            // instead of encoding the full frame. This saves significant data
            // for videos with duplicate frames (e.g. 24fps→60fps upconversion).
            if let prevInput = previousInputPlane {
                let isDuplicate = isPlaneIdentical(a: plane, b: prevInput)
                if isDuplicate {
                    encodedFrames.append([]) // empty = copy frame (FrameLen=0)
                    // previousReconstructed stays the same (reuse previous reconstruction)
                    // previousInputPlane stays the same
                    continue
                }
            }
            previousInputPlane = plane
            
            // Per-frame rate control:
            // - I-frame (first non-copy encoded frame): uses baseQt from GOP-level rate control
            // - P-frames: dynamically adjust qStep based on frame-level SAD activity
            //   using the existing RateController.calculatePFrameQStep() method.
            let frameQtY: QuantizationTable
            let frameQtC: QuantizationTable
            let isPFrame = previousReconstructed != nil
            
            if isPFrame {
                // Compute frame-level SAD by sampling Y plane differences
                let frameSAD = estimateFrameSAD(current: plane, previous: previousReconstructed!)
                let adjustedStep = rateController.calculatePFrameQStep(currentSAD: frameSAD, baseStep: baseStep)
                frameQtY = QuantizationTable(baseStep: max(1, adjustedStep), isChroma: false, layerIndex: 0)
                frameQtC = QuantizationTable(baseStep: max(1, adjustedStep), isChroma: true, layerIndex: 0)
            } else {
                frameQtY = QuantizationTable(baseStep: max(1, baseStep), isChroma: false, layerIndex: 0)
                frameQtC = QuantizationTable(baseStep: max(1, baseStep), isChroma: true, layerIndex: 0)
            }
            
            let (bytes, reconstructed) = try await encodeSpatialLayers(
                pd: plane, predictedPd: previousReconstructed, maxbitrate: localMaxbitrate,
                qtY: frameQtY, qtC: frameQtC, zeroThreshold: localZeroThreshold
            )
            encodedFrames.append(bytes)
            
            // Feed back to rate controller for next frame's qStep prediction
            let encodedBits = bytes.count * 8
            if isFirstEncoded {
                rateController.consumeIFrame(bits: encodedBits, qStep: Int(frameQtY.step))
                isFirstEncoded = false
            } else {
                let frameSAD = isPFrame ? estimateFrameSAD(current: plane, previous: previousReconstructed!) : 0.0
                rateController.consumePFrame(bits: encodedBits, qStep: Int(frameQtY.step), sad: frameSAD)
            }
            previousReconstructed = reconstructed
        }
        
        var gopBody: [UInt8] = []
        appendUInt32BE(&gopBody, UInt32(images.count)) // GOP size
        
        // GOP bitstream layout: [GOPBodySize(4B)] [GOPSize(4B)] [FrameLen(4B) FrameData]...
        // Convention: FrameLen == 0 indicates a "copy frame" — the frame is
        // pixel-identical to its predecessor and carries no encoded payload.
        // This avoids encoding redundant data for duplicate input frames,
        // which is common in telecine/pulldown converted content (e.g. 24fps→60fps,
        // where approximately 60% of frames are exact duplicates).
        // The decoder recognizes FrameLen == 0 and reuses the previous decoded frame.
        for encoded in encodedFrames {
            appendUInt32BE(&gopBody, UInt32(encoded.count))
            gopBody.append(contentsOf: encoded)
        }
        
        var out: [UInt8] = []
        appendUInt32BE(&out, UInt32(gopBody.count))
        out.append(contentsOf: gopBody)
        
        return out
    }
    
    // For handling remainder frames at the end of the stream
    func encodeSingleFrame(image: YCbCrImage) async throws -> [UInt8] {
        let curr = toPlaneData420(images: [image])[0]
        
        if keyint <= framesSinceKeyframe || frameIndex == 0 {
            let targetBits = rateController.beginGOP()
            self.qt = estimateQuantization(img: image, targetBits: targetBits)
            framesSinceKeyframe = 0
        }
        guard let qt = self.qt else { throw NSError(domain: "vevc.Encoder", code: 1, userInfo: nil) }
        
        let qtY = QuantizationTable(baseStep: max(1, Int(qt.step)), isChroma: false, layerIndex: 0)
        let qtC = QuantizationTable(baseStep: max(1, Int(qt.step) * 2), isChroma: true, layerIndex: 0)
        
        let (bytes, _) = try await encodeSpatialLayers(pd: curr, predictedPd: nil, maxbitrate: maxbitrate, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
        
        framesSinceKeyframe += 1
        frameIndex += 1
        
        var gopBody: [UInt8] = []
        appendUInt32BE(&gopBody, 1) // GOP size
        
        appendUInt32BE(&gopBody, UInt32(bytes.count))
        gopBody.append(contentsOf: bytes)
        
        var out: [UInt8] = []
        appendUInt32BE(&out, UInt32(gopBody.count))
        out.append(contentsOf: gopBody)
        
        return out
    }
}


/// Estimate frame-level SAD (Sum of Absolute Differences) between current
/// and previous PlaneData420 Y planes by sampling representative blocks.
/// Returns average per-pixel SAD as a Double for RateController input.
@inline(__always)
func estimateFrameSAD(current: PlaneData420, previous: PlaneData420) -> Double {
    let width = current.width
    let height = current.height
    guard width > 0 && height > 0 else { return 0.0 }
    
    let blockSize = 32
    let bw = min(blockSize, width)
    let bh = min(blockSize, height)
    
    // Sample 8 blocks at strategic positions (same as estimateQuantization)
    let points: [(Int, Int)] = [
        (0, 0),
        (max(0, width - bw), 0),
        (0, max(0, height - bh)),
        (max(0, width - bw), max(0, height - bh)),
        (max(0, (width - bw) / 2), 0),
        (max(0, width - bw), max(0, (height - bh) / 2)),
        (max(0, (width - bw) / 2), max(0, height - bh)),
        (0, max(0, (height - bh) / 2)),
    ]
    
    var totalSAD: Int = 0
    var totalPixels: Int = 0
    
    for (sx, sy) in points {
        for y in sy..<min(sy + bh, height) {
            let rowOffset = y * width
            for x in sx..<min(sx + bw, width) {
                let idx = rowOffset + x
                totalSAD += abs(Int(current.y[idx]) - Int(previous.y[idx]))
            }
        }
        totalPixels += bw * bh
    }
    
    return totalPixels > 0 ? Double(totalSAD) / Double(totalPixels) : 0.0
}

func estimateQuantization(img: YCbCrImage, targetBits: Int) -> QuantizationTable {
    let probeStep = 64
    let qt = QuantizationTable(baseStep: probeStep)
    
    let w = (img.width / 8)
    let h = (img.height / 8)
    
    let points: [(Int, Int)] = [
        (0, 0),
        ((img.width - w), 0),
        (0, (img.height - h)),
        ((img.width - w), (img.height - h)),
        (((img.width - w) / 2), 0),
        ((img.width - w), ((img.height - h) / 2)),
        (((img.width - w) / 2), (img.height - h)),
        (0, ((img.height - h) / 2)),
    ]
    
    var totalSampleBits = 0
    let reader = ImageReader(img: img)
    @inline(__always)
    func fetchBlockY(reader: ImageReader, x: Int, y: Int, w: Int, h: Int) -> Block2D {
        var block = Block2D(width: w, height: h)
        block.withView { view in
            for i in 0..<h {
                view.setRow(offsetY: i, row: reader.rowY(x: x, y: y + i, size: w))
            }
        }
        return block
    }

    @inline(__always)
    func fetchBlockCb(reader: ImageReader, x: Int, y: Int, w: Int, h: Int) -> Block2D {
        var block = Block2D(width: w, height: h)
        block.withView { view in
            for i in 0..<h {
                view.setRow(offsetY: i, row: reader.rowCb(x: x, y: y + i, size: w))
            }
        }
        return block
    }

    @inline(__always)
    func fetchBlockCr(reader: ImageReader, x: Int, y: Int, w: Int, h: Int) -> Block2D {
        var block = Block2D(width: w, height: h)
        block.withView { view in
            for i in 0..<h {
                view.setRow(offsetY: i, row: reader.rowCr(x: x, y: y + i, size: w))
            }
        }
        return block
    }
    
    for (sx, sy) in points {
        var blockY = fetchBlockY(reader: reader, x: sx, y: sy, w: w, h: h)
        totalSampleBits += measureBlockBits8(block: &blockY, qt: qt)
        
        var blockCb = fetchBlockCb(reader: reader, x: sx, y: sy, w: w, h: h)
        totalSampleBits += measureBlockBits8(block: &blockCb, qt: qt)
        
        var blockCr = fetchBlockCr(reader: reader, x: sx, y: sy, w: w, h: h)
        totalSampleBits += measureBlockBits8(block: &blockCr, qt: qt)
    }
    
    let samplePixels = points.count * (w * h) * 3
    let totalPixels = img.width * img.height * 3
    
    let estimatedTotalBits = Double(totalSampleBits) * (Double(totalPixels) / Double(samplePixels))
    let ratio = estimatedTotalBits / Double(targetBits)
    // I-frame quality bias balances compression vs quality.
    // 0.85 is tuned for SSIM Min ~0.85 with 22% I-frame allocation.
    let predictedStep = Double(probeStep) * ratio * 0.85

    let q = min(256, Int(max(1, predictedStep)))
    
    return QuantizationTable(baseStep: q)
}

private func estimateRiceBitsDPCM4(block: BlockView, lastVal: inout Int16) -> Int {
    let count = 4 * 4
    let ptr0 = block.rowPointer(y: 0)
    let ptr1 = block.rowPointer(y: 1)
    let ptr2 = block.rowPointer(y: 2)
    let ptr3 = block.rowPointer(y: 3)
    
    @inline(__always)
    func errorMED(_ x: Int16, _ a: Int16, _ b: Int16, _ c: Int16) -> Int {
        let ia = Int(a), ib = Int(b), ic = Int(c)
        let predicted: Int
        if ia <= ic && ib <= ic {
            predicted = min(ia, ib)
        } else if ic <= ia && ic <= ib {
            predicted = max(ia, ib)
        } else {
            predicted = ia + ib - ic
        }
        return abs(Int(x) - predicted)
    }

    var sumDiffAbs = abs(Int(ptr0[0]) - Int(lastVal))
    sumDiffAbs += abs(Int(ptr0[1]) - Int(ptr0[0]))
    sumDiffAbs += abs(Int(ptr0[2]) - Int(ptr0[1]))
    sumDiffAbs += abs(Int(ptr0[3]) - Int(ptr0[2]))

    sumDiffAbs += abs(Int(ptr1[0]) - Int(ptr0[0]))
    sumDiffAbs += errorMED(ptr1[1], ptr1[0], ptr0[1], ptr0[0])
    sumDiffAbs += errorMED(ptr1[2], ptr1[1], ptr0[2], ptr0[1])
    sumDiffAbs += errorMED(ptr1[3], ptr1[2], ptr0[3], ptr0[2])
    
    sumDiffAbs += abs(Int(ptr2[0]) - Int(ptr1[0]))
    sumDiffAbs += errorMED(ptr2[1], ptr2[0], ptr1[1], ptr1[0])
    sumDiffAbs += errorMED(ptr2[2], ptr2[1], ptr1[2], ptr1[1])
    sumDiffAbs += errorMED(ptr2[3], ptr2[2], ptr1[3], ptr1[2])

    sumDiffAbs += abs(Int(ptr3[0]) - Int(ptr2[0]))
    sumDiffAbs += errorMED(ptr3[1], ptr3[0], ptr2[1], ptr2[0])
    sumDiffAbs += errorMED(ptr3[2], ptr3[1], ptr2[2], ptr2[1])
    sumDiffAbs += errorMED(ptr3[3], ptr3[2], ptr2[3], ptr2[2])

    lastVal = ptr3[3]
    
    let meanInt = sumDiffAbs / count
    let k: Int
    if meanInt < 1 {
        k = 0
    } else {
        k = (Int.bitWidth - 1) - meanInt.leadingZeroBitCount
    }
    
    let divisorShift = max(0, k - 1)
    let bodyBits = sumDiffAbs >> divisorShift
    let headerBits = count * (1 + k)
    
    return bodyBits + headerBits
}

private func estimateRiceBitsDPCM16(block: BlockView, lastVal: inout Int16) -> Int {
    let count = 16 * 16
    
    @inline(__always)
    func errorMED(_ x: Int16, _ a: Int16, _ b: Int16, _ c: Int16) -> Int {
        let ia = Int(a), ib = Int(b), ic = Int(c)
        let predicted: Int
        if ia <= ic && ib <= ic {
            predicted = min(ia, ib)
        } else if ic <= ia && ic <= ib {
            predicted = max(ia, ib)
        } else {
            predicted = ia + ib - ic
        }
        return abs(Int(x) - predicted)
    }

    var sumDiffAbs = 0
    var last = lastVal
    for y in 0..<16 {
        let ptrY = block.rowPointer(y: y)
        if y == 0 {
            for x in 0..<16 {
                if x == 0 {
                    sumDiffAbs += abs(Int(ptrY[0]) - Int(last))
                } else {
                    sumDiffAbs += abs(Int(ptrY[x]) - Int(ptrY[x - 1]))
                }
            }
        } else {
            let ptrPrevY = block.rowPointer(y: y - 1)
            for x in 0..<16 {
                if x == 0 {
                    sumDiffAbs += abs(Int(ptrY[0]) - Int(ptrPrevY[0]))
                } else {
                    sumDiffAbs += errorMED(ptrY[x], ptrY[x - 1], ptrPrevY[x], ptrPrevY[x - 1])
                }
            }
        }
        last = ptrY[15]
    }
    lastVal = last
    
    let meanInt = sumDiffAbs / count
    let k: Int
    if meanInt < 1 {
        k = 0
    } else {
        k = (Int.bitWidth - 1) - meanInt.leadingZeroBitCount
    }
    
    let divisorShift = max(0, k - 1)
    let bodyBits = sumDiffAbs >> divisorShift
    let headerBits = count * (1 + k)
    
    return bodyBits + headerBits
}

private func measureBlockBits8(block: inout Block2D, qt: QuantizationTable) -> Int {
    var sub = block.withView { view in
        return dwt2d_8_sb(&view)
    }
    
    quantizeSIMD(&sub.ll, q: qt.qLow)
    quantizeSIMD(&sub.hl, q: qt.qMid)
    quantizeSIMD(&sub.lh, q: qt.qMid)
    quantizeSIMD(&sub.hh, q: qt.qHigh)
    
    let isZero = block.data.withUnsafeMutableBufferPointer { ptr in
        return isEffectivelyZeroBase4(data: ptr, threshold: 0)
    }
    if isZero {
        return 1
    }
    
    var bits = 1
    var lastVal: Int16 = 0
    bits += estimateRiceBitsDPCM4(block: sub.ll, lastVal: &lastVal)
    bits += estimateRiceBits4(block: sub.hl)
    bits += estimateRiceBits4(block: sub.lh)
    bits += estimateRiceBits4(block: sub.hh)
    
    return bits
}

private func estimateRiceBits32(block: BlockView) -> Int {
    var sumAbs = 0
    let count = (32 * 32)
    
    for y in 0..<32 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<32 {
            sumAbs += abs(Int(ptr[x]))
        }
    }
    let meanInt = sumAbs / count
    let k: Int
    if meanInt < 1 {
        k = 0
    } else {
        k = (Int.bitWidth - 1) - meanInt.leadingZeroBitCount
    }
    
    let divisorShift = max(0, k - 1)
    let bodyBits = sumAbs >> divisorShift
    let headerBits = count * (1 + k)
    
    return bodyBits + headerBits
}

private func estimateRiceBits16(block: BlockView) -> Int {
    var sumAbs = 0
    let count = (16 * 16)
    
    for y in 0..<16 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<16 {
            sumAbs += abs(Int(ptr[x]))
        }
    }
    let meanInt = sumAbs / count
    let k: Int
    if meanInt < 1 {
        k = 0
    } else {
        k = (Int.bitWidth - 1) - meanInt.leadingZeroBitCount
    }
    
    let divisorShift = max(0, k - 1)
    let bodyBits = sumAbs >> divisorShift
    let headerBits = count * (1 + k)
    
    return bodyBits + headerBits
}

private func estimateRiceBits8(block: BlockView) -> Int {
    var sumAbs = 0
    let count = (8 * 8)
    
    for y in 0..<8 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<8 {
            sumAbs += abs(Int(ptr[x]))
        }
    }
    let meanInt = sumAbs / count
    let k: Int
    if meanInt < 1 {
        k = 0
    } else {
        k = (Int.bitWidth - 1) - meanInt.leadingZeroBitCount
    }
    
    let divisorShift = max(0, k - 1)
    let bodyBits = sumAbs >> divisorShift
    let headerBits = count * (1 + k)
    
    return bodyBits + headerBits
}

private func estimateRiceBits4(block: BlockView) -> Int {
    var sumAbs = 0
    let count = (4 * 4)
    
    for y in 0..<4 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<4 {
            sumAbs += abs(Int(ptr[x]))
        }
    }
    let meanInt = sumAbs / count
    let k: Int
    if meanInt < 1 {
        k = 0
    } else {
        k = (Int.bitWidth - 1) - meanInt.leadingZeroBitCount
    }
    
    let divisorShift = max(0, k - 1)
    let bodyBits = sumAbs >> divisorShift
    let headerBits = count * (1 + k)
    
    return bodyBits + headerBits
}

private func measureBlockBits32(block: inout Block2D, qt: QuantizationTable) -> Int {
    var sub = block.withView { view in
        return dwt2d_32_sb(&view)
    }
    
    quantizeSIMD(&sub.ll, q: qt.qLow)
    quantizeSIMD(&sub.hl, q: qt.qMid)
    quantizeSIMD(&sub.lh, q: qt.qMid)
    quantizeSIMD(&sub.hh, q: qt.qHigh)
    
    let isZero = block.data.withUnsafeMutableBufferPointer { ptr in
        return isEffectivelyZeroBase32(data: ptr, threshold: 0)
    }
    if isZero {
        return 1
    }
    
    var bits = 1
    var lastVal: Int16 = 0
    bits += estimateRiceBitsDPCM16(block: sub.ll, lastVal: &lastVal)
    bits += estimateRiceBits16(block: sub.hl)
    bits += estimateRiceBits16(block: sub.lh)
    bits += estimateRiceBits16(block: sub.hh)
    
    return bits
}

