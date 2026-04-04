import Foundation

@inline(__always)
private func buildVEVCHeader(width: Int, height: Int, framerate: Int) -> [UInt8] {
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
    
    public init(width: Int, height: Int, maxbitrate: Int, framerate: Int = 30, zeroThreshold: Int = 3, keyint: Int = 60, sceneChangeThreshold: Int = 15, maxConcurrency: Int = 4) {
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
    
    /// Encode images and return concatenated byte array.
    /// Convenience for roundtrip tests and simple usage.
    public func encodeToData(images: [YCbCrImage]) async throws -> [UInt8] {
        let chunks = try await encode(images: images)
        var result: [UInt8] = []
        for chunk in chunks {
            result.append(contentsOf: chunk)
        }
        return result
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
                let pool = BlockViewPool()
                let coreEncoder = LayersEncodeActor(
                    width: width,
                    height: height,
                    maxbitrate: maxbitrate,
                    framerate: framerate,
                    zeroThreshold: zeroThreshold,
                    keyint: keyint,
                    sceneChangeThreshold: sceneChangeThreshold,
                    pool: pool
                )
                
                do {
                    continuation.yield(buildVEVCHeader(width: width, height: height, framerate: framerate))
                    
                    var buffer: [YCbCrImage] = []
                    while let img = await iterator.next() {
                        // シーンチェンジ判定: 既存バッファがある場合、最後のフレームと比較
                        if let lastImg = buffer.last {
                            let sad = estimateFastSAD(a: img, b: lastImg)
                            if sceneChangeThreshold < sad {
                                // シーンチェンジ検知：現在のGOPバッファを強制終了してエンコード
                                let gopBytes = try await coreEncoder.encodeTemporalGOPChunk(images: buffer)
                                continuation.yield(gopBytes)
                                buffer.removeAll(keepingCapacity: true)
                            }
                        }
                        
                        buffer.append(img)
                        if buffer.count == keyint {
                            let gopBytes = try await coreEncoder.encodeTemporalGOPChunk(images: buffer)
                            continuation.yield(gopBytes)
                            buffer.removeAll(keepingCapacity: true)
                        }
                    }
                    if !buffer.isEmpty {
                        let gopBytes = try await coreEncoder.encodeTemporalGOPChunk(images: buffer)
                        continuation.yield(gopBytes)
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
    let pool: BlockViewPool
    
    private var rateController: RateController
    private var framesSinceKeyframe = 0
    private var frameIndex = 0
    private var qt: QuantizationTable?
    
    init(width: Int, height: Int, maxbitrate: Int, framerate: Int, zeroThreshold: Int, keyint: Int, sceneChangeThreshold: Int, pool: BlockViewPool) {
        self.width = width
        self.height = height
        self.maxbitrate = maxbitrate
        self.framerate = framerate
        self.zeroThreshold = zeroThreshold
        self.keyint = keyint
        self.sceneChangeThreshold = sceneChangeThreshold
        self.pool = pool
        self.rateController = RateController(maxbitrate: maxbitrate, framerate: framerate, keyint: keyint)
    }
    
    func encodeTemporalGOPChunk(images: [YCbCrImage]) async throws -> [UInt8] {
        guard !images.isEmpty else {
            throw NSError(domain: "vevc.Encoder", code: 2, userInfo: [NSLocalizedDescriptionKey: "TemporalGOP4 requires at least 1 frame"])
        }
        
        // Rate control
        var baseQt: QuantizationTable
        if keyint <= framesSinceKeyframe || frameIndex == 0 {
            let targetBits = rateController.beginGOP()
            baseQt = estimateQuantization(img: images[0], targetBits: targetBits)
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
        
        // 双方向予測用: GOP先頭フレーム（I-frame）の復元結果を後方参照として保持
        // デコーダ側でもI-frameの復元結果を使用するため、入力原データではなく復元結果を使用
        var firstReconstructed: PlaneData420? = nil
        
        for (_, plane) in planes.enumerated() {
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
            
            // 双方向予測の適用判定:
            // 【実験】全P-frameに双方向予測を適用し、効果の上限を測定する。
            // I-frameの復元結果を後方参照として全P-frameで利用する。
            let useBidirectional = isPFrame && firstReconstructed != nil && planes.count >= 2
            
            let bytes: [UInt8]
            let reconstructed: PlaneData420
            
            if useBidirectional {
                (bytes, reconstructed) = try await encodeSpatialLayers(
                    pd: plane, pool: pool, predictedPd: previousReconstructed, nextPd: firstReconstructed,
                    maxbitrate: localMaxbitrate, qtY: frameQtY, qtC: frameQtC, zeroThreshold: localZeroThreshold
                )
            } else {
                (bytes, reconstructed) = try await encodeSpatialLayers(
                    pd: plane, pool: pool, predictedPd: previousReconstructed, maxbitrate: localMaxbitrate,
                    qtY: frameQtY, qtC: frameQtC, zeroThreshold: localZeroThreshold
                )
            }
            encodedFrames.append(bytes)
            
            // Feed back to rate controller for next frame's qStep prediction
            let encodedBits = bytes.count * 8
            if isFirstEncoded {
                rateController.consumeIFrame(bits: encodedBits, qStep: Int(frameQtY.step))
                isFirstEncoded = false
            } else {
                let frameSAD = isPFrame ? estimateFrameSAD(current: plane, previous: previousReconstructed!) : 0
                rateController.consumePFrame(bits: encodedBits, qStep: Int(frameQtY.step), sad: frameSAD)
            }
            previousReconstructed = reconstructed
            
            // I-frame（先頭フレーム）の復元結果を保存
            if firstReconstructed == nil {
                firstReconstructed = reconstructed
            }
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
        
        let (bytes, _) = try await encodeSpatialLayers(pd: curr, pool: pool, predictedPd: nil, maxbitrate: maxbitrate, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
        
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

@inline(__always)
private func estimateFastSAD(a: YCbCrImage, b: YCbCrImage) -> Int {
    guard a.yPlane.count == b.yPlane.count, 0 < a.yPlane.count else { return 0 }
    let count = a.yPlane.count
    var sum: UInt64 = 0
    a.yPlane.withUnsafeBufferPointer { aPtr in
        b.yPlane.withUnsafeBufferPointer { bPtr in
            for i in stride(from: 0, to: count, by: 4) {
                sum += UInt64(abs(Int(aPtr[i]) - Int(bPtr[i])))
            }
        }
    }
    return Int((sum * 4) / UInt64(count))
}

/// Estimate frame-level SAD (Sum of Absolute Differences) between current
/// and previous PlaneData420 Y planes by sampling representative blocks.
/// Returns average per-pixel SAD as an Int for RateController input.
@inline(__always)
private func estimateFrameSAD(current: PlaneData420, previous: PlaneData420) -> Int {
    let width = current.width
    let height = current.height
    guard 0 < width && 0 < height else { return 0 }
    
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
    
    return 0 < totalPixels ? totalSAD / totalPixels : 0
}

@inline(__always)
private func estimateQuantization(img: YCbCrImage, targetBits: Int) -> QuantizationTable {
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
    func fetchBlockY(reader: ImageReader, x: Int, y: Int, w: Int, h: Int, pool: BlockViewPool) -> BlockView {
        let block = pool.get(width: w, height: h)
        reader.readBlockY(x: x, y: y, width: w, height: h, into: block.view)
        return block
    }

    @inline(__always)
    func fetchBlockCb(reader: ImageReader, x: Int, y: Int, w: Int, h: Int, pool: BlockViewPool) -> BlockView {
        let block = pool.get(width: w, height: h)
        reader.readBlockCb(x: x, y: y, width: w, height: h, into: block.view)
        return block
    }

    @inline(__always)
    func fetchBlockCr(reader: ImageReader, x: Int, y: Int, w: Int, h: Int, pool: BlockViewPool) -> BlockView {
        let block = pool.get(width: w, height: h)
        reader.readBlockCr(x: x, y: y, width: w, height: h, into: block.view)
        return block
    }
    
    let estPool = BlockViewPool(maxPerSize: 8)
    for (sx, sy) in points {
        var blockY = fetchBlockY(reader: reader, x: sx, y: sy, w: w, h: h, pool: estPool)
        totalSampleBits += measureBlockBits8(block: &blockY, qt: qt)
        estPool.put(blockY)
        
        var blockCb = fetchBlockCb(reader: reader, x: sx, y: sy, w: w, h: h, pool: estPool)
        totalSampleBits += measureBlockBits8(block: &blockCb, qt: qt)
        estPool.put(blockCb)
        
        var blockCr = fetchBlockCr(reader: reader, x: sx, y: sy, w: w, h: h, pool: estPool)
        totalSampleBits += measureBlockBits8(block: &blockCr, qt: qt)
        estPool.put(blockCr)
    }
    
    let samplePixels = points.count * (w * h) * 3
    let totalPixels = img.width * img.height * 3
    
    // Use Int64 to prevent overflow in multiplication:
    // estimatedTotalBits = totalSampleBits * (totalPixels / samplePixels)
    // predictedStep = probeStep * estimatedTotalBits * 85 / (targetBits * 100)
    let estimatedTotalBits64 = (Int64(totalSampleBits) * Int64(totalPixels)) / Int64(samplePixels)
    // I-frame quality bias: 0.85 = 85/100
    let predictedStep64 = (Int64(probeStep) * estimatedTotalBits64 * 85) / (Int64(targetBits) * 100)

    let q = min(256, Int(max(1, predictedStep64)))
    
    return QuantizationTable(baseStep: q)
}

@inline(__always)
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

@inline(__always)
private func measureBlockBits8(block: inout BlockView, qt: QuantizationTable) -> Int {
    let view = block.view
    let sub = dwt2d_8_sb(view)
    
    quantizeSIMD(sub.ll, q: qt.qLow)
    quantizeSIMD(sub.hl, q: qt.qMid)
    quantizeSIMD(sub.lh, q: qt.qMid)
    quantizeSIMD(sub.hh, q: qt.qHigh)
    
    let isZero = isEffectivelyZeroBase4(data: block.base, threshold: 0)
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

@inline(__always)
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

