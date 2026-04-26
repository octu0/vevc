import Foundation

// MARK: - LayersEncoder & LayersCoreEncoder (Temporal DWT, Mode=0x00)

public actor VEVCEncoder {
    public nonisolated let width: Int
    public nonisolated let height: Int
    public nonisolated let maxbitrate: Int
    public nonisolated let framerate: Int
    public nonisolated let zeroThreshold: Int
    public nonisolated let keyint: Int
    public nonisolated let sceneChangeThreshold: Int
    public nonisolated let maxConcurrency: Int
    
    private let coreEncoder: LayersEncodeActor
    private var lastImg: YCbCrImage? = nil
    private var frameIndex = 0
    private let pool: BlockViewPool
    
    public init(width: Int, height: Int, maxbitrate: Int, framerate: Int = 30, zeroThreshold: Int = 3, keyint: Int = 60, sceneChangeThreshold: Int = 10, maxConcurrency: Int = 4) {
        self.width = width
        self.height = height
        self.maxbitrate = maxbitrate
        self.framerate = framerate
        self.zeroThreshold = zeroThreshold
        self.keyint = keyint
        self.sceneChangeThreshold = sceneChangeThreshold
        self.maxConcurrency = maxConcurrency
        
        self.pool = BlockViewPool()
        self.coreEncoder = LayersEncodeActor(
            width: width,
            height: height,
            maxbitrate: maxbitrate,
            framerate: framerate,
            zeroThreshold: zeroThreshold,
            keyint: keyint,
            sceneChangeThreshold: sceneChangeThreshold,
            pool: pool
        )
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

    public func encode(image: YCbCrImage) async throws -> [UInt8] {
        let isSceneChange: Bool
        if let last = lastImg {
            let sad = estimateFastSAD(a: image, b: last)
            isSceneChange = (sceneChangeThreshold < sad)
        } else {
            isSceneChange = false
        }
        
        let bytes = try await coreEncoder.encodeNextFrame(image: image, isSceneChange: isSceneChange)
        
        var result: [UInt8] = []
        if frameIndex == 0 {
            let fileHeader = VEVCFileHeader(width: width, height: height, framerate: framerate)
            result.append(contentsOf: fileHeader.serialize())
        }
        result.append(contentsOf: bytes)
        
        lastImg = image
        frameIndex += 1
        
        return result
    }

    public func encode(stream: AsyncStream<YCbCrImage>) -> AsyncThrowingStream<[UInt8], Error> {
        return AsyncThrowingStream { continuation in
            Task {
                var iterator = stream.makeAsyncIterator()
                do {
                    while let img = await iterator.next() {
                        let bytes = try await self.encode(image: img)
                        continuation.yield(bytes)
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
    
    private var previousInputPlane: PlaneData420?
    private var releasePreviousInput: (@Sendable () -> Void)?
    
    private var firstReconstructed: PlaneData420?
    private var releaseFirstRecon: (@Sendable () -> Void)?
    
    private var previousReconstructed: PlaneData420?
    private var releasePreviousRecon: (@Sendable () -> Void)?
    
    internal init(width: Int, height: Int, maxbitrate: Int, framerate: Int, zeroThreshold: Int, keyint: Int, sceneChangeThreshold: Int, pool: BlockViewPool) {
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
    
    public init(width: Int, height: Int, maxbitrate: Int, framerate: Int, zeroThreshold: Int, keyint: Int, sceneChangeThreshold: Int) {
        self.width = width
        self.height = height
        self.maxbitrate = maxbitrate
        self.framerate = framerate
        self.zeroThreshold = zeroThreshold
        self.keyint = keyint
        self.sceneChangeThreshold = sceneChangeThreshold
        self.pool = BlockViewPool()
        self.rateController = RateController(maxbitrate: maxbitrate, framerate: framerate, keyint: keyint)
    }
    
    deinit {
        releasePreviousInput?()
        releasePreviousRecon?()
        releaseFirstRecon?()
    }
    
    @inline(__always)
    public func encodeNextFrame(image: YCbCrImage, isSceneChange: Bool) async throws -> [UInt8] {
        let (plane, releasePlane) = toPlaneData420(image: image, pool: pool)
        
        let isIFrame = (keyint <= framesSinceKeyframe || frameIndex == 0 || isSceneChange)
        
        if isIFrame {
            // Rate control
            let targetBits = rateController.beginGOP()
            let baseQt = estimateQuantization(img: image, targetBits: targetBits)
            self.qt = baseQt
            framesSinceKeyframe = 0
            
            let baseStep = Int(baseQt.step)
            let qtY = QuantizationTable(baseStep: max(1, baseStep), isChroma: false, layerIndex: 0)
            let qtC = QuantizationTable(baseStep: max(1, baseStep), isChroma: true, layerIndex: 0)
            
            let (bytes, reconstructed, releaseRecon) = try await encodeSpatialLayers(
                pd: plane, pool: pool, maxbitrate: maxbitrate,
                qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold, roundOffset: 0
            )
            
            rateController.consumeIFrame(bits: bytes.count * 8, qStep: Int(qtY.step))
            
            // Clean up old state
            releasePreviousInput?()
            releasePreviousRecon?()
            releaseFirstRecon?()
            
            previousInputPlane = plane
            releasePreviousInput = releasePlane
            
            firstReconstructed = reconstructed
            releaseFirstRecon = releaseRecon
            
            previousReconstructed = reconstructed
            releasePreviousRecon = nil
            
            framesSinceKeyframe += 1
            frameIndex += 1
            
            return bytes
        } else {
            // Duplicate frame detection
            if let prevIn = previousInputPlane {
                let isDuplicate = isPlaneIdentical(a: plane, b: prevIn)
                if isDuplicate {
                    releasePlane()
                    framesSinceKeyframe += 1
                    frameIndex += 1
                    return VEVCFrameHeader(frameType: .copyFrame).serialize()
                }
            }
            
            releasePreviousInput?()
            previousInputPlane = plane
            releasePreviousInput = releasePlane
            
            guard let baseQt = self.qt, let prevRecon = previousReconstructed, let firstRecon = firstReconstructed else {
                throw NSError(domain: "vevc.Encoder", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing reference frames for P-frame"])
            }
            let baseStep = Int(baseQt.step)
            
            let frameSAD = estimateFrameSAD(current: plane, previous: prevRecon)
            let adjustedStep = rateController.calculatePFrameQStep(currentSAD: frameSAD, baseStep: baseStep)
            let qtY = QuantizationTable(baseStep: max(1, adjustedStep), isChroma: false, layerIndex: 0)
            let qtC = QuantizationTable(baseStep: max(1, adjustedStep), isChroma: true, layerIndex: 0)
            
            let (bytes, reconstructed, releaseRecon) = try await encodeSpatialLayers(
                pd: plane, pool: pool, predictedPd: prevRecon, nextPd: firstRecon,
                maxbitrate: maxbitrate, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold,
                roundOffset: framesSinceKeyframe % 2, gopPosition: framesSinceKeyframe
            )
            
            rateController.consumePFrame(bits: bytes.count * 8, qStep: Int(qtY.step), sad: frameSAD)
            
            let oldRecon = previousReconstructed!
            let oldRelease = releasePreviousRecon
            
            let isPrevFirst = oldRecon.y.withUnsafeBufferPointer { p in
                firstRecon.y.withUnsafeBufferPointer { f in p.baseAddress == f.baseAddress }
            }
            if isPrevFirst != true {
                oldRelease?()
            }
            
            previousReconstructed = reconstructed
            releasePreviousRecon = releaseRecon
            
            framesSinceKeyframe += 1
            frameIndex += 1
            
            return bytes
        }
    }
}

@inline(__always)
private func estimateFastSAD(a: YCbCrImage, b: YCbCrImage) -> Int {
    guard a.yPlane.count == b.yPlane.count, 0 < a.yPlane.count else { return 0 }
    let yCount = a.yPlane.count
    var sumY: UInt64 = 0
    a.yPlane.withUnsafeBufferPointer { aPtr in
        b.yPlane.withUnsafeBufferPointer { bPtr in
            for i in stride(from: 0, to: yCount, by: 4) {
                sumY += UInt64(abs(Int(aPtr[i]) - Int(bPtr[i])))
            }
        }
    }
    let ySAD = Int((sumY * 4) / UInt64(yCount))
    
    // Chroma SAD: detect scene changes where luminance is similar but color differs
    // (e.g. dark scene to dark scene with different color palette)
    let cbCount = a.cbPlane.count
    guard a.cbPlane.count == b.cbPlane.count, 0 < cbCount else { return ySAD }
    
    var sumCb: UInt64 = 0
    var sumCr: UInt64 = 0
    withUnsafePointers(a.cbPlane, b.cbPlane, a.crPlane, b.crPlane) { aCb, bCb, aCr, bCr in
        for i in stride(from: 0, to: cbCount, by: 4) {
            sumCb += UInt64(abs(Int(aCb[i]) - Int(bCb[i])))
            sumCr += UInt64(abs(Int(aCr[i]) - Int(bCr[i])))
        }
    }
    let chromaSAD = Int(((sumCb + sumCr) * 4) / UInt64(cbCount * 2))
    
    // Weight: Y dominates but Chroma provides critical color-change detection
    return ySAD + chromaSAD
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
    
    if 0 < totalPixels {
        return totalSAD / totalPixels
    }
    return 0
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
        reader.readBlockY(x: x, y: y, width: w, height: h, into: block)
        return block
    }

    @inline(__always)
    func fetchBlockCb(reader: ImageReader, x: Int, y: Int, w: Int, h: Int, pool: BlockViewPool) -> BlockView {
        let block = pool.get(width: w, height: h)
        reader.readBlockCb(x: x, y: y, width: w, height: h, into: block)
        return block
    }

    @inline(__always)
    func fetchBlockCr(reader: ImageReader, x: Int, y: Int, w: Int, h: Int, pool: BlockViewPool) -> BlockView {
        let block = pool.get(width: w, height: h)
        reader.readBlockCr(x: x, y: y, width: w, height: h, into: block)
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

    let q = min(256, Int(max(2, predictedStep64)))
    
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
        switch true {
        case ia <= ic && ib <= ic:
            predicted = min(ia, ib)
        case ic <= ia && ic <= ib:
            predicted = max(ia, ib)
        default:
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
    let view = block
    let sub = dwt2DBlock8Subbands(view)
    
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

