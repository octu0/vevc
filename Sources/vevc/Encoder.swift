import Foundation
// MARK: - VEVCEncoder

public actor VEVCEncoder {
    public nonisolated let width: Int
    public nonisolated let height: Int
    public nonisolated let maxbitrate: Int
    public nonisolated let framerate: Int
    public nonisolated let zeroThreshold: Int
    public nonisolated let keyint: Int
    public nonisolated let sceneChangeThreshold: Int
    public nonisolated let maxConcurrency: Int
    public nonisolated let qstep: Int?
    public nonisolated let profile: UInt8
    public nonisolated let skipThreshold: Int
    
    private let coreEncoder: LayersEncodeActor
    private var frameIndex = 0
    private let pool: BlockViewPool
    
    public init(width: Int, height: Int, maxbitrate: Int, framerate: Int = 30, zeroThreshold: Int = 3, keyint: Int = 30, sceneChangeThreshold: Int = 10, maxConcurrency: Int = 4, profile: UInt8 = 0x01, skipThreshold: Int = 2) {
        self.width = width
        self.height = height
        self.maxbitrate = maxbitrate
        self.framerate = framerate
        self.zeroThreshold = zeroThreshold
        self.keyint = keyint
        self.sceneChangeThreshold = sceneChangeThreshold
        self.maxConcurrency = maxConcurrency
        self.qstep = nil
        self.profile = profile
        self.skipThreshold = EncoderTuning.envInt(key: "VEVC_SKIP_THRESH", defaultValue: skipThreshold)
        
        self.pool = BlockViewPool()
        self.coreEncoder = LayersEncodeActor(
            width: width,
            height: height,
            maxbitrate: self.maxbitrate,
            framerate: framerate,
            zeroThreshold: zeroThreshold,
            keyint: keyint,
            sceneChangeThreshold: sceneChangeThreshold,
            pool: pool,
            qstep: nil,
            profile: profile,
            skipThreshold: self.skipThreshold
        )
    }

    public init(width: Int, height: Int, qstep: Int, framerate: Int = 30, zeroThreshold: Int = 3, keyint: Int = 30, sceneChangeThreshold: Int = 10, maxConcurrency: Int = 4, profile: UInt8 = 0x01, skipThreshold: Int = 2) {
        self.width = width
        self.height = height
        self.maxbitrate = 0
        self.framerate = framerate
        self.zeroThreshold = zeroThreshold
        self.keyint = keyint
        self.sceneChangeThreshold = sceneChangeThreshold
        self.maxConcurrency = maxConcurrency
        self.qstep = qstep
        self.profile = profile
        self.skipThreshold = EncoderTuning.envInt(key: "VEVC_SKIP_THRESH", defaultValue: skipThreshold)
        
        self.pool = BlockViewPool()
        self.coreEncoder = LayersEncodeActor(
            width: width,
            height: height,
            maxbitrate: 0,
            framerate: framerate,
            zeroThreshold: zeroThreshold,
            keyint: keyint,
            sceneChangeThreshold: sceneChangeThreshold,
            pool: pool,
            qstep: qstep,
            profile: profile,
            skipThreshold: self.skipThreshold
        )
    }
    
    @inline(__always)
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
    @inline(__always)
    public func encodeToData(images: [YCbCrImage]) async throws -> [UInt8] {
        let stream = self.encode(stream: AsyncStream<YCbCrImage> { c in
            for img in images { c.yield(img) }; c.finish()
        })
        var out = [UInt8]()
        for try await data in stream {
            out.append(contentsOf: data)
        }
#if VEVC_ME_STATS
        MEStats.printStats()
#endif
        return out
    }

    @inline(__always)
    public func encode(image: YCbCrImage, forceKeyFrame: Bool = false) async throws -> [UInt8] {
        let bytes = try await coreEncoder.encodeFrame(image: image, forceKeyFrame: forceKeyFrame)
        
        var result: [UInt8] = []
        if frameIndex == 0 {
            let fileHeader = VEVCFileHeader(width: width, height: height, framerate: framerate, profile: profile)
            result.append(contentsOf: fileHeader.serialize())
        }
        result.append(contentsOf: bytes)
        
        frameIndex += 1
        
        return result
    }

    @inline(__always)
    public func encode<S: AsyncSequence & Sendable>(stream: S) -> AsyncThrowingStream<[UInt8], Error> where S.Element == YCbCrImage {
        return AsyncThrowingStream { continuation in
            Task {
                var iterator = stream.makeAsyncIterator()
                do {
                    while let img = try await iterator.next() {
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
    let qstep: Int?
    let profile: UInt8
    let skipThreshold: Int
    
    private var rateController: RateController
    private var framesSinceKeyframe = 0
    private var frameIndex = 0
    private var qt: QuantizationTable?
    
    private var previousInputPlane: PlaneData420?
    private var releasePreviousInput: (@Sendable () -> Void)?
    private var previousMVs: MotionVectors?
    
    private var firstReconstructed: PlaneData420?
    private var releaseFirstRecon: (@Sendable () -> Void)?
    
    private var previousReconstructed: PlaneData420?
    private var releasePreviousRecon: (@Sendable () -> Void)?
    
    private var firstInputPlane: PlaneData420?
    private var releaseFirstInput: (@Sendable () -> Void)?
    private var staticCounters: [Int] = []
    private var cachedNextSub2: [Int16]?
    private var cachedNextSub1: [Int16]?
    
    internal init(width: Int, height: Int, maxbitrate: Int, framerate: Int, zeroThreshold: Int, keyint: Int, sceneChangeThreshold: Int, pool: BlockViewPool, qstep: Int? = nil, profile: UInt8 = 0x01, skipThreshold: Int = 2) {
        self.width = width
        self.height = height
        self.maxbitrate = maxbitrate
        self.framerate = framerate
        self.zeroThreshold = zeroThreshold
        self.keyint = keyint
        self.sceneChangeThreshold = sceneChangeThreshold
        self.pool = pool
        self.qstep = qstep
        self.profile = profile
        self.skipThreshold = skipThreshold
        self.rateController = RateController(maxbitrate: maxbitrate, framerate: framerate, keyint: keyint)
        
        let bw = (width + 31) / 32
        let bh = (height + 31) / 32
        self.staticCounters = [Int](repeating: 0, count: bw * bh)
    }
    
    public init(width: Int, height: Int, maxbitrate: Int, framerate: Int, zeroThreshold: Int, keyint: Int, sceneChangeThreshold: Int, profile: UInt8 = 0x01, skipThreshold: Int = 2) {
        self.width = width
        self.height = height
        self.maxbitrate = maxbitrate
        self.framerate = framerate
        self.zeroThreshold = zeroThreshold
        self.keyint = keyint
        self.sceneChangeThreshold = sceneChangeThreshold
        self.pool = BlockViewPool()
        self.qstep = nil
        self.profile = profile
        self.skipThreshold = skipThreshold
        self.rateController = RateController(maxbitrate: maxbitrate, framerate: framerate, keyint: keyint)
        
        let bw = (width + 31) / 32
        let bh = (height + 31) / 32
        self.staticCounters = [Int](repeating: 0, count: bw * bh)
    }
    
    deinit {
        releasePreviousInput?()
        releasePreviousRecon?()
        releaseFirstRecon?()
        releaseFirstInput?()
    }
    
    @inline(__always)
    public func encodeFrame(image: YCbCrImage, forceKeyFrame: Bool = false) async throws -> [UInt8] {
        let (plane, releasePlane) = toPlaneData420(image: image, pool: pool)
        
        var isSceneChange = false
        if let prev = previousInputPlane {
            let sad = estimateFastSAD(a: plane, b: prev)
            isSceneChange = (sceneChangeThreshold < sad)
        }
        
        var forceIFrame = forceKeyFrame
        if forceIFrame != true && rateController.isDriftAccelerating {
            forceIFrame = true
        }
        
        let isIFrame = (keyint <= framesSinceKeyframe || frameIndex == 0 || isSceneChange || forceIFrame)
        
        if isIFrame {
            // Rate control
            let baseStep: Int
            if let fixedStep = self.qstep {
                baseStep = fixedStep
                self.qt = QuantizationTable(baseStep: max(16, baseStep), isChroma: false, layerIndex: 0)
            } else {
                let targetBits = rateController.beginGOP()
                let baseQt = estimateQuantization(img: image, targetBits: targetBits, rateController: rateController)
                self.qt = baseQt
                baseStep = Int(baseQt.step)
            }
            
            framesSinceKeyframe = 0
            
            let qtY = QuantizationTable(baseStep: max(16, baseStep), isChroma: false, layerIndex: 0)
            let qtC = QuantizationTable(baseStep: max(16, baseStep), isChroma: true, layerIndex: 0)
            
            let (bytes, reconstructed, mvs, _, releaseRecon) = try await encodeSpatialLayers(
                pd: plane, pool: pool, maxbitrate: maxbitrate,
                qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold, roundOffset: 0, profile: profile, skipThreshold: self.skipThreshold
            )
            
            if self.qstep == nil {
                rateController.consumeIFrame(bits: bytes.count * 8, qStep: Int(qtY.step))
            }
            
            // Clean up old state
            releasePreviousInput?()
            releasePreviousRecon?()
            releaseFirstRecon?()
            releaseFirstInput?()
            
            previousInputPlane = plane
            releasePreviousInput = releasePlane
            
            firstReconstructed = reconstructed
            releaseFirstRecon = releaseRecon
            
            cachedNextSub2 = nil
            cachedNextSub1 = nil
            
            previousReconstructed = reconstructed
            releasePreviousRecon = nil
            
            let (firstIn, releaseFirstIn) = toPlaneData420(image: image, pool: pool)
            firstInputPlane = firstIn
            releaseFirstInput = releaseFirstIn
            
            for i in 0..<self.staticCounters.count {
                self.staticCounters[i] = 0
            }
            
            previousMVs = mvs
            
            framesSinceKeyframe += 1
            frameIndex += 1
            
            return bytes
        }
        
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
        
        guard let baseQt = self.qt, let prevRecon = previousReconstructed, let firstRecon = firstReconstructed, let firstIn = firstInputPlane, let prevIn = previousInputPlane else {
            throw EncodeError.missingReferenceFramesForPFrame
        }
        let baseStep = Int(baseQt.step)
        
        let frameSAD = estimateFrameSAD(current: plane, previous: prevRecon)
        let adjustedStep: Int
        if let fixedStep = self.qstep {
            adjustedStep = fixedStep
        } else {
            adjustedStep = rateController.calculatePFrameQStep(currentSAD: frameSAD, baseStep: baseStep)
        }
        let qtY = QuantizationTable(baseStep: max(16, adjustedStep), isChroma: false, layerIndex: 0)
        let qtC = QuantizationTable(baseStep: max(16, adjustedStep), isChroma: true, layerIndex: 0)
        
        var localCounters = self.staticCounters
        let (bytes, reconstructed, mvs, sads, releaseRecon, nSub2, nSub1) = try await encodeSpatialLayers(
            pd: plane, pool: pool, predictedPd: prevRecon, nextPd: firstRecon, prevInput: prevIn, ltrInput: firstIn, prevMVs: previousMVs,
            maxbitrate: maxbitrate, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold,
            roundOffset: framesSinceKeyframe % 2, gopPosition: framesSinceKeyframe, profile: profile, skipThreshold: self.skipThreshold, staticCounters: &localCounters,
            cachedNextSub2: self.cachedNextSub2, cachedNextSub1: self.cachedNextSub1
        )
        self.staticCounters = localCounters
        self.cachedNextSub2 = nSub2
        self.cachedNextSub1 = nSub1
        
        // Using masked recon distortion for quality metric
        let reconDistortion = computeMaskedReconDistortion(original: plane, reconstructed: reconstructed, sads: sads)
        
        if self.qstep == nil {
            rateController.consumePFrame(bits: bytes.count * 8, qStep: Int(adjustedStep), sad: frameSAD, distortion: reconDistortion)
        }
        
        releasePreviousInput?()
        previousInputPlane = plane
        releasePreviousInput = releasePlane

        let oldRecon = previousReconstructed!
        let oldRelease = releasePreviousRecon
        
        let isPrevFirst = withUnsafePointers(oldRecon.y, firstRecon.y) { p, f in
            p == f
        }
        if isPrevFirst != true {
            oldRelease?()
        }
        
        previousReconstructed = reconstructed
        releasePreviousRecon = releaseRecon
        previousMVs = mvs
        
        framesSinceKeyframe += 1
        frameIndex += 1
        
        return bytes
    }
}

@inline(__always)
private func estimateFastSAD(a: PlaneData420, b: PlaneData420) -> Int {
    guard a.y.count == b.y.count, 0 < a.y.count else { return 0 }
    let yCount = a.y.count
    var sumY: UInt64 = 0
    withUnsafePointers(a.y, b.y) { aPtr, bPtr in
        for i in stride(from: 0, to: yCount, by: 4) {
            sumY += UInt64(abs(Int(aPtr[i]) - Int(bPtr[i])))
        }
    }
    let ySAD = Int((sumY * 4) / UInt64(yCount))
    
    // Chroma SAD: detect scene changes where luminance is similar but color differs
    // (e.g. dark scene to dark scene with different color palette)
    let cbCount = a.cb.count
    guard a.cb.count == b.cb.count, 0 < cbCount else { return ySAD }
    
    var sumCb: UInt64 = 0
    var sumCr: UInt64 = 0
    withUnsafePointers(a.cb, b.cb, a.cr, b.cr) { aCb, bCb, aCr, bCr in
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

/// Full block scan with activity mask-based reconstruction distortion measurement.
/// Returns per-pixel average SAD (same unit as traditional estimateFrameSAD).
@inline(__always)
func computeMaskedReconDistortion(
    original: PlaneData420,
    reconstructed: PlaneData420,
    sads: [Int]?
) -> Int {
    let width = original.width
    let height = original.height
    
    let blockSize = 32
    let colCount = (width + 31) / 32
    let rowCount = (height + 31) / 32
    
    var totalSAD: Int = 0
    var activePixels: Int = 0
    var totalFallbackSAD: Int = 0
    var totalPixels: Int = 0
    
    withUnsafePointers(original.y, reconstructed.y) { oBase, rBase in
            
            for r in 0..<rowCount {
                let sy = r * blockSize
                let bh = min(blockSize, height - sy)
                let rowOffset = r * colCount
                
                for c in 0..<colCount {
                    let sx = c * blockSize
                    let bw = min(blockSize, width - sx)
                    
                    var blockSAD = 0
                    for y in sy..<sy+bh {
                        let oRow = oBase + y * width + sx
                        let rRow = rBase + y * width + sx
                        
                        for x in 0..<bw {
                            blockSAD += abs(Int(oRow[x]) - Int(rRow[x]))
                        }
                    }
                    
                    let pixels = bw * bh
                    totalPixels += pixels
                    totalFallbackSAD += blockSAD
                    
                    if let sads = sads {
                        if 256 < sads[rowOffset + c] {
                            totalSAD += blockSAD
                            activePixels += pixels
                        }
                    }
                }
            }
        }
    
    // Fallback: Use full block average if active blocks are less than 5% of total
    if sads == nil || activePixels < (totalPixels / 20) {
        if 0 < totalPixels {
            return (totalFallbackSAD << 8) / totalPixels
        }
        return 0
    }
    
    if 0 < activePixels {
        return (totalSAD << 8) / activePixels
    }
    return 0
}

@inline(__always)
private func estimateQuantization(img: YCbCrImage, targetBits: Int, rateController: RateController) -> QuantizationTable {
    let probeStep = 1024
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
    // I-frame quality bias: 0.78 = 78/100
    // A lower factor produces larger (higher quality) I-frames, providing
    // a stronger structural base for subsequent P-frames to reference.
    let predictedStep64 = (Int64(probeStep) * estimatedTotalBits64 * Int64(EncoderTuning.shared.iFrameQuantizationScale)) / (Int64(targetBits) * 100)
    let correctedStep64 = (predictedStep64 * Int64(rateController.rateGainQ8)) >> 8

    // I-Frame QP floor = 1: allows near-lossless quality at high bitrates.
    // The cliff-edge discontinuity at low baseStep (previously requiring
    // floor=5) is now resolved by RateController.calculatePFrameQStep
    // guaranteeing maxStep>=40 independently of baseStep.
    let q = min(4096, Int(max(16, correctedStep64)))
    
    var finalQ = q
    if rateController.isQualitySaturated {
        let safeAvg = max(1, rateController.avgDistortionQ8)
        let ratioQ8 = min(512, (rateController.targetDistortionQ8 * 256) / safeAvg)
        finalQ = max(q, min((q * ratioQ8) / 256, q * 2))
    }
    
    return QuantizationTable(baseStep: finalQ)
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
    
    quantizeDPCM(sub.ll, q: qt.qLow)
    quantize4(sub.hl, q: qt.qMid)
    quantize4(sub.lh, q: qt.qMid)
    quantize4(sub.hh, q: qt.qHigh)
    
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

public struct EncoderTuning: @unchecked Sendable {
    public static let shared = EncoderTuning()

    public let l0LumaThresholdPFrame: Int
    public let l0ChromaThresholdScale: Int
    public let iFrameQuantizationScale: Int
    public let l16LumaThreshold: Int
    public let l32LumaThreshold: Int
    
    public init(
        l0LumaThresholdPFrame: Int = 4,
        l0ChromaThresholdScale: Int = 8,
        iFrameQuantizationScale: Int = 100,
        l16LumaThreshold: Int = 3,
        l32LumaThreshold: Int = 4
    ) {
        self.l0LumaThresholdPFrame = Self.envInt(key: "VEVC_TUNE_L0_LUMA", defaultValue: l0LumaThresholdPFrame)
        self.l0ChromaThresholdScale = Self.envInt(key: "VEVC_TUNE_L0_CHROMA", defaultValue: l0ChromaThresholdScale)
        self.iFrameQuantizationScale = Self.envInt(key: "VEVC_TUNE_IFRAME_SCALE", defaultValue: iFrameQuantizationScale)
        self.l16LumaThreshold = Self.envInt(key: "VEVC_TUNE_L16_LUMA", defaultValue: l16LumaThreshold)
        self.l32LumaThreshold = Self.envInt(key: "VEVC_TUNE_L32_LUMA", defaultValue: l32LumaThreshold)
    }
    
    @inline(__always)
    internal static func envInt(key: String, defaultValue: Int) -> Int {
        if let valStr = ProcessInfo.processInfo.environment[key], let val = Int(valStr) {
            return val
        }
        return defaultValue
    }
}
