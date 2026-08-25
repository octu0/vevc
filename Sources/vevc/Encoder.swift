import Foundation

/// Quality-ceiling floor for the I-frame coding step (Q4 units, 512 = real
/// step 32). The live-streaming target does not need the SSIM peaks that
/// fine I-frame quantization produces; the unspent I-frame bits stay in the
/// GOP budget and flow to the P frames (min-SSIM support). P-frame budgeting
/// derives from the unfloored estimate, so P quality is free to exceed the
/// floored I-frame level.
let iFrameQStepFloorQ4: Int = 512

private final class SafeReleaseAction: @unchecked Sendable {
    private var action: (@Sendable () -> Void)?
    init(_ action: @escaping @Sendable () -> Void) {
        self.action = action
    }
    func release() {
        if let act = action {
            action = nil
            act()
        }
    }
}

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
    public nonisolated let reconThresholdScale: Int
    public nonisolated let gop: Int
    public nonisolated let l2Cadence: Int
    public nonisolated let l1Cadence: Int
    public nonisolated let l0Cadence: Int
    public nonisolated let motionMaskingPx: Int
    public nonisolated let smoothL2: Int
    public nonisolated let smoothL1: Int
    public nonisolated let smoothL0: Int
    public nonisolated let temporalLayers: Int
    
    private let coreEncoder: LayersEncodeActor
    private var frameIndex = 0
    private let pool: BlockViewPool
    
    public init(width: Int, height: Int, maxbitrate: Int, framerate: Int = 30, zeroThreshold: Int = 3, keyint: Int = 30, sceneChangeThreshold: Int = 500, maxConcurrency: Int = 4, profile: UInt8 = 0x01, skipThreshold: Int = 2, reconThresholdScale: Int = 1, gop: Int = 12, l2Cadence: Int = 4, l1Cadence: Int = 2, l0Cadence: Int = 1, motionMaskingPx: Int = 2, smoothL2: Int = 1, smoothL1: Int = 2, smoothL0: Int = 1, temporalLayers: Int = 1) {
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
        self.skipThreshold = EncoderTuning.shared.skipThreshold ?? skipThreshold
        self.reconThresholdScale = EncoderTuning.shared.reconThresholdScale ?? reconThresholdScale
        self.gop = gop
        self.l2Cadence = l2Cadence
        self.l1Cadence = l1Cadence
        self.l0Cadence = l0Cadence
        self.motionMaskingPx = motionMaskingPx
        self.smoothL2 = smoothL2
        self.smoothL1 = smoothL1
        self.smoothL0 = smoothL0
        self.temporalLayers = temporalLayers
        
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
            skipThreshold: self.skipThreshold,
            reconThresholdScale: self.reconThresholdScale,
            gop: gop,
            l2Cadence: l2Cadence,
            l1Cadence: l1Cadence,
            l0Cadence: l0Cadence,
            motionMaskingPx: motionMaskingPx,
            smoothL2: smoothL2,
            smoothL1: smoothL1,
            smoothL0: smoothL0,
            temporalLayers: temporalLayers
        )
    }

    public init(width: Int, height: Int, qstep: Int, framerate: Int = 30, zeroThreshold: Int = 3, keyint: Int = 30, sceneChangeThreshold: Int = 500, maxConcurrency: Int = 4, profile: UInt8 = 0x01, skipThreshold: Int = 2, reconThresholdScale: Int = 1, gop: Int = 12, l2Cadence: Int = 4, l1Cadence: Int = 2, l0Cadence: Int = 1, motionMaskingPx: Int = 2, smoothL2: Int = 1, smoothL1: Int = 2, smoothL0: Int = 1, temporalLayers: Int = 1) {
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
        self.skipThreshold = EncoderTuning.shared.skipThreshold ?? skipThreshold
        self.reconThresholdScale = EncoderTuning.shared.reconThresholdScale ?? reconThresholdScale
        self.gop = gop
        self.l2Cadence = l2Cadence
        self.l1Cadence = l1Cadence
        self.l0Cadence = l0Cadence
        self.motionMaskingPx = motionMaskingPx
        self.smoothL2 = smoothL2
        self.smoothL1 = smoothL1
        self.smoothL0 = smoothL0
        self.temporalLayers = temporalLayers
        
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
            skipThreshold: self.skipThreshold,
            reconThresholdScale: self.reconThresholdScale,
            gop: gop,
            l2Cadence: l2Cadence,
            l1Cadence: l1Cadence,
            l0Cadence: l0Cadence,
            motionMaskingPx: motionMaskingPx,
            smoothL2: smoothL2,
            smoothL1: smoothL1,
            smoothL0: smoothL0,
            temporalLayers: temporalLayers
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
            let fileHeader = VEVCFileHeader(width: width, height: height, framerate: framerate, profile: profile, gop: gop, temporalLayers: temporalLayers)
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
    let reconThresholdScale: Int
    let gop: Int
    let l2Cadence: Int
    let l1Cadence: Int
    let l0Cadence: Int
    let motionMaskingPx: Int
    let smoothL2: Int
    let smoothL1: Int
    let smoothL0: Int
    let temporalLayers: Int
    
    private var rateController: RateController
    private var framesSinceKeyframe = 0
    private var framesSinceLtrUpdate = 0
    private var frameIndex = 0
    private var temporalFrameIndex = 0
    private var qt: QuantizationTable?
    
    private var previousInputPlane: PlaneData420?
    private var releasePreviousInput: (@Sendable () -> Void)?
    private var previousT0InputPlane: PlaneData420?
    private var releasePreviousT0Input: (@Sendable () -> Void)?
    private var previousMVs: MotionVectors?
    private var previousT0MVs: MotionVectors?
    
    private var firstReconstructed: PlaneData420?
    private var releaseFirstRecon: (@Sendable () -> Void)?
    
    var previousReconstructed: PlaneData420? // internal for drift diagnostics
    private var releasePreviousRecon: (@Sendable () -> Void)?
    private var previousT0Reconstructed: PlaneData420?
    private var releasePreviousT0Recon: (@Sendable () -> Void)?
    
    private var firstInputPlane: PlaneData420?
    private var releaseFirstInput: (@Sendable () -> Void)?
    private var staticCounters: [Int] = []
    private var cachedNextSub2: [Int16]?
    private var cachedNextSub1: [Int16]?
    var entropyHistories: FrameEntropyHistories? // internal for history-consistency gate tests
    var mvPayloadHistory: MVPayloadHistory?
    // Quarter-resolution L0 reference chain (One-Pyramid §4, profile 0x02).
    // Internal so the L0 bit-exactness gate tests can compare chains.
    let l0State = L0RefState()
    private var consecutiveCopyFrames = 0
    private var sadBaseline: Int?

    internal init(width: Int, height: Int, maxbitrate: Int, framerate: Int, zeroThreshold: Int, keyint: Int, sceneChangeThreshold: Int, pool: BlockViewPool, qstep: Int? = nil, profile: UInt8 = 0x01, skipThreshold: Int = 2, reconThresholdScale: Int = 1, gop: Int = 12, l2Cadence: Int = 4, l1Cadence: Int = 2, l0Cadence: Int = 1, motionMaskingPx: Int = 2, smoothL2: Int = 1, smoothL1: Int = 2, smoothL0: Int = 1, temporalLayers: Int = 1) {
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
        self.reconThresholdScale = reconThresholdScale
        self.gop = gop
        self.l2Cadence = l2Cadence
        self.l1Cadence = l1Cadence
        self.l0Cadence = l0Cadence
        self.motionMaskingPx = motionMaskingPx
        self.smoothL2 = smoothL2
        self.smoothL1 = smoothL1
        self.smoothL0 = smoothL0
        self.temporalLayers = temporalLayers
        self.framesSinceLtrUpdate = 0
        self.rateController = RateController(maxbitrate: maxbitrate, framerate: framerate, keyint: keyint)
        
        let bw = (width + 31) / 32
        let bh = (height + 31) / 32
        self.staticCounters = [Int](repeating: 0, count: bw * bh)
    }
    
    public init(width: Int, height: Int, maxbitrate: Int, framerate: Int, zeroThreshold: Int, keyint: Int, sceneChangeThreshold: Int, profile: UInt8 = 0x01, skipThreshold: Int = 2, reconThresholdScale: Int = 1, gop: Int = 12, l2Cadence: Int = 4, l1Cadence: Int = 2, l0Cadence: Int = 1, motionMaskingPx: Int = 2, smoothL2: Int = 1, smoothL1: Int = 2, smoothL0: Int = 1, temporalLayers: Int = 1) {
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
        self.reconThresholdScale = reconThresholdScale
        self.gop = gop
        self.l2Cadence = l2Cadence
        self.l1Cadence = l1Cadence
        self.l0Cadence = l0Cadence
        self.motionMaskingPx = motionMaskingPx
        self.smoothL2 = smoothL2
        self.smoothL1 = smoothL1
        self.smoothL0 = smoothL0
        self.temporalLayers = temporalLayers
        self.framesSinceLtrUpdate = 0
        self.rateController = RateController(maxbitrate: maxbitrate, framerate: framerate, keyint: keyint)
        
        let bw = (width + 31) / 32
        let bh = (height + 31) / 32
        self.staticCounters = [Int](repeating: 0, count: bw * bh)
    }
    
    deinit {
        releasePreviousInput?()
        releasePreviousT0Input?()
        releasePreviousRecon?()
        releasePreviousT0Recon?()
        releaseFirstRecon?()
        releaseFirstInput?()
    }
    
    @inline(__always)
    public func encodeFrame(image: YCbCrImage, forceKeyFrame: Bool = false) async throws -> [UInt8] {
        let (plane, releasePlane) = toPlaneData420(image: image, pool: pool)
        
        var isSceneChange = false
        var fastSADToPrevInput = Int.max
        if let prev = previousInputPlane {
            let sad = estimateFastSAD(a: plane, b: prev)
            fastSADToPrevInput = sad
            if profile == 0x02 {
                let baseline: Int
                if let cur = sadBaseline {
                    baseline = cur
                } else {
                    baseline = sad
                }
                let exceedsBaseline = (baseline * sceneCutBaselineRatio) < sad
                isSceneChange = (sceneChangeThreshold < sad) && exceedsBaseline
                if isSceneChange != true {
                    if sceneChangeThreshold <= maxEstimateFastSAD && sceneCutMinLumaMAD <= sad && exceedsBaseline {
                        isSceneChange = detectSceneCut(source: plane.y, reference: prev.y, width: plane.width, height: plane.height)
                    }
                }
                if isSceneChange != true {
                    if let cur = sadBaseline {
                        sadBaseline = ((cur * 7) + sad) / 8
                    } else {
                        sadBaseline = sad
                    }
                }
            } else {
                isSceneChange = (sceneChangeThreshold < sad)
                // Sign-mix cut detector: a threshold above maxEstimateFastSAD
                // means scene detection is intentionally off (deterministic
                // tests); the fastSAD gate keeps the extra luma pass off normal
                // frames.
                if isSceneChange != true && sceneChangeThreshold <= maxEstimateFastSAD && sceneCutMinLumaMAD <= sad {
                    isSceneChange = detectSceneCut(source: plane.y, reference: prev.y, width: plane.width, height: plane.height)
                }
            }
        }
        
        var forceIFrame = forceKeyFrame
        if forceIFrame != true && rateController.isDriftAccelerating(framesSinceKeyframe: framesSinceKeyframe) {
            forceIFrame = true
        }
        
        var isStaticRefresh = false
        if self.qstep == nil {
            isStaticRefresh = rateController.shouldRefreshStaticScene(framesSinceKeyframe: framesSinceKeyframe)
        }
        
        if forceIFrame != true && isStaticRefresh {
            forceIFrame = true
        }
        
        let isPeriodicIFrame = (keyint <= framesSinceKeyframe || frameIndex == 0 || forceIFrame)
        let isIFrame = (isPeriodicIFrame || isSceneChange)
        
        if isIFrame {
            // Rate control
            let baseStep: Int
            if let fixedStep = self.qstep {
                baseStep = fixedStep
                self.qt = QuantizationTable(baseStep: max(16, baseStep), isChroma: false, layerIndex: 0)
            } else {
                let targetBits = rateController.beginGOP()
                let baseQt = estimateQuantization(img: image, targetBits: targetBits, rateController: rateController)
                let loopStep = rateController.calculateIFrameQStep(targetBits: targetBits, estimatedStep: Int(baseQt.step))
                self.qt = QuantizationTable(baseStep: max(16, loopStep), isChroma: false, layerIndex: 0)
                // Quality-ceiling floor for the I-frame coding step,
                // rate-scaled: 512 (real step 32) at 500kbps, relaxing
                // inversely with bitrate so higher-rate operating points
                // keep their finer I-frames. self.qt keeps the unfloored
                // estimate, so P-frame budgeting is untouched and the bits
                // the I-frame does not spend flow to the GOP's P frames.
                // Fixed-qstep mode (explicit quality request) is never
                // floored.
                let scaledFloor = min(iFrameQStepFloorQ4, (iFrameQStepFloorQ4 * 500_000) / max(1, maxbitrate))
                baseStep = max(loopStep, scaledFloor)
            }
            
            // A cut-driven I-frame keeps the periodic keyint grid: the
            // counter restarts mid-phase so the next periodic I still lands
            // on the original frameIndex % keyint boundary. Without this the
            // whole downstream I-grid shifts with every cut, which moves
            // the reference distance of unrelated later frames (measured
            // min-SSIM −0.05 on miko when the flash plateau lost its
            // adjacent I). staticCounters start from the same base so the
            // skip_ltr eligibility (staticCounters[i] == gopPosition) keeps
            // meaning "static since this I-frame".
            framesSinceKeyframe = isPeriodicIFrame ? 0 : (frameIndex % keyint)
            framesSinceLtrUpdate = framesSinceKeyframe
            consecutiveCopyFrames = 0
            if temporalLayers == 2 {
                temporalFrameIndex = 0
            }

            // Backward-adaptive entropy tables: random-access boundary reset.
            if profile == 0x02 {
                if entropyHistories == nil { entropyHistories = FrameEntropyHistories() }
                entropyHistories?.reset()
                if mvPayloadHistory == nil { mvPayloadHistory = MVPayloadHistory() }
                mvPayloadHistory?.reset()
            }

            let qtY = QuantizationTable(baseStep: max(16, baseStep), isChroma: false, layerIndex: 0)
            let qtC = QuantizationTable(baseStep: max(16, baseStep), isChroma: true, layerIndex: 0)
            
            let (bytes, reconstructed, mvs, _, releaseRecon): ([UInt8], PlaneData420, MotionVectors, [Int], @Sendable () -> Void)
            if profile == 0x02 {
                (bytes, reconstructed, mvs, _, releaseRecon) = try await encodeSpatialLayersIntraForProfile2(
                    pd: plane, pool: pool, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold, l0State: l0State
                )
            } else {
                (bytes, reconstructed, mvs, _, releaseRecon) = try await encodeSpatialLayersIntra(
                    pd: plane, pool: pool, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold
                )
            }
            
            if self.qstep == nil {
                rateController.consumeIFrame(bits: bytes.count * 8, qStep: Int(qtY.step))
            }
            
            // Clean up old state
            releasePreviousInput?()
            releasePreviousT0Input?()
            releasePreviousRecon?()
            releasePreviousT0Recon?()
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

            if temporalLayers == 2 {
                previousT0InputPlane = firstIn
                previousT0Reconstructed = reconstructed
                previousT0MVs = mvs
            }
            
            for i in 0..<self.staticCounters.count {
                self.staticCounters[i] = framesSinceKeyframe
            }
            
            previousMVs = mvs
            
            framesSinceKeyframe += 1
            frameIndex += 1
            if temporalLayers == 2 {
                temporalFrameIndex += 1
            }
            
            return bytes
        }
        
        // Near-duplicate frame detection: emit a CopyFrame when the input is
        // within copyFrameMADLimitQ8 of the LAST CODED input (previousInputPlane
        // is not updated on copy, so chained copies keep measuring drift
        // against the anchor and the chain breaks once cumulative change
        // exceeds the bound). Calibrated on miko_700: at 96/256 the worst
        // affected pair SSIM is 0.9947. consecutiveCopyFrames caps a chain at
        // maxConsecutiveCopyFrames (~83ms at 60fps) so sub-threshold motion
        // such as HUD ticks cannot freeze indefinitely.
        // The sampled fast SAD is a free prefilter: every calibrated
        // MAD-eligible pair measured fastSAD == 0, so nonzero fast SAD can
        // never be a near-duplicate and skips the scans. Exact identity
        // (memcmp speed, common for pulldown/static content) is tried before
        // the accumulate-to-bound MAD scan.
        var allowCopyFrame = true
        if temporalLayers == 2 {
            if temporalFrameIndex % 2 == 0 {
                allowCopyFrame = false
            }
        }
        if allowCopyFrame && consecutiveCopyFrames < maxConsecutiveCopyFrames, let prevIn = previousInputPlane, fastSADToPrevInput == 0, isPlaneIdentical(a: plane, b: prevIn) || isNearDuplicate(a: plane, b: prevIn, limitQ8: copyFrameMADLimitQ8) {
            releasePlane()

            // A copied frame is static by definition: advance the per-block
            // static counters together with the GOP position, or the
            // skip_ltr eligibility (staticCounters[i] == ltrAge) breaks
            // for the rest of the GOP. The LTR pixel match itself is
            // re-verified on every coded frame, so this cannot fabricate a
            // false skip_ltr.
            for i in staticCounters.indices {
                staticCounters[i] += 1
            }

            consecutiveCopyFrames += 1
            let bytes = VEVCFrameHeader(frameType: .copyFrame, hasRefDir: false, skipMapSize: 0, mvsSize: 0, refDirSize: 0, lumaOffset: 0, chromaOffset: 0, layer0Size: 0, layer1Size: 0, layer2Size: 0).serialize()
            // Route the unspent frame budget to the rest of the GOP: the
            // remaining coded frames inherit it and the rate controller can
            // lower their qstep (min-SSIM support).
            if self.qstep == nil {
                rateController.consumeCopyFrame(bits: bytes.count * 8)
            } else {
                rateController.resetStaticStreak()
            }

            var canPromoteLTR = false
            if profile == 0x02 {
                if 0 < gop {
                    if 0 < framesSinceKeyframe {
                        if framesSinceKeyframe % gop == 0 {
                            if temporalLayers == 2 {
                                if temporalFrameIndex % 2 == 0 {
                                    canPromoteLTR = true
                                }
                            } else {
                                canPromoteLTR = true
                            }
                        }
                    }
                }
            }

            if canPromoteLTR {
                releaseFirstRecon?()
                firstReconstructed = previousReconstructed
                releaseFirstRecon = releasePreviousRecon
                releasePreviousRecon = nil

                releaseFirstInput?()
                let (firstIn, releaseFirstIn) = toPlaneData420(image: image, pool: pool)
                firstInputPlane = firstIn
                releaseFirstInput = releaseFirstIn

                if temporalLayers == 2 {
                    releasePreviousT0Recon?()
                    previousT0Reconstructed = previousReconstructed
                    releasePreviousT0Recon = nil

                    releasePreviousT0Input?()
                    previousT0InputPlane = firstIn
                    releasePreviousT0Input = nil
                }

                for i in 0..<self.staticCounters.count {
                    self.staticCounters[i] = 0
                }
                framesSinceLtrUpdate = 0
                cachedNextSub2 = nil
                cachedNextSub1 = nil
            } else {
                framesSinceLtrUpdate += 1
                if temporalLayers == 2 {
                    if temporalFrameIndex % 2 == 0 {
                        previousT0Reconstructed = previousReconstructed
                        previousT0InputPlane = previousInputPlane
                    }
                }
            }

            framesSinceKeyframe += 1
            frameIndex += 1
            if temporalLayers == 2 {
                temporalFrameIndex += 1
            }
            return bytes
        }
        consecutiveCopyFrames = 0
        
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
        
        let isT0 = (temporalFrameIndex % 2 == 0)
        let refPrevRecon: PlaneData420
        let refPrevIn: PlaneData420
        let refPrevMVs: MotionVectors?
        if temporalLayers == 2 {
            guard let t0Recon = previousT0Reconstructed, let t0In = previousT0InputPlane else {
                throw EncodeError.missingReferenceFramesForPFrame
            }
            refPrevRecon = t0Recon
            refPrevIn = t0In
            refPrevMVs = previousT0MVs
        } else {
            refPrevRecon = prevRecon
            refPrevIn = prevIn
            refPrevMVs = previousMVs
        }

        let updateL0: Bool
        if temporalLayers == 2 {
            updateL0 = isT0
        } else {
            updateL0 = true
        }

        let effSmoothL2: Int
        let effSmoothL1: Int
        let effSmoothL0: Int
        if temporalLayers == 2 && isT0 {
            effSmoothL2 = 0
            effSmoothL1 = 0
            effSmoothL0 = 0
        } else {
            effSmoothL2 = self.smoothL2
            effSmoothL1 = self.smoothL1
            effSmoothL0 = self.smoothL0
        }

        // The two P-frame pipelines are separate functions so each stays
        // branch-free; the profile decides here, once.
        let encoded: ([UInt8], PlaneData420, MotionVectors, [Int], @Sendable () -> Void, [Int16], [Int16])
        let interRatioQ8: Int
        let detailThinned: Bool
        if profile == 0x02 {
            var localCounters = self.staticCounters
            let ltrAge = framesSinceLtrUpdate + 1
            let (bytes, recon, mvs, sads, releaseRecon, nSub2, nSub1, skipMap) = try await encodeSpatialLayersForProfile2(
                pd: plane, pool: pool, predictedPd: refPrevRecon, nextPd: firstRecon, prevInput: refPrevIn, ltrInput: firstIn, prevMVs: refPrevMVs,
                maxbitrate: maxbitrate, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold,
                roundOffset: framesSinceKeyframe % 2, gopPosition: framesSinceKeyframe, ltrAge: ltrAge, skipThreshold: self.skipThreshold, reconThresholdScale: self.reconThresholdScale, staticCounters: &localCounters,
                cachedNextSub2: self.cachedNextSub2, cachedNextSub1: self.cachedNextSub1,
                entropyHistories: self.entropyHistories,
                mvPayloadHistory: self.mvPayloadHistory,
                l0State: l0State,
                l2Cadence: self.l2Cadence,
                l1Cadence: self.l1Cadence,
                l0Cadence: self.l0Cadence,
                framerate: self.framerate,
                motionMaskingPx: self.motionMaskingPx,
                adjustedStep: adjustedStep,
                smoothL2: effSmoothL2,
                smoothL1: effSmoothL1,
                smoothL0: effSmoothL0,
                updateL0Prev: updateL0
            )
            self.staticCounters = localCounters
            encoded = (bytes, recon, mvs, sads, releaseRecon, nSub2, nSub1)

            var interCount = 0
            for mode in skipMap {
                if mode == .inter {
                    interCount += 1
                }
            }
            interRatioQ8 = (interCount * 256) / skipMap.count
            detailThinned = shouldZeroCadence(cadence: self.l2Cadence, gopPosition: framesSinceKeyframe) || shouldZeroCadence(cadence: self.l1Cadence, gopPosition: framesSinceKeyframe) || shouldZeroCadence(cadence: self.l0Cadence, gopPosition: framesSinceKeyframe)
        } else {
            encoded = try await encodeSpatialLayers(
                pd: plane, pool: pool, predictedPd: refPrevRecon, nextPd: firstRecon, prevInput: refPrevIn, ltrInput: firstIn, prevMVs: refPrevMVs,
                maxbitrate: maxbitrate, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold,
                roundOffset: framesSinceKeyframe % 2, gopPosition: framesSinceKeyframe,
                cachedNextSub2: self.cachedNextSub2, cachedNextSub1: self.cachedNextSub1
            )
            interRatioQ8 = 0
            detailThinned = false
        }
        let (bytes, reconstructed, mvs, sads, releaseRecon, nSub2, nSub1) = encoded
        self.cachedNextSub2 = nSub2
        self.cachedNextSub1 = nSub1
        
        // Using masked recon distortion for quality metric
        let safeRecon = SafeReleaseAction(releaseRecon)
        let safePlane = SafeReleaseAction(releasePlane)
        let safeReleaseRecon: @Sendable () -> Void = { safeRecon.release() }
        let safeReleasePlane: @Sendable () -> Void = { safePlane.release() }

        let reconDistortion = computeMaskedReconDistortion(original: plane, reconstructed: reconstructed, sads: sads)
        
        if self.qstep == nil {
            rateController.consumePFrame(bits: bytes.count * 8, qStep: Int(adjustedStep), sad: frameSAD, distortion: reconDistortion, interRatioQ8: interRatioQ8, detailThinned: detailThinned)
        }
        
        var canPromoteLTR = false
        if profile == 0x02 {
            if 0 < gop {
                if 0 < framesSinceKeyframe {
                    if framesSinceKeyframe % gop == 0 {
                        if temporalLayers == 2 {
                            if isT0 {
                                canPromoteLTR = true
                            }
                        } else {
                            canPromoteLTR = true
                        }
                    }
                }
            }
        }

        if canPromoteLTR {
            releaseFirstRecon?()
            firstReconstructed = reconstructed
            releaseFirstRecon = safeReleaseRecon
            releasePreviousRecon = nil

            releaseFirstInput?()
            let (firstIn, releaseFirstIn) = toPlaneData420(image: image, pool: pool)
            firstInputPlane = firstIn
            releaseFirstInput = releaseFirstIn

            releasePreviousInput?()
            previousInputPlane = plane
            releasePreviousInput = safeReleasePlane

            if temporalLayers == 2 {
                releasePreviousT0Recon?()
                previousT0Reconstructed = reconstructed
                releasePreviousT0Recon = nil

                releasePreviousT0Input?()
                previousT0InputPlane = plane
                releasePreviousT0Input = nil
            }

            for i in 0..<self.staticCounters.count {
                self.staticCounters[i] = 0
            }
            framesSinceLtrUpdate = 0
            cachedNextSub2 = nil
            cachedNextSub1 = nil
        } else {
            framesSinceLtrUpdate += 1

            let oldRecon = previousReconstructed!
            let oldRelease = releasePreviousRecon
            
            let isPrevFirst = withUnsafePointers(oldRecon.y, firstRecon.y) { p, f in
                p == f
            }
            let isPrevT0: Bool
            if temporalLayers == 2, let t0 = previousT0Reconstructed {
                isPrevT0 = withUnsafePointers(oldRecon.y, t0.y) { p, t in
                    p == t
                }
            } else {
                isPrevT0 = false
            }
            if isPrevFirst != true && isPrevT0 != true {
                oldRelease?()
            }
            
            releasePreviousRecon = safeReleaseRecon

            releasePreviousInput?()
            previousInputPlane = plane
            releasePreviousInput = safeReleasePlane

            if temporalLayers == 2 {
                if isT0 {
                    if let oldT0Recon = previousT0Reconstructed {
                        let oldT0Release = releasePreviousT0Recon
                        let isT0First = withUnsafePointers(oldT0Recon.y, firstRecon.y) { p, f in
                            p == f
                        }
                        let isT0Prev = withUnsafePointers(oldT0Recon.y, oldRecon.y) { p, o in
                            p == o
                        }
                        if isT0First != true && isT0Prev != true {
                            oldT0Release?()
                        }
                    }
                    previousT0Reconstructed = reconstructed
                    releasePreviousT0Recon = safeReleaseRecon

                    releasePreviousT0Input?()
                    previousT0InputPlane = plane
                    releasePreviousT0Input = safeReleasePlane
                }
            }
        }

        previousReconstructed = reconstructed
        previousMVs = mvs
        if temporalLayers == 2 {
            if isT0 {
                previousT0MVs = mvs
            }
        }
        
        framesSinceKeyframe += 1
        frameIndex += 1
        if temporalLayers == 2 {
            temporalFrameIndex += 1
        }
        
        return bytes
    }
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

/// Encoder tuning knobs. The singleton captures every environment override
/// exactly once at init — environment variables are constant for the process
/// lifetime, so no call site re-reads them. Values with built-in defaults
/// are plain Ints; values whose default belongs to the caller (init
/// parameters like skipThreshold) are optionals, consumed as
/// `EncoderTuning.shared.x ?? callerDefault`.
public struct EncoderTuning: @unchecked Sendable {
    public static let shared = EncoderTuning()

    public let l0LumaThresholdPFrame: Int
    public let l0ChromaThresholdScale: Int
    public let iFrameQuantizationScale: Int
    public let l16LumaThreshold: Int
    public let l32LumaThreshold: Int
    /// VEVC_SKIP_THRESH: overrides the caller's skipThreshold when set.
    public let skipThreshold: Int?
    /// VEVC_RECON_THRESH_SCALE: overrides the caller's reconThresholdScale.
    public let reconThresholdScale: Int?
    /// σ-normalized AQ knobs (SAD.swift defaults; VEVC_AQ_BIAS=0 disables —
    /// the flat/textured quantizer variants collapse to qMid/qHigh).
    public let aqFlatVarianceMax: Int
    public let aqTexturedVarianceMin: Int
    public let aqBiasDeltaQ16: Int

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
        self.skipThreshold = Self.envIntOptional(key: "VEVC_SKIP_THRESH")
        self.reconThresholdScale = Self.envIntOptional(key: "VEVC_RECON_THRESH_SCALE")
        self.aqFlatVarianceMax = Self.envInt(key: "VEVC_AQ_FLAT", defaultValue: aqFlatVarianceMaxDefault)
        self.aqTexturedVarianceMin = Self.envInt(key: "VEVC_AQ_TEX", defaultValue: aqTexturedVarianceMinDefault)
        self.aqBiasDeltaQ16 = Self.envInt(key: "VEVC_AQ_BIAS", defaultValue: aqBiasDeltaQ16Default)
    }

    @inline(__always)
    private static func envInt(key: String, defaultValue: Int) -> Int {
        envIntOptional(key: key) ?? defaultValue
    }

    @inline(__always)
    private static func envIntOptional(key: String) -> Int? {
        if let valStr = ProcessInfo.processInfo.environment[key], let val = Int(valStr) {
            return val
        }
        return nil
    }
}

// MARK: - EncoderTuning

// Measurement hooks for encoder tuning. Inactive (nil singleton, no
// per-block work) unless the corresponding environment variable is set.

// Multi-reference skip oracle. Answers, without any bitstream change: if the
// encoder could skip-copy from any of the last N reconstructed frames (not
// just prev/LTR), how many blocks currently coded as inter would become
// skips, and how far back do the matching references sit?
//
// Enabled by VEVC_MULTIREF_ORACLE=<poolSize> (e.g. 15). Per P-frame it tests
// every non-skip 32x32 block against each pooled reconstruction with the
// exact production criterion: zero-MV SAD over the four 16x16 sub-blocks
// (luma + chroma, computeZeroSAD16x16 / computeZeroSADSubBlock) against the
// same skipThreshold-per-pixel budget, requiring a single reference to
// satisfy all four sub-blocks. The pool is cleared at every I-frame so no
// candidate crosses a random-access boundary.
//
// Output: one cumulative "MRORACLE" line per frame on stderr; the last line
// holds the totals. dist1 counts nearest-match age 1 (frames the current
// skip_prev rules missed), dist2_4 / dist5_15 older re-appearances.
// Measured 2026-08-16 (miko/ToS): upgrades ≈0 weighted by bits — negative
// result, multi-reference pools are not worth a bitstream change.
final class MultiRefOracle: @unchecked Sendable {
    static let shared: MultiRefOracle? = {
        guard let v = ProcessInfo.processInfo.environment["VEVC_MULTIREF_ORACLE"], let n = Int(v), 0 < n else { return nil }
        return MultiRefOracle(poolSize: n)
    }()

    private let poolSize: Int
    private let lock = NSLock()
    private var pool: [PlaneData420] = []

    private var frames = 0
    private var totalBlocks = 0
    private var currentSkips = 0
    private var upgrades = 0
    private var dist1 = 0
    private var dist2_4 = 0
    private var dist5_15 = 0

    init(poolSize: Int) {
        self.poolSize = poolSize
    }

    /// I-frame: random-access boundary — no reference crosses it.
    func reset() {
        lock.lock()
        pool.removeAll()
        lock.unlock()
    }

    /// Called with the final reconstruction of every frame (I and P alike),
    /// newest first in the pool.
    func push(recon: PlaneData420) {
        var y = [Int16](repeating: 0, count: recon.y.count)
        var cb = [Int16](repeating: 0, count: recon.cb.count)
        var cr = [Int16](repeating: 0, count: recon.cr.count)
        y.withUnsafeMutableBufferPointer { d in recon.y.withUnsafeBufferPointer { d.baseAddress!.update(from: $0.baseAddress!, count: $0.count) } }
        cb.withUnsafeMutableBufferPointer { d in recon.cb.withUnsafeBufferPointer { d.baseAddress!.update(from: $0.baseAddress!, count: $0.count) } }
        cr.withUnsafeMutableBufferPointer { d in recon.cr.withUnsafeBufferPointer { d.baseAddress!.update(from: $0.baseAddress!, count: $0.count) } }
        let copy = PlaneData420(width: recon.width, height: recon.height, y: y, cb: cb, cr: cr)
        lock.lock()
        pool.insert(copy, at: 0)
        if poolSize < pool.count { pool.removeLast() }
        lock.unlock()
    }

    /// Called on P-frames after the production skipMap is decided.
    func evaluate(pd: PlaneData420, skipMap: [BlockMode], skipThreshold: Int) {
        lock.lock()
        let refs = pool
        lock.unlock()
        guard refs.isEmpty != true else { return }

        let dx = pd.width
        let dy = pd.height
        let bw = (dx + 31) / 32
        var fUpgrades = 0
        var fDist1 = 0
        var fDist2_4 = 0
        var fDist5_15 = 0
        var fSkips = 0

        for i in 0..<skipMap.count {
            if skipMap[i] != .inter {
                fSkips += 1
                continue
            }
            let bx = (i % bw) * 32
            let by = (i / bw) * 32
            var nearest = -1
            for (age0, ref) in refs.enumerated() {
                if blockMatches(cur: pd, ref: ref, bx: bx, by: by, dx: dx, dy: dy, skipThreshold: skipThreshold) {
                    nearest = age0 + 1
                    break
                }
            }
            if 0 < nearest {
                fUpgrades += 1
                switch nearest {
                case 1: fDist1 += 1
                case 2...4: fDist2_4 += 1
                default: fDist5_15 += 1
                }
            }
        }

        lock.lock()
        frames += 1
        totalBlocks += skipMap.count
        currentSkips += fSkips
        upgrades += fUpgrades
        dist1 += fDist1
        dist2_4 += fDist2_4
        dist5_15 += fDist5_15
        let line = "MRORACLE frames=\(frames) blocks=\(totalBlocks) skips=\(currentSkips) upgrades=\(upgrades) dist1=\(dist1) dist2_4=\(dist2_4) dist5_15=\(dist5_15)\n"
        lock.unlock()
        fputs(line, stderr)
    }

    /// Production skip criterion: all four 16x16 sub-blocks of the 32x32
    /// block within skipThreshold-per-pixel SAD (luma + chroma) against a
    /// single reference's reconstruction.
    private func blockMatches(cur: PlaneData420, ref: PlaneData420, bx: Int, by: Int, dx: Int, dy: Int, skipThreshold: Int) -> Bool {
        withUnsafePlanePointers(cur, ref) { c, r in
            for sy in 0..<2 {
                for sx in 0..<2 {
                    let subX = bx + sx * 16
                    let subY = by + sy * 16
                    let mw = min(16, dx - subX)
                    let mh = min(16, dy - subY)
                    if mw <= 0 || mh <= 0 { continue }
                    let mwc = (mw + 1) / 2
                    let mhc = (mh + 1) / 2
                    let area = mw * mh + mwc * mhc * 2
                    let blockThreshold = skipThreshold * area
                    let sad: Int
                    if mw == 16 && mh == 16 && mwc == 8 && mhc == 8 {
                        sad = computeZeroSAD16x16(cY: c.y, rY: r.y, cCb: c.cb, rCb: r.cb, cCr: c.cr, rCr: r.cr, bx: subX, by: subY, width: dx, limit: blockThreshold)
                    } else {
                        sad = computeZeroSADSubBlock(cY: c.y, rY: r.y, cCb: c.cb, rCb: r.cb, cCr: c.cr, rCr: r.cr, bx: subX, by: subY, width: dx, height: dy, subWidth: mw, subHeight: mh, subWc: mwc, subHc: mhc, limit: blockThreshold)
                    }
                    if blockThreshold < sad {
                        return false
                    }
                }
            }
            return true
        }
    }
}
