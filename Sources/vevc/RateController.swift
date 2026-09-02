/// Restricts static scene refreshes to skip-dominated frames. Cheap+distorted frames with many inter blocks are typical under saturation or cadence thinning rather than evidence of a static scene (based on measurements where src_1 blew up to 60.6% I-frames).
let staticInterRatioMaxQ8: Int = 26

struct RateController {
    let baseMaxBitrate: Int
    /// Bitrate the controller actually plans against.
    /// On profile 0x02 the requested bitrate prices the whole stream, every
    /// spatial layer included, so the plan is the request unscaled. The older
    /// 1.3x layer allowance priced `-b` for the full-resolution layer alone and
    /// reserved 30% for the layers below it, but L1+L2 measure 7.0%-10.1% of a
    /// profile-2 stream, so the allowance overshot the request by ~1.3x with no
    /// layer cost to account for it. Profile 0x01 keeps the historical
    /// allowance: its streams and the quality baselines drawn from them are
    /// frozen.
    let plannedBitrate: Int
    /// Whether the I-frame share is priced per coded frame rather than per
    /// frame slot. On for profile 0x02 only: profile 0x01's streams are frozen
    /// on the same grounds as `plannedBitrate` above, and its own pulldown
    /// undershoot is left where it is.
    let pricesCodedFrames: Bool
    let framerate: Int
    let keyint: Int
    let targetDistortionQ8: Int
    
    private(set) var consumedTotalBits: Int = 0
    private(set) var plannedTotalScaled: Int = 0
    
    private(set) var gopTargetBits: Int = 0
    private(set) var gopRemainingBits: Int = 0
    private(set) var gopRemainingFrames: Int = 0
    
    private(set) var lastIFrameBits: Int = 0
    private(set) var lastIFrameQStep: Int = 0

    private(set) var avgPFrameSAD: Int = 0
    private(set) var lastPFrameBits: Int = 0
    private(set) var lastPFrameQStep: Int = 0
    private(set) var lastPFrameSAD: Int = 0
    
    private(set) var avgPFrameBits: Int = 0
    private(set) var staticStreak: Int = 0
    
    // Reconstruction distortion tracking for quality-consistent QP adjustment.
    // avgDistortionQ8: EMA of per-pixel reconstruction SAD in Q8 (target quality level)
    // lastDistortionQ8: previous frame's per-pixel reconstruction SAD in Q8
    private(set) var avgDistortionQ8: Int = 0
    private(set) var lastDistortionQ8: Int = 0
    
    // Closed-loop rate correction gain in Q8 fixed-point (256 = 1.0).
    // actual/budget consumption ratio of past GOPs, tracked by EMA.
    private(set) var rateGainQ8: Int = 256
    
    // budgetRatioQ8: Consumption pace within GOP (larger value means more budget remaining)
    private(set) var budgetRatioQ8: Int = 64
    
    // pPaceRatioQ8: Consumption pace specific to P-frames (relative to after I-frame consumption)
    private(set) var pPlanBits: Int = 0
    private(set) var pPlanFrames: Int = 0
    private(set) var pPaceRatioQ8: Int = 256
    
    // isQualitySaturated: Unified state for quality saturation (D*)
    private(set) var isQualitySaturated: Bool = false
    
    private(set) var saturationAnchorStep: Int = 0
    private(set) var driftStreak: Int = 0

    // Frame slots and coded frames seen since the current GOP opened, and the
    // ratio the previous GOP realized (Q8). A CopyFrame occupies a slot but
    // spends only its 64-bit header, so the number of frames a GOP codes is
    // not `keyint` on duplicated input - see `beginGOP` for what reads this.
    private(set) var gopSlots: Int = 0
    private(set) var gopCodedFrames: Int = 0
    private(set) var prevGopCodedRatioQ8: Int = 256

    /// Single distortion spikes are noise or normal fluctuations; requires 2 consecutive frames of sustained distortion plus an in-GOP position guard to force an I-frame only during runaway drift (based on measurements where drift-I blew up to 5,287 frames = 61% on src_1). Real scene changes are caught by the input-based cut detector.
    @inline(__always)
    func isDriftAccelerating(framesSinceKeyframe: Int) -> Bool {
        let limit = min(8, self.keyint / 2)
        if 2 <= self.driftStreak && limit <= framesSinceKeyframe {
            return true
        }
        return false
    }

    /// `profile` selects how the requested bitrate is priced; it defaults to
    /// 0x01 to match `VEVCEncoder`'s own default, so a caller that does not
    /// name a profile keeps the historical allowance.
    init(maxbitrate: Int, framerate: Int, keyint: Int, profile: UInt8 = 0x01, targetDistortion: Int = 600) {
        self.baseMaxBitrate = maxbitrate
        self.plannedBitrate = (profile == 0x02) ? maxbitrate : (maxbitrate * 13) / 10
        self.pricesCodedFrames = (profile == 0x02)
        self.framerate = framerate
        self.keyint = keyint
        self.targetDistortionQ8 = targetDistortion
    }

    /// Records whether the frame just handed to the controller was coded or
    /// copied. Read once per GOP, in `beginGOP`.
    @inline(__always)
    private mutating func noteFrameCoded(_ coded: Bool) {
        self.gopSlots += 1
        if coded {
            self.gopCodedFrames += 1
        }
    }

    @inline(__always)
    private mutating func updateSaturationState() {
        if self.isQualitySaturated {
            if (self.targetDistortionQ8 * 5) / 4 < self.avgDistortionQ8 {
                self.isQualitySaturated = false
                self.saturationAnchorStep = 0
            }
        } else {
            if 0 < self.avgDistortionQ8 && self.avgDistortionQ8 < self.targetDistortionQ8 && 230 <= self.pPaceRatioQ8 {
                self.isQualitySaturated = true
                self.saturationAnchorStep = max(16, self.lastPFrameQStep)
            }
        }
    }
    
    @inline(__always)
    mutating func beginGOP() -> Int {
        // Close the GOP that just ended: its coded-frame share is the estimate
        // the next GOP's I-frame price is drawn from. Clamped to [1/8, 1] so a
        // pathological run of copies cannot blow the divisor up.
        if 0 < self.gopSlots {
            self.prevGopCodedRatioQ8 = max(32, min(256, (self.gopCodedFrames * 256) / self.gopSlots))
        }
        self.gopSlots = 0
        self.gopCodedFrames = 0

        // Update closed-loop gain from the previous GOP's actual consumption.
        // Scene-change may cut a GOP short, so compare against the budget
        // prorated by the number of frames actually consumed.
        let framesUsed = self.keyint - self.gopRemainingFrames
        if 0 < self.gopTargetBits && 0 < framesUsed {
            let expected = max(1, (self.gopTargetBits * framesUsed) / self.keyint)
            let consumed = self.gopTargetBits - self.gopRemainingBits
            if 0 < consumed {
                let instantQ8 = (consumed << 8) / expected
                let clamped = max(64, min(2048, instantQ8))   // [0.25x, 8.0x]
                // EMA: gain = gain * 0.75 + instant * 0.25
                self.rateGainQ8 = max(64, min(2048, ((self.rateGainQ8 * 3) + clamped) / 4))
            }
        }
        
        let baseGOPBits = (self.plannedBitrate * self.keyint) / self.framerate
        let balance = (self.plannedTotalScaled - (self.consumedTotalBits * self.framerate)) / self.framerate
        var targetBits = baseGOPBits + balance
        if self.isQualitySaturated {
            targetBits = min(targetBits, baseGOPBits)
        }
        self.gopTargetBits = max(baseGOPBits / 4, min(baseGOPBits * 2, targetBits))
        self.gopRemainingBits = self.gopTargetBits
        self.gopRemainingFrames = self.keyint
        
        let absoluteFloor = (self.plannedBitrate * 6) / self.framerate
        // The I frame is priced at five coded frames' worth of the GOP budget:
        // (coded - 1) P frames plus its own 5 shares. Written against `keyint`
        // the denominator counts frame slots, and a CopyFrame slot costs 64
        // bits, so on duplicated input (2:3 pulldown measures 41% coded) the I
        // frame is priced at 41% of what the GOP can afford. calculateIFrameQStep
        // then chases that target across GOPs - measured on 1920x800 ToS
        // duplicated to 60fps, the I step ratcheted 512 -> 1261 -> 1805 -> ...
        // until it pinned at the quantization cap 4096, and because P frames
        // may not quantize finer than the I frame, the whole stream ran at the
        // coarsest step available while spending 47% of the requested rate.
        // The previous GOP's realized coded-frame share is the estimate: it is
        // exact rather than smoothed, and it is a per-GOP quantity for what is
        // a per-GOP decision.
        let iShareFrames = self.pricesCodedFrames ? max(1, (self.keyint * self.prevGopCodedRatioQ8) / 256) : self.keyint
        let iFrameBitsProp = (self.gopTargetBits * 5) / (iShareFrames + 4)
        return max(1000, max(absoluteFloor, iFrameBitsProp))
    }
    
    @inline(__always)
    mutating func consumeIFrame(bits: Int, qStep: Int) {
        self.lastIFrameBits = bits
        self.lastIFrameQStep = qStep

        self.consumedTotalBits += bits
        self.plannedTotalScaled += self.plannedBitrate
        
        self.gopRemainingBits -= bits
        self.gopRemainingFrames -= 1
        
        self.pPlanBits = self.gopRemainingBits
        self.pPlanFrames = self.gopRemainingFrames
        
        self.staticStreak = 0
        self.driftStreak = 0
        self.noteFrameCoded(true)
    }

    /// Closed-loop I-frame step: proportional prediction from the previous
    /// I-frame's (bits, step) pair, the same model calculatePFrameQStep uses
    /// for P-frames. estimateQuantization's open-loop prediction misses by
    /// up to 5x on live content (measured 332KB against a 65KB budget at
    /// step 64); one previous sample corrects it. The open-loop estimate is
    /// unreliably fine at stream start (measured ~8 at -b 4000 where the operative
    /// value was the rate-scaled floor 64); a conservative absolute seed costs
    /// 0.5s of startup quality and the closed loop corrects from the second I-frame.
    /// A scene-cut I-frame mispredicts once (different content) and recovers on the
    /// next sample.
    @inline(__always)
    func calculateIFrameQStep(targetBits: Int, estimatedStep: Int) -> Int {
        if lastIFrameBits <= 0 || lastIFrameQStep <= 0 {
            return min(16384, max(512, estimatedStep))
        }
        let val = (Int64(lastIFrameQStep) * Int64(lastIFrameBits)) / Int64(max(1, targetBits))
        return Int(max(16, min(16384, val)))
    }
    
    @inline(__always)
    mutating func calculatePFrameQStep(currentSAD: Int, baseStep: Int) -> Int {
        if self.avgPFrameSAD == 0 { self.avgPFrameSAD = currentSAD }
        // EMA: avg = avg * 0.8 + current * 0.2 → (avg * 4 + current) / 5
        self.avgPFrameSAD = ((self.avgPFrameSAD * 4) + currentSAD) / 5
        
        // Floor under a P-frame's bit target so a starved GOP cannot collapse
        // P-frame quality entirely. Written as a flat 2% of the GOP budget it
        // is keyint-proportional once read per frame:
        // (gopTargetBits / 50) / (gopTargetBits / keyint) == keyint / 50.
        // Neutral at the keyint of 30-50 it was tuned for, it becomes 2.4x the
        // GOP's own average per-frame share at the profile-2 default keyint of
        // 120 and 4.8x at 240, so the floor outranks the pacing term below it
        // on every P frame (measured: 100% of P-frame decisions at keyint 120
        // and 240 against 18.3% at keyint 30) and the controller loses its only
        // way to tighten. Capping the divisor at keyint holds the floor at or
        // under one frame's nominal share: keyint <= 50 stays bit-exact and the
        // pacing term governs above it.
        let fallbackBits = gopTargetBits / max(50, keyint)
        let avgBitsPerFrame = max(fallbackBits, gopRemainingBits / max(1, gopRemainingFrames))
        
        // Weight by activity variation: multiplier = currentSAD / avgPFrameSAD
        // Use Q16 fixed-point for multiplier: multiplier16 = (currentSAD << 16) / avgPFrameSAD
        let safeAvg = max(1, self.avgPFrameSAD)
        let multiplier16 = (Int64(currentSAD) << 16) / Int64(safeAvg)
        
        // Clamp multiplier to [0.2, 5.0] in Q16: [13107, 327680]
        let clampedMul16 = max(13107, min(327680, multiplier16))
        let targetFrameBits = Int((Int64(avgBitsPerFrame) * clampedMul16) >> 16)
        
        let plannedRemaining = max(1, (gopTargetBits * gopRemainingFrames) / max(1, keyint))
        
        if gopRemainingBits < 0 {
            self.budgetRatioQ8 = 64
        } else {
            self.budgetRatioQ8 = max(64, min(1024, (gopRemainingBits << 8) / plannedRemaining))
        }
        let budgetRatioQ8 = self.budgetRatioQ8
        
        let plannedP = max(1, (self.pPlanBits * self.gopRemainingFrames) / max(1, self.pPlanFrames))
        self.pPaceRatioQ8 = max(0, min(1024, (self.gopRemainingBits << 8) / plannedP))

        // maxStep scales proportionally to baseStep: P-Frame worst-case quality
        // tracks I-Frame quality level. At high bitrates (low baseStep), this
        // prevents P-Frames from degrading to poor quality even if bits are tight.
        // Allowing maxStep to drop below 64 prevents the SSIM inversion bug.
        var maxStep = max(baseStep * 2, min(8192, baseStep * 4))
        
        var newStepInt = (baseStep * 3) / 2
        // P-Frame QP floor: baseStep ensures P-Frames never use finer
        // quantization than the I-Frame. At high bitrates (baseStep=16-32),
        // this allows near-lossless P-frame quality.
        var minStep = max(16, baseStep)

        switch true {
        case budgetRatioQ8 < 192:
            // Budget tight (<0.75): release the baseStep anchor entirely — the
            // quantization table clamps at 4096 (Q4), so 16384 here means "as
            // coarse as the format allows". Anchoring to baseStep kept P-frames
            // finer than the budget demanded (measured 2.4x overshoot at -b 4000).
            maxStep = 16384
        case 320 < budgetRatioQ8:
            // Budget surplus (>1.25): Allow finer quantization than I-frame
            minStep = max(16, (baseStep * 3) / 4)
        default: break
        }

        // Quality ceiling by Distortion target D*
        if self.isQualitySaturated && 0 < self.saturationAnchorStep {
            let safeAvg = max(1, self.avgDistortionQ8)
            let ratioQ8 = min(512, (self.targetDistortionQ8 * 256) / safeAvg)
            let qualityFloor = (self.saturationAnchorStep * ratioQ8) / 256
            minStep = max(minStep, min(qualityFloor, self.saturationAnchorStep * 2))
        }
        
        if 0 < lastPFrameBits && 0 < lastPFrameQStep && 0 < lastPFrameSAD {
            // Predict the amount of bits we'd get if we used the same Q as last P-frame
            // The bits should scale with SAD relative to the last frame, NOT the average.
            let SADRatio16 = (Int64(currentSAD) << 16) / Int64(lastPFrameSAD)
            let clampedSadRatio16 = max(13107, min(327680, SADRatio16))
            let predictedBits64 = (Int64(lastPFrameBits) * clampedSadRatio16) >> 16
            
            // ratio = predictedCurrentBits / targetFrameBits
            let safeTarget = max(1, targetFrameBits)
            // val = lastPFrameQStep * ratio
            let val = (Int64(lastPFrameQStep) * predictedBits64) / Int64(safeTarget)
            
            newStepInt = Int(max(Int64(minStep), min(Int64(maxStep), val)))
        }
        
        // Distortion feedback: adjust QP based on actual reconstruction quality.
        // If the previous frame had higher-than-average distortion (poor quality),
        // reduce QP to improve quality. If it had lower-than-average distortion
        // (good quality), allow QP to increase.
        // This is content-adaptive: no fixed parameters, responds to actual quality.
        // Half-strength blending: apply only 50% of the correction to avoid
        // over-reacting and causing excessive size increase.
        if 256 <= budgetRatioQ8 && 0 < lastDistortionQ8 && 0 < avgDistortionQ8 {
            // fullCorrection = newStep * avgDistortion / lastDistortion
            // blended = (newStep + fullCorrection) / 2 → 50% correction strength
            let fullCorrection = (Int64(newStepInt) * Int64(avgDistortionQ8)) / Int64(lastDistortionQ8)
            let blended = (Int64(newStepInt) + fullCorrection) / 2
            newStepInt = Int(max(Int64(minStep), min(Int64(maxStep), blended)))
        }
        
        // Adaptive EMA Smoothing: blend new QP with previous QP based on scene stability.
        // sceneSadRatio = currentSAD / avgPFrameSAD:
        //   ratio ≈ 1.0: scene is stable → strong smoothing (favor previous QP)
        //   ratio >> 1.0: scene is changing → weak smoothing (follow new QP)
        //
        // alpha = clamp(sceneSadRatio, 0.3, 1.0) in Q16
        // finalQP = alpha * newQP + (1-alpha) * lastQP
        //
        // This has no fixed parameters — the smoothing strength adapts to content.
        if 256 <= budgetRatioQ8 && 0 < lastPFrameQStep && 0 < avgPFrameSAD {
            let sceneSADRatio16 = (Int64(currentSAD) << 16) / Int64(max(1, avgPFrameSAD))
            // When transitioning to static scene (sceneSadRatio16 < 32768), use higher alpha to drop QP faster.
            var minAlpha16: Int64 = 19661 // 0.3
            if sceneSADRatio16 < 32768 {
                minAlpha16 = 29491 // 0.45
            }
            let alpha16 = max(minAlpha16, min(65536, sceneSADRatio16))
            let smoothed = ((Int64(newStepInt) * alpha16) + (Int64(lastPFrameQStep) * (65536 - alpha16))) >> 16
            newStepInt = Int(max(Int64(minStep), min(Int64(maxStep), smoothed)))
        }
        
        let finalStep = min(16384, max(minStep, min(maxStep, newStepInt)))
        return finalStep
    }
    
    @inline(__always)
    mutating func consumePFrame(bits: Int, qStep: Int, sad: Int, distortion: Int, interRatioQ8: Int, detailThinned: Bool) {
        self.consumedTotalBits += bits
        self.plannedTotalScaled += self.plannedBitrate
        
        self.gopRemainingBits -= bits
        self.gopRemainingFrames -= 1
        
        self.lastPFrameBits = bits
        self.lastPFrameQStep = qStep
        self.lastPFrameSAD = sad
        
        // Track reconstruction distortion with EMA.
        // Slow adaptation (7/8 weight on history) to establish a stable target.
        let isCheap = 0 < self.avgPFrameBits && (bits * 2) < self.avgPFrameBits
        let isDistorted = (self.targetDistortionQ8 * 4) < distortion
        
        if isCheap && isDistorted && interRatioQ8 <= staticInterRatioMaxQ8 {
            self.staticStreak += 1
        } else {
            self.staticStreak = 0
        }
        
        self.lastDistortionQ8 = distortion
        if detailThinned != true && 0 < self.avgDistortionQ8 && (self.avgDistortionQ8 * 2) < distortion && (32 * 256) < distortion {
            self.driftStreak += 1
        } else {
            self.driftStreak = 0
        }
        // TODO: Consider mitigating the issue where increased recon distortion from thinned frames (detailThinned == true) pollutes the avgDistortionQ8 EMA.
        if self.avgDistortionQ8 == 0 {
            self.avgDistortionQ8 = distortion
        } else {
            self.avgDistortionQ8 = ((self.avgDistortionQ8 * 7) + distortion) / 8
        }
        
        if self.avgPFrameBits == 0 {
            self.avgPFrameBits = bits
        } else {
            self.avgPFrameBits = ((self.avgPFrameBits * 4) + bits) / 5
        }
        
        updateSaturationState()
        self.noteFrameCoded(true)
    }
    
    @inline(__always)
    func shouldRefreshStaticScene(framesSinceKeyframe: Int) -> Bool {
        let limit = min(8, self.keyint / 2)
        if 3 <= self.staticStreak && limit <= framesSinceKeyframe {
            return true
        }
        return false
    }
    
    @inline(__always)
    mutating func resetStaticStreak() {
        self.staticStreak = 0
    }

    /// A CopyFrame passed (header-only bits): advance the GOP frame
    /// bookkeeping so the remaining coded frames inherit the unspent
    /// per-frame budget, without touching the P-frame statistics — a copy is
    /// not a coding sample.
    @inline(__always)
    mutating func consumeCopyFrame(bits: Int) {
        self.consumedTotalBits += bits
        self.plannedTotalScaled += self.plannedBitrate
        
        self.gopRemainingBits -= bits
        self.gopRemainingFrames -= 1
        self.staticStreak = 0
        self.noteFrameCoded(false)
    }
}

// MARK: - Quality-floor driven early I frames (#31)
//
// With `keyint` acting as an upper bound rather than a fixed period, the GOP
// can run long enough for requantization drift to open a visible gap between
// the reconstruction and the source. The encoder already holds both, so the
// gap is measured rather than predicted: the luma MSE of every coded frame is
// compared against the luma MSE of the I frame that opened the GOP, and the
// next frame is coded as I once the ratio passes the floor.
//
// Nothing here changes the bitstream syntax. A floor-fired frame is an
// ordinary I frame, so the decoder needs no knowledge of the mechanism.

/// Integer luma MSE between a reconstruction and its source, truncating.
/// `Σ(recon − src)² / N` over the width x height luma samples, so the value is
/// deterministic and free of floating point.
@inline(__always)
func lumaMSEInteger(reconstructed: PlaneData420, source: PlaneData420) -> Int {
    let width = min(reconstructed.width, source.width)
    let height = min(reconstructed.height, source.height)
    let n = width * height
    guard 0 < n else { return 0 }

    var acc = 0
    withUnsafePointers(reconstructed.y, source.y) { rBase, sBase in
        for row in 0..<height {
            let rRow = rBase + (row * reconstructed.width)
            let sRow = sBase + (row * source.width)
            for col in 0..<width {
                let d = Int(rRow[col]) - Int(sRow[col])
                acc += d * d
            }
        }
    }
    return acc / n
}

/// Per-encoder state for the floor. Allocated only when the floor is enabled
/// and applicable (profile 0x02), so a disabled encoder never runs the MSE
/// pass at all.
final class QualityFloorState: @unchecked Sendable {
    /// Floor ratio in Q8: the next frame is coded as I once
    /// `alphaQ8 * iMSE < frameMSE * 256`.
    let alphaQ8: Int
    /// Luma MSE of the I frame that opened the current GOP.
    private(set) var iMSE: Int = 0
    /// Set when a coded P frame crossed the floor; consumed by the next frame.
    private(set) var pendingIFrame: Bool = false

    // Diagnostics only; none of this reaches the bitstream.
    /// `k` is the periodic-grid position and `dist` the true distance from the
    /// last coded I frame. The guard uses `dist`; the two differ after a cut.
    private(set) var firings: [(frame: Int, k: Int, dist: Int, frameMSE: Int, iMSE: Int)] = []
    private(set) var codedFrames: Int = 0

    init(alphaQ8: Int) {
        self.alphaQ8 = alphaQ8
    }

    /// Records the I frame that opens a GOP. Called for every I frame,
    /// including scene-change and floor-fired ones.
    @inline(__always)
    func noteIFrame(mse: Int) {
        iMSE = mse
        codedFrames += 1
    }

    /// Evaluates a coded P frame against the floor. `k` is the periodic-grid
    /// position of the frame just coded and `dist` its true distance from the
    /// last coded I frame. Returns true when this frame armed an early I.
    @discardableResult
    @inline(__always)
    func notePFrame(mse: Int, frameIndex: Int, k: Int, dist: Int) -> Bool {
        codedFrames += 1
        // A GOP needs a few frames before drift is meaningful, and firing on
        // the frames right after an I would collapse the GOP length. Measured
        // from the last coded I, not from the periodic grid, so a cut-driven I
        // also restarts the window.
        guard 8 < dist else { return false }
        guard (alphaQ8 * iMSE) < (mse * 256) else { return false }
        pendingIFrame = true
        firings.append((frame: frameIndex, k: k, dist: dist, frameMSE: mse, iMSE: iMSE))
        return true
    }

    /// Consumes the pending request, if any.
    @inline(__always)
    func takePendingIFrame() -> Bool {
        let pending = pendingIFrame
        pendingIFrame = false
        return pending
    }
}
