struct RateController {
    let baseMaxBitrate: Int
    private(set) var maxbitrate: Int
    let framerate: Int
    let keyint: Int
    let targetDistortionQ8: Int
    
    private(set) var gopTargetBits: Int = 0
    private(set) var gopRemainingBits: Int = 0
    private(set) var gopRemainingFrames: Int = 0
    
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
    
    @inline(__always)
    var isDriftAccelerating: Bool {
        if avgDistortionQ8 == 0 { return false }
        return (avgDistortionQ8 * 2) < lastDistortionQ8 && (32 * 256) < lastDistortionQ8
    }

    init(maxbitrate: Int, framerate: Int, keyint: Int, targetDistortion: Int = 600) {
        self.baseMaxBitrate = maxbitrate
        self.maxbitrate = (maxbitrate * 410) / 256
        self.framerate = framerate
        self.keyint = keyint
        self.targetDistortionQ8 = targetDistortion
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
        
        // Distortion adaptation of 1.6x gain
        if self.isQualitySaturated {
            self.maxbitrate = self.baseMaxBitrate
        } else {
            let margin = max(1, self.targetDistortionQ8 / 2)
            let diff = self.avgDistortionQ8 - self.targetDistortionQ8
            let rawGain = 256 + (diff * (410 - 256)) / margin
            let gainQ8 = max(256, min(410, rawGain))
            self.maxbitrate = (self.baseMaxBitrate * gainQ8) / 256
        }
        
        self.maxbitrate = max(100, self.maxbitrate)
        
        let baseGOPBits = (self.maxbitrate * self.keyint) / self.framerate
        // Carry over unused bits from the previous GOP (up to 1 GOP's worth) to handle complex scenes
        var carryOver = max(0, min(baseGOPBits, self.gopRemainingBits))
        
        // Suppress carry-over when quality hits the ceiling to prevent useless size bloat
        if self.isQualitySaturated {
            carryOver = 0
        }
        
        self.gopTargetBits = baseGOPBits + carryOver
        self.gopRemainingBits = self.gopTargetBits
        self.gopRemainingFrames = self.keyint
        // lastPFrameBits / lastPFrameQStep / lastPFrameSAD are intentionally
        // NOT reset here. Carrying over the previous GOP's last P-frame data
        // allows the first P-frame of the new GOP to use it as a prediction
        // reference, preventing the quality discontinuity at GOP boundaries.
        // I-frame bit allocation with keyint-independent quality floor.
        //
        // Problem: with shorter GOPs, the GOP budget shrinks proportionally,
        // and 15% of a smaller budget produces lower-quality I-frames.
        // This defeats the purpose of shorter GOPs (better drift reset).
        //
        // Solution: compute the I-frame budget as if keyint=60 (reference GOP),
        // then use that as the absolute floor. This ensures I-frame quality
        // remains constant regardless of GOP length.
        //
        // referenceGOPBits = maxbitrate * 60 / framerate (keyint=60 equivalent)
        // absoluteFloor = referenceGOPBits * 10% = maxbitrate * 60 / (framerate * 10)
        //               = maxbitrate * 6 / framerate
        let absoluteFloor = (self.maxbitrate * 6) / self.framerate
        let iFrameBitsProp = (self.gopTargetBits * 5) / (self.keyint + 4)
        return max(1000, max(absoluteFloor, iFrameBitsProp))
    }
    
    @inline(__always)
    mutating func consumeIFrame(bits: Int, qStep: Int) {
        self.gopRemainingBits -= bits
        self.gopRemainingFrames -= 1
        
        self.pPlanBits = self.gopRemainingBits
        self.pPlanFrames = self.gopRemainingFrames
        
        self.staticStreak = 0
    }
    
    @inline(__always)
    mutating func calculatePFrameQStep(currentSAD: Int, baseStep: Int) -> Int {
        if self.avgPFrameSAD == 0 { self.avgPFrameSAD = currentSAD }
        // EMA: avg = avg * 0.8 + current * 0.2 → (avg * 4 + current) / 5
        self.avgPFrameSAD = ((self.avgPFrameSAD * 4) + currentSAD) / 5
        
        // Ensure P-Frames always get at least 2% of GOP bits to avoid total quality collapse
        // 2% = 1/50
        let fallbackBits = gopTargetBits / 50
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
            // Budget tight (<0.75): Allow coarser quantization
            maxStep = min(8192, baseStep * 8)
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
        if 0 < lastDistortionQ8 && 0 < avgDistortionQ8 {
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
        if 0 < lastPFrameQStep && 0 < avgPFrameSAD {
            let sceneSADRatio16 = (Int64(currentSAD) << 16) / Int64(max(1, avgPFrameSAD))
            // When transitioning to static scene (sceneSadRatio16 < 32768), use higher alpha to drop QP faster.
            let minAlpha16 = if sceneSADRatio16 < 32768 {
                Int64(29491) // 0.45
            } else {
                Int64(19661) // 0.3
            }
            let alpha16 = max(minAlpha16, min(65536, sceneSADRatio16))
            let smoothed = (Int64(newStepInt) * alpha16 + Int64(lastPFrameQStep) * (65536 - alpha16)) >> 16
            newStepInt = Int(max(Int64(minStep), min(Int64(maxStep), smoothed)))
        }
        
        let finalStep = min(16384, max(minStep, min(maxStep, newStepInt)))
        return finalStep
    }
    
    @inline(__always)
    mutating func consumePFrame(bits: Int, qStep: Int, sad: Int, distortion: Int) {
        self.gopRemainingBits -= bits
        self.gopRemainingFrames -= 1
        
        self.lastPFrameBits = bits
        self.lastPFrameQStep = qStep
        self.lastPFrameSAD = sad
        
        // Track reconstruction distortion with EMA.
        // Slow adaptation (7/8 weight on history) to establish a stable target.
        let isCheap = self.avgPFrameBits > 0 && (bits * 2) < self.avgPFrameBits
        let isDistorted = (self.targetDistortionQ8 * 4) < distortion
        
        if isCheap && isDistorted {
            self.staticStreak += 1
        } else {
            self.staticStreak = 0
        }
        
        self.lastDistortionQ8 = distortion
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
}
