import Foundation

struct RateController {
    let maxbitrate: Int
    let framerate: Int
    let keyint: Int
    
    private(set) var gopTargetBits: Int = 0
    private(set) var gopRemainingBits: Int = 0
    private(set) var gopRemainingFrames: Int = 0
    
    private(set) var avgPFrameSAD: Int = 0
    private(set) var lastPFrameBits: Int = 0
    private(set) var lastPFrameQStep: Int = 0
    private(set) var lastPFrameSAD: Int = 0
    
    // Reconstruction distortion tracking for quality-consistent QP adjustment.
    // avgDistortion: EMA of per-pixel reconstruction SAD (target quality level)
    // lastDistortion: previous frame's per-pixel reconstruction SAD
    private(set) var avgDistortion: Int = 0
    private(set) var lastDistortion: Int = 0
    
    @inline(__always)
    var isDriftAccelerating: Bool {
        if avgDistortion == 0 { return false }
        return (avgDistortion * 2) < lastDistortion && 32 < lastDistortion
    }
    
    init(maxbitrate: Int, framerate: Int, keyint: Int) {
        self.maxbitrate = maxbitrate
        self.framerate = framerate
        self.keyint = keyint
    }
    
    @inline(__always)
    mutating func beginGOP() -> Int {
        let baseGOPBits = (maxbitrate * keyint) / framerate
        // Carry over unused bits from the previous GOP (up to 1 GOP's worth) to handle complex scenes
        let carryOver = max(0, min(baseGOPBits, self.gopRemainingBits))
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
        
        // maxStep scales proportionally to baseStep: P-Frame worst-case quality
        // tracks I-Frame quality level. At high bitrates (low baseStep), this
        // prevents P-Frames from degrading to bitrate-500 quality (step=40) even
        // though budget is abundant. At baseStep>=5, baseStep*8>=40 so behavior
        // is identical to before.
        let maxStep = max(2, min(512, baseStep * 4))
        
        var newStepInt = (baseStep * 3) / 2
        // P-Frame QP floor: baseStep ensures P-Frames never use finer
        // quantization than the I-Frame. At high bitrates (baseStep=1-2),
        // this allows near-lossless P-frame quality.
        let minStep = max(1, baseStep)
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
        if 0 < lastDistortion && 0 < avgDistortion {
            // fullCorrection = newStep * avgDistortion / lastDistortion
            // blended = (newStep + fullCorrection) / 2 → 50% correction strength
            let fullCorrection = (Int64(newStepInt) * Int64(avgDistortion)) / Int64(lastDistortion)
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
        
        let finalStep = max(minStep, min(maxStep, newStepInt))
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
        self.lastDistortion = distortion
        if self.avgDistortion == 0 {
            self.avgDistortion = distortion
        } else {
            self.avgDistortion = ((self.avgDistortion * 7) + distortion) / 8
        }
    }
}
