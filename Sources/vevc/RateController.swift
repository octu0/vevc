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
        
        self.lastPFrameBits = 0
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
        
        self.lastPFrameBits = 0 // Reset
        self.lastPFrameQStep = qStep
        self.lastPFrameSAD = 0
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
        
        // SSIM Min 0.71, Max 0.99
        let maxStep = max(baseStep * 2, min(512, baseStep * 8))
        
        var newStepInt = (baseStep * 3) / 2
        let minStep = max(2, baseStep)
        if 0 < lastPFrameBits && 0 < lastPFrameQStep && 0 < lastPFrameSAD {
            // Predict the amount of bits we'd get if we used the same Q as last P-frame
            // The bits should scale with SAD relative to the last frame, NOT the average.
            let sadRatio16 = (Int64(currentSAD) << 16) / Int64(lastPFrameSAD)
            let clampedSadRatio16 = max(13107, min(327680, sadRatio16))
            let predictedBits64 = (Int64(lastPFrameBits) * clampedSadRatio16) >> 16
            
            // ratio = predictedCurrentBits / targetFrameBits
            let safeTarget = max(1, targetFrameBits)
            // val = lastPFrameQStep * ratio
            let val = (Int64(lastPFrameQStep) * predictedBits64) / Int64(safeTarget)
            
            newStepInt = Int(max(Int64(minStep), min(Int64(maxStep), val)))
        }
        
        let finalStep = max(minStep, min(maxStep, newStepInt))
        return finalStep
    }
    
    @inline(__always)
    mutating func consumePFrame(bits: Int, qStep: Int, sad: Int) {
        self.gopRemainingBits -= bits
        self.gopRemainingFrames -= 1
        
        self.lastPFrameBits = bits
        self.lastPFrameQStep = qStep
        self.lastPFrameSAD = sad
    }
}
