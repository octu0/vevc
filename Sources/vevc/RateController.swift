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
        self.gopTargetBits = (maxbitrate * keyint) / framerate
        self.gopRemainingBits = self.gopTargetBits
        self.gopRemainingFrames = self.keyint
        
        self.lastPFrameBits = 0
        // I-Frame receives roughly 5x the bits of an average P-frame.
        // However, to ensure a strong structural base (I-frame SSIM > 0.92) so that P-frames
        // don't degrade below 0.85, we guarantee a MINIMUM of 15% of the GOP budget to the I-Frame.
        // iFrameRatio = max(0.15, 5.0 / (keyint + 4))
        // Expressed as integer: max(gopTargetBits * 3 / 20, gopTargetBits * 5 / (keyint + 4))
        let iFrameBitsFloor = (self.gopTargetBits * 3) / 20        // 15% = 3/20
        let iFrameBitsProp = (self.gopTargetBits * 5) / (self.keyint + 4)
        return max(1000, max(iFrameBitsFloor, iFrameBitsProp))
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
        let maxStep = max(baseStep * 4, 512)
        
        var newStepInt = baseStep
        if 0 < lastPFrameBits && 0 < lastPFrameQStep {
            // Predict the amount of bits we'd get if we used the same Q as last P-frame
            // predictedCurrentBits = lastPFrameBits * multiplier
            let predictedBits64 = (Int64(lastPFrameBits) * multiplier16) >> 16
            // ratio = predictedCurrentBits / targetFrameBits
            let safeTarget = max(1, targetFrameBits)
            // val = lastPFrameQStep * ratio
            let val = (Int64(lastPFrameQStep) * predictedBits64) / Int64(safeTarget)
            
            newStepInt = Int(max(1, min(Int64(maxStep), val)))
        }
        
        let finalStep = max(1, min(maxStep, newStepInt))
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
