import Foundation

struct RateController {
    let maxbitrate: Int
    let framerate: Int
    let keyint: Int
    
    private(set) var gopTargetBits: Int = 0
    private(set) var gopRemainingBits: Int = 0
    private(set) var gopRemainingFrames: Int = 0
    
    private(set) var avgPFrameSAD: Double = 0.0
    private(set) var lastPFrameBits: Int = 0
    private(set) var lastPFrameQStep: Int = 0
    private(set) var lastPFrameSAD: Double = 0.0
    
    init(maxbitrate: Int, framerate: Int, keyint: Int) {
        self.maxbitrate = maxbitrate
        self.framerate = framerate
        self.keyint = keyint
    }
    
    @inline(__always)
    mutating func beginGOP() -> Int {
        self.gopTargetBits = Int((Double(maxbitrate) / Double(framerate)) * Double(keyint))
        self.gopRemainingBits = self.gopTargetBits
        self.gopRemainingFrames = self.keyint
        
        self.lastPFrameBits = 0
        // I-Frame receives roughly 5x the bits of an average P-frame.
        // However, to ensure a strong structural base (I-frame SSIM > 0.92) so that P-frames
        // don't degrade below 0.85, we guarantee a MINIMUM of 15% of the GOP budget to the I-Frame.
        let iFrameRatio = max(0.15, 5.0 / Double(self.keyint + 4))
        return max(1000, Int(Double(self.gopTargetBits) * iFrameRatio))
    }
    
    @inline(__always)
    mutating func consumeIFrame(bits: Int, qStep: Int) {
        self.gopRemainingBits -= bits
        self.gopRemainingFrames -= 1
        
        self.lastPFrameBits = 0 // Reset
        self.lastPFrameQStep = qStep
        self.lastPFrameSAD = 0.0
    }
    
    @inline(__always)
    mutating func calculatePFrameQStep(currentSAD: Double, baseStep: Int) -> Int {
        if self.avgPFrameSAD == 0.0 { self.avgPFrameSAD = currentSAD }
        self.avgPFrameSAD = (self.avgPFrameSAD * 0.8) + (currentSAD * 0.2)
        
        // Ensure P-Frames always get at least 2% of GOP bits to avoid total quality collapse
        let fallbackBits = Int(Double(gopTargetBits) * 0.02)
        let avgBitsPerFrame = max(fallbackBits, Int(gopRemainingBits / max(1, gopRemainingFrames)))
        
        // Weight by activity variation
        let multiplier = currentSAD / max(1.0, self.avgPFrameSAD)
        let targetFrameBits = Int(Double(avgBitsPerFrame) * max(0.2, min(5.0, multiplier)))
        
        // SSIM Min 0.71, Max 0.99
        let maxStep = max(baseStep, 48)
        
        var newStepInt = baseStep
        if lastPFrameBits > 0 && lastPFrameQStep > 0 {
            // Predict the amount of bits we'd get if we used the same Q as last P-frame
            let predictedCurrentBits = Double(lastPFrameBits) * multiplier
            let ratio = predictedCurrentBits / Double(max(1, targetFrameBits))
            let val = Double(lastPFrameQStep) * ratio
            
            if val.isNaN || val.isInfinite {
                newStepInt = maxStep
            } else {
                newStepInt = Int(min(Double(maxStep), max(1.0, val)))
            }
        }
        
        let finalStep = Int(max(1, min(maxStep, newStepInt)))
        return finalStep
    }
    
    @inline(__always)
    mutating func consumePFrame(bits: Int, qStep: Int, sad: Double) {
        self.gopRemainingBits -= bits
        self.gopRemainingFrames -= 1
        
        self.lastPFrameBits = bits
        self.lastPFrameQStep = qStep
        self.lastPFrameSAD = sad
    }
}
