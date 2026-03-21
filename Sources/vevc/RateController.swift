import Foundation

public struct RateController {
    public let maxbitrate: Int
    public let framerate: Int
    public let gopSize: Int
    
    public private(set) var gopTargetBits: Int = 0
    public private(set) var gopRemainingBits: Int = 0
    public private(set) var gopRemainingFrames: Int = 0
    
    public private(set) var avgPFrameSAD: Double = 0.0
    public private(set) var lastPFrameBits: Int = 0
    public private(set) var lastPFrameQStep: Int = 0
    public private(set) var lastPFrameSAD: Double = 0.0
    
    public init(maxbitrate: Int, framerate: Int, gopSize: Int) {
        self.maxbitrate = maxbitrate
        self.framerate = framerate
        self.gopSize = gopSize
    }
    
    public mutating func beginGOP() -> Int {
        self.gopTargetBits = Int((Double(maxbitrate) / Double(framerate)) * Double(gopSize))
        self.gopRemainingBits = self.gopTargetBits
        self.gopRemainingFrames = self.gopSize
        
        self.lastPFrameBits = 0
        self.lastPFrameQStep = 0
        self.lastPFrameSAD = 0.0
        
        // I-Frame receives a substantial portion of the GOP bits (e.g. ~25%)
        return max(1000, self.gopTargetBits / 4)
    }
    
    public mutating func consumeIFrame(bits: Int, qStep: Int) {
        self.gopRemainingBits -= bits
        self.gopRemainingFrames -= 1
        
        self.lastPFrameBits = 0 // Reset
        self.lastPFrameQStep = qStep
        self.lastPFrameSAD = 0.0
    }
    
    public mutating func calculatePFrameQStep(currentSAD: Double, baseStep: Int) -> Int {
        if self.avgPFrameSAD == 0.0 { self.avgPFrameSAD = currentSAD }
        self.avgPFrameSAD = (self.avgPFrameSAD * 0.8) + (currentSAD * 0.2)
        
        // Ensure P-Frames always get at least 2% of GOP bits to avoid total quality collapse
        let fallbackBits = Int(Double(gopTargetBits) * 0.02)
        let avgBitsPerFrame = max(fallbackBits, Int(gopRemainingBits / max(1, gopRemainingFrames)))
        
        // Weight by activity variation
        let multiplier = currentSAD / max(1.0, self.avgPFrameSAD)
        let targetFrameBits = Int(Double(avgBitsPerFrame) * max(0.2, min(5.0, multiplier)))
        
        var newStepInt = baseStep
        if lastPFrameBits > 0 && lastPFrameQStep > 0 {
            // Predict the amount of bits we'd get if we used the same Q as last P-frame
            let predictedCurrentBits = Double(lastPFrameBits) * multiplier
            let ratio = predictedCurrentBits / Double(max(1, targetFrameBits))
            let val = Double(lastPFrameQStep) * ratio
            
            if val.isNaN || val.isInfinite {
                newStepInt = 128
            } else {
                newStepInt = Int(min(128.0, max(1.0, val)))
            }
        }
        
        return Int(max(1, min(128, newStepInt)))
    }
    
    public mutating func consumePFrame(bits: Int, qStep: Int, sad: Double) {
        self.gopRemainingBits -= bits
        self.gopRemainingFrames -= 1
        
        self.lastPFrameBits = bits
        self.lastPFrameQStep = qStep
        self.lastPFrameSAD = sad
    }
}
