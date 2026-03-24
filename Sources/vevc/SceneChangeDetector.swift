import Foundation

struct SceneChangeDetector {
    let threshold: Int
    
    init(threshold: Int) {
        self.threshold = threshold
    }
    
    @inline(__always)
    func isSceneChanged(prev: PlaneData420, curr: PlaneData420) -> Bool {
        var diffSum: Int = 0
        var count: Int = 0
        
        // Sample every 64th pixel for extreme performance
        let step = 64
        let yCount = min(curr.y.count, prev.y.count)
        if yCount == 0 { return false }
        
        withUnsafePointers(curr.y, prev.y) { currPtr, prevPtr in
            var i = 0
            while i < yCount {
                let c = Int(currPtr[i])
                let p = Int(prevPtr[i])
                diffSum += abs(c - p)
                count += 1
                i += step
            }
        }
        
        if count == 0 { return false }
        let meanSAD = diffSum / count
        
        return meanSAD > threshold
    }
}
