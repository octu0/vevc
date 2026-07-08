import Foundation

internal class Diagnostic: @unchecked Sendable {
    nonisolated(unsafe) internal static var enabled = false
    nonisolated(unsafe) internal static var nonZero = [String: Int]()
    nonisolated(unsafe) internal static var absOne = [String: Int]()
    nonisolated(unsafe) internal static var total = [String: Int]()
    internal static let lock = NSLock()
    
    internal static func record(block: BlockView, layer: Int, subband: String) {
        if !enabled { return }
        let key = "L\(layer)_\(subband)"
        let w = block.width
        let h = block.height
        
        var nz = 0
        var a1 = 0
        for y in 0..<h {
            let ptr = block.rowPointer(y: y)
            for x in 0..<w {
                let v = abs(ptr[x])
                if v > 0 { nz += 1 }
                if v == 1 { a1 += 1 }
            }
        }
        
        lock.lock()
        total[key, default: 0] += w * h
        nonZero[key, default: 0] += nz
        absOne[key, default: 0] += a1
        lock.unlock()
    }
    
    internal static func printStats() {
        print("--- Diagnostic Stats ---")
        lock.lock()
        let keys = total.keys.sorted()
        for k in keys {
            let t = total[k]!
            let nz = nonZero[k]!
            let a1 = absOne[k]!
            let nzRate = Double(nz) / Double(t) * 100.0
            let a1Rate = nz > 0 ? Double(a1) / Double(nz) * 100.0 : 0.0
            print(String(format: "%-8@ : NonZero: %5.2f%%, |q|=1 / NonZero: %5.2f%%", k, nzRate, a1Rate))
        }
        lock.unlock()
    }
    
    internal static func reset() {
        lock.lock()
        nonZero.removeAll()
        absOne.removeAll()
        total.removeAll()
        lock.unlock()
    }
}
