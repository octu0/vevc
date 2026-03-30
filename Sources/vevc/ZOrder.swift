// MARK: - Z-Order (Morton Code) Utility
//
// Converts 2D spatial coordinates into 1D continuous array index (Z-curve)
// Used in Entropy coding scanning to maximize continuous zero-runs (run-length encoding)

struct ZOrder {
    
    /// Pre-computed 1D arrays of Z-Order (Morton Code) coordinates for DWT macroblocks
    static let coords4: [(x: Int, y: Int)] = buildZOrder(size: 4)
    static let coords8: [(x: Int, y: Int)] = buildZOrder(size: 8)
    static let coords16: [(x: Int, y: Int)] = buildZOrder(size: 16)
    static let coords32: [(x: Int, y: Int)] = buildZOrder(size: 32)
    
    @inline(__always)
    static func index(x: Int, y: Int) -> Int {
        var z = 0
        // size max is 32x32, which only requires 5 bits. 8 bit loop is safe.
        for i in 0..<8 {
            z |= ((x >> i) & 1) << (2 * i)
            z |= ((y >> i) & 1) << (2 * i + 1)
        }
        return z
    }
    
    private static func buildZOrder(size: Int) -> [(x: Int, y: Int)] {
        var arr = [(x: Int, y: Int)](repeating: (0, 0), count: size * size)
        for y in 0..<size {
            for x in 0..<size {
                let z = index(x: x, y: y)
                arr[z] = (x, y)
            }
        }
        return arr
    }
}
