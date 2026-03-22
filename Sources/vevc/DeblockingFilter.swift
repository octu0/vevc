import Foundation

public struct DeblockingFilter {
    /// In-place applies deblocking filter to the reconstructed image.
    @inline(__always)
    public static func apply(plane: inout [Int16], width: Int, height: Int, blockSize: Int, qStep: Int) {
        // Post-processing: out-of-loop like filtering for the whole generated plane.
        // It smooths boundaries between `blockSize` regions.
        
        let tc = Int16(max(2, qStep / 4)) // Threshold based on quantization step, reduced to prevent over-smoothing
        
        // Vertical Edges (x = blockSize, 2*blockSize, ...)
        plane.withUnsafeMutableBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            
            for x in stride(from: blockSize, to: width, by: blockSize) {
                for yStart in stride(from: 0, to: height, by: 16) {
                    let rowsToProcess = min(16, height - yStart)
                    if rowsToProcess == 16 {
                        filterVerticalEdgeSIMD16(base: base, width: width, x: x, y: yStart, tc: tc)
                    } else {
                        // Fallback for non-multiple of 16 heights (rare since we pad to 32)
                        filterVerticalEdgeScalar(base: base, width: width, x: x, y: yStart, count: rowsToProcess, tc: tc)
                    }
                }
            }
        }
        
        // Horizontal Edges (y = blockSize, 2*blockSize, ...)
        plane.withUnsafeMutableBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            
            for y in stride(from: blockSize, to: height, by: blockSize) {
                for xStart in stride(from: 0, to: width, by: 16) {
                    let colsToProcess = min(16, width - xStart)
                    if colsToProcess == 16 {
                        filterHorizontalEdgeSIMD16(base: base, width: width, x: xStart, y: y, tc: tc)
                    } else {
                        filterHorizontalEdgeScalar(base: base, width: width, x: xStart, y: y, count: colsToProcess, tc: tc)
                    }
                }
            }
        }
    }
    
    @inline(__always)
    private static func filterVerticalEdgeSIMD16(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, tc: Int16) {
        // Vertical edge: p1, p0 | q0, q1
        // They are adjacent in memory along the row, but we need 16 rows.
        // To use SIMD, we must gather them.
        var p1Arr = [Int16](repeating: 0, count: 16)
        var p0Arr = [Int16](repeating: 0, count: 16)
        var q0Arr = [Int16](repeating: 0, count: 16)
        var q1Arr = [Int16](repeating: 0, count: 16)
        
        for dy in 0..<16 {
            let offset = (y + dy) * width + x
            p1Arr[dy] = base[offset - 2]
            p0Arr[dy] = base[offset - 1]
            q0Arr[dy] = base[offset + 0]
            q1Arr[dy] = base[offset + 1]
        }
        
        let p1 = SIMD16<Int16>(p1Arr)
        let p0 = SIMD16<Int16>(p0Arr)
        let q0 = SIMD16<Int16>(q0Arr)
        let q1 = SIMD16<Int16>(q1Arr)
        
        let (newP0, newQ0) = computeFilter(p1: p1, p0: p0, q0: q0, q1: q1, tc: tc)
        
        for dy in 0..<16 {
            let offset = (y + dy) * width + x
            base[offset - 1] = newP0[dy]
            base[offset + 0] = newQ0[dy]
        }
    }
    
    @inline(__always)
    private static func filterVerticalEdgeScalar(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, count: Int, tc: Int16) {
        for dy in 0..<count {
            let offset = (y + dy) * width + x
            let p1 = base[offset - 2]
            var p0 = base[offset - 1]
            var q0 = base[offset + 0]
            let q1 = base[offset + 1]
            
            let delta = Int32(q0) - Int32(p0)
            if abs(delta) < Int32(tc) {
                let d = (9 * (Int32(q0) - Int32(p0)) - 3 * (Int32(q1) - Int32(p1)) + 8) >> 4
                let dClipped = Int16(max(-Int32(tc), min(Int32(tc), d)))
                p0 = p0 &+ dClipped
                q0 = q0 &- dClipped
                base[offset - 1] = p0
                base[offset + 0] = q0
            }
        }
    }
    
    @inline(__always)
    private static func filterHorizontalEdgeSIMD16(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, tc: Int16) {
        // Horizontal edge: p1, p0 | q0, q1
        // Memory layout for 16 columns is continuous!
        let offP1 = (y - 2) * width + x
        let offP0 = (y - 1) * width + x
        let offQ0 = (y + 0) * width + x
        let offQ1 = (y + 1) * width + x
        
        let p1 = SIMD16<Int16>(UnsafeBufferPointer(start: base + offP1, count: 16))
        let p0 = SIMD16<Int16>(UnsafeBufferPointer(start: base + offP0, count: 16))
        let q0 = SIMD16<Int16>(UnsafeBufferPointer(start: base + offQ0, count: 16))
        let q1 = SIMD16<Int16>(UnsafeBufferPointer(start: base + offQ1, count: 16))
        
        let (newP0, newQ0) = computeFilter(p1: p1, p0: p0, q0: q0, q1: q1, tc: tc)
        
        let p0Ptr = UnsafeMutableRawPointer(base + offP0).assumingMemoryBound(to: SIMD16<Int16>.self)
        let q0Ptr = UnsafeMutableRawPointer(base + offQ0).assumingMemoryBound(to: SIMD16<Int16>.self)
        
        p0Ptr.pointee = newP0
        q0Ptr.pointee = newQ0
    }
    
    @inline(__always)
    private static func filterHorizontalEdgeScalar(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, count: Int, tc: Int16) {
        for dx in 0..<count {
            let offset = (y * width) + x + dx
            let p1 = base[offset - 2 * width]
            var p0 = base[offset - 1 * width]
            var q0 = base[offset + 0 * width]
            let q1 = base[offset + 1 * width]
            
            let delta = Int32(q0) - Int32(p0)
            if abs(delta) < Int32(tc) {
                let d = (9 * (Int32(q0) - Int32(p0)) - 3 * (Int32(q1) - Int32(p1)) + 8) >> 4
                let dClipped = Int16(max(-Int32(tc), min(Int32(tc), d)))
                p0 = p0 &+ dClipped
                q0 = q0 &- dClipped
                base[offset - 1 * width] = p0
                base[offset + 0 * width] = q0
            }
        }
    }
    
    @inline(__always)
    private static func computeFilter(p1: SIMD16<Int16>, p0: SIMD16<Int16>, q0: SIMD16<Int16>, q1: SIMD16<Int16>, tc: Int16) -> (SIMD16<Int16>, SIMD16<Int16>) {
        // Extracted integer upcast for overflow prevention
        let p1x = SIMD16<Int32>(truncatingIfNeeded: p1)
        let p0x = SIMD16<Int32>(truncatingIfNeeded: p0)
        let q0x = SIMD16<Int32>(truncatingIfNeeded: q0)
        let q1x = SIMD16<Int32>(truncatingIfNeeded: q1)
        
        // delta = q0 - p0
        let delta = q0x &- p0x
        
        // absDelta < tc mask
        let absDelta: SIMD16<Int32> = SIMD16<Int32>(
            abs(delta[0]), abs(delta[1]), abs(delta[2]), abs(delta[3]),
            abs(delta[4]), abs(delta[5]), abs(delta[6]), abs(delta[7]),
            abs(delta[8]), abs(delta[9]), abs(delta[10]), abs(delta[11]),
            abs(delta[12]), abs(delta[13]), abs(delta[14]), abs(delta[15])
        )
        
        let threshold = Int32(tc)
        let mask = absDelta .< threshold // bool mask
        
        // d = (9*(q0-p0) - 3*(q1-p1) + 8) >> 4
        let dUnclipped = ((delta &* 9) &- ((q1x &- p1x) &* 3) &+ 8) &>> 4
        
        // clamp: max(-tc, min(tc, d))
        let lower = SIMD16<Int32>(repeating: -threshold)
        let upper = SIMD16<Int32>(repeating: threshold)
        var d = dUnclipped
        d.clamp(lowerBound: lower, upperBound: upper)
        
        let dMasked = SIMD16<Int32>(repeating: 0).replacing(with: d, where: mask)
        let d16 = SIMD16<Int16>(truncatingIfNeeded: dMasked)
        
        let newP0 = p0 &+ d16
        let newQ0 = q0 &- d16
        
        return (newP0, newQ0)
    }
}
