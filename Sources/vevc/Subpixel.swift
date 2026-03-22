import Foundation

public struct SubpixelInterpolator {
    // HEVC Luma 8-tap filter coefficients
    // Denominator = 64 (shift by 6)
    
    // Quarter-pel: alpha = 1/4
    @inline(__always)
    public static func interpolateQuarterX(ptr: UnsafePointer<Int16>, offset: Int) -> Int16 {
        let p = ptr.advanced(by: offset - 3)
        // [-1, 4, -10, 58, 17, -5, 1, 0]
        let sum = -1 * Int(p[0])
                +  4 * Int(p[1])
                - 10 * Int(p[2])
                + 58 * Int(p[3])
                + 17 * Int(p[4])
                -  5 * Int(p[5])
                +  1 * Int(p[6])
                +  0 * Int(p[7])
        
        // Add 32 for rounding before shift right by 6
        let val = (sum + 32) >> 6
        return Int16(clamping: val)
    }

    // Half-pel: alpha = 2/4 = 1/2
    @inline(__always)
    public static func interpolateHalfX(ptr: UnsafePointer<Int16>, offset: Int) -> Int16 {
        let p = ptr.advanced(by: offset - 3)
        // [-1, 4, -11, 40, 40, -11, 4, -1]
        let sum = -1 * Int(p[0])
                +  4 * Int(p[1])
                - 11 * Int(p[2])
                + 40 * Int(p[3])
                + 40 * Int(p[4])
                - 11 * Int(p[5])
                +  4 * Int(p[6])
                -  1 * Int(p[7])
        
        let val = (sum + 32) >> 6
        return Int16(clamping: val)
    }

    // Three Quarter-pel: alpha = 3/4
    @inline(__always)
    public static func interpolateThreeQuarterX(ptr: UnsafePointer<Int16>, offset: Int) -> Int16 {
        let p = ptr.advanced(by: offset - 3)
        // [0, 1, -5, 17, 58, -10, 4, -1]
        let sum =  0 * Int(p[0])
                +  1 * Int(p[1])
                -  5 * Int(p[2])
                + 17 * Int(p[3])
                + 58 * Int(p[4])
                - 10 * Int(p[5])
                +  4 * Int(p[6])
                -  1 * Int(p[7])
        
        let val = (sum + 32) >> 6
        return Int16(clamping: val)
    }
    
    // Y-direction interpolation
    @inline(__always)
    public static func interpolateY(ptr: UnsafePointer<Int16>, offset: Int, stride: Int, fracY: Int) -> Int16 {
        let p0 = ptr.advanced(by: offset - 3 * stride)
        let p1 = ptr.advanced(by: offset - 2 * stride)
        let p2 = ptr.advanced(by: offset - 1 * stride)
        let p3 = ptr.advanced(by: offset)
        let p4 = ptr.advanced(by: offset + 1 * stride)
        let p5 = ptr.advanced(by: offset + 2 * stride)
        let p6 = ptr.advanced(by: offset + 3 * stride)
        let p7 = ptr.advanced(by: offset + 4 * stride)
        
        var sum = 0
        switch fracY {
        case 1:
            sum = -1 * Int(p0.pointee) + 4 * Int(p1.pointee) - 10 * Int(p2.pointee) + 58 * Int(p3.pointee) + 17 * Int(p4.pointee) - 5 * Int(p5.pointee) + 1 * Int(p6.pointee)
        case 2:
            sum = -1 * Int(p0.pointee) + 4 * Int(p1.pointee) - 11 * Int(p2.pointee) + 40 * Int(p3.pointee) + 40 * Int(p4.pointee) - 11 * Int(p5.pointee) + 4 * Int(p6.pointee) - 1 * Int(p7.pointee)
        case 3:
            sum = 1 * Int(p1.pointee) - 5 * Int(p2.pointee) + 17 * Int(p3.pointee) + 58 * Int(p4.pointee) - 10 * Int(p5.pointee) + 4 * Int(p6.pointee) - 1 * Int(p7.pointee)
        default:
            return ptr.advanced(by: offset).pointee
        }
        
        let val = (sum + 32) >> 6
        return Int16(clamping: val)
    }
    
    // 2D Interpolation routine
    // fracX, fracY = [0, 1, 2, 3] representing 0, 1/4, 2/4, 3/4 pel offset
    @inline(__always)
    public static func interpolateBlock(
        src: UnsafePointer<Int16>, srcStride: Int,
        dst: UnsafeMutablePointer<Int16>, dstStride: Int,
        width: Int, height: Int,
        fracX: Int, fracY: Int,
        startX: Int, startY: Int
    ) {
        if fracX == 0 && fracY == 0 {
            for y in 0..<height {
                let srcPtr = src.advanced(by: (startY + y) * srcStride + startX)
                let dstPtr = dst.advanced(by: y * dstStride)
                dstPtr.update(from: srcPtr, count: width)
            }
            return
        }
        
        if fracY == 0 {
            for y in 0..<height {
                let srcYOffset = (startY + y) * srcStride
                let dstYOffset = y * dstStride
                for x in 0..<width {
                    let rx = startX + x
                    switch fracX {
                    case 1: dst[dstYOffset + x] = interpolateQuarterX(ptr: src, offset: srcYOffset + rx)
                    case 2: dst[dstYOffset + x] = interpolateHalfX(ptr: src, offset: srcYOffset + rx)
                    case 3: dst[dstYOffset + x] = interpolateThreeQuarterX(ptr: src, offset: srcYOffset + rx)
                    default: break
                    }
                }
            }
            return
        }
        
        if fracX == 0 {
            for y in 0..<height {
                let dstYOffset = y * dstStride
                for x in 0..<width {
                    let offset = (startY + y) * srcStride + startX + x
                    dst[dstYOffset + x] = interpolateY(ptr: src, offset: offset, stride: srcStride, fracY: fracY)
                }
            }
            return
        }
        
        // 2D Fractional: First interpolate vertically (with extra lines), then horizontally
        // HEVC typically does horizontal first then vertical with intermediate 16-bit values (with 6-bit shift intermediate precision).
        // Since we are operating on int16 arrays directly, we can do intermediate buffering using standard shifts.
        // For simplicity and speed in this PoC, we will compute it sequentially.
        
        let extraLines = 7
        let intermediateCount = width * (height + extraLines)
        
        withUnsafeTemporaryAllocation(of: Int16.self, capacity: intermediateCount) { interPtr in
            guard let intBase = interPtr.baseAddress else { return }
            
            for y in 0..<(height + extraLines) {
                let srcYOffset = (startY + y - 3) * srcStride
                let intYOffset = y * width
                for x in 0..<width {
                    let rx = startX + x
                    switch fracX {
                    case 1: intBase[intYOffset + x] = interpolateQuarterX(ptr: src, offset: srcYOffset + rx)
                    case 2: intBase[intYOffset + x] = interpolateHalfX(ptr: src, offset: srcYOffset + rx)
                    case 3: intBase[intYOffset + x] = interpolateThreeQuarterX(ptr: src, offset: srcYOffset + rx)
                    default: break
                    }
                }
            }
            
            for y in 0..<height {
                let dstYOffset = y * dstStride
                for x in 0..<width {
                    let offset = (y + 3) * width + x
                    dst[dstYOffset + x] = interpolateY(ptr: intBase, offset: offset, stride: width, fracY: fracY)
                }
            }
        }
    }
}
