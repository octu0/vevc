// MARK: - FIR

private let FIRCHROMACoeffs: [[Int]] = [
    [ 0,  8,  0,  0],
    [-1,  8,  2, -1],
    [-1,  7,  3, -1],
    [-1,  6,  4, -1],
    [-1,  5,  5, -1],
    [-1,  4,  6, -1],
    [-1,  3,  7, -1],
    [-1,  2,  8, -1]
]

@usableFromInline
let FIRLUMACoeffs: [[Int]] = [
    [0, 8, 0, 0],
    [-1, 7, 2, 0],
    [-1, 5, 5, -1],
    [0, 2, 7, -1]
]

// SIMD8<Int32> horizontal FIR helper for Luma 4-tap filter
// Loads 4 shifted SIMD8<Int16> vectors from row pointer, widens to Int32, multiplies by coefficients and sums
@inline(__always)
func horizontalFIRLuma16(
    _ src: UnsafePointer<Int16>,
    _ offset: Int,
    _ c0: SIMD16<Int32>, _ c1: SIMD16<Int32>, _ c2: SIMD16<Int32>, _ c3: SIMD16<Int32>
) -> SIMD16<Int32> {
    let p = UnsafeRawPointer(src.advanced(by: offset - 1))
    let s0 = SIMD16<Int32>(truncatingIfNeeded: p.loadUnaligned(as: SIMD16<Int16>.self))
    let s1 = SIMD16<Int32>(truncatingIfNeeded: p.advanced(by: 2).loadUnaligned(as: SIMD16<Int16>.self))
    let s2 = SIMD16<Int32>(truncatingIfNeeded: p.advanced(by: 4).loadUnaligned(as: SIMD16<Int16>.self))
    let s3 = SIMD16<Int32>(truncatingIfNeeded: p.advanced(by: 6).loadUnaligned(as: SIMD16<Int16>.self))
    return c0 &* s0 &+ c1 &* s1 &+ c2 &* s2 &+ c3 &* s3
}

@inline(__always)
func horizontalFIRLuma8(
    _ row: UnsafePointer<Int16>, _ offset: Int,
    _ vcX0: SIMD8<Int32>, _ vcX1: SIMD8<Int32>, _ vcX2: SIMD8<Int32>, _ vcX3: SIMD8<Int32>
) -> SIMD8<Int32> {
    let s0 = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(row.advanced(by: offset - 1)).loadUnaligned(as: SIMD8<Int16>.self))
    let s1 = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(row.advanced(by: offset)).loadUnaligned(as: SIMD8<Int16>.self))
    let s2 = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(row.advanced(by: offset + 1)).loadUnaligned(as: SIMD8<Int16>.self))
    let s3 = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(row.advanced(by: offset + 2)).loadUnaligned(as: SIMD8<Int16>.self))
    return vcX0 &* s0 &+ vcX1 &* s1 &+ vcX2 &* s2 &+ vcX3 &* s3
}

@inline(__always)
fileprivate func subMCBlockLuma32Inner(
    dstBase: UnsafeMutablePointer<Int16>, srcBase: UnsafePointer<Int16>,
    width: Int, height: Int, blockX: Int, blockY: Int,
    shiftX: Int, shiftY: Int, fractX: Int, fractY: Int, bw: Int, bh: Int, roundOffset: Int
) {
    let fX = FIRLUMACoeffs[fractX]
    let fY = FIRLUMACoeffs[fractY]
    let cX0 = Int32(fX[0]), cX1 = Int32(fX[1]), cX2 = Int32(fX[2]), cX3 = Int32(fX[3])
    let cY0 = Int32(fY[0]), cY1 = Int32(fY[1]), cY2 = Int32(fY[2]), cY3 = Int32(fY[3])
    let useFIR = (fractX != 0 || fractY != 0)
    switch true {
    case useFIR != true:
        for y in 0..<bh {
            let sy = blockY + shiftY + y
            let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
            let r = srcBase.advanced(by: sy * width + blockX + shiftX)
            var x = 0
            while x < bw - 15 {
                let d = UnsafeRawPointer(dstPtr.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
                let s = UnsafeRawPointer(r.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
                UnsafeMutableRawPointer(dstPtr.advanced(by: x)).storeBytes(of: d &- s, as: SIMD16<Int16>.self)
                x &+= 16
            }
            while x < bw { dstPtr[x] = dstPtr[x] &- r[x]; x &+= 1 }
        }
    case fractY == 0:
        // why: fractY==0 means vertical FIR = 8*h0, skip 3 row loads
        let vcX0 = SIMD8<Int32>(repeating: cX0), vcX1 = SIMD8<Int32>(repeating: cX1)
        let vcX2 = SIMD8<Int32>(repeating: cX2), vcX3 = SIMD8<Int32>(repeating: cX3)
        let vRound = SIMD8<Int32>(repeating: Int32(31) &+ Int32(roundOffset))
        let v8 = SIMD8<Int32>(repeating: 8)
        for y in 0..<bh {
            let sy = blockY + shiftY + y
            let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
            let r0 = srcBase.advanced(by: sy * width + blockX + shiftX)
            var x = 0
            while x < bw - 7 {
                let h0 = horizontalFIRLuma8(r0, x, vcX0, vcX1, vcX2, vcX3)
                let v = v8 &* h0
                let res16 = SIMD8<Int16>(truncatingIfNeeded: (v &+ vRound) &>> 6)
                let d = UnsafeRawPointer(dstPtr.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self)
                UnsafeMutableRawPointer(dstPtr.advanced(by: x)).storeBytes(of: d &- res16, as: SIMD8<Int16>.self)
                x &+= 8
            }
            while x < bw {
                let v0 = cX0 &* Int32(r0[x - 1]) &+ cX1 &* Int32(r0[x]) &+ cX2 &* Int32(r0[x + 1]) &+ cX3 &* Int32(r0[x + 2])
                let v = Int32(8) &* v0
                let res = Int16(truncatingIfNeeded: (v &+ 31 &+ Int32(roundOffset)) >> 6)
                dstPtr[x] = dstPtr[x] &- res
                x &+= 1
            }
        }
    case fractX == 0:
        // why: fractX==0 means horizontal FIR = 8*pixel, skip horizontalFIRLuma8
        let vcY0 = SIMD8<Int32>(repeating: cY0), vcY1 = SIMD8<Int32>(repeating: cY1)
        let vcY2 = SIMD8<Int32>(repeating: cY2), vcY3 = SIMD8<Int32>(repeating: cY3)
        let vRound = SIMD8<Int32>(repeating: Int32(31) &+ Int32(roundOffset))
        let v8 = SIMD8<Int32>(repeating: 8)
        for y in 0..<bh {
            let sy = blockY + shiftY + y
            let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
            let rM1 = srcBase.advanced(by: (sy - 1) * width + blockX + shiftX)
            let r0 = srcBase.advanced(by: sy * width + blockX + shiftX)
            let rP1 = srcBase.advanced(by: (sy + 1) * width + blockX + shiftX)
            let rP2 = srcBase.advanced(by: (sy + 2) * width + blockX + shiftX)
            var x = 0
            while x < bw - 7 {
                let pM1 = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(rM1.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self))
                let p0  = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(r0.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self))
                let pP1 = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(rP1.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self))
                let pP2 = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(rP2.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self))
                let vertFIR = vcY0 &* pM1 &+ vcY1 &* p0 &+ vcY2 &* pP1 &+ vcY3 &* pP2
                let v = v8 &* vertFIR
                let res16 = SIMD8<Int16>(truncatingIfNeeded: (v &+ vRound) &>> 6)
                let d = UnsafeRawPointer(dstPtr.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self)
                UnsafeMutableRawPointer(dstPtr.advanced(by: x)).storeBytes(of: d &- res16, as: SIMD8<Int16>.self)
                x &+= 8
            }
            while x < bw {
                let vM1 = Int32(8) &* Int32(rM1[x])
                let v0  = Int32(8) &* Int32(r0[x])
                let vP1 = Int32(8) &* Int32(rP1[x])
                let vP2 = Int32(8) &* Int32(rP2[x])
                let v = cY0 &* vM1 &+ cY1 &* v0 &+ cY2 &* vP1 &+ cY3 &* vP2
                let res = Int16(truncatingIfNeeded: (v &+ 31 &+ Int32(roundOffset)) >> 6)
                dstPtr[x] = dstPtr[x] &- res
                x &+= 1
            }
        }
    default:
        // Full 2D FIR: both fractX != 0 and fractY != 0
        let vcX0 = SIMD8<Int32>(repeating: cX0), vcX1 = SIMD8<Int32>(repeating: cX1)
        let vcX2 = SIMD8<Int32>(repeating: cX2), vcX3 = SIMD8<Int32>(repeating: cX3)
        let vcY0 = SIMD8<Int32>(repeating: cY0), vcY1 = SIMD8<Int32>(repeating: cY1)
        let vcY2 = SIMD8<Int32>(repeating: cY2), vcY3 = SIMD8<Int32>(repeating: cY3)
        let vRound = SIMD8<Int32>(repeating: Int32(31) &+ Int32(roundOffset))
        for y in 0..<bh {
            let sy = blockY + shiftY + y
            let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
            let rM1 = srcBase.advanced(by: (sy - 1) * width + blockX + shiftX)
            let r0 = srcBase.advanced(by: sy * width + blockX + shiftX)
            let rP1 = srcBase.advanced(by: (sy + 1) * width + blockX + shiftX)
            let rP2 = srcBase.advanced(by: (sy + 2) * width + blockX + shiftX)
            var x = 0
            while x < bw - 7 {
                let hM1 = horizontalFIRLuma8(rM1, x, vcX0, vcX1, vcX2, vcX3)
                let h0  = horizontalFIRLuma8(r0,  x, vcX0, vcX1, vcX2, vcX3)
                let hP1 = horizontalFIRLuma8(rP1, x, vcX0, vcX1, vcX2, vcX3)
                let hP2 = horizontalFIRLuma8(rP2, x, vcX0, vcX1, vcX2, vcX3)
                let v = vcY0 &* hM1 &+ vcY1 &* h0 &+ vcY2 &* hP1 &+ vcY3 &* hP2
                let res16 = SIMD8<Int16>(truncatingIfNeeded: (v &+ vRound) &>> 6)
                let d = UnsafeRawPointer(dstPtr.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self)
                UnsafeMutableRawPointer(dstPtr.advanced(by: x)).storeBytes(of: d &- res16, as: SIMD8<Int16>.self)
                x &+= 8
            }
            while x < bw {
                let vM1 = cX0 &* Int32(rM1[x - 1]) &+ cX1 &* Int32(rM1[x]) &+ cX2 &* Int32(rM1[x + 1]) &+ cX3 &* Int32(rM1[x + 2])
                let v0  = cX0 &* Int32(r0[x - 1])  &+ cX1 &* Int32(r0[x])  &+ cX2 &* Int32(r0[x + 1])  &+ cX3 &* Int32(r0[x + 2])
                let vP1 = cX0 &* Int32(rP1[x - 1]) &+ cX1 &* Int32(rP1[x]) &+ cX2 &* Int32(rP1[x + 1]) &+ cX3 &* Int32(rP1[x + 2])
                let vP2 = cX0 &* Int32(rP2[x - 1]) &+ cX1 &* Int32(rP2[x]) &+ cX2 &* Int32(rP2[x + 1]) &+ cX3 &* Int32(rP2[x + 2])
                let v = cY0 &* vM1 &+ cY1 &* v0 &+ cY2 &* vP1 &+ cY3 &* vP2
                let res = Int16(truncatingIfNeeded: (v &+ 31 &+ Int32(roundOffset)) >> 6)
                dstPtr[x] = dstPtr[x] &- res
                x &+= 1
            }
        }
    }
}

@inline(__always)
fileprivate func subMCBlockLuma32Edge(
    dstBase: UnsafeMutablePointer<Int16>, srcBase: UnsafePointer<Int16>,
    width: Int, height: Int, blockX: Int, blockY: Int,
    shiftX: Int, shiftY: Int, fractX: Int, fractY: Int, bw: Int, bh: Int, roundOffset: Int
) {
    let fX = FIRLUMACoeffs[fractX]
    let fY = FIRLUMACoeffs[fractY]
    let cX0 = Int32(fX[0]), cX1 = Int32(fX[1]), cX2 = Int32(fX[2]), cX3 = Int32(fX[3])
    let cY0 = Int32(fY[0]), cY1 = Int32(fY[1]), cY2 = Int32(fY[2]), cY3 = Int32(fY[3])
    let useFIR = (fractX != 0 || fractY != 0)
    if useFIR  {
        for y in 0..<bh {
            let sy = blockY + shiftY + y
            let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
            let syM1 = max(0, min(sy - 1, height - 1))
            let sy0  = max(0, min(sy, height - 1))
            let syP1 = max(0, min(sy + 1, height - 1))
            let syP2 = max(0, min(sy + 2, height - 1))
            let rM1 = srcBase.advanced(by: syM1 * width)
            let r0  = srcBase.advanced(by: sy0 * width)
            let rP1 = srcBase.advanced(by: syP1 * width)
            let rP2 = srcBase.advanced(by: syP2 * width)
            for x in 0..<bw {
                let cx = blockX + shiftX + x
                let sxM1 = max(0, min(cx - 1, width - 1))
                let sx0  = max(0, min(cx, width - 1))
                let sxP1 = max(0, min(cx + 1, width - 1))
                let sxP2 = max(0, min(cx + 2, width - 1))
                let vM1 = cX0 &* Int32(rM1[sxM1]) &+ cX1 &* Int32(rM1[sx0]) &+ cX2 &* Int32(rM1[sxP1]) &+ cX3 &* Int32(rM1[sxP2])
                let v0  = cX0 &* Int32(r0[sxM1])  &+ cX1 &* Int32(r0[sx0])  &+ cX2 &* Int32(r0[sxP1])  &+ cX3 &* Int32(r0[sxP2])
                let vP1 = cX0 &* Int32(rP1[sxM1]) &+ cX1 &* Int32(rP1[sx0]) &+ cX2 &* Int32(rP1[sxP1]) &+ cX3 &* Int32(rP1[sxP2])
                let vP2 = cX0 &* Int32(rP2[sxM1]) &+ cX1 &* Int32(rP2[sx0]) &+ cX2 &* Int32(rP2[sxP1]) &+ cX3 &* Int32(rP2[sxP2])
                let v = cY0 &* vM1 &+ cY1 &* v0 &+ cY2 &* vP1 &+ cY3 &* vP2
                let res = Int16(truncatingIfNeeded: (v &+ 31 &+ Int32(roundOffset)) >> 6)
                dstPtr[x] = dstPtr[x] &- res
            }
        }
    } else {
        for y in 0..<bh {
            let sy = blockY + shiftY + y
            let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
            let safeSy = max(0, min(sy, height - 1))
            let r = srcBase.advanced(by: safeSy * width)
            for x in 0..<bw {
                let sx = max(0, min(blockX + shiftX + x, width - 1))
                dstPtr[x] = dstPtr[x] &- r[sx]
            }
        }
    }
}

@inline(__always)
fileprivate func subMCBlockLuma32(
    dstBase: UnsafeMutablePointer<Int16>, srcBase: UnsafePointer<Int16>,
    width: Int, height: Int, blockX: Int, blockY: Int,
    mv: MotionVector, roundOffset: Int, blockSize: Int = 32
) {
    if mv.isIntra { return }
    
    let mvDx = Int(mv.dx)
    let mvDy = Int(mv.dy)
    let shiftX = (mvDx >> 2)
    let shiftY = (mvDy >> 2)
    let fractX = (mvDx & 3)
    let fractY = (mvDy & 3)
    
    let bw = min(blockSize, width - blockX)
    let bh = min(blockSize, height - blockY)
    if bw <= 0 || bh <= 0 { return }
    
    let safe = (0 <= blockX + shiftX - 1) && (0 <= blockY + shiftY - 1) && (blockX + shiftX + bw + 2 < width) && (blockY + shiftY + bh + 2 < height)
    if safe {
        subMCBlockLuma32Inner(dstBase: dstBase, srcBase: srcBase, width: width, height: height, blockX: blockX, blockY: blockY, shiftX: shiftX, shiftY: shiftY, fractX: fractX, fractY: fractY, bw: bw, bh: bh, roundOffset: roundOffset)
    } else {
        subMCBlockLuma32Edge(dstBase: dstBase, srcBase: srcBase, width: width, height: height, blockX: blockX, blockY: blockY, shiftX: shiftX, shiftY: shiftY, fractX: fractX, fractY: fractY, bw: bw, bh: bh, roundOffset: roundOffset)
    }
}

@inline(__always)
fileprivate func subMCBlockChroma16Inner(
    dstBase: UnsafeMutablePointer<Int16>, srcBase: UnsafePointer<Int16>,
    width: Int, height: Int, blockX: Int, blockY: Int,
    shiftX: Int, shiftY: Int, fractX: Int, fractY: Int, bw: Int, bh: Int, roundOffset: Int
) {
    let fX = FIRCHROMACoeffs[fractX]
    let fY = FIRCHROMACoeffs[fractY]
    let cX0 = Int32(fX[0]), cX1 = Int32(fX[1]), cX2 = Int32(fX[2]), cX3 = Int32(fX[3])
    let cY0 = Int32(fY[0]), cY1 = Int32(fY[1]), cY2 = Int32(fY[2]), cY3 = Int32(fY[3])
    let useFIR = (fractX != 0 || fractY != 0)
    switch true {
    case useFIR != true:
        for y in 0..<bh {
            let sy = blockY + shiftY + y
            let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
            let r = srcBase.advanced(by: sy * width + blockX + shiftX)
            var x = 0
            while x < bw - 15 {
                let d = UnsafeRawPointer(dstPtr.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
                let s = UnsafeRawPointer(r.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
                UnsafeMutableRawPointer(dstPtr.advanced(by: x)).storeBytes(of: d &- s, as: SIMD16<Int16>.self)
                x &+= 16
            }
            while x < bw { dstPtr[x] = dstPtr[x] &- r[x]; x &+= 1 }
        }
    case fractY == 0:
        let vcX0 = SIMD8<Int32>(repeating: cX0), vcX1 = SIMD8<Int32>(repeating: cX1)
        let vcX2 = SIMD8<Int32>(repeating: cX2), vcX3 = SIMD8<Int32>(repeating: cX3)
        let vRound = SIMD8<Int32>(repeating: Int32(31) &+ Int32(roundOffset))
        let v8 = SIMD8<Int32>(repeating: 8)
        for y in 0..<bh {
            let sy = blockY + shiftY + y
            let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
            let r0 = srcBase.advanced(by: sy * width + blockX + shiftX)
            var x = 0
            while x < bw - 7 {
                let h0 = horizontalFIRLuma8(r0, x, vcX0, vcX1, vcX2, vcX3)
                let v = v8 &* h0
                let res16 = SIMD8<Int16>(truncatingIfNeeded: (v &+ vRound) &>> 6)
                let d = UnsafeRawPointer(dstPtr.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self)
                UnsafeMutableRawPointer(dstPtr.advanced(by: x)).storeBytes(of: d &- res16, as: SIMD8<Int16>.self)
                x &+= 8
            }
            while x < bw {
                let v0 = cX0 &* Int32(r0[x - 1]) &+ cX1 &* Int32(r0[x]) &+ cX2 &* Int32(r0[x + 1]) &+ cX3 &* Int32(r0[x + 2])
                let v = Int32(8) &* v0
                let res = Int16(truncatingIfNeeded: (v &+ 31 &+ Int32(roundOffset)) >> 6)
                dstPtr[x] = dstPtr[x] &- res
                x &+= 1
            }
        }
    case fractX == 0:
        let vcY0 = SIMD8<Int32>(repeating: cY0), vcY1 = SIMD8<Int32>(repeating: cY1)
        let vcY2 = SIMD8<Int32>(repeating: cY2), vcY3 = SIMD8<Int32>(repeating: cY3)
        let vRound = SIMD8<Int32>(repeating: Int32(31) &+ Int32(roundOffset))
        let v8 = SIMD8<Int32>(repeating: 8)
        for y in 0..<bh {
            let sy = blockY + shiftY + y
            let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
            let rM1 = srcBase.advanced(by: (sy - 1) * width + blockX + shiftX)
            let r0 = srcBase.advanced(by: sy * width + blockX + shiftX)
            let rP1 = srcBase.advanced(by: (sy + 1) * width + blockX + shiftX)
            let rP2 = srcBase.advanced(by: (sy + 2) * width + blockX + shiftX)
            var x = 0
            while x < bw - 7 {
                let pM1 = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(rM1.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self))
                let p0  = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(r0.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self))
                let pP1 = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(rP1.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self))
                let pP2 = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(rP2.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self))
                let vertFIR = vcY0 &* pM1 &+ vcY1 &* p0 &+ vcY2 &* pP1 &+ vcY3 &* pP2
                let v = v8 &* vertFIR
                let res16 = SIMD8<Int16>(truncatingIfNeeded: (v &+ vRound) &>> 6)
                let d = UnsafeRawPointer(dstPtr.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self)
                UnsafeMutableRawPointer(dstPtr.advanced(by: x)).storeBytes(of: d &- res16, as: SIMD8<Int16>.self)
                x &+= 8
            }
            while x < bw {
                let vM1 = Int32(8) &* Int32(rM1[x])
                let v0  = Int32(8) &* Int32(r0[x])
                let vP1 = Int32(8) &* Int32(rP1[x])
                let vP2 = Int32(8) &* Int32(rP2[x])
                let v = cY0 &* vM1 &+ cY1 &* v0 &+ cY2 &* vP1 &+ cY3 &* vP2
                let res = Int16(truncatingIfNeeded: (v &+ 31 &+ Int32(roundOffset)) >> 6)
                dstPtr[x] = dstPtr[x] &- res
                x &+= 1
            }
        }
    default:
        let vcX0 = SIMD8<Int32>(repeating: cX0), vcX1 = SIMD8<Int32>(repeating: cX1)
        let vcX2 = SIMD8<Int32>(repeating: cX2), vcX3 = SIMD8<Int32>(repeating: cX3)
        let vcY0 = SIMD8<Int32>(repeating: cY0), vcY1 = SIMD8<Int32>(repeating: cY1)
        let vcY2 = SIMD8<Int32>(repeating: cY2), vcY3 = SIMD8<Int32>(repeating: cY3)
        let vRound = SIMD8<Int32>(repeating: Int32(31) &+ Int32(roundOffset))
        for y in 0..<bh {
            let sy = blockY + shiftY + y
            let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
            let rM1 = srcBase.advanced(by: (sy - 1) * width + blockX + shiftX)
            let r0 = srcBase.advanced(by: sy * width + blockX + shiftX)
            let rP1 = srcBase.advanced(by: (sy + 1) * width + blockX + shiftX)
            let rP2 = srcBase.advanced(by: (sy + 2) * width + blockX + shiftX)
            var x = 0
            while x < bw - 7 {
                let hM1 = horizontalFIRLuma8(rM1, x, vcX0, vcX1, vcX2, vcX3)
                let h0  = horizontalFIRLuma8(r0,  x, vcX0, vcX1, vcX2, vcX3)
                let hP1 = horizontalFIRLuma8(rP1, x, vcX0, vcX1, vcX2, vcX3)
                let hP2 = horizontalFIRLuma8(rP2, x, vcX0, vcX1, vcX2, vcX3)
                let v = vcY0 &* hM1 &+ vcY1 &* h0 &+ vcY2 &* hP1 &+ vcY3 &* hP2
                let res16 = SIMD8<Int16>(truncatingIfNeeded: (v &+ vRound) &>> 6)
                let d = UnsafeRawPointer(dstPtr.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self)
                UnsafeMutableRawPointer(dstPtr.advanced(by: x)).storeBytes(of: d &- res16, as: SIMD8<Int16>.self)
                x &+= 8
            }
            while x < bw {
                let vM1 = cX0 &* Int32(rM1[x - 1]) &+ cX1 &* Int32(rM1[x]) &+ cX2 &* Int32(rM1[x + 1]) &+ cX3 &* Int32(rM1[x + 2])
                let v0  = cX0 &* Int32(r0[x - 1])  &+ cX1 &* Int32(r0[x])  &+ cX2 &* Int32(r0[x + 1])  &+ cX3 &* Int32(r0[x + 2])
                let vP1 = cX0 &* Int32(rP1[x - 1]) &+ cX1 &* Int32(rP1[x]) &+ cX2 &* Int32(rP1[x + 1]) &+ cX3 &* Int32(rP1[x + 2])
                let vP2 = cX0 &* Int32(rP2[x - 1]) &+ cX1 &* Int32(rP2[x]) &+ cX2 &* Int32(rP2[x + 1]) &+ cX3 &* Int32(rP2[x + 2])
                let v = cY0 &* vM1 &+ cY1 &* v0 &+ cY2 &* vP1 &+ cY3 &* vP2
                let res = Int16(truncatingIfNeeded: (v &+ 31 &+ Int32(roundOffset)) >> 6)
                dstPtr[x] = dstPtr[x] &- res
                x &+= 1
            }
        }
    }
}

@inline(__always)
fileprivate func subMCBlockChroma16Edge(
    dstBase: UnsafeMutablePointer<Int16>, srcBase: UnsafePointer<Int16>,
    width: Int, height: Int, blockX: Int, blockY: Int,
    shiftX: Int, shiftY: Int, fractX: Int, fractY: Int, bw: Int, bh: Int, roundOffset: Int
) {
    let fX = FIRCHROMACoeffs[fractX]
    let fY = FIRCHROMACoeffs[fractY]
    let cX0 = Int32(fX[0]), cX1 = Int32(fX[1]), cX2 = Int32(fX[2]), cX3 = Int32(fX[3])
    let cY0 = Int32(fY[0]), cY1 = Int32(fY[1]), cY2 = Int32(fY[2]), cY3 = Int32(fY[3])
    let useFIR = (fractX != 0 || fractY != 0)
    if useFIR {
        for y in 0..<bh {
            let sy = blockY + shiftY + y
            let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
            let syM1 = max(0, min(sy - 1, height - 1))
            let sy0  = max(0, min(sy, height - 1))
            let syP1 = max(0, min(sy + 1, height - 1))
            let syP2 = max(0, min(sy + 2, height - 1))
            let rM1 = srcBase.advanced(by: syM1 * width)
            let r0  = srcBase.advanced(by: sy0 * width)
            let rP1 = srcBase.advanced(by: syP1 * width)
            let rP2 = srcBase.advanced(by: syP2 * width)
            for x in 0..<bw {
                let cx = blockX + shiftX + x
                let sxM1 = max(0, min(cx - 1, width - 1))
                let sx0  = max(0, min(cx, width - 1))
                let sxP1 = max(0, min(cx + 1, width - 1))
                let sxP2 = max(0, min(cx + 2, width - 1))
                let vM1 = cX0 &* Int32(rM1[sxM1]) &+ cX1 &* Int32(rM1[sx0]) &+ cX2 &* Int32(rM1[sxP1]) &+ cX3 &* Int32(rM1[sxP2])
                let v0  = cX0 &* Int32(r0[sxM1])  &+ cX1 &* Int32(r0[sx0])  &+ cX2 &* Int32(r0[sxP1])  &+ cX3 &* Int32(r0[sxP2])
                let vP1 = cX0 &* Int32(rP1[sxM1]) &+ cX1 &* Int32(rP1[sx0]) &+ cX2 &* Int32(rP1[sxP1]) &+ cX3 &* Int32(rP1[sxP2])
                let vP2 = cX0 &* Int32(rP2[sxM1]) &+ cX1 &* Int32(rP2[sx0]) &+ cX2 &* Int32(rP2[sxP1]) &+ cX3 &* Int32(rP2[sxP2])
                let v = cY0 &* vM1 &+ cY1 &* v0 &+ cY2 &* vP1 &+ cY3 &* vP2
                let res = Int16(truncatingIfNeeded: (v &+ 31 &+ Int32(roundOffset)) >> 6)
                dstPtr[x] = dstPtr[x] &- res
            }
        }
    } else {
        for y in 0..<bh {
            let sy = blockY + shiftY + y
            let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
            let safeSy = max(0, min(sy, height - 1))
            let r = srcBase.advanced(by: safeSy * width)
            for x in 0..<bw {
                let sx = max(0, min(blockX + shiftX + x, width - 1))
                dstPtr[x] = dstPtr[x] &- r[sx]
            }
        }
    }
}

@inline(__always)
fileprivate func subMCBlockChroma16(
    dstBase: UnsafeMutablePointer<Int16>, srcBase: UnsafePointer<Int16>,
    width: Int, height: Int, blockX: Int, blockY: Int,
    mv: MotionVector, roundOffset: Int, blockSize: Int = 16
) {
    if mv.isIntra { return }
    
    let mvDx = Int(mv.dx)
    let mvDy = Int(mv.dy)
    let shiftX = (mvDx >> 3)
    let shiftY = (mvDy >> 3)
    let fractX = (mvDx & 7)
    let fractY = (mvDy & 7)
    
    let bw = min(blockSize, width - blockX)
    let bh = min(blockSize, height - blockY)
    if bw <= 0 || bh <= 0 { return }
    
    let safe = (0 <= blockX + shiftX - 1) && (0 <= blockY + shiftY - 1) && (blockX + shiftX + bw + 2 < width) && (blockY + shiftY + bh + 2 < height)
    if safe {
        subMCBlockChroma16Inner(dstBase: dstBase, srcBase: srcBase, width: width, height: height, blockX: blockX, blockY: blockY, shiftX: shiftX, shiftY: shiftY, fractX: fractX, fractY: fractY, bw: bw, bh: bh, roundOffset: roundOffset)
    } else {
        subMCBlockChroma16Edge(dstBase: dstBase, srcBase: srcBase, width: width, height: height, blockX: blockX, blockY: blockY, shiftX: shiftX, shiftY: shiftY, fractX: fractX, fractY: fractY, bw: bw, bh: bh, roundOffset: roundOffset)
    }
}

@inline(__always)
fileprivate func addMCBlockLuma32Inner(
    dstBase: UnsafeMutablePointer<Int16>, srcBase: UnsafePointer<Int16>,
    width: Int, height: Int, blockX: Int, blockY: Int,
    shiftX: Int, shiftY: Int, fractX: Int, fractY: Int, bw: Int, bh: Int, roundOffset: Int
) {
    let fX = FIRLUMACoeffs[fractX]
    let fY = FIRLUMACoeffs[fractY]
    let cX0 = Int32(fX[0]), cX1 = Int32(fX[1]), cX2 = Int32(fX[2]), cX3 = Int32(fX[3])
    let cY0 = Int32(fY[0]), cY1 = Int32(fY[1]), cY2 = Int32(fY[2]), cY3 = Int32(fY[3])
    let useFIR = (fractX != 0 || fractY != 0)
    
    if useFIR != true {
        if bw == 32 {
            for y in 0..<bh {
                let sy = blockY + shiftY + y
                let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
                let r = srcBase.advanced(by: sy * width + blockX + shiftX)
                let d0 = UnsafeRawPointer(dstPtr).loadUnaligned(as: SIMD16<Int16>.self)
                let s0 = UnsafeRawPointer(r).loadUnaligned(as: SIMD16<Int16>.self)
                let d1 = UnsafeRawPointer(dstPtr.advanced(by: 16)).loadUnaligned(as: SIMD16<Int16>.self)
                let s1 = UnsafeRawPointer(r.advanced(by: 16)).loadUnaligned(as: SIMD16<Int16>.self)
                UnsafeMutableRawPointer(dstPtr).storeBytes(of: d0 &+ s0, as: SIMD16<Int16>.self)
                UnsafeMutableRawPointer(dstPtr.advanced(by: 16)).storeBytes(of: d1 &+ s1, as: SIMD16<Int16>.self)
            }
            return
        }
        for y in 0..<bh {
            let sy = blockY + shiftY + y
            let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
            let r = srcBase.advanced(by: sy * width + blockX + shiftX)
            var x = 0
            while x < bw - 31 {
                let d0 = UnsafeRawPointer(dstPtr.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
                let s0 = UnsafeRawPointer(r.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
                let d1 = UnsafeRawPointer(dstPtr.advanced(by: x + 16)).loadUnaligned(as: SIMD16<Int16>.self)
                let s1 = UnsafeRawPointer(r.advanced(by: x + 16)).loadUnaligned(as: SIMD16<Int16>.self)
                UnsafeMutableRawPointer(dstPtr.advanced(by: x)).storeBytes(of: d0 &+ s0, as: SIMD16<Int16>.self)
                UnsafeMutableRawPointer(dstPtr.advanced(by: x + 16)).storeBytes(of: d1 &+ s1, as: SIMD16<Int16>.self)
                x &+= 32
            }
            while x < bw - 15 {
                let d = UnsafeRawPointer(dstPtr.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
                let s = UnsafeRawPointer(r.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
                UnsafeMutableRawPointer(dstPtr.advanced(by: x)).storeBytes(of: d &+ s, as: SIMD16<Int16>.self)
                x &+= 16
            }
            while x < bw { dstPtr[x] = dstPtr[x] &+ r[x]; x &+= 1 }
        }
        return
    }
    
    if fractY == 0 {
        let vcX0 = SIMD16<Int32>(repeating: cX0), vcX1 = SIMD16<Int32>(repeating: cX1)
        let vcX2 = SIMD16<Int32>(repeating: cX2), vcX3 = SIMD16<Int32>(repeating: cX3)
        let vRound = SIMD16<Int32>(repeating: Int32(31) &+ Int32(roundOffset))
        let v8 = SIMD16<Int32>(repeating: 8)
        for y in 0..<bh {
            let sy = blockY + shiftY + y
            let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
            let r0 = srcBase.advanced(by: sy * width + blockX + shiftX)
            var x = 0
            while x < bw - 15 {
                let h0 = horizontalFIRLuma16(r0, x, vcX0, vcX1, vcX2, vcX3)
                let res16 = SIMD16<Int16>(truncatingIfNeeded: ((v8 &* h0) &+ vRound) &>> 6)
                let d = UnsafeRawPointer(dstPtr.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
                UnsafeMutableRawPointer(dstPtr.advanced(by: x)).storeBytes(of: d &+ res16, as: SIMD16<Int16>.self)
                x &+= 16
            }
            while x < bw {
                let vM1 = cX0 &* Int32(r0[x - 1]) &+ cX1 &* Int32(r0[x]) &+ cX2 &* Int32(r0[x + 1]) &+ cX3 &* Int32(r0[x + 2])
                let v = Int32(8) &* vM1
                let res = Int16(truncatingIfNeeded: (v &+ 31 &+ Int32(roundOffset)) >> 6)
                dstPtr[x] = dstPtr[x] &+ res
                x &+= 1
            }
        }
        return
    }
    
    if fractX == 0 {
        let vcY0 = SIMD16<Int32>(repeating: cY0), vcY1 = SIMD16<Int32>(repeating: cY1)
        let vcY2 = SIMD16<Int32>(repeating: cY2), vcY3 = SIMD16<Int32>(repeating: cY3)
        let vRound = SIMD16<Int32>(repeating: Int32(31) &+ Int32(roundOffset))
        let v8 = SIMD16<Int32>(repeating: 8)
        
        var x = 0
        while x < bw - 15 {
            let srcX = srcBase.advanced(by: blockX + shiftX + x)
            let sy0 = blockY + shiftY
            
            var pM1 = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(srcX.advanced(by: (sy0 - 1) * width)).loadUnaligned(as: SIMD16<Int16>.self))
            var p0  = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(srcX.advanced(by: sy0 * width)).loadUnaligned(as: SIMD16<Int16>.self))
            var pP1 = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(srcX.advanced(by: (sy0 + 1) * width)).loadUnaligned(as: SIMD16<Int16>.self))
            
            for y in 0..<bh {
                let pP2 = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(srcX.advanced(by: (sy0 + y + 2) * width)).loadUnaligned(as: SIMD16<Int16>.self))
                let vertFIR = vcY0 &* pM1 &+ vcY1 &* p0 &+ vcY2 &* pP1 &+ vcY3 &* pP2
                let res16 = SIMD16<Int16>(truncatingIfNeeded: ((v8 &* vertFIR) &+ vRound) &>> 6)
                
                let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX + x)
                let d = UnsafeRawPointer(dstPtr).loadUnaligned(as: SIMD16<Int16>.self)
                UnsafeMutableRawPointer(dstPtr).storeBytes(of: d &+ res16, as: SIMD16<Int16>.self)
                
                pM1 = p0
                p0 = pP1
                pP1 = pP2
            }
            x &+= 16
        }
        while x < bw {
            for y in 0..<bh {
                let sy = blockY + shiftY + y
                let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
                let rM1 = srcBase.advanced(by: (sy - 1) * width + blockX + shiftX)
                let r0 = srcBase.advanced(by: sy * width + blockX + shiftX)
                let rP1 = srcBase.advanced(by: (sy + 1) * width + blockX + shiftX)
                let rP2 = srcBase.advanced(by: (sy + 2) * width + blockX + shiftX)
                
                let vertFIR = cY0 &* Int32(rM1[x]) &+ cY1 &* Int32(r0[x]) &+ cY2 &* Int32(rP1[x]) &+ cY3 &* Int32(rP2[x])
                let v = Int32(8) &* vertFIR
                let res = Int16(truncatingIfNeeded: (v &+ 31 &+ Int32(roundOffset)) >> 6)
                dstPtr[x] = dstPtr[x] &+ res
            }
            x &+= 1
        }
        return
    }
    
    // Full 2D FIR
    let vcX0 = SIMD16<Int32>(repeating: cX0), vcX1 = SIMD16<Int32>(repeating: cX1)
    let vcX2 = SIMD16<Int32>(repeating: cX2), vcX3 = SIMD16<Int32>(repeating: cX3)
    let vcY0 = SIMD16<Int32>(repeating: cY0), vcY1 = SIMD16<Int32>(repeating: cY1)
    let vcY2 = SIMD16<Int32>(repeating: cY2), vcY3 = SIMD16<Int32>(repeating: cY3)
    let vRound = SIMD16<Int32>(repeating: Int32(31) &+ Int32(roundOffset))
    
    var x = 0
    while x < bw - 15 {
        let srcX = srcBase.advanced(by: blockX + shiftX)
        let sy0 = blockY + shiftY
        
        var hM1 = horizontalFIRLuma16(srcX.advanced(by: (sy0 - 1) * width), x, vcX0, vcX1, vcX2, vcX3)
        var h0  = horizontalFIRLuma16(srcX.advanced(by: sy0 * width), x, vcX0, vcX1, vcX2, vcX3)
        var hP1 = horizontalFIRLuma16(srcX.advanced(by: (sy0 + 1) * width), x, vcX0, vcX1, vcX2, vcX3)
        
        for y in 0..<bh {
            let hP2 = horizontalFIRLuma16(srcX.advanced(by: (sy0 + y + 2) * width), x, vcX0, vcX1, vcX2, vcX3)
            let v = vcY0 &* hM1 &+ vcY1 &* h0 &+ vcY2 &* hP1 &+ vcY3 &* hP2
            let res16 = SIMD16<Int16>(truncatingIfNeeded: (v &+ vRound) &>> 6)
            
            let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX + x)
            let d = UnsafeRawPointer(dstPtr).loadUnaligned(as: SIMD16<Int16>.self)
            UnsafeMutableRawPointer(dstPtr).storeBytes(of: d &+ res16, as: SIMD16<Int16>.self)
            
            hM1 = h0
            h0 = hP1
            hP1 = hP2
        }
        x &+= 16
    }
    
    while x < bw {
        for y in 0..<bh {
            let sy = blockY + shiftY + y
            let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
            let rM1 = srcBase.advanced(by: (sy - 1) * width + blockX + shiftX)
            let r0 = srcBase.advanced(by: sy * width + blockX + shiftX)
            let rP1 = srcBase.advanced(by: (sy + 1) * width + blockX + shiftX)
            let rP2 = srcBase.advanced(by: (sy + 2) * width + blockX + shiftX)
            
            let vM1 = cX0 &* Int32(rM1[x - 1]) &+ cX1 &* Int32(rM1[x]) &+ cX2 &* Int32(rM1[x + 1]) &+ cX3 &* Int32(rM1[x + 2])
            let v0  = cX0 &* Int32(r0[x - 1])  &+ cX1 &* Int32(r0[x])  &+ cX2 &* Int32(r0[x + 1])  &+ cX3 &* Int32(r0[x + 2])
            let vP1 = cX0 &* Int32(rP1[x - 1]) &+ cX1 &* Int32(rP1[x]) &+ cX2 &* Int32(rP1[x + 1]) &+ cX3 &* Int32(rP1[x + 2])
            let vP2 = cX0 &* Int32(rP2[x - 1]) &+ cX1 &* Int32(rP2[x]) &+ cX2 &* Int32(rP2[x + 1]) &+ cX3 &* Int32(rP2[x + 2])
            let v = cY0 &* vM1 &+ cY1 &* v0 &+ cY2 &* vP1 &+ cY3 &* vP2
            let res = Int16(truncatingIfNeeded: (v &+ 31 &+ Int32(roundOffset)) >> 6)
            dstPtr[x] = dstPtr[x] &+ res
        }
        x &+= 1
    }
}

@inline(__always)
fileprivate func addMCBlockLuma32Edge(
    dstBase: UnsafeMutablePointer<Int16>, srcBase: UnsafePointer<Int16>,
    width: Int, height: Int, blockX: Int, blockY: Int,
    shiftX: Int, shiftY: Int, fractX: Int, fractY: Int, bw: Int, bh: Int, roundOffset: Int
) {
    let fX = FIRLUMACoeffs[fractX]
    let fY = FIRLUMACoeffs[fractY]
    let cX0 = Int32(fX[0]), cX1 = Int32(fX[1]), cX2 = Int32(fX[2]), cX3 = Int32(fX[3])
    let cY0 = Int32(fY[0]), cY1 = Int32(fY[1]), cY2 = Int32(fY[2]), cY3 = Int32(fY[3])
    let useFIR = (fractX != 0 || fractY != 0)
    if useFIR  {
        for y in 0..<bh {
            let sy = blockY + shiftY + y
            let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
            let syM1 = max(0, min(sy - 1, height - 1))
            let sy0  = max(0, min(sy, height - 1))
            let syP1 = max(0, min(sy + 1, height - 1))
            let syP2 = max(0, min(sy + 2, height - 1))
            let rM1 = srcBase.advanced(by: syM1 * width)
            let r0  = srcBase.advanced(by: sy0 * width)
            let rP1 = srcBase.advanced(by: syP1 * width)
            let rP2 = srcBase.advanced(by: syP2 * width)
            for x in 0..<bw {
                let cx = blockX + shiftX + x
                let sxM1 = max(0, min(cx - 1, width - 1))
                let sx0  = max(0, min(cx, width - 1))
                let sxP1 = max(0, min(cx + 1, width - 1))
                let sxP2 = max(0, min(cx + 2, width - 1))
                let vM1 = cX0 &* Int32(rM1[sxM1]) &+ cX1 &* Int32(rM1[sx0]) &+ cX2 &* Int32(rM1[sxP1]) &+ cX3 &* Int32(rM1[sxP2])
                let v0  = cX0 &* Int32(r0[sxM1])  &+ cX1 &* Int32(r0[sx0])  &+ cX2 &* Int32(r0[sxP1])  &+ cX3 &* Int32(r0[sxP2])
                let vP1 = cX0 &* Int32(rP1[sxM1]) &+ cX1 &* Int32(rP1[sx0]) &+ cX2 &* Int32(rP1[sxP1]) &+ cX3 &* Int32(rP1[sxP2])
                let vP2 = cX0 &* Int32(rP2[sxM1]) &+ cX1 &* Int32(rP2[sx0]) &+ cX2 &* Int32(rP2[sxP1]) &+ cX3 &* Int32(rP2[sxP2])
                let v = cY0 &* vM1 &+ cY1 &* v0 &+ cY2 &* vP1 &+ cY3 &* vP2
                let res = Int16(truncatingIfNeeded: (v &+ 31 &+ Int32(roundOffset)) >> 6)
                dstPtr[x] = dstPtr[x] &+ res
            }
        }
    } else {
        for y in 0..<bh {
            let sy = blockY + shiftY + y
            let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
            let safeSy = max(0, min(sy, height - 1))
            let r = srcBase.advanced(by: safeSy * width)
            for x in 0..<bw {
                let sx = max(0, min(blockX + shiftX + x, width - 1))
                dstPtr[x] = dstPtr[x] &+ r[sx]
            }
        }
    }
}

@inline(__always)
fileprivate func addMCBlockLuma32(
    dstBase: UnsafeMutablePointer<Int16>, srcBase: UnsafePointer<Int16>,
    width: Int, height: Int, blockX: Int, blockY: Int,
    mv: MotionVector, roundOffset: Int, blockSize: Int = 32
) {
    if mv.isIntra { return }
    
    let mvDx = Int(mv.dx)
    let mvDy = Int(mv.dy)
    let shiftX = (mvDx >> 2)
    let shiftY = (mvDy >> 2)
    let fractX = (mvDx & 3)
    let fractY = (mvDy & 3)
    
    let bw = min(blockSize, width - blockX)
    let bh = min(blockSize, height - blockY)
    if bw <= 0 || bh <= 0 { return }
    
    let srcX = blockX + shiftX
    let srcY = blockY + shiftY
    let safe = 1 <= srcX && 1 <= srcY && (srcX &+ bw &+ 2) < width && (srcY &+ bh &+ 2) < height
    if safe {
        if fractX == 0 && fractY == 0 && bw == 32 && bh == 32 {
            for y in 0..<32 {
                let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
                let r = srcBase.advanced(by: (srcY + y) * width + srcX)
                let d0 = UnsafeRawPointer(dstPtr).loadUnaligned(as: SIMD16<Int16>.self)
                let s0 = UnsafeRawPointer(r).loadUnaligned(as: SIMD16<Int16>.self)
                let d1 = UnsafeRawPointer(dstPtr.advanced(by: 16)).loadUnaligned(as: SIMD16<Int16>.self)
                let s1 = UnsafeRawPointer(r.advanced(by: 16)).loadUnaligned(as: SIMD16<Int16>.self)
                UnsafeMutableRawPointer(dstPtr).storeBytes(of: d0 &+ s0, as: SIMD16<Int16>.self)
                UnsafeMutableRawPointer(dstPtr.advanced(by: 16)).storeBytes(of: d1 &+ s1, as: SIMD16<Int16>.self)
            }
            return
        }
        addMCBlockLuma32Inner(dstBase: dstBase, srcBase: srcBase, width: width, height: height, blockX: blockX, blockY: blockY, shiftX: shiftX, shiftY: shiftY, fractX: fractX, fractY: fractY, bw: bw, bh: bh, roundOffset: roundOffset)
    } else {
        addMCBlockLuma32Edge(dstBase: dstBase, srcBase: srcBase, width: width, height: height, blockX: blockX, blockY: blockY, shiftX: shiftX, shiftY: shiftY, fractX: fractX, fractY: fractY, bw: bw, bh: bh, roundOffset: roundOffset)
    }
}

@inline(__always)
fileprivate func addMCBlockChroma16Inner(
    dstBase: UnsafeMutablePointer<Int16>, srcBase: UnsafePointer<Int16>,
    width: Int, height: Int, blockX: Int, blockY: Int,
    shiftX: Int, shiftY: Int, fractX: Int, fractY: Int, bw: Int, bh: Int, roundOffset: Int
) {
    let fX = FIRCHROMACoeffs[fractX]
    let fY = FIRCHROMACoeffs[fractY]
    let cX0 = Int32(fX[0]), cX1 = Int32(fX[1]), cX2 = Int32(fX[2]), cX3 = Int32(fX[3])
    let cY0 = Int32(fY[0]), cY1 = Int32(fY[1]), cY2 = Int32(fY[2]), cY3 = Int32(fY[3])
    let useFIR = (fractX != 0 || fractY != 0)
    switch true {
    case useFIR != true:
        if bw == 16 {
            for y in 0..<bh {
                let sy = blockY + shiftY + y
                let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
                let r = srcBase.advanced(by: sy * width + blockX + shiftX)
                let d = UnsafeRawPointer(dstPtr).loadUnaligned(as: SIMD16<Int16>.self)
                let s = UnsafeRawPointer(r).loadUnaligned(as: SIMD16<Int16>.self)
                UnsafeMutableRawPointer(dstPtr).storeBytes(of: d &+ s, as: SIMD16<Int16>.self)
            }
        } else {
            for y in 0..<bh {
                let sy = blockY + shiftY + y
                let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
                let r = srcBase.advanced(by: sy * width + blockX + shiftX)
                var x = 0
                while x < bw - 31 {
                    let d0 = UnsafeRawPointer(dstPtr.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
                    let s0 = UnsafeRawPointer(r.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
                    let d1 = UnsafeRawPointer(dstPtr.advanced(by: x + 16)).loadUnaligned(as: SIMD16<Int16>.self)
                    let s1 = UnsafeRawPointer(r.advanced(by: x + 16)).loadUnaligned(as: SIMD16<Int16>.self)
                    UnsafeMutableRawPointer(dstPtr.advanced(by: x)).storeBytes(of: d0 &+ s0, as: SIMD16<Int16>.self)
                    UnsafeMutableRawPointer(dstPtr.advanced(by: x + 16)).storeBytes(of: d1 &+ s1, as: SIMD16<Int16>.self)
                    x &+= 32
                }
                while x < bw - 15 {
                    let d = UnsafeRawPointer(dstPtr.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
                    let s = UnsafeRawPointer(r.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
                    UnsafeMutableRawPointer(dstPtr.advanced(by: x)).storeBytes(of: d &+ s, as: SIMD16<Int16>.self)
                    x &+= 16
                }
                while x < bw { dstPtr[x] = dstPtr[x] &+ r[x]; x &+= 1 }
            }
        }
    case fractY == 0:
        let vcX0 = SIMD8<Int32>(repeating: cX0), vcX1 = SIMD8<Int32>(repeating: cX1)
        let vcX2 = SIMD8<Int32>(repeating: cX2), vcX3 = SIMD8<Int32>(repeating: cX3)
        let vRound = SIMD8<Int32>(repeating: Int32(31) &+ Int32(roundOffset))
        let v8 = SIMD8<Int32>(repeating: 8)
        for y in 0..<bh {
            let sy = blockY + shiftY + y
            let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
            let r0 = srcBase.advanced(by: sy * width + blockX + shiftX)
            var x = 0
            while x < bw - 15 {
                let h0 = horizontalFIRLuma8(r0, x, vcX0, vcX1, vcX2, vcX3)
                let h1 = horizontalFIRLuma8(r0, x + 8, vcX0, vcX1, vcX2, vcX3)
                
                let v0 = v8 &* h0
                let v1 = v8 &* h1
                
                let res16_0 = SIMD8<Int16>(truncatingIfNeeded: (v0 &+ vRound) &>> 6)
                let res16_1 = SIMD8<Int16>(truncatingIfNeeded: (v1 &+ vRound) &>> 6)
                
                let d0 = UnsafeRawPointer(dstPtr.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self)
                let d1 = UnsafeRawPointer(dstPtr.advanced(by: x + 8)).loadUnaligned(as: SIMD8<Int16>.self)
                
                UnsafeMutableRawPointer(dstPtr.advanced(by: x)).storeBytes(of: d0 &+ res16_0, as: SIMD8<Int16>.self)
                UnsafeMutableRawPointer(dstPtr.advanced(by: x + 8)).storeBytes(of: d1 &+ res16_1, as: SIMD8<Int16>.self)
                x &+= 16
            }
            while x < bw - 7 {
                let h0 = horizontalFIRLuma8(r0, x, vcX0, vcX1, vcX2, vcX3)
                let v = v8 &* h0
                let res16 = SIMD8<Int16>(truncatingIfNeeded: (v &+ vRound) &>> 6)
                let d = UnsafeRawPointer(dstPtr.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self)
                UnsafeMutableRawPointer(dstPtr.advanced(by: x)).storeBytes(of: d &+ res16, as: SIMD8<Int16>.self)
                x &+= 8
            }
            while x < bw {
                let v0 = cX0 &* Int32(r0[x - 1]) &+ cX1 &* Int32(r0[x]) &+ cX2 &* Int32(r0[x + 1]) &+ cX3 &* Int32(r0[x + 2])
                let v = Int32(8) &* v0
                let res = Int16(truncatingIfNeeded: (v &+ 31 &+ Int32(roundOffset)) >> 6)
                dstPtr[x] = dstPtr[x] &+ res
                x &+= 1
            }
        }
    case fractX == 0:
        let vcY0 = SIMD8<Int32>(repeating: cY0), vcY1 = SIMD8<Int32>(repeating: cY1)
        let vcY2 = SIMD8<Int32>(repeating: cY2), vcY3 = SIMD8<Int32>(repeating: cY3)
        let vRound = SIMD8<Int32>(repeating: Int32(31) &+ Int32(roundOffset))
        let v8 = SIMD8<Int32>(repeating: 8)
        for y in 0..<bh {
            let sy = blockY + shiftY + y
            let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
            let rM1 = srcBase.advanced(by: (sy - 1) * width + blockX + shiftX)
            let r0 = srcBase.advanced(by: sy * width + blockX + shiftX)
            let rP1 = srcBase.advanced(by: (sy + 1) * width + blockX + shiftX)
            let rP2 = srcBase.advanced(by: (sy + 2) * width + blockX + shiftX)
            var x = 0
            while x < bw - 15 {
                let pM1_0 = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(rM1.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self))
                let p0_0  = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(r0.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self))
                let pP1_0 = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(rP1.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self))
                let pP2_0 = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(rP2.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self))
                
                let pM1_1 = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(rM1.advanced(by: x + 8)).loadUnaligned(as: SIMD8<Int16>.self))
                let p0_1  = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(r0.advanced(by: x + 8)).loadUnaligned(as: SIMD8<Int16>.self))
                let pP1_1 = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(rP1.advanced(by: x + 8)).loadUnaligned(as: SIMD8<Int16>.self))
                let pP2_1 = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(rP2.advanced(by: x + 8)).loadUnaligned(as: SIMD8<Int16>.self))
                
                let vertFIR_0 = vcY0 &* pM1_0 &+ vcY1 &* p0_0 &+ vcY2 &* pP1_0 &+ vcY3 &* pP2_0
                let vertFIR_1 = vcY0 &* pM1_1 &+ vcY1 &* p0_1 &+ vcY2 &* pP1_1 &+ vcY3 &* pP2_1
                
                let v_0 = v8 &* vertFIR_0
                let v_1 = v8 &* vertFIR_1
                
                let res16_0 = SIMD8<Int16>(truncatingIfNeeded: (v_0 &+ vRound) &>> 6)
                let res16_1 = SIMD8<Int16>(truncatingIfNeeded: (v_1 &+ vRound) &>> 6)
                
                let d_0 = UnsafeRawPointer(dstPtr.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self)
                let d_1 = UnsafeRawPointer(dstPtr.advanced(by: x + 8)).loadUnaligned(as: SIMD8<Int16>.self)
                
                UnsafeMutableRawPointer(dstPtr.advanced(by: x)).storeBytes(of: d_0 &+ res16_0, as: SIMD8<Int16>.self)
                UnsafeMutableRawPointer(dstPtr.advanced(by: x + 8)).storeBytes(of: d_1 &+ res16_1, as: SIMD8<Int16>.self)
                
                x &+= 16
            }
            while x < bw - 7 {
                let pM1 = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(rM1.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self))
                let p0  = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(r0.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self))
                let pP1 = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(rP1.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self))
                let pP2 = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(rP2.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self))
                let vertFIR = vcY0 &* pM1 &+ vcY1 &* p0 &+ vcY2 &* pP1 &+ vcY3 &* pP2
                let v = v8 &* vertFIR
                let res16 = SIMD8<Int16>(truncatingIfNeeded: (v &+ vRound) &>> 6)
                let d = UnsafeRawPointer(dstPtr.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self)
                UnsafeMutableRawPointer(dstPtr.advanced(by: x)).storeBytes(of: d &+ res16, as: SIMD8<Int16>.self)
                x &+= 8
            }
            while x < bw {
                let vM1 = Int32(8) &* Int32(rM1[x])
                let v0  = Int32(8) &* Int32(r0[x])
                let vP1 = Int32(8) &* Int32(rP1[x])
                let vP2 = Int32(8) &* Int32(rP2[x])
                let v = cY0 &* vM1 &+ cY1 &* v0 &+ cY2 &* vP1 &+ cY3 &* vP2
                let res = Int16(truncatingIfNeeded: (v &+ 31 &+ Int32(roundOffset)) >> 6)
                dstPtr[x] = dstPtr[x] &+ res
                x &+= 1
            }
        }
    default:
        let vcX0 = SIMD8<Int32>(repeating: cX0), vcX1 = SIMD8<Int32>(repeating: cX1)
        let vcX2 = SIMD8<Int32>(repeating: cX2), vcX3 = SIMD8<Int32>(repeating: cX3)
        let vcY0 = SIMD8<Int32>(repeating: cY0), vcY1 = SIMD8<Int32>(repeating: cY1)
        let vcY2 = SIMD8<Int32>(repeating: cY2), vcY3 = SIMD8<Int32>(repeating: cY3)
        let vRound = SIMD8<Int32>(repeating: Int32(31) &+ Int32(roundOffset))
        for y in 0..<bh {
            let sy = blockY + shiftY + y
            let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
            let rM1 = srcBase.advanced(by: (sy - 1) * width + blockX + shiftX)
            let r0 = srcBase.advanced(by: sy * width + blockX + shiftX)
            let rP1 = srcBase.advanced(by: (sy + 1) * width + blockX + shiftX)
            let rP2 = srcBase.advanced(by: (sy + 2) * width + blockX + shiftX)
            var x = 0
            while x < bw - 15 {
                let hM1_0 = horizontalFIRLuma8(rM1, x, vcX0, vcX1, vcX2, vcX3)
                let h0_0  = horizontalFIRLuma8(r0,  x, vcX0, vcX1, vcX2, vcX3)
                let hP1_0 = horizontalFIRLuma8(rP1, x, vcX0, vcX1, vcX2, vcX3)
                let hP2_0 = horizontalFIRLuma8(rP2, x, vcX0, vcX1, vcX2, vcX3)
                
                let hM1_1 = horizontalFIRLuma8(rM1, x + 8, vcX0, vcX1, vcX2, vcX3)
                let h0_1  = horizontalFIRLuma8(r0,  x + 8, vcX0, vcX1, vcX2, vcX3)
                let hP1_1 = horizontalFIRLuma8(rP1, x + 8, vcX0, vcX1, vcX2, vcX3)
                let hP2_1 = horizontalFIRLuma8(rP2, x + 8, vcX0, vcX1, vcX2, vcX3)
                
                let v_0 = vcY0 &* hM1_0 &+ vcY1 &* h0_0 &+ vcY2 &* hP1_0 &+ vcY3 &* hP2_0
                let v_1 = vcY0 &* hM1_1 &+ vcY1 &* h0_1 &+ vcY2 &* hP1_1 &+ vcY3 &* hP2_1
                
                let res16_0 = SIMD8<Int16>(truncatingIfNeeded: (v_0 &+ vRound) &>> 6)
                let res16_1 = SIMD8<Int16>(truncatingIfNeeded: (v_1 &+ vRound) &>> 6)
                
                let d_0 = UnsafeRawPointer(dstPtr.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self)
                let d_1 = UnsafeRawPointer(dstPtr.advanced(by: x + 8)).loadUnaligned(as: SIMD8<Int16>.self)
                
                UnsafeMutableRawPointer(dstPtr.advanced(by: x)).storeBytes(of: d_0 &+ res16_0, as: SIMD8<Int16>.self)
                UnsafeMutableRawPointer(dstPtr.advanced(by: x + 8)).storeBytes(of: d_1 &+ res16_1, as: SIMD8<Int16>.self)
                
                x &+= 16
            }
            while x < bw - 7 {
                let hM1 = horizontalFIRLuma8(rM1, x, vcX0, vcX1, vcX2, vcX3)
                let h0  = horizontalFIRLuma8(r0,  x, vcX0, vcX1, vcX2, vcX3)
                let hP1 = horizontalFIRLuma8(rP1, x, vcX0, vcX1, vcX2, vcX3)
                let hP2 = horizontalFIRLuma8(rP2, x, vcX0, vcX1, vcX2, vcX3)
                let v = vcY0 &* hM1 &+ vcY1 &* h0 &+ vcY2 &* hP1 &+ vcY3 &* hP2
                let res16 = SIMD8<Int16>(truncatingIfNeeded: (v &+ vRound) &>> 6)
                let d = UnsafeRawPointer(dstPtr.advanced(by: x)).loadUnaligned(as: SIMD8<Int16>.self)
                UnsafeMutableRawPointer(dstPtr.advanced(by: x)).storeBytes(of: d &+ res16, as: SIMD8<Int16>.self)
                x &+= 8
            }
            while x < bw {
                let vM1 = cX0 &* Int32(rM1[x - 1]) &+ cX1 &* Int32(rM1[x]) &+ cX2 &* Int32(rM1[x + 1]) &+ cX3 &* Int32(rM1[x + 2])
                let v0  = cX0 &* Int32(r0[x - 1])  &+ cX1 &* Int32(r0[x])  &+ cX2 &* Int32(r0[x + 1])  &+ cX3 &* Int32(r0[x + 2])
                let vP1 = cX0 &* Int32(rP1[x - 1]) &+ cX1 &* Int32(rP1[x]) &+ cX2 &* Int32(rP1[x + 1]) &+ cX3 &* Int32(rP1[x + 2])
                let vP2 = cX0 &* Int32(rP2[x - 1]) &+ cX1 &* Int32(rP2[x]) &+ cX2 &* Int32(rP2[x + 1]) &+ cX3 &* Int32(rP2[x + 2])
                let v = cY0 &* vM1 &+ cY1 &* v0 &+ cY2 &* vP1 &+ cY3 &* vP2
                let res = Int16(truncatingIfNeeded: (v &+ 31 &+ Int32(roundOffset)) >> 6)
                dstPtr[x] = dstPtr[x] &+ res
                x &+= 1
            }
        }
    }
}

@inline(__always)
fileprivate func addMCBlockChroma16Edge(
    dstBase: UnsafeMutablePointer<Int16>, srcBase: UnsafePointer<Int16>,
    width: Int, height: Int, blockX: Int, blockY: Int,
    shiftX: Int, shiftY: Int, fractX: Int, fractY: Int, bw: Int, bh: Int, roundOffset: Int
) {
    let fX = FIRCHROMACoeffs[fractX]
    let fY = FIRCHROMACoeffs[fractY]
    let cX0 = Int32(fX[0]), cX1 = Int32(fX[1]), cX2 = Int32(fX[2]), cX3 = Int32(fX[3])
    let cY0 = Int32(fY[0]), cY1 = Int32(fY[1]), cY2 = Int32(fY[2]), cY3 = Int32(fY[3])
    let useFIR = (fractX != 0 || fractY != 0)
    if useFIR {
        for y in 0..<bh {
            let sy = blockY + shiftY + y
            let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
            let syM1 = max(0, min(sy - 1, height - 1))
            let sy0  = max(0, min(sy, height - 1))
            let syP1 = max(0, min(sy + 1, height - 1))
            let syP2 = max(0, min(sy + 2, height - 1))
            let rM1 = srcBase.advanced(by: syM1 * width)
            let r0  = srcBase.advanced(by: sy0 * width)
            let rP1 = srcBase.advanced(by: syP1 * width)
            let rP2 = srcBase.advanced(by: syP2 * width)
            for x in 0..<bw {
                let cx = blockX + shiftX + x
                let sxM1 = max(0, min(cx - 1, width - 1))
                let sx0  = max(0, min(cx, width - 1))
                let sxP1 = max(0, min(cx + 1, width - 1))
                let sxP2 = max(0, min(cx + 2, width - 1))
                let vM1 = cX0 &* Int32(rM1[sxM1]) &+ cX1 &* Int32(rM1[sx0]) &+ cX2 &* Int32(rM1[sxP1]) &+ cX3 &* Int32(rM1[sxP2])
                let v0  = cX0 &* Int32(r0[sxM1])  &+ cX1 &* Int32(r0[sx0])  &+ cX2 &* Int32(r0[sxP1])  &+ cX3 &* Int32(r0[sxP2])
                let vP1 = cX0 &* Int32(rP1[sxM1]) &+ cX1 &* Int32(rP1[sx0]) &+ cX2 &* Int32(rP1[sxP1]) &+ cX3 &* Int32(rP1[sxP2])
                let vP2 = cX0 &* Int32(rP2[sxM1]) &+ cX1 &* Int32(rP2[sx0]) &+ cX2 &* Int32(rP2[sxP1]) &+ cX3 &* Int32(rP2[sxP2])
                let v = cY0 &* vM1 &+ cY1 &* v0 &+ cY2 &* vP1 &+ cY3 &* vP2
                let res = Int16(truncatingIfNeeded: (v &+ 31 &+ Int32(roundOffset)) >> 6)
                dstPtr[x] = dstPtr[x] &+ res
            }
        }
    } else {
        for y in 0..<bh {
            let sy = blockY + shiftY + y
            let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
            let safeSy = max(0, min(sy, height - 1))
            let r = srcBase.advanced(by: safeSy * width)
            for x in 0..<bw {
                let sx = max(0, min(blockX + shiftX + x, width - 1))
                dstPtr[x] = dstPtr[x] &+ r[sx]
            }
        }
    }
}

@inline(__always)
fileprivate func addMCBlockChroma16(
    dstBase: UnsafeMutablePointer<Int16>, srcBase: UnsafePointer<Int16>,
    width: Int, height: Int, blockX: Int, blockY: Int,
    mv: MotionVector, roundOffset: Int, blockSize: Int = 16
) {
    if mv.isIntra { return }
    
    let mvDx = Int(mv.dx)
    let mvDy = Int(mv.dy)
    let shiftX = (mvDx >> 3)
    let shiftY = (mvDy >> 3)
    let fractX = (mvDx & 7)
    let fractY = (mvDy & 7)
    
    let bw = min(blockSize, width - blockX)
    let bh = min(blockSize, height - blockY)
    if bw <= 0 || bh <= 0 { return }
    
    let safe = (0 <= blockX + shiftX - 1) && (0 <= blockY + shiftY - 1) && (blockX + shiftX + bw + 2 < width) && (blockY + shiftY + bh + 2 < height)
    
    if safe {
        addMCBlockChroma16Inner(dstBase: dstBase, srcBase: srcBase, width: width, height: height, blockX: blockX, blockY: blockY, shiftX: shiftX, shiftY: shiftY, fractX: fractX, fractY: fractY, bw: bw, bh: bh, roundOffset: roundOffset)
    } else {
        addMCBlockChroma16Edge(dstBase: dstBase, srcBase: srcBase, width: width, height: height, blockX: blockX, blockY: blockY, shiftX: shiftX, shiftY: shiftY, fractX: fractX, fractY: fractY, bw: bw, bh: bh, roundOffset: roundOffset)
    }
}

// MARK: - Multi-Resolution MC (Layer0/Layer1/Layer2 Shared)
// MV is stored in quarter-pixel precision of Layer2.
// mvShift specifies the downscale amount for each layer.
// layer2: mvShift=0, lumaBlockSize=32, chromaBlockSize=16 (used as is)
// layer1: mvShift=1, lumaBlockSize=16, chromaBlockSize=8  (divided by 2)
// layer0: mvShift=2, lumaBlockSize=8,  chromaBlockSize=4  (divided by 4)

@inline(__always)
func scaledMV(_ mv: MotionVector, rightShift: Int) -> MotionVector {
    if mv.isIntra { return mv }
    if rightShift == 0 { return mv }
    // (v + half) >> s rounds correctly for both signs combined with arithmetic right shift floor behavior.

    let half = 1 << (rightShift - 1)
    return MotionVector(
        dx: Int16((Int(mv.dx) + half) >> rightShift),
        dy: Int16((Int(mv.dy) + half) >> rightShift)
    )
}

fileprivate struct SendableMutableInt16Ptr: @unchecked Sendable {
    let ptr: UnsafeMutablePointer<Int16>
}
fileprivate struct SendableInt16Ptr: @unchecked Sendable {
    let ptr: UnsafePointer<Int16>
}

@inline(__always)
func applyScaledMotionCompensationLuma(plane: inout [Int16], prevPlane: [Int16], mvs: MotionVectors, skipMap: [BlockMode]? = nil, width: Int, height: Int, lumaBlockSize: Int, mvShift: Int, roundOffset: Int) async {
    let colCount = (width + lumaBlockSize - 1) / lumaBlockSize
    let rowCount = (height + lumaBlockSize - 1) / lumaBlockSize
    let maxMvIndex = mvs.count - 1
    
    withUnsafePointers(mut: &plane, prevPlane, mvs.dx, mvs.dy) { dstBase, prevBase, dxBase, dyBase in
        if let sMap = skipMap {
            let skipCount = sMap.count
            for row in 0..<rowCount {
                for col in 0..<colCount {
                    let mvIndex = min(row * colCount + col, maxMvIndex)
                    let isSkip = mvIndex < skipCount && sMap[mvIndex] != .inter
                    if isSkip {
                        let blockX = col * lumaBlockSize
                        let blockY = row * lumaBlockSize
                        let bw = min(lumaBlockSize, width - blockX)
                        let bh = min(lumaBlockSize, height - blockY)
                        switch bw {
                        case 32:
                            for y in 0..<bh {
                                let dstPtr = UnsafeMutableRawPointer(dstBase.advanced(by: (blockY + y) * width + blockX))
                                let srcPtr = UnsafeRawPointer(prevBase.advanced(by: (blockY + y) * width + blockX))
                                dstPtr.storeBytes(of: srcPtr.loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
                                dstPtr.advanced(by: 32).storeBytes(of: srcPtr.advanced(by: 32).loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
                            }
                        case 16:
                            for y in 0..<bh {
                                let dstPtr = UnsafeMutableRawPointer(dstBase.advanced(by: (blockY + y) * width + blockX))
                                let srcPtr = UnsafeRawPointer(prevBase.advanced(by: (blockY + y) * width + blockX))
                                dstPtr.storeBytes(of: srcPtr.loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
                            }
                        case 8:
                            for y in 0..<bh {
                                let dstPtr = UnsafeMutableRawPointer(dstBase.advanced(by: (blockY + y) * width + blockX))
                                let srcPtr = UnsafeRawPointer(prevBase.advanced(by: (blockY + y) * width + blockX))
                                dstPtr.storeBytes(of: srcPtr.loadUnaligned(as: SIMD8<Int16>.self), as: SIMD8<Int16>.self)
                            }
                        default:
                            if 0 < bw && 0 < bh {
                                for y in 0..<bh {
                                    let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
                                    let srcPtr = prevBase.advanced(by: (blockY + y) * width + blockX)
                                    dstPtr.update(from: srcPtr, count: bw)
                                }
                            }
                        }
                    } else {
                        let smv = scaledMV(MotionVector(dx: dxBase[mvIndex], dy: dyBase[mvIndex]), rightShift: mvShift)
                        addMCBlockLuma32(dstBase: dstBase, srcBase: prevBase, width: width, height: height, blockX: col * lumaBlockSize, blockY: row * lumaBlockSize, mv: smv, roundOffset: roundOffset, blockSize: lumaBlockSize)
                    }
                }
            }
        } else {
            for row in 0..<rowCount {
                for col in 0..<colCount {
                    let mvIndex = min(row * colCount + col, maxMvIndex)
                    let smv = scaledMV(MotionVector(dx: dxBase[mvIndex], dy: dyBase[mvIndex]), rightShift: mvShift)
                    addMCBlockLuma32(dstBase: dstBase, srcBase: prevBase, width: width, height: height, blockX: col * lumaBlockSize, blockY: row * lumaBlockSize, mv: smv, roundOffset: roundOffset, blockSize: lumaBlockSize)
                }
            }
        }
    }
}

@inline(__always)
func applyScaledMotionCompensationChroma(plane: inout [Int16], prevPlane: [Int16], mvs: MotionVectors, skipMap: [BlockMode]? = nil, width: Int, height: Int, chromaBlockSize: Int, mvShift: Int, roundOffset: Int) async {
    let colCount = (width + chromaBlockSize - 1) / chromaBlockSize
    let rowCount = (height + chromaBlockSize - 1) / chromaBlockSize
    let maxMvIndex = mvs.count - 1
    
    withUnsafePointers(mut: &plane, prevPlane, mvs.dx, mvs.dy) { dstBase, prevBase, dxBase, dyBase in
        if let sMap = skipMap {
            let skipCount = sMap.count
            for row in 0..<rowCount {
                for col in 0..<colCount {
                    let mvIndex = min(row * colCount + col, maxMvIndex)
                    let isSkip = mvIndex < skipCount && sMap[mvIndex] != .inter
                    if isSkip {
                        let blockX = col * chromaBlockSize
                        let blockY = row * chromaBlockSize
                        let bw = min(chromaBlockSize, width - blockX)
                        let bh = min(chromaBlockSize, height - blockY)
                        switch bw {
                        case 32:
                            for y in 0..<bh {
                                let dstPtr = UnsafeMutableRawPointer(dstBase.advanced(by: (blockY + y) * width + blockX))
                                let srcPtr = UnsafeRawPointer(prevBase.advanced(by: (blockY + y) * width + blockX))
                                dstPtr.storeBytes(of: srcPtr.loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
                                dstPtr.advanced(by: 32).storeBytes(of: srcPtr.advanced(by: 32).loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
                            }
                        case 16:
                            for y in 0..<bh {
                                let dstPtr = UnsafeMutableRawPointer(dstBase.advanced(by: (blockY + y) * width + blockX))
                                let srcPtr = UnsafeRawPointer(prevBase.advanced(by: (blockY + y) * width + blockX))
                                dstPtr.storeBytes(of: srcPtr.loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
                            }
                        case 8:
                            for y in 0..<bh {
                                let dstPtr = UnsafeMutableRawPointer(dstBase.advanced(by: (blockY + y) * width + blockX))
                                let srcPtr = UnsafeRawPointer(prevBase.advanced(by: (blockY + y) * width + blockX))
                                dstPtr.storeBytes(of: srcPtr.loadUnaligned(as: SIMD8<Int16>.self), as: SIMD8<Int16>.self)
                            }
                        default:
                            if 0 < bw && 0 < bh {
                                for y in 0..<bh {
                                    let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
                                    let srcPtr = prevBase.advanced(by: (blockY + y) * width + blockX)
                                    dstPtr.update(from: srcPtr, count: bw)
                                }
                            }
                        }
                    } else {
                        let smv = scaledMV(MotionVector(dx: dxBase[mvIndex], dy: dyBase[mvIndex]), rightShift: mvShift)
                        addMCBlockChroma16(dstBase: dstBase, srcBase: prevBase, width: width, height: height, blockX: col * chromaBlockSize, blockY: row * chromaBlockSize, mv: smv, roundOffset: roundOffset, blockSize: chromaBlockSize)
                    }
                }
            }
        } else {
            for row in 0..<rowCount {
                for col in 0..<colCount {
                    let mvIndex = min(row * colCount + col, maxMvIndex)
                    let smv = scaledMV(MotionVector(dx: dxBase[mvIndex], dy: dyBase[mvIndex]), rightShift: mvShift)
                    addMCBlockChroma16(dstBase: dstBase, srcBase: prevBase, width: width, height: height, blockX: col * chromaBlockSize, blockY: row * chromaBlockSize, mv: smv, roundOffset: roundOffset, blockSize: chromaBlockSize)
                }
            }
        }
    }
}

@inline(__always)
func subtractScaledMotionCompensationLuma(plane: inout [Int16], prevPlane: [Int16], mvs: MotionVectors, skipMap: [BlockMode]?, width: Int, height: Int, lumaBlockSize: Int, mvShift: Int, roundOffset: Int) {
    let colCount = (width + lumaBlockSize - 1) / lumaBlockSize
    let rowCount = (height + lumaBlockSize - 1) / lumaBlockSize
    withUnsafePointers(prevPlane, mut: &plane, mvs.dx, mvs.dy) { prevBase, dstBase, dxBase, dyBase in
        let maxMvIndex = mvs.count - 1
        for row in 0..<rowCount {
            for col in 0..<colCount {
                let mvIndex = min(row * colCount + col, maxMvIndex)
                let isSkip = skipMap != nil && skipMap!.count > mvIndex && skipMap![mvIndex] != .inter
                if isSkip {
                    for y in 0..<lumaBlockSize {
                        if height <= row * lumaBlockSize + y { break }
                        let dstPtr = dstBase.advanced(by: (row * lumaBlockSize + y) * width + col * lumaBlockSize)
                        let limit = min(lumaBlockSize, width - col * lumaBlockSize)
                        if 0 < limit {
                            dstPtr.initialize(repeating: 0, count: limit)
                        }
                    }
                } else {
                    let smv = scaledMV(MotionVector(dx: dxBase[mvIndex], dy: dyBase[mvIndex]), rightShift: mvShift)
                    subMCBlockLuma32(dstBase: dstBase, srcBase: prevBase, width: width, height: height, blockX: col * lumaBlockSize, blockY: row * lumaBlockSize, mv: smv, roundOffset: roundOffset, blockSize: lumaBlockSize)
                }
            }
        }
    }
}

@inline(__always)
func subtractScaledMotionCompensationChroma(plane: inout [Int16], prevPlane: [Int16], mvs: MotionVectors, skipMap: [BlockMode]?, width: Int, height: Int, chromaBlockSize: Int, mvShift: Int, roundOffset: Int) {
    let colCount = (width + chromaBlockSize - 1) / chromaBlockSize
    let rowCount = (height + chromaBlockSize - 1) / chromaBlockSize
    withUnsafePointers(prevPlane, mut: &plane, mvs.dx, mvs.dy) { prevBase, dstBase, dxBase, dyBase in
        let maxMvIndex = mvs.count - 1
        for row in 0..<rowCount {
            for col in 0..<colCount {
                let mvIndex = min(row * colCount + col, maxMvIndex)
                let isSkip = skipMap != nil && skipMap!.count > mvIndex && skipMap![mvIndex] != .inter
                if isSkip {
                    for y in 0..<chromaBlockSize {
                        if height <= row * chromaBlockSize + y { break }
                        let dstPtr = dstBase.advanced(by: (row * chromaBlockSize + y) * width + col * chromaBlockSize)
                        let limit = min(chromaBlockSize, width - col * chromaBlockSize)
                        if 0 < limit {
                            dstPtr.initialize(repeating: 0, count: limit)
                        }
                    }
                } else {
                    let smv = scaledMV(MotionVector(dx: dxBase[mvIndex], dy: dyBase[mvIndex]), rightShift: mvShift)
                    subMCBlockChroma16(dstBase: dstBase, srcBase: prevBase, width: width, height: height, blockX: col * chromaBlockSize, blockY: row * chromaBlockSize, mv: smv, roundOffset: roundOffset, blockSize: chromaBlockSize)
                }
            }
        }
    }
}

fileprivate struct SendableBoolPtr: @unchecked Sendable {
    let ptr: UnsafePointer<Bool>
}

// Bidirectional version
@inline(__always)
func applyScaledBidirectionalMotionCompensationLuma(plane: inout [Int16], prevPlane: [Int16], nextPlane: [Int16], mvs: MotionVectors, refDirs: [Bool], skipMap: [BlockMode]? = nil, width: Int, height: Int, lumaBlockSize: Int, mvShift: Int, roundOffset: Int) async {
    let colCount = (width + lumaBlockSize - 1) / lumaBlockSize
    let rowCount = (height + lumaBlockSize - 1) / lumaBlockSize
    let maxMvIndex = mvs.count - 1
    let dirCount = refDirs.count
    
    withUnsafePointers(mut: &plane, prevPlane, nextPlane, mvs.dx, mvs.dy, refDirs) { dstBase, prevBase, nextBase, dxBase, dyBase, dirBase in
        if let sMap = skipMap {
            let skipCount = sMap.count
            for row in 0..<rowCount {
                for col in 0..<colCount {
                    let mvIndex = min(row * colCount + col, maxMvIndex)
                    let isBackward = if mvIndex < dirCount { dirBase[mvIndex] } else { false }
                    let srcBase = isBackward ? nextBase : prevBase
                    
                    let isSkip = mvIndex < skipCount && sMap[mvIndex] != .inter
                    if isSkip {
                        let blockX = col * lumaBlockSize
                        let blockY = row * lumaBlockSize
                        let bw = min(lumaBlockSize, width - blockX)
                        let bh = min(lumaBlockSize, height - blockY)
                        switch bw {
                        case 32:
                            for y in 0..<bh {
                                let dstPtr = UnsafeMutableRawPointer(dstBase.advanced(by: (blockY + y) * width + blockX))
                                let srcPtr = UnsafeRawPointer(srcBase.advanced(by: (blockY + y) * width + blockX))
                                dstPtr.storeBytes(of: srcPtr.loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
                                dstPtr.advanced(by: 32).storeBytes(of: srcPtr.advanced(by: 32).loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
                            }
                        case 16:
                            for y in 0..<bh {
                                let dstPtr = UnsafeMutableRawPointer(dstBase.advanced(by: (blockY + y) * width + blockX))
                                let srcPtr = UnsafeRawPointer(srcBase.advanced(by: (blockY + y) * width + blockX))
                                dstPtr.storeBytes(of: srcPtr.loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
                            }
                        case 8:
                            for y in 0..<bh {
                                let dstPtr = UnsafeMutableRawPointer(dstBase.advanced(by: (blockY + y) * width + blockX))
                                let srcPtr = UnsafeRawPointer(srcBase.advanced(by: (blockY + y) * width + blockX))
                                dstPtr.storeBytes(of: srcPtr.loadUnaligned(as: SIMD8<Int16>.self), as: SIMD8<Int16>.self)
                            }
                        default:
                            if 0 < bw && 0 < bh {
                                for y in 0..<bh {
                                    let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
                                    let srcPtr = srcBase.advanced(by: (blockY + y) * width + blockX)
                                    dstPtr.update(from: srcPtr, count: bw)
                                }
                            }
                        }
                    } else {
                        let smv = scaledMV(MotionVector(dx: dxBase[mvIndex], dy: dyBase[mvIndex]), rightShift: mvShift)
                        addMCBlockLuma32(dstBase: dstBase, srcBase: srcBase, width: width, height: height, blockX: col * lumaBlockSize, blockY: row * lumaBlockSize, mv: smv, roundOffset: roundOffset, blockSize: lumaBlockSize)
                    }
                }
            }
        } else {
            for row in 0..<rowCount {
                for col in 0..<colCount {
                    let mvIndex = min(row * colCount + col, maxMvIndex)
                    let isBackward = if mvIndex < dirCount { dirBase[mvIndex] } else { false }
                    let srcBase = isBackward ? nextBase : prevBase
                    let smv = scaledMV(MotionVector(dx: dxBase[mvIndex], dy: dyBase[mvIndex]), rightShift: mvShift)
                    addMCBlockLuma32(dstBase: dstBase, srcBase: srcBase, width: width, height: height, blockX: col * lumaBlockSize, blockY: row * lumaBlockSize, mv: smv, roundOffset: roundOffset, blockSize: lumaBlockSize)
                }
            }
        }
    }
}

@inline(__always)
func applyScaledBidirectionalMotionCompensationChroma(plane: inout [Int16], prevPlane: [Int16], nextPlane: [Int16], mvs: MotionVectors, refDirs: [Bool], skipMap: [BlockMode]? = nil, width: Int, height: Int, chromaBlockSize: Int, mvShift: Int, roundOffset: Int) async {
    let colCount = (width + chromaBlockSize - 1) / chromaBlockSize
    let rowCount = (height + chromaBlockSize - 1) / chromaBlockSize
    let maxMvIndex = mvs.count - 1
    let dirCount = refDirs.count
    
    withUnsafePointers(mut: &plane, prevPlane, nextPlane, mvs.dx, mvs.dy, refDirs) { dstBase, prevBase, nextBase, dxBase, dyBase, dirBase in
        if let sMap = skipMap {
            let skipCount = sMap.count
            for row in 0..<rowCount {
                for col in 0..<colCount {
                    let mvIndex = min(row * colCount + col, maxMvIndex)
                    let isBackward = if mvIndex < dirCount { dirBase[mvIndex] } else { false }
                    let srcBase = isBackward ? nextBase : prevBase
                    
                    let isSkip = mvIndex < skipCount && sMap[mvIndex] != .inter
                    if isSkip {
                        let blockX = col * chromaBlockSize
                        let blockY = row * chromaBlockSize
                        let bw = min(chromaBlockSize, width - blockX)
                        let bh = min(chromaBlockSize, height - blockY)
                        switch bw {
                        case 32:
                            for y in 0..<bh {
                                let dstPtr = UnsafeMutableRawPointer(dstBase.advanced(by: (blockY + y) * width + blockX))
                                let srcPtr = UnsafeRawPointer(srcBase.advanced(by: (blockY + y) * width + blockX))
                                dstPtr.storeBytes(of: srcPtr.loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
                                dstPtr.advanced(by: 32).storeBytes(of: srcPtr.advanced(by: 32).loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
                            }
                        case 16:
                            for y in 0..<bh {
                                let dstPtr = UnsafeMutableRawPointer(dstBase.advanced(by: (blockY + y) * width + blockX))
                                let srcPtr = UnsafeRawPointer(srcBase.advanced(by: (blockY + y) * width + blockX))
                                dstPtr.storeBytes(of: srcPtr.loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
                            }
                        case 8:
                            for y in 0..<bh {
                                let dstPtr = UnsafeMutableRawPointer(dstBase.advanced(by: (blockY + y) * width + blockX))
                                let srcPtr = UnsafeRawPointer(srcBase.advanced(by: (blockY + y) * width + blockX))
                                dstPtr.storeBytes(of: srcPtr.loadUnaligned(as: SIMD8<Int16>.self), as: SIMD8<Int16>.self)
                            }
                        default:
                            if 0 < bw && 0 < bh {
                                for y in 0..<bh {
                                    let dstPtr = dstBase.advanced(by: (blockY + y) * width + blockX)
                                    let srcPtr = srcBase.advanced(by: (blockY + y) * width + blockX)
                                    dstPtr.update(from: srcPtr, count: bw)
                                }
                            }
                        }
                    } else {
                        let smv = scaledMV(MotionVector(dx: dxBase[mvIndex], dy: dyBase[mvIndex]), rightShift: mvShift)
                        addMCBlockChroma16(dstBase: dstBase, srcBase: srcBase, width: width, height: height, blockX: col * chromaBlockSize, blockY: row * chromaBlockSize, mv: smv, roundOffset: roundOffset, blockSize: chromaBlockSize)
                    }
                }
            }
        } else {
            for row in 0..<rowCount {
                for col in 0..<colCount {
                    let mvIndex = min(row * colCount + col, maxMvIndex)
                    let isBackward = if mvIndex < dirCount { dirBase[mvIndex] } else { false }
                    let srcBase = isBackward ? nextBase : prevBase
                    let smv = scaledMV(MotionVector(dx: dxBase[mvIndex], dy: dyBase[mvIndex]), rightShift: mvShift)
                    addMCBlockChroma16(dstBase: dstBase, srcBase: srcBase, width: width, height: height, blockX: col * chromaBlockSize, blockY: row * chromaBlockSize, mv: smv, roundOffset: roundOffset, blockSize: chromaBlockSize)
                }
            }
        }
    }
}
@inline(__always)
func subtractScaledBidirectionalMotionCompensationLuma(plane: inout [Int16], prevPlane: [Int16], nextPlane: [Int16], mvs: MotionVectors, refDirs: [Bool], skipMap: [BlockMode]?, width: Int, height: Int, lumaBlockSize: Int, mvShift: Int, roundOffset: Int) {
    let colCount = (width + lumaBlockSize - 1) / lumaBlockSize
    let rowCount = (height + lumaBlockSize - 1) / lumaBlockSize
    withUnsafePointers(prevPlane, nextPlane, mut: &plane, mvs.dx, mvs.dy, refDirs) { prevBase, nextBase, dstBase, dxBase, dyBase, refDirsBase in
        let maxMvIndex = mvs.count - 1
        for row in 0..<rowCount {
            for col in 0..<colCount {
                let mvIndex = min(row * colCount + col, maxMvIndex)
                let isSkip = skipMap != nil && skipMap!.count > mvIndex && skipMap![mvIndex] != .inter
                if isSkip {
                    for y in 0..<lumaBlockSize {
                        if height <= row * lumaBlockSize + y { break }
                        let dstPtr = dstBase.advanced(by: (row * lumaBlockSize + y) * width + col * lumaBlockSize)
                        let limit = min(lumaBlockSize, width - col * lumaBlockSize)
                        if 0 < limit {
                            dstPtr.initialize(repeating: 0, count: limit)
                        }
                    }
                } else {
                    let isBackward = refDirsBase[mvIndex]
                    let srcBase = isBackward ? nextBase : prevBase
                    let smv = scaledMV(MotionVector(dx: dxBase[mvIndex], dy: dyBase[mvIndex]), rightShift: mvShift)
                    subMCBlockLuma32(dstBase: dstBase, srcBase: srcBase, width: width, height: height, blockX: col * lumaBlockSize, blockY: row * lumaBlockSize, mv: smv, roundOffset: roundOffset, blockSize: lumaBlockSize)
                }
            }
        }
    }
}

@inline(__always)
func subtractScaledBidirectionalMotionCompensationChroma(plane: inout [Int16], prevPlane: [Int16], nextPlane: [Int16], mvs: MotionVectors, refDirs: [Bool], skipMap: [BlockMode]?, width: Int, height: Int, chromaBlockSize: Int, mvShift: Int, roundOffset: Int) {
    let colCount = (width + chromaBlockSize - 1) / chromaBlockSize
    let rowCount = (height + chromaBlockSize - 1) / chromaBlockSize
    withUnsafePointers(prevPlane, nextPlane, mut: &plane, mvs.dx, mvs.dy, refDirs) { prevBase, nextBase, dstBase, dxBase, dyBase, refDirsBase in
        let maxMvIndex = mvs.count - 1
        for row in 0..<rowCount {
            for col in 0..<colCount {
                let mvIndex = min(row * colCount + col, maxMvIndex)
                let isSkip = skipMap != nil && skipMap!.count > mvIndex && skipMap![mvIndex] != .inter
                if isSkip {
                    for y in 0..<chromaBlockSize {
                        if height <= row * chromaBlockSize + y { break }
                        let dstPtr = dstBase.advanced(by: (row * chromaBlockSize + y) * width + col * chromaBlockSize)
                        let limit = min(chromaBlockSize, width - col * chromaBlockSize)
                        if 0 < limit {
                            dstPtr.initialize(repeating: 0, count: limit)
                        }
                    }
                } else {
                    let isBackward = refDirsBase[mvIndex]
                    let srcBase = isBackward ? nextBase : prevBase
                    let smv = scaledMV(MotionVector(dx: dxBase[mvIndex], dy: dyBase[mvIndex]), rightShift: mvShift)
                    subMCBlockChroma16(dstBase: dstBase, srcBase: srcBase, width: width, height: height, blockX: col * chromaBlockSize, blockY: row * chromaBlockSize, mv: smv, roundOffset: roundOffset, blockSize: chromaBlockSize)
                }
            }
        }
    }
}
