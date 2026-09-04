// MARK: - MotionVector

struct MotionVector: Sendable {
    let dx: Int16
    let dy: Int16
    
    init(dx: Int16, dy: Int16) {
        self.dx = dx
        self.dy = dy
    }
    
    @inline(__always)
    var isIntra: Bool {
        return dx == 32767 && dy == 32767
    }
}

/// SoA (Structure of Arrays) layout for motion vectors.
/// dx and dy are stored in separate contiguous arrays for cache-friendly access.
struct MotionVectors: Sendable {
    var dx: [Int16]
    var dy: [Int16]
    
    var count: Int { dx.count }
    var isEmpty: Bool { dx.isEmpty }
    
    init(dx: [Int16], dy: [Int16]) {
        self.dx = dx
        self.dy = dy
    }
    
    init(count: Int) {
        self.dx = [Int16](repeating: 0, count: count)
        self.dy = [Int16](repeating: 0, count: count)
    }
    
    static let empty = MotionVectors(dx: [], dy: [])
}

/// Per-block motion search results of *both* reference directions. The coding
/// path keeps a single vector per block — the direction the SAD comparison
/// picked — so a consumer that has to weigh "copy from prev" against "copy from
/// LTR" cannot reuse it: reading the chosen vector for both makes the two
/// candidates the same block. `computeBidirectionalMotionVectors` fills this
/// only when asked, and filling it does not change the vectors it returns.
final class DualMVSink: @unchecked Sendable {
    var prevMVs: MotionVectors = .empty
    var ltrMVs: MotionVectors = .empty
    /// The per-direction SADs the search already produced. `prev` includes the
    /// chroma term the coding path adds; `ltr` is the luma search SAD plus the
    /// same chroma term.
    var prevSADs: [Int] = []
    var ltrSADs: [Int] = []
}

/// Temporal SNN membrane potentials tracking motion search direction affinity.
/// Negative: Prev affinity (active motion)
/// Positive: LTR affinity (static / background persistence)
final class MEMembraneState: @unchecked Sendable {
    var potentials: [Int8]

    init(count: Int = 0) {
        if 0 < count {
            self.potentials = [Int8](repeating: 0, count: count)
        } else {
            self.potentials = []
        }
    }

    func ensure(count: Int) {
        if potentials.count == count {
            return
        }
        potentials = [Int8](repeating: 0, count: count)
    }

    func reset() {
        for i in 0..<potentials.count {
            potentials[i] = 0
        }
    }

    func update(mvs: MotionVectors, refDirs: [Bool], skipMap: [BlockMode]) {
        let n = potentials.count
        let count = min(n, min(mvs.dx.count, refDirs.count))
        let hasSkipMap = 0 < skipMap.count
        for i in 0..<count {
            let isSkipLtr: Bool
            let isSkipPrev: Bool
            if hasSkipMap && i < skipMap.count {
                switch skipMap[i] {
                case .skip_ltr:
                    isSkipLtr = true
                    isSkipPrev = false
                case .skip_prev:
                    isSkipLtr = false
                    isSkipPrev = true
                default:
                    isSkipLtr = false
                    isSkipPrev = false
                }
            } else {
                isSkipLtr = false
                isSkipPrev = false
            }

            let isLtr = refDirs[i] || isSkipLtr
            let isStatic = isSkipPrev || (mvs.dx[i] == 0 && mvs.dy[i] == 0)

            var v = Int(potentials[i])
            switch true {
            case isLtr:
                // LTR reference chosen: increase LTR affinity strongly
                v = min(8, v + 3)
            case isStatic:
                // Static persistence: steady increase in LTR affinity
                v = min(8, v + 1)
            default:
                // Active motion with Prev reference: decrease LTR affinity
                v = max(-8, v - 2)
            }
            potentials[i] = Int8(v)
        }
    }
}

private let meFineOffsets: [(Int, Int)] = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1)
]

private let meSearchOffsetX: [Int] = [-1, 0, 1, -1, 1, -1, 0, 1]
private let meSearchOffsetY: [Int] = [-1, -1, -1, 0, 0, 1, 1, 1]

let meRefine2OffsetsX: [Int] = [
    // r=1 (8 points)
    -1,  0,  1, -1,  1, -1,  0,  1,
    // r=2 (16 points: y=-2 (5), y=-1 (2), y=0 (2), y=1 (2), y=2 (5))
    -2, -1,  0,  1,  2, -2,  2, -2,  2, -2,  2, -2, -1,  0,  1,  2
]
let meRefine2OffsetsY: [Int] = [
    // r=1 (8 points)
    -1, -1, -1,  0,  0,  1,  1,  1,
    // r=2 (16 points)
    -2, -2, -2, -2, -2, -1, -1,  0,  0,  1,  1,  2,  2,  2,  2,  2
]

// MARK: - MotionEstimation

struct MotionEstimation {

    @inline(__always)
    fileprivate static func median(_ a: Int, _ b: Int, _ c: Int) -> Int {
        return max(min(a, b), min(max(a, b), c))
    }

    // Lagrange multiplier for rate-distortion optimization:
    // penalizes Motion Vector Difference (MVD) to favor pmv
    @inline(__always)
    static func getMVDPenalty(dx: Int, dy: Int, pmvDx: Int, pmvDy: Int) -> Int {
        return Int((dx - pmvDx).magnitude) + Int((dy - pmvDy).magnitude)
    }

    @inline(__always)
    static func fetchPixelsBlock8(plane: UnsafePointer<Int16>, width: Int, height: Int, x: Int, y: Int, dest: UnsafeMutablePointer<Int16>) {
        if 0 <= x && 0 <= y && x + 8 <= width && y + 8 <= height {
            for ry in 0..<8 {
                let offset = (y + ry) * width + x
                let sPtr = UnsafeRawPointer(plane.advanced(by: offset))
                let dPtr = UnsafeMutableRawPointer(dest.advanced(by: ry * 8))
                dPtr.storeBytes(of: sPtr.loadUnaligned(as: SIMD8<Int16>.self), as: SIMD8<Int16>.self)
            }
        } else {
            for i in 0..<64 {
                let ry = i >> 3
                let rx = i & 7
                let srcY = max(0, min(y + ry, height - 1))
                let srcX = max(0, min(x + rx, width - 1))
                dest[i] = plane[srcY * width + srcX]
            }
        }
    }

    @inline(__always)
    static func fetchHalfPixelBlock8(plane: UnsafePointer<Int16>, width: Int, height: Int, intX: Int, intY: Int, fractX: Int, fractY: Int, dest: UnsafeMutablePointer<Int16>, roundOffset: Int) {
        if fractX == 0 && fractY == 0 {
            fetchPixelsBlock8(plane: plane, width: width, height: height, x: intX, y: intY, dest: dest)
            return
        }
        
        if 0 <= intX && 0 <= intY && ((intX + 8) + fractX) <= width && ((intY + 8) + fractY) <= height {
            if fractY == 0 {
                let vRound = SIMD8<Int16>(repeating: Int16(roundOffset))
                for ry in 0..<8 {
                    let row = plane.advanced(by: (intY + ry) * width + intX)
                    let dst = dest.advanced(by: ry * 8)
                    let r0 = UnsafeRawPointer(row).loadUnaligned(as: SIMD8<Int16>.self)
                    let r1 = UnsafeRawPointer(row.advanced(by: 1)).loadUnaligned(as: SIMD8<Int16>.self)
                    let res = (r0 &+ r1 &+ vRound) &>> 1
                    UnsafeMutableRawPointer(dst).storeBytes(of: res, as: SIMD8<Int16>.self)
                }
                return
            }
            if fractX == 0 {
                let vRound = SIMD8<Int16>(repeating: Int16(roundOffset))
                for ry in 0..<8 {
                    let row0 = plane.advanced(by: (intY + ry) * width + intX)
                    let row1 = plane.advanced(by: (intY + ry + 1) * width + intX)
                    let dst = dest.advanced(by: ry * 8)
                    let r0 = UnsafeRawPointer(row0).loadUnaligned(as: SIMD8<Int16>.self)
                    let r1 = UnsafeRawPointer(row1).loadUnaligned(as: SIMD8<Int16>.self)
                    let res = (r0 &+ r1 &+ vRound) &>> 1
                    UnsafeMutableRawPointer(dst).storeBytes(of: res, as: SIMD8<Int16>.self)
                }
                return
            }
            
            let vRound = SIMD8<Int16>(repeating: Int16(1 + roundOffset))
            for ry in 0..<8 {
                let row0 = plane.advanced(by: (intY + ry) * width + intX)
                let row1 = plane.advanced(by: (intY + ry + 1) * width + intX)
                let dst = dest.advanced(by: ry * 8)
                let r00 = UnsafeRawPointer(row0).loadUnaligned(as: SIMD8<Int16>.self)
                let r01 = UnsafeRawPointer(row0.advanced(by: 1)).loadUnaligned(as: SIMD8<Int16>.self)
                let r10 = UnsafeRawPointer(row1).loadUnaligned(as: SIMD8<Int16>.self)
                let r11 = UnsafeRawPointer(row1.advanced(by: 1)).loadUnaligned(as: SIMD8<Int16>.self)
                let res = (r00 &+ r01 &+ r10 &+ r11 &+ vRound) &>> 2
                UnsafeMutableRawPointer(dst).storeBytes(of: res, as: SIMD8<Int16>.self)
            }
            return
        }
        
        for ry in 0..<8 {
            let sy0 = max(0, min(intY + ry, height - 1))
            let sy1 = max(0, min(intY + ry + fractY, height - 1))
            let row0 = plane.advanced(by: sy0 * width)
            let row1 = plane.advanced(by: sy1 * width)
            let dstPtr = dest.advanced(by: ry * 8)
            for rx in 0..<8 {
                let sx0 = max(0, min(intX + rx, width - 1))
                let sx1 = max(0, min(intX + rx + fractX, width - 1))
                switch true {
                case fractY == 0:
                    dstPtr[rx] = Int16((Int(row0[sx0]) + Int(row0[sx1]) + roundOffset) >> 1)
                case fractX == 0:
                    dstPtr[rx] = Int16((Int(row0[sx0]) + Int(row1[sx0]) + roundOffset) >> 1)
                default:
                    dstPtr[rx] = Int16((Int(row0[sx0]) + Int(row0[sx1]) + Int(row1[sx0]) + Int(row1[sx1]) + 1 + roundOffset) >> 2)
                }
            }
        }
    }

    @inline(__always)
    static func fetchQuarterPixelBlock8(plane: UnsafePointer<Int16>, width: Int, height: Int, intX: Int, intY: Int, remX: Int, remY: Int, dest: UnsafeMutablePointer<Int16>, roundOffset: Int) {
        if remX == 0 && remY == 0 {
            fetchPixelsBlock8(plane: plane, width: width, height: height, x: intX, y: intY, dest: dest)
            return
        }
        let nextX = if remX == 0 { 0 } else { 1 }
        let nextY = if remY == 0 { 0 } else { 1 }
        let wA = 4 - remX
        let wB = remX
        let wC = 4 - remY
        let wD = remY
        
        if 0 <= intX && 0 <= intY && ((intX + 8) + nextX) <= width && ((intY + 8) + nextY) <= height {
            for ry in 0..<8 {
                let row0 = plane.advanced(by: (intY + ry) * width + intX)
                let row1 = plane.advanced(by: (intY + ry + nextY) * width + intX)
                let dst = dest.advanced(by: ry * 8)
                for rx in 0..<8 {
                    let v0 = (wA * wC) * Int(row0[rx])
                    let v1 = (wB * wC) * Int(row0[rx + nextX])
                    let v2 = (wA * wD) * Int(row1[rx])
                    let v3 = (wB * wD) * Int(row1[rx + nextX])
                    let v = (v0 + v1) + (v2 + v3)
                    dst[rx] = Int16((v + 7 + roundOffset) >> 4)
                }
            }
            return
        }
        for ry in 0..<8 {
            let sy0 = max(0, min(intY + ry, height - 1))
            let sy1 = max(0, min(intY + ry + nextY, height - 1))
            let row0 = plane.advanced(by: sy0 * width)
            let row1 = plane.advanced(by: sy1 * width)
            let dst = dest.advanced(by: ry * 8)
            for rx in 0..<8 {
                let sx0 = max(0, min(intX + rx, width - 1))
                let sx1 = max(0, min(intX + rx + nextX, width - 1))
                let v0 = (wA * wC) * Int(row0[sx0])
                let v1 = (wB * wC) * Int(row0[sx1])
                let v2 = (wA * wD) * Int(row1[sx0])
                let v3 = (wB * wD) * Int(row1[sx1])
                let v = (v0 + v1) + (v2 + v3)
                dst[rx] = Int16((v + 7 + roundOffset) >> 4)
            }
        }
    }

    @inline(__always)
    static func compute64PointSADBlocksWithStride(cBase: UnsafePointer<Int16>, pBase: UnsafePointer<Int16>, pStride: Int) -> Int {
        var sad: Int32 = 0
        for ry in 0..<8 {
            let c = UnsafeRawPointer(cBase.advanced(by: ry * 8)).loadUnaligned(as: SIMD8<Int16>.self)
            let p = UnsafeRawPointer(pBase.advanced(by: ry * pStride)).loadUnaligned(as: SIMD8<Int16>.self)
            let d = pointwiseMax(c, p) &- pointwiseMin(c, p)
            sad &+= SIMD8<Int32>(truncatingIfNeeded: d).wrappedSum()
        }
        return Int(sad)
    }

    @inline(__always)
    static func compute64PointSADBlocks(cBase: UnsafePointer<Int16>, pBase: UnsafePointer<Int16>) -> Int {
        let c0 = UnsafeRawPointer(cBase).loadUnaligned(as: SIMD16<Int16>.self)
        let p0 = UnsafeRawPointer(pBase).loadUnaligned(as: SIMD16<Int16>.self)
        let d0 = pointwiseMax(c0, p0) &- pointwiseMin(c0, p0)

        let c1 = UnsafeRawPointer(cBase.advanced(by: 16)).loadUnaligned(as: SIMD16<Int16>.self)
        let p1 = UnsafeRawPointer(pBase.advanced(by: 16)).loadUnaligned(as: SIMD16<Int16>.self)
        let d1 = pointwiseMax(c1, p1) &- pointwiseMin(c1, p1)

        let c2 = UnsafeRawPointer(cBase.advanced(by: 32)).loadUnaligned(as: SIMD16<Int16>.self)
        let p2 = UnsafeRawPointer(pBase.advanced(by: 32)).loadUnaligned(as: SIMD16<Int16>.self)
        let d2 = pointwiseMax(c2, p2) &- pointwiseMin(c2, p2)

        let c3 = UnsafeRawPointer(cBase.advanced(by: 48)).loadUnaligned(as: SIMD16<Int16>.self)
        let p3 = UnsafeRawPointer(pBase.advanced(by: 48)).loadUnaligned(as: SIMD16<Int16>.self)
        let d3 = pointwiseMax(c3, p3) &- pointwiseMin(c3, p3)

        let sum = (SIMD16<Int32>(truncatingIfNeeded: d0) &+ SIMD16<Int32>(truncatingIfNeeded: d1))
            &+ (SIMD16<Int32>(truncatingIfNeeded: d2) &+ SIMD16<Int32>(truncatingIfNeeded: d3))
        return Int(sum.wrappedSum())
    }

    private static let dsLdspX: [Int] = [0, 1, 2, 1, 0, -1, -2, -1]
    private static let dsLdspY: [Int] = [-2, -1, 0, 1, 2, 1, 0, -1]
    private static let dsSdspX: [Int] = [0, 1, 0, -1]
    private static let dsSdspY: [Int] = [-1, 0, 1, 0]

    @inline(__always)
    private static func refineNear2(
        candCenterDx: Int, candCenterDy: Int,
        candCost: Int, candSAD: Int,
        minDx: Int, maxDx: Int, minDy: Int, maxDy: Int,
        pmvBaseX: Int, pmvBaseY: Int,
        pBase: UnsafePointer<Int16>, cPtr: UnsafePointer<Int16>,
        width: Int, bx: Int, by: Int
    ) -> (Int, Int, Int, Int) {
        let candidateBias: Int
        if candSAD < 128 {
            candidateBias = 33
        } else {
            candidateBias = 32
        }
        var bestRefineDx = candCenterDx
        var bestRefineDy = candCenterDy
        var bestRefineCost = candCost - candidateBias
        var bestRefineSAD = candSAD

        let isSafeInner = (minDx <= candCenterDx - 2) && (candCenterDx + 2 <= maxDx) && (minDy <= candCenterDy - 2) && (candCenterDy + 2 <= maxDy)
        if isSafeInner {
            let centerBase = pBase.advanced(by: (by + candCenterDy) * width + (bx + candCenterDx))

            for i in 0..<24 {
                let rx = candCenterDx + meRefine2OffsetsX[i]
                let ry = candCenterDy + meRefine2OffsetsY[i]
                let penalty = getMVDPenalty(dx: rx, dy: ry, pmvDx: pmvBaseX, pmvDy: pmvBaseY) * 4
                let maxAllowed = bestRefineCost - penalty
                if maxAllowed <= 0 { continue }

                let pPtr = centerBase.advanced(by: meRefine2OffsetsY[i] * width + meRefine2OffsetsX[i])
                let sad = compute64PointSADBlocksWithStride(cBase: cPtr, pBase: pPtr, pStride: width)
                let totalCost = sad + penalty
                if totalCost < bestRefineCost {
                    bestRefineCost = totalCost
                    bestRefineSAD = sad
                    bestRefineDx = rx
                    bestRefineDy = ry
                }
            }
        } else {
            for i in 0..<24 {
                let rx = candCenterDx + meRefine2OffsetsX[i]
                let ry = candCenterDy + meRefine2OffsetsY[i]
                if rx < minDx { continue }
                if maxDx < rx { continue }
                if ry < minDy { continue }
                if maxDy < ry { continue }

                let penalty = getMVDPenalty(dx: rx, dy: ry, pmvDx: pmvBaseX, pmvDy: pmvBaseY) * 4
                let maxAllowed = bestRefineCost - penalty
                if maxAllowed <= 0 { continue }

                let pPtr = pBase.advanced(by: (by + ry) * width + (bx + rx))
                let sad = compute64PointSADBlocksWithStride(cBase: cPtr, pBase: pPtr, pStride: width)
                let totalCost = sad + penalty
                if totalCost < bestRefineCost {
                    bestRefineCost = totalCost
                    bestRefineSAD = sad
                    bestRefineDx = rx
                    bestRefineDy = ry
                }
            }
        }

        if bestRefineDx != candCenterDx || bestRefineDy != candCenterDy {
            return (bestRefineDx, bestRefineDy, bestRefineCost, bestRefineSAD)
        }
        return (candCenterDx, candCenterDy, candCost, candSAD)
    }

    @inline(__always)
    private static func searchLdsp(
        centerX: Int, centerY: Int,
        startCost: Int, startSAD: Int,
        minDx: Int, maxDx: Int, minDy: Int, maxDy: Int,
        pmvBaseX: Int, pmvBaseY: Int,
        pBase: UnsafePointer<Int16>, cPtr: UnsafePointer<Int16>,
        width: Int, bx: Int, by: Int
    ) -> (Int, Int, Int, Int) {
        var curCenterX = centerX
        var curCenterDy = centerY
        var bestCost = startCost
        var bestSAD = startSAD
        var ldspIter = 0
        while ldspIter < 2 {
            ldspIter += 1
            let stepThresholdCost = bestCost - 32
            var minCost = stepThresholdCost
            var minSAD = bestSAD
            var minDxPos = curCenterX
            var minDyPos = curCenterDy
            var foundSmaller = false

            let isSafeInner = (minDx <= curCenterX - 2) && (curCenterX + 2 <= maxDx) && (minDy <= curCenterDy - 2) && (curCenterDy + 2 <= maxDy)
            if isSafeInner {
                let centerBase = pBase.advanced(by: (by + curCenterDy) * width + (bx + curCenterX))

                for i in 0..<8 {
                    let dx = curCenterX + dsLdspX[i]
                    let dy = curCenterDy + dsLdspY[i]
                    let penalty = getMVDPenalty(dx: dx, dy: dy, pmvDx: pmvBaseX, pmvDy: pmvBaseY) * 4
                    let maxAllowed = minCost - penalty
                    if maxAllowed <= 0 { continue }

                    let pPtr = centerBase.advanced(by: dsLdspY[i] * width + dsLdspX[i])
                    let sad = compute64PointSADBlocksWithStride(cBase: cPtr, pBase: pPtr, pStride: width)
                    let total = sad + penalty
                    if total < minCost {
                        minCost = total
                        minSAD = sad
                        minDxPos = dx
                        minDyPos = dy
                        foundSmaller = true
                    }
                }
            } else {
                for i in 0..<8 {
                    let dx = curCenterX + dsLdspX[i]
                    let dy = curCenterDy + dsLdspY[i]
                    if dx < minDx { continue }
                    if maxDx < dx { continue }
                    if dy < minDy { continue }
                    if maxDy < dy { continue }

                    let penalty = getMVDPenalty(dx: dx, dy: dy, pmvDx: pmvBaseX, pmvDy: pmvBaseY) * 4
                    let maxAllowed = minCost - penalty
                    if maxAllowed <= 0 { continue }

                    let pPtr = pBase.advanced(by: (by + dy) * width + (bx + dx))
                    let sad = compute64PointSADBlocksWithStride(cBase: cPtr, pBase: pPtr, pStride: width)
                    let total = sad + penalty
                    if total < minCost {
                        minCost = total
                        minSAD = sad
                        minDxPos = dx
                        minDyPos = dy
                        foundSmaller = true
                    }
                }
            }
            if foundSmaller != true {
                break
            }
            bestCost = minCost
            bestSAD = minSAD
            curCenterX = minDxPos
            curCenterDy = minDyPos
            if bestSAD < 128 {
                break
            }
        }
        return (curCenterX, curCenterDy, bestCost, bestSAD)
    }

    @inline(__always)
    private static func evaluateSearch(
        cPtr: UnsafePointer<Int16>, 
        pBase: UnsafePointer<Int16>, 
        oPtr: UnsafeMutablePointer<Int16>,
        tPtr: UnsafeMutablePointer<Int16>,
        width: Int, height: Int, bx: Int, by: Int, range: Int,
        pmv: MotionVector,
        tmv: MotionVector = MotionVector(dx: 0, dy: 0),
        spatialA: MotionVector = MotionVector(dx: 0, dy: 0),
        spatialB: MotionVector = MotionVector(dx: 0, dy: 0),
        spatialC: MotionVector = MotionVector(dx: 0, dy: 0),
        roundOffset: Int,
        contrast: Int = 0,
        membrane: Int = 0,
        gopPosition: Int = 0,
        weights: MEDeciderWeights = MEDeciderWeights.shared
    ) -> (Int, Int, Int) {
        fetchPixelsBlock8(plane: pBase, width: width, height: height, x: bx, y: by, dest: oPtr)
        let zeroSAD: Int = compute64PointSADBlocks(cBase: cPtr, pBase: oPtr)
        
        if zeroSAD < 34 {
            return (0, 0, zeroSAD)
        }
        
        let minDy = max(-1 * range, -1 * by)
        let maxDy = min(range, height - by - 8)
        let minDx = max(-1 * range, -1 * bx)
        let maxDx = min(range, width - bx - 8)
        
        if maxDy < minDy || maxDx < minDx {
            return (0, 0, zeroSAD)
        }
        
        let pmvBaseX = Int(pmv.dx) / 4
        let pmvBaseY = Int(pmv.dy) / 4
        let tmvBaseX = Int(tmv.dx) / 4
        let tmvBaseY = Int(tmv.dy) / 4
        let spABaseX = Int(spatialA.dx) / 4
        let spABaseY = Int(spatialA.dy) / 4
        let spBBaseX = Int(spatialB.dx) / 4
        let spBBaseY = Int(spatialB.dy) / 4
        let spCBaseX = Int(spatialC.dx) / 4
        let spCBaseY = Int(spatialC.dy) / 4

        let pmvClampedX = max(minDx, min(maxDx, pmvBaseX))
        let pmvClampedY = max(minDy, min(maxDy, pmvBaseY))
        
        let zeroPenalty = getMVDPenalty(dx: 0, dy: 0, pmvDx: pmvBaseX, pmvDy: pmvBaseY) * 4
        let zeroCost = zeroSAD + zeroPenalty
        var bestCoarseCost = zeroCost
        var bestCoarseSAD = zeroSAD
        var bestCoarseDx = 0
        var bestCoarseDy = 0
        var pmvSAD = Int.max
        
        // 1. Evaluate Candidate MVs: PMV, TMV, Spatial neighbors (A, B, C) in Base8 integer pixels
        let cands: [(Int, Int)] = [
            (pmvClampedX, pmvClampedY),
            (max(minDx, min(maxDx, tmvBaseX)), max(minDy, min(maxDy, tmvBaseY))),
            (max(minDx, min(maxDx, spABaseX)), max(minDy, min(maxDy, spABaseY))),
            (max(minDx, min(maxDx, spBBaseX)), max(minDy, min(maxDy, spBBaseY))),
            (max(minDx, min(maxDx, spCBaseX)), max(minDy, min(maxDy, spCBaseY)))
        ]
        
        for cand in cands {
            let cx = cand.0
            let cy = cand.1
            if cx == 0 && cy == 0 {
                if cx == pmvClampedX && cy == pmvClampedY {
                    pmvSAD = zeroSAD
                }
                continue
            }
            if cx == bestCoarseDx && cy == bestCoarseDy {
                if cx == pmvClampedX && cy == pmvClampedY {
                    pmvSAD = bestCoarseSAD
                }
                continue
            }
            
            let pPtr = pBase.advanced(by: (by + cy) * width + (bx + cx))
            let sad = compute64PointSADBlocksWithStride(cBase: cPtr, pBase: pPtr, pStride: width)
            if cx == pmvClampedX && cy == pmvClampedY {
                pmvSAD = sad
            }
            let penalty = getMVDPenalty(dx: cx, dy: cy, pmvDx: pmvBaseX, pmvDy: pmvBaseY) * 4
            let totalCost = sad + penalty
            if totalCost < bestCoarseCost {
                bestCoarseCost = totalCost
                bestCoarseSAD = sad
                bestCoarseDx = cx
                bestCoarseDy = cy
            }
        }
        
        // Prioritize candidate (0, 0) or PMV among candidates
        if zeroSAD < 48 || zeroCost <= (bestCoarseCost + 32) {
            bestCoarseDx = 0
            bestCoarseDy = 0
            bestCoarseSAD = zeroSAD
            bestCoarseCost = zeroCost
        } else {
            if (bestCoarseDx != pmvClampedX || bestCoarseDy != pmvClampedY) && (pmvClampedX != 0 || pmvClampedY != 0) {
                let pmvPenalty = getMVDPenalty(dx: pmvClampedX, dy: pmvClampedY, pmvDx: pmvBaseX, pmvDy: pmvBaseY) * 4
                let pmvCost = pmvSAD + pmvPenalty
                if pmvCost <= (bestCoarseCost + 25) {
                    bestCoarseDx = pmvClampedX
                    bestCoarseDy = pmvClampedY
                    bestCoarseSAD = pmvSAD
                    bestCoarseCost = pmvCost
                }
            }
        }
        
        // 2. Classification-guided integer motion search regime
        let candMvMag = Int((Int(bestCoarseDx)).magnitude) + Int((Int(bestCoarseDy)).magnitude)
        let q = SIMD8<Int32>(
            Int32(zeroSAD),
            Int32(bestCoarseSAD),
            Int32(clamping: pmvSAD),
            Int32(zeroCost - bestCoarseCost),
            Int32(candMvMag),
            Int32(contrast),
            Int32(membrane),
            Int32(gopPosition)
        )
        let decision = meDeciderClassify(q: q, weights: weights)

        switch decision.mode {
        case .earlyCandidate:
            break
        case .refineNear:
            let (rx, ry, rcost, rsad) = refineNear2(
                candCenterDx: bestCoarseDx, candCenterDy: bestCoarseDy,
                candCost: bestCoarseCost, candSAD: bestCoarseSAD,
                minDx: minDx, maxDx: maxDx, minDy: minDy, maxDy: maxDy,
                pmvBaseX: pmvBaseX, pmvBaseY: pmvBaseY,
                pBase: pBase, cPtr: cPtr, width: width, bx: bx, by: by
            )
            bestCoarseDx = rx
            bestCoarseDy = ry
            bestCoarseCost = rcost
            bestCoarseSAD = rsad
            if 384 <= rsad {
                let (lx, ly, lcost, lsad) = searchLdsp(
                    centerX: rx, centerY: ry,
                    startCost: rcost, startSAD: rsad,
                    minDx: minDx, maxDx: maxDx, minDy: minDy, maxDy: maxDy,
                    pmvBaseX: pmvBaseX, pmvBaseY: pmvBaseY,
                    pBase: pBase, cPtr: cPtr, width: width, bx: bx, by: by
                )
                bestCoarseDx = lx
                bestCoarseDy = ly
                bestCoarseCost = lcost
                bestCoarseSAD = lsad
            }
        case .wideDiamond:
            let (rx, ry, rcost, rsad) = refineNear2(
                candCenterDx: bestCoarseDx, candCenterDy: bestCoarseDy,
                candCost: bestCoarseCost, candSAD: bestCoarseSAD,
                minDx: minDx, maxDx: maxDx, minDy: minDy, maxDy: maxDy,
                pmvBaseX: pmvBaseX, pmvBaseY: pmvBaseY,
                pBase: pBase, cPtr: cPtr, width: width, bx: bx, by: by
            )
            bestCoarseDx = rx
            bestCoarseDy = ry
            bestCoarseCost = rcost
            bestCoarseSAD = rsad
            if 384 <= rsad {
                let (lx, ly, lcost, lsad) = searchLdsp(
                    centerX: rx, centerY: ry,
                    startCost: rcost, startSAD: rsad,
                    minDx: minDx, maxDx: maxDx, minDy: minDy, maxDy: maxDy,
                    pmvBaseX: pmvBaseX, pmvBaseY: pmvBaseY,
                    pBase: pBase, cPtr: cPtr, width: width, bx: bx, by: by
                )
                bestCoarseDx = lx
                bestCoarseDy = ly
                bestCoarseCost = lcost
                bestCoarseSAD = lsad
            }
        }
        
        // 3. Candidate consistency bias: snap to (0, 0) or PMV if cost is nearly identical
        if bestCoarseDx != 0 || bestCoarseDy != 0 {
            if zeroSAD < 48 || zeroCost <= (bestCoarseCost + 32) {
                bestCoarseDx = 0
                bestCoarseDy = 0
                bestCoarseSAD = zeroSAD
                bestCoarseCost = zeroCost
            }
        }
        if (bestCoarseDx != pmvClampedX || bestCoarseDy != pmvClampedY) && (pmvClampedX != 0 || pmvClampedY != 0) {
            let pmvPenalty = getMVDPenalty(dx: pmvClampedX, dy: pmvClampedY, pmvDx: pmvBaseX, pmvDy: pmvBaseY) * 4
            let pmvCost = pmvSAD + pmvPenalty
            if pmvCost <= (bestCoarseCost + 25) {
                bestCoarseDx = pmvClampedX
                bestCoarseDy = pmvClampedY
                bestCoarseSAD = pmvSAD
                bestCoarseCost = pmvCost
            }
        }
        
        // 4. Subpixel refinement: Half-pixel search around best integer position
        var bestHpDx: Int = bestCoarseDx * 2
        var bestHpDy: Int = bestCoarseDy * 2
        var bestHpSAD: Int = bestCoarseSAD
        
        if 160 < bestCoarseSAD {
            let pmvHpDx = Int(pmv.dx) / 2
            let pmvHpDy = Int(pmv.dy) / 2
            for oi in 0..<8 {
                let hx = meSearchOffsetX[oi]
                let hy = meSearchOffsetY[oi]
                let hpDx: Int = bestCoarseDx * 2 + hx
                let hpDy: Int = bestCoarseDy * 2 + hy
                
                let intDx: Int = hpDx >> 1
                let intDy: Int = hpDy >> 1
                let fractX: Int = hpDx & 1
                let fractY: Int = hpDy & 1
                
                let blurPenalty = (fractX + fractY) * 16
                let penalty = getMVDPenalty(dx: hpDx, dy: hpDy, pmvDx: pmvHpDx, pmvDy: pmvHpDy) + blurPenalty
                let maxSAD = bestHpSAD - penalty
                if maxSAD <= 0 { continue }
                
                fetchHalfPixelBlock8(
                    plane: pBase, width: width, height: height,
                    intX: bx + intDx, intY: by + intDy,
                    fractX: fractX, fractY: fractY, dest: tPtr, roundOffset: roundOffset
                )
                let sad = compute64PointSADBlocks(cBase: cPtr, pBase: tPtr)
                
                let totalSAD = sad + penalty
                if totalSAD < bestHpSAD {
                    bestHpSAD = totalSAD
                    bestHpDx = hpDx
                    bestHpDy = hpDy
                }
            }
        }
        
        // 5. Subpixel refinement: Quarter-pixel search around best half-pixel position
        var bestQpDx: Int = bestHpDx * 2
        var bestQpDy: Int = bestHpDy * 2
        var bestQpSAD: Int = bestHpSAD
        
        if 64 < bestHpSAD {
            let pmvQpDx = Int(pmv.dx)
            let pmvQpDy = Int(pmv.dy)
            for oi in 0..<8 {
                let qpDx: Int = bestHpDx * 2 + meSearchOffsetX[oi]
                let qpDy: Int = bestHpDy * 2 + meSearchOffsetY[oi]
                
                let intDx: Int = qpDx >> 2
                let intDy: Int = qpDy >> 2
                let remX: Int = qpDx & 3
                let remY: Int = qpDy & 3
                
                let blurPenalty = (remX + remY) * 4
                let penalty = getMVDPenalty(dx: qpDx, dy: qpDy, pmvDx: pmvQpDx, pmvDy: pmvQpDy) + blurPenalty
                let maxSAD = bestQpSAD - penalty
                if maxSAD <= 0 { continue }
                
                fetchQuarterPixelBlock8(
                    plane: pBase, width: width, height: height,
                    intX: bx + intDx, intY: by + intDy,
                    remX: remX, remY: remY, dest: tPtr, roundOffset: roundOffset
                )
                let sad = compute64PointSADBlocks(cBase: cPtr, pBase: tPtr)
                
                let totalSAD = sad + penalty
                if totalSAD < bestQpSAD {
                    bestQpSAD = totalSAD
                    bestQpDx = qpDx
                    bestQpDy = qpDy
                }
            }
        }
        
        let bestMVIsZero = (bestQpDx == 0 && bestQpDy == 0)
        if bestMVIsZero != true {
            if zeroSAD < 32 || zeroSAD <= (bestQpSAD + 24) {
                return (0, 0, zeroSAD)
            }
        }
        if pmvClampedX != 0 || pmvClampedY != 0 {
            let pmvQpX = pmvClampedX * 4
            let pmvQpY = pmvClampedY * 4
            if bestQpDx != pmvQpX || bestQpDy != pmvQpY {
                if pmvSAD <= (bestQpSAD + 16) {
                    return (pmvQpX, pmvQpY, pmvSAD)
                }
            }
        }
        
        return (bestQpDx, bestQpDy, bestQpSAD)
    }

    @inline(__always)
    static func searchPixelsFromArrays(
        currPlane: [Int16], 
        prevPlane: [Int16], 
        cPtr: UnsafeMutablePointer<Int16>,
        oPtr: UnsafeMutablePointer<Int16>,
        tPtr: UnsafeMutablePointer<Int16>,
        width: Int, height: Int, bx: Int, by: Int, range: Int = 8,
        pmv: MotionVector,
        tmv: MotionVector = MotionVector(dx: 0, dy: 0),
        spatialA: MotionVector = MotionVector(dx: 0, dy: 0),
        spatialB: MotionVector = MotionVector(dx: 0, dy: 0),
        spatialC: MotionVector = MotionVector(dx: 0, dy: 0),
        roundOffset: Int,
        contrast: Int = 0,
        membrane: Int = 0,
        gopPosition: Int = 0,
        weights: MEDeciderWeights = MEDeciderWeights.shared
    ) -> (MotionVector, Int) {
        return withUnsafePointers(currPlane, prevPlane) { cBase, pBase in
            fetchPixelsBlock8(plane: cBase, width: width, height: height, x: bx, y: by, dest: cPtr)
            return searchPixels(
                cBase: cBase, pBase: pBase, cPtr: cPtr, oPtr: oPtr, tPtr: tPtr,
                width: width, height: height, bx: bx, by: by, range: range,
                pmv: pmv, tmv: tmv, spatialA: spatialA, spatialB: spatialB, spatialC: spatialC,
                roundOffset: roundOffset,
                contrast: contrast,
                membrane: membrane,
                gopPosition: gopPosition,
                weights: weights
            )
        }
    }
    
    @inline(__always)
    static func searchPixels(
        cBase: UnsafePointer<Int16>, 
        pBase: UnsafePointer<Int16>, 
        cPtr: UnsafeMutablePointer<Int16>,
        oPtr: UnsafeMutablePointer<Int16>,
        tPtr: UnsafeMutablePointer<Int16>,
        width: Int, height: Int, bx: Int, by: Int, range: Int = 8,
        pmv: MotionVector,
        tmv: MotionVector = MotionVector(dx: 0, dy: 0),
        spatialA: MotionVector = MotionVector(dx: 0, dy: 0),
        spatialB: MotionVector = MotionVector(dx: 0, dy: 0),
        spatialC: MotionVector = MotionVector(dx: 0, dy: 0),
        roundOffset: Int,
        contrast: Int = 0,
        membrane: Int = 0,
        gopPosition: Int = 0,
        weights: MEDeciderWeights = MEDeciderWeights.shared
    ) -> (MotionVector, Int) {
        let (dx, dy, sad) = evaluateSearch(
            cPtr: cPtr, pBase: pBase, oPtr: oPtr, tPtr: tPtr,
            width: width, height: height, bx: bx, by: by, range: range,
            pmv: pmv, tmv: tmv, spatialA: spatialA, spatialB: spatialB, spatialC: spatialC,
            roundOffset: roundOffset,
            contrast: contrast,
            membrane: membrane,
            gopPosition: gopPosition,
            weights: weights
        )
        return (MotionVector(dx: Int16(dx), dy: Int16(dy)), sad)
    }

    /// Extract approximate structure contrast (max - min) from 8x8 block
    /// Zero-cost feature extraction without additional SIMD loop overheads.
    @inline(__always)
    static func extractContrast8x8(base: UnsafePointer<Int16>, width: Int, height: Int, bx: Int, by: Int) -> Int {
        let isSafeX = (0 <= bx) && (bx + 8 <= width)
        let isSafeY = (0 <= by) && (by + 8 <= height)
        if isSafeX && isSafeY {
            var vMin = SIMD8<Int16>(repeating: 32767)
            var vMax = SIMD8<Int16>(repeating: -32768)
            for y in 0..<8 {
                let row = UnsafeRawPointer(base.advanced(by: (by + y) * width + bx)).loadUnaligned(as: SIMD8<Int16>.self)
                vMin = pointwiseMin(vMin, row)
                vMax = pointwiseMax(vMax, row)
            }
            return Int(vMax.max() &- vMin.min())
        }

        var minVal: Int32 = 32767
        var maxVal: Int32 = -32768
        for y in 0..<8 {
            let sy = max(0, min(by + y, height - 1))
            let row = base.advanced(by: sy * width)
            for x in 0..<8 {
                let sx = max(0, min(bx + x, width - 1))
                let val = Int32(row[sx])
                minVal = min(minVal, val)
                maxVal = max(maxVal, val)
            }
        }
        
        return Int(maxVal - minVal)
    }

    @inline(__always)
    static func computeChromaSAD(
        currCb: UnsafePointer<Int16>, currCr: UnsafePointer<Int16>,
        refCb: UnsafePointer<Int16>, refCr: UnsafePointer<Int16>,
        cbw: Int, cbh: Int,
        bx: Int, by: Int, refDx: Int, refDy: Int
    ) -> Int {
        // bx/by are quarter-resolution block coordinates; the chroma planes
        // are half resolution, so the block's chroma region is 16×16 at
        // (2bx, 2by). refDx/refDy are the full-resolution-pixel motion vector
        // (ChromaSADCoordinateTests pins this contract), so the chroma
        // displacement is refDx>>1. Callers must NOT pre-scale the MV — the
        // previous call sites passed mv*2, sampling the reference at DOUBLE
        // the true motion displacement (correct only for zero MVs, wrong for
        // exactly the moving blocks the penalty exists for).
        let cx = bx << 1
        let cy = by << 1
        let crx = cx + (refDx >> 1)
        let cry = cy + (refDy >> 1)

        // 4×4 samples strided by 4 across the whole 16×16 chroma region (the
        // corner-only version read the top-left 4×4 — 1/16 coverage, which
        // let chroma mismatches in the remaining area go unseen).
        let isCurrSafe = (0 <= cx) && (0 <= cy) && (cx + 13 <= cbw) && (cy + 13 <= cbh)
        let isRefSafe = (0 <= crx) && (0 <= cry) && (crx + 13 <= cbw) && (cry + 13 <= cbh)

        if isCurrSafe && isRefSafe {
            var sad: Int32 = 0
            for y in 0..<4 {
                let currOffset = (cy + y * 4) * cbw + cx
                let refOffset = (cry + y * 4) * cbw + crx

                let cCb = SIMD4<Int16>(currCb[currOffset], currCb[currOffset + 4], currCb[currOffset + 8], currCb[currOffset + 12])
                let rCb = SIMD4<Int16>(refCb[refOffset], refCb[refOffset + 4], refCb[refOffset + 8], refCb[refOffset + 12])
                let dCb = pointwiseMax(cCb, rCb) &- pointwiseMin(cCb, rCb)

                let cCr = SIMD4<Int16>(currCr[currOffset], currCr[currOffset + 4], currCr[currOffset + 8], currCr[currOffset + 12])
                let rCr = SIMD4<Int16>(refCr[refOffset], refCr[refOffset + 4], refCr[refOffset + 8], refCr[refOffset + 12])
                let dCr = pointwiseMax(cCr, rCr) &- pointwiseMin(cCr, rCr)

                sad &+= (SIMD4<Int32>(truncatingIfNeeded: dCb) &+ SIMD4<Int32>(truncatingIfNeeded: dCr)).wrappedSum()
            }
            return Int(sad) * 4 // Luma SAD scale matching (16 sample positions × 2 planes vs 64 luma points)
        }
        return 1000
    }

    @inline(__always)
    static func computeQuarterPixelSADSubsampled32_Safe_NoFIR(
        curr: UnsafePointer<Int16>, prev: UnsafePointer<Int16>,
        width: Int, bx: Int, by: Int, intDx: Int, intDy: Int
    ) -> Int {
        var sad: Int32 = 0
        for ry in stride(from: 0, to: 32, by: 8) {
            let cy = by + ry
            let rowC = curr.advanced(by: cy * width + bx)
            let py = by + intDy + ry
            let r = prev.advanced(by: py * width + bx + intDx)
            let c32 = UnsafeRawPointer(rowC).loadUnaligned(as: SIMD32<Int16>.self)
            let p32 = UnsafeRawPointer(r).loadUnaligned(as: SIMD32<Int16>.self)
            let cEven = c32.evenHalf
            let pEven = p32.evenHalf
            let d = pointwiseMax(cEven, pEven) &- pointwiseMin(cEven, pEven)
            sad &+= Int32(d.wrappedSum())
        }
        return Int(sad)
    }

    @inline(__always)
    static func computeQuarterPixelSADSubsampled32_Safe_FIR_Y0(
        curr: UnsafePointer<Int16>, prev: UnsafePointer<Int16>,
        width: Int, bx: Int, by: Int, intDx: Int, intDy: Int,
        cX0: Int32, cX1: Int32, cX2: Int32, cX3: Int32
    ) -> Int {
        var sad: Int32 = 0
        let vCX0 = SIMD16<Int32>(repeating: cX0)
        let vCX1 = SIMD16<Int32>(repeating: cX1)
        let vCX2 = SIMD16<Int32>(repeating: cX2)
        let vCX3 = SIMD16<Int32>(repeating: cX3)
        for ry in stride(from: 0, to: 32, by: 8) {
            let cy = by + ry
            let rowC = curr.advanced(by: cy * width + bx)
            let py = by + intDy + ry
            let r0 = prev.advanced(by: py * width + bx + intDx)
            
            let cEven = UnsafeRawPointer(rowC).loadUnaligned(as: SIMD32<Int16>.self).evenHalf
            let m1Even = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(r0.advanced(by: -1)).loadUnaligned(as: SIMD32<Int16>.self).evenHalf)
            let r0Even = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(r0).loadUnaligned(as: SIMD32<Int16>.self).evenHalf)
            let p1Even = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(r0.advanced(by: 1)).loadUnaligned(as: SIMD32<Int16>.self).evenHalf)
            let p2Even = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(r0.advanced(by: 2)).loadUnaligned(as: SIMD32<Int16>.self).evenHalf)
            
            let h0 = (vCX0 &* m1Even &+ vCX1 &* r0Even) &+ (vCX2 &* p1Even &+ vCX3 &* p2Even)
            let pVal = (h0 &+ 3) &>> 3
            let cWide = SIMD16<Int32>(truncatingIfNeeded: cEven)
            let diff = pointwiseMax(cWide, pVal) &- pointwiseMin(cWide, pVal)
            sad &+= diff.wrappedSum()
        }
        return Int(sad)
    }

    @inline(__always)
    static func computeQuarterPixelSADSubsampled32_Safe_FIR_X0(
        curr: UnsafePointer<Int16>, prev: UnsafePointer<Int16>,
        width: Int, bx: Int, by: Int, intDx: Int, intDy: Int,
        cY0: Int32, cY1: Int32, cY2: Int32, cY3: Int32
    ) -> Int {
        var sad: Int32 = 0
        let vCY0 = SIMD16<Int32>(repeating: cY0)
        let vCY1 = SIMD16<Int32>(repeating: cY1)
        let vCY2 = SIMD16<Int32>(repeating: cY2)
        let vCY3 = SIMD16<Int32>(repeating: cY3)
        for ry in stride(from: 0, to: 32, by: 8) {
            let cy = by + ry
            let rowC = curr.advanced(by: cy * width + bx)
            let py = by + intDy + ry
            let rM1 = prev.advanced(by: (py - 1) * width + bx + intDx)
            let r0 = prev.advanced(by: py * width + bx + intDx)
            let rP1 = prev.advanced(by: (py + 1) * width + bx + intDx)
            let rP2 = prev.advanced(by: (py + 2) * width + bx + intDx)
            
            let cEven = UnsafeRawPointer(rowC).loadUnaligned(as: SIMD32<Int16>.self).evenHalf
            let m1Even = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(rM1).loadUnaligned(as: SIMD32<Int16>.self).evenHalf)
            let r0Even = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(r0).loadUnaligned(as: SIMD32<Int16>.self).evenHalf)
            let p1Even = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(rP1).loadUnaligned(as: SIMD32<Int16>.self).evenHalf)
            let p2Even = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(rP2).loadUnaligned(as: SIMD32<Int16>.self).evenHalf)
            
            let vertSum = (vCY0 &* m1Even &+ vCY1 &* r0Even) &+ (vCY2 &* p1Even &+ vCY3 &* p2Even)
            let pVal = (vertSum &+ 3) &>> 3
            let cWide = SIMD16<Int32>(truncatingIfNeeded: cEven)
            let diff = pointwiseMax(cWide, pVal) &- pointwiseMin(cWide, pVal)
            sad &+= diff.wrappedSum()
        }
        return Int(sad)
    }

    @inline(__always)
    static func computeQuarterPixelSADSubsampled32_Safe_FIR_XY(
        curr: UnsafePointer<Int16>, prev: UnsafePointer<Int16>,
        width: Int, bx: Int, by: Int, intDx: Int, intDy: Int,
        cX0: Int32, cX1: Int32, cX2: Int32, cX3: Int32,
        cY0: Int32, cY1: Int32, cY2: Int32, cY3: Int32
    ) -> Int {
        var sad: Int32 = 0
        let vCX0 = SIMD16<Int32>(repeating: cX0)
        let vCX1 = SIMD16<Int32>(repeating: cX1)
        let vCX2 = SIMD16<Int32>(repeating: cX2)
        let vCX3 = SIMD16<Int32>(repeating: cX3)
        let vCY0 = SIMD16<Int32>(repeating: cY0)
        let vCY1 = SIMD16<Int32>(repeating: cY1)
        let vCY2 = SIMD16<Int32>(repeating: cY2)
        let vCY3 = SIMD16<Int32>(repeating: cY3)
        
        for ry in stride(from: 0, to: 32, by: 8) {
            let cy = by + ry
            let rowC = curr.advanced(by: cy * width + bx)
            let py = by + intDy + ry
            let rM1 = prev.advanced(by: (py - 1) * width + bx + intDx)
            let r0 = prev.advanced(by: py * width + bx + intDx)
            let rP1 = prev.advanced(by: (py + 1) * width + bx + intDx)
            let rP2 = prev.advanced(by: (py + 2) * width + bx + intDx)
            
            let cEven = UnsafeRawPointer(rowC).loadUnaligned(as: SIMD32<Int16>.self).evenHalf
            
            let m1_m1 = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(rM1.advanced(by: -1)).loadUnaligned(as: SIMD32<Int16>.self).evenHalf)
            let m1_0  = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(rM1).loadUnaligned(as: SIMD32<Int16>.self).evenHalf)
            let m1_p1 = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(rM1.advanced(by: 1)).loadUnaligned(as: SIMD32<Int16>.self).evenHalf)
            let m1_p2 = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(rM1.advanced(by: 2)).loadUnaligned(as: SIMD32<Int16>.self).evenHalf)
            let vM1 = (vCX0 &* m1_m1 &+ vCX1 &* m1_0) &+ (vCX2 &* m1_p1 &+ vCX3 &* m1_p2)
            
            let r0_m1 = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(r0.advanced(by: -1)).loadUnaligned(as: SIMD32<Int16>.self).evenHalf)
            let r0_0  = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(r0).loadUnaligned(as: SIMD32<Int16>.self).evenHalf)
            let r0_p1 = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(r0.advanced(by: 1)).loadUnaligned(as: SIMD32<Int16>.self).evenHalf)
            let r0_p2 = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(r0.advanced(by: 2)).loadUnaligned(as: SIMD32<Int16>.self).evenHalf)
            let v0 = (vCX0 &* r0_m1 &+ vCX1 &* r0_0) &+ (vCX2 &* r0_p1 &+ vCX3 &* r0_p2)

            let p1_m1 = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(rP1.advanced(by: -1)).loadUnaligned(as: SIMD32<Int16>.self).evenHalf)
            let p1_0  = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(rP1).loadUnaligned(as: SIMD32<Int16>.self).evenHalf)
            let p1_p1 = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(rP1.advanced(by: 1)).loadUnaligned(as: SIMD32<Int16>.self).evenHalf)
            let p1_p2 = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(rP1.advanced(by: 2)).loadUnaligned(as: SIMD32<Int16>.self).evenHalf)
            let vP1 = (vCX0 &* p1_m1 &+ vCX1 &* p1_0) &+ (vCX2 &* p1_p1 &+ vCX3 &* p1_p2)

            let p2_m1 = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(rP2.advanced(by: -1)).loadUnaligned(as: SIMD32<Int16>.self).evenHalf)
            let p2_0  = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(rP2).loadUnaligned(as: SIMD32<Int16>.self).evenHalf)
            let p2_p1 = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(rP2.advanced(by: 1)).loadUnaligned(as: SIMD32<Int16>.self).evenHalf)
            let p2_p2 = SIMD16<Int32>(truncatingIfNeeded: UnsafeRawPointer(rP2.advanced(by: 2)).loadUnaligned(as: SIMD32<Int16>.self).evenHalf)
            let vP2 = (vCX0 &* p2_m1 &+ vCX1 &* p2_0) &+ (vCX2 &* p2_p1 &+ vCX3 &* p2_p2)

            let refVal = (vCY0 &* vM1 &+ vCY1 &* v0) &+ (vCY2 &* vP1 &+ vCY3 &* vP2)
            let pVal = (refVal &+ 31) &>> 6
            let cWide = SIMD16<Int32>(truncatingIfNeeded: cEven)
            let diff = pointwiseMax(cWide, pVal) &- pointwiseMin(cWide, pVal)
            sad &+= diff.wrappedSum()
        }
        return Int(sad)
    }

    @inline(__always)
    static func computeQuarterPixelSADSubsampled32_Unsafe_NoFIR(
        curr: UnsafePointer<Int16>, prev: UnsafePointer<Int16>,
        width: Int, height: Int, bx: Int, by: Int, intDx: Int, intDy: Int
    ) -> Int {
        var sad: Int32 = 0
        for ry in stride(from: 0, to: 32, by: 8) {
            let cy = min(by + ry, height - 1)
            let rowC = curr.advanced(by: cy * width)
            let py = by + intDy + ry
            let sy0 = max(0, min(py, height - 1))
            let r = prev.advanced(by: sy0 * width)
            for rx in stride(from: 0, to: 32, by: 2) {
                let sx = max(0, min(bx + intDx + rx, width - 1))
                sad &+= Int32((Int32(rowC[min(bx + rx, width - 1)]) - Int32(r[sx])).magnitude)
            }
        }
        return Int(sad)
    }

    @inline(__always)
    static func computeQuarterPixelSADSubsampled32_Unsafe_FIR(
        curr: UnsafePointer<Int16>, prev: UnsafePointer<Int16>,
        width: Int, height: Int, bx: Int, by: Int, intDx: Int, intDy: Int,
        cX0: Int32, cX1: Int32, cX2: Int32, cX3: Int32,
        cY0: Int32, cY1: Int32, cY2: Int32, cY3: Int32
    ) -> Int {
        var sad: Int32 = 0
        for ry in stride(from: 0, to: 32, by: 8) {
            let cy = min(by + ry, height - 1)
            let rowC = curr.advanced(by: cy * width)
            let py = by + intDy + ry
            let syM1 = max(0, min(py - 1, height - 1))
            let sy0  = max(0, min(py, height - 1))
            let syP1 = max(0, min(py + 1, height - 1))
            let syP2 = max(0, min(py + 2, height - 1))
            let rM1 = prev.advanced(by: syM1 * width)
            let r0  = prev.advanced(by: sy0 * width)
            let rP1 = prev.advanced(by: syP1 * width)
            let rP2 = prev.advanced(by: syP2 * width)
            for rx in stride(from: 0, to: 32, by: 2) {
                let px = bx &+ intDx &+ rx
                let sxM1 = max(0, min(px - 1, width - 1))
                let sx0  = max(0, min(px, width - 1))
                let sxP1 = max(0, min(px + 1, width - 1))
                let sxP2 = max(0, min(px + 2, width - 1))
                
                let vM1 = cX0 &* Int32(rM1[sxM1]) &+ cX1 &* Int32(rM1[sx0]) &+ cX2 &* Int32(rM1[sxP1]) &+ cX3 &* Int32(rM1[sxP2])
                let v0  = cX0 &* Int32(r0[sxM1])  &+ cX1 &* Int32(r0[sx0])  &+ cX2 &* Int32(r0[sxP1])  &+ cX3 &* Int32(r0[sxP2])
                let vP1 = cX0 &* Int32(rP1[sxM1]) &+ cX1 &* Int32(rP1[sx0]) &+ cX2 &* Int32(rP1[sxP1]) &+ cX3 &* Int32(rP1[sxP2])
                let vP2 = cX0 &* Int32(rP2[sxM1]) &+ cX1 &* Int32(rP2[sx0]) &+ cX2 &* Int32(rP2[sxP1]) &+ cX3 &* Int32(rP2[sxP2])
                
                let refVal = cY0 &* vM1 &+ cY1 &* v0 &+ cY2 &* vP1 &+ cY3 &* vP2
                let pVal = (refVal &+ 31) >> 6
                
                sad &+= Int32((Int32(rowC[min(bx &+ rx, width - 1)]) &- pVal).magnitude)
            }
        }
        return Int(sad)
    }

    @inline(__always)
    static func computeQuarterPixelSADSubsampled32(
        curr: UnsafePointer<Int16>, 
        prev: UnsafePointer<Int16>,
        width: Int, height: Int, bx: Int, by: Int,
        qDx: Int, qDy: Int
    ) -> Int {
        let intDx = qDx >> 2
        let intDy = qDy >> 2
        let fractX = qDx & 3
        let fractY = qDy & 3
        
        let fX = FIRLUMACoeffs[fractX]
        let fY = FIRLUMACoeffs[fractY]
        
        let safe = (0 <= ((bx + intDx) - 1)) && (0 <= ((by + intDy) - 1)) && ((((bx + intDx) + 32) + 2) < width) && ((((by + intDy) + 32) + 2) < height) && ((bx + 32) <= width) && ((by + 32) <= height)
        let useFIR = (fractX != 0 || fractY != 0)
        
        let cX0 = Int32(fX[0]), cX1 = Int32(fX[1]), cX2 = Int32(fX[2]), cX3 = Int32(fX[3])
        let cY0 = Int32(fY[0]), cY1 = Int32(fY[1]), cY2 = Int32(fY[2]), cY3 = Int32(fY[3])
        
        if safe {
            if useFIR {
                if fractY == 0 {
                    return computeQuarterPixelSADSubsampled32_Safe_FIR_Y0(curr: curr, prev: prev, width: width, bx: bx, by: by, intDx: intDx, intDy: intDy, cX0: cX0, cX1: cX1, cX2: cX2, cX3: cX3)
                }
                if fractX == 0 {
                    return computeQuarterPixelSADSubsampled32_Safe_FIR_X0(curr: curr, prev: prev, width: width, bx: bx, by: by, intDx: intDx, intDy: intDy, cY0: cY0, cY1: cY1, cY2: cY2, cY3: cY3)
                }
                return computeQuarterPixelSADSubsampled32_Safe_FIR_XY(curr: curr, prev: prev, width: width, bx: bx, by: by, intDx: intDx, intDy: intDy, cX0: cX0, cX1: cX1, cX2: cX2, cX3: cX3, cY0: cY0, cY1: cY1, cY2: cY2, cY3: cY3)
            }
            return computeQuarterPixelSADSubsampled32_Safe_NoFIR(curr: curr, prev: prev, width: width, bx: bx, by: by, intDx: intDx, intDy: intDy)
        }
        
        if useFIR {
            return computeQuarterPixelSADSubsampled32_Unsafe_FIR(curr: curr, prev: prev, width: width, height: height, bx: bx, by: by, intDx: intDx, intDy: intDy, cX0: cX0, cX1: cX1, cX2: cX2, cX3: cX3, cY0: cY0, cY1: cY1, cY2: cY2, cY3: cY3)
        }
        return computeQuarterPixelSADSubsampled32_Unsafe_NoFIR(curr: curr, prev: prev, width: width, height: height, bx: bx, by: by, intDx: intDx, intDy: intDy)
    }

    static let searchOffsets = [
        (0, -1), (0, 1), (-1, 0), (1, 0),
        (-1, -1), (1, -1), (-1, 1), (1, 1)
    ]

    @inline(__always)
    static func searchPixelsSubpixelRefinement32(
        curr: UnsafePointer<Int16>,
        prev: UnsafePointer<Int16>,
        width: Int, height: Int, bx: Int, by: Int, pmv: MotionVector
    ) -> (MotionVector, Int) {
        // pmv is in full units of Luma dx
        // We convert it to 1/4 units of Luma dx by multiplying by 4
        let baseQx = Int(pmv.dx) * 4
        let baseQy = Int(pmv.dy) * 4
        
        var bestSAD = computeQuarterPixelSADSubsampled32(curr: curr, prev: prev, width: width, height: height, bx: bx, by: by, qDx: baseQx, qDy: baseQy)
        if bestSAD < 256 { return (MotionVector(dx: Int16(baseQx), dy: Int16(baseQy)), bestSAD) }
        
        // 1. Half-pixel search (step = 2 quarter pixels)
        var hpBestQx = baseQx
        var hpBestQy = baseQy
        for (ox, oy) in searchOffsets {
            let qx = baseQx + ox * 2
            let qy = baseQy + oy * 2
            
            let intDx = qx >> 2
            let intDy = qy >> 2
            if (bx + intDx) < -32 || (width + 32) < (bx + intDx + 32) { continue }
            if (by + intDy) < -32 || (height + 32) < (by + intDy + 32) { continue }
            
            let penalty = (Int((ox).magnitude) + Int((oy).magnitude)) * 6
            let maxSAD = bestSAD - penalty
            if maxSAD <= 0 { continue }
            
            let sad = computeQuarterPixelSADSubsampled32(curr: curr, prev: prev, width: width, height: height, bx: bx, by: by, qDx: qx, qDy: qy)
            let totalSAD = sad + penalty
            if totalSAD < bestSAD {
                bestSAD = totalSAD
                hpBestQx = qx
                hpBestQy = qy
            }
        }
        
        if bestSAD < 384 { return (MotionVector(dx: Int16(hpBestQx), dy: Int16(hpBestQy)), bestSAD) }
        
        // 2. Quarter-pixel search (step = 1 quarter pixel)
        var qpBestQx = hpBestQx
        var qpBestQy = hpBestQy
        for (ox, oy) in searchOffsets {
            let qx = hpBestQx + ox
            let qy = hpBestQy + oy
            
            let intDx = qx >> 2
            let intDy = qy >> 2
            if (bx + intDx) < -32 || (width + 32) < (bx + intDx + 32) { continue }
            if (by + intDy) < -32 || (height + 32) < (by + intDy + 32) { continue }
            
            let penalty = (Int((ox).magnitude) + Int((oy).magnitude)) * 4
            let maxSAD = bestSAD - penalty
            if maxSAD <= 0 { continue }
            
            let sad = computeQuarterPixelSADSubsampled32(curr: curr, prev: prev, width: width, height: height, bx: bx, by: by, qDx: qx, qDy: qy)
            let totalSAD = sad + penalty
            if totalSAD < bestSAD {
                bestSAD = totalSAD
                qpBestQx = qx
                qpBestQy = qy
            }
        }
        
        return (MotionVector(dx: Int16(qpBestQx), dy: Int16(qpBestQy)), bestSAD)
    }

    @inline(__always)
    static func searchPixelsSubpixelRefinement32FromArrays(
        currPlane: [Int16],
        prevPlane: [Int16],
        width: Int, height: Int, bx: Int, by: Int, pmv: MotionVector
    ) -> (MotionVector, Int) {
        return withUnsafePointers(currPlane, prevPlane) { cBase, pBase in
            searchPixelsSubpixelRefinement32(curr: cBase, prev: pBase, width: width, height: height, bx: bx, by: by, pmv: pmv)
        }
    }

    @inline(__always)
    static func computeOcclusionScores(
        currPlane: [Int16],
        prevPlane: [Int16],
        width: Int,
        height: Int,
        globalPrior: MotionVector
    ) -> [Int] {
        let colCount = (width + 7) / 8
        let rowCount = (height + 7) / 8
        var scores = [Int](repeating: 0, count: rowCount * colCount)
        let gdy = Int(globalPrior.dy) / 4
        let gdx = Int(globalPrior.dx) / 4
        
        withUnsafePointers(currPlane, prevPlane, mut: &scores) { cBase, pBase, sBase in
            for row in 0..<rowCount {
                let by = row * 8
                let bHeight = min(8, height - by)
                if bHeight <= 0 { continue }
                
                let pBy = min(max(0, by + gdy), height - bHeight)
                for col in 0..<colCount {
                    let bx = col * 8
                    let bWidth = min(8, width - bx)
                    if bWidth <= 0 { continue }
                    
                    var cProfile0: Int32 = 0, cProfile1: Int32 = 0, cProfile2: Int32 = 0, cProfile3: Int32 = 0
                    var cProfile4: Int32 = 0, cProfile5: Int32 = 0, cProfile6: Int32 = 0, cProfile7: Int32 = 0
                    
                    var pProfile0: Int32 = 0, pProfile1: Int32 = 0, pProfile2: Int32 = 0, pProfile3: Int32 = 0
                    var pProfile4: Int32 = 0, pProfile5: Int32 = 0, pProfile6: Int32 = 0, pProfile7: Int32 = 0
                    
                    // apply gdx
                    let px0 = max(0, min(width - 1, bx + 0 + gdx)) - bx
                    let px1 = max(0, min(width - 1, bx + 1 + gdx)) - bx
                    let px2 = max(0, min(width - 1, bx + 2 + gdx)) - bx
                    let px3 = max(0, min(width - 1, bx + 3 + gdx)) - bx
                    let px4 = max(0, min(width - 1, bx + 4 + gdx)) - bx
                    let px5 = max(0, min(width - 1, bx + 5 + gdx)) - bx
                    let px6 = max(0, min(width - 1, bx + 6 + gdx)) - bx
                    let px7 = max(0, min(width - 1, bx + 7 + gdx)) - bx
                    
                    for y in 0..<bHeight {
                        let cOffset = (by + y) * width + bx
                        let cRow = cBase.advanced(by: cOffset)
                        if 0 < bWidth { cProfile0 &+= Int32(cRow[0]) }
                        if 1 < bWidth { cProfile1 &+= Int32(cRow[1]) }
                        if 2 < bWidth { cProfile2 &+= Int32(cRow[2]) }
                        if 3 < bWidth { cProfile3 &+= Int32(cRow[3]) }
                        if 4 < bWidth { cProfile4 &+= Int32(cRow[4]) }
                        if 5 < bWidth { cProfile5 &+= Int32(cRow[5]) }
                        if 6 < bWidth { cProfile6 &+= Int32(cRow[6]) }
                        if 7 < bWidth { cProfile7 &+= Int32(cRow[7]) }
                        
                        let pOffset = (pBy + y) * width + bx
                        let pRow = pBase.advanced(by: pOffset)
                        
                        if 0 < bWidth { pProfile0 &+= Int32(pRow[px0]) }
                        if 1 < bWidth { pProfile1 &+= Int32(pRow[px1]) }
                        if 2 < bWidth { pProfile2 &+= Int32(pRow[px2]) }
                        if 3 < bWidth { pProfile3 &+= Int32(pRow[px3]) }
                        if 4 < bWidth { pProfile4 &+= Int32(pRow[px4]) }
                        if 5 < bWidth { pProfile5 &+= Int32(pRow[px5]) }
                        if 6 < bWidth { pProfile6 &+= Int32(pRow[px6]) }
                        if 7 < bWidth { pProfile7 &+= Int32(pRow[px7]) }
                    }
                    
                    var score = 0
                    if 0 < bWidth { score &+= Int(Int32((cProfile0 - pProfile0).magnitude)) }
                    if 1 < bWidth { score &+= Int(Int32((cProfile1 - pProfile1).magnitude)) }
                    if 2 < bWidth { score &+= Int(Int32((cProfile2 - pProfile2).magnitude)) }
                    if 3 < bWidth { score &+= Int(Int32((cProfile3 - pProfile3).magnitude)) }
                    if 4 < bWidth { score &+= Int(Int32((cProfile4 - pProfile4).magnitude)) }
                    if 5 < bWidth { score &+= Int(Int32((cProfile5 - pProfile5).magnitude)) }
                    if 6 < bWidth { score &+= Int(Int32((cProfile6 - pProfile6).magnitude)) }
                    if 7 < bWidth { score &+= Int(Int32((cProfile7 - pProfile7).magnitude)) }
                    
                    sBase[row * colCount + col] = score / (bHeight * bWidth)
                }
            }
        }
        return scores
    }
}

private struct SliceResult: Sendable {
    let startIdx: Int
    let dx: [Int16]
    let dy: [Int16]
    let sads: [Int]
    let refDirs: [Bool]
    let pdx: [Int16]
    let pdy: [Int16]
    let ldx: [Int16]
    let ldy: [Int16]
    let psad: [Int]
    let lsad: [Int]
}

private struct BidirSliceContext: Sendable {
    let targetWidth: Int
    let targetHeight: Int
    let colCount: Int
    let cbw: Int
    let cbh: Int
    let roundOffset: Int
    let gopPosition: Int
    let wantDual: Bool
    let hasSkipMap: Bool
    let skipMap: [BlockMode]
    let prevMVs: MotionVectors
    let cBase: UnsafeSendablePointer<Int16>
    let pBase: UnsafeSendablePointer<Int16>
    let nBase: UnsafeSendablePointer<Int16>
    let cY: UnsafeSendablePointer<Int16>
    let pY: UnsafeSendablePointer<Int16>
    let nY: UnsafeSendablePointer<Int16>
    let cCb: UnsafeSendablePointer<Int16>
    let cCr: UnsafeSendablePointer<Int16>
    let pCb: UnsafeSendablePointer<Int16>
    let pCr: UnsafeSendablePointer<Int16>
    let nCb: UnsafeSendablePointer<Int16>
    let nCr: UnsafeSendablePointer<Int16>
    let memWrapper: UnsafeSendablePointer<Int8>?
    let memCount: Int
    let pool: BlockViewPool
}

@inline(__always)
private func executeBidirSlice(
    ctx: BidirSliceContext,
    startIdx: Int,
    sliceBlocksCount: Int
) -> SliceResult {
    let tmpC = ctx.pool.get64()
    let tmpO = ctx.pool.get64()
    let tmpT = ctx.pool.get64()
    defer {
        ctx.pool.put(tmpC)
        ctx.pool.put(tmpO)
        ctx.pool.put(tmpT)
    }
    let cPtr = tmpC.base
    let oPtr = tmpO.base
    let tPtr = tmpT.base
    
    let cBase = ctx.cBase.ptr
    let pBase = ctx.pBase.ptr
    let nBase = ctx.nBase.ptr
    
    let cY = ctx.cY.ptr
    let pY = ctx.pY.ptr
    let nY = ctx.nY.ptr
    let cCb = ctx.cCb.ptr
    let cCr = ctx.cCr.ptr
    let pCb = ctx.pCb.ptr
    let pCr = ctx.pCr.ptr
    let nCb = ctx.nCb.ptr
    let nCr = ctx.nCr.ptr

    let targetWidth = ctx.targetWidth
    let targetHeight = ctx.targetHeight
    let colCount = ctx.colCount
    let cbw = ctx.cbw
    let cbh = ctx.cbh
    let roundOffset = ctx.roundOffset
    let gopPosition = ctx.gopPosition
    let wantDual = ctx.wantDual
    let hasSkipMap = ctx.hasSkipMap
    let skipMapConst = ctx.skipMap
    let prevMVsConst = ctx.prevMVs
    let memPtr = ctx.memWrapper?.ptr
    let memCount = ctx.memCount

    var dx = [Int16](repeating: 0, count: sliceBlocksCount)
    var dy = [Int16](repeating: 0, count: sliceBlocksCount)
    var sads = [Int](repeating: 0, count: sliceBlocksCount)
    var refDirs = [Bool](repeating: false, count: sliceBlocksCount)
    let dualSliceCount: Int
    if wantDual {
        dualSliceCount = sliceBlocksCount
    } else {
        dualSliceCount = 0
    }
    var pdx = [Int16](repeating: 0, count: dualSliceCount)
    var pdy = [Int16](repeating: 0, count: dualSliceCount)
    var ldx = [Int16](repeating: 0, count: dualSliceCount)
    var ldy = [Int16](repeating: 0, count: dualSliceCount)
    var psad = [Int](repeating: 0, count: dualSliceCount)
    var lsad = [Int](repeating: 0, count: dualSliceCount)
    let meWeights = MEDeciderWeights.shared

    for i in 0..<sliceBlocksCount {
        let idx = startIdx + i
        let col = idx % colCount
        let row = idx / colCount
        let bx = col * 8
        let by = row * 8

        let isSkipBlock = hasSkipMap && skipMapConst[idx] != .inter
        if isSkipBlock {
            if wantDual != true {
                dx[i] = 0
                dy[i] = 0
                sads[i] = 0
                refDirs[i] = false
                continue
            }
        }

        MotionEstimation.fetchPixelsBlock8(plane: cBase, width: targetWidth, height: targetHeight, x: bx, y: by, dest: cPtr)

        let mvADx = if 0 < col { dx[i - 1] } else { Int16(0) }
        let mvADy = if 0 < col { dy[i - 1] } else { Int16(0) }
        let mvBDx = if colCount <= i { dx[i - colCount] } else { Int16(0) }
        let mvBDy = if colCount <= i { dy[i - colCount] } else { Int16(0) }
        let mvCDx = if colCount <= (i - 1) && col < (colCount - 1) { dx[i - colCount + 1] } else { Int16(0) }
        let mvCDy = if colCount <= (i - 1) && col < (colCount - 1) { dy[i - colCount + 1] } else { Int16(0) }

        let sADx = Int(mvADx) >> 2
        let sADy = Int(mvADy) >> 2
        let sBDx = Int(mvBDx) >> 2
        let sBDy = Int(mvBDy) >> 2
        let sCDx = Int(mvCDx) >> 2
        let sCDy = Int(mvCDy) >> 2

        let mvA = MotionVector(dx: Int16(sADx), dy: Int16(sADy))
        let mvB = MotionVector(dx: Int16(sBDx), dy: Int16(sBDy))
        let mvC = MotionVector(dx: Int16(sCDx), dy: Int16(sCDy))

        let pmvDx = MotionEstimation.median(sADx, sBDx, sCDx)
        let pmvDy = MotionEstimation.median(sADy, sBDy, sCDy)
        let pmv = MotionVector(dx: Int16(pmvDx), dy: Int16(pmvDy))

        let tmv: MotionVector
        if idx < prevMVsConst.count {
            tmv = MotionVector(dx: Int16(Int(prevMVsConst.dx[idx]) >> 2), dy: Int16(Int(prevMVsConst.dy[idx]) >> 2))
        } else {
            tmv = MotionVector(dx: 0, dy: 0)
        }

        let currContrast = MotionEstimation.extractContrast8x8(base: cBase, width: targetWidth, height: targetHeight, bx: bx, by: by)

        let memVal: Int
        if let mem = memPtr, idx < memCount {
            memVal = Int(mem[idx])
        } else {
            memVal = 0
        }
        let hasLtrAffinity = 2 < memVal

        var mvPrev = MotionVector(dx: 0, dy: 0)
        var mutSADPrev = 0
        var prevSAD = 0
        var mvNext = MotionVector(dx: 0, dy: 0)
        var mutSADNext = 0
        var bestMV = MotionVector(dx: 0, dy: 0)
        var dir = false
        var cachedNextChromaSad: Int = -1
        // Evaluate Prev direction
        let (pMV, pSAD) = MotionEstimation.searchPixels(
            cBase: cBase, pBase: pBase,
            cPtr: cPtr, oPtr: oPtr, tPtr: tPtr,
            width: targetWidth, height: targetHeight, bx: bx, by: by, range: 8,
            pmv: pmv, tmv: tmv, spatialA: mvA, spatialB: mvB, spatialC: mvC,
            roundOffset: roundOffset,
            contrast: currContrast,
            membrane: memVal,
            gopPosition: gopPosition,
            weights: meWeights
        )
        mvPrev = pMV
        mutSADPrev = pSAD
        let prevChromaSad = MotionEstimation.computeChromaSAD(currCb: cCb, currCr: cCr, refCb: pCb, refCr: pCr, cbw: cbw, cbh: cbh, bx: bx, by: by, refDx: Int(mvPrev.dx), refDy: Int(mvPrev.dy))
        mutSADPrev += prevChromaSad / 4
        prevSAD = mutSADPrev

        if 1024 < mutSADPrev {
            let dynT = max(1024, currContrast * 48)
            if dynT < mutSADPrev {
                var mvVariance = 0
                if 0 < col && colCount <= i {
                    let dxDiff = Int((Int(dx[i - 1]) - Int(dx[i - colCount])).magnitude)
                    let dyDiff = Int((Int(dy[i - 1]) - Int(dy[i - colCount])).magnitude)
                    mvVariance = dxDiff + dyDiff
                }
                if 32 < mvVariance && (dynT * 2) < mutSADPrev {
                    let (expMv, expSad) = MotionEstimation.searchPixels(
                        cBase: cBase, pBase: pBase,
                        cPtr: cPtr, oPtr: oPtr, tPtr: tPtr,
                        width: targetWidth, height: targetHeight, bx: bx, by: by, range: 12,
                        pmv: mvPrev, tmv: tmv, spatialA: mvA, spatialB: mvB, spatialC: mvC,
                        roundOffset: roundOffset,
                        contrast: currContrast,
                        membrane: memVal,
                        gopPosition: gopPosition,
                        weights: meWeights
                    )
                    if expSad < mutSADPrev {
                        mvPrev = expMv
                        mutSADPrev = expSad
                        prevSAD = mutSADPrev
                    }
                }
            }
        }

        bestMV = mvPrev
        dir = false

        // Evaluate Next (LTR) direction
        let (nMV, nSAD) = MotionEstimation.searchPixels(
            cBase: cBase, pBase: nBase,
            cPtr: cPtr, oPtr: oPtr, tPtr: tPtr,
            width: targetWidth, height: targetHeight, bx: bx, by: by, range: 8,
            pmv: pmv, tmv: tmv, spatialA: mvA, spatialB: mvB, spatialC: mvC,
            roundOffset: roundOffset,
            contrast: currContrast,
            membrane: memVal,
            gopPosition: gopPosition,
            weights: meWeights
        )
        mvNext = nMV
        mutSADNext = nSAD

        let nextChromaSad = MotionEstimation.computeChromaSAD(currCb: cCb, currCr: cCr, refCb: nCb, refCr: nCr, cbw: cbw, cbh: cbh, bx: bx, by: by, refDx: Int(mvNext.dx), refDy: Int(mvNext.dy))
        cachedNextChromaSad = nextChromaSad

        let earlyExitThreshold = min(1536, 512 + (gopPosition * 16))
        let gopPenalty = min(1024, gopPosition * 16)

        if earlyExitThreshold <= prevSAD {
            let mvEnergyNext = Int((Int(mvNext.dx)).magnitude) + Int((Int(mvNext.dy)).magnitude)
            let effectiveGopPenalty: Int
            if mutSADNext < 384 || hasLtrAffinity {
                effectiveGopPenalty = 0
            } else {
                effectiveGopPenalty = gopPenalty
            }
            let baselinePenalty = (mvEnergyNext * 8) + 32 + effectiveGopPenalty

            if mutSADNext + baselinePenalty < prevSAD {
                let nextContrast = MotionEstimation.extractContrast8x8(base: nBase, width: targetWidth, height: targetHeight, bx: bx + (Int(mvNext.dx) >> 2), by: by + (Int(mvNext.dy) >> 2))
                let contrastDiff = Int((currContrast - nextContrast).magnitude)
                let structurePenalty = (contrastDiff * contrastDiff) / 4
                let chromaPenalty = nextChromaSad / 4

                let totalNextPenalty = ((mutSADNext + baselinePenalty) + (structurePenalty + chromaPenalty))
                let energyNext = (mvNext.dy * mvNext.dy) + (mvNext.dx * mvNext.dx)
                let energyPrev = (mvPrev.dy * mvPrev.dy) + (mvPrev.dx * mvPrev.dx)

                switch true {
                case totalNextPenalty < prevSAD:
                    bestMV = mvNext
                    dir = true
                case (totalNextPenalty == prevSAD) && (energyNext < energyPrev):
                    bestMV = mvNext
                    dir = true
                default:
                    break
                }
            }
        }

        let dynamicThreshold = max(1024, currContrast * 48)
        let resolvedNextChromaSad: Int
        if cachedNextChromaSad != -1 {
            resolvedNextChromaSad = cachedNextChromaSad
        } else {
            resolvedNextChromaSad = MotionEstimation.computeChromaSAD(currCb: cCb, currCr: cCr, refCb: nCb, refCr: nCr, cbw: cbw, cbh: cbh, bx: bx, by: by, refDx: Int(mvNext.dx), refDy: Int(mvNext.dy))
        }
        let finalSAD: Int
        if dir {
            finalSAD = mutSADNext + (resolvedNextChromaSad / 4)
        } else {
            finalSAD = prevSAD
        }
        switch true {
        case dynamicThreshold < finalSAD:
            dx[i] = 0
            dy[i] = 0
        case finalSAD < 256:
            dx[i] = bestMV.dx * 4
            dy[i] = bestMV.dy * 4
        default:
            let refPlaneY: UnsafePointer<Int16>
            if dir {
                refPlaneY = nY
            } else {
                refPlaneY = pY
            }
            let (refinedMV, _) = MotionEstimation.searchPixelsSubpixelRefinement32(
                curr: cY, prev: refPlaneY,
                width: targetWidth, height: targetHeight, bx: col * 32, by: row * 32, pmv: bestMV
            )
            dx[i] = refinedMV.dx
            dy[i] = refinedMV.dy
        }
        sads[i] = finalSAD
        refDirs[i] = dir

        if wantDual {
            let ltrSAD = mutSADNext + (resolvedNextChromaSad / 4)
            psad[i] = prevSAD
            lsad[i] = ltrSAD

            if dir != true {
                pdx[i] = dx[i]
                pdy[i] = dy[i]
            } else {
                switch true {
                case dynamicThreshold < prevSAD:
                    pdx[i] = 0
                    pdy[i] = 0
                case prevSAD < 256:
                    pdx[i] = mvPrev.dx * 4
                    pdy[i] = mvPrev.dy * 4
                default:
                    let (rmv, _) = MotionEstimation.searchPixelsSubpixelRefinement32(
                        curr: cY, prev: pY,
                        width: targetWidth, height: targetHeight, bx: col * 32, by: row * 32, pmv: mvPrev
                    )
                    pdx[i] = rmv.dx
                    pdy[i] = rmv.dy
                }
            }

            if dir {
                ldx[i] = dx[i]
                ldy[i] = dy[i]
            } else {
                switch true {
                case dynamicThreshold < ltrSAD:
                    ldx[i] = 0
                    ldy[i] = 0
                case ltrSAD < 256:
                    ldx[i] = mvNext.dx * 4
                    ldy[i] = mvNext.dy * 4
                default:
                    let (rmv, _) = MotionEstimation.searchPixelsSubpixelRefinement32(
                        curr: cY, prev: nY,
                        width: targetWidth, height: targetHeight, bx: col * 32, by: row * 32, pmv: mvNext
                    )
                    ldx[i] = rmv.dx
                    ldy[i] = rmv.dy
                }
            }
        }
        if isSkipBlock {
            dx[i] = 0
            dy[i] = 0
            sads[i] = 0
            refDirs[i] = false
        }
    }
    return SliceResult(startIdx: startIdx, dx: dx, dy: dy, sads: sads, refDirs: refDirs, pdx: pdx, pdy: pdy, ldx: ldx, ldy: ldy, psad: psad, lsad: lsad)
}

/// Bidirectional MV calculation: searches MV in both forward (prev) and backward (next) frames,
@inline(__always)
func computeBidirectionalMotionVectors(curr: PlaneData420, prev: PlaneData420, next: PlaneData420, prevMVs: MotionVectors, pool: BlockViewPool, roundOffset: Int, gopPosition: Int, skipMap: [BlockMode], cachedNextSub2: [Int16]?, cachedNextSub1: [Int16]?, dualOut: DualMVSink?, membrane: MEMembraneState? = nil) async -> (MotionVectors, [Int], [Bool], [Int], [Int16], [Int16]) {
    let dx = curr.width
    let dy = curr.height
    let l1dx = (dx + 1) / 2
    let l1dy = (dy + 1) / 2
    let l0dx = (l1dx + 1) / 2
    let l0dy = (l1dy + 1) / 2
    
    // Compute DWT LL band (Base8 resolution) for current frame
    async let (currSub2, rCurrSub2) = extractSingleTransformSubband32(r: curr.rY, width: dx, height: dy, pool: pool)
    
    // Forward reference DWT LL band
    async let (prevSub2, rPrevSub2) = extractSingleTransformSubband32(r: prev.rY, width: dx, height: dy, pool: pool)
    
    let cS2 = await currSub2
    let pS2 = await prevSub2
    
    let nS2: [Int16]
    let nR2: @Sendable () -> Void
    if let cached = cachedNextSub2 {
        nS2 = cached
        nR2 = {}
    } else {
        let (sub2, releaseFn) = await extractSingleTransformSubband32(r: next.rY, width: dx, height: dy, pool: pool)
        nS2 = sub2
        nR2 = releaseFn
    }
    
    async let (currSub1, rCurrSub1) = extractSingleTransformSubband16(r: Int16Reader(data: cS2, width: l1dx, height: l1dy), width: l1dx, height: l1dy, pool: pool)
    async let (prevSub1, rPrevSub1) = extractSingleTransformSubband16(r: Int16Reader(data: pS2, width: l1dx, height: l1dy), width: l1dx, height: l1dy, pool: pool)
    
    let (cS1, cR1) = await (currSub1, rCurrSub1)
    let (pS1, pR1) = await (prevSub1, rPrevSub1)
    
    let nS1: [Int16]
    let nR1: @Sendable () -> Void
    if let cached = cachedNextSub1 {
        nS1 = cached
        nR1 = {}
    } else {
        let (sub1, releaseFn) = await extractSingleTransformSubband16(r: Int16Reader(data: nS2, width: l1dx, height: l1dy), width: l1dx, height: l1dy, pool: pool)
        nS1 = sub1
        nR1 = releaseFn
    }
    
    let cR2 = await rCurrSub2
    let pR2 = await rPrevSub2
    
    defer {
        cR2()
        cR1()
        pR2()
        pR1()
        nR2()
        nR1()
    }
    
    let targetWidth = l0dx
    let targetHeight = l0dy
    let colCount = (targetWidth + 7) / 8
    let rowCount = (targetHeight + 7) / 8
    let blocks8Count = colCount * rowCount
    
    var mvs = MotionVectors(count: blocks8Count)
    var sads = [Int](repeating: 0, count: blocks8Count)
    var refDirs = [Bool](repeating: false, count: blocks8Count)
    
    let wantDual = (dualOut != nil)

    let cS1BaseWrapper = UnsafeSendablePointer(ptr: cS1.withUnsafeBufferPointer { $0.baseAddress! })
    let pS1BaseWrapper = UnsafeSendablePointer(ptr: pS1.withUnsafeBufferPointer { $0.baseAddress! })
    let nS1BaseWrapper = UnsafeSendablePointer(ptr: nS1.withUnsafeBufferPointer { $0.baseAddress! })
    
    let cYWrapper = UnsafeSendablePointer(ptr: curr.y.withUnsafeBufferPointer { $0.baseAddress! })
    let pYWrapper = UnsafeSendablePointer(ptr: prev.y.withUnsafeBufferPointer { $0.baseAddress! })
    let nYWrapper = UnsafeSendablePointer(ptr: next.y.withUnsafeBufferPointer { $0.baseAddress! })
    let cCbWrapper = UnsafeSendablePointer(ptr: curr.cb.withUnsafeBufferPointer { $0.baseAddress! })
    let cCrWrapper = UnsafeSendablePointer(ptr: curr.cr.withUnsafeBufferPointer { $0.baseAddress! })
    let pCbWrapper = UnsafeSendablePointer(ptr: prev.cb.withUnsafeBufferPointer { $0.baseAddress! })
    let pCrWrapper = UnsafeSendablePointer(ptr: prev.cr.withUnsafeBufferPointer { $0.baseAddress! })
    let nCbWrapper = UnsafeSendablePointer(ptr: next.cb.withUnsafeBufferPointer { $0.baseAddress! })
    let nCrWrapper = UnsafeSendablePointer(ptr: next.cr.withUnsafeBufferPointer { $0.baseAddress! })
    let cbw = (curr.width + 1) / 2
    let cbh = (curr.height + 1) / 2
    
    let hasSkipMap = 0 < skipMap.count
    let memWrapper: UnsafeSendablePointer<Int8>?
    let memCount: Int
    if let mem = membrane, 0 < mem.potentials.count {
        memWrapper = UnsafeSendablePointer(ptr: mem.potentials.withUnsafeBufferPointer { $0.baseAddress! })
        memCount = mem.potentials.count
    } else {
        memWrapper = nil
        memCount = 0
    }
    
    let ctx = BidirSliceContext(
        targetWidth: targetWidth,
        targetHeight: targetHeight,
        colCount: colCount,
        cbw: cbw,
        cbh: cbh,
        roundOffset: roundOffset,
        gopPosition: gopPosition,
        wantDual: wantDual,
        hasSkipMap: hasSkipMap,
        skipMap: skipMap,
        prevMVs: prevMVs,
        cBase: cS1BaseWrapper,
        pBase: pS1BaseWrapper,
        nBase: nS1BaseWrapper,
        cY: cYWrapper,
        pY: pYWrapper,
        nY: nYWrapper,
        cCb: cCbWrapper,
        cCr: cCrWrapper,
        pCb: pCbWrapper,
        pCr: pCrWrapper,
        nCb: nCbWrapper,
        nCr: nCrWrapper,
        memWrapper: memWrapper,
        memCount: memCount,
        pool: pool
    )
    
    let numSlices = 4
    let rowsPerSlice = (rowCount + numSlices - 1) / numSlices
    
    let results: [SliceResult] = await withTaskGroup(of: SliceResult.self) { group in
        for sliceIdx in 0..<numSlices {
            let startRow = sliceIdx * rowsPerSlice
            let endRow = min(rowCount, startRow + rowsPerSlice)
            if endRow <= startRow { continue }
            
            let startIdx = startRow * colCount
            let sliceBlocksCount = (endRow - startRow) * colCount
            
            group.addTask { [ctx] in
                executeBidirSlice(
                    ctx: ctx,
                    startIdx: startIdx,
                    sliceBlocksCount: sliceBlocksCount
                )
            }
        }
        
        var collected = [SliceResult]()
        for await res in group {
            collected.append(res)
        }
        return collected
    }
    
    let dualCount: Int
    if wantDual {
        dualCount = blocks8Count
    } else {
        dualCount = 0
    }
    var dualPrev = MotionVectors(count: dualCount)
    var dualLtr = MotionVectors(count: dualCount)
    var dualPrevSAD = [Int](repeating: 0, count: dualCount)
    var dualLtrSAD = [Int](repeating: 0, count: dualCount)
    for res in results {
        let endIdx = res.startIdx + res.dx.count
        mvs.dx.replaceSubrange(res.startIdx..<endIdx, with: res.dx)
        mvs.dy.replaceSubrange(res.startIdx..<endIdx, with: res.dy)
        sads.replaceSubrange(res.startIdx..<endIdx, with: res.sads)
        refDirs.replaceSubrange(res.startIdx..<endIdx, with: res.refDirs)
        if wantDual {
            dualPrev.dx.replaceSubrange(res.startIdx..<endIdx, with: res.pdx)
            dualPrev.dy.replaceSubrange(res.startIdx..<endIdx, with: res.pdy)
            dualLtr.dx.replaceSubrange(res.startIdx..<endIdx, with: res.ldx)
            dualLtr.dy.replaceSubrange(res.startIdx..<endIdx, with: res.ldy)
            dualPrevSAD.replaceSubrange(res.startIdx..<endIdx, with: res.psad)
            dualLtrSAD.replaceSubrange(res.startIdx..<endIdx, with: res.lsad)
        }
    }
    if let sink = dualOut {
        sink.prevMVs = dualPrev
        sink.ltrMVs = dualLtr
        sink.prevSADs = dualPrevSAD
        sink.ltrSADs = dualLtrSAD
    }
    
    let occlusionScores = MotionEstimation.computeOcclusionScores(currPlane: cS1, prevPlane: pS1, width: targetWidth, height: targetHeight, globalPrior: MotionVector(dx: 0, dy: 0))
    return (mvs, sads, refDirs, occlusionScores, nS2, nS1)
}
