import Foundation

@inline(__always)
func calculateSAD32x32(pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, currStride: Int, prevStride: Int) -> Int {
    var sumVec0 = SIMD16<UInt16>()
    var sumVec1 = SIMD16<UInt16>()
    
    for y in 0..<32 {
        let currRow = pCurr.advanced(by: y * currStride)
        let prevRow = pPrev.advanced(by: y * prevStride)

        let c0 = UnsafeRawPointer(currRow).loadUnaligned(as: SIMD16<Int16>.self)
        let p0 = UnsafeRawPointer(prevRow).loadUnaligned(as: SIMD16<Int16>.self)
        let c1 = UnsafeRawPointer(currRow.advanced(by: 16)).loadUnaligned(as: SIMD16<Int16>.self)
        let p1 = UnsafeRawPointer(prevRow.advanced(by: 16)).loadUnaligned(as: SIMD16<Int16>.self)

        let diff0 = c0 &- p0
        let mask0 = diff0 &>> 15
        let abs0 = (diff0 ^ mask0) &- mask0

        let diff1 = c1 &- p1
        let mask1 = diff1 &>> 15
        let abs1 = (diff1 ^ mask1) &- mask1

        sumVec0 &+= SIMD16<UInt16>(truncatingIfNeeded: abs0)
        sumVec1 &+= SIMD16<UInt16>(truncatingIfNeeded: abs1)
    }
    
    let total0 = SIMD16<UInt32>(truncatingIfNeeded: sumVec0).wrappedSum()
    let total1 = SIMD16<UInt32>(truncatingIfNeeded: sumVec1).wrappedSum()
    return Int(total0 &+ total1)
}

@inline(__always)
func calculateSAD64x64(pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, currStride: Int, prevStride: Int) -> Int {
    var sumVec0 = SIMD16<UInt16>()
    var sumVec1 = SIMD16<UInt16>()
    var sumVec2 = SIMD16<UInt16>()
    var sumVec3 = SIMD16<UInt16>()
    
    for y in 0..<64 {
        let currRow = pCurr.advanced(by: y * currStride)
        let prevRow = pPrev.advanced(by: y * prevStride)

        let c0 = UnsafeRawPointer(currRow).loadUnaligned(as: SIMD16<Int16>.self)
        let p0 = UnsafeRawPointer(prevRow).loadUnaligned(as: SIMD16<Int16>.self)
        let c1 = UnsafeRawPointer(currRow.advanced(by: 16)).loadUnaligned(as: SIMD16<Int16>.self)
        let p1 = UnsafeRawPointer(prevRow.advanced(by: 16)).loadUnaligned(as: SIMD16<Int16>.self)
        let c2 = UnsafeRawPointer(currRow.advanced(by: 32)).loadUnaligned(as: SIMD16<Int16>.self)
        let p2 = UnsafeRawPointer(prevRow.advanced(by: 32)).loadUnaligned(as: SIMD16<Int16>.self)
        let c3 = UnsafeRawPointer(currRow.advanced(by: 48)).loadUnaligned(as: SIMD16<Int16>.self)
        let p3 = UnsafeRawPointer(prevRow.advanced(by: 48)).loadUnaligned(as: SIMD16<Int16>.self)

        let diff0 = c0 &- p0
        let mask0 = diff0 &>> 15
        let abs0 = (diff0 ^ mask0) &- mask0

        let diff1 = c1 &- p1
        let mask1 = diff1 &>> 15
        let abs1 = (diff1 ^ mask1) &- mask1

        let diff2 = c2 &- p2
        let mask2 = diff2 &>> 15
        let abs2 = (diff2 ^ mask2) &- mask2

        let diff3 = c3 &- p3
        let mask3 = diff3 &>> 15
        let abs3 = (diff3 ^ mask3) &- mask3

        sumVec0 &+= SIMD16<UInt16>(truncatingIfNeeded: abs0)
        sumVec1 &+= SIMD16<UInt16>(truncatingIfNeeded: abs1)
        sumVec2 &+= SIMD16<UInt16>(truncatingIfNeeded: abs2)
        sumVec3 &+= SIMD16<UInt16>(truncatingIfNeeded: abs3)
    }
    
    let total0 = SIMD16<UInt32>(truncatingIfNeeded: sumVec0).wrappedSum()
    let total1 = SIMD16<UInt32>(truncatingIfNeeded: sumVec1).wrappedSum()
    let total2 = SIMD16<UInt32>(truncatingIfNeeded: sumVec2).wrappedSum()
    let total3 = SIMD16<UInt32>(truncatingIfNeeded: sumVec3).wrappedSum()
    
    return Int(total0 &+ total1 &+ total2 &+ total3)
}

@inline(__always)
func downscale8x(pd: PlaneData420) -> (data: [Int16], w: Int, h: Int) {
    let w = pd.width / 8
    let h = pd.height / 8
    var out = [Int16](repeating: 0, count: w * h)
    
    let pdWidth = pd.width
    pd.y.withUnsafeBufferPointer { ptr in
        guard let pY = ptr.baseAddress else { return }
        out.withUnsafeMutableBufferPointer { oPtr in
            guard let pOut = oPtr.baseAddress else { return }
            
            for y in 0..<h {
                let py = y * 8
                let outRow = y * w
                for x in 0..<w {
                    let px = x * 8
                    var sum: Int32 = 0
                    for dy in 0..<8 {
                        let off = (py + dy) * pdWidth + px
                        let v = UnsafeRawPointer(pY.advanced(by: off)).loadUnaligned(as: SIMD8<Int16>.self)
                        let v32 = SIMD8<Int32>(clamping: v)
                        sum &+= v32[0] &+ v32[1] &+ v32[2] &+ v32[3] &+ v32[4] &+ v32[5] &+ v32[6] &+ v32[7]
                    }
                    pOut[outRow + x] = Int16(sum / 64)
                }
            }
        }
    }
    return (out, w, h)
}

@inline(__always)
func calculateDownscaledSADStats(layer0Curr: [Int16], layer0Prev: [Int16], w: Int, h: Int) -> (meanSAD: Int, maxBlockSAD: Int) {
    // 64x64 block is 8x8 in downscaled layer0
    let mbSize = 8
    let mbCols = (w + mbSize - 1) / mbSize
    let mbRows = (h + mbSize - 1) / mbSize
    
    var totalSAD = 0
    var maxSAD = 0
    
    layer0Curr.withUnsafeBufferPointer { cPtr in
        guard let pC = cPtr.baseAddress else { return }
        layer0Prev.withUnsafeBufferPointer { pPtr in
            guard let pP = pPtr.baseAddress else { return }
            
            for mbY in 0..<mbRows {
                let startY = mbY * mbSize
                let actH = min(mbSize, h - startY)
                for mbX in 0..<mbCols {
                    let startX = mbX * mbSize
                    let actW = min(mbSize, w - startX)
                    
                    var blockSAD = 0
                    for y in 0..<actH {
                        let row = (startY + y) * w
                        for x in 0..<actW {
                            let idx = row + startX + x
                            let diff = Int(pC[idx]) - Int(pP[idx])
                            blockSAD += diff > 0 ? diff : -diff
                        }
                    }
                    
                    // Scale up by 64 to closely estimate the original 64x64 block SAD scale 
                    // (since layer0 has 1/64 the pixel count and the per-pixel diff is comparable)
                    let scaledSAD = blockSAD * 64
                    totalSAD += scaledSAD
                    if scaledSAD > maxSAD { maxSAD = scaledSAD }
                }
            }
        }
    }
    
    let meanSAD = mbCols * mbRows > 0 ? totalSAD / (mbCols * mbRows) : 0
    return (meanSAD, maxSAD)
}

struct MotionVector: Sendable {
    let dx: Int
    let dy: Int
}

struct MotionVectors: Sendable {
    var vectors: [SIMD2<Int16>]

    init(count: Int) {
        self.vectors = [SIMD2<Int16>](repeating: .zero, count: count)
    }
}

enum MotionNode: Sendable {
    case leaf(mv: SIMD2<Int16>)
    indirect case split(tl: MotionNode, tr: MotionNode, bl: MotionNode, br: MotionNode)
}

struct MotionTree: Sendable {
    var ctuNodes: [MotionNode]
    var width: Int
    var height: Int
}

struct MVGrid {
    var grid: [SIMD2<Int16>]
    let stride: Int
    let minSize: Int

    init(width: Int, height: Int, minSize: Int) {
        self.stride = (width + minSize - 1) / minSize
        let rows = (height + minSize - 1) / minSize
        self.grid = [SIMD2<Int16>](repeating: .zero, count: stride * rows)
        self.minSize = minSize
    }

    func getPMV(x: Int, y: Int, w: Int) -> SIMD2<Int16> {
        let gx = x / minSize
        let gy = y / minSize
        let gw = w / minSize
        
        let hasLeft = gx > 0
        let hasTop = gy > 0
        let hasTopRight = gy > 0 && (gx + gw) < stride

        let idxLeft = hasLeft ? (gy * stride + (gx - 1)) : -1
        let idxTop = hasTop ? ((gy - 1) * stride + gx) : -1
        let idxTopRight = hasTopRight ? ((gy - 1) * stride + (gx + gw)) : -1

        var count = 0
        if hasLeft { count += 1 }
        if hasTop { count += 1 }
        if hasTopRight { count += 1 }

        if count == 0 { return .zero }
        if count == 1 {
            return grid[hasLeft ? idxLeft : (hasTop ? idxTop : idxTopRight)]
        }
        if count == 2 {
            var dxSum = 0
            var dySum = 0
            if hasLeft { let v = grid[idxLeft]; dxSum += Int(v.x); dySum += Int(v.y) }
            if hasTop { let v = grid[idxTop]; dxSum += Int(v.x); dySum += Int(v.y) }
            if hasTopRight { let v = grid[idxTopRight]; dxSum += Int(v.x); dySum += Int(v.y) }
            return SIMD2<Int16>(Int16(dxSum / 2), Int16(dySum / 2))
        }

        let lvec = grid[idxLeft]
        let tvec = grid[idxTop]
        let rvec = grid[idxTopRight]

        let minX = min(lvec.x, min(tvec.x, rvec.x))
        let maxX = max(lvec.x, max(tvec.x, rvec.x))
        let pmvX = lvec.x + tvec.x + rvec.x - minX - maxX

        let minY = min(lvec.y, min(tvec.y, rvec.y))
        let maxY = max(lvec.y, max(tvec.y, rvec.y))
        let pmvY = lvec.y + tvec.y + rvec.y - minY - maxY

        return SIMD2<Int16>(pmvX, pmvY)
    }

    mutating func fill(x: Int, y: Int, w: Int, h: Int, mv: SIMD2<Int16>) {
        let gx = x / minSize
        let gy = y / minSize
        let gw = w / minSize
        let gh = h / minSize
        
        for i in 0..<gh {
            let row = (gy + i) * stride
            for j in 0..<gw {
                grid[row + gx + j] = mv
            }
        }
    }
}

@inline(__always)
func estimateMBMEBlock64x64(
    pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, w: Int, h: Int,
    startX: Int, startY: Int, searchRange: Int,
    mvs: inout MotionVectors, mvIdx: Int
) {
    var bestSAD = calculateSAD64x64(
        pCurr: pCurr.advanced(by: startY * w + startX),
        pPrev: pPrev.advanced(by: startY * w + startX),
        currStride: w, prevStride: w
    )
    var bestDX = 0
    var bestDY = 0

    let earlyExitThreshold = 64 * 64 * 1
    if earlyExitThreshold < bestSAD {
        let minSafeDX = max(-1 * searchRange, -startX)
        let maxSafeDX = min(searchRange, w - 64 - startX)
        let minSafeDY = max(-1 * searchRange, -startY)
        let maxSafeDY = min(searchRange, h - 64 - startY)

        let negSearchRange = -1 * searchRange
        let posSearchRange = searchRange

        var step = searchRange / 2
        while step >= 1 {
            var currentBestDX = bestDX
            var currentBestDY = bestDY
            var currentBestSAD = bestSAD

            for j in -1...1 {
                for i in -1...1 {
                    if i == 0 && j == 0 { continue }
                    
                    let dx = bestDX + i * step
                    let dy = bestDY + j * step

                    if dx < negSearchRange || posSearchRange < dx || dy < negSearchRange || posSearchRange < dy { continue }

                    let diffX = dx >= 0 ? dx : -1 * dx
                    let diffY = dy >= 0 ? dy : -1 * dy
                    let penalty = diffX + diffY
                    if currentBestSAD <= penalty { continue }

                    let sad: Int
                    let isDYSafe = dy >= minSafeDY && dy <= maxSafeDY
                    if isDYSafe && dx >= minSafeDX && dx <= maxSafeDX {
                        let refYRowOffset = (startY + dy) * w
                        sad = calculateSAD64x64(
                            pCurr: pCurr.advanced(by: startY * w + startX),
                            pPrev: pPrev.advanced(by: refYRowOffset + startX + dx),
                            currStride: w, prevStride: w
                        )
                    } else {
                        sad = calculateSADEdge(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, actW: 64, actH: 64, dx: dx, dy: dy)
                    }

                    let totalSad = sad &+ penalty
                    if totalSad < currentBestSAD {
                        currentBestSAD = totalSad
                        currentBestDX = dx
                        currentBestDY = dy
                    }
                }
            }

            bestDX = currentBestDX
            bestDY = currentBestDY
            bestSAD = currentBestSAD

            step /= 2
        }
    }

    mvs.vectors[mvIdx] = SIMD2(Int16(bestDX * 4), Int16(bestDY * 4))
}

@inline(__always)
func estimateMBMEBlock32x32(
    pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, w: Int, h: Int,
    startX: Int, startY: Int, searchRange: Int,
    mvs: inout MotionVectors, mvIdx: Int
) {
    var bestSAD = calculateSAD32x32(
        pCurr: pCurr.advanced(by: startY * w + startX),
        pPrev: pPrev.advanced(by: startY * w + startX),
        currStride: w, prevStride: w
    )
    var bestDX = 0
    var bestDY = 0

    let earlyExitThreshold = 32 * 32 * 1
    if earlyExitThreshold < bestSAD {
        let minSafeDX = max(-1 * searchRange, -startX)
        let maxSafeDX = min(searchRange, w - 32 - startX)
        let minSafeDY = max(-1 * searchRange, -startY)
        let maxSafeDY = min(searchRange, h - 32 - startY)

        let negSearchRange = -1 * searchRange
        let posSearchRange = searchRange

        var step = searchRange / 2
        while 1 <= step {
            var currentBestDX = bestDX
            var currentBestDY = bestDY
            var currentBestSAD = bestSAD

            for j in -1...1 {
                for i in -1...1 {
                    if i == 0 && j == 0 { continue }
                    
                    let dx = bestDX + i * step
                    let dy = bestDY + j * step

                    if dx < negSearchRange || posSearchRange < dx || dy < negSearchRange || posSearchRange < dy { continue }

                    let diffX = dx >= 0 ? dx : -1 * dx
                    let diffY = dy >= 0 ? dy : -1 * dy
                    let penalty = diffX + diffY
                    if currentBestSAD <= penalty { continue }

                    let sad: Int
                    let isDYSafe = dy >= minSafeDY && dy <= maxSafeDY
                    if isDYSafe && dx >= minSafeDX && dx <= maxSafeDX {
                        let refYRowOffset = (startY + dy) * w
                        sad = calculateSAD32x32(
                            pCurr: pCurr.advanced(by: startY * w + startX),
                            pPrev: pPrev.advanced(by: refYRowOffset + startX + dx),
                            currStride: w, prevStride: w
                        )
                    } else {
                        sad = calculateSADEdge(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, actW: 32, actH: 32, dx: dx, dy: dy)
                    }

                    let totalSad = sad &+ penalty
                    if totalSad < currentBestSAD {
                        currentBestSAD = totalSad
                        currentBestDX = dx
                        currentBestDY = dy
                    }
                }
            }

            bestDX = currentBestDX
            bestDY = currentBestDY
            bestSAD = currentBestSAD

            step /= 2
        }
    }

    mvs.vectors[mvIdx] = SIMD2(Int16(bestDX * 4), Int16(bestDY * 4))
}

@inline(__always)
func estimateMBMEBlockEdge(
    pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, w: Int, h: Int,
    startX: Int, startY: Int, actW: Int, actH: Int, searchRange: Int,
    mvs: inout MotionVectors, mvIdx: Int,
    fracRefBuf: UnsafeMutablePointer<Int16>, fracExtBuf: UnsafeMutablePointer<Int16>
) {
    var bestSAD = calculateSADEdge(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, actW: actW, actH: actH, dx: 0, dy: 0)
    var bestDX = 0
    var bestDY = 0

    let earlyExitThreshold = actW * actH * 1
    if earlyExitThreshold < bestSAD {
        let negSearchRange = -1 * searchRange
        let posSearchRange = searchRange

        var step = searchRange / 2
        while 1 <= step {
            var currentBestDX = bestDX
            var currentBestDY = bestDY
            var currentBestSAD = bestSAD

            for j in -1...1 {
                for i in -1...1 {
                    if i == 0 && j == 0 { continue }

                    let dx = bestDX + i * step
                    let dy = bestDY + j * step

                    if dx < negSearchRange || posSearchRange < dx || dy < negSearchRange || posSearchRange < dy { continue }

                    let diffX = dx >= 0 ? dx : -1 * dx
                    let diffY = dy >= 0 ? dy : -1 * dy
                    let penalty = diffX + diffY
                    if currentBestSAD <= penalty { continue }

                    let refX = startX + dx
                    let refY = startY + dy
                    let sad: Int

                    if 0 <= refX && 0 <= refY && refX + actW <= w && refY + actH <= h {
                        var s: UInt = 0
                        for y in 0..<actH {
                            let currRow = (startY + y) * w + startX
                            let prevRow = (refY + y) * w + refX
                            let pCurrRow = pCurr.advanced(by: currRow)
                            let pPrevRow = pPrev.advanced(by: prevRow)
                            for x in 0..<actW {
                                let diff = Int(pCurrRow[x]) - Int(pPrevRow[x])
                                let mask = diff >> 31
                                s &+= UInt((diff ^ mask) - mask)
                            }
                        }
                        sad = Int(s)
                    } else {
                        sad = calculateSADEdge(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, actW: actW, actH: actH, dx: dx, dy: dy)
                    }

                    let totalSAD = sad &+ penalty
                    if totalSAD < currentBestSAD {
                        currentBestSAD = totalSAD
                        currentBestDX = dx
                        currentBestDY = dy
                    }
                }
            }

            bestDX = currentBestDX
            bestDY = currentBestDY
            bestSAD = currentBestSAD
            step /= 2
        }
    }

    let refinedQMV = refineFractionalMBME(
        pCurr: pCurr, pPrev: pPrev, w: w, h: h,
        startX: startX, startY: startY, actW: actW, actH: actH,
        bestIntDX: bestDX, bestIntDY: bestDY, bestIntSAD: bestSAD,
        fracRefBuffer: fracRefBuf, fracExtBuffer: fracExtBuf
    )

    mvs.vectors[mvIdx] = refinedQMV
}

// MARK: - Subsampled SAD for Coarse Search

/// 64x64ブロックのSADを1行おき（32行分）で計算する。
/// ダウンスケールバッファを必要としないゼロコストの粗探索用SAD。
/// 結果は近似値であり、正確なSADの約50%の値を返す。
@inline(__always)
func calculateSAD64x64_Subsample(pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, currStride: Int, prevStride: Int) -> Int {
    var sumVec0 = SIMD16<UInt16>()
    var sumVec1 = SIMD16<UInt16>()
    var sumVec2 = SIMD16<UInt16>()
    var sumVec3 = SIMD16<UInt16>()

    // 1行おきに計算（y=0,2,4,...62の32行分）
    for y in stride(from: 0, to: 64, by: 2) {
        let currRow = pCurr.advanced(by: y * currStride)
        let prevRow = pPrev.advanced(by: y * prevStride)

        let c0 = UnsafeRawPointer(currRow).loadUnaligned(as: SIMD16<Int16>.self)
        let p0 = UnsafeRawPointer(prevRow).loadUnaligned(as: SIMD16<Int16>.self)
        let c1 = UnsafeRawPointer(currRow.advanced(by: 16)).loadUnaligned(as: SIMD16<Int16>.self)
        let p1 = UnsafeRawPointer(prevRow.advanced(by: 16)).loadUnaligned(as: SIMD16<Int16>.self)
        let c2 = UnsafeRawPointer(currRow.advanced(by: 32)).loadUnaligned(as: SIMD16<Int16>.self)
        let p2 = UnsafeRawPointer(prevRow.advanced(by: 32)).loadUnaligned(as: SIMD16<Int16>.self)
        let c3 = UnsafeRawPointer(currRow.advanced(by: 48)).loadUnaligned(as: SIMD16<Int16>.self)
        let p3 = UnsafeRawPointer(prevRow.advanced(by: 48)).loadUnaligned(as: SIMD16<Int16>.self)

        let diff0 = c0 &- p0
        let mask0 = diff0 &>> 15
        let abs0 = (diff0 ^ mask0) &- mask0

        let diff1 = c1 &- p1
        let mask1 = diff1 &>> 15
        let abs1 = (diff1 ^ mask1) &- mask1

        let diff2 = c2 &- p2
        let mask2 = diff2 &>> 15
        let abs2 = (diff2 ^ mask2) &- mask2

        let diff3 = c3 &- p3
        let mask3 = diff3 &>> 15
        let abs3 = (diff3 ^ mask3) &- mask3

        sumVec0 &+= SIMD16<UInt16>(truncatingIfNeeded: abs0)
        sumVec1 &+= SIMD16<UInt16>(truncatingIfNeeded: abs1)
        sumVec2 &+= SIMD16<UInt16>(truncatingIfNeeded: abs2)
        sumVec3 &+= SIMD16<UInt16>(truncatingIfNeeded: abs3)
    }

    let total0 = SIMD16<UInt32>(truncatingIfNeeded: sumVec0).wrappedSum()
    let total1 = SIMD16<UInt32>(truncatingIfNeeded: sumVec1).wrappedSum()
    let total2 = SIMD16<UInt32>(truncatingIfNeeded: sumVec2).wrappedSum()
    let total3 = SIMD16<UInt32>(truncatingIfNeeded: sumVec3).wrappedSum()

    return Int(total0 &+ total1 &+ total2 &+ total3)
}

// MARK: - estimateMBME (Subsample SAD + PMV)

/// サブサンプリングSADによる高速動き推定。
/// 従来のステップサーチ構造をそのまま維持し、SADの計算を1行おき（50%計算量）にすることで高速化。
/// 加えて、PMV（周辺ブロックのMV中央値）を探索開始点として使用し、収束を加速。
/// デコーダへの出力フォーマットは従来と同一（変更不要）。
@inline(__always)
func estimateMBME(curr: PlaneData420, prev: PlaneData420) -> MotionVectors {
    let mbSize = 64
    let w = curr.width
    let h = curr.height
    let mbCols = (w + mbSize - 1) / mbSize

    var mvs = MotionVectors(count: mbCols * ((h + mbSize - 1) / mbSize))

    let searchRange = 16

    var fracRefBuffer = [Int16](repeating: 0, count: 64 * 64)
    var fracExtBuffer = [Int16](repeating: 0, count: 71 * 71)

    curr.y.withUnsafeBufferPointer { currPtr in
        guard let pCurr = currPtr.baseAddress else { return }
        prev.y.withUnsafeBufferPointer { prevPtr in
            guard let pPrev = prevPtr.baseAddress else { return }
            fracRefBuffer.withUnsafeMutableBufferPointer { fracRefPtr in
                guard let pFracRef = fracRefPtr.baseAddress else { return }
                fracExtBuffer.withUnsafeMutableBufferPointer { fracExtPtr in
                    guard let pFracExt = fracExtPtr.baseAddress else { return }

            let fullMbCols = w / mbSize
            let fullMbRows = h / mbSize
            let remW = w % mbSize
            let remH = h % mbSize

            // --- フルサイズブロック: サブサンプリングSAD + PMV初期値 ---
            for mbY in 0..<fullMbRows {
                let startY = mbY * mbSize
                for mbX in 0..<fullMbCols {
                    let startX = mbX * mbSize
                    let idx = mbY * mbCols + mbX

                    let minSafeDX = max(-1 * searchRange, -startX)
                    let maxSafeDX = min(searchRange, w - 64 - startX)
                    let minSafeDY = max(-1 * searchRange, -startY)
                    let maxSafeDY = min(searchRange, h - 64 - startY)

                    // PMVを探索開始点として活用
                    let pmv = calculatePMV(mvs: mvs, mbX: mbX, mbY: mbY, mbCols: mbCols)
                    var bestDX = max(minSafeDX, min(maxSafeDX, pmv.dx >> 2))
                    var bestDY = max(minSafeDY, min(maxSafeDY, pmv.dy >> 2))

                    // 初期SADの計算（サブサンプリング）
                    var bestSAD: Int
                    if bestDY >= minSafeDY && bestDY <= maxSafeDY && bestDX >= minSafeDX && bestDX <= maxSafeDX {
                        bestSAD = calculateSAD64x64_Subsample(
                            pCurr: pCurr.advanced(by: startY * w + startX),
                            pPrev: pPrev.advanced(by: (startY + bestDY) * w + startX + bestDX),
                            currStride: w, prevStride: w
                        )
                    } else {
                        bestSAD = calculateSADEdge(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, actW: 64, actH: 64, dx: bestDX, dy: bestDY)
                    }

                    // (0,0)のSADも必ず評価（PMVより良い場合があるため）
                    if bestDX != 0 || bestDY != 0 {
                        let zeroSAD = calculateSAD64x64_Subsample(
                            pCurr: pCurr.advanced(by: startY * w + startX),
                            pPrev: pPrev.advanced(by: startY * w + startX),
                            currStride: w, prevStride: w
                        )
                        if zeroSAD < bestSAD {
                            bestSAD = zeroSAD
                            bestDX = 0
                            bestDY = 0
                        }
                    }

                    // earlyExitの閾値（サブサンプリングは半分の行数なので閾値も調整）
                    let earlyExitThreshold = 64 * 32 * 1
                    if earlyExitThreshold < bestSAD {
                        let negSearchRange = -1 * searchRange
                        let posSearchRange = searchRange

                        var step = searchRange / 2
                        while 1 <= step {
                            var currentBestDX = bestDX
                            var currentBestDY = bestDY
                            var currentBestSAD = bestSAD

                            // step==1のとき: フルSADでbestSADを再計算（サブサンプリング→フルの精度補正）
                            if step == 1 {
                                if bestDY >= minSafeDY && bestDY <= maxSafeDY && bestDX >= minSafeDX && bestDX <= maxSafeDX {
                                    currentBestSAD = calculateSAD64x64(
                                        pCurr: pCurr.advanced(by: startY * w + startX),
                                        pPrev: pPrev.advanced(by: (startY + bestDY) * w + startX + bestDX),
                                        currStride: w, prevStride: w
                                    )
                                } else {
                                    currentBestSAD = calculateSADEdge(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, actW: 64, actH: 64, dx: bestDX, dy: bestDY)
                                }
                            }

                            for j in -1...1 {
                                for i in -1...1 {
                                    if i == 0 && j == 0 { continue }

                                    let dx = bestDX + i * step
                                    let dy = bestDY + j * step

                                    if dx < negSearchRange || posSearchRange < dx || dy < negSearchRange || posSearchRange < dy { continue }

                                    let diffX = dx >= 0 ? dx : -1 * dx
                                    let diffY = dy >= 0 ? dy : -1 * dy
                                    let penalty = diffX + diffY
                                    if currentBestSAD <= penalty { continue }

                                    let sad: Int
                                    let isDYSafe = dy >= minSafeDY && dy <= maxSafeDY
                                    if isDYSafe && dx >= minSafeDX && dx <= maxSafeDX {
                                        if step == 1 {
                                            // 最終ステップ: フルSADで精密比較
                                            sad = calculateSAD64x64(
                                                pCurr: pCurr.advanced(by: startY * w + startX),
                                                pPrev: pPrev.advanced(by: (startY + dy) * w + startX + dx),
                                                currStride: w, prevStride: w
                                            )
                                        } else {
                                            // 粗ステップ: サブサンプリングSADで高速比較
                                            sad = calculateSAD64x64_Subsample(
                                                pCurr: pCurr.advanced(by: startY * w + startX),
                                                pPrev: pPrev.advanced(by: (startY + dy) * w + startX + dx),
                                                currStride: w, prevStride: w
                                            )
                                        }
                                    } else {
                                        sad = calculateSADEdge(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, actW: 64, actH: 64, dx: dx, dy: dy)
                                    }

                                    let totalSad = sad &+ penalty
                                    if totalSad < currentBestSAD {
                                        currentBestSAD = totalSad
                                        currentBestDX = dx
                                        currentBestDY = dy
                                    }
                                }
                            }

                            bestDX = currentBestDX
                            bestDY = currentBestDY
                            bestSAD = currentBestSAD

                            step /= 2
                        }
                    }

                    // Fractional Component Refinement
                    let refinedQMV = refineFractionalMBME(
                        pCurr: pCurr, pPrev: pPrev, w: w, h: h,
                        startX: startX, startY: startY, actW: 64, actH: 64,
                        bestIntDX: bestDX, bestIntDY: bestDY, bestIntSAD: bestSAD,
                        fracRefBuffer: pFracRef, fracExtBuffer: pFracExt
                    )
                    
                    mvs.vectors[idx] = refinedQMV
                }
            }

            // --- 端部ブロック: 従来方式にフォールバック ---
            if 0 < remW {
                let mbX = fullMbCols
                let startX = mbX * mbSize
                for mbY in 0..<fullMbRows {
                    let startY = mbY * mbSize
                    let idx = mbY * mbCols + mbX
                    estimateMBMEBlockEdge(
                        pCurr: pCurr, pPrev: pPrev, w: w, h: h,
                        startX: startX, startY: startY, actW: remW, actH: mbSize, searchRange: searchRange,
                        mvs: &mvs, mvIdx: idx, fracRefBuf: pFracRef, fracExtBuf: pFracExt
                    )
                }
            }

            if 0 < remH {
                let mbY = fullMbRows
                let startY = mbY * mbSize
                for mbX in 0..<fullMbCols {
                    let startX = mbX * mbSize
                    let idx = mbY * mbCols + mbX
                    estimateMBMEBlockEdge(
                        pCurr: pCurr, pPrev: pPrev, w: w, h: h,
                        startX: startX, startY: startY, actW: mbSize, actH: remH, searchRange: searchRange,
                        mvs: &mvs, mvIdx: idx, fracRefBuf: pFracRef, fracExtBuf: pFracExt
                    )
                }
            }

            if 0 < remW && 0 < remH {
                let mbX = fullMbCols
                let mbY = fullMbRows
                let startX = mbX * mbSize
                let startY = mbY * mbSize
                let idx = mbY * mbCols + mbX
                estimateMBMEBlockEdge(
                        pCurr: pCurr, pPrev: pPrev, w: w, h: h,
                        startX: startX, startY: startY, actW: remW, actH: remH, searchRange: searchRange,
                        mvs: &mvs, mvIdx: idx, fracRefBuf: pFracRef, fracExtBuf: pFracExt
                    )
            }
                }
            }
        }
    }

    return mvs
}

@inline(__always)
func calculateSADEdge(pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, w: Int, h: Int, startX: Int, startY: Int, actW: Int, actH: Int, dx: Int, dy: Int) -> Int {
    var sad: UInt = 0
    let refX = startX + dx
    let minSafeX = max(0, min(actW, -refX))
    let maxSafeX = max(0, min(actW, w - refX))

    for y in 0..<actH {
        let cy = startY + y
        let currRow = cy * w

        let py = max(0, min(h - 1, cy + dy))
        let prevRow = py * w

        let pCurrRow = pCurr.advanced(by: currRow + startX)
        let pPrevRow = pPrev.advanced(by: prevRow)

        if 0 < minSafeX {
            let leftEdgeVal = pPrevRow[0]
            for x in 0..<minSafeX {
                let diff = Int(pCurrRow[x]) - Int(leftEdgeVal)
                let mask = diff >> 31
                sad &+= UInt((diff ^ mask) - mask)
            }
        }
        
        let copyCount = maxSafeX - minSafeX
        if 0 < copyCount {
            let pPrevSafe = pPrevRow.advanced(by: refX + minSafeX)
            let pCurrSafe = pCurrRow.advanced(by: minSafeX)
            
            var x = 0
            if 16 <= copyCount {
                var sumVec = SIMD16<UInt16>()
                while x <= copyCount - 16 {
                    let c0 = UnsafeRawPointer(pCurrSafe.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
                    let p0 = UnsafeRawPointer(pPrevSafe.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
                    let diff0 = c0 &- p0
                    let mask0 = diff0 &>> 15
                    let abs0 = (diff0 ^ mask0) &- mask0
                    sumVec &+= SIMD16<UInt16>(truncatingIfNeeded: abs0)
                    x += 16
                }
                sad &+= UInt(SIMD16<UInt32>(truncatingIfNeeded: sumVec).wrappedSum())
            }
            
            while x < copyCount {
                let diff = Int(pCurrSafe[x]) - Int(pPrevSafe[x])
                let mask = diff >> 31
                sad &+= UInt((diff ^ mask) - mask)
                x += 1
            }
        }

        if maxSafeX < actW {
            let rightEdgeVal = pPrevRow[w - 1]
            for x in maxSafeX..<actW {
                let diff = Int(pCurrRow[x]) - Int(rightEdgeVal)
                let mask = diff >> 31
                sad &+= UInt((diff ^ mask) - mask)
            }
        }
    }
    return Int(sad)
}

@inline(__always)
func applyMBME(prev: PlaneData420, mvs: MotionVectors) async -> PlaneData420 {
    let mbSize = 64
    let w = prev.width
    let h = prev.height
    let mbCols = (w + mbSize - 1) / mbSize

    @Sendable @inline(__always)
    func apply(data: [Int16], pW: Int, pH: Int, div: Int) async -> [Int16] {
        if pW == 0 || pH == 0 { return data }
        var out = [Int16](repeating: 0, count: pW * pH)

        let localMbCols = mbCols
        let pMbSize = mbSize / div

        let fullMbCols = pW / pMbSize
        let fullMbRows = pH / pMbSize
        let remW = pW % pMbSize
        let remH = pH % pMbSize

        data.withUnsafeBufferPointer { pPtr in
            guard let pData = pPtr.baseAddress else { return }
            out.withUnsafeMutableBufferPointer { oPtr in
                guard let pOut = oPtr.baseAddress else { return }

                var extBuffer = [Int16](repeating: 0, count: 71 * 71)

                @inline(__always)
                func applyBlock(mbX: Int, mbY: Int, actW: Int, actH: Int) {
                    let startX = mbX * pMbSize
                    let startY = mbY * pMbSize
                    let idx = mbY * localMbCols + mbX
                    let vec = mvs.vectors[idx]
                    
                    let qdx = Int(vec.x) / div
                    let qdy = Int(vec.y) / div
                    let dx = qdx >> 2
                    let dy = qdy >> 2
                    let fracX = qdx & 3
                    let fracY = qdy & 3

                    let refX = startX + dx
                    let refY = startY + dy
                    
                    let isSafe = (refX - 3 >= 0) && (refY - 3 >= 0) && (refX + actW + 4 <= pW) && (refY + actH + 4 <= pH)

                    if isSafe {
                        if fracX == 0 && fracY == 0 {
                            switch actW {
                            case 64:
                                for y in 0..<actH {
                                    let dstRow = (startY + y) * pW
                                    let srcRow = (refY + y) * pW
                                    let c0 = UnsafeRawPointer(pData.advanced(by: srcRow + refX)).loadUnaligned(as: SIMD16<Int16>.self)
                                    let c1 = UnsafeRawPointer(pData.advanced(by: srcRow + refX + 16)).loadUnaligned(as: SIMD16<Int16>.self)
                                    let c2 = UnsafeRawPointer(pData.advanced(by: srcRow + refX + 32)).loadUnaligned(as: SIMD16<Int16>.self)
                                    let c3 = UnsafeRawPointer(pData.advanced(by: srcRow + refX + 48)).loadUnaligned(as: SIMD16<Int16>.self)
                                    let pDst = UnsafeMutableRawPointer(pOut.advanced(by: dstRow + startX))
                                    pDst.storeBytes(of: c0, as: SIMD16<Int16>.self)
                                    pDst.advanced(by: 32).storeBytes(of: c1, as: SIMD16<Int16>.self)
                                    pDst.advanced(by: 64).storeBytes(of: c2, as: SIMD16<Int16>.self)
                                    pDst.advanced(by: 96).storeBytes(of: c3, as: SIMD16<Int16>.self)
                                }
                            case 32:
                                for y in 0..<actH {
                                    let dstRow = (startY + y) * pW
                                    let srcRow = (refY + y) * pW
                                    let c0 = UnsafeRawPointer(pData.advanced(by: srcRow + refX)).loadUnaligned(as: SIMD16<Int16>.self)
                                    let c1 = UnsafeRawPointer(pData.advanced(by: srcRow + refX + 16)).loadUnaligned(as: SIMD16<Int16>.self)
                                    let pDst = UnsafeMutableRawPointer(pOut.advanced(by: dstRow + startX))
                                    pDst.storeBytes(of: c0, as: SIMD16<Int16>.self)
                                    pDst.advanced(by: 32).storeBytes(of: c1, as: SIMD16<Int16>.self)
                                }
                            default:
                                for y in 0..<actH {
                                    let dstRow = (startY + y) * pW
                                    let srcRow = (refY + y) * pW
                                    pOut.advanced(by: dstRow + startX).update(from: pData.advanced(by: srcRow + refX), count: actW)
                                }
                            }
                        } else {
                            subpixelInterpolateBlock(
                                src: pData, srcStride: pW,
                                dst: pOut.advanced(by: startY * pW + startX), dstStride: pW,
                                width: actW, height: actH,
                                fracX: fracX, fracY: fracY,
                                startX: refX, startY: refY
                            )
                        }
                    } else {
                        if fracX == 0 && fracY == 0 {
                            let minSafeX = max(0, min(actW, -refX))
                            let maxSafeX = max(0, min(actW, pW - refX))

                            for y in 0..<actH {
                                let dstY = startY + y
                                let srcY = max(0, min(pH - 1, dstY + dy))
                                let dstRow = dstY * pW
                                let srcRow = srcY * pW
                                
                                let pDstBase = pOut.advanced(by: dstRow + startX)

                                if 0 < minSafeX {
                                    let leftEdgeVal = pData[srcRow]
                                    for x in 0..<minSafeX {
                                        pDstBase[x] = leftEdgeVal
                                    }
                                }
                                
                                let copyCount = maxSafeX - minSafeX
                                if 0 < copyCount {
                                    pDstBase.advanced(by: minSafeX).update(from: pData.advanced(by: srcRow + refX + minSafeX), count: copyCount)
                                }
                                
                                if maxSafeX < actW {
                                    let rightEdgeVal = pData[srcRow + pW - 1]
                                    for x in maxSafeX..<actW {
                                        pDstBase[x] = rightEdgeVal
                                    }
                                }
                            }
                        } else {
                            // Extract padded block for fractional interpolation near edge
                            let extW = actW + 7
                            let extH = actH + 7
                            let extStartX = refX - 3
                            let extStartY = refY - 3
                            
                            extBuffer.withUnsafeMutableBufferPointer { extPtr in
                                guard let pExt = extPtr.baseAddress else { return }
                                
                                let minSafeX = max(0, min(extW, -extStartX))
                                let maxSafeX = max(0, min(extW, pW - extStartX))
                                
                                for y in 0..<extH {
                                    let srcY = max(0, min(pH - 1, extStartY + y))
                                    let srcRow = srcY * pW
                                    let dstRow = y * extW
                                    let pDstBase = pExt.advanced(by: dstRow)
                                    
                                    if 0 < minSafeX {
                                        let leftEdgeVal = pData[srcRow]
                                        for x in 0..<minSafeX {
                                            pDstBase[x] = leftEdgeVal
                                        }
                                    }
                                    
                                    let copyCount = maxSafeX - minSafeX
                                    if 0 < copyCount {
                                        pDstBase.advanced(by: minSafeX).update(from: pData.advanced(by: srcRow + extStartX + minSafeX), count: copyCount)
                                    }
                                    
                                    if maxSafeX < extW {
                                        let rightEdgeVal = pData[srcRow + pW - 1]
                                        for x in maxSafeX..<extW {
                                            pDstBase[x] = rightEdgeVal
                                        }
                                    }
                                }
                                
                                // Now interpolate from extBuffer to pOut
                                subpixelInterpolateBlock(
                                    src: pExt, srcStride: extW,
                                    dst: pOut.advanced(by: startY * pW + startX), dstStride: pW,
                                    width: actW, height: actH,
                                    fracX: fracX, fracY: fracY,
                                    startX: 3, startY: 3
                                )
                            }
                        }
                    }
                }

                for mbY in 0..<fullMbRows {
                    for mbX in 0..<fullMbCols {
                        applyBlock(mbX: mbX, mbY: mbY, actW: pMbSize, actH: pMbSize)
                    }
                }

                if 0 < remW {
                    let mbX = fullMbCols
                    for mbY in 0..<fullMbRows {
                        applyBlock(mbX: mbX, mbY: mbY, actW: remW, actH: pMbSize)
                    }
                }

                if 0 < remH {
                    let mbY = fullMbRows
                    for mbX in 0..<fullMbCols {
                        applyBlock(mbX: mbX, mbY: mbY, actW: pMbSize, actH: remH)
                    }
                }

                if 0 < remW && 0 < remH {
                    let mbX = fullMbCols
                    let mbY = fullMbRows
                    applyBlock(mbX: mbX, mbY: mbY, actW: remW, actH: remH)
                }
            }
        }
        return out
    }

    async let yTask = apply(data: prev.y, pW: w, pH: h, div: 1)
    async let cbTask = apply(data: prev.cb, pW: (w + 1) / 2, pH: (h + 1) / 2, div: 2)
    async let crTask = apply(data: prev.cr, pW: (w + 1) / 2, pH: (h + 1) / 2, div: 2)

    return PlaneData420(width: w, height: h, y: await yTask, cb: await cbTask, cr: await crTask)
}

@inline(__always)
func calculatePMV(mvs: MotionVectors, mbX: Int, mbY: Int, mbCols: Int) -> (dx: Int, dy: Int) {
    let hasLeft = mbX > 0
    let hasTop = mbY > 0
    let hasTopRight = mbY > 0 && mbX < mbCols - 1

    let idxLeft = hasLeft ? (mbY * mbCols + (mbX - 1)) : -1
    let idxTop = hasTop ? ((mbY - 1) * mbCols + mbX) : -1
    let idxTopRight = hasTopRight ? ((mbY - 1) * mbCols + (mbX + 1)) : -1

    var count = 0
    if hasLeft { count += 1 }
    if hasTop { count += 1 }
    if hasTopRight { count += 1 }

    if count == 0 {
        return (0, 0)
    }
    if count == 1 {
        let vec = mvs.vectors[hasLeft ? idxLeft : (hasTop ? idxTop : idxTopRight)]
        return (Int(vec.x), Int(vec.y))
    }
    if count == 2 {
        var dxSum = 0
        var dySum = 0
        if hasLeft { let v = mvs.vectors[idxLeft]; dxSum += Int(v.x); dySum += Int(v.y) }
        if hasTop { let v = mvs.vectors[idxTop]; dxSum += Int(v.x); dySum += Int(v.y) }
        if hasTopRight { let v = mvs.vectors[idxTopRight]; dxSum += Int(v.x); dySum += Int(v.y) }
        return (dxSum / 2, dySum / 2)
    }
    
    let lVec = mvs.vectors[idxLeft]; let lx = Int(lVec.x); let ly = Int(lVec.y)
    let tVec = mvs.vectors[idxTop]; let tx = Int(tVec.x); let ty = Int(tVec.y)
    let rVec = mvs.vectors[idxTopRight]; let rx = Int(rVec.x); let ry = Int(rVec.y)

    let minX = min(lx, min(tx, rx))
    let maxX = max(lx, max(tx, rx))
    let pmvX = lx + tx + rx - minX - maxX

    let minY = min(ly, min(ty, ry))
    let maxY = max(ly, max(ty, ry))
    let pmvY = ly + ty + ry - minY - maxY

    return (pmvX, pmvY)
}

import Foundation

@inline(__always)
func calculateSAD16x16(pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, currStride: Int, prevStride: Int) -> Int {
    var sumVec0 = SIMD16<UInt16>()
    for y in 0..<16 {
        let currRow = pCurr.advanced(by: y * currStride)
        let prevRow = pPrev.advanced(by: y * prevStride)

        let c0 = UnsafeRawPointer(currRow).loadUnaligned(as: SIMD16<Int16>.self)
        let p0 = UnsafeRawPointer(prevRow).loadUnaligned(as: SIMD16<Int16>.self)

        let diff0 = c0 &- p0
        let mask0 = diff0 &>> 15
        let abs0 = (diff0 ^ mask0) &- mask0

        sumVec0 &+= SIMD16<UInt16>(truncatingIfNeeded: abs0)
    }
    return Int(SIMD16<UInt32>(truncatingIfNeeded: sumVec0).wrappedSum())
}

@inline(__always)
func calculateSAD8x8(pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, currStride: Int, prevStride: Int) -> Int {
    var sumVec0 = SIMD8<UInt16>()
    for y in 0..<8 {
        let currRow = pCurr.advanced(by: y * currStride)
        let prevRow = pPrev.advanced(by: y * prevStride)

        let cv = UnsafeRawPointer(currRow).loadUnaligned(as: SIMD8<Int16>.self)
        let pv = UnsafeRawPointer(prevRow).loadUnaligned(as: SIMD8<Int16>.self)
        
        let diff0 = cv &- pv
        let mask0 = diff0 &>> 15
        let abs0 = (diff0 ^ mask0) &- mask0
        sumVec0 &+= SIMD8<UInt16>(truncatingIfNeeded: abs0)
    }
    return Int(SIMD8<UInt32>(truncatingIfNeeded: sumVec0).wrappedSum())
}

@inline(__always)
func evaluateMotionQuadtreeNode(
    pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, w: Int, h: Int,
    startX: Int, startY: Int, size: Int, searchRange: Int,
    coarseDX: Int, coarseDY: Int,
    grid: inout MVGrid,
    fracRefBuf: UnsafeMutablePointer<Int16>,
    fracExtBuf: UnsafeMutablePointer<Int16>
) -> (node: MotionNode, sad: Int) {
    if startX >= w || startY >= h {
        return (.leaf(mv: .zero), 0)
    }

    let actW = min(size, w - startX)
    let actH = min(size, h - startY)

    let minX = -startX
    let minY = -startY
    let maxX = w - startX - actW
    let maxY = h - startY - actH

    let pmv = grid.getPMV(x: startX, y: startY, w: size)
    
    // Layer0 Coarse MV serves as the search center at the root (64x64) level
    var centerDX = Int(pmv.x)
    var centerDY = Int(pmv.y)
    if size == 64 {
        centerDX = coarseDX
        centerDY = coarseDY
    }
    
    // Extremely tight fine search around the center
    // E.g., for ±3 range, we only check 49 points.
    var bestDX = Int(pmv.x) >> 2
    var bestDY = Int(pmv.y) >> 2
    
    bestDX = max(minX, min(maxX, bestDX))
    bestDY = max(minY, min(maxY, bestDY))

    var bestSAD: Int
    if size == 64 && actW == 64 && actH == 64 {
        bestSAD = calculateSAD64x64(
            pCurr: pCurr.advanced(by: startY * w + startX),
            pPrev: pPrev.advanced(by: (startY + bestDY) * w + startX + bestDX),
            currStride: w, prevStride: w
        )
        let zeroSAD = calculateSAD64x64(
            pCurr: pCurr.advanced(by: startY * w + startX),
            pPrev: pPrev.advanced(by: startY * w + startX),
            currStride: w, prevStride: w
        )
        if zeroSAD < bestSAD { bestSAD = zeroSAD; bestDX = 0; bestDY = 0 }
    } else if size == 32 && actW == 32 && actH == 32 {
        bestSAD = calculateSAD32x32(
            pCurr: pCurr.advanced(by: startY * w + startX),
            pPrev: pPrev.advanced(by: (startY + bestDY) * w + startX + bestDX),
            currStride: w, prevStride: w
        )
        let zeroSAD = calculateSAD32x32(
            pCurr: pCurr.advanced(by: startY * w + startX),
            pPrev: pPrev.advanced(by: startY * w + startX),
            currStride: w, prevStride: w
        )
        if zeroSAD < bestSAD { bestSAD = zeroSAD; bestDX = 0; bestDY = 0 }
    } else if size == 16 && actW == 16 && actH == 16 {
        bestSAD = calculateSAD16x16(
            pCurr: pCurr.advanced(by: startY * w + startX),
            pPrev: pPrev.advanced(by: (startY + bestDY) * w + startX + bestDX),
            currStride: w, prevStride: w
        )
        let zeroSAD = calculateSAD16x16(
            pCurr: pCurr.advanced(by: startY * w + startX),
            pPrev: pPrev.advanced(by: startY * w + startX),
            currStride: w, prevStride: w
        )
        if zeroSAD < bestSAD { bestSAD = zeroSAD; bestDX = 0; bestDY = 0 }
    } else if size == 8 && actW == 8 && actH == 8 {
        bestSAD = calculateSAD8x8(
            pCurr: pCurr.advanced(by: startY * w + startX),
            pPrev: pPrev.advanced(by: (startY + bestDY) * w + startX + bestDX),
            currStride: w, prevStride: w
        )
        let zeroSAD = calculateSAD8x8(
            pCurr: pCurr.advanced(by: startY * w + startX),
            pPrev: pPrev.advanced(by: startY * w + startX),
            currStride: w, prevStride: w
        )
        if zeroSAD < bestSAD { bestSAD = zeroSAD; bestDX = 0; bestDY = 0 }
    } else {
        bestSAD = calculateSADEdge(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, actW: actW, actH: actH, dx: centerDX, dy: centerDY)
        let zeroSAD = calculateSADEdge(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, actW: actW, actH: actH, dx: 0, dy: 0)
        
        // Zero-bias (0,0 is generally preferred if close)
        if zeroSAD <= bestSAD + (actW * actH / 16) {
            bestDX = 0
            bestDY = 0
            bestSAD = zeroSAD
        } else {
            bestDX = centerDX
            bestDY = centerDY
        }
    }
    // Fast Mode: If PMV or ZeroMV already yields an excellent match.
    // Given that typical quantization noise adds SAD=3~5 per pixel, we set the threshold to SAD=8 per pixel.
    // If the error is under this, it's mostly quantization noise and the MV is correct.
    let earlyExitThreshold = size * size * 8
    if earlyExitThreshold < bestSAD {
        var step = max(1, searchRange / 2)
        while 1 <= step {
            var currentBestDX = bestDX
            var currentBestDY = bestDY
            var currentBestSAD = bestSAD

            if step == 1 && size == 64 && actW == 64 && actH == 64 {
                currentBestSAD = calculateSAD64x64(
                    pCurr: pCurr.advanced(by: startY * w + startX),
                    pPrev: pPrev.advanced(by: (startY + bestDY) * w + startX + bestDX),
                    currStride: w, prevStride: w
                )
            }

            for j in -1...1 {
                for i in -1...1 {
                    if i == 0 && j == 0 { continue }
                    let dx = bestDX + i * step
                    let dy = bestDY + j * step
                    
                    let maxAbsoluteMV = 64
                    if dx < -maxAbsoluteMV || maxAbsoluteMV < dx || dy < -maxAbsoluteMV || maxAbsoluteMV < dy { continue }
                    if dx < minX || maxX < dx || dy < minY || maxY < dy { continue }
                    
                    let penalty = abs(dx) + abs(dy)
                    if currentBestSAD <= penalty { continue }
                    
                    let sad: Int
                    if size == 64 && actW == 64 && actH == 64 {
                        sad = step == 1 ? calculateSAD64x64(
                            pCurr: pCurr.advanced(by: startY * w + startX),
                            pPrev: pPrev.advanced(by: (startY + dy) * w + startX + dx),
                            currStride: w, prevStride: w
                        ) : calculateSAD64x64_Subsample(
                            pCurr: pCurr.advanced(by: startY * w + startX),
                            pPrev: pPrev.advanced(by: (startY + dy) * w + startX + dx),
                            currStride: w, prevStride: w
                        )
                    } else if size == 32 && actW == 32 && actH == 32 {
                        sad = calculateSAD32x32(
                            pCurr: pCurr.advanced(by: startY * w + startX),
                            pPrev: pPrev.advanced(by: (startY + dy) * w + startX + dx),
                            currStride: w, prevStride: w
                        )
                    } else if size == 16 && actW == 16 && actH == 16 {
                        sad = calculateSAD16x16(
                            pCurr: pCurr.advanced(by: startY * w + startX),
                            pPrev: pPrev.advanced(by: (startY + dy) * w + startX + dx),
                            currStride: w, prevStride: w
                        )
                    } else if size == 8 && actW == 8 && actH == 8 {
                        sad = calculateSAD8x8(
                            pCurr: pCurr.advanced(by: startY * w + startX),
                            pPrev: pPrev.advanced(by: (startY + dy) * w + startX + dx),
                            currStride: w, prevStride: w
                        )
                    } else {
                        sad = calculateSADEdge(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, actW: actW, actH: actH, dx: dx, dy: dy)
                    }
                    
                    let totalSAD = sad &+ penalty
                    if totalSAD < currentBestSAD {
                        currentBestSAD = totalSAD
                        currentBestDX = dx
                        currentBestDY = dy
                    }
                }
            }
            bestDX = currentBestDX
            bestDY = currentBestDY
            bestSAD = currentBestSAD
            step /= 2
        }
    }

    // Fractional ME should ONLY run if we decide NOT to split (to save massive CPU time)
    let penaltyValues = [64: 0, 32: 200, 16: 400, 8: 600] // RDO penalty based on size
    let baseLeafSAD = bestSAD + (penaltyValues[size] ?? 0)

    let minSize = 8
    let splitEarlyExitThreshold = actW * actH * 8 // SAD=8 per pixel (tolerates quantization noise)
    var performSplit = false
    var splitNode: MotionNode? = nil
    var splitSAD = Int.max

    // Evaluate Split
    if size > minSize && actW == size && actH == size && bestSAD > splitEarlyExitThreshold {
        let half = size / 2
        // Save grid state in case we don't pick split
        let savedGrid = grid
        
        // Provide integer PMV for children during their search to keep it accurate
        let intPMV = SIMD2<Int16>(Int16(bestDX * 4), Int16(bestDY * 4))
        grid.fill(x: startX, y: startY, w: size, h: size, mv: intPMV)
        
        // Hierarchical ME (HME): Reduce the search range exponentially for child nodes.
        // Parent already found the bulk motion vector. Children only need to slightly refine it.
        let childSearchRange = max(1, searchRange / 2) // Drastically shrink for children
        let tl = evaluateMotionQuadtreeNode(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, size: half, searchRange: childSearchRange, coarseDX: 0, coarseDY: 0, grid: &grid, fracRefBuf: fracRefBuf, fracExtBuf: fracExtBuf)
        let tr = evaluateMotionQuadtreeNode(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX + half, startY: startY, size: half, searchRange: childSearchRange, coarseDX: 0, coarseDY: 0, grid: &grid, fracRefBuf: fracRefBuf, fracExtBuf: fracExtBuf)
        let bl = evaluateMotionQuadtreeNode(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY + half, size: half, searchRange: childSearchRange, coarseDX: 0, coarseDY: 0, grid: &grid, fracRefBuf: fracRefBuf, fracExtBuf: fracExtBuf)
        let br = evaluateMotionQuadtreeNode(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX + half, startY: startY + half, size: half, searchRange: childSearchRange, coarseDX: 0, coarseDY: 0, grid: &grid, fracRefBuf: fracRefBuf, fracExtBuf: fracExtBuf)
        // Penalty for sending 4 MVs instead of 1 MV.
        // It must scale with area to prevent overfitting to quantization noise!
        let splitPenalty = (actW * actH) * 1 // Requires at least an average of SAD=1 reduction per pixel to justify a split
        splitSAD = tl.sad + tr.sad + bl.sad + br.sad + splitPenalty
        
        if splitSAD < baseLeafSAD {
            // Split preferred
            performSplit = true
            splitNode = .split(tl: tl.node, tr: tr.node, bl: bl.node, br: br.node)
        } else {
            // No Split preferred: Revert grid
            grid = savedGrid
        }
    }

    if performSplit {
        return (splitNode!, splitSAD)
    } else {
        // Run heavy subpixel refinement ONLY on the finalized leaf node!
        let refinedQMV = refineFractionalMBME(
            pCurr: pCurr, pPrev: pPrev, w: w, h: h,
            startX: startX, startY: startY, actW: actW, actH: actH,
            bestIntDX: bestDX, bestIntDY: bestDY, bestIntSAD: bestSAD,
            fracRefBuffer: fracRefBuf, fracExtBuffer: fracExtBuf
        )
        grid.fill(x: startX, y: startY, w: actW, h: actH, mv: refinedQMV)
        return (.leaf(mv: refinedQMV), baseLeafSAD)
    }
}

@inline(__always)
func evaluateLayer0Motion(
    pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, w: Int, h: Int,
    startX: Int, startY: Int, size: Int, searchRange: Int
) -> (dx: Int, dy: Int) {
    if startX >= w || startY >= h { return (0, 0) }
    
    let actW = min(size, w - startX)
    let actH = min(size, h - startY)
    
    var bestSAD = Int.max
    var bestDX = 0
    var bestDY = 0

    // Evaluate Center (from Coarse MV or PMV depending on size)
    let zeroSAD = calculateSADEdge(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, actW: actW, actH: actH, dx: 0, dy: 0)
    bestSAD = zeroSAD
    
    for dy in -searchRange...searchRange {
        for dx in -searchRange...searchRange {
            if dx == 0 && dy == 0 { continue }
            
            let sad = calculateSADEdge(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, actW: actW, actH: actH, dx: dx, dy: dy)
            let penalty = (abs(dx) + abs(dy)) * 2
            
            if sad + penalty < bestSAD {
                bestSAD = sad + penalty
                bestDX = dx
                bestDY = dy
            }
        }
    }
    
    return (bestDX, bestDY)
}

func estimateMotionQuadtree(curr: PlaneData420, prev: PlaneData420, layer0Curr: (data: [Int16], w: Int, h: Int), layer0Prev: (data: [Int16], w: Int, h: Int)) -> MotionTree {
    let mbSize = 64
    let w = curr.width
    let h = curr.height
    let mbCols = (w + mbSize - 1) / mbSize
    let mbRows = (h + mbSize - 1) / mbSize

    var ctuNodes = [MotionNode]()
    var grid = MVGrid(width: w, height: h, minSize: 8)

    var fracRefBuffer = [Int16](repeating: 0, count: 64 * 64)
    var fracExtBuffer = [Int16](repeating: 0, count: 71 * 71)

    curr.y.withUnsafeBufferPointer { currPtr in
        guard let pCurr = currPtr.baseAddress else { return }
        prev.y.withUnsafeBufferPointer { prevPtr in
            guard let pPrev = prevPtr.baseAddress else { return }
            fracRefBuffer.withUnsafeMutableBufferPointer { fracRefPtr in
                guard let pFracRef = fracRefPtr.baseAddress else { return }
                fracExtBuffer.withUnsafeMutableBufferPointer { fracExtPtr in
                    guard let pFracExt = fracExtPtr.baseAddress else { return }

            layer0Curr.data.withUnsafeBufferPointer { layer0CPtr in
                guard let pLayer0C = layer0CPtr.baseAddress else { return }
                layer0Prev.data.withUnsafeBufferPointer { layer0PPtr in
                    guard let pLayer0P = layer0PPtr.baseAddress else { return }
                    
                    for mbY in 0..<mbRows {
                        let startY = mbY * mbSize
                        for mbX in 0..<mbCols {
                            let startX = mbX * mbSize
                            
                            // 1. DWT/Layer0 Coarse ME: Search in the 1/8 downscaled plane
                            // The 64x64 block maps to an 8x8 block in Layer0. Search range ±3 means ±24 in full res!
                            let cDXDY = evaluateLayer0Motion(
                                pCurr: pLayer0C, pPrev: pLayer0P, w: layer0Curr.w, h: layer0Curr.h,
                                startX: mbX * 8, startY: mbY * 8, size: 8, searchRange: 3
                            )
                            
                            // 2. Full-res Fine ME: Refine around the scaled Coarse MV
                            // Fine search range of ±3 is extremely light (49 evaluations) instead of ±16 (1089 evaluations)
                            let nodeEval = evaluateMotionQuadtreeNode(
                                pCurr: pCurr, pPrev: pPrev, w: w, h: h,
                                startX: startX, startY: startY, size: mbSize, searchRange: 3,
                                coarseDX: cDXDY.dx * 8, coarseDY: cDXDY.dy * 8,
                                grid: &grid, fracRefBuf: pFracRef, fracExtBuf: pFracExt
                            )
                            ctuNodes.append(nodeEval.node)
                        }
                    }
                }
            }
                }
            }
        }
    }
    return MotionTree(ctuNodes: ctuNodes, width: w, height: h)
}

func applyMotionQuadtreeNode(
    node: MotionNode,
    pDstY: UnsafeMutablePointer<Int16>,
    pDstCb: UnsafeMutablePointer<Int16>,
    pDstCr: UnsafeMutablePointer<Int16>,
    pPrevY: UnsafePointer<Int16>,
    pPrevCb: UnsafePointer<Int16>,
    pPrevCr: UnsafePointer<Int16>,
    w: Int, h: Int,
    startX: Int, startY: Int, size: Int,
    extBuffer: UnsafeMutablePointer<Int16>
) {
    if startX >= w || startY >= h { return }
    let actW = min(size, w - startX)
    let actH = min(size, h - startY)

    switch node {
    case .split(let tl, let tr, let bl, let br):
        let half = size / 2
        applyMotionQuadtreeNode(node: tl, pDstY: pDstY, pDstCb: pDstCb, pDstCr: pDstCr, pPrevY: pPrevY, pPrevCb: pPrevCb, pPrevCr: pPrevCr, w: w, h: h, startX: startX, startY: startY, size: half, extBuffer: extBuffer)
        applyMotionQuadtreeNode(node: tr, pDstY: pDstY, pDstCb: pDstCb, pDstCr: pDstCr, pPrevY: pPrevY, pPrevCb: pPrevCb, pPrevCr: pPrevCr, w: w, h: h, startX: startX + half, startY: startY, size: half, extBuffer: extBuffer)
        applyMotionQuadtreeNode(node: bl, pDstY: pDstY, pDstCb: pDstCb, pDstCr: pDstCr, pPrevY: pPrevY, pPrevCb: pPrevCb, pPrevCr: pPrevCr, w: w, h: h, startX: startX, startY: startY + half, size: half, extBuffer: extBuffer)
        applyMotionQuadtreeNode(node: br, pDstY: pDstY, pDstCb: pDstCb, pDstCr: pDstCr, pPrevY: pPrevY, pPrevCb: pPrevCb, pPrevCr: pPrevCr, w: w, h: h, startX: startX + half, startY: startY + half, size: half, extBuffer: extBuffer)
    case .leaf(let mv):
        let dx = Int(mv.x)
        let dy = Int(mv.y)
        let intDX = dx >> 2
        let intDY = dy >> 2
        let fracX = dx & 3
        let fracY = dy & 3

        let refX = startX + intDX
        let refY = startY + intDY

        let isYSafe = (refX - 3 >= 0) && (refY - 3 >= 0) && (refX + actW + 4 <= w) && (refY + actH + 4 <= h)

        let dstPtrY = pDstY.advanced(by: startY * w + startX)

        if isYSafe {
            subpixelInterpolateBlock(
                src: pPrevY, srcStride: w, dst: dstPtrY, dstStride: w,
                width: actW, height: actH, fracX: fracX, fracY: fracY, startX: refX, startY: refY
            )
        } else {
            let extW = actW + 7
            let extH = actH + 7
            let extStartX = refX - 3
            let extStartY = refY - 3
            let minSafeX = max(0, min(extW, -extStartX))
            let maxSafeX = max(0, min(extW, w - extStartX))
            
            for y in 0..<extH {
                let srcY = max(0, min(h - 1, extStartY + y))
                let srcRow = srcY * w
                let dstRow = y * extW
                let pDstBase = extBuffer.advanced(by: dstRow)

                if 0 < minSafeX {
                    let leftEdgeVal = pPrevY[srcRow]
                    for x in 0..<minSafeX { pDstBase[x] = leftEdgeVal }
                }

                let copyCount = maxSafeX - minSafeX
                if 0 < copyCount {
                    pDstBase.advanced(by: minSafeX).update(from: pPrevY.advanced(by: srcRow + extStartX + minSafeX), count: copyCount)
                }

                if maxSafeX < extW {
                    let rightEdgeVal = pPrevY[srcRow + w - 1]
                    for x in maxSafeX..<extW { pDstBase[x] = rightEdgeVal }
                }
            }
            
            subpixelInterpolateBlock(
                src: extBuffer, srcStride: extW, dst: dstPtrY, dstStride: w,
                width: actW, height: actH, fracX: fracX, fracY: fracY, startX: 3, startY: 3
            )
        }

        // Chroma
        let cw = w / 2
        let ch = h / 2
        let actCW = actW / 2
        let actCH = actH / 2
        let startCX = startX / 2
        let startCY = startY / 2

        let chromaDX = dx >> 1 // Since luma is 1/4 pel, chroma resolution makes it 1/8 pel, but we map to 1/4 pel filter precision
        let chromaDY = dy >> 1
        let cIntDX = chromaDX >> 2
        let cIntDY = chromaDY >> 2
        let cFracX = chromaDX & 3
        let cFracY = chromaDY & 3

        let crefX = startCX + cIntDX
        let crefY = startCY + cIntDY

        let isCSafe = (crefX - 3 >= 0) && (crefY - 3 >= 0) && (crefX + actCW + 4 <= cw) && (crefY + actCH + 4 <= ch)

        let dstPtrCb = pDstCb.advanced(by: startCY * cw + startCX)
        let dstPtrCr = pDstCr.advanced(by: startCY * cw + startCX)

        if isCSafe {
            subpixelInterpolateBlock(
                src: pPrevCb, srcStride: cw, dst: dstPtrCb, dstStride: cw,
                width: actCW, height: actCH, fracX: cFracX, fracY: cFracY, startX: crefX, startY: crefY
            )
            subpixelInterpolateBlock(
                src: pPrevCr, srcStride: cw, dst: dstPtrCr, dstStride: cw,
                width: actCW, height: actCH, fracX: cFracX, fracY: cFracY, startX: crefX, startY: crefY
            )
        } else {
            let extCW = actCW + 7
            let extCH = actCH + 7
            let cExtStartX = crefX - 3
            let cExtStartY = crefY - 3
            let cMinSafeX = max(0, min(extCW, -cExtStartX))
            let cMaxSafeX = max(0, min(extCW, cw - cExtStartX))

            func copyChromaEdge(pPrevC: UnsafePointer<Int16>, dstC: UnsafeMutablePointer<Int16>) {
                for y in 0..<extCH {
                    let srcY = max(0, min(ch - 1, cExtStartY + y))
                    let srcRow = srcY * cw
                    let dstRow = y * extCW
                    let pDstBase = extBuffer.advanced(by: dstRow)

                    if 0 < cMinSafeX {
                        let leftEdgeVal = pPrevC[srcRow]
                        for x in 0..<cMinSafeX { pDstBase[x] = leftEdgeVal }
                    }
                    let copyCount = cMaxSafeX - cMinSafeX
                    if 0 < copyCount {
                        pDstBase.advanced(by: cMinSafeX).update(from: pPrevC.advanced(by: srcRow + cExtStartX + cMinSafeX), count: copyCount)
                    }
                    if cMaxSafeX < extCW {
                        let rightEdgeVal = pPrevC[srcRow + cw - 1]
                        for x in cMaxSafeX..<extCW { pDstBase[x] = rightEdgeVal }
                    }
                }
                subpixelInterpolateBlock(
                    src: extBuffer, srcStride: extCW, dst: dstC, dstStride: cw,
                    width: actCW, height: actCH, fracX: cFracX, fracY: cFracY, startX: 3, startY: 3
                )
            }
            
            copyChromaEdge(pPrevC: pPrevCb, dstC: dstPtrCb)
            copyChromaEdge(pPrevC: pPrevCr, dstC: dstPtrCr)
        }
    }
}

func applyMotionQuadtree(prev: PlaneData420, tree: MotionTree) async -> PlaneData420 {
    let w = prev.width
    let h = prev.height
    var result = PlaneData420(
        width: w, height: h,
        y: [Int16](repeating: 0, count: w * h),
        cb: [Int16](repeating: 0, count: (w * h) / 4),
        cr: [Int16](repeating: 0, count: (w * h) / 4)
    )

    let mbSize = 64
    let mbCols = (w + mbSize - 1) / mbSize
    let mbRows = (h + mbSize - 1) / mbSize

    result.y.withUnsafeMutableBufferPointer { resYPtr in
        guard let pResY = resYPtr.baseAddress else { return }
        result.cb.withUnsafeMutableBufferPointer { resCbPtr in
            guard let pResCb = resCbPtr.baseAddress else { return }
            result.cr.withUnsafeMutableBufferPointer { resCrPtr in
                guard let pResCr = resCrPtr.baseAddress else { return }
                prev.y.withUnsafeBufferPointer { prevYPtr in
                    guard let pPrevY = prevYPtr.baseAddress else { return }
                    prev.cb.withUnsafeBufferPointer { prevCbPtr in
                        guard let pPrevCb = prevCbPtr.baseAddress else { return }
                        prev.cr.withUnsafeBufferPointer { prevCrPtr in
                            guard let pPrevCr = prevCrPtr.baseAddress else { return }
                            
                            var extBuffer = [Int16](repeating: 0, count: 71 * 71)
                            extBuffer.withUnsafeMutableBufferPointer { extPtr in
                                guard let pExt = extPtr.baseAddress else { return }

                                for mbY in 0..<mbRows {
                                    let startY = mbY * mbSize
                                    for mbX in 0..<mbCols {
                                        let startX = mbX * mbSize
                                        let nodeIndex = mbY * mbCols + mbX
                                        if nodeIndex < tree.ctuNodes.count {
                                            let node = tree.ctuNodes[nodeIndex]
                                            applyMotionQuadtreeNode(
                                                node: node,
                                                pDstY: pResY, pDstCb: pResCb, pDstCr: pResCr,
                                                pPrevY: pPrevY, pPrevCb: pPrevCb, pPrevCr: pPrevCr,
                                                w: w, h: h,
                                                startX: startX, startY: startY, size: mbSize,
                                                extBuffer: pExt
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return result
}

func encodeMotionQuadtreeNode(
    node: MotionNode,
    w: Int, h: Int,
    startX: Int, startY: Int, size: Int,
    grid: inout MVGrid,
    bw: inout EntropyEncoder
) {
    if startX >= w || startY >= h { return }
    let minSize = 8

    switch node {
    case .split(let tl, let tr, let bl, let br):
        if size > minSize {
            bw.encodeBypass(binVal: 1)
        }
        let half = size / 2
        encodeMotionQuadtreeNode(node: tl, w: w, h: h, startX: startX, startY: startY, size: half, grid: &grid, bw: &bw)
        encodeMotionQuadtreeNode(node: tr, w: w, h: h, startX: startX + half, startY: startY, size: half, grid: &grid, bw: &bw)
        encodeMotionQuadtreeNode(node: bl, w: w, h: h, startX: startX, startY: startY + half, size: half, grid: &grid, bw: &bw)
        encodeMotionQuadtreeNode(node: br, w: w, h: h, startX: startX + half, startY: startY + half, size: half, grid: &grid, bw: &bw)
        
    case .leaf(let mv):
        if size > minSize {
            bw.encodeBypass(binVal: 0)
        }
        let actW = min(size, w - startX)
        let actH = min(size, h - startY)

        let pmv = grid.getPMV(x: startX, y: startY, w: actW)
        grid.fill(x: startX, y: startY, w: actW, h: actH, mv: mv)

        let mvdX = Int(mv.x) - Int(pmv.x)
        let mvdY = Int(mv.y) - Int(pmv.y)

        if mvdX == 0 && mvdY == 0 {
            bw.encodeBypass(binVal: 0)
        } else {
            bw.encodeBypass(binVal: 1)
            let sx: UInt8 = mvdX <= -1 ? 1 : 0
            bw.encodeBypass(binVal: sx)
            let mx = UInt32(abs(mvdX))
            encodeExpGolomb(val: mx, encoder: &bw)

            let sy: UInt8 = mvdY <= -1 ? 1 : 0
            bw.encodeBypass(binVal: sy)
            let my = UInt32(abs(mvdY))
            encodeExpGolomb(val: my, encoder: &bw)
        }
    }
}

func decodeMotionQuadtreeNode(
    w: Int, h: Int,
    startX: Int, startY: Int, size: Int,
    grid: inout MVGrid,
    br: inout EntropyDecoder
) throws -> MotionNode {
    if startX >= w || startY >= h { return .leaf(mv: .zero) }
    let minSize = 8

    var isSplit = false
    if size > minSize {
        isSplit = try br.decodeBypass() == 1
    }

    if isSplit {
        let half = size / 2
        let tl = try decodeMotionQuadtreeNode(w: w, h: h, startX: startX, startY: startY, size: half, grid: &grid, br: &br)
        let tr = try decodeMotionQuadtreeNode(w: w, h: h, startX: startX + half, startY: startY, size: half, grid: &grid, br: &br)
        let bl = try decodeMotionQuadtreeNode(w: w, h: h, startX: startX, startY: startY + half, size: half, grid: &grid, br: &br)
        let brNode = try decodeMotionQuadtreeNode(w: w, h: h, startX: startX + half, startY: startY + half, size: half, grid: &grid, br: &br)
        return .split(tl: tl, tr: tr, bl: bl, br: brNode)
    } else {
        let actW = min(size, w - startX)
        let actH = min(size, h - startY)

        let pmv = grid.getPMV(x: startX, y: startY, w: actW)

        let hasMVD = try br.decodeBypass() == 1
        var mvdX: Int = 0
        var mvdY: Int = 0

        if hasMVD {
            let sx = try br.decodeBypass() == 1
            let mx = Int(try decodeExpGolomb(decoder: &br))
            mvdX = sx ? -mx : mx

            let sy = try br.decodeBypass() == 1
            let my = Int(try decodeExpGolomb(decoder: &br))
            mvdY = sy ? -my : my
        }

        let mv = SIMD2<Int16>(Int16(clamping: mvdX + Int(pmv.x)), Int16(clamping: mvdY + Int(pmv.y)))
        grid.fill(x: startX, y: startY, w: actW, h: actH, mv: mv)
        
        return .leaf(mv: mv)
    }
}
