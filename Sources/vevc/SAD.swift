/// CopyFrame near-duplicate bound: per-pixel mean abs diff (Y plus chroma at
/// half weight) in Q8 — 96 = 0.375/px. Calibrated on miko_700 consecutive
/// source frames: 14.7% of pairs fall under it and the worst affected pair
/// SSIM is 0.9947 (the live-streaming quality target tolerates well below
/// that; the floor is protected because fades and motion exceed the bound).
let copyFrameMADLimitQ8: Int = 32

/// Copy-chain cap: sub-threshold motion (e.g. 1px HUD ticks) never breaks
/// the MAD bound, so a hard cap bounds how long it can stay frozen
/// (5 frames ≈ 83ms at 60fps).
let maxConsecutiveCopyFrames: Int = 5

/// Integer square root (floor).
/// Returns the largest integer n such that n*n <= value.
@inline(__always)
func isqrt(_ value: Int) -> Int {
    guard 0 < value else { return 0 }
    var x = value
    var y = (x + 1) / 2
    while y < x {
        x = y
        y = (x + (value / x)) / 2
    }
    return x
}

/// sqrt(2) in 1024-scale fixed-point: 1.41421356... * 1024 ≈ 1448
private let kSqrt2Scaled: Int = 1448

/// Compute a spatial weight for a block at (blockCol, blockRow) in a grid of (colCount x rowCount).
/// Returns 1024 at the center of the image and increases toward edges/corners (1024-scale fixed-point).
/// 1024 corresponds to weight 1.0.
/// Used to apply more aggressive compression on peripheral blocks where
/// human visual attention is naturally lower.
///
/// - Parameters:
///   - blockCol, blockRow: Block position (0-indexed).
///   - colCount, rowCount: Total grid dimensions.
///   - edgeScale: Maximum weight at corners in 1024-scale (default 1536 = 1.5x).
/// - Returns: Weight in [1024, edgeScale] (1024-scale fixed-point).
@inline(__always)
func spatialWeight(blockCol: Int, blockRow: Int, colCount: Int, rowCount: Int, edgeScale: Int = 1536) -> Int {
    guard 1 < colCount && 1 < rowCount else { return 1024 }
    
    // Normalize block position to [-1024, 1024] centered coordinates (1024-scale)
    let cx = ((blockCol * 2048) / (colCount - 1)) - 1024
    let cy = ((blockRow * 2048) / (rowCount - 1)) - 1024
    
    // Euclidean distance from center in 1024-scale, normalized by sqrt(2)
    // dist = sqrt(cx*cx + cy*cy) / sqrt(2), all in 1024-scale
    let distSquared = ((cx * cx) + (cy * cy))
    let dist1024 = isqrt(distSquared)
    // Divide by SQRT2_SCALED and clamp to [0, 1024]
    let distNorm = min(1024, (dist1024 * 1024) / kSqrt2Scaled)
    
    // Linear interpolation: center → 1024, corner → edgeScale
    return 1024 + (((edgeScale - 1024) * distNorm) / 1024)
}

/// Compute spatially-adaptive SAD threshold for zero-block skip decisions.
/// Edge blocks get higher thresholds → more likely to be fully skipped.
@inline(__always)
func spatialSADThreshold(baseSAD: Int, blockCol: Int, blockRow: Int, colCount: Int, rowCount: Int) -> Int {
    let weight = spatialWeight(blockCol: blockCol, blockRow: blockRow, colCount: colCount, rowCount: rowCount)
    return (baseSAD * weight) / 1024
}

@inline(__always)
func scaledSADThreshold(_ defaultSAD: Int, step: Int) -> Int {
    return (defaultSAD * min(step, 256)) / 48
}

/// Zero-MV SAD of one 16x16 luma sub-block plus its 8x8 chroma, with an
/// early-out once `limit` is exceeded — the production skip criterion.
@inline(__always)
func computeZeroSAD16x16(
    cY: UnsafePointer<Int16>, rY: UnsafePointer<Int16>,
    cCb: UnsafePointer<Int16>, rCb: UnsafePointer<Int16>,
    cCr: UnsafePointer<Int16>, rCr: UnsafePointer<Int16>,
    bx: Int, by: Int, width: Int, limit: Int
) -> Int {
    var sad: Int = 0
    let strideY = width
    let strideC = (width + 1) / 2
    let bxC = bx / 2
    let byC = by / 2

    for y in 0..<16 {
        let offset = (by + y) * strideY + bx
        for x in 0..<16 {
            sad &+= Int((Int32(cY[offset + x]) - Int32(rY[offset + x])).magnitude)
        }
        if limit < sad { return sad }
    }

    for y in 0..<8 {
        let offset = (byC + y) * strideC + bxC
        for x in 0..<8 {
            sad &+= Int((Int32(cCb[offset + x]) - Int32(rCb[offset + x])).magnitude)
            sad &+= Int((Int32(cCr[offset + x]) - Int32(rCr[offset + x])).magnitude)
        }
        if limit < sad { return sad }
    }
    return sad
}

/// computeZeroSAD16x16 for partial edge blocks (arbitrary sub-block size).
@inline(__always)
func computeZeroSADSubBlock(
    cY: UnsafePointer<Int16>, rY: UnsafePointer<Int16>,
    cCb: UnsafePointer<Int16>, rCb: UnsafePointer<Int16>,
    cCr: UnsafePointer<Int16>, rCr: UnsafePointer<Int16>,
    bx: Int, by: Int, width: Int, height: Int,
    subWidth: Int, subHeight: Int, subWc: Int, subHc: Int,
    limit: Int
) -> Int {
    var sad: Int = 0
    let strideY = width
    for y in 0..<subHeight {
        let yy = by + y
        let offset = yy * strideY + bx
        for x in 0..<subWidth {
            sad &+= Int((Int32(cY[offset + x]) - Int32(rY[offset + x])).magnitude)
        }
        if limit < sad { return sad }
    }

    let bxC = bx / 2
    let byC = by / 2
    let strideC = (width + 1) / 2
    for y in 0..<subHc {
        let yy = byC + y
        let offset = yy * strideC + bxC
        for x in 0..<subWc {
            sad &+= Int((Int32(cCb[offset + x]) - Int32(rCb[offset + x])).magnitude)
            sad &+= Int((Int32(cCr[offset + x]) - Int32(rCr[offset + x])).magnitude)
        }
        if limit < sad { return sad }
    }
    return sad
}

/// Exact near-duplicate test for CopyFrame emission: mean abs diff over ALL
/// pixels (Y plus chroma at half weight), early-exiting once the bound is
/// crossed. Sampling estimators (estimateFastSAD) are unusable here — they
/// miss off-grid changes such as 1px HUD ticks, which must break a copy
/// chain. limitQ8 is the per-pixel bound in Q8 (256 = 1.0/px): the combined
/// metric madY + madC/2 == ((ySum + 2·cSum) << 8) / yCount for 4:2:0.
@inline(__always)
func isNearDuplicate(a: PlaneData420, b: PlaneData420, limitQ8: Int) -> Bool {
    guard a.y.count == b.y.count, a.cb.count == b.cb.count, 0 < a.y.count else { return false }
    let bound = (limitQ8 * a.y.count) >> 8
    var sum = 0
    let exceeded = withUnsafePointers(a.y, b.y) { aY, bY -> Bool in
        lumaAbsDiffExceedsProgressive(aY, bY, count: a.y.count, sum: &sum, bound: bound)
    }
    if exceeded { return false }
    let cbExceeded = withUnsafePointers(a.cb, b.cb) { aCb, bCb -> Bool in
        planeAbsDiffExceeds(aCb, bCb, count: a.cb.count, weight: 2, sum: &sum, bound: bound)
    }
    if cbExceeded { return false }
    let crExceeded = withUnsafePointers(a.cr, b.cr) { aCr, bCr -> Bool in
        planeAbsDiffExceeds(aCr, bCr, count: a.cr.count, weight: 2, sum: &sum, bound: bound)
    }
    return crExceeded != true
}

/// Luma pass with a progressive rejection bound: a frame that finishes within
/// `bound` can have accumulated at most bound·(p + count/4)/count after p
/// pixels unless its difference is pathologically back-loaded (the count/4
/// slack tolerates front-loading). Ordinary non-duplicates sit far above the
/// bound and reject after a few percent of the plane instead of accumulating
/// all the way to it. The early exit is safe-direction only: it can forgo a
/// copy opportunity, never produce a wrong copy.
@inline(__always)
private func lumaAbsDiffExceedsProgressive(_ a: UnsafePointer<Int16>, _ b: UnsafePointer<Int16>, count: Int, sum: inout Int, bound: Int) -> Bool {
    let slack = count / 4
    var i = 0
    var chunkAcc = 0
    while i + 16 <= count {
        let va = UnsafeRawPointer(a + i).loadUnaligned(as: SIMD16<Int16>.self)
        let vb = UnsafeRawPointer(b + i).loadUnaligned(as: SIMD16<Int16>.self)
        let d = va &- vb
        let ad = d.replacing(with: 0 &- d, where: d .< 0)
        chunkAcc += Int(ad.wrappedSum())
        i += 16
        if i % 4096 == 0 {
            sum += chunkAcc
            chunkAcc = 0
            if bound * (i + slack) < sum * count {
                return true
            }
        }
    }
    while i < count {
        chunkAcc += Int((Int32(a[i]) - Int32(b[i])).magnitude)
        i += 1
    }
    sum += chunkAcc
    return bound < sum
}

/// Accumulates weight·Σ|a−b| into sum, returning true as soon as it exceeds
/// bound. SIMD16 lanes; per-chunk lane sums stay below Int16 range (16×255).
@inline(__always)
private func planeAbsDiffExceeds(_ a: UnsafePointer<Int16>, _ b: UnsafePointer<Int16>, count: Int, weight: Int, sum: inout Int, bound: Int) -> Bool {
    var i = 0
    var chunkAcc = 0
    while i + 16 <= count {
        let va = UnsafeRawPointer(a + i).loadUnaligned(as: SIMD16<Int16>.self)
        let vb = UnsafeRawPointer(b + i).loadUnaligned(as: SIMD16<Int16>.self)
        let d = va &- vb
        let ad = d.replacing(with: 0 &- d, where: d .< 0)
        chunkAcc += Int(ad.wrappedSum())
        i += 16
        if i % 4096 == 0 {
            sum += weight * chunkAcc
            chunkAcc = 0
            if bound < sum { return true }
        }
    }
    while i < count {
        chunkAcc += Int((Int32(a[i]) - Int32(b[i])).magnitude)
        i += 1
    }
    sum += weight * chunkAcc
    return bound < sum
}

/// Fast whole-frame SAD estimate (every 4th pixel) for scene-change
/// detection: per-pixel average over Y plus a chroma term that catches
/// scene changes where luminance is similar but the color palette differs
/// (e.g. dark scene to dark scene).
@inline(__always)
func estimateFastSAD(a: PlaneData420, b: PlaneData420) -> Int {
    guard a.y.count == b.y.count, 0 < a.y.count else { return 0 }
    let yCount = a.y.count
    var sumY: UInt64 = 0
    withUnsafePointers(a.y, b.y) { aPtr, bPtr in
        for i in stride(from: 0, to: yCount, by: 4) {
            sumY += UInt64(abs(Int(aPtr[i]) - Int(bPtr[i])))
        }
    }
    let ySAD = Int((sumY * 4) / UInt64(yCount))

    let cbCount = a.cb.count
    guard a.cb.count == b.cb.count, 0 < cbCount else { return ySAD }

    var sumCb: UInt64 = 0
    var sumCr: UInt64 = 0
    withUnsafePointers(a.cb, b.cb, a.cr, b.cr) { aCb, bCb, aCr, bCr in
        for i in stride(from: 0, to: cbCount, by: 4) {
            sumCb += UInt64(abs(Int(aCb[i]) - Int(bCb[i])))
            sumCr += UInt64(abs(Int(aCr[i]) - Int(bCr[i])))
        }
    }
    let chromaSAD = Int(((sumCb + sumCr) * 4) / UInt64(cbCount * 2))

    // Weight: Y dominates but chroma provides critical color-change detection
    return ySAD + chromaSAD
}

/// Estimate frame-level SAD (Sum of Absolute Differences) between current
/// and previous PlaneData420 Y planes by sampling representative blocks.
/// Returns average per-pixel SAD as an Int for RateController input.
@inline(__always)
func estimateFrameSAD(current: PlaneData420, previous: PlaneData420) -> Int {
    let width = current.width
    let height = current.height

    let blockSize = 32
    let bw = min(blockSize, width)
    let bh = min(blockSize, height)

    // Sample 8 blocks at strategic positions (same as estimateQuantization)
    let points: [(Int, Int)] = [
        (0, 0),
        (max(0, width - bw), 0),
        (0, max(0, height - bh)),
        (max(0, width - bw), max(0, height - bh)),
        (max(0, (width - bw) / 2), 0),
        (max(0, width - bw), max(0, (height - bh) / 2)),
        (max(0, (width - bw) / 2), max(0, height - bh)),
        (0, max(0, (height - bh) / 2)),
    ]

    var totalSAD: Int = 0
    var totalPixels: Int = 0

    for (sx, sy) in points {
        for y in sy..<min(sy + bh, height) {
            let rowOffset = y * width
            for x in sx..<min(sx + bw, width) {
                let idx = rowOffset + x
                totalSAD += abs(Int(current.y[idx]) - Int(previous.y[idx]))
            }
        }
        totalPixels += bw * bh
    }

    if 0 < totalPixels {
        return totalSAD / totalPixels
    }
    return 0
}

/// Full block scan with activity mask-based reconstruction distortion measurement.
/// Returns per-pixel average SAD (same unit as traditional estimateFrameSAD).
@inline(__always)
func computeMaskedReconDistortion(
    original: PlaneData420,
    reconstructed: PlaneData420,
    sads: [Int]?
) -> Int {
    let width = original.width
    let height = original.height

    let blockSize = 32
    let colCount = (width + 31) / 32
    let rowCount = (height + 31) / 32

    var totalSAD: Int = 0
    var activePixels: Int = 0
    var totalFallbackSAD: Int = 0
    var totalPixels: Int = 0

    withUnsafePointers(original.y, reconstructed.y) { oBase, rBase in
        for r in 0..<rowCount {
            let sy = r * blockSize
            let bh = min(blockSize, height - sy)
            let rowOffset = r * colCount

            for c in 0..<colCount {
                let sx = c * blockSize
                let bw = min(blockSize, width - sx)

                var blockSAD = 0
                for y in sy..<sy+bh {
                    let oRow = oBase + y * width + sx
                    let rRow = rBase + y * width + sx

                    for x in 0..<bw {
                        blockSAD += abs(Int(oRow[x]) - Int(rRow[x]))
                    }
                }

                let pixels = bw * bh
                totalPixels += pixels
                totalFallbackSAD += blockSAD

                if let sads = sads {
                    if 256 < sads[rowOffset + c] {
                        totalSAD += blockSAD
                        activePixels += pixels
                    }
                }
            }
        }
    }

    // Fallback: use full block average if active blocks are less than 5% of total
    if sads == nil || activePixels < (totalPixels / 20) {
        if 0 < totalPixels {
            return (totalFallbackSAD << 8) / totalPixels
        }
        return 0
    }

    if 0 < activePixels {
        return (totalSAD << 8) / activePixels
    }
    return 0
}
