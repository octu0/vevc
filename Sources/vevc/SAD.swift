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

/// Weighted-prediction luma offset estimate (#21): subsampled mean
/// difference between the source and its prediction reference. Fades shift
/// the whole frame's luminance, which motion compensation cannot express —
/// signaling this offset and forming P′ = P + offset removes the DC of the
/// fade residual (measured −30% MAD on fade frames, inert elsewhere).
/// Encoder-only estimator: the value is signaled, so precision is a rate
/// question, not a sync question. Works on per-32px-block means (8×8 samples
/// at stride 4 per block) instead of a whole-plane mean: the offset is only
/// valid when the luminance shift is spatially uniform. A localized flash
/// produces a bimodal block-mean distribution, and at saturated qsteps the
/// residual cannot repair a misapplied global offset (measured −2〜−3 dB on
/// flash-exit frames), so the estimate is the MEDIAN block mean and is
/// rejected entirely when the block-mean deviation exceeds
/// maxBlockMeanStd. Zeroed inside the ±1 noise band and clamped to the
/// signaled Int8 range.
@inline(__always)
func estimateLumaOffset(source: [Int16], reference: [Int16], width: Int, height: Int) -> Int {
    guard source.count == reference.count, source.count == width * height else { return 0 }
    let bw = width / 32
    let bh = height / 32
    let blockCount = bw * bh
    guard 0 < blockCount else { return 0 }
    // Per-block sums of 64 samples each (8×8 at stride 4); block mean =
    // sum / 64. Row-major over the plane so the reads stay sequential.
    var sums = [Int](repeating: 0, count: blockCount)
    sums.withUnsafeMutableBufferPointer { sumsBuf in
        let sumsBase = sumsBuf.baseAddress!
        withUnsafePointers(source, reference) { s, r in
            let xEnd = bw * 32
            var y = 0
            let yEnd = bh * 32
            while y < yEnd {
                let rowBase = y * width
                let blockRowBase = (y / 32) * bw
                var x = 0
                while x < xEnd {
                    let i = rowBase + x
                    sumsBase[blockRowBase + (x >> 5)] += Int(s[i]) - Int(r[i])
                    x += 4
                }
                y += 4
            }
        }
    }
    var total = 0
    var totalSq = 0
    for sum in sums {
        total += sum
        totalSq += sum * sum
    }
    // Uniformity gate: std of block means ≤ maxBlockMeanStd. On the block-sum
    // scale (mean × 64): var_sum ≤ (maxBlockMeanStd × 64)².
    let maxBlockMeanStd = 12
    let meanSum = total / blockCount
    let varSum = totalSq / blockCount - meanSum * meanSum
    if (maxBlockMeanStd * 64) * (maxBlockMeanStd * 64) < varSum { return 0 }
    sums.sort()
    let median = sums[blockCount / 2]
    let offset = median < 0 ? (median - 32) / 64 : (median + 32) / 64
    if -2 < offset && offset < 2 { return 0 }
    return max(-127, min(127, offset))
}

// MARK: - Block activity (σ-normalized AQ)

/// SSIM weights a coded error by the local variance denominator 1/(2σ²+C):
/// the same error is visible on flat blocks and masked on textured blocks.
/// The per-32px-block luma variance map drives the block-adaptive dead-zone
/// selection (Quant.swift) — encoder-only, the signaled step never changes.
public enum BlockActivityClass: UInt8, Sendable {
    case normal = 0
    case flat = 1
    case textured = 2
    case incoherentTextured = 3
}

/// Compute gradient direction coherence C in [0.0, 1.0] for a 32x32 block using 3x3 Sobel filter.
/// C = sqrt((sum Gx)^2 + (sum Gy)^2) / sum sqrt(Gx^2 + Gy^2)
@inline(__always)
func computeBlockCoherence(
    source: UnsafePointer<Int16>,
    stride: Int,
    width: Int,
    height: Int,
    bx: Int,
    by: Int,
    bw: Int,
    bh: Int
) -> Double {
    let startY = max(1, by)
    let endY = min(height - 2, by + bh - 1)
    let startX = max(1, bx)
    let endX = min(width - 2, bx + bw - 1)

    if endY < startY {
        return 1.0
    }
    if endX < startX {
        return 1.0
    }

    var sumGx: Double = 0.0
    var sumGy: Double = 0.0
    var sumMag: Double = 0.0

    for y in startY...endY {
        let prevRow = source.advanced(by: (y - 1) * stride)
        let currRow = source.advanced(by: y * stride)
        let nextRow = source.advanced(by: (y + 1) * stride)

        for x in startX...endX {
            let p00 = Double(prevRow[x - 1])
            let p02 = Double(prevRow[x + 1])
            let p10 = Double(currRow[x - 1])
            let p12 = Double(currRow[x + 1])
            let p20 = Double(nextRow[x - 1])
            let p22 = Double(nextRow[x + 1])

            let gx = (p02 + (2.0 * p12) + p22) - (p00 + (2.0 * p10) + p20)

            let p01 = Double(prevRow[x])
            let p21 = Double(nextRow[x])

            let gy = (p20 + (2.0 * p21) + p22) - (p00 + (2.0 * p01) + p02)

            let mag = ((gx * gx) + (gy * gy)).squareRoot()
            sumGx += gx
            sumGy += gy
            sumMag += mag
        }
    }

    if sumMag <= 1e-6 {
        return 1.0
    }

    let totalNorm = ((sumGx * sumGx) + (sumGy * sumGy)).squareRoot()
    return totalNorm / sumMag
}

/// Default classification thresholds on the variance scale (σ² of stride-4
/// samples). Swept on miko_700 500k against the fixed-dead-zone baseline:
/// the TEXTURED side carries all of the gain (masked coefficients drop, the
/// rate-control loop reinvests the freed bits: min +0.0030 / avg +0.0032
/// SSIM at −16.6% size with σ² ≥ 600, delta 65536). The FLAT side (bias
/// toward round-to-nearest) costs bits with no measurable SSIM return —
/// disabled by default (threshold 0 classifies no block as flat).
let aqFlatVarianceMaxDefault = 0
let aqTexturedVarianceMinDefault = 1600
/// Dead-zone bias delta in Q16 step units (16384 = 0.25 step) at full
/// saturation; QuantizationTable ramps it with the qHigh saturation
/// extension (zero at baseStep ≤ 2048). Stronger deltas measured better on
/// SSIM (65536/σ²≥600: min +0.0030, −16.6% size) but visibly erode HUD
/// text — text strokes are high-variance yet unmasked, and the erosion
/// lives in HL/LH where the variance class cannot separate strokes from
/// noise. 0.25 step only drops coefficients near the dead-zone edge.
let aqBiasDeltaQ16Default = 16384

/// Per-32px-block luma variance map (single row-major stride-4 pass,
/// sum + sum-of-squares). Grid = ceil(width/32) × ceil(height/32) — 1:1 with
/// the skip map and with every layer's luma block grid.
@inline(__always)
func computeBlockActivityMap(source: [Int16], width: Int, height: Int) -> [Int32] {
    let bw = (width + 31) / 32
    let bh = (height + 31) / 32
    let blockCount = bw * bh
    var sums = [Int](repeating: 0, count: blockCount)
    var sumSqs = [Int](repeating: 0, count: blockCount)
    var counts = [Int](repeating: 0, count: blockCount)
    sums.withUnsafeMutableBufferPointer { sumsBuf in
        sumSqs.withUnsafeMutableBufferPointer { sumSqsBuf in
            counts.withUnsafeMutableBufferPointer { countsBuf in
                let sumsBase = sumsBuf.baseAddress!
                let sumSqsBase = sumSqsBuf.baseAddress!
                let countsBase = countsBuf.baseAddress!
                withUnsafePointers(source) { s in
                    var y = 0
                    while y < height {
                        let rowBase = y * width
                        let blockRowBase = (y / 32) * bw
                        var x = 0
                        while x < width {
                            let v = Int(s[rowBase + x])
                            let idx = blockRowBase + (x >> 5)
                            sumsBase[idx] += v
                            sumSqsBase[idx] += v * v
                            countsBase[idx] += 1
                            x += 4
                        }
                        y += 4
                    }
                }
            }
        }
    }
    var variances = [Int32](repeating: 0, count: blockCount)
    for i in 0..<blockCount {
        let n = counts[i]
        if n == 0 { continue }
        let mean = sums[i] / n
        variances[i] = Int32(clamping: sumSqs[i] / n - mean * mean)
    }
    return variances
}

/// Per-block gradient energy and local contrast, on the same 32px grid as
/// `computeBlockActivityMap`. `maxGrad` is the largest centre-difference
/// gradient magnitude |dI/dx| + |dI/dy| in the block (sampled every other
/// pixel), `avgGrad` its mean over the same samples, and `localRange` the
/// largest max-minus-min of any 8x8 sub-block. Together they separate a block
/// that is genuinely featureless from one that is low-variance only because a
/// single strong edge splits two flat halves.
@inline(__always)
func computeBlockGradientAndContrast(
    source: UnsafePointer<Int16>,
    stride: Int,
    width: Int,
    height: Int,
    bx: Int,
    by: Int,
    bw: Int,
    bh: Int
) -> (maxGrad: Int32, avgGrad: Int32, localRange: Int32) {
    var maxG: Int32 = 0
    var sumG: Int32 = 0
    var countG: Int32 = 0
    var maxSubRange: Int32 = 0

    let subBlocksY = (bh + 7) / 8
    let subBlocksX = (bw + 7) / 8

    for sby in 0..<subBlocksY {
        let subY = by + sby * 8
        let subH = min(8, height - subY)
        for sbx in 0..<subBlocksX {
            let subX = bx + sbx * 8
            let subW = min(8, width - subX)

            var minV: Int16 = 32767
            var maxV: Int16 = -32768

            for y in 0..<subH {
                let row = source.advanced(by: (subY + y) * stride + subX)
                for x in 0..<subW {
                    let v = row[x]
                    if v < minV { minV = v }
                    if maxV < v { maxV = v }
                }
            }
            let r = Int32(maxV - minV)
            if maxSubRange < r {
                maxSubRange = r
            }
        }
    }

    let startY = max(1, by)
    let endY = min(height - 2, by + bh - 1)
    let startX = max(1, bx)
    let endX = min(width - 2, bx + bw - 1)

    if startY <= endY {
        if startX <= endX {
            var y = startY
            while y <= endY {
                let prevRow = source.advanced(by: (y - 1) * stride)
                let currRow = source.advanced(by: y * stride)
                let nextRow = source.advanced(by: (y + 1) * stride)
                var x = startX
                while x <= endX {
                    let g = abs(Int32(currRow[x + 1]) - Int32(currRow[x - 1])) + abs(Int32(nextRow[x]) - Int32(prevRow[x]))
                    if maxG < g {
                        maxG = g
                    }
                    sumG += g
                    countG += 1
                    x += 2
                }
                y += 2
            }
        }
    }
    var avgG: Int32 = 0
    if 0 < countG {
        avgG = sumG / countG
    }
    return (maxG, avgG, maxSubRange)
}

@inline(__always)
func classifyBlockActivity(
    varianceMap: [Int32],
    flatVarianceMax: Int,
    texturedVarianceMin: Int,
    source: [Int16]? = nil,
    width: Int = 0,
    height: Int = 0,
    coherenceEnabled: Bool = true
) -> [BlockActivityClass] {
    var classes = [BlockActivityClass](repeating: .normal, count: varianceMap.count)
    let bw = (width + 31) / 32
    let colCount = bw
    for i in 0..<varianceMap.count {
        if varianceMap[i] < Int32(flatVarianceMax) {
            classes[i] = .flat
        }
        if Int32(texturedVarianceMin) <= varianceMap[i] {
            if coherenceEnabled {
                if let src = source {
                    if 0 < width {
                        if 0 < height {
                            let r = i / colCount
                            let c = i % colCount
                            let bx = c * 32
                            let by = r * 32
                            let blockW = min(32, width - bx)
                            let blockH = min(32, height - by)
                            let coherence = withUnsafePointers(src) { ptr in
                                computeBlockCoherence(source: ptr, stride: width, width: width, height: height, bx: bx, by: by, bw: blockW, bh: blockH)
                            }
                            if coherence < 0.85 {
                                classes[i] = .incoherentTextured
                            }
                            if 0.85 <= coherence {
                                classes[i] = .textured
                            }
                        }
                    }
                }
            }
            if coherenceEnabled != true || source == nil {
                classes[i] = .textured
            }
        }
    }
    return classes
}

/// Scene-cut detector constants (measured on miko_700 source pairs):
/// a cut replaces content so per-32px-block mean diffs go BOTH ways
/// (548→549: luma MAD 39.7, minority-side fraction 0.169), while a
/// flash/fade shifts everything in ONE direction (worst flash transition
/// 613→614: MAD 38.5, minority 0.000; fades ≤ 17.7 MAD, minority 0.000;
/// normal motion 2.9 MAD, minority 0.012). MAD alone cannot separate a cut
/// from a flash — the sign mix can, with a wide margin.
let sceneCutMinLumaMAD = 16
/// A block counts toward a side when |block mean| > 8 (on the 64-sample
/// block-sum scale: 8 × 64).
let sceneCutSideBlockSum = 8 * 64
/// Cut when the minority side holds ≥ 1/16 of the blocks.
let sceneCutMinorityDenominator = 16
/// estimateFastSAD is a per-pixel mean (luma + both chroma terms), so it
/// can never exceed 3 × 255. A sceneChangeThreshold above this value means
/// the caller wants scene detection off entirely (e.g. deterministic
/// tests) — the cut detector honors that too.
let maxEstimateFastSAD = 765
/// Treats a 4x spike above the noise floor as a cut. Live-action noise elevates the floor, avoiding false triggers. Based on empirical measurements where src_1 produced 11 false cuts/sec.
let sceneCutBaselineRatio = 4

/// Scene-cut detector: one row-major stride-4 pass over the luma planes
/// accumulating per-32px-block signed sums and the global abs sum, then
/// the two-sided test described above. Encoder-only (drives the I-frame
/// decision, nothing is signaled).
@inline(__always)
func detectSceneCut(source: [Int16], reference: [Int16], width: Int, height: Int) -> Bool {
    guard source.count == reference.count, source.count == width * height else { return false }
    let bw = width / 32
    let bh = height / 32
    let blockCount = bw * bh
    guard 0 < blockCount else { return false }
    var sums = [Int](repeating: 0, count: blockCount)
    var sumAbs = 0
    var sampleCount = 0
    sums.withUnsafeMutableBufferPointer { sumsBuf in
        let sumsBase = sumsBuf.baseAddress!
        withUnsafePointers(source, reference) { s, r in
            let xEnd = bw * 32
            var y = 0
            let yEnd = bh * 32
            while y < yEnd {
                let rowBase = y * width
                let blockRowBase = (y / 32) * bw
                var x = 0
                while x < xEnd {
                    let i = rowBase + x
                    let d = Int(s[i]) - Int(r[i])
                    sumsBase[blockRowBase + (x >> 5)] += d
                    sumAbs += abs(d)
                    x += 4
                }
                y += 4
            }
        }
    }
    sampleCount = (bw * 8) * (bh * 8)
    if sumAbs < sceneCutMinLumaMAD * sampleCount { return false }
    var pos = 0
    var neg = 0
    for sum in sums {
        if sceneCutSideBlockSum < sum {
            pos += 1
        } else if sum < -sceneCutSideBlockSum {
            neg += 1
        }
    }
    return blockCount <= min(pos, neg) * sceneCutMinorityDenominator
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
