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
