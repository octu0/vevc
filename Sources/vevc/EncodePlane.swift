// MARK: - Encode Plane Arrays

import Foundation

@usableFromInline
struct __VevcSendableInt16Ptr: @unchecked Sendable {
    @usableFromInline let ptr: UnsafeMutablePointer<Int16>
    @usableFromInline init(_ ptr: UnsafeMutablePointer<Int16>) { self.ptr = ptr }
}

// MARK: - Spatial Adaptive Weight

/// Integer square root (floor).
/// Returns the largest integer n such that n*n <= value.
@inline(__always)
private func isqrt(_ value: Int) -> Int {
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

final class ConcurrentBox<T>: @unchecked Sendable {
    var value: T
    init(_ value: T) { self.value = value }
}

@inline(__always)
func evaluateQuantizeLayer32(block: inout BlockView, qt: QuantizationTable) {
    let view = block
    let subs = getSubbands32(view: view)
    let hl = subs.hl
    let lh = subs.lh
    let hh = subs.hh
    quantizeSIMDSignedMapping16(hl, q: qt.qMid)
    quantizeSIMDSignedMapping16(lh, q: qt.qMid)
    quantizeSIMDSignedMapping16(hh, q: qt.qHigh)
}

@inline(__always)
func evaluateQuantizeLayer16(block: inout BlockView, qt: QuantizationTable) {
    let view = block
    let subs = getSubbands16(view: view)
    let hl = subs.hl
    let lh = subs.lh
    let hh = subs.hh
    quantizeSIMDSignedMapping8(hl, q: qt.qMid)
    quantizeSIMDSignedMapping8(lh, q: qt.qMid)
    quantizeSIMDSignedMapping8(hh, q: qt.qHigh)
}

@inline(__always)
func evaluateQuantizeBase8(block: inout BlockView, qt: QuantizationTable) {
    let view = block
    let subs = getSubbands8(view: view)
    let ll = subs.ll
    let hl = subs.hl
    let lh = subs.lh
    let hh = subs.hh
    quantizeSIMD4(ll, q: qt.qLow)
    quantizeSIMDSignedMapping4(hl, q: qt.qMid)
    quantizeSIMDSignedMapping4(lh, q: qt.qMid)
    quantizeSIMDSignedMapping4(hh, q: qt.qHigh)
}

@inline(__always)
func evaluateQuantizeBase32(block: inout BlockView, qt: QuantizationTable) {
    let view = block
    let subs = getSubbands32(view: view)
    let ll = subs.ll
    let hl = subs.hl
    let lh = subs.lh
    let hh = subs.hh
    quantizeSIMD16(ll, q: qt.qLow)
    quantizeSIMDSignedMapping16(hl, q: qt.qMid)
    quantizeSIMDSignedMapping16(lh, q: qt.qMid)
    quantizeSIMDSignedMapping16(hh, q: qt.qHigh)
}

@inline(__always)
func extractSingleTransformBlocks32(r: Int16Reader, width: Int, height: Int, pool: BlockViewPool, onBlock: (@Sendable (Int, BlockView) -> Void)? = nil) async -> (blocks: [BlockView], subband: [Int16]) {
    let subWidth = ((width + 1) / 2)
    let subHeight = ((height + 1) / 2)
    let dstBaseAlloc = UnsafeMutablePointer<Int16>.allocate(capacity: subWidth * subHeight)
    dstBaseAlloc.initialize(repeating: 0, count: subWidth * subHeight)
    defer { 
        dstBaseAlloc.deinitialize(count: subWidth * subHeight)
        dstBaseAlloc.deallocate() 
    }
    let safeDst = __VevcSendableInt16Ptr(dstBaseAlloc)
    
    let rowCount = ((height + 32 - 1) / 32)
    let colCount = ((width + 32 - 1) / 32)
    let totalBlocks = rowCount * colCount
    
    var tmpBlocks: [BlockView] = []
    tmpBlocks.reserveCapacity(totalBlocks)
    for _ in 0..<totalBlocks {
        tmpBlocks.append(pool.get(width: 32, height: 32))
    }
    let blocks = tmpBlocks
    
    let chunkSize = 4
    await withTaskGroup(of: Void.self) { group in
        for sRow in stride(from: 0, to: rowCount, by: chunkSize) {
            let endRow = min(sRow + chunkSize, rowCount)
            group.addTask { [blocks, safeDst] in
                let dstBase = safeDst.ptr
                for i in sRow..<endRow {
                    let h = (i * 32)
                    for j in 0..<colCount {
                        let w = (j * 32)
                        if width <= w || height <= h { continue }
                        let view = blocks[(i * colCount) + j]
                        r.readBlock(x: w, y: h, width: 32, height: 32, into: view)
                        dwt2d_32(view)
                        
                        let destStartX = (w / 2)
                        let destStartY = (h / 2)
                        let subSize = (32 / 2)
                        let subs = getSubbands32(view: view)
                        let srcBase = subs.ll.base
                        let limit = min(subSize, (subWidth - destStartX))
                        
                        if 0 < limit {
                            if limit == subSize && (destStartY + subSize) <= subHeight {
                                let dstBasePtr = dstBase.advanced(by: (destStartY * subWidth) + destStartX)
                                dstBasePtr.advanced(by: subWidth * 0).update(from: srcBase.advanced(by: 32 * 0), count: 16)
                                dstBasePtr.advanced(by: subWidth * 1).update(from: srcBase.advanced(by: 32 * 1), count: 16)
                                dstBasePtr.advanced(by: subWidth * 2).update(from: srcBase.advanced(by: 32 * 2), count: 16)
                                dstBasePtr.advanced(by: subWidth * 3).update(from: srcBase.advanced(by: 32 * 3), count: 16)
                                dstBasePtr.advanced(by: subWidth * 4).update(from: srcBase.advanced(by: 32 * 4), count: 16)
                                dstBasePtr.advanced(by: subWidth * 5).update(from: srcBase.advanced(by: 32 * 5), count: 16)
                                dstBasePtr.advanced(by: subWidth * 6).update(from: srcBase.advanced(by: 32 * 6), count: 16)
                                dstBasePtr.advanced(by: subWidth * 7).update(from: srcBase.advanced(by: 32 * 7), count: 16)
                                dstBasePtr.advanced(by: subWidth * 8).update(from: srcBase.advanced(by: 32 * 8), count: 16)
                                dstBasePtr.advanced(by: subWidth * 9).update(from: srcBase.advanced(by: 32 * 9), count: 16)
                                dstBasePtr.advanced(by: subWidth * 10).update(from: srcBase.advanced(by: 32 * 10), count: 16)
                                dstBasePtr.advanced(by: subWidth * 11).update(from: srcBase.advanced(by: 32 * 11), count: 16)
                                dstBasePtr.advanced(by: subWidth * 12).update(from: srcBase.advanced(by: 32 * 12), count: 16)
                                dstBasePtr.advanced(by: subWidth * 13).update(from: srcBase.advanced(by: 32 * 13), count: 16)
                                dstBasePtr.advanced(by: subWidth * 14).update(from: srcBase.advanced(by: 32 * 14), count: 16)
                                dstBasePtr.advanced(by: subWidth * 15).update(from: srcBase.advanced(by: 32 * 15), count: 16)
                            } else {
                                for blockY in 0..<subSize {
                                    let dstY = (destStartY + blockY)
                                    if subHeight > dstY {
                                        let srcPtr = srcBase.advanced(by: (blockY * 32))
                                        let dstIdx = ((dstY * subWidth) + destStartX)
                                        dstBase.advanced(by: dstIdx).update(from: srcPtr, count: limit)
                                    }
                                }
                            }
                        }
                        
                        onBlock?((i * colCount) + j, view)
                    }
                }
            }
        }
    }
    
    var subband = pool.getInt16(count: subWidth * subHeight)
    subband.withUnsafeMutableBufferPointer { buf in
        if let base = buf.baseAddress {
            base.update(from: dstBaseAlloc, count: subWidth * subHeight)
        }
    }
    return (blocks, subband)
}

@inline(__always)
func extractSingleTransformSubband32(r: Int16Reader, width: Int, height: Int, pool: BlockViewPool) async -> [Int16] {
    let subWidth = ((width + 1) / 2)
    let subHeight = ((height + 1) / 2)
    var subband = pool.getInt16(count: subWidth * subHeight)
    let rowCount = ((height + 32 - 1) / 32)
    let colCount = ((width + 32 - 1) / 32)
    let totalBlocks = rowCount * colCount
    
    var tmpBlocks: [BlockView] = []
    tmpBlocks.reserveCapacity(totalBlocks)
    for _ in 0..<totalBlocks {
        tmpBlocks.append(pool.get(width: 32, height: 32))
    }
    let blocks = tmpBlocks
    
    let chunkSize = 8
    await withTaskGroup(of: Void.self) { group in
        for sRow in stride(from: 0, to: rowCount, by: chunkSize) {
            let endRow = min(sRow + chunkSize, rowCount)
            group.addTask { [blocks] in
                for i in sRow..<endRow {
                    let h = (i * 32)
                    for j in 0..<colCount {
                        let w = (j * 32)
                        if width <= w || height <= h { continue }
                        let view = blocks[(i * colCount) + j]
                        r.readBlock(x: w, y: h, width: 32, height: 32, into: view)
                        dwt2d_32(view)
                    }
                }
            }
        }
    }
    
    subband.withUnsafeMutableBufferPointer { dstBuf in
        guard let dstBase = dstBuf.baseAddress else { return }
        for i in 0..<rowCount {
            let h = (i * 32)
            for j in 0..<colCount {
                let w = (j * 32)
                if width <= w || height <= h { continue }
                let llBlock = blocks[(i * colCount) + j]
                
                let destStartX = (w / 2)
                let destStartY = (h / 2)
                let subSize = (32 / 2)

                let view = llBlock
                let subs = getSubbands32(view: view)
                let srcBase = subs.ll.base
                let limit = min(subSize, (subWidth - destStartX))

                guard 0 < limit else { continue }

                if limit == subSize && (destStartY + subSize) <= subHeight {
                    let dstBasePtr = dstBase.advanced(by: (destStartY * subWidth) + destStartX)
                    dstBasePtr.advanced(by: subWidth * 0).update(from: srcBase.advanced(by: 32 * 0), count: 16)
                    dstBasePtr.advanced(by: subWidth * 1).update(from: srcBase.advanced(by: 32 * 1), count: 16)
                    dstBasePtr.advanced(by: subWidth * 2).update(from: srcBase.advanced(by: 32 * 2), count: 16)
                    dstBasePtr.advanced(by: subWidth * 3).update(from: srcBase.advanced(by: 32 * 3), count: 16)
                    dstBasePtr.advanced(by: subWidth * 4).update(from: srcBase.advanced(by: 32 * 4), count: 16)
                    dstBasePtr.advanced(by: subWidth * 5).update(from: srcBase.advanced(by: 32 * 5), count: 16)
                    dstBasePtr.advanced(by: subWidth * 6).update(from: srcBase.advanced(by: 32 * 6), count: 16)
                    dstBasePtr.advanced(by: subWidth * 7).update(from: srcBase.advanced(by: 32 * 7), count: 16)
                    dstBasePtr.advanced(by: subWidth * 8).update(from: srcBase.advanced(by: 32 * 8), count: 16)
                    dstBasePtr.advanced(by: subWidth * 9).update(from: srcBase.advanced(by: 32 * 9), count: 16)
                    dstBasePtr.advanced(by: subWidth * 10).update(from: srcBase.advanced(by: 32 * 10), count: 16)
                    dstBasePtr.advanced(by: subWidth * 11).update(from: srcBase.advanced(by: 32 * 11), count: 16)
                    dstBasePtr.advanced(by: subWidth * 12).update(from: srcBase.advanced(by: 32 * 12), count: 16)
                    dstBasePtr.advanced(by: subWidth * 13).update(from: srcBase.advanced(by: 32 * 13), count: 16)
                    dstBasePtr.advanced(by: subWidth * 14).update(from: srcBase.advanced(by: 32 * 14), count: 16)
                    dstBasePtr.advanced(by: subWidth * 15).update(from: srcBase.advanced(by: 32 * 15), count: 16)
                } else {
                    for blockY in 0..<subSize {
                        let dstY = (destStartY + blockY)
                        if subHeight <= dstY { continue }
                        let srcPtr = srcBase.advanced(by: (blockY * 32))
                        let dstIdx = ((dstY * subWidth) + destStartX)
                        dstBase.advanced(by: dstIdx).update(from: srcPtr, count: limit)
                    }
                }
            }
        }
    }
    
    pool.putAll(blocks)
    return subband
}

@inline(__always)
func extractSingleTransformBlocks16(r: Int16Reader, width: Int, height: Int, pool: BlockViewPool, onBlock: (@Sendable (Int, BlockView) -> Void)? = nil) async -> (blocks: [BlockView], subband: [Int16]) {
    let subWidth = ((width + 1) / 2)
    let subHeight = ((height + 1) / 2)
    let dstBaseAlloc = UnsafeMutablePointer<Int16>.allocate(capacity: subWidth * subHeight)
    dstBaseAlloc.initialize(repeating: 0, count: subWidth * subHeight)
    defer { 
        dstBaseAlloc.deinitialize(count: subWidth * subHeight)
        dstBaseAlloc.deallocate() 
    }
    let safeDst = __VevcSendableInt16Ptr(dstBaseAlloc)
    
    let rowCount = ((height + 16 - 1) / 16)
    let colCount = ((width + 16 - 1) / 16)
    let totalBlocks = rowCount * colCount
    
    var tmpBlocks: [BlockView] = []
    tmpBlocks.reserveCapacity(totalBlocks)
    for _ in 0..<totalBlocks {
        tmpBlocks.append(pool.get(width: 16, height: 16))
    }
    let blocks = tmpBlocks
    
    let chunkSize = 4
    await withTaskGroup(of: Void.self) { group in
        for sRow in stride(from: 0, to: rowCount, by: chunkSize) {
            let endRow = min(sRow + chunkSize, rowCount)
            group.addTask { [blocks, safeDst] in
                let dstBase = safeDst.ptr
                for i in sRow..<endRow {
                    let h = (i * 16)
                    for j in 0..<colCount {
                        let w = (j * 16)
                        if width <= w || height <= h { continue }
                        let view = blocks[(i * colCount) + j]
                        r.readBlock(x: w, y: h, width: 16, height: 16, into: view)
                        dwt2d_16(view)
                        
                        let destStartX = (w / 2)
                        let destStartY = (h / 2)
                        let subSize = (16 / 2)
                        let subs = getSubbands16(view: view)
                        let srcBase = subs.ll.base
                        let limit = min(subSize, (subWidth - destStartX))
                        
                        if 0 < limit {
                            if limit == subSize && (destStartY + subSize) <= subHeight {
                                let dstBasePtr = dstBase.advanced(by: (destStartY * subWidth) + destStartX)
                                dstBasePtr.advanced(by: subWidth * 0).update(from: srcBase.advanced(by: 16 * 0), count: 8)
                                dstBasePtr.advanced(by: subWidth * 1).update(from: srcBase.advanced(by: 16 * 1), count: 8)
                                dstBasePtr.advanced(by: subWidth * 2).update(from: srcBase.advanced(by: 16 * 2), count: 8)
                                dstBasePtr.advanced(by: subWidth * 3).update(from: srcBase.advanced(by: 16 * 3), count: 8)
                                dstBasePtr.advanced(by: subWidth * 4).update(from: srcBase.advanced(by: 16 * 4), count: 8)
                                dstBasePtr.advanced(by: subWidth * 5).update(from: srcBase.advanced(by: 16 * 5), count: 8)
                                dstBasePtr.advanced(by: subWidth * 6).update(from: srcBase.advanced(by: 16 * 6), count: 8)
                                dstBasePtr.advanced(by: subWidth * 7).update(from: srcBase.advanced(by: 16 * 7), count: 8)
                            } else {
                                for blockY in 0..<subSize {
                                    let dstY = (destStartY + blockY)
                                    if subHeight > dstY {
                                        let srcPtr = srcBase.advanced(by: (blockY * 16))
                                        let dstIdx = ((dstY * subWidth) + destStartX)
                                        dstBase.advanced(by: dstIdx).update(from: srcPtr, count: limit)
                                    }
                                }
                            }
                        }
                        
                        onBlock?((i * colCount) + j, view)
                    }
                }
            }
        }
    }
    
    var subband = pool.getInt16(count: subWidth * subHeight)
    subband.withUnsafeMutableBufferPointer { buf in
        if let base = buf.baseAddress {
            base.update(from: dstBaseAlloc, count: subWidth * subHeight)
        }
    }
    return (blocks, subband)
}

@inline(__always)
func extractSingleTransformSubband16(r: Int16Reader, width: Int, height: Int, pool: BlockViewPool) async -> [Int16] {
    let subWidth = ((width + 1) / 2)
    let subHeight = ((height + 1) / 2)
    var subband = pool.getInt16(count: subWidth * subHeight)
    let rowCount = ((height + 16 - 1) / 16)
    let colCount = ((width + 16 - 1) / 16)
    let totalBlocks = rowCount * colCount
    
    var tmpBlocks: [BlockView] = []
    tmpBlocks.reserveCapacity(totalBlocks)
    for _ in 0..<totalBlocks {
        tmpBlocks.append(pool.get(width: 16, height: 16))
    }
    let blocks = tmpBlocks
    
    let chunkSize = 8
    await withTaskGroup(of: Void.self) { group in
        for sRow in stride(from: 0, to: rowCount, by: chunkSize) {
            let endRow = min(sRow + chunkSize, rowCount)
            group.addTask { [blocks] in
                for i in sRow..<endRow {
                    let h = (i * 16)
                    for j in 0..<colCount {
                        let w = (j * 16)
                        if width <= w || height <= h { continue }
                        let view = blocks[(i * colCount) + j]
                        r.readBlock(x: w, y: h, width: 16, height: 16, into: view)
                        dwt2d_16(view)
                    }
                }
            }
        }
    }
    
    subband.withUnsafeMutableBufferPointer { dstBuf in
        guard let dstBase = dstBuf.baseAddress else { return }
        for i in 0..<rowCount {
            let h = (i * 16)
            for j in 0..<colCount {
                let w = (j * 16)
                if width <= w || height <= h { continue }
                let llBlock = blocks[(i * colCount) + j]
                
                let destStartX = (w / 2)
                let destStartY = (h / 2)
                let subSize = (16 / 2)

                let view = llBlock
                let subs = getSubbands16(view: view)
                let srcBase = subs.ll.base
                let limit = min(subSize, (subWidth - destStartX))

                guard 0 < limit else { continue }

                if limit == subSize && (destStartY + subSize) <= subHeight {
                    let dstBasePtr = dstBase.advanced(by: (destStartY * subWidth) + destStartX)
                    dstBasePtr.advanced(by: subWidth * 0).update(from: srcBase.advanced(by: 16 * 0), count: 8)
                    dstBasePtr.advanced(by: subWidth * 1).update(from: srcBase.advanced(by: 16 * 1), count: 8)
                    dstBasePtr.advanced(by: subWidth * 2).update(from: srcBase.advanced(by: 16 * 2), count: 8)
                    dstBasePtr.advanced(by: subWidth * 3).update(from: srcBase.advanced(by: 16 * 3), count: 8)
                    dstBasePtr.advanced(by: subWidth * 4).update(from: srcBase.advanced(by: 16 * 4), count: 8)
                    dstBasePtr.advanced(by: subWidth * 5).update(from: srcBase.advanced(by: 16 * 5), count: 8)
                    dstBasePtr.advanced(by: subWidth * 6).update(from: srcBase.advanced(by: 16 * 6), count: 8)
                    dstBasePtr.advanced(by: subWidth * 7).update(from: srcBase.advanced(by: 16 * 7), count: 8)
                } else {
                    for blockY in 0..<subSize {
                        let dstY = (destStartY + blockY)
                        if subHeight <= dstY { continue }
                        let srcPtr = srcBase.advanced(by: (blockY * 16))
                        let dstIdx = ((dstY * subWidth) + destStartX)
                        dstBase.advanced(by: dstIdx).update(from: srcPtr, count: limit)
                    }
                }
            }
        }
    }
    
    pool.putAll(blocks)
    return subband
}

@inline(__always)
func extractSingleTransformBlocksBase8(r: Int16Reader, width: Int, height: Int, pool: BlockViewPool) async -> [BlockView] {
    let rowCount = ((height + 8 - 1) / 8)
    let colCount = ((width + 8 - 1) / 8)
    let totalBlocks = rowCount * colCount
    
    var tmpBlocks: [BlockView] = []
    tmpBlocks.reserveCapacity(totalBlocks)
    for _ in 0..<totalBlocks {
        tmpBlocks.append(pool.get(width: 8, height: 8))
    }
    let blocks = tmpBlocks
    
    let chunkSize = 4
    await withTaskGroup(of: Void.self) { group in
        for sRow in stride(from: 0, to: rowCount, by: chunkSize) {
            let endRow = min(sRow + chunkSize, rowCount)
            group.addTask { [blocks] in
                for i in sRow..<endRow {
                    let h = (i * 8)
                    for j in 0..<colCount {
                        let w = (j * 8)
                        if width <= w || height <= h { continue }
                        let view = blocks[(i * colCount) + j]
                        r.readBlock(x: w, y: h, width: 8, height: 8, into: view)
                        dwt2d_8(view)
                    }
                }
            }
        }
    }    
    return blocks
}

@inline(__always)
func extractSingleTransformBlocksBase32(r: Int16Reader, width: Int, height: Int, pool: BlockViewPool) async -> [BlockView] {
    let rowCount = ((height + 32 - 1) / 32)
    let colCount = ((width + 32 - 1) / 32)
    let totalBlocks = rowCount * colCount
    
    var tmpBlocks: [BlockView] = []
    tmpBlocks.reserveCapacity(totalBlocks)
    for _ in 0..<totalBlocks {
        tmpBlocks.append(pool.get(width: 32, height: 32))
    }
    let blocks = tmpBlocks
    
    let chunkSize = 4
    await withTaskGroup(of: Void.self) { group in
        for sRow in stride(from: 0, to: rowCount, by: chunkSize) {
            let endRow = min(sRow + chunkSize, rowCount)
            group.addTask { [blocks] in
                for i in sRow..<endRow {
                    let h = (i * 32)
                    for j in 0..<colCount {
                        let w = (j * 32)
                        if width <= w || height <= h { continue }
                        let view = blocks[(i * colCount) + j]
                        r.readBlock(x: w, y: h, width: 32, height: 32, into: view)
                        dwt2d_32(view)
                    }
                }
            }
        }
    }    
    return blocks
}

@inline(__always)
func subtractCoeffs32(currBlocks: inout [BlockView], predBlocks: inout [BlockView]) {
    for i in currBlocks.indices {
        let vC = currBlocks[i]
        let vP = predBlocks[i]
        let cBase = vC.base
        let pBase = vP.base
        for y in 0..<16 {
            let ptrC = cBase.advanced(by: y * 32 + 16)
            let ptrP = pBase.advanced(by: y * 32 + 16)
            let vecC = UnsafeRawPointer(ptrC).loadUnaligned(as: SIMD16<Int16>.self)
            let vecP = UnsafeRawPointer(ptrP).loadUnaligned(as: SIMD16<Int16>.self)
            let res = vecC &- vecP
            UnsafeMutableRawPointer(ptrC).storeBytes(of: res, as: SIMD16<Int16>.self)
        }
        let ptrC_bot = cBase.advanced(by: 512)
        let ptrP_bot = pBase.advanced(by: 512)
        for offset in stride(from: 0, to: 512, by: 16) {
            let vecC = UnsafeRawPointer(ptrC_bot.advanced(by: offset)).loadUnaligned(as: SIMD16<Int16>.self)
            let vecP = UnsafeRawPointer(ptrP_bot.advanced(by: offset)).loadUnaligned(as: SIMD16<Int16>.self)
            let res = vecC &- vecP
            UnsafeMutableRawPointer(ptrC_bot.advanced(by: offset)).storeBytes(of: res, as: SIMD16<Int16>.self)
        }
    }
}

@inline(__always)
func subtractCoeffs16(currBlocks: inout [BlockView], predBlocks: inout [BlockView]) {
    for i in currBlocks.indices {
        let vC = currBlocks[i]
        let vP = predBlocks[i]
        let cBase = vC.base
        let pBase = vP.base
        for y in 0..<8 {
            let ptrC = cBase.advanced(by: y * 16 + 8)
            let ptrP = pBase.advanced(by: y * 16 + 8)
            let vecC = UnsafeRawPointer(ptrC).loadUnaligned(as: SIMD8<Int16>.self)
            let vecP = UnsafeRawPointer(ptrP).loadUnaligned(as: SIMD8<Int16>.self)
            let res = vecC &- vecP
            UnsafeMutableRawPointer(ptrC).storeBytes(of: res, as: SIMD8<Int16>.self)
        }
        let ptrC_bot = cBase.advanced(by: 128)
        let ptrP_bot = pBase.advanced(by: 128)
        for offset in stride(from: 0, to: 128, by: 8) {
            let vecC = UnsafeRawPointer(ptrC_bot.advanced(by: offset)).loadUnaligned(as: SIMD8<Int16>.self)
            let vecP = UnsafeRawPointer(ptrP_bot.advanced(by: offset)).loadUnaligned(as: SIMD8<Int16>.self)
            let res = vecC &- vecP
            UnsafeMutableRawPointer(ptrC_bot.advanced(by: offset)).storeBytes(of: res, as: SIMD8<Int16>.self)
        }
    }
}

@inline(__always)
func subtractCoeffsBase8(currBlocks: inout [BlockView], predBlocks: inout [BlockView]) {
    for i in currBlocks.indices {
        let vC = currBlocks[i]
        let vP = predBlocks[i]
        let ptrC = vC.base
        let ptrP = vP.base
        for offset in stride(from: 0, to: 64, by: 8) {
            let vecC = UnsafeRawPointer(ptrC.advanced(by: offset)).loadUnaligned(as: SIMD8<Int16>.self)
            let vecP = UnsafeRawPointer(ptrP.advanced(by: offset)).loadUnaligned(as: SIMD8<Int16>.self)
            let res = vecC &- vecP
            UnsafeMutableRawPointer(ptrC.advanced(by: offset)).storeBytes(of: res, as: SIMD8<Int16>.self)
        }
    }
}

@inline(__always)
func subtractCoeffsBase32(currBlocks: inout [BlockView], predBlocks: inout [BlockView]) {
    for i in currBlocks.indices {
        let vC = currBlocks[i]
        let vP = predBlocks[i]
        let ptrC = vC.base
        let ptrP = vP.base
        for offset in stride(from: 0, to: 1024, by: 16) {
            let vecC = UnsafeRawPointer(ptrC.advanced(by: offset)).loadUnaligned(as: SIMD16<Int16>.self)
            let vecP = UnsafeRawPointer(ptrP.advanced(by: offset)).loadUnaligned(as: SIMD16<Int16>.self)
            let res = vecC &- vecP
            UnsafeMutableRawPointer(ptrC.advanced(by: offset)).storeBytes(of: res, as: SIMD16<Int16>.self)
        }
    }
}

@inline(__always)
func preparePlaneLayer32(pd: PlaneData420, pool: BlockViewPool, sads: [Int]?, layer: UInt8, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int) async throws -> (PlaneData420, [BlockView], [BlockView], [BlockView]) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    
    let yColCount32 = (dx + 31) / 32
    let cbColCount32 = (cbDx + 31) / 32
    let yRowCount32 = (dy + 31) / 32
    let cbRowCount32 = (cbDy + 31) / 32
    
    async let taskBufY = { () -> ([Int16], [BlockView]) in
        let (blocks, subband) = await extractSingleTransformBlocks32(r: pd.rY, width: dx, height: dy, pool: pool) { index, view in
            if let sList = sads, index < sList.count {
                let col = index % yColCount32
                let row = index / yColCount32
                let threshold = spatialSADThreshold(baseSAD: 150, blockCol: col, blockRow: row, colCount: yColCount32, rowCount: yRowCount32)
                if sList[index] < threshold { view.clearAll() }
            }
            var v = view
            evaluateQuantizeLayer32(block: &v, qt: qtY)
        }
        return (subband, blocks)
    }()
    
    async let taskBufCb = { () -> ([Int16], [BlockView]) in
        let (blocks, subband) = await extractSingleTransformBlocks32(r: pd.rCb, width: cbDx, height: cbDy, pool: pool) { index, view in
            var sadVal = Int.max
            if let sList = sads {
                let col = index % cbColCount32
                let row = index / cbColCount32
                let lumaIdx = (row * 2) * yColCount32 + (col * 2)
                if lumaIdx < sList.count { sadVal = sList[lumaIdx] }

                let threshold = spatialSADThreshold(baseSAD: 150, blockCol: col, blockRow: row, colCount: cbColCount32, rowCount: cbRowCount32)
                if sadVal < threshold { view.clearAll() }
            }
            var v = view
            evaluateQuantizeLayer32(block: &v, qt: qtC)
        }
        return (subband, blocks)
    }()
    
    async let taskBufCr = { () -> ([Int16], [BlockView]) in
        let (blocks, subband) = await extractSingleTransformBlocks32(r: pd.rCr, width: cbDx, height: cbDy, pool: pool) { index, view in
            var sadVal = Int.max
            if let sList = sads {
                let col = index % cbColCount32
                let row = index / cbColCount32
                let lumaIdx = (row * 2) * yColCount32 + (col * 2)
                if lumaIdx < sList.count { sadVal = sList[lumaIdx] }

                let threshold = spatialSADThreshold(baseSAD: 75, blockCol: col, blockRow: row, colCount: cbColCount32, rowCount: cbRowCount32)
                if sadVal < threshold { view.clearAll() }
            }
            var v = view
            evaluateQuantizeLayer32(block: &v, qt: qtC)
        }
        return (subband, blocks)
    }()

    let (subY, yBlocks) = await taskBufY
    let (subCb, cbBlocks) = await taskBufCb
    let (subCr, crBlocks) = await taskBufCr

    let subPlane = PlaneData420(width: (dx + 1) / 2, height: (dy + 1) / 2, y: subY, cb: subCb, cr: subCr)
    return (subPlane, yBlocks, cbBlocks, crBlocks)
}

@inline(__always)
func preparePlaneLayer16(pd: PlaneData420, pool: BlockViewPool, sads: [Int]?, layer: UInt8, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int) async throws -> (PlaneData420, [BlockView], [BlockView], [BlockView]) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    
    let yColCount16 = (dx + 15) / 16
    let cbColCount16 = (cbDx + 15) / 16
    let yRowCount16 = (dy + 15) / 16
    let cbRowCount16 = (cbDy + 15) / 16
    
    async let taskBufY = { () -> ([Int16], [BlockView]) in
        let (blocks, subband) = await extractSingleTransformBlocks16(r: pd.rY, width: dx, height: dy, pool: pool) { index, view in
            if let sList = sads, index < sList.count {
                let col = index % yColCount16
                let row = index / yColCount16
                let threshold = spatialSADThreshold(baseSAD: 75, blockCol: col, blockRow: row, colCount: yColCount16, rowCount: yRowCount16)
                if sList[index] < threshold { view.clearAll() }
            }
            var v = view
            evaluateQuantizeLayer16(block: &v, qt: qtY)
        }
        return (subband, blocks)
    }()
    
    async let taskBufCb = { () -> ([Int16], [BlockView]) in
        let (blocks, subband) = await extractSingleTransformBlocks16(r: pd.rCb, width: cbDx, height: cbDy, pool: pool) { index, view in
            var sadVal = Int.max
            if let sList = sads {
                let col = index % cbColCount16
                let row = index / cbColCount16
                let lumaIdx = (row * 2) * yColCount16 + (col * 2)
                if lumaIdx < sList.count { sadVal = sList[lumaIdx] }

                let threshold = spatialSADThreshold(baseSAD: 150, blockCol: col, blockRow: row, colCount: cbColCount16, rowCount: cbRowCount16)
                if sadVal < threshold { view.clearAll() }
            }
            var v = view
            evaluateQuantizeLayer16(block: &v, qt: qtC)
        }
        return (subband, blocks)
    }()
    
    async let taskBufCr = { () -> ([Int16], [BlockView]) in
        let (blocks, subband) = await extractSingleTransformBlocks16(r: pd.rCr, width: cbDx, height: cbDy, pool: pool) { index, view in
            var sadVal = Int.max
            if let sList = sads {
                let col = index % cbColCount16
                let row = index / cbColCount16
                let lumaIdx = (row * 2) * yColCount16 + (col * 2)
                if lumaIdx < sList.count { sadVal = sList[lumaIdx] }

                let threshold = spatialSADThreshold(baseSAD: 150, blockCol: col, blockRow: row, colCount: cbColCount16, rowCount: cbRowCount16)
                if sadVal < threshold { view.clearAll() }
            }
            var v = view
            evaluateQuantizeLayer16(block: &v, qt: qtC)
        }
        return (subband, blocks)
    }()

    let (subY, yBlocks) = await taskBufY
    let (subCb, cbBlocks) = await taskBufCb
    let (subCr, crBlocks) = await taskBufCr

    let subPlane = PlaneData420(width: (dx + 1) / 2, height: (dy + 1) / 2, y: subY, cb: subCb, cr: subCr)
    return (subPlane, yBlocks, cbBlocks, crBlocks)
}

@inline(__always)
func entropyEncodeLayer32(dx: Int, dy: Int, layer: UInt8, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, isPFrame: Bool = false, yBlocks: inout [BlockView], cbBlocks: inout [BlockView], crBlocks: inout [BlockView], parentYBlocks: [BlockView]?, parentCbBlocks: [BlockView]?, parentCrBlocks: [BlockView]?) -> [UInt8] {
    // Layer2 (32x32) contains the highest-frequency DWT subbands with the
    // lowest CSF sensitivity. P-frame residuals at this level can be zeroed
    // more aggressively (threshold=4) than Layer1 (threshold=2) without
    // perceptible quality loss.
    let safeThresholdY = max(0, zeroThreshold - (Int(qtY.step) / 2))
    let safeThresholdC = max(0, zeroThreshold - (Int(qtC.step) / 2))
    
    let colCountY = (dx + 31) / 32
    let rowCountY = (dy + 31) / 32
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let colCountC = (cbDx + 31) / 32
    let rowCountC = (cbDy + 31) / 32
    
    let bufY = encodePlaneSubbands32(blocks: &yBlocks, zeroThreshold: safeThresholdY, parentBlocks: parentYBlocks, colCount: colCountY, rowCount: rowCountY)
    let bufCb = encodePlaneSubbands32(blocks: &cbBlocks, zeroThreshold: safeThresholdC, parentBlocks: parentCbBlocks, colCount: colCountC, rowCount: rowCountC)
    let bufCr = encodePlaneSubbands32(blocks: &crBlocks, zeroThreshold: safeThresholdC, parentBlocks: parentCrBlocks, colCount: colCountC, rowCount: rowCountC)
    
    debugLog("  [Layer \\(layer)] qtY=\\(qtY.step), qtC=\\(qtC.step) Y=\\(bufY.count) Cb=\\(bufCb.count) Cr=\\(bufCr.count) bytes")
    
    var out: [UInt8] = []
    appendUInt16BE(&out, UInt16(qtY.step))
    appendUInt16BE(&out, UInt16(qtC.step))
    
    appendUInt32BE(&out, UInt32(bufY.count))
    out.append(contentsOf: bufY)
    
    appendUInt32BE(&out, UInt32(bufCb.count))
    out.append(contentsOf: bufCb)
    
    appendUInt32BE(&out, UInt32(bufCr.count))
    out.append(contentsOf: bufCr)
    
    return out
}

@inline(__always)
func entropyEncodeLayer16(dx: Int, dy: Int, layer: UInt8, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, isPFrame: Bool = false, yBlocks: inout [BlockView], cbBlocks: inout [BlockView], crBlocks: inout [BlockView], parentYBlocks: [BlockView]?, parentCbBlocks: [BlockView]?, parentCrBlocks: [BlockView]?) -> [UInt8] {
    let safeThresholdY = max(0, zeroThreshold - (Int(qtY.step) / 2))
    let safeThresholdC = max(0, zeroThreshold - (Int(qtC.step) / 2))
    
    let colCountY = (dx + 15) / 16
    let rowCountY = (dy + 15) / 16
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let colCountC = (cbDx + 15) / 16
    let rowCountC = (cbDy + 15) / 16
    
    let bufY = encodePlaneSubbands16(blocks: &yBlocks, zeroThreshold: safeThresholdY, parentBlocks: parentYBlocks, colCount: colCountY, rowCount: rowCountY)
    let bufCb = encodePlaneSubbands16(blocks: &cbBlocks, zeroThreshold: safeThresholdC, parentBlocks: parentCbBlocks, colCount: colCountC, rowCount: rowCountC)
    let bufCr = encodePlaneSubbands16(blocks: &crBlocks, zeroThreshold: safeThresholdC, parentBlocks: parentCrBlocks, colCount: colCountC, rowCount: rowCountC)
    
    debugLog("  [Layer \\(layer)] qtY=\\(qtY.step), qtC=\\(qtC.step) Y=\\(bufY.count) Cb=\\(bufCb.count) Cr=\\(bufCr.count) bytes")
    
    var out: [UInt8] = []
    appendUInt16BE(&out, UInt16(qtY.step))
    appendUInt16BE(&out, UInt16(qtC.step))
    
    appendUInt32BE(&out, UInt32(bufY.count))
    out.append(contentsOf: bufY)
    
    appendUInt32BE(&out, UInt32(bufCb.count))
    out.append(contentsOf: bufCb)
    
    appendUInt32BE(&out, UInt32(bufCr.count))
    out.append(contentsOf: bufCr)
    
    return out
}

@inline(__always)
func reconstructPlaneBase8(blocks: [BlockView], width: Int, height: Int, qt: QuantizationTable, pool: BlockViewPool) -> [Int16] {
    let colCount = (width + 7) / 8
    let rowCount = (height + 7) / 8
    var plane = pool.getInt16(count: width * height)
    plane.withUnsafeMutableBufferPointer { dstBuf in
        guard let dstBase = dstBuf.baseAddress else { return }
        var idx = 0
        for row in 0..<rowCount {
            let startY = row * 8
            let validEndY = min(height, startY + 8)
            let loopH = validEndY - startY
            let isEdgeY = (loopH < 8)
            
            for col in 0..<colCount {
                let startX = col * 8
                let validEndX = min(width, startX + 8)
                let loopW = validEndX - startX
                let isEdgeX = (loopW < 8)
                
                let blk = blocks[idx]
                idx += 1
                
                let view = blk
                let base = view.base
                let llView = BlockView(base: base, width: 4, height: 4, stride: 8)
                let hlView = BlockView(base: base.advanced(by: 4), width: 4, height: 4, stride: 8)
                let lhView = BlockView(base: base.advanced(by: 32), width: 4, height: 4, stride: 8)
                let hhView = BlockView(base: base.advanced(by: 36), width: 4, height: 4, stride: 8)
                dequantizeSIMD4(llView, q: qt.qLow)
                dequantizeSIMDSignedMapping4(hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping4(lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping4(hhView, q: qt.qHigh)
                invDwt2d_8(view)
                            
                if isEdgeY != true && isEdgeX != true {
                    let v = blk
                    for h in 0..<8 {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: 8)
                    }
                } else if loopH > 0 && loopW > 0 {
                    let v = blk
                    for h in 0..<loopH {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: loopW)
                    }
                }
            }
        }
    }
    return plane
}

@inline(__always)
func reconstructPlaneLayer32Y(blocks: [BlockView], prevImg: Image16, width: Int, height: Int, qt: QuantizationTable, pool: BlockViewPool) -> [Int16] {
    let colCount = (width + 31) / 32
    let rowCount = (height + 31) / 32
    var plane = pool.getInt16(count: width * height)
    plane.withUnsafeMutableBufferPointer { dstBuf in
        guard let dstBase = dstBuf.baseAddress else { return }
        var idx = 0
        for row in 0..<rowCount {
            let startY = row * 32
            let validEndY = min(height, startY + 32)
            let loopH = validEndY - startY
            let isEdgeY = (loopH < 32)
            
            for col in 0..<colCount {
                let startX = col * 32
                let validEndX = min(width, startX + 32)
                let loopW = validEndX - startX
                let isEdgeX = (loopW < 32)
                
                let blk = blocks[idx]
                idx += 1
                
                let llX = startX / 2
                let llY = startY / 2
                prevImg.readY(x: llX, y: llY, size: 16, into: blk)
                                        
                let view = blk
                let base = view.base
                let hlView = BlockView(base: base.advanced(by: 16), width: 16, height: 16, stride: 32)
                let lhView = BlockView(base: base.advanced(by: 16 * 32), width: 16, height: 16, stride: 32)
                let hhView = BlockView(base: base.advanced(by: 16 * 32 + 16), width: 16, height: 16, stride: 32)
                dequantizeSIMDSignedMapping16(hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping16(lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping16(hhView, q: qt.qHigh)
                invDwt2d_32(view)
                            
                if isEdgeY != true && isEdgeX != true {
                    let v = blk
                    for h in 0..<32 {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: 32)
                    }
                } else if loopH > 0 && loopW > 0 {
                    let v = blk
                    for h in 0..<loopH {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: loopW)
                    }
                }
            }
        }
    }
    return plane
}

@inline(__always)
func reconstructPlaneLayer32Cb(blocks: [BlockView], prevImg: Image16, width: Int, height: Int, qt: QuantizationTable, pool: BlockViewPool) -> [Int16] {
    let colCount = (width + 31) / 32
    let rowCount = (height + 31) / 32
    var plane = pool.getInt16(count: width * height)
    plane.withUnsafeMutableBufferPointer { dstBuf in
        guard let dstBase = dstBuf.baseAddress else { return }
        var idx = 0
        for row in 0..<rowCount {
            let startY = row * 32
            let validEndY = min(height, startY + 32)
            let loopH = validEndY - startY
            let isEdgeY = (loopH < 32)
            
            for col in 0..<colCount {
                let startX = col * 32
                let validEndX = min(width, startX + 32)
                let loopW = validEndX - startX
                let isEdgeX = (loopW < 32)
                
                let blk = blocks[idx]
                idx += 1
                
                let llX = startX / 2
                let llY = startY / 2
                prevImg.readCb(x: llX, y: llY, size: 16, into: blk)
                                        
                let view = blk
                let base = view.base
                let hlView = BlockView(base: base.advanced(by: 16), width: 16, height: 16, stride: 32)
                let lhView = BlockView(base: base.advanced(by: 16 * 32), width: 16, height: 16, stride: 32)
                let hhView = BlockView(base: base.advanced(by: 16 * 32 + 16), width: 16, height: 16, stride: 32)
                dequantizeSIMDSignedMapping16(hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping16(lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping16(hhView, q: qt.qHigh)
                invDwt2d_32(view)
                            
                if isEdgeY != true && isEdgeX != true {
                    let v = blk
                    for h in 0..<32 {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: 32)
                    }
                } else if loopH > 0 && loopW > 0 {
                    let v = blk
                    for h in 0..<loopH {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: loopW)
                    }
                }
            }
        }
    }
    return plane
}

@inline(__always)
func reconstructPlaneLayer32Cr(blocks: [BlockView], prevImg: Image16, width: Int, height: Int, qt: QuantizationTable, pool: BlockViewPool) -> [Int16] {
    let colCount = (width + 31) / 32
    let rowCount = (height + 31) / 32
    var plane = pool.getInt16(count: width * height)
    plane.withUnsafeMutableBufferPointer { dstBuf in
        guard let dstBase = dstBuf.baseAddress else { return }
        var idx = 0
        for row in 0..<rowCount {
            let startY = row * 32
            let validEndY = min(height, startY + 32)
            let loopH = validEndY - startY
            let isEdgeY = (loopH < 32)
            
            for col in 0..<colCount {
                let startX = col * 32
                let validEndX = min(width, startX + 32)
                let loopW = validEndX - startX
                let isEdgeX = (loopW < 32)
                
                let blk = blocks[idx]
                idx += 1
                
                let llX = startX / 2
                let llY = startY / 2
                prevImg.readCr(x: llX, y: llY, size: 16, into: blk)
                                        
                let view = blk
                let base = view.base
                let hlView = BlockView(base: base.advanced(by: 16), width: 16, height: 16, stride: 32)
                let lhView = BlockView(base: base.advanced(by: 16 * 32), width: 16, height: 16, stride: 32)
                let hhView = BlockView(base: base.advanced(by: 16 * 32 + 16), width: 16, height: 16, stride: 32)
                dequantizeSIMDSignedMapping16(hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping16(lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping16(hhView, q: qt.qHigh)
                invDwt2d_32(view)
                            
                if isEdgeY != true && isEdgeX != true {
                    let v = blk
                    for h in 0..<32 {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: 32)
                    }
                } else if loopH > 0 && loopW > 0 {
                    let v = blk
                    for h in 0..<loopH {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: loopW)
                    }
                }
            }
        }
    }
    return plane
}

@inline(__always)
func reconstructPlaneLayer16Y(blocks: [BlockView], prevImg: Image16, width: Int, height: Int, qt: QuantizationTable, pool: BlockViewPool) -> [Int16] {
    let colCount = (width + 15) / 16
    let rowCount = (height + 15) / 16
    var plane = pool.getInt16(count: width * height)
    plane.withUnsafeMutableBufferPointer { dstBuf in
        guard let dstBase = dstBuf.baseAddress else { return }
        var idx = 0
        for row in 0..<rowCount {
            let startY = row * 16
            let validEndY = min(height, startY + 16)
            let loopH = validEndY - startY
            let isEdgeY = (loopH < 16)
            
            for col in 0..<colCount {
                let startX = col * 16
                let validEndX = min(width, startX + 16)
                let loopW = validEndX - startX
                let isEdgeX = (loopW < 16)
                
                let blk = blocks[idx]
                idx += 1
                
                let llX = startX / 2
                let llY = startY / 2
                prevImg.readY(x: llX, y: llY, size: 8, into: blk)
                                        
                let view = blk
                let base = view.base
                let hlView = BlockView(base: base.advanced(by: 8), width: 8, height: 8, stride: 16)
                let lhView = BlockView(base: base.advanced(by: 8 * 16), width: 8, height: 8, stride: 16)
                let hhView = BlockView(base: base.advanced(by: 8 * 16 + 8), width: 8, height: 8, stride: 16)
                dequantizeSIMDSignedMapping8(hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping8(lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping8(hhView, q: qt.qHigh)
                invDwt2d_16(view)
                            
                if isEdgeY != true && isEdgeX != true {
                    let v = blk
                    for h in 0..<16 {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: 16)
                    }
                } else if loopH > 0 && loopW > 0 {
                    let v = blk
                    for h in 0..<loopH {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: loopW)
                    }
                }
            }
        }
    }
    return plane
}

@inline(__always)
func reconstructPlaneLayer16Cb(blocks: [BlockView], prevImg: Image16, width: Int, height: Int, qt: QuantizationTable, pool: BlockViewPool) -> [Int16] {
    let colCount = (width + 15) / 16
    let rowCount = (height + 15) / 16
    var plane = pool.getInt16(count: width * height)
    plane.withUnsafeMutableBufferPointer { dstBuf in
        guard let dstBase = dstBuf.baseAddress else { return }
        var idx = 0
        for row in 0..<rowCount {
            let startY = row * 16
            let validEndY = min(height, startY + 16)
            let loopH = validEndY - startY
            let isEdgeY = (loopH < 16)
            
            for col in 0..<colCount {
                let startX = col * 16
                let validEndX = min(width, startX + 16)
                let loopW = validEndX - startX
                let isEdgeX = (loopW < 16)
                
                let blk = blocks[idx]
                idx += 1
                
                let llX = startX / 2
                let llY = startY / 2
                prevImg.readCb(x: llX, y: llY, size: 8, into: blk)
                                        
                let view = blk
                let base = view.base
                let hlView = BlockView(base: base.advanced(by: 8), width: 8, height: 8, stride: 16)
                let lhView = BlockView(base: base.advanced(by: 8 * 16), width: 8, height: 8, stride: 16)
                let hhView = BlockView(base: base.advanced(by: 8 * 16 + 8), width: 8, height: 8, stride: 16)
                dequantizeSIMDSignedMapping8(hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping8(lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping8(hhView, q: qt.qHigh)
                invDwt2d_16(view)
                            
                if isEdgeY != true && isEdgeX != true {
                    let v = blk
                    for h in 0..<16 {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: 16)
                    }
                } else if loopH > 0 && loopW > 0 {
                    let v = blk
                    for h in 0..<loopH {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: loopW)
                    }
                }
            }
        }
    }
    return plane
}

@inline(__always)
func reconstructPlaneLayer16Cr(blocks: [BlockView], prevImg: Image16, width: Int, height: Int, qt: QuantizationTable, pool: BlockViewPool) -> [Int16] {
    let colCount = (width + 15) / 16
    let rowCount = (height + 15) / 16
    var plane = pool.getInt16(count: width * height)
    plane.withUnsafeMutableBufferPointer { dstBuf in
        guard let dstBase = dstBuf.baseAddress else { return }
        var idx = 0
        for row in 0..<rowCount {
            let startY = row * 16
            let validEndY = min(height, startY + 16)
            let loopH = validEndY - startY
            let isEdgeY = (loopH < 16)
            
            for col in 0..<colCount {
                let startX = col * 16
                let validEndX = min(width, startX + 16)
                let loopW = validEndX - startX
                let isEdgeX = (loopW < 16)
                
                let blk = blocks[idx]
                idx += 1
                
                let llX = startX / 2
                let llY = startY / 2
                prevImg.readCr(x: llX, y: llY, size: 8, into: blk)
                                        
                let view = blk
                let base = view.base
                let hlView = BlockView(base: base.advanced(by: 8), width: 8, height: 8, stride: 16)
                let lhView = BlockView(base: base.advanced(by: 8 * 16), width: 8, height: 8, stride: 16)
                let hhView = BlockView(base: base.advanced(by: 8 * 16 + 8), width: 8, height: 8, stride: 16)
                dequantizeSIMDSignedMapping8(hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping8(lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping8(hhView, q: qt.qHigh)
                invDwt2d_16(view)
                            
                if isEdgeY != true && isEdgeX != true {
                    let v = blk
                    for h in 0..<16 {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: 16)
                    }
                } else if loopH > 0 && loopW > 0 {
                    let v = blk
                    for h in 0..<loopH {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: loopW)
                    }
                }
            }
        }
    }
    return plane
}

@inline(__always)
func encodePlaneBase8(pd: PlaneData420, pool: BlockViewPool, sads: [Int]?, layer: UInt8, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int) async throws -> ([UInt8], PlaneData420, [BlockView], [BlockView], [BlockView]) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    
    let yColCount8 = (dx + 7) / 8
    let yRowCount8 = (dy + 7) / 8
    
    async let taskBufY = { () -> ([UInt8], [Int16], [BlockView]) in
        var blocks = await extractSingleTransformBlocksBase8(r: pd.rY, width: dx, height: dy, pool: pool)
        let isIFrame = (sads == nil)
        let isPFrame = !isIFrame
        for i in blocks.indices {
            if let sList = sads, i < sList.count {
                let col = i % yColCount8
                let row = i / yColCount8
                let threshold = spatialSADThreshold(baseSAD: 150, blockCol: col, blockRow: row, colCount: yColCount8, rowCount: yRowCount8)
                if sList[i] < threshold { blocks[i].clearAll() }
            }
            evaluateQuantizeBase8(block: &blocks[i], qt: qtY)
        }
        
        // DPCM is already perfectly handled inside encodePlaneBaseSubbands8 via blockEncodeDPCM4 (MED)
        
        // P-frame Base8: apply safeThreshold to zero out imperceptible residuals
        let safeThreshold = max(0, zeroThreshold - (Int(qtY.step) / 2))
        let buf = isPFrame
            ? encodePlaneBaseSubbands8PFrame(blocks: &blocks, zeroThreshold: safeThreshold)
            : encodePlaneBaseSubbands8(blocks: &blocks, zeroThreshold: safeThreshold)
        
        let quantizedBlocks = blocks
        let reconPlane = reconstructPlaneBase8(blocks: blocks, width: dx, height: dy, qt: qtY, pool: pool)
        return (buf, reconPlane, quantizedBlocks)
    }()
    
    let lumaColCount = (dx + 7) / 8
    let chromaColCount = (cbDx + 7) / 8
    
    async let taskBufCb = { () -> ([UInt8], [Int16], [BlockView]) in
        var blocks = await extractSingleTransformBlocksBase8(r: pd.rCb, width: cbDx, height: cbDy, pool: pool)
        let isIFrame = (sads == nil)
        let isPFrame = !isIFrame
        for i in blocks.indices {
            var sadVal: Int = Int.max
            if let sList = sads {
                let r: Int = i / chromaColCount
                let c: Int = i % chromaColCount
                let r2: Int = r * 2
                let c2: Int = c * 2
                let lumaIdx: Int = r2 * lumaColCount + c2
                if lumaIdx < sList.count { sadVal = sList[lumaIdx] }
            }
            if sadVal < 75 { blocks[i].clearAll() }
            evaluateQuantizeBase8(block: &blocks[i], qt: qtC)
        }
        
        // DPCM is already perfectly handled inside encodePlaneBaseSubbands8 via blockEncodeDPCM4 (MED)
        
        let safeThreshold = max(0, zeroThreshold - (Int(qtC.step)  / 2))
        let buf = isPFrame
            ? encodePlaneBaseSubbands8PFrame(blocks: &blocks, zeroThreshold: safeThreshold)
            : encodePlaneBaseSubbands8(blocks: &blocks, zeroThreshold: safeThreshold)
        
        let quantizedBlocks = blocks
        let reconPlane = reconstructPlaneBase8(blocks: blocks, width: cbDx, height: cbDy, qt: qtC, pool: pool)
        return (buf, reconPlane, quantizedBlocks)
    }()
    
    async let taskBufCr = { () -> ([UInt8], [Int16], [BlockView]) in
        var blocks = await extractSingleTransformBlocksBase8(r: pd.rCr, width: cbDx, height: cbDy, pool: pool)
        let isIFrame = (sads == nil)
        let isPFrame = !isIFrame
        for i in blocks.indices {
            var sadVal: Int = Int.max
            if let sList = sads {
                let r: Int = i / chromaColCount
                let c: Int = i % chromaColCount
                let r2: Int = r * 2
                let c2: Int = c * 2
                let lumaIdx: Int = r2 * lumaColCount + c2
                if lumaIdx < sList.count { sadVal = sList[lumaIdx] }
            }
            if sadVal < 75 { blocks[i].clearAll() }
            evaluateQuantizeBase8(block: &blocks[i], qt: qtC)
        }
        
        // DPCM is already perfectly handled inside encodePlaneBaseSubbands8 via blockEncodeDPCM4 (MED)
        
        let safeThreshold = max(0, zeroThreshold - (Int(qtC.step) / 2))
        let buf = isPFrame
            ? encodePlaneBaseSubbands8PFrame(blocks: &blocks, zeroThreshold: safeThreshold)
            : encodePlaneBaseSubbands8(blocks: &blocks, zeroThreshold: safeThreshold)
        
        let quantizedBlocks = blocks
        let reconPlane = reconstructPlaneBase8(blocks: blocks, width: cbDx, height: cbDy, qt: qtC, pool: pool)
        return (buf, reconPlane, quantizedBlocks)
    }()

    let (bufY, reconY, base8YBlocks) = await taskBufY
    let (bufCb, reconCb, base8CbBlocks) = await taskBufCb
    let (bufCr, reconCr, base8CrBlocks) = await taskBufCr
    
    let reconstructed = PlaneData420(width: dx, height: dy, y: reconY, cb: reconCb, cr: reconCr)
    
    debugLog("  [Layer \(layer)/Base] Y=\(bufY.count) Cb=\(bufCb.count) Cr=\(bufCr.count) bytes")
    
    var out: [UInt8] = []
    appendUInt16BE(&out, UInt16(qtY.step))
    appendUInt16BE(&out, UInt16(qtC.step))
    
    appendUInt32BE(&out, UInt32(bufY.count))
    out.append(contentsOf: bufY)
    
    appendUInt32BE(&out, UInt32(bufCb.count))
    out.append(contentsOf: bufCb)
    
    appendUInt32BE(&out, UInt32(bufCr.count))
    out.append(contentsOf: bufCr)
    
    return (out, reconstructed, base8YBlocks, base8CbBlocks, base8CrBlocks)
}

// encoder-side reconstruction:
// rANS decode without entropy, using quantized blocks directly
@inline(__always)
func reconstructPlaneBase32(blocks: [BlockView], width: Int, height: Int, qt: QuantizationTable, pool: BlockViewPool) -> [Int16] {
    let colCount = (width + 32 - 1) / 32
    var plane = pool.getInt16(count: width * height)
    plane.withUnsafeMutableBufferPointer { dstBuf in
        guard let dstBase = dstBuf.baseAddress else { return }
        for idx in blocks.indices {
            let blk = blocks[idx]
            let row = idx / colCount
            let col = idx % colCount
            let startY = row * 32
            let startX = col * 32
            
            let view = blk
            let half = 16
            let base = view.base
            let llView = BlockView(base: base, width: half, height: half, stride: 32)
            let hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 32)
            let lhView = BlockView(base: base.advanced(by: half * 32), width: half, height: half, stride: 32)
            let hhView = BlockView(base: base.advanced(by: half * 32 + half), width: half, height: half, stride: 32)
            dequantizeSIMD(llView, q: qt.qLow)
            dequantizeSIMDSignedMapping(hlView, q: qt.qMid)
            dequantizeSIMDSignedMapping(lhView, q: qt.qMid)
            dequantizeSIMDSignedMapping(hhView, q: qt.qHigh)
            invDwt2d_32(view)
                    
            let validEndY = min(height, startY + 32)
            let validEndX = min(width, startX + 32)
            let loopH = validEndY - startY
            let loopW = validEndX - startX
            
            if 0 < loopH && 0 < loopW {
                let v = blk
                for h in 0..<loopH {
                    let srcPtr = v.rowPointer(y: h)
                    let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                    destPtr.update(from: srcPtr, count: loopW)
                }
            }
        }
    }
    return plane
}

@inline(__always)
func encodePlaneBase32(pd: PlaneData420, pool: BlockViewPool, predictedPd: PlaneData420?, layer: UInt8, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int) async throws -> ([UInt8], PlaneData420) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    
    async let taskBufY = { () -> ([UInt8], [Int16]) in
        var blocks = await extractSingleTransformBlocksBase32(r: pd.rY, width: dx, height: dy, pool: pool)
        if let pPd = predictedPd {
            var pBlocks = await extractSingleTransformBlocksBase32(r: pPd.rY, width: dx, height: dy, pool: pool)
            subtractCoeffsBase32(currBlocks: &blocks, predBlocks: &pBlocks)
        }
        for i in blocks.indices {
            evaluateQuantizeBase32(block: &blocks[i], qt: qtY)
        }
        let safeThreshold = max(0, zeroThreshold - (Int(qtY.step) / 2))
        let buf = encodePlaneBaseSubbands32(blocks: &blocks, zeroThreshold: safeThreshold)
        let reconPlane = reconstructPlaneBase32(blocks: blocks, width: dx, height: dy, qt: qtY, pool: pool)
        return (buf, reconPlane)
    }()
    
    async let taskBufCb = { () -> ([UInt8], [Int16]) in
        var blocks = await extractSingleTransformBlocksBase32(r: pd.rCb, width: cbDx, height: cbDy, pool: pool)
        if let pPd = predictedPd {
            var pBlocks = await extractSingleTransformBlocksBase32(r: pPd.rCb, width: cbDx, height: cbDy, pool: pool)
            subtractCoeffsBase32(currBlocks: &blocks, predBlocks: &pBlocks)
        }
        for i in blocks.indices {
            evaluateQuantizeBase32(block: &blocks[i], qt: qtC)
        }
        let safeThreshold = max(0, zeroThreshold - (Int(qtC.step)  / 2))
        let buf = encodePlaneBaseSubbands32(blocks: &blocks, zeroThreshold: safeThreshold)
        let reconPlane = reconstructPlaneBase32(blocks: blocks, width: cbDx, height: cbDy, qt: qtC, pool: pool)
        return (buf, reconPlane)
    }()
    
    async let taskBufCr = { () -> ([UInt8], [Int16]) in
        var blocks = await extractSingleTransformBlocksBase32(r: pd.rCr, width: cbDx, height: cbDy, pool: pool)
        if let pPd = predictedPd {
            var pBlocks = await extractSingleTransformBlocksBase32(r: pPd.rCr, width: cbDx, height: cbDy, pool: pool)
            subtractCoeffsBase32(currBlocks: &blocks, predBlocks: &pBlocks)
        }
        for i in blocks.indices {
            evaluateQuantizeBase32(block: &blocks[i], qt: qtC)
        }
        let safeThreshold = max(0, zeroThreshold - (Int(qtC.step)  / 2))
        let buf = encodePlaneBaseSubbands32(blocks: &blocks, zeroThreshold: safeThreshold)
        let reconPlane = reconstructPlaneBase32(blocks: blocks, width: cbDx, height: cbDy, qt: qtC, pool: pool)
        return (buf, reconPlane)
    }()

    let (bufY, reconY) = await taskBufY
    let (bufCb, reconCb) = await taskBufCb
    let (bufCr, reconCr) = await taskBufCr
    
    var mutReconY = reconY
    var mutReconCb = reconCb
    var mutReconCr = reconCr
    
    applyDeblockingFilter(plane: &mutReconY, width: dx, height: dy, blockSize: 32, qStep: Int(qtY.step))
    applyDeblockingFilter(plane: &mutReconCb, width: cbDx, height: cbDy, blockSize: 32, qStep: Int(qtC.step))
    applyDeblockingFilter(plane: &mutReconCr, width: cbDx, height: cbDy, blockSize: 32, qStep: Int(qtC.step))
    
    let reconstructed = PlaneData420(width: dx, height: dy, y: mutReconY, cb: mutReconCb, cr: mutReconCr)
    
    debugLog("  [Layer \(layer)/Base] Y=\(bufY.count) Cb=\(bufCb.count) Cr=\(bufCr.count) bytes")
    
    var out: [UInt8] = []
    appendUInt16BE(&out, UInt16(qtY.step))
    appendUInt16BE(&out, UInt16(qtC.step))
    
    appendUInt32BE(&out, UInt32(bufY.count))
    out.append(contentsOf: bufY)
    
    appendUInt32BE(&out, UInt32(bufCb.count))
    out.append(contentsOf: bufCb)
    
    appendUInt32BE(&out, UInt32(bufCr.count))
    out.append(contentsOf: bufCr)
    
    return (out, reconstructed)
}

@inline(__always)
func encodeSpatialLayers(pd: PlaneData420, pool: BlockViewPool, maxbitrate: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, roundOffset: Int) async throws -> ([UInt8], PlaneData420) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    
    let qtY2 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 2)
    let qtC2 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 2)
    let qtY1 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 1)
    let qtC1 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 1)
    let qtY0 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 0)
    let qtC0 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 0)
    
    let resPd = PlaneData420(width: dx, height: dy, y: pd.y, cb: pd.cb, cr: pd.cr)
    let isPFrame = false
    
    var (sub2, l2yBlocks, l2cbBlocks, l2crBlocks) = try await preparePlaneLayer32(pd: resPd, pool: pool, sads: nil, layer: 2, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold)
    var (sub1, l1yBlocks, l1cbBlocks, l1crBlocks) = try await preparePlaneLayer16(pd: sub2, pool: pool, sads: nil, layer: 1, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold)
    let (layer0, baseRecon, base8YBlocks, base8CbBlocks, base8CrBlocks) = try await encodePlaneBase8(pd: sub1, pool: pool, sads: nil, layer: 0, qtY: qtY0, qtC: qtC0, zeroThreshold: zeroThreshold)
    
    let baseImg = Image16(width: baseRecon.width, height: baseRecon.height, y: baseRecon.y, cb: baseRecon.cb, cr: baseRecon.cr)
    
    let l1dx = sub2.width
    let l1dy = sub2.height
    let l1cbDx = ((l1dx + 1) / 2)
    let l1cbDy = ((l1dy + 1) / 2)
    let layer1 = entropyEncodeLayer16(dx: sub2.width, dy: sub2.height, layer: 1, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold, isPFrame: isPFrame, yBlocks: &l1yBlocks, cbBlocks: &l1cbBlocks, crBlocks: &l1crBlocks, parentYBlocks: base8YBlocks, parentCbBlocks: base8CbBlocks, parentCrBlocks: base8CrBlocks)
    
    let mutReconL1Y = reconstructPlaneLayer16Y(blocks: l1yBlocks, prevImg: baseImg, width: l1dx, height: l1dy, qt: qtY1, pool: pool)
    let mutReconL1Cb = reconstructPlaneLayer16Cb(blocks: l1cbBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    let mutReconL1Cr = reconstructPlaneLayer16Cr(blocks: l1crBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    
    let l1Img = Image16(width: l1dx, height: l1dy, y: mutReconL1Y, cb: mutReconL1Cb, cr: mutReconL1Cr)
    
    let layer2 = entropyEncodeLayer32(dx: pd.width, dy: pd.height, layer: 2, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold, isPFrame: isPFrame, yBlocks: &l2yBlocks, cbBlocks: &l2cbBlocks, crBlocks: &l2crBlocks, parentYBlocks: l1yBlocks, parentCbBlocks: l1cbBlocks, parentCrBlocks: l1crBlocks)
    
    var mutReconL2Y = reconstructPlaneLayer32Y(blocks: l2yBlocks, prevImg: l1Img, width: dx, height: dy, qt: qtY2, pool: pool)
    var mutReconL2Cb = reconstructPlaneLayer32Cb(blocks: l2cbBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
    var mutReconL2Cr = reconstructPlaneLayer32Cr(blocks: l2crBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
    
    applyDeblockingFilter(plane: &mutReconL2Y, width: dx, height: dy, blockSize: 32, qStep: Int(qtY2.step))
    applyDeblockingFilter(plane: &mutReconL2Cb, width: cbDx, height: cbDy, blockSize: 32, qStep: Int(qtC2.step))
    applyDeblockingFilter(plane: &mutReconL2Cr, width: cbDx, height: cbDy, blockSize: 32, qStep: Int(qtC2.step))
    
    let reconstructed = PlaneData420(width: dx, height: dy, y: mutReconL2Y, cb: mutReconL2Cb, cr: mutReconL2Cr)
    
    debugLog("  [Summary] Layer0=\(layer0.count) Layer1=\(layer1.count) Layer2=\(layer2.count) total=\(layer0.count + layer1.count + layer2.count) bytes")
    
    var out: [UInt8] = []
    appendUInt32BE(&out, 0)
    appendUInt32BE(&out, 0)
    appendUInt32BE(&out, UInt32(layer0.count))
    out.append(contentsOf: layer0)
    appendUInt32BE(&out, UInt32(layer1.count))
    out.append(contentsOf: layer1)
    appendUInt32BE(&out, UInt32(layer2.count))
    out.append(contentsOf: layer2)
    
    return (out, reconstructed)
}

@inline(__always)
func encodeSpatialLayers(pd: PlaneData420, pool: BlockViewPool, predictedPd: PlaneData420, maxbitrate: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, roundOffset: Int) async throws -> ([UInt8], PlaneData420) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    
    let qtY2 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 2)
    let qtC2 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 2)
    let qtY1 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 1)
    let qtC1 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 1)
    let qtY0 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 0)
    let qtC0 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 0)
    
    let (mvs, sads) = await computeMotionVectors(curr: pd, prev: predictedPd, pool: pool, roundOffset: roundOffset)
    
    var mutPdY = pool.getInt16(count: pd.y.count)
    var mutPdCb = pool.getInt16(count: pd.cb.count)
    var mutPdCr = pool.getInt16(count: pd.cr.count)
    mutPdY.withUnsafeMutableBufferPointer { dst in pd.y.withUnsafeBufferPointer({ _ = dst.update(from: $0) }) }
    mutPdCb.withUnsafeMutableBufferPointer { dst in pd.cb.withUnsafeBufferPointer({ _ = dst.update(from: $0) }) }
    mutPdCr.withUnsafeMutableBufferPointer { dst in pd.cr.withUnsafeBufferPointer({ _ = dst.update(from: $0) }) }
    
    subtractMotionCompensationPixels(plane: &mutPdY, prevPlane: predictedPd.y, mvs: mvs, width: dx, height: dy, blockSize: 32, shiftMultiplierX2: 2, roundOffset: roundOffset)
    subtractMotionCompensationPixels(plane: &mutPdCb, prevPlane: predictedPd.cb, mvs: mvs, width: cbDx, height: cbDy, blockSize: 16, shiftMultiplierX2: 1, roundOffset: roundOffset)
    subtractMotionCompensationPixels(plane: &mutPdCr, prevPlane: predictedPd.cr, mvs: mvs, width: cbDx, height: cbDy, blockSize: 16, shiftMultiplierX2: 1, roundOffset: roundOffset)
    
    let resPd = PlaneData420(width: dx, height: dy, y: mutPdY, cb: mutPdCb, cr: mutPdCr)
    let isPFrame = true
    
    var (sub2, l2yBlocks, l2cbBlocks, l2crBlocks) = try await preparePlaneLayer32(pd: resPd, pool: pool, sads: sads, layer: 2, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold)
    var (sub1, l1yBlocks, l1cbBlocks, l1crBlocks) = try await preparePlaneLayer16(pd: sub2, pool: pool, sads: sads, layer: 1, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold)
    let (layer0, baseRecon, base8YBlocks, base8CbBlocks, base8CrBlocks) = try await encodePlaneBase8(pd: sub1, pool: pool, sads: sads, layer: 0, qtY: qtY0, qtC: qtC0, zeroThreshold: zeroThreshold)
    
    let baseImg = Image16(width: baseRecon.width, height: baseRecon.height, y: baseRecon.y, cb: baseRecon.cb, cr: baseRecon.cr)
    
    let l1dx = sub2.width
    let l1dy = sub2.height
    let l1cbDx = ((l1dx + 1) / 2)
    let l1cbDy = ((l1dy + 1) / 2)
    let layer1 = entropyEncodeLayer16(dx: sub2.width, dy: sub2.height, layer: 1, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold, isPFrame: isPFrame, yBlocks: &l1yBlocks, cbBlocks: &l1cbBlocks, crBlocks: &l1crBlocks, parentYBlocks: base8YBlocks, parentCbBlocks: base8CbBlocks, parentCrBlocks: base8CrBlocks)
    
    let mutReconL1Y = reconstructPlaneLayer16Y(blocks: l1yBlocks, prevImg: baseImg, width: l1dx, height: l1dy, qt: qtY1, pool: pool)
    let mutReconL1Cb = reconstructPlaneLayer16Cb(blocks: l1cbBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    let mutReconL1Cr = reconstructPlaneLayer16Cr(blocks: l1crBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    
    let l1Img = Image16(width: l1dx, height: l1dy, y: mutReconL1Y, cb: mutReconL1Cb, cr: mutReconL1Cr)
    
    let layer2 = entropyEncodeLayer32(dx: pd.width, dy: pd.height, layer: 2, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold, isPFrame: isPFrame, yBlocks: &l2yBlocks, cbBlocks: &l2cbBlocks, crBlocks: &l2crBlocks, parentYBlocks: l1yBlocks, parentCbBlocks: l1cbBlocks, parentCrBlocks: l1crBlocks)
    
    var mutReconL2Y = reconstructPlaneLayer32Y(blocks: l2yBlocks, prevImg: l1Img, width: dx, height: dy, qt: qtY2, pool: pool)
    var mutReconL2Cb = reconstructPlaneLayer32Cb(blocks: l2cbBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
    var mutReconL2Cr = reconstructPlaneLayer32Cr(blocks: l2crBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
    
    applyMotionCompensationPixels(plane: &mutReconL2Y, prevPlane: predictedPd.y, mvs: mvs, width: dx, height: dy, blockSize: 32, shiftMultiplierX2: 2, roundOffset: roundOffset)
    applyMotionCompensationPixels(plane: &mutReconL2Cb, prevPlane: predictedPd.cb, mvs: mvs, width: cbDx, height: cbDy, blockSize: 16, shiftMultiplierX2: 1, roundOffset: roundOffset)
    applyMotionCompensationPixels(plane: &mutReconL2Cr, prevPlane: predictedPd.cr, mvs: mvs, width: cbDx, height: cbDy, blockSize: 16, shiftMultiplierX2: 1, roundOffset: roundOffset)
    
    applyDeblockingFilter(plane: &mutReconL2Y, width: dx, height: dy, blockSize: 32, qStep: Int(qtY2.step))
    applyDeblockingFilter(plane: &mutReconL2Cb, width: cbDx, height: cbDy, blockSize: 32, qStep: Int(qtC2.step))
    applyDeblockingFilter(plane: &mutReconL2Cr, width: cbDx, height: cbDy, blockSize: 32, qStep: Int(qtC2.step))
    
    let reconstructed = PlaneData420(width: dx, height: dy, y: mutReconL2Y, cb: mutReconL2Cb, cr: mutReconL2Cr)
    
    debugLog("  [Summary] Layer0=\(layer0.count) Layer1=\(layer1.count) Layer2=\(layer2.count) total=\(layer0.count + layer1.count + layer2.count) bytes")
    
    var out: [UInt8] = []
    
    let mvCount = mvs.count
    appendUInt32BE(&out, UInt32(mvCount))
    let mvData = encodeMVs(mvs: mvs)
    appendUInt32BE(&out, UInt32(mvData.count))
    out.append(contentsOf: mvData)

    appendUInt32BE(&out, UInt32(layer0.count))
    out.append(contentsOf: layer0)
    appendUInt32BE(&out, UInt32(layer1.count))
    out.append(contentsOf: layer1)
    appendUInt32BE(&out, UInt32(layer2.count))
    out.append(contentsOf: layer2)
    
    return (out, reconstructed)
}

@inline(__always)
func encodeSpatialLayers(pd: PlaneData420, pool: BlockViewPool, predictedPd: PlaneData420, nextPd: PlaneData420, maxbitrate: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, roundOffset: Int) async throws -> ([UInt8], PlaneData420) {
    let pPd = predictedPd
    let nPd = nextPd
    
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    
    let qtY2 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 2)
    let qtC2 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 2)
    let qtY1 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 1)
    let qtC1 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 1)
    let qtY0 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 0)
    let qtC0 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 0)
    
    // bidirectional MV calculation: search MVs for both forward and backward and select the one with the smaller SAD for each block
    let (mvs, sads, refDirs) = await computeBidirectionalMotionVectors(curr: pd, prev: pPd, next: nPd, pool: pool, roundOffset: roundOffset)
    
    // pixel level residual calculation based on reference direction
    var mutPdY = pool.getInt16(count: pd.y.count)
    var mutPdCb = pool.getInt16(count: pd.cb.count)
    var mutPdCr = pool.getInt16(count: pd.cr.count)
    mutPdY.withUnsafeMutableBufferPointer { dst in pd.y.withUnsafeBufferPointer({ _ = dst.update(from: $0) }) }
    mutPdCb.withUnsafeMutableBufferPointer { dst in pd.cb.withUnsafeBufferPointer({ _ = dst.update(from: $0) }) }
    mutPdCr.withUnsafeMutableBufferPointer { dst in pd.cr.withUnsafeBufferPointer({ _ = dst.update(from: $0) }) }
    subtractBidirectionalMotionCompensationPixels(plane: &mutPdY, prevPlane: pPd.y, nextPlane: nPd.y, mvs: mvs, refDirs: refDirs, width: dx, height: dy, blockSize: 32, shiftMultiplierX2: 2, roundOffset: roundOffset)
    subtractBidirectionalMotionCompensationPixels(plane: &mutPdCb, prevPlane: pPd.cb, nextPlane: nPd.cb, mvs: mvs, refDirs: refDirs, width: cbDx, height: cbDy, blockSize: 16, shiftMultiplierX2: 1, roundOffset: roundOffset)
    subtractBidirectionalMotionCompensationPixels(plane: &mutPdCr, prevPlane: pPd.cr, nextPlane: nPd.cr, mvs: mvs, refDirs: refDirs, width: cbDx, height: cbDy, blockSize: 16, shiftMultiplierX2: 1, roundOffset: roundOffset)
    let resPd = PlaneData420(width: dx, height: dy, y: mutPdY, cb: mutPdCb, cr: mutPdCr)

    let isPFrame = true
    
    var (sub2, l2yBlocks, l2cbBlocks, l2crBlocks) = try await preparePlaneLayer32(pd: resPd, pool: pool, sads: sads, layer: 2, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold)
    var (sub1, l1yBlocks, l1cbBlocks, l1crBlocks) = try await preparePlaneLayer16(pd: sub2, pool: pool, sads: sads, layer: 1, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold)
    let (layer0, baseRecon, base8YBlocks, base8CbBlocks, base8CrBlocks) = try await encodePlaneBase8(pd: sub1, pool: pool, sads: sads, layer: 0, qtY: qtY0, qtC: qtC0, zeroThreshold: zeroThreshold)
    
    let baseImg = Image16(width: baseRecon.width, height: baseRecon.height, y: baseRecon.y, cb: baseRecon.cb, cr: baseRecon.cr)
    
    let l1dx = sub2.width
    let l1dy = sub2.height
    let l1cbDx = ((l1dx + 1) / 2)
    let l1cbDy = ((l1dy + 1) / 2)
    let layer1 = entropyEncodeLayer16(dx: sub2.width, dy: sub2.height, layer: 1, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold, isPFrame: isPFrame, yBlocks: &l1yBlocks, cbBlocks: &l1cbBlocks, crBlocks: &l1crBlocks, parentYBlocks: base8YBlocks, parentCbBlocks: base8CbBlocks, parentCrBlocks: base8CrBlocks)
    
    let mutReconL1Y = reconstructPlaneLayer16Y(blocks: l1yBlocks, prevImg: baseImg, width: l1dx, height: l1dy, qt: qtY1, pool: pool)
    let mutReconL1Cb = reconstructPlaneLayer16Cb(blocks: l1cbBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    let mutReconL1Cr = reconstructPlaneLayer16Cr(blocks: l1crBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    
    let l1Img = Image16(width: l1dx, height: l1dy, y: mutReconL1Y, cb: mutReconL1Cb, cr: mutReconL1Cr)
    
    let layer2 = entropyEncodeLayer32(dx: pd.width, dy: pd.height, layer: 2, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold, isPFrame: isPFrame, yBlocks: &l2yBlocks, cbBlocks: &l2cbBlocks, crBlocks: &l2crBlocks, parentYBlocks: l1yBlocks, parentCbBlocks: l1cbBlocks, parentCrBlocks: l1crBlocks)
    
    let reconL2Y = reconstructPlaneLayer32Y(blocks: l2yBlocks, prevImg: l1Img, width: dx, height: dy, qt: qtY2, pool: pool)
    let reconL2Cb = reconstructPlaneLayer32Cb(blocks: l2cbBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
    let reconL2Cr = reconstructPlaneLayer32Cr(blocks: l2crBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
    
    var mutReconL2Y = reconL2Y
    var mutReconL2Cb = reconL2Cb
    var mutReconL2Cr = reconL2Cr
    
    // bidirectional motion compensation addition (reconstruction)
    applyBidirectionalMotionCompensationPixels(plane: &mutReconL2Y, prevPlane: pPd.y, nextPlane: nPd.y, mvs: mvs, refDirs: refDirs, width: dx, height: dy, blockSize: 32, shiftMultiplierX2: 2, roundOffset: roundOffset)
    applyBidirectionalMotionCompensationPixels(plane: &mutReconL2Cb, prevPlane: pPd.cb, nextPlane: nPd.cb, mvs: mvs, refDirs: refDirs, width: cbDx, height: cbDy, blockSize: 16, shiftMultiplierX2: 1, roundOffset: roundOffset)
    applyBidirectionalMotionCompensationPixels(plane: &mutReconL2Cr, prevPlane: pPd.cr, nextPlane: nPd.cr, mvs: mvs, refDirs: refDirs, width: cbDx, height: cbDy, blockSize: 16, shiftMultiplierX2: 1, roundOffset: roundOffset)
    
    applyDeblockingFilter(plane: &mutReconL2Y, width: dx, height: dy, blockSize: 32, qStep: Int(qtY2.step))
    applyDeblockingFilter(plane: &mutReconL2Cb, width: cbDx, height: cbDy, blockSize: 32, qStep: Int(qtC2.step))
    applyDeblockingFilter(plane: &mutReconL2Cr, width: cbDx, height: cbDy, blockSize: 32, qStep: Int(qtC2.step))
    
    let reconstructed = PlaneData420(width: dx, height: dy, y: mutReconL2Y, cb: mutReconL2Cb, cr: mutReconL2Cr)
    
    debugLog("  [Summary/BiDir] Layer0=\(layer0.count) Layer1=\(layer1.count) Layer2=\(layer2.count) total=\(layer0.count + layer1.count + layer2.count) bytes")
    
    var out: [UInt8] = []
    
    let mvCount = mvs.count
    appendUInt32BE(&out, UInt32(mvCount))
    
    // serialize MV data + reference direction flags
    let mvData = encodeMVs(mvs: mvs)
    appendUInt32BE(&out, UInt32(mvData.count))
    out.append(contentsOf: mvData)
    
    // reference direction flags: 1 bit per block, packed in bytes
    let refDirByteCount = (refDirs.count + 7) / 8
    appendUInt32BE(&out, UInt32(refDirByteCount))
    var refDirBuf = [UInt8](repeating: 0, count: refDirByteCount)
    for i in refDirs.indices {
        if refDirs[i] {
            refDirBuf[i / 8] |= UInt8(1 << (i % 8))
        }
    }
    out.append(contentsOf: refDirBuf)

    appendUInt32BE(&out, UInt32(layer0.count))
    out.append(contentsOf: layer0)
    
    appendUInt32BE(&out, UInt32(layer1.count))
    out.append(contentsOf: layer1)
    
    appendUInt32BE(&out, UInt32(layer2.count))
    out.append(contentsOf: layer2)
    
    return (out, reconstructed)
}

@inline(__always)
func computeMotionVectors(curr: PlaneData420, prev: PlaneData420, pool: BlockViewPool, roundOffset: Int) async -> ([MotionVector], [Int]) {
    let dx = curr.width
    let dy = curr.height
    let l1dx = (dx + 1) / 2
    let l1dy = (dy + 1) / 2
    let l0dx = (l1dx + 1) / 2
    let l0dy = (l1dy + 1) / 2
    
    let currSub2 = await extractSingleTransformSubband32(r: curr.rY, width: dx, height: dy, pool: pool)
    let currSub1 = await extractSingleTransformSubband16(r: Int16Reader(data: currSub2, width: l1dx, height: l1dy), width: l1dx, height: l1dy, pool: pool)
    let currBlocks8 = await extractSingleTransformBlocksBase8(r: Int16Reader(data: currSub1, width: l0dx, height: l0dy), width: l0dx, height: l0dy, pool: pool)

    let prevSub2 = await extractSingleTransformSubband32(r: prev.rY, width: dx, height: dy, pool: pool)
    let prevSub1 = await extractSingleTransformSubband16(r: Int16Reader(data: prevSub2, width: l1dx, height: l1dy), width: l1dx, height: l1dy, pool: pool)
    
    let targetWidth = l0dx
    let targetHeight = l0dy
    let colCount = (targetWidth + 7) / 8
    
    var mvs = [MotionVector](repeating: MotionVector(dx: 0, dy: 0), count: currBlocks8.count)
    var sads = [Int](repeating: 0, count: currBlocks8.count)
    
    let tmpC = pool.get(width: 8, height: 8)
    let tmpO = pool.get(width: 8, height: 8)
    let tmpT = pool.get(width: 8, height: 8)
    defer {
        pool.put(tmpC)
        pool.put(tmpO)
        pool.put(tmpT)
    }
    let cPtr = tmpC.base
    let oPtr = tmpO.base
    let tPtr = tmpT.base

    mvs.withUnsafeMutableBufferPointer { mvsPtr in
        sads.withUnsafeMutableBufferPointer { sadsPtr in
            for idx in currBlocks8.indices {
                let col = idx % colCount
                let row = idx / colCount
                let bx = col * 8
                let by = row * 8
                let pmv = (col > 0) ? mvsPtr[idx - 1] : MotionVector(dx: 0, dy: 0)
                let (mv, sad) = MotionEstimation.searchPixels(
                    currPlane: currSub1, prevPlane: prevSub1, 
                    cPtr: cPtr, oPtr: oPtr, tPtr: tPtr,
                    width: targetWidth, height: targetHeight, bx: bx, by: by, range: 2, pmv: pmv,
                    roundOffset: roundOffset,
                )
                mvsPtr[idx] = mv
                sadsPtr[idx] = sad
            }
        }
    }
    return (mvs, sads)
}

/// Bidirectional MV calculation: searches MV in both forward (prev) and backward (next) frames, 
/// and selects the one with smaller SAD per block.
/// - Returns: (mvs, sads, refDirs) where refDirs is the reference direction flag per block (false=forward, true=backward)
@inline(__always)
func computeBidirectionalMotionVectors(curr: PlaneData420, prev: PlaneData420, next: PlaneData420, pool: BlockViewPool, roundOffset: Int) async -> ([MotionVector], [Int], [Bool]) {
    let dx = curr.width
    let dy = curr.height
    let l1dx = (dx + 1) / 2
    let l1dy = (dy + 1) / 2
    let l0dx = (l1dx + 1) / 2
    let l0dy = (l1dy + 1) / 2
    
    // Compute DWT LL band (Base8 resolution) for current frame
    let currSub2 = await extractSingleTransformSubband32(r: curr.rY, width: dx, height: dy, pool: pool)
    let currSub1 = await extractSingleTransformSubband16(r: Int16Reader(data: currSub2, width: l1dx, height: l1dy), width: l1dx, height: l1dy, pool: pool)
    let currBlocks8 = await extractSingleTransformBlocksBase8(r: Int16Reader(data: currSub1, width: l0dx, height: l0dy), width: l0dx, height: l0dy, pool: pool)

    // Forward reference DWT LL band
    let prevSub2 = await extractSingleTransformSubband32(r: prev.rY, width: dx, height: dy, pool: pool)
    let prevSub1 = await extractSingleTransformSubband16(r: Int16Reader(data: prevSub2, width: l1dx, height: l1dy), width: l1dx, height: l1dy, pool: pool)
    
    // Backward reference DWT LL band
    let nextSub2 = await extractSingleTransformSubband32(r: next.rY, width: dx, height: dy, pool: pool)
    let nextSub1 = await extractSingleTransformSubband16(r: Int16Reader(data: nextSub2, width: l1dx, height: l1dy), width: l1dx, height: l1dy, pool: pool)
    
    let targetWidth = l0dx
    let targetHeight = l0dy
    let colCount = (targetWidth + 7) / 8
    
    var mvs = [MotionVector](repeating: MotionVector(dx: 0, dy: 0), count: currBlocks8.count)
    var sads = [Int](repeating: 0, count: currBlocks8.count)
    var refDirs = [Bool](repeating: false, count: currBlocks8.count)
    
    let tmpC = pool.get(width: 8, height: 8)
    let tmpO = pool.get(width: 8, height: 8)
    let tmpT = pool.get(width: 8, height: 8)
    defer {
        pool.put(tmpC)
        pool.put(tmpO)
        pool.put(tmpT)
    }
    let cPtr = tmpC.base
    let oPtr = tmpO.base
    let tPtr = tmpT.base

    let body: (UnsafeMutablePointer<MotionVector>, UnsafeMutablePointer<Int>, UnsafeMutablePointer<Bool>) -> Void = { mvsPtr, sadsPtr, refDirsPtr in
        for idx in currBlocks8.indices {
            let col = idx % colCount
            let row = idx / colCount
            let bx = col * 8
            let by = row * 8
            
            let pmv = (col > 0) ? mvsPtr[idx - 1] : MotionVector(dx: 0, dy: 0)
            
            let (mvPrev, sadPrev) = MotionEstimation.searchPixels(
                currPlane: currSub1, prevPlane: prevSub1,
                cPtr: cPtr, oPtr: oPtr, tPtr: tPtr,
                width: targetWidth, height: targetHeight, bx: bx, by: by, range: 2, pmv: pmv, roundOffset: roundOffset
            )
            
            let (mvNext, sadNext) = MotionEstimation.searchPixels(
                currPlane: currSub1, prevPlane: nextSub1,
                cPtr: cPtr, oPtr: oPtr, tPtr: tPtr,
                width: targetWidth, height: targetHeight, bx: bx, by: by, range: 2, pmv: pmv, roundOffset: roundOffset
            )
            
            var bestMv = mvPrev
            var bestSad = sadPrev
            var dir = false
            
            if sadNext < sadPrev {
                bestMv = mvNext
                bestSad = sadNext
                dir = true
            } else if sadNext == sadPrev && (mvNext.dy * mvNext.dy + mvNext.dx * mvNext.dx) < (mvPrev.dy * mvPrev.dy + mvPrev.dx * mvPrev.dx) {
                bestMv = mvNext
                bestSad = sadNext
                dir = true
            }
            
            mvsPtr[idx] = bestMv
            sadsPtr[idx] = bestSad
            refDirsPtr[idx] = dir
        }
    }
    withUnsafePointers(mut: &mvs, mut: &sads, mut: &refDirs, body)
    return (mvs, sads, refDirs)
}

/// bidirectional motion compensation pixel subtraction
/// fractHalf branch is hoisted to block level to eliminate per-pixel branch prediction misses
@inline(__always)
func subtractBidirectionalMotionCompensationPixels(plane: inout [Int16], prevPlane: [Int16], nextPlane: [Int16], mvs: [MotionVector], refDirs: [Bool], width: Int, height: Int, blockSize: Int, shiftMultiplierX2: Int, roundOffset: Int) {
    let colCount = (width + blockSize - 1) / blockSize
    let rowCount = (height + blockSize - 1) / blockSize
    let body: (UnsafePointer<Int16>, UnsafePointer<Int16>, UnsafeMutablePointer<Int16>) -> Void = { prevBase, nextBase, dstBase in
        for row in 0..<rowCount {
            for col in 0..<colCount {
                let mvIndex = min(row * colCount + col, mvs.count - 1)
                let mv = mvs[mvIndex]
                let isBackward = (mvIndex < refDirs.count) ? refDirs[mvIndex] : false
                let srcBase = isBackward ? nextBase : prevBase
                let blockX = col * blockSize
                let blockY = row * blockSize
                let rawShiftX = Int(mv.dx) * shiftMultiplierX2
                let rawShiftY = Int(mv.dy) * shiftMultiplierX2
                let shiftX = rawShiftX >> 2
                let fractHalfX = (rawShiftX & 2) >> 1
                let shiftY = rawShiftY >> 2
                let fractHalfY = (rawShiftY & 2) >> 1
                let bw = min(blockSize, width - blockX)
                let bh = min(blockSize, height - blockY)

                if fractHalfX == 0 && fractHalfY == 0 {
                    // integer pixel path: no interpolation needed
                    for y in 0..<bh {
                        let dstY = blockY + y
                        let srcY = blockY + shiftY + y
                        let safeSrcY = max(0, min(srcY, height - 1))
                        let dstPtr = dstBase.advanced(by: dstY * width + blockX)
                        let srcRow = srcBase.advanced(by: safeSrcY * width)
                        for x in 0..<bw {
                            let sx = max(0, min(blockX + shiftX + x, width - 1))
                            dstPtr[x] = dstPtr[x] &- srcRow[sx]
                        }
                    }
                } else if fractHalfY == 0 {
                    // horizontal half-pel only
                    for y in 0..<bh {
                        let dstY = blockY + y
                        let srcY = blockY + shiftY + y
                        let safeSrcY = max(0, min(srcY, height - 1))
                        let dstPtr = dstBase.advanced(by: dstY * width + blockX)
                        let srcRow0 = srcBase.advanced(by: safeSrcY * width)
                        for x in 0..<bw {
                            let srcX = blockX + shiftX + x
                            let sx0 = max(0, min(srcX, width - 1))
                            let sx1 = max(0, min(srcX + 1, width - 1))
                            dstPtr[x] = dstPtr[x] &- Int16((Int(srcRow0[sx0]) + Int(srcRow0[sx1]) + 1) >> 1)
                        }
                    }
                } else if fractHalfX == 0 {
                    // vertical half-pel only
                    for y in 0..<bh {
                        let dstY = blockY + y
                        let srcY = blockY + shiftY + y
                        let safeSrcY0 = max(0, min(srcY, height - 1))
                        let safeSrcY1 = max(0, min(srcY + 1, height - 1))
                        let dstPtr = dstBase.advanced(by: dstY * width + blockX)
                        let srcRow0 = srcBase.advanced(by: safeSrcY0 * width)
                        let srcRow1 = srcBase.advanced(by: safeSrcY1 * width)
                        for x in 0..<bw {
                            let sx = max(0, min(blockX + shiftX + x, width - 1))
                            dstPtr[x] = dstPtr[x] &- Int16((Int(srcRow0[sx]) + Int(srcRow1[sx]) + 1) >> 1)
                        }
                    }
                } else {
                    // bilinear (both half-pel)
                    for y in 0..<bh {
                        let dstY = blockY + y
                        let srcY = blockY + shiftY + y
                        let safeSrcY0 = max(0, min(srcY, height - 1))
                        let safeSrcY1 = max(0, min(srcY + 1, height - 1))
                        let dstPtr = dstBase.advanced(by: dstY * width + blockX)
                        let srcRow0 = srcBase.advanced(by: safeSrcY0 * width)
                        let srcRow1 = srcBase.advanced(by: safeSrcY1 * width)
                        for x in 0..<bw {
                            let srcX = blockX + shiftX + x
                            let sx0 = max(0, min(srcX, width - 1))
                            let sx1 = max(0, min(srcX + 1, width - 1))
                            dstPtr[x] = dstPtr[x] &- Int16((Int(srcRow0[sx0]) + Int(srcRow0[sx1]) + Int(srcRow1[sx0]) + Int(srcRow1[sx1]) + 2) >> 2)
                        }
                    }
                }
            }
        }
    }
    withUnsafePointers(prevPlane, nextPlane, mut: &plane, body)
}

/// Pixel addition for bidirectional motion compensation based on reference direction flag (for decoder/reconstruction)
/// fractHalf branch is hoisted to block level to eliminate per-pixel branch prediction misses
@inline(__always)
func applyBidirectionalMotionCompensationPixels(plane: inout [Int16], prevPlane: [Int16], nextPlane: [Int16], mvs: [MotionVector], refDirs: [Bool], width: Int, height: Int, blockSize: Int, shiftMultiplierX2: Int, roundOffset: Int) {
    let colCount = (width + blockSize - 1) / blockSize
    let rowCount = (height + blockSize - 1) / blockSize
    let body: (UnsafePointer<Int16>, UnsafePointer<Int16>, UnsafeMutablePointer<Int16>) -> Void = { prevBase, nextBase, dstBase in
        for row in 0..<rowCount {
            for col in 0..<colCount {
                let mvIndex = min(row * colCount + col, mvs.count - 1)
                let mv = mvs[mvIndex]
                let isBackward = (mvIndex < refDirs.count) ? refDirs[mvIndex] : false
                let srcBase = isBackward ? nextBase : prevBase
                let blockX = col * blockSize
                let blockY = row * blockSize
                let rawShiftX = Int(mv.dx) * shiftMultiplierX2
                let rawShiftY = Int(mv.dy) * shiftMultiplierX2
                let shiftX = rawShiftX >> 2
                let fractHalfX = (rawShiftX & 2) >> 1
                let shiftY = rawShiftY >> 2
                let fractHalfY = (rawShiftY & 2) >> 1
                let bw = min(blockSize, width - blockX)
                let bh = min(blockSize, height - blockY)

                if fractHalfX == 0 && fractHalfY == 0 {
                    // integer pixel path: no interpolation needed
                    for y in 0..<bh {
                        let dstY = blockY + y
                        let srcY = blockY + shiftY + y
                        let safeSrcY = max(0, min(srcY, height - 1))
                        let dstPtr = dstBase.advanced(by: dstY * width + blockX)
                        let srcRow = srcBase.advanced(by: safeSrcY * width)
                        for x in 0..<bw {
                            let sx = max(0, min(blockX + shiftX + x, width - 1))
                            dstPtr[x] = dstPtr[x] &+ srcRow[sx]
                        }
                    }
                } else if fractHalfY == 0 {
                    // horizontal half-pel only
                    for y in 0..<bh {
                        let dstY = blockY + y
                        let srcY = blockY + shiftY + y
                        let safeSrcY = max(0, min(srcY, height - 1))
                        let dstPtr = dstBase.advanced(by: dstY * width + blockX)
                        let srcRow0 = srcBase.advanced(by: safeSrcY * width)
                        for x in 0..<bw {
                            let srcX = blockX + shiftX + x
                            let sx0 = max(0, min(srcX, width - 1))
                            let sx1 = max(0, min(srcX + 1, width - 1))
                            dstPtr[x] = dstPtr[x] &+ Int16((Int(srcRow0[sx0]) + Int(srcRow0[sx1]) + roundOffset) >> 1)
                        }
                    }
                } else if fractHalfX == 0 {
                    // vertical half-pel only
                    for y in 0..<bh {
                        let dstY = blockY + y
                        let srcY = blockY + shiftY + y
                        let safeSrcY0 = max(0, min(srcY, height - 1))
                        let safeSrcY1 = max(0, min(srcY + 1, height - 1))
                        let dstPtr = dstBase.advanced(by: dstY * width + blockX)
                        let srcRow0 = srcBase.advanced(by: safeSrcY0 * width)
                        let srcRow1 = srcBase.advanced(by: safeSrcY1 * width)
                        for x in 0..<bw {
                            let sx = max(0, min(blockX + shiftX + x, width - 1))
                            dstPtr[x] = dstPtr[x] &+ Int16((Int(srcRow0[sx]) + Int(srcRow1[sx]) + roundOffset) >> 1)
                        }
                    }
                } else {
                    // bilinear (both half-pel)
                    for y in 0..<bh {
                        let dstY = blockY + y
                        let srcY = blockY + shiftY + y
                        let safeSrcY0 = max(0, min(srcY, height - 1))
                        let safeSrcY1 = max(0, min(srcY + 1, height - 1))
                        let dstPtr = dstBase.advanced(by: dstY * width + blockX)
                        let srcRow0 = srcBase.advanced(by: safeSrcY0 * width)
                        let srcRow1 = srcBase.advanced(by: safeSrcY1 * width)
                        for x in 0..<bw {
                            let srcX = blockX + shiftX + x
                            let sx0 = max(0, min(srcX, width - 1))
                            let sx1 = max(0, min(srcX + 1, width - 1))
                            dstPtr[x] = dstPtr[x] &+ Int16((Int(srcRow0[sx0]) + Int(srcRow0[sx1]) + Int(srcRow1[sx0]) + Int(srcRow1[sx1]) + 1 + roundOffset) >> 2)
                        }
                    }
                }
            }
        }
    }
    withUnsafePointers(prevPlane, nextPlane, mut: &plane, body)
}

@inline(__always)
fileprivate func getOBMCPredPixel(
    srcBase: UnsafePointer<Int16>, 
    mvs: [MotionVector],
    x: Int, y: Int, 
    blockX: Int, blockY: Int, 
    row: Int, col: Int, 
    bh: Int, bw: Int, 
    predBase: Int32,
    colCount: Int, rowCount: Int,
    shiftMultiplierX2: Int,
    width: Int, height: Int
) -> Int32 {
    let obmcBorder = 2
    var pred = predBase
    if y < obmcBorder && 0 < row {
        let nIdx = min((row - 1) * colCount + col, mvs.count - 1)
        let nMV = mvs[nIdx]
        let nSrcY = max(0, min(blockY + (Int(nMV.dy) * shiftMultiplierX2) / 4 + y, height - 1))
        let nSrcX = max(0, min(blockX + (Int(nMV.dx) * shiftMultiplierX2) / 4 + x, width - 1))
        pred = (pred * 3 + Int32(srcBase[nSrcY * width + nSrcX]) + 2) >> 2
    } else if y >= bh - obmcBorder && row < rowCount - 1 {
        let nIdx = min((row + 1) * colCount + col, mvs.count - 1)
        let nMV = mvs[nIdx]
        let nSrcY = max(0, min(blockY + (Int(nMV.dy) * shiftMultiplierX2) / 4 + y, height - 1))
        let nSrcX = max(0, min(blockX + (Int(nMV.dx) * shiftMultiplierX2) / 4 + x, width - 1))
        pred = (pred * 3 + Int32(srcBase[nSrcY * width + nSrcX]) + 2) >> 2
    }
    if x < obmcBorder && 0 < col {
        let nIdx = min(row * colCount + (col - 1), mvs.count - 1)
        let nMV = mvs[nIdx]
        let nSrcY = max(0, min(blockY + (Int(nMV.dy) * shiftMultiplierX2) / 4 + y, height - 1))
        let nSrcX = max(0, min(blockX + (Int(nMV.dx) * shiftMultiplierX2) / 4 + x, width - 1))
        pred = (pred * 3 + Int32(srcBase[nSrcY * width + nSrcX]) + 2) >> 2
    } else if x >= bw - obmcBorder && col < colCount - 1 {
        let nIdx = min(row * colCount + (col + 1), mvs.count - 1)
        let nMV = mvs[nIdx]
        let nSrcY = max(0, min(blockY + (Int(nMV.dy) * shiftMultiplierX2) / 4 + y, height - 1))
        let nSrcX = max(0, min(blockX + (Int(nMV.dx) * shiftMultiplierX2) / 4 + x, width - 1))
        pred = (pred * 3 + Int32(srcBase[nSrcY * width + nSrcX]) + 2) >> 2
    }
    return pred
}

@inline(__always)
func subtractMotionCompensationPixels(plane: inout [Int16], prevPlane: [Int16], mvs: [MotionVector], width: Int, height: Int, blockSize: Int, shiftMultiplierX2: Int, roundOffset: Int) {
    let colCount = (width + blockSize - 1) / blockSize
    let rowCount = (height + blockSize - 1) / blockSize
    let obmcBorder = 2


    @inline(__always)
    func processBlocks(srcBase: UnsafePointer<Int16>, dstBase: UnsafeMutablePointer<Int16>) {
        for row in 0..<rowCount {
            for col in 0..<colCount {
                let mvIndex = min(row * colCount + col, mvs.count - 1)
                let mv = mvs[mvIndex]
                let blockX = col * blockSize
                let blockY = row * blockSize
                
                let rawShiftX = Int(mv.dx) * shiftMultiplierX2
                let rawShiftY = Int(mv.dy) * shiftMultiplierX2
                let shiftX = rawShiftX >> 2
                let fractHalfX = (rawShiftX & 2) >> 1
                let shiftY = rawShiftY >> 2
                let fractHalfY = (rawShiftY & 2) >> 1
                let bw = min(blockSize, width - blockX)
                let bh = min(blockSize, height - blockY)
                
                let startInnerY = (0 < row) ? min(obmcBorder, bh) : 0
                let endInnerY = (row < rowCount - 1) ? max(startInnerY, bh - obmcBorder) : bh
                let startInnerX = (0 < col) ? min(obmcBorder, bw) : 0
                let endInnerX = (col < colCount - 1) ? max(startInnerX, bw - obmcBorder) : bw

                if fractHalfX == 0 && fractHalfY == 0 {
                    for y in 0..<startInnerY {
                        let dstY = blockY + y
                        let srcY = blockY + shiftY + y
                        let safeSrcY = max(0, min(srcY, height - 1))
                        let dstRowPtr = dstBase.advanced(by: dstY * width + blockX)
                        let srcRowPtr = srcBase.advanced(by: safeSrcY * width)
                        for x in 0..<bw {
                            let sx = max(0, min(blockX + shiftX + x, width - 1))
                            let basePixel = Int32(srcRowPtr[sx])
                            let predPixel = getOBMCPredPixel(srcBase: srcBase, mvs: mvs, x: x, y: y, blockX: blockX, blockY: blockY, row: row, col: col, bh: bh, bw: bw, predBase: basePixel, colCount: colCount, rowCount: rowCount, shiftMultiplierX2: shiftMultiplierX2, width: width, height: height)
                            dstRowPtr[x] = dstRowPtr[x] &- Int16(truncatingIfNeeded: predPixel)
                        }
                    }
                    for y in startInnerY..<endInnerY {
                        let dstY = blockY + y
                        let srcY = blockY + shiftY + y
                        let safeSrcY = max(0, min(srcY, height - 1))
                        let dstRowPtr = dstBase.advanced(by: dstY * width + blockX)
                        let srcRowPtr = srcBase.advanced(by: safeSrcY * width)
                        for x in 0..<startInnerX {
                            let sx = max(0, min(blockX + shiftX + x, width - 1))
                            let basePixel = Int32(srcRowPtr[sx])
                            let predPixel = getOBMCPredPixel(srcBase: srcBase, mvs: mvs, x: x, y: y, blockX: blockX, blockY: blockY, row: row, col: col, bh: bh, bw: bw, predBase: basePixel, colCount: colCount, rowCount: rowCount, shiftMultiplierX2: shiftMultiplierX2, width: width, height: height)
                            dstRowPtr[x] = dstRowPtr[x] &- Int16(truncatingIfNeeded: predPixel)
                        }
                        for x in startInnerX..<endInnerX {
                            let sx = max(0, min(blockX + shiftX + x, width - 1))
                            let basePixel = Int32(srcRowPtr[sx])
                            dstRowPtr[x] = dstRowPtr[x] &- Int16(truncatingIfNeeded: basePixel)
                        }
                        for x in endInnerX..<bw {
                            let sx = max(0, min(blockX + shiftX + x, width - 1))
                            let basePixel = Int32(srcRowPtr[sx])
                            let predPixel = getOBMCPredPixel(srcBase: srcBase, mvs: mvs, x: x, y: y, blockX: blockX, blockY: blockY, row: row, col: col, bh: bh, bw: bw, predBase: basePixel, colCount: colCount, rowCount: rowCount, shiftMultiplierX2: shiftMultiplierX2, width: width, height: height)
                            dstRowPtr[x] = dstRowPtr[x] &- Int16(truncatingIfNeeded: predPixel)
                        }
                    }
                    for y in endInnerY..<bh {
                        let dstY = blockY + y
                        let srcY = blockY + shiftY + y
                        let safeSrcY = max(0, min(srcY, height - 1))
                        let dstRowPtr = dstBase.advanced(by: dstY * width + blockX)
                        let srcRowPtr = srcBase.advanced(by: safeSrcY * width)
                        for x in 0..<bw {
                            let sx = max(0, min(blockX + shiftX + x, width - 1))
                            let basePixel = Int32(srcRowPtr[sx])
                            let predPixel = getOBMCPredPixel(srcBase: srcBase, mvs: mvs, x: x, y: y, blockX: blockX, blockY: blockY, row: row, col: col, bh: bh, bw: bw, predBase: basePixel, colCount: colCount, rowCount: rowCount, shiftMultiplierX2: shiftMultiplierX2, width: width, height: height)
                            dstRowPtr[x] = dstRowPtr[x] &- Int16(truncatingIfNeeded: predPixel)
                        }
                    }
                } else {
                    for y in 0..<startInnerY {
                        let dstY = blockY + y
                        let srcY = blockY + shiftY + y
                        let safeSrcY0 = max(0, min(srcY, height - 1))
                        let safeSrcY1 = max(0, min(srcY + fractHalfY, height - 1))
                        let dstRowPtr = dstBase.advanced(by: dstY * width + blockX)
                        let srcRow0 = srcBase.advanced(by: safeSrcY0 * width)
                        let srcRow1 = srcBase.advanced(by: safeSrcY1 * width)
                        for x in 0..<bw {
                            let srcX = blockX + shiftX + x
                            let sx0 = max(0, min(srcX, width - 1))
                            let sx1 = max(0, min(srcX + fractHalfX, width - 1))
                            let basePixel: Int32
                            if fractHalfY == 0 {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow0[sx1]) + roundOffset) >> 1)
                            } else if fractHalfX == 0 {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow1[sx0]) + roundOffset) >> 1)
                            } else {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow0[sx1]) + Int(srcRow1[sx0]) + Int(srcRow1[sx1]) + 1 + roundOffset) >> 2)
                            }
                            let predPixel = getOBMCPredPixel(srcBase: srcBase, mvs: mvs, x: x, y: y, blockX: blockX, blockY: blockY, row: row, col: col, bh: bh, bw: bw, predBase: basePixel, colCount: colCount, rowCount: rowCount, shiftMultiplierX2: shiftMultiplierX2, width: width, height: height)
                            dstRowPtr[x] = dstRowPtr[x] &- Int16(truncatingIfNeeded: predPixel)
                        }
                    }
                    for y in startInnerY..<endInnerY {
                        let dstY = blockY + y
                        let srcY = blockY + shiftY + y
                        let safeSrcY0 = max(0, min(srcY, height - 1))
                        let safeSrcY1 = max(0, min(srcY + fractHalfY, height - 1))
                        let dstRowPtr = dstBase.advanced(by: dstY * width + blockX)
                        let srcRow0 = srcBase.advanced(by: safeSrcY0 * width)
                        let srcRow1 = srcBase.advanced(by: safeSrcY1 * width)
                        
                        for x in 0..<startInnerX {
                            let srcX = blockX + shiftX + x
                            let sx0 = max(0, min(srcX, width - 1))
                            let sx1 = max(0, min(srcX + fractHalfX, width - 1))
                            let basePixel: Int32
                            if fractHalfY == 0 {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow0[sx1]) + roundOffset) >> 1)
                            } else if fractHalfX == 0 {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow1[sx0]) + roundOffset) >> 1)
                            } else {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow0[sx1]) + Int(srcRow1[sx0]) + Int(srcRow1[sx1]) + 1 + roundOffset) >> 2)
                            }
                            let predPixel = getOBMCPredPixel(srcBase: srcBase, mvs: mvs, x: x, y: y, blockX: blockX, blockY: blockY, row: row, col: col, bh: bh, bw: bw, predBase: basePixel, colCount: colCount, rowCount: rowCount, shiftMultiplierX2: shiftMultiplierX2, width: width, height: height)
                            dstRowPtr[x] = dstRowPtr[x] &- Int16(truncatingIfNeeded: predPixel)
                        }
                        for x in startInnerX..<endInnerX {
                            let srcX = blockX + shiftX + x
                            let sx0 = max(0, min(srcX, width - 1))
                            let sx1 = max(0, min(srcX + fractHalfX, width - 1))
                            let basePixel: Int32
                            if fractHalfY == 0 {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow0[sx1]) + roundOffset) >> 1)
                            } else if fractHalfX == 0 {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow1[sx0]) + roundOffset) >> 1)
                            } else {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow0[sx1]) + Int(srcRow1[sx0]) + Int(srcRow1[sx1]) + 1 + roundOffset) >> 2)
                            }
                            dstRowPtr[x] = dstRowPtr[x] &- Int16(truncatingIfNeeded: basePixel)
                        }
                        for x in endInnerX..<bw {
                            let srcX = blockX + shiftX + x
                            let sx0 = max(0, min(srcX, width - 1))
                            let sx1 = max(0, min(srcX + fractHalfX, width - 1))
                            let basePixel: Int32
                            if fractHalfY == 0 {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow0[sx1]) + roundOffset) >> 1)
                            } else if fractHalfX == 0 {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow1[sx0]) + roundOffset) >> 1)
                            } else {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow0[sx1]) + Int(srcRow1[sx0]) + Int(srcRow1[sx1]) + 1 + roundOffset) >> 2)
                            }
                            let predPixel = getOBMCPredPixel(srcBase: srcBase, mvs: mvs, x: x, y: y, blockX: blockX, blockY: blockY, row: row, col: col, bh: bh, bw: bw, predBase: basePixel, colCount: colCount, rowCount: rowCount, shiftMultiplierX2: shiftMultiplierX2, width: width, height: height)
                            dstRowPtr[x] = dstRowPtr[x] &- Int16(truncatingIfNeeded: predPixel)
                        }
                    }
                    for y in endInnerY..<bh {
                        let dstY = blockY + y
                        let srcY = blockY + shiftY + y
                        let safeSrcY0 = max(0, min(srcY, height - 1))
                        let safeSrcY1 = max(0, min(srcY + fractHalfY, height - 1))
                        let dstRowPtr = dstBase.advanced(by: dstY * width + blockX)
                        let srcRow0 = srcBase.advanced(by: safeSrcY0 * width)
                        let srcRow1 = srcBase.advanced(by: safeSrcY1 * width)
                        for x in 0..<bw {
                            let srcX = blockX + shiftX + x
                            let sx0 = max(0, min(srcX, width - 1))
                            let sx1 = max(0, min(srcX + fractHalfX, width - 1))
                            let basePixel: Int32
                            if fractHalfY == 0 {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow0[sx1]) + roundOffset) >> 1)
                            } else if fractHalfX == 0 {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow1[sx0]) + roundOffset) >> 1)
                            } else {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow0[sx1]) + Int(srcRow1[sx0]) + Int(srcRow1[sx1]) + 1 + roundOffset) >> 2)
                            }
                            let predPixel = getOBMCPredPixel(srcBase: srcBase, mvs: mvs, x: x, y: y, blockX: blockX, blockY: blockY, row: row, col: col, bh: bh, bw: bw, predBase: basePixel, colCount: colCount, rowCount: rowCount, shiftMultiplierX2: shiftMultiplierX2, width: width, height: height)
                            dstRowPtr[x] = dstRowPtr[x] &- Int16(truncatingIfNeeded: predPixel)
                        }
                    }
                }
            }
        }
    }
    withUnsafePointers(prevPlane, mut: &plane, processBlocks)
}

@inline(__always)
func applyMotionCompensationPixels(plane: inout [Int16], prevPlane: [Int16], mvs: [MotionVector], width: Int, height: Int, blockSize: Int, shiftMultiplierX2: Int, roundOffset: Int) {
    let colCount = (width + blockSize - 1) / blockSize
    let rowCount = (height + blockSize - 1) / blockSize
    let obmcBorder = 2


    @inline(__always)
    func processBlocks(srcBase: UnsafePointer<Int16>, dstBase: UnsafeMutablePointer<Int16>) {
        for row in 0..<rowCount {
            for col in 0..<colCount {
                let mvIndex = min(row * colCount + col, mvs.count - 1)
                let mv = mvs[mvIndex]
                let blockX = col * blockSize
                let blockY = row * blockSize
                
                let rawShiftX = Int(mv.dx) * shiftMultiplierX2
                let rawShiftY = Int(mv.dy) * shiftMultiplierX2
                let shiftX = rawShiftX >> 2
                let fractHalfX = (rawShiftX & 2) >> 1
                let shiftY = rawShiftY >> 2
                let fractHalfY = (rawShiftY & 2) >> 1
                let bw = min(blockSize, width - blockX)
                let bh = min(blockSize, height - blockY)
                
                let startInnerY = (0 < row) ? min(obmcBorder, bh) : 0
                let endInnerY = (row < rowCount - 1) ? max(startInnerY, bh - obmcBorder) : bh
                let startInnerX = (0 < col) ? min(obmcBorder, bw) : 0
                let endInnerX = (col < colCount - 1) ? max(startInnerX, bw - obmcBorder) : bw

                if fractHalfX == 0 && fractHalfY == 0 {
                    for y in 0..<startInnerY {
                        let dstY = blockY + y
                        let srcY = blockY + shiftY + y
                        let safeSrcY = max(0, min(srcY, height - 1))
                        let dstRowPtr = dstBase.advanced(by: dstY * width + blockX)
                        let srcRowPtr = srcBase.advanced(by: safeSrcY * width)
                        for x in 0..<bw {
                            let sx = max(0, min(blockX + shiftX + x, width - 1))
                            let basePixel = Int32(srcRowPtr[sx])
                            let predPixel = getOBMCPredPixel(srcBase: srcBase, mvs: mvs, x: x, y: y, blockX: blockX, blockY: blockY, row: row, col: col, bh: bh, bw: bw, predBase: basePixel, colCount: colCount, rowCount: rowCount, shiftMultiplierX2: shiftMultiplierX2, width: width, height: height)
                            dstRowPtr[x] = dstRowPtr[x] &+ Int16(truncatingIfNeeded: predPixel)
                        }
                    }
                    for y in startInnerY..<endInnerY {
                        let dstY = blockY + y
                        let srcY = blockY + shiftY + y
                        let safeSrcY = max(0, min(srcY, height - 1))
                        let dstRowPtr = dstBase.advanced(by: dstY * width + blockX)
                        let srcRowPtr = srcBase.advanced(by: safeSrcY * width)
                        for x in 0..<startInnerX {
                            let sx = max(0, min(blockX + shiftX + x, width - 1))
                            let basePixel = Int32(srcRowPtr[sx])
                            let predPixel = getOBMCPredPixel(srcBase: srcBase, mvs: mvs, x: x, y: y, blockX: blockX, blockY: blockY, row: row, col: col, bh: bh, bw: bw, predBase: basePixel, colCount: colCount, rowCount: rowCount, shiftMultiplierX2: shiftMultiplierX2, width: width, height: height)
                            dstRowPtr[x] = dstRowPtr[x] &+ Int16(truncatingIfNeeded: predPixel)
                        }
                        for x in startInnerX..<endInnerX {
                            let sx = max(0, min(blockX + shiftX + x, width - 1))
                            let basePixel = Int32(srcRowPtr[sx])
                            dstRowPtr[x] = dstRowPtr[x] &+ Int16(truncatingIfNeeded: basePixel)
                        }
                        for x in endInnerX..<bw {
                            let sx = max(0, min(blockX + shiftX + x, width - 1))
                            let basePixel = Int32(srcRowPtr[sx])
                            let predPixel = getOBMCPredPixel(srcBase: srcBase, mvs: mvs, x: x, y: y, blockX: blockX, blockY: blockY, row: row, col: col, bh: bh, bw: bw, predBase: basePixel, colCount: colCount, rowCount: rowCount, shiftMultiplierX2: shiftMultiplierX2, width: width, height: height)
                            dstRowPtr[x] = dstRowPtr[x] &+ Int16(truncatingIfNeeded: predPixel)
                        }
                    }
                    for y in endInnerY..<bh {
                        let dstY = blockY + y
                        let srcY = blockY + shiftY + y
                        let safeSrcY = max(0, min(srcY, height - 1))
                        let dstRowPtr = dstBase.advanced(by: dstY * width + blockX)
                        let srcRowPtr = srcBase.advanced(by: safeSrcY * width)
                        for x in 0..<bw {
                            let sx = max(0, min(blockX + shiftX + x, width - 1))
                            let basePixel = Int32(srcRowPtr[sx])
                            let predPixel = getOBMCPredPixel(srcBase: srcBase, mvs: mvs, x: x, y: y, blockX: blockX, blockY: blockY, row: row, col: col, bh: bh, bw: bw, predBase: basePixel, colCount: colCount, rowCount: rowCount, shiftMultiplierX2: shiftMultiplierX2, width: width, height: height)
                            dstRowPtr[x] = dstRowPtr[x] &+ Int16(truncatingIfNeeded: predPixel)
                        }
                    }
                } else {
                    for y in 0..<startInnerY {
                        let dstY = blockY + y
                        let srcY = blockY + shiftY + y
                        let safeSrcY0 = max(0, min(srcY, height - 1))
                        let safeSrcY1 = max(0, min(srcY + fractHalfY, height - 1))
                        let dstRowPtr = dstBase.advanced(by: dstY * width + blockX)
                        let srcRow0 = srcBase.advanced(by: safeSrcY0 * width)
                        let srcRow1 = srcBase.advanced(by: safeSrcY1 * width)
                        for x in 0..<bw {
                            let srcX = blockX + shiftX + x
                            let sx0 = max(0, min(srcX, width - 1))
                            let sx1 = max(0, min(srcX + fractHalfX, width - 1))
                            let basePixel: Int32
                            if fractHalfY == 0 {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow0[sx1]) + roundOffset) >> 1)
                            } else if fractHalfX == 0 {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow1[sx0]) + roundOffset) >> 1)
                            } else {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow0[sx1]) + Int(srcRow1[sx0]) + Int(srcRow1[sx1]) + 1 + roundOffset) >> 2)
                            }
                            let predPixel = getOBMCPredPixel(srcBase: srcBase, mvs: mvs, x: x, y: y, blockX: blockX, blockY: blockY, row: row, col: col, bh: bh, bw: bw, predBase: basePixel, colCount: colCount, rowCount: rowCount, shiftMultiplierX2: shiftMultiplierX2, width: width, height: height)
                            dstRowPtr[x] = dstRowPtr[x] &+ Int16(truncatingIfNeeded: predPixel)
                        }
                    }
                    for y in startInnerY..<endInnerY {
                        let dstY = blockY + y
                        let srcY = blockY + shiftY + y
                        let safeSrcY0 = max(0, min(srcY, height - 1))
                        let safeSrcY1 = max(0, min(srcY + fractHalfY, height - 1))
                        let dstRowPtr = dstBase.advanced(by: dstY * width + blockX)
                        let srcRow0 = srcBase.advanced(by: safeSrcY0 * width)
                        let srcRow1 = srcBase.advanced(by: safeSrcY1 * width)
                        
                        for x in 0..<startInnerX {
                            let srcX = blockX + shiftX + x
                            let sx0 = max(0, min(srcX, width - 1))
                            let sx1 = max(0, min(srcX + fractHalfX, width - 1))
                            let basePixel: Int32
                            if fractHalfY == 0 {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow0[sx1]) + roundOffset) >> 1)
                            } else if fractHalfX == 0 {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow1[sx0]) + roundOffset) >> 1)
                            } else {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow0[sx1]) + Int(srcRow1[sx0]) + Int(srcRow1[sx1]) + 1 + roundOffset) >> 2)
                            }
                            let predPixel = getOBMCPredPixel(srcBase: srcBase, mvs: mvs, x: x, y: y, blockX: blockX, blockY: blockY, row: row, col: col, bh: bh, bw: bw, predBase: basePixel, colCount: colCount, rowCount: rowCount, shiftMultiplierX2: shiftMultiplierX2, width: width, height: height)
                            dstRowPtr[x] = dstRowPtr[x] &+ Int16(truncatingIfNeeded: predPixel)
                        }
                        for x in startInnerX..<endInnerX {
                            let srcX = blockX + shiftX + x
                            let sx0 = max(0, min(srcX, width - 1))
                            let sx1 = max(0, min(srcX + fractHalfX, width - 1))
                            let basePixel: Int32
                            if fractHalfY == 0 {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow0[sx1]) + roundOffset) >> 1)
                            } else if fractHalfX == 0 {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow1[sx0]) + roundOffset) >> 1)
                            } else {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow0[sx1]) + Int(srcRow1[sx0]) + Int(srcRow1[sx1]) + 1 + roundOffset) >> 2)
                            }
                            dstRowPtr[x] = dstRowPtr[x] &+ Int16(truncatingIfNeeded: basePixel)
                        }
                        for x in endInnerX..<bw {
                            let srcX = blockX + shiftX + x
                            let sx0 = max(0, min(srcX, width - 1))
                            let sx1 = max(0, min(srcX + fractHalfX, width - 1))
                            let basePixel: Int32
                            if fractHalfY == 0 {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow0[sx1]) + roundOffset) >> 1)
                            } else if fractHalfX == 0 {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow1[sx0]) + roundOffset) >> 1)
                            } else {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow0[sx1]) + Int(srcRow1[sx0]) + Int(srcRow1[sx1]) + 1 + roundOffset) >> 2)
                            }
                            let predPixel = getOBMCPredPixel(srcBase: srcBase, mvs: mvs, x: x, y: y, blockX: blockX, blockY: blockY, row: row, col: col, bh: bh, bw: bw, predBase: basePixel, colCount: colCount, rowCount: rowCount, shiftMultiplierX2: shiftMultiplierX2, width: width, height: height)
                            dstRowPtr[x] = dstRowPtr[x] &+ Int16(truncatingIfNeeded: predPixel)
                        }
                    }
                    for y in endInnerY..<bh {
                        let dstY = blockY + y
                        let srcY = blockY + shiftY + y
                        let safeSrcY0 = max(0, min(srcY, height - 1))
                        let safeSrcY1 = max(0, min(srcY + fractHalfY, height - 1))
                        let dstRowPtr = dstBase.advanced(by: dstY * width + blockX)
                        let srcRow0 = srcBase.advanced(by: safeSrcY0 * width)
                        let srcRow1 = srcBase.advanced(by: safeSrcY1 * width)
                        for x in 0..<bw {
                            let srcX = blockX + shiftX + x
                            let sx0 = max(0, min(srcX, width - 1))
                            let sx1 = max(0, min(srcX + fractHalfX, width - 1))
                            let basePixel: Int32
                            if fractHalfY == 0 {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow0[sx1]) + roundOffset) >> 1)
                            } else if fractHalfX == 0 {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow1[sx0]) + roundOffset) >> 1)
                            } else {
                                basePixel = Int32((Int(srcRow0[sx0]) + Int(srcRow0[sx1]) + Int(srcRow1[sx0]) + Int(srcRow1[sx1]) + 1 + roundOffset) >> 2)
                            }
                            let predPixel = getOBMCPredPixel(srcBase: srcBase, mvs: mvs, x: x, y: y, blockX: blockX, blockY: blockY, row: row, col: col, bh: bh, bw: bw, predBase: basePixel, colCount: colCount, rowCount: rowCount, shiftMultiplierX2: shiftMultiplierX2, width: width, height: height)
                            dstRowPtr[x] = dstRowPtr[x] &+ Int16(truncatingIfNeeded: predPixel)
                        }
                    }
                }
            }
        }
    }
    withUnsafePointers(prevPlane, mut: &plane, processBlocks)
}
