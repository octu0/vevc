import Foundation

fileprivate struct SendableInt16Ptr: @unchecked Sendable {
    let ptr: UnsafeMutablePointer<Int16>
    init(_ ptr: UnsafeMutablePointer<Int16>) { self.ptr = ptr }
}

// MARK: - Spatial Adaptive Weight

final class ConcurrentBox<T>: @unchecked Sendable {
    var value: T
    init(_ value: T) { self.value = value }
}

@inline(__always)
func isBlockAllSkip(skipMap: [BlockMode], mapWidth: Int, lxStart: Int, lyStart: Int, countX: Int, countY: Int) -> Bool {
    for y in 0..<countY {
        let ly = lyStart + y
        for x in 0..<countX {
            let lx = lxStart + x
            let idx = ly * mapWidth + lx
            if idx < skipMap.count, skipMap[idx] == .inter {
                return false
            }
        }
    }
    return true
}

@inline(__always)
func evaluateQuantizeLayer32(view: BlockView, qt: QuantizationTable) {
    let subs = getSubbands32(view: view)
    let hl = subs.hl
    let lh = subs.lh
    let hh = subs.hh
    quantize16(hl, q: qt.qMid)
    quantize16(lh, q: qt.qMid)
    quantize16(hh, q: qt.qHigh)
}

/// σ-normalized AQ variant: the block's source-activity class selects the
/// dead-zone variant (same steps — the bitstream and decoder are unchanged).
@inline(__always)
func evaluateQuantizeLayer32WithActivity(view: BlockView, qt: QuantizationTable, activity: BlockActivityClass) {
    let subs = getSubbands32(view: view)
    switch activity {
    case .flat:
        quantize16(subs.hl, q: qt.qMidFlat)
        quantize16(subs.lh, q: qt.qMidFlat)
        quantize16(subs.hh, q: qt.qHighFlat)
    case .textured:
        quantize16(subs.hl, q: qt.qMidTextured)
        quantize16(subs.lh, q: qt.qMidTextured)
        quantize16(subs.hh, q: qt.qHighTextured)
    case .normal:
        quantize16(subs.hl, q: qt.qMid)
        quantize16(subs.lh, q: qt.qMid)
        quantize16(subs.hh, q: qt.qHigh)
    }
}

@inline(__always)
func evaluateQuantizeLayer16(view: BlockView, qt: QuantizationTable) {
    let subs = getSubbands16(view: view)
    let hl = subs.hl
    let lh = subs.lh
    let hh = subs.hh
    quantize8(hl, q: qt.qMid)
    quantize8(lh, q: qt.qMid)
    quantize8(hh, q: qt.qHigh)
}

@inline(__always)
func evaluateQuantizeLayer16WithActivity(view: BlockView, qt: QuantizationTable, activity: BlockActivityClass) {
    let subs = getSubbands16(view: view)
    switch activity {
    case .flat:
        quantize8(subs.hl, q: qt.qMidFlat)
        quantize8(subs.lh, q: qt.qMidFlat)
        quantize8(subs.hh, q: qt.qHighFlat)
    case .textured:
        quantize8(subs.hl, q: qt.qMidTextured)
        quantize8(subs.lh, q: qt.qMidTextured)
        quantize8(subs.hh, q: qt.qHighTextured)
    case .normal:
        quantize8(subs.hl, q: qt.qMid)
        quantize8(subs.lh, q: qt.qMid)
        quantize8(subs.hh, q: qt.qHigh)
    }
}

@inline(__always)
func evaluateQuantizeBase8(view: BlockView, qt: QuantizationTable) {
    let subs = getSubbands8(view: view)
    let ll = subs.ll
    let hl = subs.hl
    let lh = subs.lh
    let hh = subs.hh
    quantizeDPCM(ll, q: qt.qLow)
    quantize4(hl, q: qt.qMid)
    quantize4(lh, q: qt.qMid)
    quantize4(hh, q: qt.qHigh)
}


/// Per-block skip flags for a layer's luma extract: every layer's luma block
/// grid is the same ceil(width/32) lattice as the skip map, so one block maps
/// 1:1 to one entry.
@inline(__always)
func lumaSkipFlags(skipMap: [BlockMode], mapWidth: Int, rowCount: Int, colCount: Int) -> [Bool] {
    var flags = [Bool](repeating: false, count: rowCount * colCount)
    for i in 0..<rowCount {
        for j in 0..<colCount {
            flags[i * colCount + j] = isBlockAllSkip(skipMap: skipMap, mapWidth: mapWidth, lxStart: j, lyStart: i, countX: 1, countY: 1)
        }
    }
    return flags
}

/// Per-block skip flags for a layer's chroma extract: a chroma block on the
/// half-resolution plane spans 2×2 skip-map entries — all four must be
/// non-inter for the block to be skippable.
@inline(__always)
func chromaSkipFlags(skipMap: [BlockMode], mapWidth: Int, rowCount: Int, colCount: Int) -> [Bool] {
    var flags = [Bool](repeating: false, count: rowCount * colCount)
    for i in 0..<rowCount {
        for j in 0..<colCount {
            flags[i * colCount + j] = isBlockAllSkip(skipMap: skipMap, mapWidth: mapWidth, lxStart: j * 2, lyStart: i * 2, countX: 2, countY: 2)
        }
    }
    return flags
}

@inline(__always)
func isZeroBlock(view: BlockView) -> Bool {
    let ptr = view.base
    let w = view.width
    let s = view.stride
    for y in 0..<view.height {
        let row = ptr.advanced(by: y * s)
        for x in 0..<w {
            if row[x] != 0 { return false }
        }
    }
    return true
}

/// Gather one 32-block's LL quadrant into the subband plane (shared by the
/// skip and no-skip extracts; nonZero blocks only — zero blocks leave the
/// pre-cleared subband untouched).
@inline(__always)
private func gatherLL32(view: BlockView, w: Int, h: Int, subWidth: Int, subHeight: Int, dstBase: UnsafeMutablePointer<Int16>) {
    let destStartX = (w / 2)
    let destStartY = (h / 2)
    let subSize = (32 / 2)
    let subs = getSubbands32(view: view)
    let srcBase = subs.ll.base
    let limit = min(subSize, (subWidth - destStartX))
    if limit <= 0 { return }

    if limit == subSize && (destStartY + subSize) <= subHeight {
        let dstBasePtr = dstBase.advanced(by: (destStartY * subWidth) + destStartX)
        let dstRaw = UnsafeMutableRawPointer(dstBasePtr)
        let srcRaw = UnsafeRawPointer(srcBase)

        // unrolled SIMD copy
        for row in 0..<16 {
            let sRow = srcRaw.advanced(by: row * 32 * 2).assumingMemoryBound(to: SIMD16<Int16>.self)
            let dRow = dstRaw.advanced(by: row * subWidth * 2).assumingMemoryBound(to: SIMD16<Int16>.self)
            dRow.pointee = sRow.pointee
        }
    } else {
        for blockY in 0..<subSize {
            let dstY = (destStartY + blockY)
            if dstY < subHeight {
                let srcPtr = srcBase.advanced(by: (blockY * 32))
                let dstIdx = ((dstY * subWidth) + destStartX)
                dstBase.advanced(by: dstIdx).update(from: srcPtr, count: limit)
            }
        }
    }
}

/// 3-tap separable [1, 2, 1]/4 binomial smoothing for full continuous residual plane.
/// Eliminates cross-block boundary discontinuities prior to block extraction & DWT.
func smoothResidualPlaneContinuous(
    src: UnsafePointer<Int16>,
    dst: UnsafeMutablePointer<Int16>,
    temp: UnsafeMutablePointer<Int16>,
    width: Int,
    height: Int,
    activityMap: [BlockActivityClass]? = nil,
    stride: Int
) {
    let twoVec16 = SIMD16<Int16>(repeating: 2)
    let chunks = width / 16

    // --- 1. Horizontal Pass (Row-wise SIMD) ---
    for y in 0..<height {
        let srcRow = src.advanced(by: y * stride)
        let tmpRow = temp.advanced(by: y * stride)

        let left0 = srcRow[0]
        let mid0 = srcRow[0]
        let right0 = srcRow[1]
        tmpRow[0] = (left0 &+ (mid0 &<< 1) &+ right0 &+ 2) &>> 2

        var x = 1
        while (x + 16) <= width {
            let leftVec = UnsafeRawPointer(srcRow.advanced(by: x - 1)).loadUnaligned(as: SIMD16<Int16>.self)
            let midVec = UnsafeRawPointer(srcRow.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
            let rightVec = UnsafeRawPointer(srcRow.advanced(by: x + 1)).loadUnaligned(as: SIMD16<Int16>.self)

            let smoothed = (leftVec &+ (midVec &<< 1) &+ rightVec &+ twoVec16) &>> 2
            UnsafeMutableRawPointer(tmpRow.advanced(by: x)).storeBytes(of: smoothed, as: SIMD16<Int16>.self)
            x += 16
        }

        while x < (width - 1) {
            let l = srcRow[x - 1]
            let m = srcRow[x]
            let r = srcRow[x + 1]
            tmpRow[x] = (l &+ (m &<< 1) &+ r &+ 2) &>> 2
            x += 1
        }

        let lastX = width - 1
        let leftLast = srcRow[lastX - 1]
        let midLast = srcRow[lastX]
        let rightLast = srcRow[lastX]
        tmpRow[lastX] = (leftLast &+ (midLast &<< 1) &+ rightLast &+ 2) &>> 2
    }

    // --- 2. Vertical Pass (Column-wise SIMD across rows) ---
    let topRow = temp
    let row1 = temp.advanced(by: stride)
    let dstTop = dst
    for c in 0..<chunks {
        let x = c * 16
        let m = UnsafeRawPointer(topRow.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
        let b = UnsafeRawPointer(row1.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
        let out = (m &+ (m &<< 1) &+ b &+ twoVec16) &>> 2
        UnsafeMutableRawPointer(dstTop.advanced(by: x)).storeBytes(of: out, as: SIMD16<Int16>.self)
    }

    for y in 1..<(height - 1) {
        let rowPrev = temp.advanced(by: (y - 1) * stride)
        let rowCurr = temp.advanced(by: y * stride)
        let rowNext = temp.advanced(by: (y + 1) * stride)
        let dstRow = dst.advanced(by: y * stride)

        for c in 0..<chunks {
            let x = c * 16
            let t = UnsafeRawPointer(rowPrev.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
            let m = UnsafeRawPointer(rowCurr.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
            let b = UnsafeRawPointer(rowNext.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
            let out = (t &+ (m &<< 1) &+ b &+ twoVec16) &>> 2
            UnsafeMutableRawPointer(dstRow.advanced(by: x)).storeBytes(of: out, as: SIMD16<Int16>.self)
        }
    }

    let botRow = temp.advanced(by: (height - 1) * stride)
    let rowBeforeBot = temp.advanced(by: (height - 2) * stride)
    let dstBot = dst.advanced(by: (height - 1) * stride)
    for c in 0..<chunks {
        let x = c * 16
        let t = UnsafeRawPointer(rowBeforeBot.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
        let m = UnsafeRawPointer(botRow.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
        let out = (t &+ (m &<< 1) &+ m &+ twoVec16) &>> 2
        UnsafeMutableRawPointer(dstBot.advanced(by: x)).storeBytes(of: out, as: SIMD16<Int16>.self)
    }

    // --- 3. Textured Block Preservation ---
    if let activity = activityMap {
        let colCount = (width + 31) / 32
        let rowCount = (height + 31) / 32
        for r in 0..<rowCount {
            let by = r * 32
            let bh = min(32, height - by)
            for c in 0..<colCount {
                let blockIdx = (r * colCount) + c
                if blockIdx < activity.count && activity[blockIdx] == .textured {
                    let bx = c * 32
                    let bw = min(32, width - bx)
                    for line in 0..<bh {
                        let srcLine = src.advanced(by: (by + line) * stride + bx)
                        let dstLine = dst.advanced(by: (by + line) * stride + bx)
                        dstLine.update(from: srcLine, count: bw)
                    }
                }
            }
        }
    }
}

@inline(__always)
func extractSingleTransformBlocks32(r: Int16Reader, width: Int, height: Int, pool: BlockViewPool, qt: QuantizationTable) async -> (blocks: [BlockView], subband: [Int16], releaseFn: @Sendable () -> Void) {
    let subWidth = ((width + 1) / 2)
    let subHeight = ((height + 1) / 2)
    var subband = pool.getInt16(count: subWidth * subHeight)
    subband.withUnsafeMutableBufferPointer { ptr in
        ptr.initialize(repeating: 0)
    }

    let rowCount = ((height + 32 - 1) / 32)
    let colCount = ((width + 32 - 1) / 32)
    let totalBlocks = rowCount * colCount

    var tmpBlocks = pool.getBlockViewArray(capacity: totalBlocks)
    tmpBlocks.reserveCapacity(totalBlocks)
    for _ in 0..<totalBlocks {
        tmpBlocks.append(pool.get1024())
    }
    let blocks = tmpBlocks
    let chunkSize = 4

    let safeSrc = r.data.withUnsafeBufferPointer { UnsafeSendablePointer(ptr: $0.baseAddress!) }
    await withTaskGroup(of: Void.self) { group in
        for sRow in stride(from: 0, to: rowCount, by: chunkSize) {
            let endRow = min(sRow + chunkSize, rowCount)
            group.addTask { [blocks, qt, safeSrc] in
                let srcBase = safeSrc.ptr
                for i in sRow..<endRow {
                    let h = (i * 32)
                    for j in 0..<colCount {
                        let w = (j * 32)
                        if width <= w || height <= h { continue }
                        let view = blocks[(i * colCount) + j]
                        r.readBlock(x: w, y: h, width: 32, height: 32, into: view, srcBase: srcBase)
                        if isZeroBlock(view: view) != true {
                            dwt2DBlock32(view)
                            evaluateQuantizeLayer32(view: view, qt: qt)
                        }
                    }
                }
            }
        }
    }

    withUnsafePointers(mut: &subband) { dstBase in
        for i in 0..<rowCount {
            for j in 0..<colCount {
                let view = blocks[(i * colCount) + j]
                if isZeroBlock(view: view) { continue }

                let w = (j * 32)
                let h = (i * 32)
                if width <= w || height <= h { continue }
                gatherLL32(view: view, w: w, h: h, subWidth: subWidth, subHeight: subHeight, dstBase: dstBase)
            }
        }
    }

    withExtendedLifetime(subband) {}
    return (tmpBlocks, subband, { [tmpBlocks, subband] in pool.putBlockViewArray(tmpBlocks); pool.putInt16(subband) })
}

/// extractSingleTransformBlocks32 with a per-block skip flag (precomputed by
/// the caller via lumaSkipFlags/chromaSkipFlags): skip blocks bypass
/// read/DWT/quant and stay zero (One-Pyramid §5).
@inline(__always)
func extractSingleTransformBlocks32WithSkipMap(r: Int16Reader, width: Int, height: Int, pool: BlockViewPool, qt: QuantizationTable, isSkip: [Bool]) async -> (blocks: [BlockView], subband: [Int16], releaseFn: @Sendable () -> Void) {
    let subWidth = ((width + 1) / 2)
    let subHeight = ((height + 1) / 2)
    var subband = pool.getInt16(count: subWidth * subHeight)
    subband.withUnsafeMutableBufferPointer { ptr in
        ptr.initialize(repeating: 0)
    }

    let rowCount = ((height + 32 - 1) / 32)
    let colCount = ((width + 32 - 1) / 32)
    let totalBlocks = rowCount * colCount

    var tmpBlocks = pool.getBlockViewArray(capacity: totalBlocks)
    tmpBlocks.reserveCapacity(totalBlocks)
    for _ in 0..<totalBlocks {
        tmpBlocks.append(pool.get1024())
    }
    let blocks = tmpBlocks
    let chunkSize = 4

    let safeSrc = r.data.withUnsafeBufferPointer { UnsafeSendablePointer(ptr: $0.baseAddress!) }
    await withTaskGroup(of: Void.self) { group in
        for sRow in stride(from: 0, to: rowCount, by: chunkSize) {
            let endRow = min(sRow + chunkSize, rowCount)
            group.addTask { [blocks, qt, isSkip, safeSrc] in
                let srcBase = safeSrc.ptr
                for i in sRow..<endRow {
                    let h = (i * 32)
                    for j in 0..<colCount {
                        let w = (j * 32)
                        if width <= w || height <= h { continue }
                        let blockIdx = (i * colCount) + j
                        let view = blocks[blockIdx]

                        if isSkip[blockIdx] {
                            clearBlockRegion(base: view.base, width: 32, height: 32, stride: view.stride)
                            continue
                        }
                        r.readBlock(x: w, y: h, width: 32, height: 32, into: view, srcBase: srcBase)
                        if isZeroBlock(view: view) != true {
                            dwt2DBlock32(view)
                            evaluateQuantizeLayer32(view: view, qt: qt)
                        }
                    }
                }
            }
        }
    }

    withUnsafePointers(mut: &subband) { dstBase in
        for i in 0..<rowCount {
            for j in 0..<colCount {
                let blockIdx = (i * colCount) + j
                let view = blocks[blockIdx]
                if isSkip[blockIdx] || isZeroBlock(view: view) { continue }

                let w = (j * 32)
                let h = (i * 32)
                if width <= w || height <= h { continue }
                gatherLL32(view: view, w: w, h: h, subWidth: subWidth, subHeight: subHeight, dstBase: dstBase)
            }
        }
    }

    withExtendedLifetime(subband) {}
    return (tmpBlocks, subband, { [tmpBlocks, subband] in pool.putBlockViewArray(tmpBlocks); pool.putInt16(subband) })
}

/// σ-normalized AQ variant of extractSingleTransformBlocks32WithSkipMap:
/// identical pipeline, the per-block quantize call selects the dead-zone
/// variant from the luma activity map (grid 1:1 with the block grid).
func extractSingleTransformBlocks32WithSkipMapAndActivity(r: Int16Reader, width: Int, height: Int, pool: BlockViewPool, qt: QuantizationTable, isSkip: [Bool], activity: [BlockActivityClass]) async -> (blocks: [BlockView], subband: [Int16], releaseFn: @Sendable () -> Void) {
    let subWidth = ((width + 1) / 2)
    let subHeight = ((height + 1) / 2)
    var subband = pool.getInt16(count: subWidth * subHeight)
    subband.withUnsafeMutableBufferPointer { ptr in
        ptr.initialize(repeating: 0)
    }

    let rowCount = ((height + 32 - 1) / 32)
    let colCount = ((width + 32 - 1) / 32)
    let totalBlocks = rowCount * colCount

    var tmpBlocks = pool.getBlockViewArray(capacity: totalBlocks)
    tmpBlocks.reserveCapacity(totalBlocks)
    for _ in 0..<totalBlocks {
        tmpBlocks.append(pool.get1024())
    }
    let blocks = tmpBlocks
    let chunkSize = 4

    let safeSrc = r.data.withUnsafeBufferPointer { UnsafeSendablePointer(ptr: $0.baseAddress!) }
    await withTaskGroup(of: Void.self) { group in
        for sRow in stride(from: 0, to: rowCount, by: chunkSize) {
            let endRow = min(sRow + chunkSize, rowCount)
            group.addTask { [blocks, qt, isSkip, activity, safeSrc] in
                let srcBase = safeSrc.ptr
                for i in sRow..<endRow {
                    let h = (i * 32)
                    for j in 0..<colCount {
                        let w = (j * 32)
                        if width <= w || height <= h { continue }
                        let blockIdx = (i * colCount) + j
                        let view = blocks[blockIdx]

                        if isSkip[blockIdx] {
                            clearBlockRegion(base: view.base, width: 32, height: 32, stride: view.stride)
                            continue
                        }
                        r.readBlock(x: w, y: h, width: 32, height: 32, into: view, srcBase: srcBase)
                        if isZeroBlock(view: view) != true {
                            dwt2DBlock32(view)
                            evaluateQuantizeLayer32WithActivity(view: view, qt: qt, activity: activity[blockIdx])
                        }
                    }
                }
            }
        }
    }

    withUnsafePointers(mut: &subband) { dstBase in
        for i in 0..<rowCount {
            for j in 0..<colCount {
                let blockIdx = (i * colCount) + j
                let view = blocks[blockIdx]
                if isSkip[blockIdx] || isZeroBlock(view: view) { continue }

                let w = (j * 32)
                let h = (i * 32)
                if width <= w || height <= h { continue }
                gatherLL32(view: view, w: w, h: h, subWidth: subWidth, subHeight: subHeight, dstBase: dstBase)
            }
        }
    }

    withExtendedLifetime(subband) {}
    return (tmpBlocks, subband, { [tmpBlocks, subband] in pool.putBlockViewArray(tmpBlocks); pool.putInt16(subband) })
}

@inline(__always)
func extractSingleTransformSubband32(r: Int16Reader, width: Int, height: Int, pool: BlockViewPool) async -> ([Int16], @Sendable () -> Void) {
    let subWidth = ((width + 1) / 2)
    let subHeight = ((height + 1) / 2)
    var subband = pool.getInt16(count: subWidth * subHeight)
    let safeDst = withUnsafePointers(mut: &subband) { SendableInt16Ptr($0) }
    
    let rowCount = ((height + 32 - 1) / 32)
    let colCount = ((width + 32 - 1) / 32)
    
    let chunkSize = 4
    await withTaskGroup(of: Void.self) { group in
        for sRow in stride(from: 0, to: rowCount, by: chunkSize) {
            let endRow = min(sRow + chunkSize, rowCount)
            group.addTask { [safeDst] in
                let dstBase = safeDst.ptr
                let view = pool.get1024()
                defer { pool.put(view) }
                
                for i in sRow..<endRow {
                    let h = (i * 32)
                    for j in 0..<colCount {
                        let w = (j * 32)
                        if width <= w || height <= h { continue }
                        r.readBlock(x: w, y: h, width: 32, height: 32, into: view)
                        
                        let destStartX = (w / 2)
                        let destStartY = (h / 2)
                        let subSize = (32 / 2)
                        let limit = min(subSize, (subWidth - destStartX))
                        if limit <= 0 { continue }
                        
                        let srcBase = view.base
                        if limit == 16 && (destStartY + 16) <= subHeight {
                            for blockY in 0..<16 {
                                let dstIdx = ((destStartY + blockY) * subWidth) + destStartX
                                let dstPtr = dstBase.advanced(by: dstIdx)
                                let sy = blockY * 2
                                let srcRow0 = srcBase.advanced(by: sy * 32)
                                let srcRow1 = srcBase.advanced(by: (sy + 1) * 32)
                                for blockX in 0..<16 {
                                    let sx = blockX * 2
                                    let p0 = Int(srcRow0[sx])
                                    let p1 = Int(srcRow0[sx + 1])
                                    let p2 = Int(srcRow1[sx])
                                    let p3 = Int(srcRow1[sx + 1])
                                    dstPtr[blockX] = Int16((p0 + p1 + p2 + p3) >> 2)
                                }
                            }
                        } else {
                            for blockY in 0..<subSize {
                                let dstY = destStartY + blockY
                                if dstY < subHeight {
                                    let dstIdx = (dstY * subWidth) + destStartX
                                    let dstPtr = dstBase.advanced(by: dstIdx)
                                    let sy = blockY * 2
                                    let srcRow0 = srcBase.advanced(by: sy * 32)
                                    let srcRow1 = srcBase.advanced(by: (sy + 1) * 32)
                                    for blockX in 0..<limit {
                                        let sx = blockX * 2
                                        let p0 = Int(srcRow0[sx])
                                        let p1 = Int(srcRow0[sx + 1])
                                        let p2 = Int(srcRow1[sx])
                                        let p3 = Int(srcRow1[sx + 1])
                                        dstPtr[blockX] = Int16((p0 + p1 + p2 + p3) >> 2)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    withExtendedLifetime(subband) {}
    return (subband, { [subband] in pool.putInt16(subband) })
}

/// Gather one 16-block's LL quadrant into the subband plane (shared by the
/// skip and no-skip extracts; nonZero blocks only).
@inline(__always)
private func gatherLL16(view: BlockView, w: Int, h: Int, subWidth: Int, subHeight: Int, dstBase: UnsafeMutablePointer<Int16>) {
    let destStartX = (w / 2)
    let destStartY = (h / 2)
    let subSize = (16 / 2)
    let subs = getSubbands16(view: view)
    let srcBase = subs.ll.base
    let limit = min(subSize, (subWidth - destStartX))
    if limit <= 0 { return }

    if limit == subSize && (destStartY + subSize) <= subHeight {
        let dstBasePtr = dstBase.advanced(by: (destStartY * subWidth) + destStartX)
        let dstRaw = UnsafeMutableRawPointer(dstBasePtr)
        let srcRaw = UnsafeRawPointer(srcBase)

        // unrolled SIMD copy
        for row in 0..<8 {
            let sRow = srcRaw.advanced(by: row * 16 * 2).assumingMemoryBound(to: SIMD8<Int16>.self)
            let dRow = dstRaw.advanced(by: row * subWidth * 2).assumingMemoryBound(to: SIMD8<Int16>.self)
            dRow.pointee = sRow.pointee
        }
    } else {
        for blockY in 0..<subSize {
            let dstY = (destStartY + blockY)
            if dstY < subHeight {
                let srcPtr = srcBase.advanced(by: (blockY * 16))
                let dstIdx = ((dstY * subWidth) + destStartX)
                dstBase.advanced(by: dstIdx).update(from: srcPtr, count: limit)
            }
        }
    }
}

@inline(__always)
func extractSingleTransformBlocks16(r: Int16Reader, width: Int, height: Int, pool: BlockViewPool, qt: QuantizationTable) async -> (blocks: [BlockView], subband: [Int16], releaseFn: @Sendable () -> Void) {
    let subWidth = ((width + 1) / 2)
    let subHeight = ((height + 1) / 2)
    var subband = pool.getInt16(count: subWidth * subHeight)
    subband.withUnsafeMutableBufferPointer { ptr in
        ptr.initialize(repeating: 0)
    }

    let rowCount = ((height + 16 - 1) / 16)
    let colCount = ((width + 16 - 1) / 16)
    let totalBlocks = rowCount * colCount

    var tmpBlocks = pool.getBlockViewArray(capacity: totalBlocks)
    tmpBlocks.reserveCapacity(totalBlocks)
    for _ in 0..<totalBlocks {
        tmpBlocks.append(pool.get256())
    }
    let blocks = tmpBlocks
    let chunkSize = 4

    let safeSrc = r.data.withUnsafeBufferPointer { UnsafeSendablePointer(ptr: $0.baseAddress!) }
    await withTaskGroup(of: Void.self) { group in
        for sRow in stride(from: 0, to: rowCount, by: chunkSize) {
            let endRow = min(sRow + chunkSize, rowCount)
            group.addTask { [blocks, qt, safeSrc] in
                let srcBase = safeSrc.ptr
                for i in sRow..<endRow {
                    let h = (i * 16)
                    for j in 0..<colCount {
                        let w = (j * 16)
                        if width <= w || height <= h { continue }
                        let view = blocks[(i * colCount) + j]
                        r.readBlock(x: w, y: h, width: 16, height: 16, into: view, srcBase: srcBase)
                        if isZeroBlock(view: view) != true {
                            dwt2DBlock16(view)
                            evaluateQuantizeLayer16(view: view, qt: qt)
                        }
                    }
                }
            }
        }
    }

    withUnsafePointers(mut: &subband) { dstBase in
        for i in 0..<rowCount {
            for j in 0..<colCount {
                let view = blocks[(i * colCount) + j]
                if isZeroBlock(view: view) { continue }

                let w = (j * 16)
                let h = (i * 16)
                if width <= w || height <= h { continue }
                gatherLL16(view: view, w: w, h: h, subWidth: subWidth, subHeight: subHeight, dstBase: dstBase)
            }
        }
    }

    withExtendedLifetime(subband) {}
    return (tmpBlocks, subband, { [tmpBlocks, subband] in pool.putBlockViewArray(tmpBlocks); pool.putInt16(subband) })
}

/// extractSingleTransformBlocks16 with a per-block skip flag (precomputed by
/// the caller via lumaSkipFlags/chromaSkipFlags): skip blocks bypass
/// read/DWT/quant and stay zero (One-Pyramid §5).
@inline(__always)
func extractSingleTransformBlocks16WithSkipMap(r: Int16Reader, width: Int, height: Int, pool: BlockViewPool, qt: QuantizationTable, isSkip: [Bool]) async -> (blocks: [BlockView], subband: [Int16], releaseFn: @Sendable () -> Void) {
    let subWidth = ((width + 1) / 2)
    let subHeight = ((height + 1) / 2)
    var subband = pool.getInt16(count: subWidth * subHeight)
    subband.withUnsafeMutableBufferPointer { ptr in
        ptr.initialize(repeating: 0)
    }

    let rowCount = ((height + 16 - 1) / 16)
    let colCount = ((width + 16 - 1) / 16)
    let totalBlocks = rowCount * colCount

    var tmpBlocks = pool.getBlockViewArray(capacity: totalBlocks)
    tmpBlocks.reserveCapacity(totalBlocks)
    for _ in 0..<totalBlocks {
        tmpBlocks.append(pool.get256())
    }
    let blocks = tmpBlocks
    let chunkSize = 4

    let safeSrc = r.data.withUnsafeBufferPointer { UnsafeSendablePointer(ptr: $0.baseAddress!) }
    await withTaskGroup(of: Void.self) { group in
        for sRow in stride(from: 0, to: rowCount, by: chunkSize) {
            let endRow = min(sRow + chunkSize, rowCount)
            group.addTask { [blocks, qt, isSkip, safeSrc] in
                let srcBase = safeSrc.ptr
                for i in sRow..<endRow {
                    let h = (i * 16)
                    for j in 0..<colCount {
                        let w = (j * 16)
                        if width <= w || height <= h { continue }
                        let blockIdx = (i * colCount) + j
                        let view = blocks[blockIdx]

                        if isSkip[blockIdx] {
                            clearBlockRegion(base: view.base, width: 16, height: 16, stride: view.stride)
                            continue
                        }
                        r.readBlock(x: w, y: h, width: 16, height: 16, into: view, srcBase: srcBase)
                        if isZeroBlock(view: view) != true {
                            dwt2DBlock16(view)
                            evaluateQuantizeLayer16(view: view, qt: qt)
                        }
                    }
                }
            }
        }
    }

    withUnsafePointers(mut: &subband) { dstBase in
        for i in 0..<rowCount {
            for j in 0..<colCount {
                let blockIdx = (i * colCount) + j
                let view = blocks[blockIdx]
                if isSkip[blockIdx] || isZeroBlock(view: view) { continue }

                let w = (j * 16)
                let h = (i * 16)
                if width <= w || height <= h { continue }
                gatherLL16(view: view, w: w, h: h, subWidth: subWidth, subHeight: subHeight, dstBase: dstBase)
            }
        }
    }

    withExtendedLifetime(subband) {}
    return (tmpBlocks, subband, { [tmpBlocks, subband] in pool.putBlockViewArray(tmpBlocks); pool.putInt16(subband) })
}

/// σ-normalized AQ variant of extractSingleTransformBlocks16WithSkipMap (the
/// half-resolution 16px luma grid is 1:1 with the full-resolution 32px grid,
/// so the same activity map indexes both).
func extractSingleTransformBlocks16WithSkipMapAndActivity(r: Int16Reader, width: Int, height: Int, pool: BlockViewPool, qt: QuantizationTable, isSkip: [Bool], activity: [BlockActivityClass]) async -> (blocks: [BlockView], subband: [Int16], releaseFn: @Sendable () -> Void) {
    let subWidth = ((width + 1) / 2)
    let subHeight = ((height + 1) / 2)
    var subband = pool.getInt16(count: subWidth * subHeight)
    subband.withUnsafeMutableBufferPointer { ptr in
        ptr.initialize(repeating: 0)
    }

    let rowCount = ((height + 16 - 1) / 16)
    let colCount = ((width + 16 - 1) / 16)
    let totalBlocks = rowCount * colCount

    var tmpBlocks = pool.getBlockViewArray(capacity: totalBlocks)
    tmpBlocks.reserveCapacity(totalBlocks)
    for _ in 0..<totalBlocks {
        tmpBlocks.append(pool.get256())
    }
    let blocks = tmpBlocks
    let chunkSize = 4

    let safeSrc = r.data.withUnsafeBufferPointer { UnsafeSendablePointer(ptr: $0.baseAddress!) }
    await withTaskGroup(of: Void.self) { group in
        for sRow in stride(from: 0, to: rowCount, by: chunkSize) {
            let endRow = min(sRow + chunkSize, rowCount)
            group.addTask { [blocks, qt, isSkip, activity, safeSrc] in
                let srcBase = safeSrc.ptr
                for i in sRow..<endRow {
                    let h = (i * 16)
                    for j in 0..<colCount {
                        let w = (j * 16)
                        if width <= w || height <= h { continue }
                        let blockIdx = (i * colCount) + j
                        let view = blocks[blockIdx]

                        if isSkip[blockIdx] {
                            clearBlockRegion(base: view.base, width: 16, height: 16, stride: view.stride)
                            continue
                        }
                        r.readBlock(x: w, y: h, width: 16, height: 16, into: view, srcBase: srcBase)
                        if isZeroBlock(view: view) != true {
                            dwt2DBlock16(view)
                            evaluateQuantizeLayer16WithActivity(view: view, qt: qt, activity: activity[blockIdx])
                        }
                    }
                }
            }
        }
    }

    withUnsafePointers(mut: &subband) { dstBase in
        for i in 0..<rowCount {
            for j in 0..<colCount {
                let blockIdx = (i * colCount) + j
                let view = blocks[blockIdx]
                if isSkip[blockIdx] || isZeroBlock(view: view) { continue }

                let w = (j * 16)
                let h = (i * 16)
                if width <= w || height <= h { continue }
                gatherLL16(view: view, w: w, h: h, subWidth: subWidth, subHeight: subHeight, dstBase: dstBase)
            }
        }
    }

    withExtendedLifetime(subband) {}
    return (tmpBlocks, subband, { [tmpBlocks, subband] in pool.putBlockViewArray(tmpBlocks); pool.putInt16(subband) })
}

@inline(__always)
func extractSingleTransformSubband16(r: Int16Reader, width: Int, height: Int, pool: BlockViewPool) async -> ([Int16], @Sendable () -> Void) {
    let subWidth = (width + 1) / 2
    let subHeight = (height + 1) / 2
    var subband = pool.getInt16(count: subWidth * subHeight)
    let safeDst = withUnsafePointers(mut: &subband) { SendableInt16Ptr($0) }
    
    let rowCount = (height + (16 - 1)) / 16
    let colCount = (width + (16 - 1)) / 16
    
    let chunkSize = 8
    await withTaskGroup(of: Void.self) { group in
        for sRow in stride(from: 0, to: rowCount, by: chunkSize) {
            let endRow = min(sRow + chunkSize, rowCount)
            group.addTask { [safeDst] in
                let dstBase = safeDst.ptr
                let view = pool.get256()
                defer { pool.put(view) }
                
                for i in sRow..<endRow {
                    let h = (i * 16)
                    for j in 0..<colCount {
                        let w = (j * 16)
                        if width <= w || height <= h { continue }
                        r.readBlock(x: w, y: h, width: 16, height: 16, into: view)
                        
                        let destStartX = (w / 2)
                        let destStartY = (h / 2)
                        let subSize = (16 / 2)
                        let limit = min(subSize, (subWidth - destStartX))
                        if limit <= 0 { continue }
                        
                        let srcBase = view.base
                        if limit == 8 && (destStartY + 8) <= subHeight {
                            for blockY in 0..<8 {
                                let dstIdx = ((destStartY + blockY) * subWidth) + destStartX
                                let dstPtr = dstBase.advanced(by: dstIdx)
                                let sy = blockY * 2
                                let srcRow0 = srcBase.advanced(by: sy * 16)
                                let srcRow1 = srcBase.advanced(by: (sy + 1) * 16)
                                for blockX in 0..<8 {
                                    let sx = blockX * 2
                                    let p0 = Int(srcRow0[sx])
                                    let p1 = Int(srcRow0[sx + 1])
                                    let p2 = Int(srcRow1[sx])
                                    let p3 = Int(srcRow1[sx + 1])
                                    dstPtr[blockX] = Int16((p0 + p1 + p2 + p3) >> 2)
                                }
                            }
                        } else {
                            for blockY in 0..<subSize {
                                let dstY = destStartY + blockY
                                if dstY < subHeight {
                                    let dstIdx = (dstY * subWidth) + destStartX
                                    let dstPtr = dstBase.advanced(by: dstIdx)
                                    let sy = blockY * 2
                                    let srcRow0 = srcBase.advanced(by: sy * 16)
                                    let srcRow1 = srcBase.advanced(by: (sy + 1) * 16)
                                    for blockX in 0..<limit {
                                        let sx = blockX * 2
                                        let p0 = Int(srcRow0[sx])
                                        let p1 = Int(srcRow0[sx + 1])
                                        let p2 = Int(srcRow1[sx])
                                        let p3 = Int(srcRow1[sx + 1])
                                        dstPtr[blockX] = Int16((p0 + p1 + p2 + p3) >> 2)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    withExtendedLifetime(subband) {}
    return (subband, { [subband] in pool.putInt16(subband) })
}

@inline(__always)
func extractSingleTransformBlocksBase8(r: Int16Reader, width: Int, height: Int, pool: BlockViewPool) async -> (blocks: [BlockView], releaseFn: @Sendable () -> Void) {
    let rowCount = ((height + 8 - 1) / 8)
    let colCount = ((width + 8 - 1) / 8)
    let totalBlocks = rowCount * colCount

    var tmpBlocks = pool.getBlockViewArray(capacity: totalBlocks)
    tmpBlocks.reserveCapacity(totalBlocks)
    for _ in 0..<totalBlocks {
        tmpBlocks.append(pool.get64())
    }
    let blocks = tmpBlocks

    let chunkSize = 4
    await withTaskGroup(of: Void.self) { group in
        for sRow in stride(from: 0, to: rowCount, by: chunkSize) {
            let endRow = min(sRow + chunkSize, rowCount)
            group.addTask { [blocks, r] in
                r.data.withUnsafeBufferPointer { srcBuf in
                    let srcBase = srcBuf.baseAddress!
                    for i in sRow..<endRow {
                        let h = (i * 8)
                        for j in 0..<colCount {
                            let w = (j * 8)
                            if width <= w || height <= h { continue }
                            let view = blocks[(i * colCount) + j]
                            r.readBlock(x: w, y: h, width: 8, height: 8, into: view, srcBase: srcBase)
                            if isZeroBlock(view: view) != true {
                                dwt2DBlock8(view)
                            }
                        }
                    }
                }
            }
        }
    }
    return (tmpBlocks, { [tmpBlocks] in pool.putBlockViewArray(tmpBlocks) })
}

/// extractSingleTransformBlocksBase8 with a per-block skip flag (precomputed
/// by the caller via lumaSkipFlags/chromaSkipFlags): skip blocks bypass
/// read/DWT and stay zero (One-Pyramid §5).
@inline(__always)
func extractSingleTransformBlocksBase8WithSkipMap(r: Int16Reader, width: Int, height: Int, pool: BlockViewPool, isSkip: [Bool], cullSAD: Int = 0) async -> (blocks: [BlockView], releaseFn: @Sendable () -> Void) {
    let rowCount = ((height + 8 - 1) / 8)
    let colCount = ((width + 8 - 1) / 8)
    let totalBlocks = rowCount * colCount

    var tmpBlocks = pool.getBlockViewArray(capacity: totalBlocks)
    tmpBlocks.reserveCapacity(totalBlocks)
    for _ in 0..<totalBlocks {
        tmpBlocks.append(pool.get64())
    }
    let blocks = tmpBlocks

    let chunkSize = 4
    await withTaskGroup(of: Void.self) { group in
        for sRow in stride(from: 0, to: rowCount, by: chunkSize) {
            let endRow = min(sRow + chunkSize, rowCount)
            group.addTask { [blocks, isSkip, r, cullSAD] in
                r.data.withUnsafeBufferPointer { srcBuf in
                    let srcBase = srcBuf.baseAddress!
                    for i in sRow..<endRow {
                        let h = (i * 8)
                        for j in 0..<colCount {
                            let w = (j * 8)
                            if width <= w || height <= h { continue }
                            let blockIdx = (i * colCount) + j
                            let view = blocks[blockIdx]

                            if isSkip[blockIdx] {
                                clearBlockRegion(base: view.base, width: 8, height: 8, stride: view.stride)
                                continue
                            }
                            r.readBlock(x: w, y: h, width: 8, height: 8, into: view, srcBase: srcBase)
                            if 0 < cullSAD && base8BlockSAD(view) < cullSAD {
                                // Small residual energy: cull the whole block
                                // pre-DWT (encoder-only policy; all-zero
                                // coefficients are always legal).
                                clearBlockRegion(base: view.base, width: 8, height: 8, stride: view.stride)
                                continue
                            }
                            if isZeroBlock(view: view) != true {
                                dwt2DBlock8(view)
                            }
                        }
                    }
                }
            }
        }
    }
    return (tmpBlocks, { [tmpBlocks] in pool.putBlockViewArray(tmpBlocks) })
}

/// Sum of absolute sample values of an 8x8 Base8 block (raw residual).
@inline(__always)
func base8BlockSAD(_ view: BlockView) -> Int {
    var sum = 0
    let base = view.base
    let stride = view.stride
    for y in 0..<8 {
        let row = base.advanced(by: y * stride)
        for x in 0..<8 {
            let v = Int(row[x])
            sum += v < 0 ? -v : v
        }
    }
    return sum
}

@inline(__always)
func preparePlaneLayer32(pd: PlaneData420, pool: BlockViewPool, qtY: QuantizationTable, qtC: QuantizationTable) async -> (PlaneData420, [BlockView], [BlockView], [BlockView], @Sendable () -> Void) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)

    async let taskBufY = { () -> ([Int16], [BlockView], @Sendable () -> Void) in
        let (blocks, subband, r) = await extractSingleTransformBlocks32(r: pd.rY, width: dx, height: dy, pool: pool, qt: qtY)
        return (subband, blocks, r)
    }()

    async let taskBufCb = { () -> ([Int16], [BlockView], @Sendable () -> Void) in
        let (blocks, subband, r) = await extractSingleTransformBlocks32(r: pd.rCb, width: cbDx, height: cbDy, pool: pool, qt: qtC)
        return (subband, blocks, r)
    }()

    async let taskBufCr = { () -> ([Int16], [BlockView], @Sendable () -> Void) in
        let (blocks, subband, r) = await extractSingleTransformBlocks32(r: pd.rCr, width: cbDx, height: cbDy, pool: pool, qt: qtC)
        return (subband, blocks, r)
    }()

    let (subY, yBlocks, relY) = await taskBufY
    let (subCb, cbBlocks, relCb) = await taskBufCb
    let (subCr, crBlocks, relCr) = await taskBufCr

    let subPlane = PlaneData420(width: (dx + 1) / 2, height: (dy + 1) / 2, y: subY, cb: subCb, cr: subCr)
    return (subPlane, yBlocks, cbBlocks, crBlocks, { relY(); relCb(); relCr() })
}

@inline(__always)
func preparePlaneLayer32WithSkipMap(pd: PlaneData420, pool: BlockViewPool, qtY: QuantizationTable, qtC: QuantizationTable, skipMap: [BlockMode], skipMapWidth: Int) async -> (PlaneData420, [BlockView], [BlockView], [BlockView], @Sendable () -> Void) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)

    let ySkip = lumaSkipFlags(skipMap: skipMap, mapWidth: skipMapWidth, rowCount: (dy + 31) / 32, colCount: (dx + 31) / 32)
    let cSkip = chromaSkipFlags(skipMap: skipMap, mapWidth: skipMapWidth, rowCount: (cbDy + 31) / 32, colCount: (cbDx + 31) / 32)

    async let taskBufY = { [ySkip] () -> ([Int16], [BlockView], @Sendable () -> Void) in
        let (blocks, subband, r) = await extractSingleTransformBlocks32WithSkipMap(r: pd.rY, width: dx, height: dy, pool: pool, qt: qtY, isSkip: ySkip)
        return (subband, blocks, r)
    }()

    async let taskBufCb = { [cSkip] () -> ([Int16], [BlockView], @Sendable () -> Void) in
        let (blocks, subband, r) = await extractSingleTransformBlocks32WithSkipMap(r: pd.rCb, width: cbDx, height: cbDy, pool: pool, qt: qtC, isSkip: cSkip)
        return (subband, blocks, r)
    }()

    async let taskBufCr = { [cSkip] () -> ([Int16], [BlockView], @Sendable () -> Void) in
        let (blocks, subband, r) = await extractSingleTransformBlocks32WithSkipMap(r: pd.rCr, width: cbDx, height: cbDy, pool: pool, qt: qtC, isSkip: cSkip)
        return (subband, blocks, r)
    }()

    let (subY, yBlocks, relY) = await taskBufY
    let (subCb, cbBlocks, relCb) = await taskBufCb
    let (subCr, crBlocks, relCr) = await taskBufCr

    let subPlane = PlaneData420(width: (dx + 1) / 2, height: (dy + 1) / 2, y: subY, cb: subCb, cr: subCr)
    return (subPlane, yBlocks, cbBlocks, crBlocks, { relY(); relCb(); relCr() })
}

/// σ-normalized AQ variant: the LUMA plane quantizes with per-block dead-zone
/// variants selected by the activity map; chroma is untouched (masking is
/// luma-driven, and the chroma block grid spans 2×2 luma blocks).
@inline(__always)
func preparePlaneLayer32WithSkipMapAndActivity(pd: PlaneData420, pool: BlockViewPool, qtY: QuantizationTable, qtC: QuantizationTable, skipMap: [BlockMode], skipMapWidth: Int, activity: [BlockActivityClass]) async -> (PlaneData420, [BlockView], [BlockView], [BlockView], @Sendable () -> Void) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)

    let ySkip = lumaSkipFlags(skipMap: skipMap, mapWidth: skipMapWidth, rowCount: (dy + 31) / 32, colCount: (dx + 31) / 32)
    let cSkip = chromaSkipFlags(skipMap: skipMap, mapWidth: skipMapWidth, rowCount: (cbDy + 31) / 32, colCount: (cbDx + 31) / 32)

    async let taskBufY = { [ySkip, activity] () -> ([Int16], [BlockView], @Sendable () -> Void) in
        let (blocks, subband, r) = await extractSingleTransformBlocks32WithSkipMapAndActivity(r: pd.rY, width: dx, height: dy, pool: pool, qt: qtY, isSkip: ySkip, activity: activity)
        return (subband, blocks, r)
    }()

    async let taskBufCb = { [cSkip] () -> ([Int16], [BlockView], @Sendable () -> Void) in
        let (blocks, subband, r) = await extractSingleTransformBlocks32WithSkipMap(r: pd.rCb, width: cbDx, height: cbDy, pool: pool, qt: qtC, isSkip: cSkip)
        return (subband, blocks, r)
    }()

    async let taskBufCr = { [cSkip] () -> ([Int16], [BlockView], @Sendable () -> Void) in
        let (blocks, subband, r) = await extractSingleTransformBlocks32WithSkipMap(r: pd.rCr, width: cbDx, height: cbDy, pool: pool, qt: qtC, isSkip: cSkip)
        return (subband, blocks, r)
    }()

    let (subY, yBlocks, relY) = await taskBufY
    let (subCb, cbBlocks, relCb) = await taskBufCb
    let (subCr, crBlocks, relCr) = await taskBufCr

    let subPlane = PlaneData420(width: (dx + 1) / 2, height: (dy + 1) / 2, y: subY, cb: subCb, cr: subCr)
    return (subPlane, yBlocks, cbBlocks, crBlocks, { relY(); relCb(); relCr() })
}

@inline(__always)
func preparePlaneLayer16(pd: PlaneData420, pool: BlockViewPool, qtY: QuantizationTable, qtC: QuantizationTable) async -> (PlaneData420, [BlockView], [BlockView], [BlockView], @Sendable () -> Void) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)

    async let taskBufY = { () -> ([Int16], [BlockView], @Sendable () -> Void) in
        let (blocks, subband, r) = await extractSingleTransformBlocks16(r: pd.rY, width: dx, height: dy, pool: pool, qt: qtY)
        return (subband, blocks, r)
    }()

    async let taskBufCb = { () -> ([Int16], [BlockView], @Sendable () -> Void) in
        let (blocks, subband, r) = await extractSingleTransformBlocks16(r: pd.rCb, width: cbDx, height: cbDy, pool: pool, qt: qtC)
        return (subband, blocks, r)
    }()

    async let taskBufCr = { () -> ([Int16], [BlockView], @Sendable () -> Void) in
        let (blocks, subband, r) = await extractSingleTransformBlocks16(r: pd.rCr, width: cbDx, height: cbDy, pool: pool, qt: qtC)
        return (subband, blocks, r)
    }()

    let (subY, yBlocks, relY) = await taskBufY
    let (subCb, cbBlocks, relCb) = await taskBufCb
    let (subCr, crBlocks, relCr) = await taskBufCr

    let subPlane = PlaneData420(width: (dx + 1) / 2, height: (dy + 1) / 2, y: subY, cb: subCb, cr: subCr)
    return (subPlane, yBlocks, cbBlocks, crBlocks, { relY(); relCb(); relCr() })
}

@inline(__always)
func preparePlaneLayer16WithSkipMap(pd: PlaneData420, pool: BlockViewPool, qtY: QuantizationTable, qtC: QuantizationTable, skipMap: [BlockMode], skipMapWidth: Int) async -> (PlaneData420, [BlockView], [BlockView], [BlockView], @Sendable () -> Void) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)

    let ySkip = lumaSkipFlags(skipMap: skipMap, mapWidth: skipMapWidth, rowCount: (dy + 15) / 16, colCount: (dx + 15) / 16)
    let cSkip = chromaSkipFlags(skipMap: skipMap, mapWidth: skipMapWidth, rowCount: (cbDy + 15) / 16, colCount: (cbDx + 15) / 16)

    async let taskBufY = { [ySkip] () -> ([Int16], [BlockView], @Sendable () -> Void) in
        let (blocks, subband, r) = await extractSingleTransformBlocks16WithSkipMap(r: pd.rY, width: dx, height: dy, pool: pool, qt: qtY, isSkip: ySkip)
        return (subband, blocks, r)
    }()

    async let taskBufCb = { [cSkip] () -> ([Int16], [BlockView], @Sendable () -> Void) in
        let (blocks, subband, r) = await extractSingleTransformBlocks16WithSkipMap(r: pd.rCb, width: cbDx, height: cbDy, pool: pool, qt: qtC, isSkip: cSkip)
        return (subband, blocks, r)
    }()

    async let taskBufCr = { [cSkip] () -> ([Int16], [BlockView], @Sendable () -> Void) in
        let (blocks, subband, r) = await extractSingleTransformBlocks16WithSkipMap(r: pd.rCr, width: cbDx, height: cbDy, pool: pool, qt: qtC, isSkip: cSkip)
        return (subband, blocks, r)
    }()

    let (subY, yBlocks, relY) = await taskBufY
    let (subCb, cbBlocks, relCb) = await taskBufCb
    let (subCr, crBlocks, relCr) = await taskBufCr

    let subPlane = PlaneData420(width: (dx + 1) / 2, height: (dy + 1) / 2, y: subY, cb: subCb, cr: subCr)
    return (subPlane, yBlocks, cbBlocks, crBlocks, { relY(); relCb(); relCr() })
}

/// σ-normalized AQ variant (luma-only, see preparePlaneLayer32WithSkipMapAndActivity).
@inline(__always)
func preparePlaneLayer16WithSkipMapAndActivity(pd: PlaneData420, pool: BlockViewPool, qtY: QuantizationTable, qtC: QuantizationTable, skipMap: [BlockMode], skipMapWidth: Int, activity: [BlockActivityClass]) async -> (PlaneData420, [BlockView], [BlockView], [BlockView], @Sendable () -> Void) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)

    let ySkip = lumaSkipFlags(skipMap: skipMap, mapWidth: skipMapWidth, rowCount: (dy + 15) / 16, colCount: (dx + 15) / 16)
    let cSkip = chromaSkipFlags(skipMap: skipMap, mapWidth: skipMapWidth, rowCount: (cbDy + 15) / 16, colCount: (cbDx + 15) / 16)

    async let taskBufY = { [ySkip, activity] () -> ([Int16], [BlockView], @Sendable () -> Void) in
        let (blocks, subband, r) = await extractSingleTransformBlocks16WithSkipMapAndActivity(r: pd.rY, width: dx, height: dy, pool: pool, qt: qtY, isSkip: ySkip, activity: activity)
        return (subband, blocks, r)
    }()

    async let taskBufCb = { [cSkip] () -> ([Int16], [BlockView], @Sendable () -> Void) in
        let (blocks, subband, r) = await extractSingleTransformBlocks16WithSkipMap(r: pd.rCb, width: cbDx, height: cbDy, pool: pool, qt: qtC, isSkip: cSkip)
        return (subband, blocks, r)
    }()

    async let taskBufCr = { [cSkip] () -> ([Int16], [BlockView], @Sendable () -> Void) in
        let (blocks, subband, r) = await extractSingleTransformBlocks16WithSkipMap(r: pd.rCr, width: cbDx, height: cbDy, pool: pool, qt: qtC, isSkip: cSkip)
        return (subband, blocks, r)
    }()

    let (subY, yBlocks, relY) = await taskBufY
    let (subCb, cbBlocks, relCb) = await taskBufCb
    let (subCr, crBlocks, relCr) = await taskBufCr

    let subPlane = PlaneData420(width: (dx + 1) / 2, height: (dy + 1) / 2, y: subY, cb: subCb, cr: subCr)
    return (subPlane, yBlocks, cbBlocks, crBlocks, { relY(); relCb(); relCr() })
}

/// Assemble the layer2 payload: derive the per-plane zero thresholds, entropy
/// encode the three coefficient planes (EncodeTransform.swift), and serialize
/// the VEVCLayerData container.
@inline(__always)
func encodeLayer32Payload(dx: Int, dy: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, yBlocks: inout [BlockView], cbBlocks: inout [BlockView], crBlocks: inout [BlockView], parentYBlocks: [BlockView], parentCbBlocks: [BlockView], parentCrBlocks: [BlockView], histories: [EntropyHistoryState]?, selectModel: ModelSelectorFn, updateHistory: Bool = true) -> [UInt8] {
    // Layer2 (32x32) contains the highest-frequency DWT subbands with the
    // lowest CSF sensitivity. P-frame residuals at this level can be zeroed
    // more aggressively (threshold=3) than Layer1 (threshold=2) without
    // perceptible quality loss.
    let safeThresholdY = min(3, min(zeroThreshold, max(0, Int(qtY.step) / 64)))
    let safeThresholdC = min(8, min(zeroThreshold, max(0, Int(qtC.step) / 64)))

    let colCountY = (dx + 31) / 32
    let rowCountY = (dy + 31) / 32
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let colCountC = (cbDx + 31) / 32
    let rowCountC = (cbDy + 31) / 32

    let bufY = encodePlaneSubbands32(blocks: &yBlocks, zeroThreshold: safeThresholdY, parentBlocks: parentYBlocks, colCount: colCountY, rowCount: rowCountY, history: histories?[0], selectModel: selectModel, updateHistory: updateHistory)
    let bufCb = encodePlaneSubbands32(blocks: &cbBlocks, zeroThreshold: safeThresholdC, parentBlocks: parentCbBlocks, colCount: colCountC, rowCount: rowCountC, history: histories?[1], selectModel: selectModel, updateHistory: updateHistory)
    let bufCr = encodePlaneSubbands32(blocks: &crBlocks, zeroThreshold: safeThresholdC, parentBlocks: parentCrBlocks, colCount: colCountC, rowCount: rowCountC, history: histories?[2], selectModel: selectModel, updateHistory: updateHistory)

    debugLog({
        return "  [Layer 2] qtY=\(qtY.step), qtC=\(qtC.step) Y=\(bufY.count) Cb=\(bufCb.count) Cr=\(bufCr.count) bytes"
    }())

    return VEVCLayerData.serialize(
        qtYStep: UInt16(qtY.step), qtCStep: UInt16(qtC.step),
        bufY: bufY, bufCb: bufCb, bufCr: bufCr
    )
}

@inline(__always)
func encodeLayer32PayloadWithSkipMap(dx: Int, dy: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, yBlocks: inout [BlockView], cbBlocks: inout [BlockView], crBlocks: inout [BlockView], parentYBlocks: [BlockView], parentCbBlocks: [BlockView], parentCrBlocks: [BlockView], ySkip: [Bool], cSkip: [Bool], isTreezY: [Bool]? = nil, isTreezCb: [Bool]? = nil, isTreezCr: [Bool]? = nil, histories: [EntropyHistoryState]?, selectModel: ModelSelectorFn, updateHistory: Bool = true) -> ([UInt8], [Bool], [Bool], [Bool]) {
    let safeThresholdY = min(3, min(zeroThreshold, max(0, Int(qtY.step) / 64)))
    let safeThresholdC = min(8, min(zeroThreshold, max(0, Int(qtC.step) / 64)))

    let colCountY = (dx + 31) / 32
    let rowCountY = (dy + 31) / 32
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let colCountC = (cbDx + 31) / 32
    let rowCountC = (cbDy + 31) / 32

    let (bufY, yZeros) = encodePlaneSubbands32WithSkipMap(blocks: &yBlocks, zeroThreshold: safeThresholdY, parentBlocks: parentYBlocks, colCount: colCountY, rowCount: rowCountY, isSkip: ySkip, isTreez: isTreezY, history: histories?[0], selectModel: selectModel, updateHistory: updateHistory)
    let (bufCb, cbZeros) = encodePlaneSubbands32WithSkipMap(blocks: &cbBlocks, zeroThreshold: safeThresholdC, parentBlocks: parentCbBlocks, colCount: colCountC, rowCount: rowCountC, isSkip: cSkip, isTreez: isTreezCb, history: histories?[1], selectModel: selectModel, updateHistory: updateHistory)
    let (bufCr, crZeros) = encodePlaneSubbands32WithSkipMap(blocks: &crBlocks, zeroThreshold: safeThresholdC, parentBlocks: parentCrBlocks, colCount: colCountC, rowCount: rowCountC, isSkip: cSkip, isTreez: isTreezCr, history: histories?[2], selectModel: selectModel, updateHistory: updateHistory)

    debugLog({
        return "  [Layer 2] qtY=\(qtY.step), qtC=\(qtC.step) Y=\(bufY.count) Cb=\(bufCb.count) Cr=\(bufCr.count) bytes"
    }())

    let layerData = VEVCLayerData.serialize(
        qtYStep: UInt16(qtY.step), qtCStep: UInt16(qtC.step),
        bufY: bufY, bufCb: bufCb, bufCr: bufCr
    )
    return (layerData, yZeros, cbZeros, crZeros)
}

/// Assemble the layer1 payload: derive the per-plane zero thresholds, entropy
/// encode the three coefficient planes (EncodeTransform.swift), and serialize
/// the VEVCLayerData container.
@inline(__always)
func encodeLayer16Payload(dx: Int, dy: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, yBlocks: inout [BlockView], cbBlocks: inout [BlockView], crBlocks: inout [BlockView], parentYBlocks: [BlockView], parentCbBlocks: [BlockView], parentCrBlocks: [BlockView], histories: [EntropyHistoryState]?, selectModel: ModelSelectorFn, updateHistory: Bool = true) -> [UInt8] {
    let safeThresholdY = min(2, min(zeroThreshold, max(0, Int(qtY.step) / 64)))
    let safeThresholdC = min(8, min(zeroThreshold, max(0, Int(qtC.step) / 64)))

    let colCountY = (dx + 15) / 16
    let rowCountY = (dy + 15) / 16
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let colCountC = (cbDx + 15) / 16
    let rowCountC = (cbDy + 15) / 16

    let bufY = encodePlaneSubbands16(blocks: &yBlocks, zeroThreshold: safeThresholdY, parentBlocks: parentYBlocks, colCount: colCountY, rowCount: rowCountY, history: histories?[0], selectModel: selectModel, updateHistory: updateHistory)
    let bufCb = encodePlaneSubbands16(blocks: &cbBlocks, zeroThreshold: safeThresholdC, parentBlocks: parentCbBlocks, colCount: colCountC, rowCount: rowCountC, history: histories?[1], selectModel: selectModel, updateHistory: updateHistory)
    let bufCr = encodePlaneSubbands16(blocks: &crBlocks, zeroThreshold: safeThresholdC, parentBlocks: parentCrBlocks, colCount: colCountC, rowCount: rowCountC, history: histories?[2], selectModel: selectModel, updateHistory: updateHistory)

    debugLog({
        return "  [Layer 1] qtY=\(qtY.step), qtC=\(qtC.step) Y=\(bufY.count) Cb=\(bufCb.count) Cr=\(bufCr.count) bytes"
    }())

    return VEVCLayerData.serialize(
        qtYStep: UInt16(qtY.step), qtCStep: UInt16(qtC.step),
        bufY: bufY, bufCb: bufCb, bufCr: bufCr
    )
}

@inline(__always)
func encodeLayer16PayloadWithSkipMap(dx: Int, dy: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, yBlocks: inout [BlockView], cbBlocks: inout [BlockView], crBlocks: inout [BlockView], parentYBlocks: [BlockView], parentCbBlocks: [BlockView], parentCrBlocks: [BlockView], ySkip: [Bool], cSkip: [Bool], isTreezY: [Bool]? = nil, isTreezCb: [Bool]? = nil, isTreezCr: [Bool]? = nil, histories: [EntropyHistoryState]?, selectModel: ModelSelectorFn, updateHistory: Bool = true) -> ([UInt8], [Bool], [Bool], [Bool]) {
    let safeThresholdY = min(2, min(zeroThreshold, max(0, Int(qtY.step) / 64)))
    let safeThresholdC = min(8, min(zeroThreshold, max(0, Int(qtC.step) / 64)))

    let colCountY = (dx + 15) / 16
    let rowCountY = (dy + 15) / 16
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let colCountC = (cbDx + 15) / 16
    let rowCountC = (cbDy + 15) / 16

    let (bufY, yZeros) = encodePlaneSubbands16WithSkipMap(blocks: &yBlocks, zeroThreshold: safeThresholdY, parentBlocks: parentYBlocks, colCount: colCountY, rowCount: rowCountY, isSkip: ySkip, isTreez: isTreezY, history: histories?[0], selectModel: selectModel, updateHistory: updateHistory)
    let (bufCb, cbZeros) = encodePlaneSubbands16WithSkipMap(blocks: &cbBlocks, zeroThreshold: safeThresholdC, parentBlocks: parentCbBlocks, colCount: colCountC, rowCount: rowCountC, isSkip: cSkip, isTreez: isTreezCb, history: histories?[1], selectModel: selectModel, updateHistory: updateHistory)
    let (bufCr, crZeros) = encodePlaneSubbands16WithSkipMap(blocks: &crBlocks, zeroThreshold: safeThresholdC, parentBlocks: parentCrBlocks, colCount: colCountC, rowCount: rowCountC, isSkip: cSkip, isTreez: isTreezCr, history: histories?[2], selectModel: selectModel, updateHistory: updateHistory)

    debugLog({
        return "  [Layer 1] qtY=\(qtY.step), qtC=\(qtC.step) Y=\(bufY.count) Cb=\(bufCb.count) Cr=\(bufCr.count) bytes"
    }())

    let layerData = VEVCLayerData.serialize(
        qtYStep: UInt16(qtY.step), qtCStep: UInt16(qtC.step),
        bufY: bufY, bufCb: bufCb, bufCr: bufCr
    )
    return (layerData, yZeros, cbZeros, crZeros)
}

@inline(__always)
func reconstructPlaneBase8(blocks: [BlockView], width: Int, height: Int, qt: QuantizationTable, pool: BlockViewPool) -> ([Int16], @Sendable () -> Void) {
    let colCount = (width + 7) / 8
    let rowCount = (height + 7) / 8
    var plane = pool.getInt16(count: width * height)
    withUnsafePointers(mut: &plane) { dstBase in
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
                dequantizeDPCM(ptr: base, stride: 8, q: qt.qLow)
                dequantize4(ptr: base.advanced(by: 4), stride: 8, q: qt.qMid)
                dequantize4(ptr: base.advanced(by: 32), stride: 8, q: qt.qMid)
                dequantize4(ptr: base.advanced(by: 36), stride: 8, q: qt.qHigh)
                inverseDWT2DBlock8(ptr: view.base, stride: view.stride)
                            
                switch true {
                case isEdgeY != true && isEdgeX != true:
                    let v = blk
                    for h in 0..<8 {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: 8)
                    }
                case 0 < loopH && 0 < loopW:
                    let v = blk
                    for h in 0..<loopH {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: loopW)
                    }
                default:
                    break
                }
            }
        }
    }
    return (plane, { [plane] in pool.putInt16(plane) })
}

@inline(__always)
func reconstructPlaneLayer32Y(blocks: [BlockView], prevImg: Image16, width: Int, height: Int, qt: QuantizationTable, pool: BlockViewPool, skipMap: [BlockMode]? = nil) -> ([Int16], @Sendable () -> Void) {
    let colCount = (width + 31) / 32
    let rowCount = (height + 31) / 32
    let sCount = skipMap?.count ?? 0
    var plane = pool.getInt16(count: width * height)
    withUnsafePointers(mut: &plane) { dstBase in
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
                
                if let map = skipMap {
                    let l2Index = row * colCount + col
                    if l2Index < sCount && map[l2Index] != .inter {
                        continue
                    }
                }
                
                let llX = startX / 2
                let llY = startY / 2
                prevImg.readY(x: llX, y: llY, size: 16, into: blk)
                                        
                let view = blk
                let base = view.base
                dequantize16(ptr: base.advanced(by: 16), stride: 32, q: qt.qMid)
                dequantize16(ptr: base.advanced(by: 512), stride: 32, q: qt.qMid)
                dequantize16(ptr: base.advanced(by: 528), stride: 32, q: qt.qHigh)
                inverseDWT2DBlock32(ptr: view.base, stride: view.stride)
                            
                switch true {
                case isEdgeY != true && isEdgeX != true:
                    let v = blk
                    for h in 0..<32 {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: 32)
                    }
                case 0 < loopH && 0 < loopW:
                    let v = blk
                    for h in 0..<loopH {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: loopW)
                    }
                default:
                    break
                }
            }
        }
    }
    return (plane, { [plane] in pool.putInt16(plane) })
}

@inline(__always)
func reconstructPlaneLayer32Cb(blocks: [BlockView], prevImg: Image16, width: Int, height: Int, qt: QuantizationTable, pool: BlockViewPool, skipMap: [BlockMode]? = nil, skipBw: Int = 0, skipBh: Int = 0) -> ([Int16], @Sendable () -> Void) {
    let colCount = (width + 31) / 32
    let rowCount = (height + 31) / 32
    var plane = pool.getInt16(count: width * height)
    withUnsafePointers(mut: &plane) { dstBase in
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
                
                // Chroma blocks span 2×2 luma-geometry skip-map entries (a
                // 32px chroma block covers 64px at full resolution).
                if let map = skipMap, 0 < skipBw, base8ChromaAllSkip(skipMap: map, bw: skipBw, bh: skipBh, c: col, r: row) {
                    continue
                }

                let llX = startX / 2
                let llY = startY / 2
                prevImg.readCb(x: llX, y: llY, size: 16, into: blk)
                
                let view = blk
                let base = view.base
                dequantize16(ptr: base.advanced(by: 16), stride: 32, q: qt.qMid)
                dequantize16(ptr: base.advanced(by: 512), stride: 32, q: qt.qMid)
                dequantize16(ptr: base.advanced(by: 528), stride: 32, q: qt.qHigh)
                inverseDWT2DBlock32(ptr: view.base, stride: view.stride)
                            
                switch true {
                case isEdgeY != true && isEdgeX != true:
                    let v = blk
                    for h in 0..<32 {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: 32)
                    }
                case 0 < loopH && 0 < loopW:
                    let v = blk
                    for h in 0..<loopH {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: loopW)
                    }
                default:
                    break
                }
            }
        }
    }
    return (plane, { [plane] in pool.putInt16(plane) })
}

@inline(__always)
func reconstructPlaneLayer32Cr(blocks: [BlockView], prevImg: Image16, width: Int, height: Int, qt: QuantizationTable, pool: BlockViewPool, skipMap: [BlockMode]? = nil, skipBw: Int = 0, skipBh: Int = 0) -> ([Int16], @Sendable () -> Void) {
    let colCount = (width + 31) / 32
    let rowCount = (height + 31) / 32
    var plane = pool.getInt16(count: width * height)
    withUnsafePointers(mut: &plane) { dstBase in
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
                
                // Chroma blocks span 2×2 luma-geometry skip-map entries (a
                // 32px chroma block covers 64px at full resolution).
                if let map = skipMap, 0 < skipBw, base8ChromaAllSkip(skipMap: map, bw: skipBw, bh: skipBh, c: col, r: row) {
                    continue
                }

                let llX = startX / 2
                let llY = startY / 2
                prevImg.readCr(x: llX, y: llY, size: 16, into: blk)
                
                let view = blk
                let base = view.base
                dequantize16(ptr: base.advanced(by: 16), stride: 32, q: qt.qMid)
                dequantize16(ptr: base.advanced(by: 512), stride: 32, q: qt.qMid)
                dequantize16(ptr: base.advanced(by: 528), stride: 32, q: qt.qHigh)
                inverseDWT2DBlock32(ptr: view.base, stride: view.stride)
                            
                switch true {
                case isEdgeY != true && isEdgeX != true:
                    let v = blk
                    for h in 0..<32 {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: 32)
                    }
                case 0 < loopH && 0 < loopW:
                    let v = blk
                    for h in 0..<loopH {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: loopW)
                    }
                default:
                    break
                }
            }
        }
    }
    return (plane, { [plane] in pool.putInt16(plane) })
}

@inline(__always)
func reconstructPlaneLayer16Y(blocks: [BlockView], prevImg: Image16, width: Int, height: Int, qt: QuantizationTable, pool: BlockViewPool) -> ([Int16], @Sendable () -> Void) {
    let colCount = (width + 15) / 16
    let rowCount = (height + 15) / 16
    var plane = pool.getInt16(count: width * height)
    withUnsafePointers(mut: &plane) { dstBase in
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
                dequantize8(ptr: base.advanced(by: 8), stride: 16, q: qt.qMid)
                dequantize8(ptr: base.advanced(by: 128), stride: 16, q: qt.qMid)
                dequantize8(ptr: base.advanced(by: 136), stride: 16, q: qt.qHigh)
                inverseDWT2DBlock16(ptr: view.base, stride: view.stride)
                            
                switch true {
                case isEdgeY != true && isEdgeX != true:
                    let v = blk
                    for h in 0..<16 {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: 16)
                    }
                case 0 < loopH && 0 < loopW:
                    let v = blk
                    for h in 0..<loopH {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: loopW)
                    }
                default:
                    break
                }
            }
        }
    }
    return (plane, { [plane] in pool.putInt16(plane) })
}

@inline(__always)
func reconstructPlaneLayer16Cb(blocks: [BlockView], prevImg: Image16, width: Int, height: Int, qt: QuantizationTable, pool: BlockViewPool) -> ([Int16], @Sendable () -> Void) {
    let colCount = (width + 15) / 16
    let rowCount = (height + 15) / 16
    var plane = pool.getInt16(count: width * height)
    withUnsafePointers(mut: &plane) { dstBase in
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
                dequantize8(ptr: base.advanced(by: 8), stride: 16, q: qt.qMid)
                dequantize8(ptr: base.advanced(by: 128), stride: 16, q: qt.qMid)
                dequantize8(ptr: base.advanced(by: 136), stride: 16, q: qt.qHigh)
                inverseDWT2DBlock16(ptr: view.base, stride: view.stride)
                            
                switch true {
                case isEdgeY != true && isEdgeX != true:
                    let v = blk
                    for h in 0..<16 {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: 16)
                    }
                case 0 < loopH && 0 < loopW:
                    let v = blk
                    for h in 0..<loopH {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: loopW)
                    }
                default:
                    break
                }
            }
        }
    }
    return (plane, { [plane] in pool.putInt16(plane) })
}

@inline(__always)
func reconstructPlaneLayer16Cr(blocks: [BlockView], prevImg: Image16, width: Int, height: Int, qt: QuantizationTable, pool: BlockViewPool) -> ([Int16], @Sendable () -> Void) {
    let colCount = (width + 15) / 16
    let rowCount = (height + 15) / 16
    var plane = pool.getInt16(count: width * height)
    withUnsafePointers(mut: &plane) { dstBase in
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
                dequantize8(ptr: base.advanced(by: 8), stride: 16, q: qt.qMid)
                dequantize8(ptr: base.advanced(by: 128), stride: 16, q: qt.qMid)
                dequantize8(ptr: base.advanced(by: 136), stride: 16, q: qt.qHigh)
                inverseDWT2DBlock16(ptr: view.base, stride: view.stride)
                            
                switch true {
                case isEdgeY != true && isEdgeX != true:
                    let v = blk
                    for h in 0..<16 {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: 16)
                    }
                case 0 < loopH && 0 < loopW:
                    let v = blk
                    for h in 0..<loopH {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: loopW)
                    }
                default:
                    break
                }
            }
        }
    }
    return (plane, { [plane] in pool.putInt16(plane) })
}

/// Base8 encode, I-frame: static-table entropy coding (DPCM handled inside
/// encodePlaneBaseSubbands8 via blockEncodeDPCM4/MED), no SAD gating, no
/// history state. selectModel picks the profile's static AC tables.
@inline(__always)
func encodePlaneBase8Intra(pd: PlaneData420, pool: BlockViewPool, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, selectModel: @escaping ModelSelectorFn) async -> ([UInt8], PlaneData420, [BlockView], [BlockView], [BlockView], @Sendable () -> Void) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)

    async let taskBufY = { () -> ([UInt8], [Int16], @Sendable () -> Void, [BlockView], @Sendable () -> Void) in
        var (blocks, relBlocks) = await extractSingleTransformBlocksBase8(r: pd.rY, width: dx, height: dy, pool: pool)
        for i in blocks.indices {
            evaluateQuantizeBase8(view: blocks[i], qt: qtY)
        }

        let safeThreshold = min(1, min(zeroThreshold, max(0, Int(qtY.step) / 64)))
        let buf = encodePlaneBaseSubbands8(blocks: &blocks, zeroThreshold: safeThreshold, selectModel: selectModel)

        let quantizedBlocks = blocks
        let (reconPlane, rPlane) = reconstructPlaneBase8(blocks: blocks, width: dx, height: dy, qt: qtY, pool: pool)
        return (buf, reconPlane, rPlane, quantizedBlocks, relBlocks)
    }()

    async let taskBufCb = { () -> ([UInt8], [Int16], @Sendable () -> Void, [BlockView], @Sendable () -> Void) in
        var (blocks, relBlocks) = await extractSingleTransformBlocksBase8(r: pd.rCb, width: cbDx, height: cbDy, pool: pool)
        for i in blocks.indices {
            evaluateQuantizeBase8(view: blocks[i], qt: qtC)
        }

        let safeThreshold = min(8, max(0, (zeroThreshold / 8) - (Int(qtC.step)  / 32)))
        let buf = encodePlaneBaseSubbands8(blocks: &blocks, zeroThreshold: safeThreshold, selectModel: selectModel)

        let quantizedBlocks = blocks
        let (reconPlane, rPlane) = reconstructPlaneBase8(blocks: blocks, width: cbDx, height: cbDy, qt: qtC, pool: pool)
        return (buf, reconPlane, rPlane, quantizedBlocks, relBlocks)
    }()

    async let taskBufCr = { () -> ([UInt8], [Int16], @Sendable () -> Void, [BlockView], @Sendable () -> Void) in
        var (blocks, relBlocks) = await extractSingleTransformBlocksBase8(r: pd.rCr, width: cbDx, height: cbDy, pool: pool)
        for i in blocks.indices {
            evaluateQuantizeBase8(view: blocks[i], qt: qtC)
        }

        let safeThreshold = min(8, max(0, (zeroThreshold / 8) - (Int(qtC.step) / 32)))
        let buf = encodePlaneBaseSubbands8(blocks: &blocks, zeroThreshold: safeThreshold, selectModel: selectModel)

        let quantizedBlocks = blocks
        let (reconPlane, rPlane) = reconstructPlaneBase8(blocks: blocks, width: cbDx, height: cbDy, qt: qtC, pool: pool)
        return (buf, reconPlane, rPlane, quantizedBlocks, relBlocks)
    }()

    let (bufY, reconY, r0Y, base8YBlocks, relYBlocks) = await taskBufY
    let (bufCb, reconCb, r0Cb, base8CbBlocks, relCbBlocks) = await taskBufCb
    let (bufCr, reconCr, r0Cr, base8CrBlocks, relCrBlocks) = await taskBufCr

    let reconstructed = PlaneData420(width: dx, height: dy, y: reconY, cb: reconCb, cr: reconCr)

    debugLog({
        return "  [Layer 0/Base] Y=\(bufY.count) Cb=\(bufCb.count) Cr=\(bufCr.count) bytes"
    }())

    let out = VEVCLayerData.serialize(
        qtYStep: UInt16(qtY.step), qtCStep: UInt16(qtC.step),
        bufY: bufY, bufCb: bufCb, bufCr: bufCr
    )

    return (out, reconstructed, base8YBlocks, base8CbBlocks, base8CrBlocks, {
        r0Y()
        r0Cb()
        r0Cr()
        relYBlocks()
        relCbBlocks()
        relCrBlocks()
    })
}

/// Base8 encode, P-frame profile 0x01: SAD-gated luma residual clearing +
/// the P-frame entropy path with the shipped static tables (no skip map, no
/// history state).
@inline(__always)
func encodePlaneBase8PFrame(pd: PlaneData420, pool: BlockViewPool, sads: [Int], qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int) async -> ([UInt8], PlaneData420, [BlockView], [BlockView], [BlockView], @Sendable () -> Void) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)

    let yColCount8 = (dx + 7) / 8
    let yRowCount8 = (dy + 7) / 8

    async let taskBufY = { () -> ([UInt8], [Int16], @Sendable () -> Void, [BlockView], @Sendable () -> Void) in
        var (blocks, relBlocks) = await extractSingleTransformBlocksBase8(r: pd.rY, width: dx, height: dy, pool: pool)
        for i in blocks.indices {
            if i < sads.count {
                let col = i % yColCount8
                let row = i / yColCount8
                let threshold = spatialSADThreshold(baseSAD: scaledSADThreshold(150, step: (Int(qtY.step) + 8) >> 4), blockCol: col, blockRow: row, colCount: yColCount8, rowCount: yRowCount8)
                if sads[i] < threshold {
                    let b = blocks[i]
                    clearBlockRegion(base: b.base, width: b.width, height: b.height, stride: b.stride)
                }
            }
            evaluateQuantizeBase8(view: blocks[i], qt: qtY)
        }

        // P-frame Base8: apply safeThreshold to zero out imperceptible residuals
        let safeThreshold = min(1, min(zeroThreshold, max(0, Int(qtY.step) / 64)))
        let buf = encodePlaneBaseSubbands8PFrame(blocks: &blocks, zeroThreshold: safeThreshold, history: nil, selectModel: unifiedSelectModel)

        let quantizedBlocks = blocks
        let (reconPlane, rPlane) = reconstructPlaneBase8(blocks: blocks, width: dx, height: dy, qt: qtY, pool: pool)
        return (buf, reconPlane, rPlane, quantizedBlocks, relBlocks)
    }()

    async let taskBufCb = { () -> ([UInt8], [Int16], @Sendable () -> Void, [BlockView], @Sendable () -> Void) in
        var (blocks, relBlocks) = await extractSingleTransformBlocksBase8(r: pd.rCb, width: cbDx, height: cbDy, pool: pool)
        for i in blocks.indices {
            evaluateQuantizeBase8(view: blocks[i], qt: qtC)
        }

        let safeThreshold = min(8, max(0, (zeroThreshold / 8) - (Int(qtC.step)  / 32)))
        let buf = encodePlaneBaseSubbands8PFrame(blocks: &blocks, zeroThreshold: safeThreshold, history: nil, selectModel: unifiedSelectModel)

        let quantizedBlocks = blocks
        let (reconPlane, rPlane) = reconstructPlaneBase8(blocks: blocks, width: cbDx, height: cbDy, qt: qtC, pool: pool)
        return (buf, reconPlane, rPlane, quantizedBlocks, relBlocks)
    }()

    async let taskBufCr = { () -> ([UInt8], [Int16], @Sendable () -> Void, [BlockView], @Sendable () -> Void) in
        var (blocks, relBlocks) = await extractSingleTransformBlocksBase8(r: pd.rCr, width: cbDx, height: cbDy, pool: pool)
        for i in blocks.indices {
            evaluateQuantizeBase8(view: blocks[i], qt: qtC)
        }

        let safeThreshold = min(8, max(0, (zeroThreshold / 8) - (Int(qtC.step) / 32)))
        let buf = encodePlaneBaseSubbands8PFrame(blocks: &blocks, zeroThreshold: safeThreshold, history: nil, selectModel: unifiedSelectModel)

        let quantizedBlocks = blocks
        let (reconPlane, rPlane) = reconstructPlaneBase8(blocks: blocks, width: cbDx, height: cbDy, qt: qtC, pool: pool)
        return (buf, reconPlane, rPlane, quantizedBlocks, relBlocks)
    }()

    let (bufY, reconY, r0Y, base8YBlocks, relYBlocks) = await taskBufY
    let (bufCb, reconCb, r0Cb, base8CbBlocks, relCbBlocks) = await taskBufCb
    let (bufCr, reconCr, r0Cr, base8CrBlocks, relCrBlocks) = await taskBufCr

    let reconstructed = PlaneData420(width: dx, height: dy, y: reconY, cb: reconCb, cr: reconCr)

    debugLog({
        return "  [Layer 0/Base] Y=\(bufY.count) Cb=\(bufCb.count) Cr=\(bufCr.count) bytes"
    }())

    let out = VEVCLayerData.serialize(
        qtYStep: UInt16(qtY.step), qtCStep: UInt16(qtC.step),
        bufY: bufY, bufCb: bufCb, bufCr: bufCr
    )

    return (out, reconstructed, base8YBlocks, base8CbBlocks, base8CrBlocks, {
        r0Y()
        r0Cb()
        r0Cr()
        relYBlocks()
        relCbBlocks()
        relCrBlocks()
    })
}

/// Base8 prepare, P-frame profile 0x02: extract, SAD-gated clearing, and quantization.
@inline(__always)
func preparePlaneBase8WithSkipMap(
    pd: PlaneData420, pool: BlockViewPool, sads: [Int],
    qtY: QuantizationTable, qtC: QuantizationTable,
    skipMap: [BlockMode], skipMapWidth: Int
) async -> ([BlockView], [BlockView], [BlockView], @Sendable () -> Void) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)

    let yColCount8 = (dx + 7) / 8
    let yRowCount8 = (dy + 7) / 8

    let ySkip = lumaSkipFlags(skipMap: skipMap, mapWidth: skipMapWidth, rowCount: yRowCount8, colCount: yColCount8)
    let cSkip = chromaSkipFlags(skipMap: skipMap, mapWidth: skipMapWidth, rowCount: (cbDy + 7) / 8, colCount: (cbDx + 7) / 8)

    // Chroma residual culling (experimental): Base8 luma gets an SAD-gated
    // whole-block clear (below) but Cb/Cr had no analog — measured on miko1
    // motion clips the chroma nonzero-block rate ran ~2.5x luma's, making
    // P-L0 chroma cost MORE than luma. Active only in the saturated regime
    // (layer-0 chroma baseStep ≥ 2048 Q4) so normal-rate streams are untouched,
    // and scaled by the same qstep curve as the luma gate. Env-tunable while
    // under evaluation.
    let chromaCullBaseSAD = Int(ProcessInfo.processInfo.environment["VEVC_L0C_CULL"] ?? "") ?? 96
    let chromaCullActive = Int(qtC.step) >= 2048
    // Same qstep-proportional curve as the luma gate: base 96 reaches the
    // measured-good 512 at the saturated step (real 256) and backs off
    // linearly below it.
    let chromaCullSAD = chromaCullActive ? scaledSADThreshold(chromaCullBaseSAD, step: (Int(qtC.step) + 8) >> 4) : 0

    async let taskY = { [ySkip] () -> ([BlockView], @Sendable () -> Void) in
        let (blocks, relBlocks) = await extractSingleTransformBlocksBase8WithSkipMap(r: pd.rY, width: dx, height: dy, pool: pool, isSkip: ySkip)
        for i in blocks.indices {
            if i < sads.count {
                let col = i % yColCount8
                let row = i / yColCount8
                let threshold = spatialSADThreshold(baseSAD: scaledSADThreshold(150, step: (Int(qtY.step) + 8) >> 4), blockCol: col, blockRow: row, colCount: yColCount8, rowCount: yRowCount8)
                if sads[i] < threshold {
                    let b = blocks[i]
                    clearBlockRegion(base: b.base, width: b.width, height: b.height, stride: b.stride)
                }
            }
            evaluateQuantizeBase8(view: blocks[i], qt: qtY)
        }
        return (blocks, relBlocks)
    }()

    async let taskCb = { [cSkip, chromaCullSAD] () -> ([BlockView], @Sendable () -> Void) in
        let (blocks, relBlocks) = await extractSingleTransformBlocksBase8WithSkipMap(r: pd.rCb, width: cbDx, height: cbDy, pool: pool, isSkip: cSkip, cullSAD: chromaCullSAD)
        for i in blocks.indices {
            evaluateQuantizeBase8(view: blocks[i], qt: qtC)
        }
        return (blocks, relBlocks)
    }()

    async let taskCr = { [cSkip, chromaCullSAD] () -> ([BlockView], @Sendable () -> Void) in
        let (blocks, relBlocks) = await extractSingleTransformBlocksBase8WithSkipMap(r: pd.rCr, width: cbDx, height: cbDy, pool: pool, isSkip: cSkip, cullSAD: chromaCullSAD)
        for i in blocks.indices {
            evaluateQuantizeBase8(view: blocks[i], qt: qtC)
        }
        return (blocks, relBlocks)
    }()

    let (base8YBlocks, relYBlocks) = await taskY
    let (base8CbBlocks, relCbBlocks) = await taskCb
    let (base8CrBlocks, relCrBlocks) = await taskCr

    return (base8YBlocks, base8CbBlocks, base8CrBlocks, {
        relYBlocks()
        relCbBlocks()
        relCrBlocks()
    })
}

/// Base8 serialize & reconstruct, P-frame profile 0x02.
@inline(__always)
func serializePlaneBase8PFrameWithSkipMap(
    pd: PlaneData420, pool: BlockViewPool,
    qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int,
    base8YBlocks: inout [BlockView], base8CbBlocks: inout [BlockView], base8CrBlocks: inout [BlockView],
    skipMap: [BlockMode], skipMapWidth: Int,
    isTreezY: [Bool]? = nil, isTreezCb: [Bool]? = nil, isTreezCr: [Bool]? = nil,
    histories: [EntropyHistoryState]?,
    updateHistory: Bool = true
) -> ([UInt8], PlaneData420, @Sendable () -> Void, [Bool], [Bool], [Bool]) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)

    let yColCount8 = (dx + 7) / 8
    let yRowCount8 = (dy + 7) / 8

    let ySkip = lumaSkipFlags(skipMap: skipMap, mapWidth: skipMapWidth, rowCount: yRowCount8, colCount: yColCount8)
    let cSkip = chromaSkipFlags(skipMap: skipMap, mapWidth: skipMapWidth, rowCount: (cbDy + 7) / 8, colCount: (cbDx + 7) / 8)

    let safeThresholdY = min(1, min(zeroThreshold, max(0, Int(qtY.step) / 64)))
    let (bufY, yZeros) = encodePlaneBaseSubbands8PFrameWithSkipMap(blocks: &base8YBlocks, zeroThreshold: safeThresholdY, isSkip: ySkip, isTreez: isTreezY, history: histories?[0], selectModel: unifiedSelectModelParentFree, updateHistory: updateHistory)
    let (reconY, r0Y) = reconstructPlaneBase8(blocks: base8YBlocks, width: dx, height: dy, qt: qtY, pool: pool)

    let safeThresholdC = min(8, max(0, (zeroThreshold / 8) - (Int(qtC.step) / 32)))
    let (bufCb, cbZeros) = encodePlaneBaseSubbands8PFrameWithSkipMap(blocks: &base8CbBlocks, zeroThreshold: safeThresholdC, isSkip: cSkip, isTreez: isTreezCb, history: histories?[1], selectModel: unifiedSelectModelParentFree, updateHistory: updateHistory)
    let (reconCb, r0Cb) = reconstructPlaneBase8(blocks: base8CbBlocks, width: cbDx, height: cbDy, qt: qtC, pool: pool)

    let (bufCr, crZeros) = encodePlaneBaseSubbands8PFrameWithSkipMap(blocks: &base8CrBlocks, zeroThreshold: safeThresholdC, isSkip: cSkip, isTreez: isTreezCr, history: histories?[2], selectModel: unifiedSelectModelParentFree, updateHistory: updateHistory)
    let (reconCr, r0Cr) = reconstructPlaneBase8(blocks: base8CrBlocks, width: cbDx, height: cbDy, qt: qtC, pool: pool)

    let reconstructed = PlaneData420(width: dx, height: dy, y: reconY, cb: reconCb, cr: reconCr)

    debugLog({
        return "  [Layer 0/Base] Y=\(bufY.count) Cb=\(bufCb.count) Cr=\(bufCr.count) bytes"
    }())

    let out = VEVCLayerData.serialize(
        qtYStep: UInt16(qtY.step), qtCStep: UInt16(qtC.step),
        bufY: bufY, bufCb: bufCb, bufCr: bufCr
    )

    return (out, reconstructed, {
        r0Y()
        r0Cb()
        r0Cr()
    }, yZeros, cbZeros, crZeros)
}

/// Base8 encode, P-frame profile 0x02: skip-block bypass (One-Pyramid §5),
/// SAD-gated luma clearing, parent-free static tables, and backward-adaptive
/// history streams.
@inline(__always)
func encodePlaneBase8PFrameWithSkipMap(pd: PlaneData420, pool: BlockViewPool, sads: [Int], qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, skipMap: [BlockMode], skipMapWidth: Int, isTreezY: [Bool]? = nil, isTreezCb: [Bool]? = nil, isTreezCr: [Bool]? = nil, histories: [EntropyHistoryState]?, updateHistory: Bool = true) async -> ([UInt8], PlaneData420, [BlockView], [BlockView], [BlockView], @Sendable () -> Void, [Bool], [Bool], [Bool]) {
    let (base8YBlocks, base8CbBlocks, base8CrBlocks, releaseBlocks) = await preparePlaneBase8WithSkipMap(
        pd: pd, pool: pool, sads: sads,
        qtY: qtY, qtC: qtC,
        skipMap: skipMap, skipMapWidth: skipMapWidth
    )
    var mutYBlocks = base8YBlocks
    var mutCbBlocks = base8CbBlocks
    var mutCrBlocks = base8CrBlocks

    let (out, reconstructed, releaseRecon, yZeros, cbZeros, crZeros) = serializePlaneBase8PFrameWithSkipMap(
        pd: pd, pool: pool,
        qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold,
        base8YBlocks: &mutYBlocks, base8CbBlocks: &mutCbBlocks, base8CrBlocks: &mutCrBlocks,
        skipMap: skipMap, skipMapWidth: skipMapWidth,
        isTreezY: isTreezY, isTreezCb: isTreezCb, isTreezCr: isTreezCr,
        histories: histories,
        updateHistory: updateHistory
    )

    return (out, reconstructed, mutYBlocks, mutCbBlocks, mutCrBlocks, {
        releaseRecon()
        releaseBlocks()
    }, yZeros, cbZeros, crZeros)
}
