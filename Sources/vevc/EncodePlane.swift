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
func extractSingleTransformBlocksBase8WithSkipMap(r: Int16Reader, width: Int, height: Int, pool: BlockViewPool, isSkip: [Bool]) async -> (blocks: [BlockView], releaseFn: @Sendable () -> Void) {
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
            group.addTask { [blocks, isSkip, r] in
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

@inline(__always)
func entropyEncodeLayer32(dx: Int, dy: Int, layer: UInt8, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, isPFrame: Bool = false, yBlocks: inout [BlockView], cbBlocks: inout [BlockView], crBlocks: inout [BlockView], parentYBlocks: [BlockView]?, parentCbBlocks: [BlockView]?, parentCrBlocks: [BlockView]?, sads: [Int]? = nil, histories: [EntropyHistoryState]? = nil, selectModel: ModelSelectorFn = unifiedSelectModel) -> [UInt8] {
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
    
    let bufY = encodePlaneSubbands32(blocks: &yBlocks, zeroThreshold: safeThresholdY, parentBlocks: parentYBlocks, sads: sads, colCount: colCountY, rowCount: rowCountY, history: histories?[0], selectModel: selectModel)
    let bufCb = encodePlaneSubbands32(blocks: &cbBlocks, zeroThreshold: safeThresholdC, parentBlocks: parentCbBlocks, colCount: colCountC, rowCount: rowCountC, history: histories?[1], selectModel: selectModel)
    let bufCr = encodePlaneSubbands32(blocks: &crBlocks, zeroThreshold: safeThresholdC, parentBlocks: parentCrBlocks, colCount: colCountC, rowCount: rowCountC, history: histories?[2], selectModel: selectModel)
    
    debugLog({
        return "  [Layer \\(layer)] qtY=\\(qtY.step), qtC=\\(qtC.step) Y=\\(bufY.count) Cb=\\(bufCb.count) Cr=\\(bufCr.count) bytes"
    }())
    
    return VEVCLayerData.serialize(
        qtYStep: UInt16(qtY.step), qtCStep: UInt16(qtC.step),
        bufY: bufY, bufCb: bufCb, bufCr: bufCr
    )
}

@inline(__always)
func entropyEncodeLayer16(dx: Int, dy: Int, layer: UInt8, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, isPFrame: Bool = false, yBlocks: inout [BlockView], cbBlocks: inout [BlockView], crBlocks: inout [BlockView], parentYBlocks: [BlockView]?, parentCbBlocks: [BlockView]?, parentCrBlocks: [BlockView]?, sads: [Int]? = nil, occlusionScores: [Int]? = nil, histories: [EntropyHistoryState]? = nil, selectModel: ModelSelectorFn = unifiedSelectModel) -> [UInt8] {
    let safeThresholdY = min(2, min(zeroThreshold, max(0, Int(qtY.step) / 64)))
    let safeThresholdC = min(8, min(zeroThreshold, max(0, Int(qtC.step) / 64)))
    
    let colCountY = (dx + 15) / 16
    let rowCountY = (dy + 15) / 16
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let colCountC = (cbDx + 15) / 16
    let rowCountC = (cbDy + 15) / 16
    
    // Note: SADs are evaluated at 32x32 granularity, so map Layer16 to Layer32 granularity
    // In layered structure, we just pass sads arrays if aligned, or map if necessary.
    // For now, only 32x32 blocks use it cleanly, but if Layer16 needs it:
    let bufY = encodePlaneSubbands16(blocks: &yBlocks, zeroThreshold: safeThresholdY, parentBlocks: parentYBlocks, sads: sads, occlusionScores: occlusionScores, colCount: colCountY, rowCount: rowCountY, history: histories?[0], selectModel: selectModel)
    let bufCb = encodePlaneSubbands16(blocks: &cbBlocks, zeroThreshold: safeThresholdC, parentBlocks: parentCbBlocks, colCount: colCountC, rowCount: rowCountC, history: histories?[1], selectModel: selectModel)
    let bufCr = encodePlaneSubbands16(blocks: &crBlocks, zeroThreshold: safeThresholdC, parentBlocks: parentCrBlocks, colCount: colCountC, rowCount: rowCountC, history: histories?[2], selectModel: selectModel)
    
    debugLog({
        return "  [Layer \\(layer)] qtY=\\(qtY.step), qtC=\\(qtC.step) Y=\\(bufY.count) Cb=\\(bufCb.count) Cr=\\(bufCr.count) bytes"
    }())
    
    return VEVCLayerData.serialize(
        qtYStep: UInt16(qtY.step), qtCStep: UInt16(qtC.step),
        bufY: bufY, bufCb: bufCb, bufCr: bufCr
    )
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
func reconstructPlaneLayer32Cb(blocks: [BlockView], prevImg: Image16, width: Int, height: Int, qt: QuantizationTable, pool: BlockViewPool, skipMap: [BlockMode]? = nil) -> ([Int16], @Sendable () -> Void) {
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
func reconstructPlaneLayer32Cr(blocks: [BlockView], prevImg: Image16, width: Int, height: Int, qt: QuantizationTable, pool: BlockViewPool, skipMap: [BlockMode]? = nil) -> ([Int16], @Sendable () -> Void) {
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

@inline(__always)
/// Base8 encode, I-frame: static-table entropy coding (DPCM handled inside
/// encodePlaneBaseSubbands8 via blockEncodeDPCM4/MED), no SAD gating, no
/// history state. selectModel picks the profile's static AC tables.
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

/// Base8 encode, P-frame profile 0x02: skip-block bypass (One-Pyramid §5),
/// SAD-gated luma clearing, parent-free static tables, and backward-adaptive
/// history streams.
func encodePlaneBase8PFrameWithSkipMap(pd: PlaneData420, pool: BlockViewPool, sads: [Int], qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, skipMap: [BlockMode], skipMapWidth: Int, histories: [EntropyHistoryState]?) async -> ([UInt8], PlaneData420, [BlockView], [BlockView], [BlockView], @Sendable () -> Void) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)

    let yColCount8 = (dx + 7) / 8
    let yRowCount8 = (dy + 7) / 8

    let ySkip = lumaSkipFlags(skipMap: skipMap, mapWidth: skipMapWidth, rowCount: yRowCount8, colCount: yColCount8)
    let cSkip = chromaSkipFlags(skipMap: skipMap, mapWidth: skipMapWidth, rowCount: (cbDy + 7) / 8, colCount: (cbDx + 7) / 8)

    async let taskBufY = { [ySkip] () -> ([UInt8], [Int16], @Sendable () -> Void, [BlockView], @Sendable () -> Void) in
        var (blocks, relBlocks) = await extractSingleTransformBlocksBase8WithSkipMap(r: pd.rY, width: dx, height: dy, pool: pool, isSkip: ySkip)
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
        let buf = encodePlaneBaseSubbands8PFrame(blocks: &blocks, zeroThreshold: safeThreshold, history: histories?[0], selectModel: unifiedSelectModelParentFree)

        let quantizedBlocks = blocks
        let (reconPlane, rPlane) = reconstructPlaneBase8(blocks: blocks, width: dx, height: dy, qt: qtY, pool: pool)
        return (buf, reconPlane, rPlane, quantizedBlocks, relBlocks)
    }()

    async let taskBufCb = { [cSkip] () -> ([UInt8], [Int16], @Sendable () -> Void, [BlockView], @Sendable () -> Void) in
        var (blocks, relBlocks) = await extractSingleTransformBlocksBase8WithSkipMap(r: pd.rCb, width: cbDx, height: cbDy, pool: pool, isSkip: cSkip)
        for i in blocks.indices {
            evaluateQuantizeBase8(view: blocks[i], qt: qtC)
        }

        let safeThreshold = min(8, max(0, (zeroThreshold / 8) - (Int(qtC.step)  / 32)))
        let buf = encodePlaneBaseSubbands8PFrame(blocks: &blocks, zeroThreshold: safeThreshold, history: histories?[1], selectModel: unifiedSelectModelParentFree)

        let quantizedBlocks = blocks
        let (reconPlane, rPlane) = reconstructPlaneBase8(blocks: blocks, width: cbDx, height: cbDy, qt: qtC, pool: pool)
        return (buf, reconPlane, rPlane, quantizedBlocks, relBlocks)
    }()

    async let taskBufCr = { [cSkip] () -> ([UInt8], [Int16], @Sendable () -> Void, [BlockView], @Sendable () -> Void) in
        var (blocks, relBlocks) = await extractSingleTransformBlocksBase8WithSkipMap(r: pd.rCr, width: cbDx, height: cbDy, pool: pool, isSkip: cSkip)
        for i in blocks.indices {
            evaluateQuantizeBase8(view: blocks[i], qt: qtC)
        }

        let safeThreshold = min(8, max(0, (zeroThreshold / 8) - (Int(qtC.step) / 32)))
        let buf = encodePlaneBaseSubbands8PFrame(blocks: &blocks, zeroThreshold: safeThreshold, history: histories?[2], selectModel: unifiedSelectModelParentFree)

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
