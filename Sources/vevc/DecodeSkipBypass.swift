// MARK: - Skip-block reconstruction bypass (Profile 0x02, One-Pyramid §5)
//
// Skip blocks carry all-zero coefficients by construction (the encoder zeroes
// their residual before the DWT), so their dequant + inverse DWT is pure
// waste. Two bypass levels, both bit-exact with the full pipeline:
//
// - Base8: the entropy stage already leaves skip blocks as cleared views, so
//   the bypass copies the zero block into the plane without dequant/IDWT.
//   Output identical for every maxLayer (dequant and the integer lifting map
//   zero to zero).
// - Layer1: skip blocks are not reconstructed at all. Their region of the
//   half-res plane stays unwritten, which is only legal when layer2 is
//   decoded: the layer2 reconstruction skips exactly the same block indices
//   (decodeLayer32Process*WithSkipMap), so the garbage is never read, and the
//   full-resolution skip regions are filled by the final skip copy. The
//   caller therefore passes the skip map here only when layer2 is present.
//
// Index mapping: luma blocks of every layer grid are 1:1 with the skip map
// (both are ceil(dx/32) × ceil(dy/32)); layer1/layer2 chroma blocks are also
// 1:1 with each other, so direct indexing keeps the encoder, layer1 and
// layer2 consistent. Base8 chroma blocks span 2×2 skip-map entries
// geometrically, so their bypass requires all four to be non-inter
// (base8ChromaAllSkip) — that is exactly the region the encoder zeroed.

@inline(__always)
func chromaAllSkip(_ map: UnsafeBufferPointer<BlockMode>, bw: Int, bh: Int, c: Int, r: Int) -> Bool {
    for dy in 0..<2 {
        let ly = min(2 * r + dy, bh - 1)
        for dx in 0..<2 {
            let lx = min(2 * c + dx, bw - 1)
            if map[ly * bw + lx] == .inter {
                return false
            }
        }
    }
    return true
}

@inline(__always)
func base8ChromaAllSkip(skipMap: [BlockMode], bw: Int, bh: Int, c: Int, r: Int) -> Bool {
    for dy in 0..<2 {
        let ly = min(2 * r + dy, bh - 1)
        for dx in 0..<2 {
            let lx = min(2 * c + dx, bw - 1)
            if skipMap[ly * bw + lx] == .inter {
                return false
            }
        }
    }
    return true
}

@Sendable @inline(__always)
func decodeBase8ProcessYWithSkipMap(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], qt: QuantizationTable, skipMap: [BlockMode], sub: inout Image16) async {
    let concurrency = min(rowCount, 4)
    let chunkSizeSlice = (rowCount + concurrency - 1) / concurrency
    let subConst = sub
    let sCount = skipMap.count
    let sSkip = skipMap.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let sDest = sub.withUnsafeY { UnsafeSendableMutablePointer(ptr: $0) }

    // Structured concurrency instead of DispatchQueue.concurrentPerform (see ProcessY note).
    let sBlocks = blocks.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let work: @Sendable (Int) -> Void = { tIdx in
                let startRow: Int = tIdx * chunkSizeSlice
                let endRow: Int = min(startRow + chunkSizeSlice, rowCount)
                guard startRow < endRow else { return }
                let destPtr = sDest.ptr
                for i in startRow..<endRow {
                    let h: Int = i * 8
                    let rowOffset = i * colCount
                    for xIdx in 0..<colCount {
                        let w = xIdx * 8
                        let blockIndex: Int = rowOffset &+ xIdx
                        var blk: BlockView = sBlocks.ptr[blockIndex]
                        if blockIndex < sCount && sSkip.ptr[blockIndex] != .inter {
                            // Skip block: coefficients are all zero — copy the
                            // cleared view as-is.
                            subConst.updateY(destBase: destPtr, data: &blk, startX: w, startY: h, size: 8)
                            continue
                        }
                        let base = blk.base
                        dequantizeDPCM(ptr: base, stride: 8, q: qt.qLow)
                        dequantize4(ptr: base.advanced(by: 4), stride: 8, q: qt.qMid)
                        dequantize4(ptr: base.advanced(by: 32), stride: 8, q: qt.qMid)
                        dequantize4(ptr: base.advanced(by: 36), stride: 8, q: qt.qHigh)
                        inverseDWT2DBlock8(ptr: base, stride: 8)
                        subConst.updateY(destBase: destPtr, data: &blk, startX: w, startY: h, size: 8)
                    }
                }
    }
    if rowCount * colCount < 1500 {
        for tIdx in 0..<concurrency {
            work(tIdx)
        }
        return
    }
    await withTaskGroup(of: Void.self) { group in
        for tIdx in 1..<concurrency {
            group.addTask { work(tIdx) }
        }
        work(0)
    }
}

@Sendable @inline(__always)
func decodeBase8ProcessCbWithSkipMap(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], qt: QuantizationTable, skipMap: [BlockMode], skipBw: Int, skipBh: Int, sub: inout Image16) async {
    let concurrency = min(rowCount, 4)
    let chunkSizeSlice = (rowCount + concurrency - 1) / concurrency
    let subConst = sub
    let skipArr = skipMap
    let sDest = sub.withUnsafeCb { UnsafeSendableMutablePointer(ptr: $0) }

    let sBlocks = blocks.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let work: @Sendable (Int) -> Void = { tIdx in
                let startRow: Int = tIdx * chunkSizeSlice
                let endRow: Int = min(startRow + chunkSizeSlice, rowCount)
                guard startRow < endRow else { return }
                let destPtr = sDest.ptr
                for i in startRow..<endRow {
                    let h: Int = i * 8
                    let rowOffset = i * colCount
                    for xIdx in 0..<colCount {
                        let w = xIdx * 8
                        let blockIndex: Int = rowOffset &+ xIdx
                        var blk: BlockView = sBlocks.ptr[blockIndex]
                        if base8ChromaAllSkip(skipMap: skipArr, bw: skipBw, bh: skipBh, c: xIdx, r: i) {
                            subConst.updateCb(destBase: destPtr, data: &blk, startX: w, startY: h, size: 8)
                            continue
                        }
                        let base = blk.base
                        dequantizeDPCM(ptr: base, stride: 8, q: qt.qLow)
                        dequantize4(ptr: base.advanced(by: 4), stride: 8, q: qt.qMid)
                        dequantize4(ptr: base.advanced(by: 32), stride: 8, q: qt.qMid)
                        dequantize4(ptr: base.advanced(by: 36), stride: 8, q: qt.qHigh)
                        inverseDWT2DBlock8(ptr: base, stride: 8)
                        subConst.updateCb(destBase: destPtr, data: &blk, startX: w, startY: h, size: 8)
                    }
                }
    }
    if rowCount * colCount < 1500 {
        for tIdx in 0..<concurrency {
            work(tIdx)
        }
        return
    }
    await withTaskGroup(of: Void.self) { group in
        for tIdx in 1..<concurrency {
            group.addTask { work(tIdx) }
        }
        work(0)
    }
}

@Sendable @inline(__always)
func decodeBase8ProcessCrWithSkipMap(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], qt: QuantizationTable, skipMap: [BlockMode], skipBw: Int, skipBh: Int, sub: inout Image16) async {
    let concurrency = min(rowCount, 4)
    let chunkSizeSlice = (rowCount + concurrency - 1) / concurrency
    let subConst = sub
    let skipArr = skipMap
    let sDest = sub.withUnsafeCr { UnsafeSendableMutablePointer(ptr: $0) }

    let sBlocks = blocks.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let work: @Sendable (Int) -> Void = { tIdx in
                let startRow: Int = tIdx * chunkSizeSlice
                let endRow: Int = min(startRow + chunkSizeSlice, rowCount)
                guard startRow < endRow else { return }
                let destPtr = sDest.ptr
                for i in startRow..<endRow {
                    let h: Int = i * 8
                    let rowOffset = i * colCount
                    for xIdx in 0..<colCount {
                        let w = xIdx * 8
                        let blockIndex: Int = rowOffset &+ xIdx
                        var blk: BlockView = sBlocks.ptr[blockIndex]
                        if base8ChromaAllSkip(skipMap: skipArr, bw: skipBw, bh: skipBh, c: xIdx, r: i) {
                            subConst.updateCr(destBase: destPtr, data: &blk, startX: w, startY: h, size: 8)
                            continue
                        }
                        let base = blk.base
                        dequantizeDPCM(ptr: base, stride: 8, q: qt.qLow)
                        dequantize4(ptr: base.advanced(by: 4), stride: 8, q: qt.qMid)
                        dequantize4(ptr: base.advanced(by: 32), stride: 8, q: qt.qMid)
                        dequantize4(ptr: base.advanced(by: 36), stride: 8, q: qt.qHigh)
                        inverseDWT2DBlock8(ptr: base, stride: 8)
                        subConst.updateCr(destBase: destPtr, data: &blk, startX: w, startY: h, size: 8)
                    }
                }
    }
    if rowCount * colCount < 1500 {
        for tIdx in 0..<concurrency {
            work(tIdx)
        }
        return
    }
    await withTaskGroup(of: Void.self) { group in
        for tIdx in 1..<concurrency {
            group.addTask { work(tIdx) }
        }
        work(0)
    }
}

@Sendable @inline(__always)
func decodeLayer16ProcessYWithSkipMap(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable, skipMap: [BlockMode], sub: inout Image16) async {
    let concurrency = min(rowCount, 4)
    let chunkSizeSlice = (rowCount + concurrency - 1) / concurrency
    let subConst = sub
    let sCount = skipMap.count
    let sSkip = skipMap.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let sPrev = prev.withUnsafeYReadOnly { UnsafeSendablePointer(ptr: $0) }
    let sDest = sub.withUnsafeY { UnsafeSendableMutablePointer(ptr: $0) }

    let sBlocks = blocks.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let work: @Sendable (Int) -> Void = { tIdx in
                let startRow: Int = tIdx * chunkSizeSlice
                let endRow: Int = min(startRow + chunkSizeSlice, rowCount)
                guard startRow < endRow else { return }
                let prevPtr = sPrev.ptr
                let destPtr = sDest.ptr
                for i in startRow..<endRow {
                    let h: Int = i * 16
                    let rowOffset = i * colCount
                    for xIdx in 0..<colCount {
                        let blockIndex: Int = rowOffset &+ xIdx
                        if blockIndex < sCount && sSkip.ptr[blockIndex] != .inter {
                            continue
                        }
                        let w = xIdx * 16
                        let block: BlockView = sBlocks.ptr[blockIndex]
                        let base = block.base
                        prev.readYDirect(srcBase: prevPtr, x: w / 2, y: h / 2, size: 8, into: block)
                        dequantize8(ptr: base.advanced(by: 8), stride: 16, q: qt.qMid)
                        dequantize8(ptr: base.advanced(by: 128), stride: 16, q: qt.qMid)
                        dequantize8(ptr: base.advanced(by: 136), stride: 16, q: qt.qHigh)
                        inverseDWT2DBlock16(ptr: base, stride: 16)
                        var blk = block
                        subConst.updateY(destBase: destPtr, data: &blk, startX: w, startY: h, size: 16)
                    }
                }
    }
    if rowCount * colCount < 1500 {
        for tIdx in 0..<concurrency {
            work(tIdx)
        }
        return
    }
    await withTaskGroup(of: Void.self) { group in
        for tIdx in 1..<concurrency {
            group.addTask { work(tIdx) }
        }
        work(0)
    }
}

@Sendable @inline(__always)
func decodeLayer16ProcessCbWithSkipMap(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable, skipMap: [BlockMode], skipBw: Int, skipBh: Int, sub: inout Image16) async {
    let concurrency = min(rowCount, 4)
    let chunkSizeSlice = (rowCount + concurrency - 1) / concurrency
    let subConst = sub
    let sSkip = skipMap.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let sPrev = prev.withUnsafeCbReadOnly { UnsafeSendablePointer(ptr: $0) }
    let sDest = sub.withUnsafeCb { UnsafeSendableMutablePointer(ptr: $0) }

    let sBlocks = blocks.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let work: @Sendable (Int) -> Void = { tIdx in
                let startRow: Int = tIdx * chunkSizeSlice
                let endRow: Int = min(startRow + chunkSizeSlice, rowCount)
                guard startRow < endRow else { return }
                let prevPtr = sPrev.ptr
                let destPtr = sDest.ptr
                for i in startRow..<endRow {
                    let h: Int = i * 16
                    let rowOffset = i * colCount
                    for xIdx in 0..<colCount {
                        let blockIndex: Int = rowOffset &+ xIdx
                        // Chroma blocks span 2×2 luma-geometry skip-map entries.
                        if chromaAllSkip(sSkip.ptr, bw: skipBw, bh: skipBh, c: xIdx, r: i) {
                            continue
                        }
                        let w = xIdx * 16
                        let block: BlockView = sBlocks.ptr[blockIndex]
                        let base = block.base
                        prev.readCbDirect(srcBase: prevPtr, x: w / 2, y: h / 2, size: 8, into: block)
                        dequantize8(ptr: base.advanced(by: 8), stride: 16, q: qt.qMid)
                        dequantize8(ptr: base.advanced(by: 128), stride: 16, q: qt.qMid)
                        dequantize8(ptr: base.advanced(by: 136), stride: 16, q: qt.qHigh)
                        inverseDWT2DBlock16(ptr: base, stride: 16)
                        var blk = block
                        subConst.updateCb(destBase: destPtr, data: &blk, startX: w, startY: h, size: 16)
                    }
                }
    }
    if rowCount * colCount < 1500 {
        for tIdx in 0..<concurrency {
            work(tIdx)
        }
        return
    }
    await withTaskGroup(of: Void.self) { group in
        for tIdx in 1..<concurrency {
            group.addTask { work(tIdx) }
        }
        work(0)
    }
}

@Sendable @inline(__always)
func decodeLayer16ProcessCrWithSkipMap(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable, skipMap: [BlockMode], skipBw: Int, skipBh: Int, sub: inout Image16) async {
    let concurrency = min(rowCount, 4)
    let chunkSizeSlice = (rowCount + concurrency - 1) / concurrency
    let subConst = sub
    let sSkip = skipMap.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let sPrev = prev.withUnsafeCrReadOnly { UnsafeSendablePointer(ptr: $0) }
    let sDest = sub.withUnsafeCr { UnsafeSendableMutablePointer(ptr: $0) }

    let sBlocks = blocks.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let work: @Sendable (Int) -> Void = { tIdx in
                let startRow: Int = tIdx * chunkSizeSlice
                let endRow: Int = min(startRow + chunkSizeSlice, rowCount)
                guard startRow < endRow else { return }
                let prevPtr = sPrev.ptr
                let destPtr = sDest.ptr
                for i in startRow..<endRow {
                    let h: Int = i * 16
                    let rowOffset = i * colCount
                    for xIdx in 0..<colCount {
                        let blockIndex: Int = rowOffset &+ xIdx
                        // Chroma blocks span 2×2 luma-geometry skip-map entries.
                        if chromaAllSkip(sSkip.ptr, bw: skipBw, bh: skipBh, c: xIdx, r: i) {
                            continue
                        }
                        let w = xIdx * 16
                        let block: BlockView = sBlocks.ptr[blockIndex]
                        let base = block.base
                        prev.readCrDirect(srcBase: prevPtr, x: w / 2, y: h / 2, size: 8, into: block)
                        dequantize8(ptr: base.advanced(by: 8), stride: 16, q: qt.qMid)
                        dequantize8(ptr: base.advanced(by: 128), stride: 16, q: qt.qMid)
                        dequantize8(ptr: base.advanced(by: 136), stride: 16, q: qt.qHigh)
                        inverseDWT2DBlock16(ptr: base, stride: 16)
                        var blk = block
                        subConst.updateCr(destBase: destPtr, data: &blk, startX: w, startY: h, size: 16)
                    }
                }
    }
    if rowCount * colCount < 1500 {
        for tIdx in 0..<concurrency {
            work(tIdx)
        }
        return
    }
    await withTaskGroup(of: Void.self) { group in
        for tIdx in 1..<concurrency {
            group.addTask { work(tIdx) }
        }
        work(0)
    }
}
