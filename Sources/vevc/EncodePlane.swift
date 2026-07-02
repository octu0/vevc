// MARK: - AC Energy Measurement for Adaptive Quantization

/// Measures the AC energy of a DWT-transformed 32x32 block.
/// AC energy = sum of |HL| + |LH| + |HH| coefficients.
/// This measures the "complexity" of the block: high energy = edges/textures,
/// low energy = flat/smooth regions.
///
/// Layout after DWT of 32x32 block (stride=32):
///   [LL 16x16] [HL 16x16]
///   [LH 16x16] [HH 16x16]
/// HL starts at base+16, LH at base+16*32, HH at base+16*32+16
@inline(__always)
func measureACEnergy32(view: BlockView) -> Int {
    let base = view.base
    let s = 32 // stride
    var totalSum: Int = 0
    
    // HL subband: rows 0..15, cols 16..31
    for y in 0..<16 {
        let ptr = base + y * s + 16
        for x in 0..<16 {
            let v = Int(ptr[x])
            let mask = v &>> 31
            totalSum &+= (v ^ mask) &- mask
        }
    }
    
    // LH subband: rows 16..31, cols 0..15
    for y in 16..<32 {
        let ptr = base + y * s
        for x in 0..<16 {
            let v = Int(ptr[x])
            let mask = v &>> 31
            totalSum &+= (v ^ mask) &- mask
        }
    }
    
    // HH subband: rows 16..31, cols 16..31
    for y in 16..<32 {
        let ptr = base + y * s + 16
        for x in 0..<16 {
            let v = Int(ptr[x])
            let mask = v &>> 31
            totalSum &+= (v ^ mask) &- mask
        }
    }
    
    return totalSum
}

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
func evaluateQuantizeLayer32(view: BlockView, qt: QuantizationTable) {
    let subs = getSubbands32(view: view)
    let hl = subs.hl
    let lh = subs.lh
    let hh = subs.hh
    quantizeSIMDSignedMapping16(hl, q: qt.qMid)
    quantizeSIMDSignedMapping16(lh, q: qt.qMid)
    quantizeSIMDSignedMapping16(hh, q: qt.qHigh)
}

@inline(__always)
func evaluateQuantizeLayer16(view: BlockView, qt: QuantizationTable) {
    let subs = getSubbands16(view: view)
    let hl = subs.hl
    let lh = subs.lh
    let hh = subs.hh
    quantizeSIMDSignedMapping8(hl, q: qt.qMid)
    quantizeSIMDSignedMapping8(lh, q: qt.qMid)
    quantizeSIMDSignedMapping8(hh, q: qt.qHigh)
}

@inline(__always)
func evaluateQuantizeBase8(view: BlockView, qt: QuantizationTable) {
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
func evaluateQuantizeBase32(view: BlockView, qt: QuantizationTable) {
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
func extractSingleTransformBlocks32AQ(r: Int16Reader, width: Int, height: Int, pool: BlockViewPool, aqTable: AQTable, sads: [Int]? = nil, occlusionScores: [Int]? = nil) async -> (blocks: [BlockView], subband: [Int16], levels: [UInt8], releaseFn: @Sendable () -> Void) {
    let subWidth = ((width + 1) / 2)
    let subHeight = ((height + 1) / 2)
    var subband = pool.getInt16(count: subWidth * subHeight)
    let safeDst = withUnsafePointers(mut: &subband) { SendableInt16Ptr($0) }
    
    let rowCount = ((height + 32 - 1) / 32)
    let colCount = ((width + 32 - 1) / 32)
    let totalBlocks = rowCount * colCount
    
    var tmpBlocks = pool.getBlockViewArray(capacity: totalBlocks)
    tmpBlocks.reserveCapacity(totalBlocks)
    for _ in 0..<totalBlocks {
        tmpBlocks.append(pool.get(width: 32, height: 32))
    }
    let blocks = tmpBlocks
    
    let chunkSize = 8
    
    let energyBox = ConcurrentBox([Int](repeating: 0, count: totalBlocks))
    let levelsBox = ConcurrentBox([UInt8](repeating: 2, count: totalBlocks))
    let safeEnergyBox = energyBox
    let safeLevelsBox = levelsBox
    await withTaskGroup(of: Void.self) { group in
        for sRow in stride(from: 0, to: rowCount, by: chunkSize) {
            let endRow = min(sRow + chunkSize, rowCount)
            group.addTask { [blocks, safeDst, safeEnergyBox] in
                let dstBase = safeDst.ptr
                for i in sRow..<endRow {
                    let h = (i * 32)
                    for j in 0..<colCount {
                        let w = (j * 32)
                        if width <= w || height <= h { continue }
                        let blockIdx = (i * colCount) + j
                        let view = blocks[blockIdx]
                        r.readBlock(x: w, y: h, width: 32, height: 32, into: view)
                        dwt2DBlock32(view)
                        
                        let sad = sads?[blockIdx] ?? 1024
                        let occ = occlusionScores?[blockIdx] ?? 0
                        
                        let isHighSAD = 1500 <= sad
                        let isHighOcc = 12 <= occ
                        
                        let isHighError = isHighSAD || isHighOcc
                        
                        if 256 < sad || isHighError {
                            var energy = measureACEnergy32(view: view)
                            // AQ injection: lower energy => finer quantization.
                            // To make occlusion finer, we can artificially lower the energy.
                            if isHighError {
                                energy = energy / 4
                            }
                            safeEnergyBox.value[blockIdx] = energy
                        } else {
                            safeEnergyBox.value[blockIdx] = -1
                        }
                        
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
                                    if dstY < subHeight {
                                        let srcPtr = srcBase.advanced(by: (blockY * 32))
                                        let dstIdx = ((dstY * subWidth) + destStartX)
                                        dstBase.advanced(by: dstIdx).update(from: srcPtr, count: limit)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let energies = energyBox.value
    var totalEnergy: Int = 0
    var validBlocks: Int = 0
    for i in 0..<totalBlocks {
        let e = energies[i]
        if e != -1 {
            totalEnergy += e
            validBlocks += 1
        }
    }
    let avgEnergy = max(1, totalEnergy / max(1, validBlocks))
    
    await withTaskGroup(of: Void.self) { group in
        for sRow in stride(from: 0, to: rowCount, by: chunkSize) {
            let endRow = min(sRow + chunkSize, rowCount)
            group.addTask { [blocks, energies, safeLevelsBox, aqTable, sads] in
                for i in sRow..<endRow {
                    let h = (i * 32)
                    for j in 0..<colCount {
                        let w = (j * 32)
                        if width <= w || height <= h { continue }
                        let blockIdx = (i * colCount) + j
                        let view = blocks[blockIdx]
                        let sad = sads?[blockIdx] ?? -1
                        let energy = energies[blockIdx]
                        
                        let blockQt: QuantizationTable
                        if energy == -1 {
                            blockQt = aqTable.base
                        } else {
                            let levelIdx = aqTable.selectIndex(energy: energy, avgEnergy: avgEnergy, sad: sad, bx: j, by: i, colCount: colCount, rowCount: rowCount)
                            blockQt = aqTable[levelIdx]
                            safeLevelsBox.value[blockIdx] = UInt8(levelIdx)
                        }
                        
                        evaluateQuantizeLayer32(view: view, qt: blockQt)
                    }
                }
            }
        }
    }
    
    withExtendedLifetime(subband) {}
    let levels = levelsBox.value
    return (tmpBlocks, subband, levels, { [tmpBlocks, subband] in pool.putBlockViewArray(tmpBlocks); pool.putInt16(subband) })
}

@inline(__always)
func extractSingleTransformBlocks32(r: Int16Reader, width: Int, height: Int, pool: BlockViewPool, qt: QuantizationTable) async -> (blocks: [BlockView], subband: [Int16], releaseFn: @Sendable () -> Void) {
    let subWidth = ((width + 1) / 2)
    let subHeight = ((height + 1) / 2)
    var subband = pool.getInt16(count: subWidth * subHeight)
    let safeDst = withUnsafePointers(mut: &subband) { SendableInt16Ptr($0) }
    
    let rowCount = ((height + 32 - 1) / 32)
    let colCount = ((width + 32 - 1) / 32)
    let totalBlocks = rowCount * colCount
    
    var tmpBlocks = pool.getBlockViewArray(capacity: totalBlocks)
    tmpBlocks.reserveCapacity(totalBlocks)
    for _ in 0..<totalBlocks {
        tmpBlocks.append(pool.get(width: 32, height: 32))
    }
    let blocks = tmpBlocks
    let chunkSize = 8
    
    await withTaskGroup(of: Void.self) { group in
        for sRow in stride(from: 0, to: rowCount, by: chunkSize) {
            let endRow = min(sRow + chunkSize, rowCount)
            group.addTask { [blocks, safeDst, qt] in
                let dstBase = safeDst.ptr
                for i in sRow..<endRow {
                    let h = (i * 32)
                    for j in 0..<colCount {
                        let w = (j * 32)
                        if width <= w || height <= h { continue }
                        let view = blocks[(i * colCount) + j]
                        r.readBlock(x: w, y: h, width: 32, height: 32, into: view)
                        dwt2DBlock32(view)
                        
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
                                    if dstY < subHeight {
                                        let srcPtr = srcBase.advanced(by: (blockY * 32))
                                        let dstIdx = ((dstY * subWidth) + destStartX)
                                        dstBase.advanced(by: dstIdx).update(from: srcPtr, count: limit)
                                    }
                                }
                            }
                        }
                        
                        evaluateQuantizeLayer32(view: view, qt: qt)
                    }
                }
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
    let rowCount = ((height + 32 - 1) / 32)
    let colCount = ((width + 32 - 1) / 32)
    let totalBlocks = rowCount * colCount
    
    var tmpBlocks = pool.getBlockViewArray(capacity: totalBlocks)
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
                        dwt2DBlock32(view)
                    }
                }
            }
        }
    }
    
    withUnsafePointers(mut: &subband) { dstBase in
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
    
    
    withExtendedLifetime(subband) {}
    return (subband, { [tmpBlocks, subband] in pool.putBlockViewArray(tmpBlocks); pool.putInt16(subband) })
}

@inline(__always)
func extractSingleTransformBlocks16(r: Int16Reader, width: Int, height: Int, pool: BlockViewPool, qt: QuantizationTable, sads: [Int]? = nil, occlusionScores: [Int]? = nil) async -> (blocks: [BlockView], subband: [Int16], releaseFn: @Sendable () -> Void) {
    let subWidth = ((width + 1) / 2)
    let subHeight = ((height + 1) / 2)
    var subband = pool.getInt16(count: subWidth * subHeight)
    let safeDst = withUnsafePointers(mut: &subband) { SendableInt16Ptr($0) }
    
    let rowCount = ((height + 16 - 1) / 16)
    let colCount = ((width + 16 - 1) / 16)
    let totalBlocks = rowCount * colCount
    
    var tmpBlocks = pool.getBlockViewArray(capacity: totalBlocks)
    tmpBlocks.reserveCapacity(totalBlocks)
    for _ in 0..<totalBlocks {
        tmpBlocks.append(pool.get(width: 16, height: 16))
    }
    let blocks = tmpBlocks
    
    let chunkSize = 4
    await withTaskGroup(of: Void.self) { group in
        for sRow in stride(from: 0, to: rowCount, by: chunkSize) {
            let endRow = min(sRow + chunkSize, rowCount)
            group.addTask { [blocks, safeDst, qt] in
                let dstBase = safeDst.ptr
                for i in sRow..<endRow {
                    let h = (i * 16)
                    for j in 0..<colCount {
                        let w = (j * 16)
                        if width <= w || height <= h { continue }
                        let view = blocks[(i * colCount) + j]
                        r.readBlock(x: w, y: h, width: 16, height: 16, into: view)
                        dwt2DBlock16(view)
                        
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
                                    if dstY < subHeight {
                                        let srcPtr = srcBase.advanced(by: (blockY * 16))
                                        let dstIdx = ((dstY * subWidth) + destStartX)
                                        dstBase.advanced(by: dstIdx).update(from: srcPtr, count: limit)
                                    }
                                }
                            }
                        }
                        
                        evaluateQuantizeLayer16(view: view, qt: qt)
                    }
                }
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
    let rowCount = (height + (16 - 1)) / 16
    let colCount = (width + (16 - 1)) / 16
    let totalBlocks = rowCount * colCount
    
    var tmpBlocks = pool.getBlockViewArray(capacity: totalBlocks)
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
                        dwt2DBlock16(view)
                    }
                }
            }
        }
    }
    
    withUnsafePointers(mut: &subband) { dstBase in
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
    
    
    withExtendedLifetime(subband) {}
    return (subband, { [tmpBlocks, subband] in pool.putBlockViewArray(tmpBlocks); pool.putInt16(subband) })
}

@inline(__always)
func extractSingleTransformBlocksBase8(r: Int16Reader, width: Int, height: Int, pool: BlockViewPool) async -> ([BlockView], @Sendable () -> Void) {
    let rowCount = ((height + 8 - 1) / 8)
    let colCount = ((width + 8 - 1) / 8)
    let totalBlocks = rowCount * colCount
    
    var tmpBlocks = pool.getBlockViewArray(capacity: totalBlocks)
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
                        dwt2DBlock8(view)
                    }
                }
            }
        }
    }    
    return (tmpBlocks, { [tmpBlocks] in pool.putBlockViewArray(tmpBlocks) })
}

@inline(__always)
func preparePlaneLayer32AQ(pd: PlaneData420, pool: BlockViewPool, sads: [Int]?, occlusionScores: [Int]?, layer: UInt8, aqYTable: AQTable, qtCTable: QuantizationTable, zeroThreshold: Int) async throws -> (PlaneData420, [BlockView], [BlockView], [BlockView], [UInt8], @Sendable () -> Void) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    
    async let taskBufY = { () -> ([Int16], [BlockView], [UInt8], @Sendable () -> Void) in
        let (blocks, subband, levels, r) = await extractSingleTransformBlocks32AQ(r: pd.rY, width: dx, height: dy, pool: pool, aqTable: aqYTable, sads: sads, occlusionScores: occlusionScores)
        return (subband, blocks, levels, r)
    }()
    
    async let taskBufCb = { () -> ([Int16], [BlockView], @Sendable () -> Void) in
        let (blocks, subband, r) = await extractSingleTransformBlocks32(r: pd.rCb, width: cbDx, height: cbDy, pool: pool, qt: qtCTable)
        return (subband, blocks, r)
    }()
    
    async let taskBufCr = { () -> ([Int16], [BlockView], @Sendable () -> Void) in
        let (blocks, subband, r) = await extractSingleTransformBlocks32(r: pd.rCr, width: cbDx, height: cbDy, pool: pool, qt: qtCTable)
        return (subband, blocks, r)
    }()

    let (subY, yBlocks, levels, relY) = await taskBufY
    let (subCb, cbBlocks, relCb) = await taskBufCb
    let (subCr, crBlocks, relCr) = await taskBufCr

    let subPlane = PlaneData420(width: (dx + 1) / 2, height: (dy + 1) / 2, y: subY, cb: subCb, cr: subCr)
    return (subPlane, yBlocks, cbBlocks, crBlocks, levels, { relY(); relCb(); relCr() })
}

@inline(__always)
func preparePlaneLayer32(pd: PlaneData420, pool: BlockViewPool, sads: [Int]?, layer: UInt8, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int) async throws -> (PlaneData420, [BlockView], [BlockView], [BlockView], @Sendable () -> Void) {
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
func preparePlaneLayer16(pd: PlaneData420, pool: BlockViewPool, sads: [Int]?, occlusionScores: [Int]?, layer: UInt8, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int) async throws -> (PlaneData420, [BlockView], [BlockView], [BlockView], @Sendable () -> Void) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    
    async let taskBufY = { () -> ([Int16], [BlockView], @Sendable () -> Void) in
        let (blocks, subband, r) = await extractSingleTransformBlocks16(r: pd.rY, width: dx, height: dy, pool: pool, qt: qtY, sads: sads, occlusionScores: occlusionScores)
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
func entropyEncodeLayer32(dx: Int, dy: Int, layer: UInt8, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, isPFrame: Bool = false, yBlocks: inout [BlockView], cbBlocks: inout [BlockView], crBlocks: inout [BlockView], parentYBlocks: [BlockView]?, parentCbBlocks: [BlockView]?, parentCrBlocks: [BlockView]?, sads: [Int]? = nil, levels: [UInt8]? = nil) -> [UInt8] {
    // Layer2 (32x32) contains the highest-frequency DWT subbands with the
    // lowest CSF sensitivity. P-frame residuals at this level can be zeroed
    // more aggressively (threshold=3) than Layer1 (threshold=2) without
    // perceptible quality loss.
    let safeThresholdY = min(3, min(zeroThreshold, max(0, Int(qtY.step) / 4)))
    let safeThresholdC = min(8, min(zeroThreshold, max(0, Int(qtC.step) / 4)))
    
    let colCountY = (dx + 31) / 32
    let rowCountY = (dy + 31) / 32
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let colCountC = (cbDx + 31) / 32
    let rowCountC = (cbDy + 31) / 32
    
    let bufY = encodePlaneSubbands32(blocks: &yBlocks, zeroThreshold: safeThresholdY, parentBlocks: parentYBlocks, sads: sads, colCount: colCountY, rowCount: rowCountY)
    let bufCb = encodePlaneSubbands32(blocks: &cbBlocks, zeroThreshold: safeThresholdC, parentBlocks: parentCbBlocks, colCount: colCountC, rowCount: rowCountC)
    let bufCr = encodePlaneSubbands32(blocks: &crBlocks, zeroThreshold: safeThresholdC, parentBlocks: parentCrBlocks, colCount: colCountC, rowCount: rowCountC)
    
    debugLog({
        return "  [Layer \\(layer)] qtY=\\(qtY.step), qtC=\\(qtC.step) Y=\\(bufY.count) Cb=\\(bufCb.count) Cr=\\(bufCr.count) bytes"
    }())
    
    let aqMap = levels.map { encodeAQMap(levels: $0) }
    
    return VEVCLayerData.serialize(
        qtYStep: UInt16(qtY.step), qtCStep: UInt16(qtC.step),
        aqMap: aqMap,
        bufY: bufY, bufCb: bufCb, bufCr: bufCr
    )
}

@inline(__always)
func entropyEncodeLayer16(dx: Int, dy: Int, layer: UInt8, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, isPFrame: Bool = false, yBlocks: inout [BlockView], cbBlocks: inout [BlockView], crBlocks: inout [BlockView], parentYBlocks: [BlockView]?, parentCbBlocks: [BlockView]?, parentCrBlocks: [BlockView]?, sads: [Int]? = nil, occlusionScores: [Int]? = nil) -> [UInt8] {
    let safeThresholdY = min(2, min(zeroThreshold, max(0, Int(qtY.step) / 4)))
    let safeThresholdC = min(8, min(zeroThreshold, max(0, Int(qtC.step) / 4)))
    
    let colCountY = (dx + 15) / 16
    let rowCountY = (dy + 15) / 16
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let colCountC = (cbDx + 15) / 16
    let rowCountC = (cbDy + 15) / 16
    
    // Note: SADs are evaluated at 32x32 granularity, so map Layer16 to Layer32 granularity
    // In layered structure, we just pass sads arrays if aligned, or map if necessary.
    // For now, only 32x32 blocks use it cleanly, but if Layer16 needs it:
    let bufY = encodePlaneSubbands16(blocks: &yBlocks, zeroThreshold: safeThresholdY, parentBlocks: parentYBlocks, sads: sads, occlusionScores: occlusionScores, colCount: colCountY, rowCount: rowCountY)
    let bufCb = encodePlaneSubbands16(blocks: &cbBlocks, zeroThreshold: safeThresholdC, parentBlocks: parentCbBlocks, colCount: colCountC, rowCount: rowCountC)
    let bufCr = encodePlaneSubbands16(blocks: &crBlocks, zeroThreshold: safeThresholdC, parentBlocks: parentCrBlocks, colCount: colCountC, rowCount: rowCountC)
    
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
                let llView = BlockView(base: base, width: 4, height: 4, stride: 8)
                let hlView = BlockView(base: base.advanced(by: 4), width: 4, height: 4, stride: 8)
                let lhView = BlockView(base: base.advanced(by: 32), width: 4, height: 4, stride: 8)
                let hhView = BlockView(base: base.advanced(by: 36), width: 4, height: 4, stride: 8)
                dequantizeSIMD4(llView, q: qt.qLow)
                dequantizeSIMDSignedMapping4(hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping4(lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping4(hhView, q: qt.qHigh)
                inverseDWT2DBlock8(view)
                            
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
func reconstructPlaneLayer32Y(blocks: [BlockView], prevImg: Image16, width: Int, height: Int, aqTable: AQTable, levels: [UInt8], pool: BlockViewPool) -> ([Int16], @Sendable () -> Void) {
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
                let level = levels[idx]
                idx += 1
                
                let llX = startX / 2
                let llY = startY / 2
                prevImg.readY(x: llX, y: llY, size: 16, into: blk)
                                        
                let view = blk
                let base = view.base
                let hlView = BlockView(base: base.advanced(by: 16), width: 16, height: 16, stride: 32)
                let lhView = BlockView(base: base.advanced(by: 16 * 32), width: 16, height: 16, stride: 32)
                let hhView = BlockView(base: base.advanced(by: 16 * 32 + 16), width: 16, height: 16, stride: 32)
                
                let qt = aqTable[Int(level)]
                dequantizeSIMDSignedMapping16(hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping16(lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping16(hhView, q: qt.qHigh)
                inverseDWT2DBlock32(view)
                            
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
func reconstructPlaneLayer32Cb(blocks: [BlockView], prevImg: Image16, width: Int, height: Int, qt: QuantizationTable, pool: BlockViewPool) -> ([Int16], @Sendable () -> Void) {
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
                inverseDWT2DBlock32(view)
                            
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
func reconstructPlaneLayer32Cr(blocks: [BlockView], prevImg: Image16, width: Int, height: Int, qt: QuantizationTable, pool: BlockViewPool) -> ([Int16], @Sendable () -> Void) {
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
                inverseDWT2DBlock32(view)
                            
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
                let hlView = BlockView(base: base.advanced(by: 8), width: 8, height: 8, stride: 16)
                let lhView = BlockView(base: base.advanced(by: 8 * 16), width: 8, height: 8, stride: 16)
                let hhView = BlockView(base: base.advanced(by: 8 * 16 + 8), width: 8, height: 8, stride: 16)
                dequantizeSIMDSignedMapping8(hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping8(lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping8(hhView, q: qt.qHigh)
                inverseDWT2DBlock16(view)
                            
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
                let hlView = BlockView(base: base.advanced(by: 8), width: 8, height: 8, stride: 16)
                let lhView = BlockView(base: base.advanced(by: 8 * 16), width: 8, height: 8, stride: 16)
                let hhView = BlockView(base: base.advanced(by: 8 * 16 + 8), width: 8, height: 8, stride: 16)
                dequantizeSIMDSignedMapping8(hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping8(lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping8(hhView, q: qt.qHigh)
                inverseDWT2DBlock16(view)
                            
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
                let hlView = BlockView(base: base.advanced(by: 8), width: 8, height: 8, stride: 16)
                let lhView = BlockView(base: base.advanced(by: 8 * 16), width: 8, height: 8, stride: 16)
                let hhView = BlockView(base: base.advanced(by: 8 * 16 + 8), width: 8, height: 8, stride: 16)
                dequantizeSIMDSignedMapping8(hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping8(lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping8(hhView, q: qt.qHigh)
                inverseDWT2DBlock16(view)
                            
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
func encodePlaneBase8(pd: PlaneData420, pool: BlockViewPool, sads: [Int]?, occlusionScores: [Int]?, layer: UInt8, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int) async throws -> ([UInt8], PlaneData420, [BlockView], [BlockView], [BlockView], @Sendable () -> Void) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    
    let yColCount8 = (dx + 7) / 8
    let yRowCount8 = (dy + 7) / 8
    
    async let taskBufY = { () -> ([UInt8], [Int16], @Sendable () -> Void, [BlockView], @Sendable () -> Void) in
        var (blocks, relBlocks) = await extractSingleTransformBlocksBase8(r: pd.rY, width: dx, height: dy, pool: pool)
        let isIFrame = (sads == nil)
        for i in blocks.indices {
            if let sList = sads, i < sList.count {
                let col = i % yColCount8
                let row = i / yColCount8
                let threshold = spatialSADThreshold(baseSAD: scaledSADThreshold(150, step: Int(qtY.step)), blockCol: col, blockRow: row, colCount: yColCount8, rowCount: yRowCount8)
                if sList[i] < threshold { 
                    let b = blocks[i]
                    clearBlockRegion(base: b.base, width: b.width, height: b.height, stride: b.stride)
                }
            }
            evaluateQuantizeBase8(view: blocks[i], qt: qtY)
        }
        
        // DPCM is already perfectly handled inside encodePlaneBaseSubbands8 via blockEncodeDPCM4 (MED)
        
        // P-frame Base8: apply safeThreshold to zero out imperceptible residuals
        let safeThreshold = min(1, min(zeroThreshold, max(0, Int(qtY.step) / 4)))
        let buf = if isIFrame != true {
            encodePlaneBaseSubbands8PFrame(blocks: &blocks, zeroThreshold: safeThreshold)
        } else {
            encodePlaneBaseSubbands8(blocks: &blocks, zeroThreshold: safeThreshold)
        }
        
        let quantizedBlocks = blocks
        let (reconPlane, rPlane) = reconstructPlaneBase8(blocks: blocks, width: dx, height: dy, qt: qtY, pool: pool)
        return (buf, reconPlane, rPlane, quantizedBlocks, relBlocks)
    }()
    
    
    async let taskBufCb = { () -> ([UInt8], [Int16], @Sendable () -> Void, [BlockView], @Sendable () -> Void) in
        var (blocks, relBlocks) = await extractSingleTransformBlocksBase8(r: pd.rCb, width: cbDx, height: cbDy, pool: pool)
        let isIFrame = (sads == nil)
        for i in blocks.indices {
            evaluateQuantizeBase8(view: blocks[i], qt: qtC)
        }
        
        let safeThreshold = min(8, max(0, (zeroThreshold / 8) - (Int(qtC.step)  / 2)))
        let buf = if isIFrame != true {
            encodePlaneBaseSubbands8PFrame(blocks: &blocks, zeroThreshold: safeThreshold)
        } else {
            encodePlaneBaseSubbands8(blocks: &blocks, zeroThreshold: safeThreshold)
        }
        
        let quantizedBlocks = blocks
        let (reconPlane, rPlane) = reconstructPlaneBase8(blocks: blocks, width: cbDx, height: cbDy, qt: qtC, pool: pool)
        return (buf, reconPlane, rPlane, quantizedBlocks, relBlocks)
    }()
    
    async let taskBufCr = { () -> ([UInt8], [Int16], @Sendable () -> Void, [BlockView], @Sendable () -> Void) in
        var (blocks, relBlocks) = await extractSingleTransformBlocksBase8(r: pd.rCr, width: cbDx, height: cbDy, pool: pool)
        let isIFrame = (sads == nil)
        for i in blocks.indices {
            evaluateQuantizeBase8(view: blocks[i], qt: qtC)
        }
        
        let safeThreshold = min(8, max(0, (zeroThreshold / 8) - (Int(qtC.step) / 2)))
        let buf = if isIFrame != true {
            encodePlaneBaseSubbands8PFrame(blocks: &blocks, zeroThreshold: safeThreshold)
        } else {
            encodePlaneBaseSubbands8(blocks: &blocks, zeroThreshold: safeThreshold)
        }
        
        let quantizedBlocks = blocks
        let (reconPlane, rPlane) = reconstructPlaneBase8(blocks: blocks, width: cbDx, height: cbDy, qt: qtC, pool: pool)
        return (buf, reconPlane, rPlane, quantizedBlocks, relBlocks)
    }()

    let (bufY, reconY, r0Y, base8YBlocks, relYBlocks) = await taskBufY
    let (bufCb, reconCb, r0Cb, base8CbBlocks, relCbBlocks) = await taskBufCb
    let (bufCr, reconCr, r0Cr, base8CrBlocks, relCrBlocks) = await taskBufCr
    
    let reconstructed = PlaneData420(width: dx, height: dy, y: reconY, cb: reconCb, cr: reconCr)
    
    debugLog({
        return "  [Layer \(layer)/Base] Y=\(bufY.count) Cb=\(bufCb.count) Cr=\(bufCr.count) bytes"
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
