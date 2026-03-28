// MARK: - Encode Plane Arrays

import Foundation

final class ConcurrentBox<T>: @unchecked Sendable {
    var value: T
    init(_ value: T) { self.value = value }
}

@inline(__always)
func evaluateQuantizeLayer32(block: inout Block2D, qt: QuantizationTable) {
    block.withView { view in
        let subs = getSubbands32(view: view)
        var hl = subs.hl
        var lh = subs.lh
        var hh = subs.hh
        quantizeSIMDSignedMapping16(&hl, q: qt.qMid)
        quantizeSIMDSignedMapping16(&lh, q: qt.qMid)
        quantizeSIMDSignedMapping16(&hh, q: qt.qHigh)
    }
}

@inline(__always)
func evaluateQuantizeLayer16(block: inout Block2D, qt: QuantizationTable) {
    block.withView { view in
        let subs = getSubbands16(view: view)
        var hl = subs.hl
        var lh = subs.lh
        var hh = subs.hh
        quantizeSIMDSignedMapping8(&hl, q: qt.qMid)
        quantizeSIMDSignedMapping8(&lh, q: qt.qMid)
        quantizeSIMDSignedMapping8(&hh, q: qt.qHigh)
    }
}

@inline(__always)
func evaluateQuantizeBase8(block: inout Block2D, qt: QuantizationTable) {
    block.withView { view in
        let subs = getSubbands8(view: view)
        var ll = subs.ll
        var hl = subs.hl
        var lh = subs.lh
        var hh = subs.hh
        quantizeSIMD4(&ll, q: qt.qLow)
        quantizeSIMDSignedMapping4(&hl, q: qt.qMid)
        quantizeSIMDSignedMapping4(&lh, q: qt.qMid)
        quantizeSIMDSignedMapping4(&hh, q: qt.qHigh)
    }
}

@inline(__always)
func evaluateQuantizeBase32(block: inout Block2D, qt: QuantizationTable) {
    block.withView { view in
        let subs = getSubbands32(view: view)
        var ll = subs.ll
        var hl = subs.hl
        var lh = subs.lh
        var hh = subs.hh
        quantizeSIMD16(&ll, q: qt.qLow)
        quantizeSIMDSignedMapping16(&hl, q: qt.qMid)
        quantizeSIMDSignedMapping16(&lh, q: qt.qMid)
        quantizeSIMDSignedMapping16(&hh, q: qt.qHigh)
    }
}

@inline(__always)
func extractSingleTransformBlocks32(r: Int16Reader, width: Int, height: Int) async -> (blocks: [Block2D], subband: [Int16]) {
    let subWidth = ((width + 1) / 2)
    let subHeight = ((height + 1) / 2)
    var subband: [Int16] = [Int16](repeating: 0, count: subWidth * subHeight)
    let rowCount = ((height + 32 - 1) / 32)
    let chunkSize = 8
    
    // Concurrent ではなく TaskGroup を用いて安全に結果をマージする
    var resultsArray = [(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCount)
    await withTaskGroup(of: [(Int, [(Block2D, Int, Int)])].self) { group in
        var startRow = 0
        while startRow < rowCount {
            let endRow = min(startRow + chunkSize, rowCount)
            let sRow = startRow
            group.addTask {
                var chunkResults: [(Int, [(Block2D, Int, Int)])] = []
                for i in sRow..<endRow {
                    let h = (i * 32)
                    var rowResults: [(Block2D, Int, Int)] = []
                    for w in stride(from: 0, to: width, by: 32) {
                        var block = Block2D(width: 32, height: 32)
                        block.withView { view in
                            r.readBlock(x: w, y: h, width: 32, height: 32, into: &view)
                            dwt2d_32(&view)
                        }
                        rowResults.append((block, w, h))
                    }
                    chunkResults.append((i, rowResults))
                }
                return chunkResults
            }
            startRow += chunkSize
        }
        for await chunk in group {
            for (i, rowResults) in chunk {
                resultsArray[i] = (i * 32, rowResults)
            }
        }
    }    
    var blocks: [Block2D] = []
    blocks.reserveCapacity((rowCount * ((width + 32 - 1) / 32)))
    subband.withUnsafeMutableBufferPointer { dstBuf in
        guard let dstBase = dstBuf.baseAddress else { return }
        for i in 0..<rowCount {
            guard let res = resultsArray[i] else { continue }
            for j in res.1.indices {
                var (llBlock, w, h) = res.1[j]
                blocks.append(llBlock)
                
                let destStartX = (w / 2)
                let destStartY = (h / 2)
                let subSize = (32 / 2)

                llBlock.withView { view in
                    let subs = getSubbands32(view: view)
                    let srcBase = subs.ll.base
                    let limit = min(subSize, (subWidth - destStartX))

                    guard 0 < limit else { return }

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
    }
    
    return (blocks, subband)
}

@inline(__always)
func extractSingleTransformBlocks16(r: Int16Reader, width: Int, height: Int) async -> (blocks: [Block2D], subband: [Int16]) {
    let subWidth = ((width + 1) / 2)
    let subHeight = ((height + 1) / 2)
    var subband: [Int16] = [Int16](repeating: 0, count: subWidth * subHeight)
    let rowCount = ((height + 16 - 1) / 16)
    let chunkSize = 8
    
    var resultsArray = [(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCount)
    await withTaskGroup(of: [(Int, [(Block2D, Int, Int)])].self) { group in
        var startRow = 0
        while startRow < rowCount {
            let endRow = min(startRow + chunkSize, rowCount)
            let sRow = startRow
            group.addTask {
                var chunkResults: [(Int, [(Block2D, Int, Int)])] = []
                for i in sRow..<endRow {
                    let h = (i * 16)
                    var rowResults: [(Block2D, Int, Int)] = []
                    for w in stride(from: 0, to: width, by: 16) {
                        var block = Block2D(width: 16, height: 16)
                        block.withView { view in
                            r.readBlock(x: w, y: h, width: 16, height: 16, into: &view)
                            dwt2d_16(&view)
                        }
                        rowResults.append((block, w, h))
                    }
                    chunkResults.append((i, rowResults))
                }
                return chunkResults
            }
            startRow += chunkSize
        }
        for await chunk in group {
            for (i, rowResults) in chunk {
                resultsArray[i] = (i * 16, rowResults)
            }
        }
    }    
    var blocks: [Block2D] = []
    blocks.reserveCapacity((rowCount * ((width + 16 - 1)  / 2)))
    subband.withUnsafeMutableBufferPointer { dstBuf in
        guard let dstBase = dstBuf.baseAddress else { return }
        for i in 0..<rowCount {
            guard let res = resultsArray[i] else { continue }
            for j in res.1.indices {
                var (llBlock, w, h) = res.1[j]
                blocks.append(llBlock)
                
                let destStartX = (w / 2)
                let destStartY = (h / 2)
                let subSize = (16 / 2)

                llBlock.withView { view in
                    let subs = getSubbands16(view: view)
                    let srcBase = subs.ll.base
                    let limit = min(subSize, (subWidth - destStartX))

                    guard 0 < limit else { return }

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
    }
    
    return (blocks, subband)
}

@inline(__always)
func extractSingleTransformBlocksBase8(r: Int16Reader, width: Int, height: Int) async -> [Block2D] {
    let rowCount = ((height + 8 - 1) / 8)
    let chunkSize = 4
    
    var resultsArray = [(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCount)
    await withTaskGroup(of: [(Int, [(Block2D, Int, Int)])].self) { group in
        var startRow = 0
        while startRow < rowCount {
            let endRow = min(startRow + chunkSize, rowCount)
            let sRow = startRow
            group.addTask {
                var chunkResults: [(Int, [(Block2D, Int, Int)])] = []
                for i in sRow..<endRow {
                    let h = (i * 8)
                    var rowResults: [(Block2D, Int, Int)] = []
                    for w in stride(from: 0, to: width, by: 8) {
                        var block = Block2D(width: 8, height: 8)
                        block.withView { view in
                            r.readBlock(x: w, y: h, width: 8, height: 8, into: &view)
                            dwt2d_8(&view)
                        }
                        rowResults.append((block, w, h))
                    }
                    chunkResults.append((i, rowResults))
                }
                return chunkResults
            }
            startRow += chunkSize
        }
        for await chunk in group {
            for (i, rowResults) in chunk {
                resultsArray[i] = (i * 8, rowResults)
            }
        }
    }    
    var blocks: [Block2D] = []
    blocks.reserveCapacity((rowCount * ((width + 8 - 1) / 8)))
    for i in 0..<rowCount {
        guard let res = resultsArray[i] else { continue }
        for j in res.1.indices {
            let (llBlock, _, _) = res.1[j]
            blocks.append(llBlock)
        }
    }
    
    return blocks
}

@inline(__always)
func extractSingleTransformBlocksBase32(r: Int16Reader, width: Int, height: Int) async -> [Block2D] {
    let rowCount = ((height + 32 - 1) / 32)
    let chunkSize = 4
    
    var resultsArray = [(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCount)
    await withTaskGroup(of: [(Int, [(Block2D, Int, Int)])].self) { group in
        var startRow = 0
        while startRow < rowCount {
            let endRow = min(startRow + chunkSize, rowCount)
            let sRow = startRow
            group.addTask {
                var chunkResults: [(Int, [(Block2D, Int, Int)])] = []
                for i in sRow..<endRow {
                    let h = (i * 32)
                    var rowResults: [(Block2D, Int, Int)] = []
                    for w in stride(from: 0, to: width, by: 32) {
                        var block = Block2D(width: 32, height: 32)
                        block.withView { view in
                            r.readBlock(x: w, y: h, width: 32, height: 32, into: &view)
                            dwt2d_32(&view)
                        }
                        rowResults.append((block, w, h))
                    }
                    chunkResults.append((i, rowResults))
                }
                return chunkResults
            }
            startRow += chunkSize
        }
        for await chunk in group {
            for (i, rowResults) in chunk {
                resultsArray[i] = (i * 32, rowResults)
            }
        }
    }    
    var blocks: [Block2D] = []
    blocks.reserveCapacity((rowCount * ((width + 32 - 1) / 32)))
    for i in 0..<rowCount {
        guard let res = resultsArray[i] else { continue }
        for j in res.1.indices {
            let (llBlock, _, _) = res.1[j]
            blocks.append(llBlock)
        }
    }
    
    return blocks
}

@inline(__always)
func subtractCoeffs32(currBlocks: inout [Block2D], predBlocks: inout [Block2D]) {
    for i in currBlocks.indices {
        currBlocks[i].withView { vC in
            predBlocks[i].withView { vP in
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
    }
}

@inline(__always)
func subtractCoeffs16(currBlocks: inout [Block2D], predBlocks: inout [Block2D]) {
    for i in currBlocks.indices {
        currBlocks[i].withView { vC in
            predBlocks[i].withView { vP in
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
    }
}

@inline(__always)
func subtractCoeffsBase8(currBlocks: inout [Block2D], predBlocks: inout [Block2D]) {
    for i in currBlocks.indices {
        currBlocks[i].withView { vC in
            predBlocks[i].withView { vP in
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
    }
}

@inline(__always)
func subtractCoeffsBase32(currBlocks: inout [Block2D], predBlocks: inout [Block2D]) {
    for i in currBlocks.indices {
        currBlocks[i].withView { vC in
            predBlocks[i].withView { vP in
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
    }
}

@inline(__always)
func preparePlaneLayer32(pd: PlaneData420, sads: [Int]?, layer: UInt8, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int) async throws -> (PlaneData420, [Block2D], [Block2D], [Block2D]) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    
    async let taskBufY = { () -> ([Int16], [Block2D]) in
        var (blocks, subband) = await extractSingleTransformBlocks32(r: pd.rY, width: dx, height: dy)
        for i in blocks.indices { if let sList = sads, i < sList.count, sList[i] < 800 { blocks[i].clearAll() } }
        for i in blocks.indices {
            evaluateQuantizeLayer32(block: &blocks[i], qt: qtY)
        }
        return (subband, blocks)
    }()
    
    async let taskBufCb = { () -> ([Int16], [Block2D]) in
        var (blocks, subband) = await extractSingleTransformBlocks32(r: pd.rCb, width: cbDx, height: cbDy)
        for i in blocks.indices { if let sList = sads, i < sList.count, sList[i] < 400 { blocks[i].clearAll() } }
        for i in blocks.indices {
            evaluateQuantizeLayer32(block: &blocks[i], qt: qtC)
        }
        return (subband, blocks)
    }()
    
    async let taskBufCr = { () -> ([Int16], [Block2D]) in
        var (blocks, subband) = await extractSingleTransformBlocks32(r: pd.rCr, width: cbDx, height: cbDy)
        for i in blocks.indices { if let sList = sads, i < sList.count, sList[i] < 400 { blocks[i].clearAll() } }
        for i in blocks.indices {
            evaluateQuantizeLayer32(block: &blocks[i], qt: qtC)
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
func preparePlaneLayer16(pd: PlaneData420, sads: [Int]?, layer: UInt8, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int) async throws -> (PlaneData420, [Block2D], [Block2D], [Block2D]) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    
    async let taskBufY = { () -> ([Int16], [Block2D]) in
        var (blocks, subband) = await extractSingleTransformBlocks16(r: pd.rY, width: dx, height: dy)
        for i in blocks.indices { if let sList = sads, i < sList.count, sList[i] < 800 { blocks[i].clearAll() } }
        for i in blocks.indices {
            evaluateQuantizeLayer16(block: &blocks[i], qt: qtY)
        }
        return (subband, blocks)
    }()
    
    async let taskBufCb = { () -> ([Int16], [Block2D]) in
        var (blocks, subband) = await extractSingleTransformBlocks16(r: pd.rCb, width: cbDx, height: cbDy)
        for i in blocks.indices { if let sList = sads, i < sList.count, sList[i] < 400 { blocks[i].clearAll() } }
        for i in blocks.indices {
            evaluateQuantizeLayer16(block: &blocks[i], qt: qtC)
        }
        return (subband, blocks)
    }()
    
    async let taskBufCr = { () -> ([Int16], [Block2D]) in
        var (blocks, subband) = await extractSingleTransformBlocks16(r: pd.rCr, width: cbDx, height: cbDy)
        for i in blocks.indices { if let sList = sads, i < sList.count, sList[i] < 400 { blocks[i].clearAll() } }
        for i in blocks.indices {
            evaluateQuantizeLayer16(block: &blocks[i], qt: qtC)
        }
        return (subband, blocks)
    }()

    let (subY, yBlocks) = await taskBufY
    let (subCb, cbBlocks) = await taskBufCb
    let (subCr, crBlocks) = await taskBufCr

    let subPlane = PlaneData420(width: (dx + 1) / 2, height: (dy + 1) / 2, y: subY, cb: subCb, cr: subCr)
    return (subPlane, yBlocks, cbBlocks, crBlocks)
}

func entropyEncodeLayer32(dx: Int, dy: Int, layer: UInt8, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, yBlocks: inout [Block2D], cbBlocks: inout [Block2D], crBlocks: inout [Block2D], parentYBlocks: [Block2D]?, parentCbBlocks: [Block2D]?, parentCrBlocks: [Block2D]?) -> [UInt8] {
    let safeThresholdY = max(0, zeroThreshold - (Int(qtY.step) / 2))
    let safeThresholdC = max(0, zeroThreshold - (Int(qtC.step) / 2))
    
    let bufY = encodePlaneSubbands32(blocks: &yBlocks, zeroThreshold: safeThresholdY, parentBlocks: parentYBlocks)
    let bufCb = encodePlaneSubbands32(blocks: &cbBlocks, zeroThreshold: safeThresholdC, parentBlocks: parentCbBlocks)
    let bufCr = encodePlaneSubbands32(blocks: &crBlocks, zeroThreshold: safeThresholdC, parentBlocks: parentCrBlocks)
    
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
func entropyEncodeLayer16(dx: Int, dy: Int, layer: UInt8, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, yBlocks: inout [Block2D], cbBlocks: inout [Block2D], crBlocks: inout [Block2D], parentYBlocks: [Block2D]?, parentCbBlocks: [Block2D]?, parentCrBlocks: [Block2D]?) -> [UInt8] {
    let safeThresholdY = max(0, zeroThreshold - (Int(qtY.step) / 2))
    let safeThresholdC = max(0, zeroThreshold - (Int(qtC.step) / 2))
    
    let bufY = encodePlaneSubbands16(blocks: &yBlocks, zeroThreshold: safeThresholdY, parentBlocks: parentYBlocks)
    let bufCb = encodePlaneSubbands16(blocks: &cbBlocks, zeroThreshold: safeThresholdC, parentBlocks: parentCbBlocks)
    let bufCr = encodePlaneSubbands16(blocks: &crBlocks, zeroThreshold: safeThresholdC, parentBlocks: parentCrBlocks)
    
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
func reconstructPlaneBase8(blocks: [Block2D], width: Int, height: Int, qt: QuantizationTable) -> [Int16] {
    let colCount = (width + 7) / 8
    let rowCount = (height + 7) / 8
    var plane = [Int16](repeating: 0, count: width * height)
    
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
                
                var blk = blocks[idx]
                idx += 1
                
                blk.withView { view in
                    let base = view.base
                    var llView = BlockView(base: base, width: 4, height: 4, stride: 8)
                    var hlView = BlockView(base: base.advanced(by: 4), width: 4, height: 4, stride: 8)
                    var lhView = BlockView(base: base.advanced(by: 32), width: 4, height: 4, stride: 8)
                    var hhView = BlockView(base: base.advanced(by: 36), width: 4, height: 4, stride: 8)
                    dequantizeSIMD4(&llView, q: qt.qLow)
                    dequantizeSIMDSignedMapping4(&hlView, q: qt.qMid)
                    dequantizeSIMDSignedMapping4(&lhView, q: qt.qMid)
                    dequantizeSIMDSignedMapping4(&hhView, q: qt.qHigh)
                    invDwt2d_8(&view)
                }
                
                if !isEdgeY && !isEdgeX {
                    blk.withView { v in
                        for h in 0..<8 {
                            let srcPtr = v.rowPointer(y: h)
                            let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                            destPtr.update(from: srcPtr, count: 8)
                        }
                    }
                } else if loopH > 0 && loopW > 0 {
                    blk.withView { v in
                        for h in 0..<loopH {
                            let srcPtr = v.rowPointer(y: h)
                            let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                            destPtr.update(from: srcPtr, count: loopW)
                        }
                    }
                }
            }
        }
    }
    return plane
}

@inline(__always)
func reconstructPlaneLayer32Y(blocks: [Block2D], prevImg: Image16, width: Int, height: Int, qt: QuantizationTable) -> [Int16] {
    let colCount = (width + 31) / 32
    let rowCount = (height + 31) / 32
    var plane = [Int16](repeating: 0, count: width * height)
    
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
                
                var blk = blocks[idx]
                idx += 1
                
                let llX = startX / 2
                let llY = startY / 2
                var ll = prevImg.getY(x: llX, y: llY, size: 16)
                
                ll.withView { srcView in
                    blk.withView { destView in
                        for yi in 0..<16 {
                            let srcPtr = srcView.rowPointer(y: yi)
                            let destPtr = destView.rowPointer(y: yi)
                            destPtr.update(from: srcPtr, count: 16)
                        }
                    }
                }
                
                blk.withView { view in
                    let base = view.base
                    var hlView = BlockView(base: base.advanced(by: 16), width: 16, height: 16, stride: 32)
                    var lhView = BlockView(base: base.advanced(by: 16 * 32), width: 16, height: 16, stride: 32)
                    var hhView = BlockView(base: base.advanced(by: 16 * 32 + 16), width: 16, height: 16, stride: 32)
                    dequantizeSIMDSignedMapping16(&hlView, q: qt.qMid)
                    dequantizeSIMDSignedMapping16(&lhView, q: qt.qMid)
                    dequantizeSIMDSignedMapping16(&hhView, q: qt.qHigh)
                    invDwt2d_32(&view)
                }
                
                if !isEdgeY && !isEdgeX {
                    blk.withView { v in
                        for h in 0..<32 {
                            let srcPtr = v.rowPointer(y: h)
                            let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                            destPtr.update(from: srcPtr, count: 32)
                        }
                    }
                } else if loopH > 0 && loopW > 0 {
                    blk.withView { v in
                        for h in 0..<loopH {
                            let srcPtr = v.rowPointer(y: h)
                            let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                            destPtr.update(from: srcPtr, count: loopW)
                        }
                    }
                }
            }
        }
    }
    return plane
}

@inline(__always)
func reconstructPlaneLayer32Cb(blocks: [Block2D], prevImg: Image16, width: Int, height: Int, qt: QuantizationTable) -> [Int16] {
    let colCount = (width + 31) / 32
    let rowCount = (height + 31) / 32
    var plane = [Int16](repeating: 0, count: width * height)
    
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
                
                var blk = blocks[idx]
                idx += 1
                
                let llX = startX / 2
                let llY = startY / 2
                var ll = prevImg.getCb(x: llX, y: llY, size: 16)
                
                ll.withView { srcView in
                    blk.withView { destView in
                        for yi in 0..<16 {
                            let srcPtr = srcView.rowPointer(y: yi)
                            let destPtr = destView.rowPointer(y: yi)
                            destPtr.update(from: srcPtr, count: 16)
                        }
                    }
                }
                
                blk.withView { view in
                    let base = view.base
                    var hlView = BlockView(base: base.advanced(by: 16), width: 16, height: 16, stride: 32)
                    var lhView = BlockView(base: base.advanced(by: 16 * 32), width: 16, height: 16, stride: 32)
                    var hhView = BlockView(base: base.advanced(by: 16 * 32 + 16), width: 16, height: 16, stride: 32)
                    dequantizeSIMDSignedMapping16(&hlView, q: qt.qMid)
                    dequantizeSIMDSignedMapping16(&lhView, q: qt.qMid)
                    dequantizeSIMDSignedMapping16(&hhView, q: qt.qHigh)
                    invDwt2d_32(&view)
                }
                
                if !isEdgeY && !isEdgeX {
                    blk.withView { v in
                        for h in 0..<32 {
                            let srcPtr = v.rowPointer(y: h)
                            let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                            destPtr.update(from: srcPtr, count: 32)
                        }
                    }
                } else if loopH > 0 && loopW > 0 {
                    blk.withView { v in
                        for h in 0..<loopH {
                            let srcPtr = v.rowPointer(y: h)
                            let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                            destPtr.update(from: srcPtr, count: loopW)
                        }
                    }
                }
            }
        }
    }
    return plane
}

@inline(__always)
func reconstructPlaneLayer32Cr(blocks: [Block2D], prevImg: Image16, width: Int, height: Int, qt: QuantizationTable) -> [Int16] {
    let colCount = (width + 31) / 32
    let rowCount = (height + 31) / 32
    var plane = [Int16](repeating: 0, count: width * height)
    
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
                
                var blk = blocks[idx]
                idx += 1
                
                let llX = startX / 2
                let llY = startY / 2
                var ll = prevImg.getCr(x: llX, y: llY, size: 16)
                
                ll.withView { srcView in
                    blk.withView { destView in
                        for yi in 0..<16 {
                            let srcPtr = srcView.rowPointer(y: yi)
                            let destPtr = destView.rowPointer(y: yi)
                            destPtr.update(from: srcPtr, count: 16)
                        }
                    }
                }
                
                blk.withView { view in
                    let base = view.base
                    var hlView = BlockView(base: base.advanced(by: 16), width: 16, height: 16, stride: 32)
                    var lhView = BlockView(base: base.advanced(by: 16 * 32), width: 16, height: 16, stride: 32)
                    var hhView = BlockView(base: base.advanced(by: 16 * 32 + 16), width: 16, height: 16, stride: 32)
                    dequantizeSIMDSignedMapping16(&hlView, q: qt.qMid)
                    dequantizeSIMDSignedMapping16(&lhView, q: qt.qMid)
                    dequantizeSIMDSignedMapping16(&hhView, q: qt.qHigh)
                    invDwt2d_32(&view)
                }
                
                if !isEdgeY && !isEdgeX {
                    blk.withView { v in
                        for h in 0..<32 {
                            let srcPtr = v.rowPointer(y: h)
                            let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                            destPtr.update(from: srcPtr, count: 32)
                        }
                    }
                } else if loopH > 0 && loopW > 0 {
                    blk.withView { v in
                        for h in 0..<loopH {
                            let srcPtr = v.rowPointer(y: h)
                            let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                            destPtr.update(from: srcPtr, count: loopW)
                        }
                    }
                }
            }
        }
    }
    return plane
}

@inline(__always)
func reconstructPlaneLayer16Y(blocks: [Block2D], prevImg: Image16, width: Int, height: Int, qt: QuantizationTable) -> [Int16] {
    let colCount = (width + 15) / 16
    let rowCount = (height + 15) / 16
    var plane = [Int16](repeating: 0, count: width * height)
    
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
                
                var blk = blocks[idx]
                idx += 1
                
                let llX = startX / 2
                let llY = startY / 2
                var ll = prevImg.getY(x: llX, y: llY, size: 8)
                
                ll.withView { srcView in
                    blk.withView { destView in
                        for yi in 0..<8 {
                            let srcPtr = srcView.rowPointer(y: yi)
                            let destPtr = destView.rowPointer(y: yi)
                            destPtr.update(from: srcPtr, count: 8)
                        }
                    }
                }
                
                blk.withView { view in
                    let base = view.base
                    var hlView = BlockView(base: base.advanced(by: 8), width: 8, height: 8, stride: 16)
                    var lhView = BlockView(base: base.advanced(by: 8 * 16), width: 8, height: 8, stride: 16)
                    var hhView = BlockView(base: base.advanced(by: 8 * 16 + 8), width: 8, height: 8, stride: 16)
                    dequantizeSIMDSignedMapping8(&hlView, q: qt.qMid)
                    dequantizeSIMDSignedMapping8(&lhView, q: qt.qMid)
                    dequantizeSIMDSignedMapping8(&hhView, q: qt.qHigh)
                    invDwt2d_16(&view)
                }
                
                if !isEdgeY && !isEdgeX {
                    blk.withView { v in
                        for h in 0..<16 {
                            let srcPtr = v.rowPointer(y: h)
                            let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                            destPtr.update(from: srcPtr, count: 16)
                        }
                    }
                } else if loopH > 0 && loopW > 0 {
                    blk.withView { v in
                        for h in 0..<loopH {
                            let srcPtr = v.rowPointer(y: h)
                            let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                            destPtr.update(from: srcPtr, count: loopW)
                        }
                    }
                }
            }
        }
    }
    return plane
}

@inline(__always)
func reconstructPlaneLayer16Cb(blocks: [Block2D], prevImg: Image16, width: Int, height: Int, qt: QuantizationTable) -> [Int16] {
    let colCount = (width + 15) / 16
    let rowCount = (height + 15) / 16
    var plane = [Int16](repeating: 0, count: width * height)
    
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
                
                var blk = blocks[idx]
                idx += 1
                
                let llX = startX / 2
                let llY = startY / 2
                var ll = prevImg.getCb(x: llX, y: llY, size: 8)
                
                ll.withView { srcView in
                    blk.withView { destView in
                        for yi in 0..<8 {
                            let srcPtr = srcView.rowPointer(y: yi)
                            let destPtr = destView.rowPointer(y: yi)
                            destPtr.update(from: srcPtr, count: 8)
                        }
                    }
                }
                
                blk.withView { view in
                    let base = view.base
                    var hlView = BlockView(base: base.advanced(by: 8), width: 8, height: 8, stride: 16)
                    var lhView = BlockView(base: base.advanced(by: 8 * 16), width: 8, height: 8, stride: 16)
                    var hhView = BlockView(base: base.advanced(by: 8 * 16 + 8), width: 8, height: 8, stride: 16)
                    dequantizeSIMDSignedMapping8(&hlView, q: qt.qMid)
                    dequantizeSIMDSignedMapping8(&lhView, q: qt.qMid)
                    dequantizeSIMDSignedMapping8(&hhView, q: qt.qHigh)
                    invDwt2d_16(&view)
                }
                
                if !isEdgeY && !isEdgeX {
                    blk.withView { v in
                        for h in 0..<16 {
                            let srcPtr = v.rowPointer(y: h)
                            let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                            destPtr.update(from: srcPtr, count: 16)
                        }
                    }
                } else if loopH > 0 && loopW > 0 {
                    blk.withView { v in
                        for h in 0..<loopH {
                            let srcPtr = v.rowPointer(y: h)
                            let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                            destPtr.update(from: srcPtr, count: loopW)
                        }
                    }
                }
            }
        }
    }
    return plane
}

@inline(__always)
func reconstructPlaneLayer16Cr(blocks: [Block2D], prevImg: Image16, width: Int, height: Int, qt: QuantizationTable) -> [Int16] {
    let colCount = (width + 15) / 16
    let rowCount = (height + 15) / 16
    var plane = [Int16](repeating: 0, count: width * height)
    
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
                
                var blk = blocks[idx]
                idx += 1
                
                let llX = startX / 2
                let llY = startY / 2
                var ll = prevImg.getCr(x: llX, y: llY, size: 8)
                
                ll.withView { srcView in
                    blk.withView { destView in
                        for yi in 0..<8 {
                            let srcPtr = srcView.rowPointer(y: yi)
                            let destPtr = destView.rowPointer(y: yi)
                            destPtr.update(from: srcPtr, count: 8)
                        }
                    }
                }
                
                blk.withView { view in
                    let base = view.base
                    var hlView = BlockView(base: base.advanced(by: 8), width: 8, height: 8, stride: 16)
                    var lhView = BlockView(base: base.advanced(by: 8 * 16), width: 8, height: 8, stride: 16)
                    var hhView = BlockView(base: base.advanced(by: 8 * 16 + 8), width: 8, height: 8, stride: 16)
                    dequantizeSIMDSignedMapping8(&hlView, q: qt.qMid)
                    dequantizeSIMDSignedMapping8(&lhView, q: qt.qMid)
                    dequantizeSIMDSignedMapping8(&hhView, q: qt.qHigh)
                    invDwt2d_16(&view)
                }
                
                if !isEdgeY && !isEdgeX {
                    blk.withView { v in
                        for h in 0..<16 {
                            let srcPtr = v.rowPointer(y: h)
                            let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                            destPtr.update(from: srcPtr, count: 16)
                        }
                    }
                } else if loopH > 0 && loopW > 0 {
                    blk.withView { v in
                        for h in 0..<loopH {
                            let srcPtr = v.rowPointer(y: h)
                            let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                            destPtr.update(from: srcPtr, count: loopW)
                        }
                    }
                }
            }
        }
    }
    return plane
}

@inline(__always)
func encodePlaneBase8(pd: PlaneData420, sads: [Int]?, layer: UInt8, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int) async throws -> ([UInt8], PlaneData420, [Block2D], [Block2D], [Block2D]) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    
    async let taskBufY = { () -> ([UInt8], [Int16], [Block2D]) in
        var blocks = await extractSingleTransformBlocksBase8(r: pd.rY, width: dx, height: dy)
        for i in blocks.indices {
            if let sList = sads, i < sList.count, sList[i] < 800 { blocks[i].clearAll() }
            evaluateQuantizeBase8(block: &blocks[i], qt: qtY)
        }
        let safeThreshold = max(0, zeroThreshold - (Int(qtY.step) / 2))
        let buf = encodePlaneBaseSubbands8(blocks: &blocks, zeroThreshold: safeThreshold)
        let quantizedBlocks = blocks
        let reconPlane = reconstructPlaneBase8(blocks: blocks, width: dx, height: dy, qt: qtY)
        return (buf, reconPlane, quantizedBlocks)
    }()
    
    async let taskBufCb = { () -> ([UInt8], [Int16], [Block2D]) in
        var blocks = await extractSingleTransformBlocksBase8(r: pd.rCb, width: cbDx, height: cbDy)
        for i in blocks.indices {
            if let sList = sads, i < sList.count, sList[i] < 400 { blocks[i].clearAll() }
            evaluateQuantizeBase8(block: &blocks[i], qt: qtC)
        }
        let safeThreshold = max(0, zeroThreshold - (Int(qtC.step)  / 2))
        let buf = encodePlaneBaseSubbands8(blocks: &blocks, zeroThreshold: safeThreshold)
        let quantizedBlocks = blocks
        let reconPlane = reconstructPlaneBase8(blocks: blocks, width: cbDx, height: cbDy, qt: qtC)
        return (buf, reconPlane, quantizedBlocks)
    }()
    
    async let taskBufCr = { () -> ([UInt8], [Int16], [Block2D]) in
        var blocks = await extractSingleTransformBlocksBase8(r: pd.rCr, width: cbDx, height: cbDy)
        for i in blocks.indices {
            if let sList = sads, i < sList.count, sList[i] < 400 { blocks[i].clearAll() }
            evaluateQuantizeBase8(block: &blocks[i], qt: qtC)
        }
        let safeThreshold = max(0, zeroThreshold - (Int(qtC.step) / 2))
        let buf = encodePlaneBaseSubbands8(blocks: &blocks, zeroThreshold: safeThreshold)
        let quantizedBlocks = blocks
        let reconPlane = reconstructPlaneBase8(blocks: blocks, width: cbDx, height: cbDy, qt: qtC)
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
func reconstructPlaneBase32(blocks: [Block2D], width: Int, height: Int, qt: QuantizationTable) -> [Int16] {
    let colCount = (width + 32 - 1) / 32
    var plane = [Int16](repeating: 0, count: width * height)
    
    plane.withUnsafeMutableBufferPointer { dstBuf in
        guard let dstBase = dstBuf.baseAddress else { return }
        for idx in blocks.indices {
            var blk = blocks[idx]
            let row = idx / colCount
            let col = idx % colCount
            let startY = row * 32
            let startX = col * 32
            
            blk.withView { view in
                let half = 16
                let base = view.base
                var llView = BlockView(base: base, width: half, height: half, stride: 32)
                var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 32)
                var lhView = BlockView(base: base.advanced(by: half * 32), width: half, height: half, stride: 32)
                var hhView = BlockView(base: base.advanced(by: half * 32 + half), width: half, height: half, stride: 32)
                dequantizeSIMD(&llView, q: qt.qLow)
                dequantizeSIMDSignedMapping(&hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping(&lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping(&hhView, q: qt.qHigh)
                invDwt2d_32(&view)
            }
            
            let validEndY = min(height, startY + 32)
            let validEndX = min(width, startX + 32)
            let loopH = validEndY - startY
            let loopW = validEndX - startX
            
            if loopH > 0 && loopW > 0 {
                blk.withView { v in
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
func encodePlaneBase32(pd: PlaneData420, predictedPd: PlaneData420?, layer: UInt8, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int) async throws -> ([UInt8], PlaneData420) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    
    async let taskBufY = { () -> ([UInt8], [Int16]) in
        var blocks = await extractSingleTransformBlocksBase32(r: pd.rY, width: dx, height: dy)
        if let pPd = predictedPd {
            var pBlocks = await extractSingleTransformBlocksBase32(r: pPd.rY, width: dx, height: dy)
            subtractCoeffsBase32(currBlocks: &blocks, predBlocks: &pBlocks)
        }
        for i in blocks.indices {
            evaluateQuantizeBase32(block: &blocks[i], qt: qtY)
        }
        let safeThreshold = max(0, zeroThreshold - (Int(qtY.step) / 2))
        let buf = encodePlaneBaseSubbands32(blocks: &blocks, zeroThreshold: safeThreshold)
        let reconPlane = reconstructPlaneBase32(blocks: blocks, width: dx, height: dy, qt: qtY)
        return (buf, reconPlane)
    }()
    
    async let taskBufCb = { () -> ([UInt8], [Int16]) in
        var blocks = await extractSingleTransformBlocksBase32(r: pd.rCb, width: cbDx, height: cbDy)
        if let pPd = predictedPd {
            var pBlocks = await extractSingleTransformBlocksBase32(r: pPd.rCb, width: cbDx, height: cbDy)
            subtractCoeffsBase32(currBlocks: &blocks, predBlocks: &pBlocks)
        }
        for i in blocks.indices {
            evaluateQuantizeBase32(block: &blocks[i], qt: qtC)
        }
        let safeThreshold = max(0, zeroThreshold - (Int(qtC.step)  / 2))
        let buf = encodePlaneBaseSubbands32(blocks: &blocks, zeroThreshold: safeThreshold)
        let reconPlane = reconstructPlaneBase32(blocks: blocks, width: cbDx, height: cbDy, qt: qtC)
        return (buf, reconPlane)
    }()
    
    async let taskBufCr = { () -> ([UInt8], [Int16]) in
        var blocks = await extractSingleTransformBlocksBase32(r: pd.rCr, width: cbDx, height: cbDy)
        if let pPd = predictedPd {
            var pBlocks = await extractSingleTransformBlocksBase32(r: pPd.rCr, width: cbDx, height: cbDy)
            subtractCoeffsBase32(currBlocks: &blocks, predBlocks: &pBlocks)
        }
        for i in blocks.indices {
            evaluateQuantizeBase32(block: &blocks[i], qt: qtC)
        }
        let safeThreshold = max(0, zeroThreshold - (Int(qtC.step)  / 2))
        let buf = encodePlaneBaseSubbands32(blocks: &blocks, zeroThreshold: safeThreshold)
        let reconPlane = reconstructPlaneBase32(blocks: blocks, width: cbDx, height: cbDy, qt: qtC)
        return (buf, reconPlane)
    }()

    let (bufY, reconY) = await taskBufY
    let (bufCb, reconCb) = await taskBufCb
    let (bufCr, reconCr) = await taskBufCr
    
    var mutReconY = reconY
    var mutReconCb = reconCb
    var mutReconCr = reconCr
    
    // Apply deblocking filter
    applyDeblockingFilter(plane: &mutReconY, width: dx, height: dy, blockSize: 32, qStep: Int(qtY.step))
    applyDeblockingFilter(plane: &mutReconCb, width: cbDx, height: cbDy, blockSize: 16, qStep: Int(qtC.step))
    applyDeblockingFilter(plane: &mutReconCr, width: cbDx, height: cbDy, blockSize: 16, qStep: Int(qtC.step))
    
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
func encodeSpatialLayers(pd: PlaneData420, predictedPd: PlaneData420?, maxbitrate: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int) async throws -> ([UInt8], PlaneData420) {
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
    
    let (mvs, sads) = predictedPd != nil ? await computeMotionVectors(curr: pd, prev: predictedPd!) : (nil, nil)
    
    var mutPdY = pd.y
    var mutPdCb = pd.cb
    var mutPdCr = pd.cr
    if let pPd = predictedPd, let mvs = mvs {
        subtractMotionCompensationPixels(plane: &mutPdY, prevPlane: pPd.y, mvs: mvs, width: dx, height: dy, blockSize: 32, shiftMultiplierX2: 8)
        subtractMotionCompensationPixels(plane: &mutPdCb, prevPlane: pPd.cb, mvs: mvs, width: cbDx, height: cbDy, blockSize: 16, shiftMultiplierX2: 4)
        subtractMotionCompensationPixels(plane: &mutPdCr, prevPlane: pPd.cr, mvs: mvs, width: cbDx, height: cbDy, blockSize: 16, shiftMultiplierX2: 4)
    }
    let resPd = PlaneData420(width: dx, height: dy, y: mutPdY, cb: mutPdCb, cr: mutPdCr)

    let (sub2, l2yBlocks, l2cbBlocks, l2crBlocks) = try await preparePlaneLayer32(pd: resPd, sads: sads, layer: 2, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold)
    let (sub1, l1yBlocks, l1cbBlocks, l1crBlocks) = try await preparePlaneLayer16(pd: sub2, sads: sads, layer: 1, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold)
    let (layer0, baseRecon, base8YBlocks, base8CbBlocks, base8CrBlocks) = try await encodePlaneBase8(pd: sub1, sads: sads, layer: 0, qtY: qtY0, qtC: qtC0, zeroThreshold: zeroThreshold)
    
    // Layer chain reconstruction: Base8 → Layer16 → Layer32
    // Build Image16 from base reconstruction for boundaryRepeat support
    let baseImg = Image16(width: baseRecon.width, height: baseRecon.height, y: baseRecon.y, cb: baseRecon.cb, cr: baseRecon.cr)
    
    // Layer16: LL = base reconstruction (via Image16.getY/Cb/Cr with boundaryRepeat)
    let l1dx = sub2.width
    let l1dy = sub2.height
    let l1cbDx = ((l1dx + 1) / 2)
    let l1cbDy = ((l1dy + 1) / 2)
    let reconL1Y = reconstructPlaneLayer16Y(blocks: l1yBlocks, prevImg: baseImg, width: l1dx, height: l1dy, qt: qtY1)
    let reconL1Cb = reconstructPlaneLayer16Cb(blocks: l1cbBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1)
    let reconL1Cr = reconstructPlaneLayer16Cr(blocks: l1crBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1)
    
    // Build Image16 from Layer16 reconstruction
    let l1Img = Image16(width: l1dx, height: l1dy, y: reconL1Y, cb: reconL1Cb, cr: reconL1Cr)
    
    var l1yBlocksMut = l1yBlocks
    var l1cbBlocksMut = l1cbBlocks
    var l1crBlocksMut = l1crBlocks
    let layer1 = entropyEncodeLayer16(dx: sub2.width, dy: sub2.height, layer: 1, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold, yBlocks: &l1yBlocksMut, cbBlocks: &l1cbBlocksMut, crBlocks: &l1crBlocksMut, parentYBlocks: base8YBlocks, parentCbBlocks: base8CbBlocks, parentCrBlocks: base8CrBlocks)
    
    var l2yBlocksMut = l2yBlocks
    var l2cbBlocksMut = l2cbBlocks
    var l2crBlocksMut = l2crBlocks
    let layer2 = entropyEncodeLayer32(dx: pd.width, dy: pd.height, layer: 2, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold, yBlocks: &l2yBlocksMut, cbBlocks: &l2cbBlocksMut, crBlocks: &l2crBlocksMut, parentYBlocks: l1yBlocksMut, parentCbBlocks: l1cbBlocksMut, parentCrBlocks: l1crBlocksMut)
    
    // Layer32: LL = layer16 reconstruction (via Image16.getY/Cb/Cr with boundaryRepeat)
    let reconL2Y = reconstructPlaneLayer32Y(blocks: l2yBlocks, prevImg: l1Img, width: dx, height: dy, qt: qtY2)
    let reconL2Cb = reconstructPlaneLayer32Cb(blocks: l2cbBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2)
    let reconL2Cr = reconstructPlaneLayer32Cr(blocks: l2crBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2)
    
    var mutReconL2Y = reconL2Y
    var mutReconL2Cb = reconL2Cb
    var mutReconL2Cr = reconL2Cr
    
    if let tPrev = predictedPd, let mvs = mvs {
        applyMotionCompensationPixels(plane: &mutReconL2Y, prevPlane: tPrev.y, mvs: mvs, width: dx, height: dy, blockSize: 32, shiftMultiplierX2: 8)
        applyMotionCompensationPixels(plane: &mutReconL2Cb, prevPlane: tPrev.cb, mvs: mvs, width: cbDx, height: cbDy, blockSize: 16, shiftMultiplierX2: 4)
        applyMotionCompensationPixels(plane: &mutReconL2Cr, prevPlane: tPrev.cr, mvs: mvs, width: cbDx, height: cbDy, blockSize: 16, shiftMultiplierX2: 4)
    }
    
    // Apply deblocking filter (blockSize corresponds to Layer32 output)
    applyDeblockingFilter(plane: &mutReconL2Y, width: dx, height: dy, blockSize: 32, qStep: Int(qtY2.step))
    applyDeblockingFilter(plane: &mutReconL2Cb, width: cbDx, height: cbDy, blockSize: 16, qStep: Int(qtC2.step))
    applyDeblockingFilter(plane: &mutReconL2Cr, width: cbDx, height: cbDy, blockSize: 16, qStep: Int(qtC2.step))
    
    let reconstructed = PlaneData420(width: dx, height: dy, y: mutReconL2Y, cb: mutReconL2Cb, cr: mutReconL2Cr)
    
    debugLog("  [Summary] Layer0=\(layer0.count) Layer1=\(layer1.count) Layer2=\(layer2.count) total=\(layer0.count + layer1.count + layer2.count) bytes")
    
    var out: [UInt8] = []
    
    let mvCount = mvs?.count ?? 0
    appendUInt32BE(&out, UInt32(mvCount))
    
    if mvCount > 0, let mvs = mvs {
        let mvData = encodeMVs(mvs: mvs)
        appendUInt32BE(&out, UInt32(mvData.count))
        out.append(contentsOf: mvData)
    } else {
        appendUInt32BE(&out, 0) // mvDataLen = 0
    }

    appendUInt32BE(&out, UInt32(layer0.count))
    out.append(contentsOf: layer0)
    
    appendUInt32BE(&out, UInt32(layer1.count))
    out.append(contentsOf: layer1)
    
    appendUInt32BE(&out, UInt32(layer2.count))
    out.append(contentsOf: layer2)
    
    return (out, reconstructed)
}


extension Block2D {
    mutating func clearAll() {
        self.data = [Int16](repeating: 0, count: self.data.count)
    }
}

@inline(__always)
func computeMotionVectors(curr: PlaneData420, prev: PlaneData420) async -> ([MotionVector], [Int]) {
    let dx = curr.width
    let dy = curr.height
    let l1dx = (dx + 1) / 2
    let l1dy = (dy + 1) / 2
    let l0dx = (l1dx + 1) / 2
    let l0dy = (l1dy + 1) / 2
    
    let (_, currSub2) = await extractSingleTransformBlocks32(r: curr.rY, width: dx, height: dy)
    let (_, currSub1) = await extractSingleTransformBlocks16(r: Int16Reader(data: currSub2, width: l1dx, height: l1dy), width: l1dx, height: l1dy)
    var currBlocks8 = await extractSingleTransformBlocksBase8(r: Int16Reader(data: currSub1, width: l0dx, height: l0dy), width: l0dx, height: l0dy)

    let (_, prevSub2) = await extractSingleTransformBlocks32(r: prev.rY, width: dx, height: dy)
    let (_, prevSub1) = await extractSingleTransformBlocks16(r: Int16Reader(data: prevSub2, width: l1dx, height: l1dy), width: l1dx, height: l1dy)
    
    let targetWidth = l0dx
    let targetHeight = l0dy
    let colCount = (targetWidth + 7) / 8
    
    var mvs = [MotionVector]()
    var sads = [Int]()
    mvs.reserveCapacity(currBlocks8.count)
    sads.reserveCapacity(currBlocks8.count)
    for idx in currBlocks8.indices {
        let col = idx % colCount
        let row = idx / colCount
        let bx = col * 8
        let by = row * 8
        let (mv, _, sad) = MotionEstimation.search(currBlock: &currBlocks8[idx], prevPlane: prevSub1, width: targetWidth, height: targetHeight, bx: bx, by: by, range: 2)
        mvs.append(mv)
        sads.append(sad)
    }
    return (mvs, sads)
}

@inline(__always)
func subtractMotionCompensationPixels(plane: inout [Int16], prevPlane: [Int16], mvs: [MotionVector], width: Int, height: Int, blockSize: Int, shiftMultiplierX2: Int) {
    let colCount = (width + blockSize - 1) / blockSize
    plane.withUnsafeMutableBufferPointer { dstBuf in
        guard let dstBase = dstBuf.baseAddress else { return }
        prevPlane.withUnsafeBufferPointer { srcBuf in
            guard let srcBase = srcBuf.baseAddress else { return }
            for row in 0..<((height + blockSize - 1) / blockSize) {
                for col in 0..<colCount {
                    let mvIndex = min(row * colCount + col, mvs.count - 1)
                    let mv = mvs[mvIndex]
                    let blockX = col * blockSize
                    let blockY = row * blockSize
                    let shiftX = (Int(mv.dx) * shiftMultiplierX2) / 2
                    let shiftY = (Int(mv.dy) * shiftMultiplierX2) / 2
                    let targetX = blockX + shiftX
                    let targetY = blockY + shiftY
                    for y in 0..<min(blockSize, height - blockY) {
                        let dstY = blockY + y
                        let srcY = targetY + y
                        let dstPtr = dstBase.advanced(by: dstY * width + blockX)
                        let safeSrcY = max(0, min(srcY, height - 1))
                        let srcRowPtr = srcBase.advanced(by: safeSrcY * width)
                        for x in 0..<min(blockSize, width - blockX) {
                            let srcX = targetX + x
                            let safeSrcX = max(0, min(srcX, width - 1))
                            let predPixel = srcRowPtr[safeSrcX]
                            dstPtr[x] = dstPtr[x] &- predPixel
                        }
                    }
                }
            }
        }
    }
}

@inline(__always)
func applyMotionCompensationPixels(plane: inout [Int16], prevPlane: [Int16], mvs: [MotionVector], width: Int, height: Int, blockSize: Int, shiftMultiplierX2: Int) {
    let colCount = (width + blockSize - 1) / blockSize
    plane.withUnsafeMutableBufferPointer { dstBuf in
        guard let dstBase = dstBuf.baseAddress else { return }
        prevPlane.withUnsafeBufferPointer { srcBuf in
            guard let srcBase = srcBuf.baseAddress else { return }
            for row in 0..<((height + blockSize - 1) / blockSize) {
                for col in 0..<colCount {
                    let mvIndex = min(row * colCount + col, mvs.count - 1)
                    let mv = mvs[mvIndex]
                    let blockX = col * blockSize
                    let blockY = row * blockSize
                    let shiftX = (Int(mv.dx) * shiftMultiplierX2) / 2
                    let shiftY = (Int(mv.dy) * shiftMultiplierX2) / 2
                    let targetX = blockX + shiftX
                    let targetY = blockY + shiftY
                    for y in 0..<min(blockSize, height - blockY) {
                        let dstY = blockY + y
                        let srcY = targetY + y
                        let dstPtr = dstBase.advanced(by: dstY * width + blockX)
                        let safeSrcY = max(0, min(srcY, height - 1))
                        let srcRowPtr = srcBase.advanced(by: safeSrcY * width)
                        for x in 0..<min(blockSize, width - blockX) {
                            let srcX = targetX + x
                            let safeSrcX = max(0, min(srcX, width - 1))
                            let predPixel = srcRowPtr[safeSrcX]
                            dstPtr[x] = dstPtr[x] &+ predPixel
                        }
                    }
                }
            }
        }
    }
}

