// MARK: - Encode Plane Arrays

import Foundation

final class ConcurrentBox<T>: @unchecked Sendable {
    var value: T
    init(_ value: T) { self.value = value }
}

struct PlaneData420 {
    let width: Int
    let height: Int
    var y: [Int16]
    var cb: [Int16]
    var cr: [Int16]
    
    var rY: Int16Reader {
        Int16Reader(data: y, width: width, height: height)
    }
    var rCb: Int16Reader {
        Int16Reader(data: cb, width: (width + 1) / 2, height: (height + 1) / 2)
    }
    var rCr: Int16Reader {
        Int16Reader(data: cr, width: (width + 1) / 2, height: (height + 1) / 2)
    }
}

extension PlaneData420 {
    init(img16: Image16) {
        self.width = img16.width
        self.height = img16.height
        var yFlat = [Int16]()
        yFlat.reserveCapacity((img16.width * img16.height))
        for row in img16.y { yFlat.append(contentsOf: row) }

        let cWidth = ((img16.width + 1) / 2)
        let cHeight = ((img16.height + 1) / 2)

        var cbFlat = [Int16]()
        cbFlat.reserveCapacity((cWidth * cHeight))
        for row in img16.cb { cbFlat.append(contentsOf: row) }

        var crFlat = [Int16]()
        crFlat.reserveCapacity((cWidth * cHeight))
        for row in img16.cr { crFlat.append(contentsOf: row) }

        self.y = yFlat
        self.cb = cbFlat
        self.cr = crFlat
    }
    
    func toYCbCr() -> YCbCrImage {
        var img = YCbCrImage(width: width, height: height)
        if width < 1 || height < 1 { return img }

        let countY = img.yPlane.count
        if countY > 0 {
            for i in 0..<countY {
                let v = self.y[i]
                img.yPlane[i] = v < -128 ? 0 : (127 < v ? 255 : UInt8(v + 128))
            }
        }

        let countCbCr = img.cbPlane.count
        if countCbCr > 0 {
            for i in 0..<countCbCr {
                let cbVal = self.cb[i]
                img.cbPlane[i] = cbVal < -128 ? 0 : (127 < cbVal ? 255 : UInt8(cbVal + 128))
                let crVal = self.cr[i]
                img.crPlane[i] = crVal < -128 ? 0 : (127 < crVal ? 255 : UInt8(crVal + 128))
            }
        }

        return img
    }
}

@inline(__always)
func evaluateQuantizeLayer(block: inout Block2D, size: Int, qt: QuantizationTable) {
    block.withView { view in
        let subs = getSubbands(view: view, size: size)
        var hl = subs.hl
        var lh = subs.lh
        var hh = subs.hh
        quantizeMidSignedMapping(&hl, qt: qt)
        quantizeMidSignedMapping(&lh, qt: qt)
        quantizeHighSignedMapping(&hh, qt: qt)
    }
}

@inline(__always)
func evaluateQuantizeBase(block: inout Block2D, size: Int, qt: QuantizationTable) {
    block.withView { view in
        let subs = getSubbands(view: view, size: size)
        var ll = subs.ll
        var hl = subs.hl
        var lh = subs.lh
        var hh = subs.hh
        quantizeLow(&ll, qt: qt)
        quantizeMidSignedMapping(&hl, qt: qt)
        quantizeMidSignedMapping(&lh, qt: qt)
        quantizeHighSignedMapping(&hh, qt: qt)
    }
}

func extractTransformBlocks(pd: PlaneData420, size: Int, qtY: QuantizationTable, qtC: QuantizationTable) async throws -> (blocksY: [Block2D], blocksCb: [Block2D], blocksCr: [Block2D], subPlane: PlaneData420) {
    let dx = pd.width
    let dy = pd.height
    
    var subY: [Int16] = [Int16](repeating: 0, count: ((dx / 2) * (dy / 2)))
    var subCb: [Int16] = [Int16](repeating: 0, count: (((dx + 1) / 2 / 2) * ((dy + 1) / 2 / 2)))
    var subCr: [Int16] = [Int16](repeating: 0, count: (((dx + 1) / 2 / 2) * ((dy + 1) / 2 / 2)))
    
    let rY = pd.rY
    let rowCountY = ((dy + size - 1) / size)
    let resultsY = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCountY))
    let errorY = ConcurrentBox<Error?>(nil)
    let chunkSize = 4
    let taskCountY = ((rowCountY + chunkSize - 1) / chunkSize)
    
    DispatchQueue.concurrentPerform(iterations: taskCountY) { taskIdx in
        let startRow = (taskIdx * chunkSize)
        let endRow = min((startRow + chunkSize), rowCountY)
        
        for i in startRow..<endRow {
            let h = (i * size)
            var rowResults: [(Block2D, Int, Int)] = []
            for w in stride(from: 0, to: dx, by: size) {
                var block = Block2D(width: size, height: size)
                block.withView { view in
                    for line in 0..<size {
                        let row = rY.row(x: w, y: (h + line), size: size)
                        view.setRow(offsetY: line, row: row)
                    }
                    _ = dwt2d(&view, size: size)
                }
                rowResults.append((block, w, h))
            }
            resultsY.value[i] = (h, rowResults)
        }
    }
    if let err = errorY.value { throw err }
    
    var blocksY: [Block2D] = []
    blocksY.reserveCapacity((rowCountY * ((dx + size - 1) / size)))
    for i in 0..<rowCountY {
        guard let res = resultsY.value[i] else { continue }
        for j in res.1.indices {
            var (llBlock, w, h) = res.1[j]
            blocksY.append(llBlock)

            
            let subDxWidth = (dx / 2)
            let subDyHeight = (dy / 2)
            let destStartX = (w / 2)
            let destStartY = (h / 2)
            let subSize = (size / 2)

            llBlock.withView { view in
                let subs = getSubbands(view: view, size: size)
                let srcBase = subs.ll.base
                for blockY in 0..<subSize {
                    let dstY = (destStartY + blockY)
                    if subDyHeight <= dstY { continue }
                    let srcPtr = srcBase.advanced(by: (blockY * size))
                    let limit = min(subSize, (subDxWidth - destStartX))

                    guard 0 < limit else { continue }

                    let dstIdx = ((dstY * subDxWidth) + destStartX)
                    subY.withUnsafeMutableBufferPointer { dstPtr in
                        guard let base = dstPtr.baseAddress else { return }
                        base.advanced(by: dstIdx).update(from: srcPtr, count: limit)
                    }
                }
            }
        }
    }
    
    let rCb = pd.rCb
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    let subCbDx = (cbDx / 2)
    let subCbDy = (cbDy / 2)
    let rowCountCb = ((cbDy + size - 1) / size)
    let resultsCb = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCountCb))
    let errorCb = ConcurrentBox<Error?>(nil)
    let taskCountCb = ((rowCountCb + chunkSize - 1) / chunkSize)
    
    DispatchQueue.concurrentPerform(iterations: taskCountCb) { taskIdx in
        let startRow = (taskIdx * chunkSize)
        let endRow = min((startRow + chunkSize), rowCountCb)
        
        for i in startRow..<endRow {
            let h = (i * size)
            var rowResults: [(Block2D, Int, Int)] = []
            for w in stride(from: 0, to: cbDx, by: size) {
                var block = Block2D(width: size, height: size)
                block.withView { view in
                    for line in 0..<size {
                        let row = rCb.row(x: w, y: (h + line), size: size)
                        view.setRow(offsetY: line, row: row)
                    }
                    _ = dwt2d(&view, size: size)
                }
                rowResults.append((block, w, h))
            }
            resultsCb.value[i] = (h, rowResults)
        }
    }
    if let err = errorCb.value { throw err }
    
    var blocksCb: [Block2D] = []
    blocksCb.reserveCapacity((rowCountCb * ((cbDx + size - 1) / size)))
    for i in 0..<rowCountCb {
        guard let res = resultsCb.value[i] else { continue }
        for j in res.1.indices {
            var (llBlock, w, h) = res.1[j]
            blocksCb.append(llBlock)

            let destStartX = (w / 2)
            let destStartY = (h / 2)
            let subSize = (size / 2)

            llBlock.withView { view in
                let subs = getSubbands(view: view, size: size)
                let srcBase = subs.ll.base
                for blockY in 0..<subSize {
                    let dstY = (destStartY + blockY)
                    if subCbDy <= dstY { continue }
                    let srcPtr = srcBase.advanced(by: (blockY * size))
                    let limit = min(subSize, (subCbDx - destStartX))

                    guard 0 < limit else { continue }

                    let dstIdx = ((dstY * subCbDx) + destStartX)
                    subCb.withUnsafeMutableBufferPointer { dstPtr in
                        guard let base = dstPtr.baseAddress else { return }
                        base.advanced(by: dstIdx).update(from: srcPtr, count: limit)
                    }
                }
            }
        }
    }
    
    let rCr = pd.rCr
    let rowCountCr = ((cbDy + size - 1) / size)
    let resultsCr = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCountCr))
    let errorCr = ConcurrentBox<Error?>(nil)
    let taskCountCr = ((rowCountCr + chunkSize - 1) / chunkSize)
    
    DispatchQueue.concurrentPerform(iterations: taskCountCr) { taskIdx in
        let startRow = (taskIdx * chunkSize)
        let endRow = min((startRow + chunkSize), rowCountCr)
        
        for i in startRow..<endRow {
            let h = (i * size)
            var rowResults: [(Block2D, Int, Int)] = []
            for w in stride(from: 0, to: cbDx, by: size) {
                var block = Block2D(width: size, height: size)
                block.withView { view in
                    for line in 0..<size {
                        let row = rCr.row(x: w, y: (h + line), size: size)
                        view.setRow(offsetY: line, row: row)
                    }
                    _ = dwt2d(&view, size: size)
                }
                rowResults.append((block, w, h))
            }
            resultsCr.value[i] = (h, rowResults)
        }
    }
    if let err = errorCr.value { throw err }
    
    var blocksCr: [Block2D] = []
    blocksCr.reserveCapacity((rowCountCr * ((cbDx + size - 1) / size)))
    for i in 0..<rowCountCr {
        guard let res = resultsCr.value[i] else { continue }
        for j in res.1.indices {
            var (llBlock, w, h) = res.1[j]
            blocksCr.append(llBlock)

            
            let destStartX = (w / 2)
            let destStartY = (h / 2)
            let subSize = (size / 2)

            llBlock.withView { view in
                let subs = getSubbands(view: view, size: size)
                let srcBase = subs.ll.base
                for blockY in 0..<subSize {
                    let dstY = (destStartY + blockY)
                    if subCbDy <= dstY { continue }
                    let srcPtr = srcBase.advanced(by: (blockY * size))
                    let limit = min(subSize, (subCbDx - destStartX))

                    guard 0 < limit else { continue }

                    let dstIdx = ((dstY * subCbDx) + destStartX)
                    subCr.withUnsafeMutableBufferPointer { dstPtr in
                        guard let base = dstPtr.baseAddress else { return }
                        base.advanced(by: dstIdx).update(from: srcPtr, count: limit)
                    }
                }
            }
        }
    }
    
    let subPlane = PlaneData420(width: (dx / 2), height: (dy / 2), y: subY, cb: subCb, cr: subCr)
    return (blocksY, blocksCb, blocksCr, subPlane)
}

func extractTransformBlocksBase(pd: PlaneData420, size: Int, qtY: QuantizationTable, qtC: QuantizationTable) async throws -> (blocksY: [Block2D], blocksCb: [Block2D], blocksCr: [Block2D]) {
    let dx = pd.width
    let dy = pd.height
    
    
    let rY = pd.rY
    let rowCountY = ((dy + size - 1) / size)
    let resultsY = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCountY))
    let errorY = ConcurrentBox<Error?>(nil)
    let chunkSize = 4
    let taskCountY = ((rowCountY + chunkSize - 1) / chunkSize)
    
    DispatchQueue.concurrentPerform(iterations: taskCountY) { taskIdx in
        let startRow = (taskIdx * chunkSize)
        let endRow = min((startRow + chunkSize), rowCountY)
        
        for i in startRow..<endRow {
            let h = (i * size)
            var rowResults: [(Block2D, Int, Int)] = []
            for w in stride(from: 0, to: dx, by: size) {
                var block = Block2D(width: size, height: size)
                block.withView { view in
                    for line in 0..<size {
                        let row = rY.row(x: w, y: (h + line), size: size)
                        view.setRow(offsetY: line, row: row)
                    }
                    _ = dwt2d(&view, size: size)
                }
                rowResults.append((block, w, h))
            }
            resultsY.value[i] = (h, rowResults)
        }
    }
    if let err = errorY.value { throw err }
    
    var blocksY: [Block2D] = []
    blocksY.reserveCapacity((rowCountY * ((dx + size - 1) / size)))
    for i in 0..<rowCountY {
        guard let res = resultsY.value[i] else { continue }
        for j in res.1.indices {
            let (llBlock, _, _) = res.1[j]
            blocksY.append(llBlock)

        }
    }
    
    let rCb = pd.rCb
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    let rowCountCb = ((cbDy + size - 1) / size)
    let resultsCb = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCountCb))
    let errorCb = ConcurrentBox<Error?>(nil)
    let taskCountCb = ((rowCountCb + chunkSize - 1) / chunkSize)
    
    DispatchQueue.concurrentPerform(iterations: taskCountCb) { taskIdx in
        let startRow = (taskIdx * chunkSize)
        let endRow = min((startRow + chunkSize), rowCountCb)
        
        for i in startRow..<endRow {
            let h = (i * size)
            var rowResults: [(Block2D, Int, Int)] = []
            for w in stride(from: 0, to: cbDx, by: size) {
                var block = Block2D(width: size, height: size)
                block.withView { view in
                    for line in 0..<size {
                        let row = rCb.row(x: w, y: (h + line), size: size)
                        view.setRow(offsetY: line, row: row)
                    }
                    _ = dwt2d(&view, size: size)
                }
                rowResults.append((block, w, h))
            }
            resultsCb.value[i] = (h, rowResults)
        }
    }
    if let err = errorCb.value { throw err }
    
    var blocksCb: [Block2D] = []
    blocksCb.reserveCapacity((rowCountCb * ((cbDx + size - 1) / size)))
    for i in 0..<rowCountCb {
        guard let res = resultsCb.value[i] else { continue }
        for j in res.1.indices {
            let (llBlock, _, _) = res.1[j]
            blocksCb.append(llBlock)

        }
    }
    
    let rCr = pd.rCr
    let rowCountCr = ((cbDy + size - 1) / size)
    let resultsCr = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCountCr))
    let errorCr = ConcurrentBox<Error?>(nil)
    let taskCountCr = ((rowCountCr + chunkSize - 1) / chunkSize)
    
    DispatchQueue.concurrentPerform(iterations: taskCountCr) { taskIdx in
        let startRow = (taskIdx * chunkSize)
        let endRow = min((startRow + chunkSize), rowCountCr)
        
        for i in startRow..<endRow {
            let h = (i * size)
            var rowResults: [(Block2D, Int, Int)] = []
            for w in stride(from: 0, to: cbDx, by: size) {
                var block = Block2D(width: size, height: size)
                block.withView { view in
                    for line in 0..<size {
                        let row = rCr.row(x: w, y: (h + line), size: size)
                        view.setRow(offsetY: line, row: row)
                    }
                    _ = dwt2d(&view, size: size)
                }
                rowResults.append((block, w, h))
            }
            resultsCr.value[i] = (h, rowResults)
        }
    }
    if let err = errorCr.value { throw err }
    
    var blocksCr: [Block2D] = []
    blocksCr.reserveCapacity((rowCountCr * ((cbDx + size - 1) / size)))
    for i in 0..<rowCountCr {
        guard let res = resultsCr.value[i] else { continue }
        for j in res.1.indices {
            let (llBlock, _, _) = res.1[j]
            blocksCr.append(llBlock)

        }
    }
    
    return (blocksY, blocksCb, blocksCr)
}

@inline(__always)
func subtractCoeffs(currBlocks: inout [Block2D], predBlocks: inout [Block2D], size: Int) {
    let half = (size / 2)
    for i in currBlocks.indices {
        currBlocks[i].withView { vC in
            predBlocks[i].withView { vP in
                for y in 0..<half {
                    let ptrC = vC.rowPointer(y: y).advanced(by: half)
                    let ptrP = vP.rowPointer(y: y).advanced(by: half)
                    for x in 0..<half { ptrC[x] &-= ptrP[x] }
                }
                for y in half..<size {
                    let ptrC = vC.rowPointer(y: y)
                    let ptrP = vP.rowPointer(y: y)
                    for x in 0..<half { ptrC[x] &-= ptrP[x] }
                }
                for y in half..<size {
                    let ptrC = vC.rowPointer(y: y).advanced(by: half)
                    let ptrP = vP.rowPointer(y: y).advanced(by: half)
                    for x in 0..<half { ptrC[x] &-= ptrP[x] }
                }
            }
        }
    }
}

@inline(__always)
func subtractCoeffsBase(currBlocks: inout [Block2D], predBlocks: inout [Block2D], size: Int) {
    let half = (size / 2)
    for i in currBlocks.indices {
        currBlocks[i].withView { vC in
            predBlocks[i].withView { vP in
                for y in 0..<half {
                    let ptrC = vC.rowPointer(y: y)
                    let ptrP = vP.rowPointer(y: y)
                    for x in 0..<half { ptrC[x] &-= ptrP[x] }
                }
                for y in 0..<half {
                    let ptrC = vC.rowPointer(y: y).advanced(by: half)
                    let ptrP = vP.rowPointer(y: y).advanced(by: half)
                    for x in 0..<half { ptrC[x] &-= ptrP[x] }
                }
                for y in half..<size {
                    let ptrC = vC.rowPointer(y: y)
                    let ptrP = vP.rowPointer(y: y)
                    for x in 0..<half { ptrC[x] &-= ptrP[x] }
                }
                for y in half..<size {
                    let ptrC = vC.rowPointer(y: y).advanced(by: half)
                    let ptrP = vP.rowPointer(y: y).advanced(by: half)
                    for x in 0..<half { ptrC[x] &-= ptrP[x] }
                }
            }
        }
    }
}

func encodePlaneLayer(pd: PlaneData420, predictedPd: PlaneData420?, layer: UInt8, size: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int) async throws -> ([UInt8], PlaneData420, PlaneData420?) {
    let dx = pd.width
    let dy = pd.height
    var (subBlocksY, subBlocksCb, subBlocksCr, subPlane) = try await extractTransformBlocks(pd: pd, size: size, qtY: qtY, qtC: qtC)
    
    var subPredPlane: PlaneData420? = nil
    if let pPd = predictedPd {
        var (pY, pCb, pCr, pSub) = try await extractTransformBlocks(pd: pPd, size: size, qtY: qtY, qtC: qtC)
        subtractCoeffs(currBlocks: &subBlocksY, predBlocks: &pY, size: size)
        subtractCoeffs(currBlocks: &subBlocksCb, predBlocks: &pCb, size: size)
        subtractCoeffs(currBlocks: &subBlocksCr, predBlocks: &pCr, size: size)
        subPredPlane = pSub
    }
    
    for i in subBlocksY.indices { evaluateQuantizeLayer(block: &subBlocksY[i], size: size, qt: qtY) }
    for i in subBlocksCb.indices { evaluateQuantizeLayer(block: &subBlocksCb[i], size: size, qt: qtC) }
    for i in subBlocksCr.indices { evaluateQuantizeLayer(block: &subBlocksCr[i], size: size, qt: qtC) }
    
    let bufY = encodePlaneSubbands(blocks: &subBlocksY, size: size, zeroThreshold: zeroThreshold)
    let bufCb = encodePlaneSubbands(blocks: &subBlocksCb, size: size, zeroThreshold: zeroThreshold)
    let bufCr = encodePlaneSubbands(blocks: &subBlocksCr, size: size, zeroThreshold: zeroThreshold)
    debugLog("  [Layer \(layer)] Y=\(bufY.count) Cb=\(bufCb.count) Cr=\(bufCr.count) bytes")
    
    var out: [UInt8] = []
    out.append(contentsOf: [0x56, 0x45, 0x56, 0x43, layer]) // 'VEVC' + layer
    appendUInt16BE(&out, UInt16(dx))
    appendUInt16BE(&out, UInt16(dy))
    out.append(UInt8(qtY.step))
    out.append(UInt8(qtC.step))
    
    appendUInt32BE(&out, UInt32(bufY.count))
    out.append(contentsOf: bufY)
    
    appendUInt32BE(&out, UInt32(bufCb.count))
    out.append(contentsOf: bufCb)
    
    appendUInt32BE(&out, UInt32(bufCr.count))
    out.append(contentsOf: bufCr)
    
    return (out, subPlane, subPredPlane)
}

func encodePlaneBase(pd: PlaneData420, predictedPd: PlaneData420?, layer: UInt8, size: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, isIFrame: Bool) async throws -> [UInt8] {
    let dx = pd.width
    let dy = pd.height
    var (subBlocksY, subBlocksCb, subBlocksCr) = try await extractTransformBlocksBase(pd: pd, size: size, qtY: qtY, qtC: qtC)
    if let pPd = predictedPd {
        var (pY, pCb, pCr) = try await extractTransformBlocksBase(pd: pPd, size: size, qtY: qtY, qtC: qtC)
        subtractCoeffsBase(currBlocks: &subBlocksY, predBlocks: &pY, size: size)
        subtractCoeffsBase(currBlocks: &subBlocksCb, predBlocks: &pCb, size: size)
        subtractCoeffsBase(currBlocks: &subBlocksCr, predBlocks: &pCr, size: size)
    }
    
    for i in subBlocksY.indices { evaluateQuantizeBase(block: &subBlocksY[i], size: size, qt: qtY) }
    for i in subBlocksCb.indices { evaluateQuantizeBase(block: &subBlocksCb[i], size: size, qt: qtC) }
    for i in subBlocksCr.indices { evaluateQuantizeBase(block: &subBlocksCr[i], size: size, qt: qtC) }
    
    let bufY = encodePlaneBaseSubbands(blocks: &subBlocksY, size: size, zeroThreshold: zeroThreshold)
    let bufCb = encodePlaneBaseSubbands(blocks: &subBlocksCb, size: size, zeroThreshold: zeroThreshold)
    let bufCr = encodePlaneBaseSubbands(blocks: &subBlocksCr, size: size, zeroThreshold: zeroThreshold)
    debugLog("  [Layer \(layer)/Base] Y=\(bufY.count) Cb=\(bufCb.count) Cr=\(bufCr.count) bytes")
    
    var out: [UInt8] = []
    out.append(contentsOf: [0x56, 0x45, 0x56, 0x43, layer]) // 'VEVC' + layer
    appendUInt16BE(&out, UInt16(dx))
    appendUInt16BE(&out, UInt16(dy))
    out.append(UInt8(qtY.step))
    out.append(UInt8(qtC.step))
    
    appendUInt32BE(&out, UInt32(bufY.count))
    out.append(contentsOf: bufY)
    
    appendUInt32BE(&out, UInt32(bufCb.count))
    out.append(contentsOf: bufCb)
    
    appendUInt32BE(&out, UInt32(bufCr.count))
    out.append(contentsOf: bufCr)
    
    return out
}

func encodeSpatialLayers(pd: PlaneData420, predictedPd: PlaneData420?, maxbitrate: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, isIFrame: Bool) async throws -> [UInt8] {
    let (layer2, sub2, subPred2) = try await encodePlaneLayer(pd: pd, predictedPd: predictedPd, layer: 2, size: 32, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
    let (layer1, sub1, subPred1) = try await encodePlaneLayer(pd: sub2, predictedPd: subPred2, layer: 1, size: 16, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
    let layer0 = try await encodePlaneBase(pd: sub1, predictedPd: subPred1, layer: 0, size: 8, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold, isIFrame: isIFrame)
    
    debugLog("  [Summary] Layer0=\(layer0.count) Layer1=\(layer1.count) Layer2=\(layer2.count) total=\(layer0.count + layer1.count + layer2.count) bytes")
    
    var out: [UInt8] = []
    
    appendUInt32BE(&out, UInt32(layer0.count))
    out.append(contentsOf: layer0)
    
    appendUInt32BE(&out, UInt32(layer1.count))
    out.append(contentsOf: layer1)
    
    appendUInt32BE(&out, UInt32(layer2.count))
    out.append(contentsOf: layer2)
    
    return out
}
