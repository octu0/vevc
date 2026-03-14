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
                switch v {
                case ..<(-128):
                    img.yPlane[i] = 0
                case 128...:
                    img.yPlane[i] = 255
                default:
                    img.yPlane[i] = UInt8(v + 128)
                }
            }
        }

        let countCbCr = img.cbPlane.count
        if 0 < countCbCr {
            for i in 0..<countCbCr {
                let cbVal = self.cb[i]
                switch cbVal {
                case ..<(-128):
                    img.cbPlane[i] = 0
                case 128...:
                    img.cbPlane[i] = 255
                default:
                    img.cbPlane[i] = UInt8(cbVal + 128)
                }

                let crVal = self.cr[i]
                switch crVal {
                case ..<(-128):
                    img.crPlane[i] = 0
                case 128...:
                    img.crPlane[i] = 255
                default:
                    img.crPlane[i] = UInt8(crVal + 128)
                }
            }
        }

        return img
    }
}

@inline(__always)
func evaluateQuantizeLayer32(block: inout Block2D, qt: QuantizationTable) {
    block.withView { view in
        let subs = getSubbands32(view: view)
        var hl = subs.hl
        var lh = subs.lh
        var hh = subs.hh
        quantizeMidSignedMapping(&hl, qt: qt)
        quantizeMidSignedMapping(&lh, qt: qt)
        quantizeHighSignedMapping(&hh, qt: qt)
    }
}

@inline(__always)
func evaluateQuantizeLayer16(block: inout Block2D, qt: QuantizationTable) {
    block.withView { view in
        let subs = getSubbands16(view: view)
        var hl = subs.hl
        var lh = subs.lh
        var hh = subs.hh
        quantizeMidSignedMapping(&hl, qt: qt)
        quantizeMidSignedMapping(&lh, qt: qt)
        quantizeHighSignedMapping(&hh, qt: qt)
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
        quantizeLow(&ll, qt: qt)
        quantizeMidSignedMapping(&hl, qt: qt)
        quantizeMidSignedMapping(&lh, qt: qt)
        quantizeHighSignedMapping(&hh, qt: qt)
    }
}

@inline(__always)
func extractSingleTransformBlocks32(r: Int16Reader, width: Int, height: Int) -> (blocks: [Block2D], subband: [Int16]) {
    let subWidth = (width / 2)
    let subHeight = (height / 2)
    var subband: [Int16] = [Int16](repeating: 0, count: subWidth * subHeight)
    let rowCount = ((height + 32 - 1) / 32)
    let results = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCount))
    let chunkSize = 4
    let taskCount = ((rowCount + chunkSize - 1) / chunkSize)
    
    DispatchQueue.concurrentPerform(iterations: taskCount) { taskIdx in
        let startRow = (taskIdx * chunkSize)
        let endRow = min((startRow + chunkSize), rowCount)
        
        for i in startRow..<endRow {
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
            results.value[i] = (h, rowResults)
        }
    }
    
    var blocks: [Block2D] = []
    blocks.reserveCapacity((rowCount * ((width + 32 - 1) / 32)))
    subband.withUnsafeMutableBufferPointer { dstBuf in
        guard let dstBase = dstBuf.baseAddress else { return }
        for i in 0..<rowCount {
            guard let res = results.value[i] else { continue }
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
func extractSingleTransformBlocks16(r: Int16Reader, width: Int, height: Int) -> (blocks: [Block2D], subband: [Int16]) {
    let subWidth = (width / 2)
    let subHeight = (height / 2)
    var subband: [Int16] = [Int16](repeating: 0, count: subWidth * subHeight)
    let rowCount = ((height + 16 - 1) / 16)
    let results = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCount))
    let chunkSize = 4
    let taskCount = ((rowCount + chunkSize - 1) / chunkSize)
    
    DispatchQueue.concurrentPerform(iterations: taskCount) { taskIdx in
        let startRow = (taskIdx * chunkSize)
        let endRow = min((startRow + chunkSize), rowCount)
        
        for i in startRow..<endRow {
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
            results.value[i] = (h, rowResults)
        }
    }
    
    var blocks: [Block2D] = []
    blocks.reserveCapacity((rowCount * ((width + 16 - 1) / 16)))
    subband.withUnsafeMutableBufferPointer { dstBuf in
        guard let dstBase = dstBuf.baseAddress else { return }
        for i in 0..<rowCount {
            guard let res = results.value[i] else { continue }
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
func extractSingleTransformBlocksBase8(r: Int16Reader, width: Int, height: Int) -> [Block2D] {
    let rowCount = ((height + 8 - 1) / 8)
    let results = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCount))
    let chunkSize = 4
    let taskCount = ((rowCount + chunkSize - 1) / chunkSize)
    
    DispatchQueue.concurrentPerform(iterations: taskCount) { taskIdx in
        let startRow = (taskIdx * chunkSize)
        let endRow = min((startRow + chunkSize), rowCount)
        
        for i in startRow..<endRow {
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
            results.value[i] = (h, rowResults)
        }
    }
    
    var blocks: [Block2D] = []
    blocks.reserveCapacity((rowCount * ((width + 8 - 1) / 8)))
    for i in 0..<rowCount {
        guard let res = results.value[i] else { continue }
        for j in res.1.indices {
            let (llBlock, _, _) = res.1[j]
            blocks.append(llBlock)
        }
    }
    
    return blocks
}

@inline(__always)
func subtractCoeffs32(currBlocks: inout [Block2D], predBlocks: inout [Block2D]) {
    let half = (32 / 2)
    for i in currBlocks.indices {
        currBlocks[i].withView { vC in
            predBlocks[i].withView { vP in
                for y in 0..<half {
                    let ptrC = vC.rowPointer(y: y).advanced(by: half)
                    let ptrP = vP.rowPointer(y: y).advanced(by: half)
                    for x in 0..<half { ptrC[x] &-= ptrP[x] }
                }
                for y in half..<32 {
                    let ptrC = vC.rowPointer(y: y)
                    let ptrP = vP.rowPointer(y: y)
                    for x in 0..<half { ptrC[x] &-= ptrP[x] }
                }
                for y in half..<32 {
                    let ptrC = vC.rowPointer(y: y).advanced(by: half)
                    let ptrP = vP.rowPointer(y: y).advanced(by: half)
                    for x in 0..<half { ptrC[x] &-= ptrP[x] }
                }
            }
        }
    }
}

@inline(__always)
func subtractCoeffs16(currBlocks: inout [Block2D], predBlocks: inout [Block2D]) {
    let half = (16 / 2)
    for i in currBlocks.indices {
        currBlocks[i].withView { vC in
            predBlocks[i].withView { vP in
                for y in 0..<half {
                    let ptrC = vC.rowPointer(y: y).advanced(by: half)
                    let ptrP = vP.rowPointer(y: y).advanced(by: half)
                    for x in 0..<half { ptrC[x] &-= ptrP[x] }
                }
                for y in half..<16 {
                    let ptrC = vC.rowPointer(y: y)
                    let ptrP = vP.rowPointer(y: y)
                    for x in 0..<half { ptrC[x] &-= ptrP[x] }
                }
                for y in half..<16 {
                    let ptrC = vC.rowPointer(y: y).advanced(by: half)
                    let ptrP = vP.rowPointer(y: y).advanced(by: half)
                    for x in 0..<half { ptrC[x] &-= ptrP[x] }
                }
            }
        }
    }
}

@inline(__always)
func subtractCoeffsBase8(currBlocks: inout [Block2D], predBlocks: inout [Block2D]) {
    let half = (8 / 2)
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
                for y in half..<8 {
                    let ptrC = vC.rowPointer(y: y)
                    let ptrP = vP.rowPointer(y: y)
                    for x in 0..<half { ptrC[x] &-= ptrP[x] }
                }
                for y in half..<8 {
                    let ptrC = vC.rowPointer(y: y).advanced(by: half)
                    let ptrP = vP.rowPointer(y: y).advanced(by: half)
                    for x in 0..<half { ptrC[x] &-= ptrP[x] }
                }
            }
        }
    }
}

@inline(__always)
func encodePlaneLayer32(pd: PlaneData420, predictedPd: PlaneData420?, layer: UInt8, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int) async throws -> ([UInt8], PlaneData420, PlaneData420?) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    
    async let taskBufY = {
        var (blocks, subband) = extractSingleTransformBlocks32(r: pd.rY, width: dx, height: dy)
        var predSubband: [Int16]? = nil
        if let pPd = predictedPd {
            var (pBlocks, pSubband) = extractSingleTransformBlocks32(r: pPd.rY, width: dx, height: dy)
            subtractCoeffs32(currBlocks: &blocks, predBlocks: &pBlocks)
            predSubband = pSubband
        }
        for i in blocks.indices { evaluateQuantizeLayer32(block: &blocks[i], qt: qtY) }
        let buf = encodePlaneSubbands32(blocks: &blocks, zeroThreshold: zeroThreshold)
        return (buf, subband, predSubband)
    }()
    
    async let taskBufCb = {
        var (blocks, subband) = extractSingleTransformBlocks32(r: pd.rCb, width: cbDx, height: cbDy)
        var predSubband: [Int16]? = nil
        if let pPd = predictedPd {
            var (pBlocks, pSubband) = extractSingleTransformBlocks32(r: pPd.rCb, width: cbDx, height: cbDy)
            subtractCoeffs32(currBlocks: &blocks, predBlocks: &pBlocks)
            predSubband = pSubband
        }
        for i in blocks.indices { evaluateQuantizeLayer32(block: &blocks[i], qt: qtC) }
        let buf = encodePlaneSubbands32(blocks: &blocks, zeroThreshold: zeroThreshold)
        return (buf, subband, predSubband)
    }()
    
    async let taskBufCr = {
        var (blocks, subband) = extractSingleTransformBlocks32(r: pd.rCr, width: cbDx, height: cbDy)
        var predSubband: [Int16]? = nil
        if let pPd = predictedPd {
            var (pBlocks, pSubband) = extractSingleTransformBlocks32(r: pPd.rCr, width: cbDx, height: cbDy)
            subtractCoeffs32(currBlocks: &blocks, predBlocks: &pBlocks)
            predSubband = pSubband
        }
        for i in blocks.indices { evaluateQuantizeLayer32(block: &blocks[i], qt: qtC) }
        let buf = encodePlaneSubbands32(blocks: &blocks, zeroThreshold: zeroThreshold)
        return (buf, subband, predSubband)
    }()

    let (bufY, subY, pSubY) = await taskBufY
    let (bufCb, subCb, pSubCb) = await taskBufCb
    let (bufCr, subCr, pSubCr) = await taskBufCr

    let subPlane = PlaneData420(width: dx / 2, height: dy / 2, y: subY, cb: subCb, cr: subCr)
    var subPredPlane: PlaneData420? = nil
    if let pY = pSubY, let pCb = pSubCb, let pCr = pSubCr {
        subPredPlane = PlaneData420(width: dx / 2, height: dy / 2, y: pY, cb: pCb, cr: pCr)
    }

    debugLog("  [Layer \(layer)] Y=\(bufY.count) Cb=\(bufCb.count) Cr=\(bufCr.count) bytes")
    
    var out: [UInt8] = []
    out.append(contentsOf: [0x56, 0x45, 0x56, 0x43, layer])
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

@inline(__always)
func encodePlaneLayer16(pd: PlaneData420, predictedPd: PlaneData420?, layer: UInt8, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int) async throws -> ([UInt8], PlaneData420, PlaneData420?) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    
    async let taskBufY = {
        var (blocks, subband) = extractSingleTransformBlocks16(r: pd.rY, width: dx, height: dy)
        var predSubband: [Int16]? = nil
        if let pPd = predictedPd {
            var (pBlocks, pSubband) = extractSingleTransformBlocks16(r: pPd.rY, width: dx, height: dy)
            subtractCoeffs16(currBlocks: &blocks, predBlocks: &pBlocks)
            predSubband = pSubband
        }
        for i in blocks.indices { evaluateQuantizeLayer16(block: &blocks[i], qt: qtY) }
        let buf = encodePlaneSubbands16(blocks: &blocks, zeroThreshold: zeroThreshold)
        return (buf, subband, predSubband)
    }()
    
    async let taskBufCb = {
        var (blocks, subband) = extractSingleTransformBlocks16(r: pd.rCb, width: cbDx, height: cbDy)
        var predSubband: [Int16]? = nil
        if let pPd = predictedPd {
            var (pBlocks, pSubband) = extractSingleTransformBlocks16(r: pPd.rCb, width: cbDx, height: cbDy)
            subtractCoeffs16(currBlocks: &blocks, predBlocks: &pBlocks)
            predSubband = pSubband
        }
        for i in blocks.indices { evaluateQuantizeLayer16(block: &blocks[i], qt: qtC) }
        let buf = encodePlaneSubbands16(blocks: &blocks, zeroThreshold: zeroThreshold)
        return (buf, subband, predSubband)
    }()
    
    async let taskBufCr = {
        var (blocks, subband) = extractSingleTransformBlocks16(r: pd.rCr, width: cbDx, height: cbDy)
        var predSubband: [Int16]? = nil
        if let pPd = predictedPd {
            var (pBlocks, pSubband) = extractSingleTransformBlocks16(r: pPd.rCr, width: cbDx, height: cbDy)
            subtractCoeffs16(currBlocks: &blocks, predBlocks: &pBlocks)
            predSubband = pSubband
        }
        for i in blocks.indices { evaluateQuantizeLayer16(block: &blocks[i], qt: qtC) }
        let buf = encodePlaneSubbands16(blocks: &blocks, zeroThreshold: zeroThreshold)
        return (buf, subband, predSubband)
    }()

    let (bufY, subY, pSubY) = await taskBufY
    let (bufCb, subCb, pSubCb) = await taskBufCb
    let (bufCr, subCr, pSubCr) = await taskBufCr

    let subPlane = PlaneData420(width: dx / 2, height: dy / 2, y: subY, cb: subCb, cr: subCr)
    var subPredPlane: PlaneData420? = nil
    if let pY = pSubY, let pCb = pSubCb, let pCr = pSubCr {
        subPredPlane = PlaneData420(width: dx / 2, height: dy / 2, y: pY, cb: pCb, cr: pCr)
    }

    debugLog("  [Layer \(layer)] Y=\(bufY.count) Cb=\(bufCb.count) Cr=\(bufCr.count) bytes")
    
    var out: [UInt8] = []
    out.append(contentsOf: [0x56, 0x45, 0x56, 0x43, layer])
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

@inline(__always)
func encodePlaneBase8(pd: PlaneData420, predictedPd: PlaneData420?, layer: UInt8, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int) async throws -> [UInt8] {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    
    async let taskBufY = {
        var blocks = extractSingleTransformBlocksBase8(r: pd.rY, width: dx, height: dy)
        if let pPd = predictedPd {
            var pBlocks = extractSingleTransformBlocksBase8(r: pPd.rY, width: dx, height: dy)
            subtractCoeffsBase8(currBlocks: &blocks, predBlocks: &pBlocks)
        }
        for i in blocks.indices { evaluateQuantizeBase8(block: &blocks[i], qt: qtY) }
        return encodePlaneBaseSubbands8(blocks: &blocks, zeroThreshold: zeroThreshold)
    }()
    
    async let taskBufCb = {
        var blocks = extractSingleTransformBlocksBase8(r: pd.rCb, width: cbDx, height: cbDy)
        if let pPd = predictedPd {
            var pBlocks = extractSingleTransformBlocksBase8(r: pPd.rCb, width: cbDx, height: cbDy)
            subtractCoeffsBase8(currBlocks: &blocks, predBlocks: &pBlocks)
        }
        for i in blocks.indices { evaluateQuantizeBase8(block: &blocks[i], qt: qtC) }
        return encodePlaneBaseSubbands8(blocks: &blocks, zeroThreshold: zeroThreshold)
    }()
    
    async let taskBufCr = {
        var blocks = extractSingleTransformBlocksBase8(r: pd.rCr, width: cbDx, height: cbDy)
        if let pPd = predictedPd {
            var pBlocks = extractSingleTransformBlocksBase8(r: pPd.rCr, width: cbDx, height: cbDy)
            subtractCoeffsBase8(currBlocks: &blocks, predBlocks: &pBlocks)
        }
        for i in blocks.indices { evaluateQuantizeBase8(block: &blocks[i], qt: qtC) }
        return encodePlaneBaseSubbands8(blocks: &blocks, zeroThreshold: zeroThreshold)
    }()

    let bufY = await taskBufY
    let bufCb = await taskBufCb
    let bufCr = await taskBufCr
    
    debugLog("  [Layer \(layer)/Base] Y=\(bufY.count) Cb=\(bufCb.count) Cr=\(bufCr.count) bytes")
    
    var out: [UInt8] = []
    out.append(contentsOf: [0x56, 0x45, 0x56, 0x43, layer])
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

@inline(__always)
func encodeSpatialLayers(pd: PlaneData420, predictedPd: PlaneData420?, maxbitrate: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int) async throws -> [UInt8] {
    let (layer2, sub2, subPred2) = try await encodePlaneLayer32(pd: pd, predictedPd: predictedPd, layer: 2, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
    let (layer1, sub1, subPred1) = try await encodePlaneLayer16(pd: sub2, predictedPd: subPred2, layer: 1, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
    let layer0 = try await encodePlaneBase8(pd: sub1, predictedPd: subPred1, layer: 0, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
    
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
