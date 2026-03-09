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
        Int16Reader(data: cb, width: ((width + 1) / 2), height: ((height + 1) / 2))
    }
    var rCr: Int16Reader {
        Int16Reader(data: cr, width: ((width + 1) / 2), height: ((height + 1) / 2))
    }
}

extension PlaneData420 {
    init(img16: Image16) {
        self.width = img16.width
        self.height = img16.height
        var yFlat = [Int16]()
        for row in img16.y { yFlat.append(contentsOf: row) }
        var cbFlat = [Int16]()
        for row in img16.cb { cbFlat.append(contentsOf: row) }
        var crFlat = [Int16]()
        for row in img16.cr { crFlat.append(contentsOf: row) }
        self.y = yFlat
        self.cb = cbFlat
        self.cr = crFlat
    }
    
    func toYCbCr() -> YCbCrImage {
        var img = YCbCrImage(width: width, height: height)
        let countY = img.yPlane.count
        for i in 0..<countY {
            let v = self.y[i]
            img.yPlane[i] = v < -128 ? 0 : (v > 127 ? 255 : UInt8(v + 128))
        }
        let countCbCr = img.cbPlane.count
        for i in 0..<countCbCr {
            let cbVal = self.cb[i]
            img.cbPlane[i] = cbVal < -128 ? 0 : (cbVal > 127 ? 255 : UInt8(cbVal + 128))
            let crVal = self.cr[i]
            img.crPlane[i] = crVal < -128 ? 0 : (crVal > 127 ? 255 : UInt8(crVal + 128))
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

func extractTransformBlocks(pd: PlaneData420, size: Int, qtY: QuantizationTable, qtC: QuantizationTable, isBase: Bool) async throws -> (blocksY: [Block2D], blocksCb: [Block2D], blocksCr: [Block2D], subPlane: PlaneData420?) {
    let dx = pd.width
    let dy = pd.height
    
    let rY = pd.rY
    let rowCountY = ((dy + size - 1) / size)
    let resultsY: [[Block2D]] = try await withThrowingTaskGroup(of: (Int, [Block2D]).self) { group in
        for i in 0..<rowCountY {
            group.addTask {
                let h = (i * size)
                var rowBlocks: [Block2D] = []
                for w in stride(from: 0, to: dx, by: size) {
                    var block = Block2D(width: size, height: size)
                    block.withView { view in
                        for line in 0..<size {
                            let row = rY.row(x: w, y: (h + line), size: size)
                            view.setRow(offsetY: line, row: row)
                        }
                        _ = dwt2d(&view, size: size)
                    }
                    rowBlocks.append(block)
                }
                return (i, rowBlocks)
            }
        }
        var res = [[Block2D]](repeating: [], count: rowCountY)
        for try await (idx, blocks) in group {
            res[idx] = blocks
        }
        return res
    }
    var blocksY = resultsY.flatMap { $0 }
    
    let rCb = pd.rCb
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    let rowCountCb = ((cbDy + size - 1) / size)
    let resultsCb: [[Block2D]] = try await withThrowingTaskGroup(of: (Int, [Block2D]).self) { group in
        for i in 0..<rowCountCb {
            group.addTask {
                let h = (i * size)
                var rowBlocks: [Block2D] = []
                for w in stride(from: 0, to: cbDx, by: size) {
                    var block = Block2D(width: size, height: size)
                    block.withView { view in
                        for line in 0..<size {
                            let row = rCb.row(x: w, y: (h + line), size: size)
                            view.setRow(offsetY: line, row: row)
                        }
                        _ = dwt2d(&view, size: size)
                    }
                    rowBlocks.append(block)
                }
                return (i, rowBlocks)
            }
        }
        var res = [[Block2D]](repeating: [], count: rowCountCb)
        for try await (idx, blocks) in group {
            res[idx] = blocks
        }
        return res
    }
    var blocksCb = resultsCb.flatMap { $0 }
    
    let rCr = pd.rCr
    let resultsCr: [[Block2D]] = try await withThrowingTaskGroup(of: (Int, [Block2D]).self) { group in
        for i in 0..<rowCountCb {
            group.addTask {
                let h = (i * size)
                var rowBlocks: [Block2D] = []
                for w in stride(from: 0, to: cbDx, by: size) {
                    var block = Block2D(width: size, height: size)
                    block.withView { view in
                        for line in 0..<size {
                            let row = rCr.row(x: w, y: (h + line), size: size)
                            view.setRow(offsetY: line, row: row)
                        }
                        _ = dwt2d(&view, size: size)
                    }
                    rowBlocks.append(block)
                }
                return (i, rowBlocks)
            }
        }
        var res = [[Block2D]](repeating: [], count: rowCountCb)
        for try await (idx, blocks) in group {
            res[idx] = blocks
        }
        return res
    }
    var blocksCr = resultsCr.flatMap { $0 }
    
    var subPlane: PlaneData420? = nil
    if !isBase {
        var sY = [Int16](repeating: 0, count: ((dx / 2) * (dy / 2)))
        var sCb = [Int16](repeating: 0, count: ((cbDx / 2) * (cbDy / 2)))
        var sCr = [Int16](repeating: 0, count: ((cbDx / 2) * (cbDy / 2)))

        let colCountY = ((dx + size - 1) / size)
        let subSize = (size / 2)
        let subDxY = (dx / 2)
        let subDyY = (dy / 2)
        for i in blocksY.indices {
            let r = (i / colCountY)
            let c = (i % colCountY)
            let destStartX = (c * subSize)
            let destStartY = (r * subSize)
            blocksY[i].withView { view in
                let subs = getSubbands(view: view, size: size)
                let srcBase = subs.ll.base
                for blockY in 0..<subSize {
                    let dstY = (destStartY + blockY)
                    if (dstY < subDyY) {
                        let srcPtr = srcBase.advanced(by: (blockY * size))
                        let limit = min(subSize, (subDxY - destStartX))
                        if (0 < limit) {
                            let dstIdx = (dstY * subDxY + destStartX)
                            sY.withUnsafeMutableBufferPointer { dstPtr in
                                dstPtr.baseAddress!.advanced(by: dstIdx).update(from: srcPtr, count: limit)
                            }
                        }
                    }
                }
            }
        }
        
        let colCountC = ((cbDx + size - 1) / size)
        let subDxC = (cbDx / 2)
        let subDyC = (cbDy / 2)
        for i in blocksCb.indices {
            let r = (i / colCountC)
            let c = (i % colCountC)
            let destStartX = (c * subSize)
            let destStartY = (r * subSize)
            blocksCb[i].withView { view in
                let subs = getSubbands(view: view, size: size)
                let srcBase = subs.ll.base
                for blockY in 0..<subSize {
                    let dstY = (destStartY + blockY)
                    if (dstY < subDyC) {
                        let srcPtr = srcBase.advanced(by: (blockY * size))
                        let limit = min(subSize, (subDxC - destStartX))
                        if (0 < limit) {
                            let dstIdx = (dstY * subDxC + destStartX)
                            sCb.withUnsafeMutableBufferPointer { dstPtr in
                                dstPtr.baseAddress!.advanced(by: dstIdx).update(from: srcPtr, count: limit)
                            }
                        }
                    }
                }
            }
        }
        for i in blocksCr.indices {
            let r = (i / colCountC)
            let c = (i % colCountC)
            let destStartX = (c * subSize)
            let destStartY = (r * subSize)
            blocksCr[i].withView { view in
                let subs = getSubbands(view: view, size: size)
                let srcBase = subs.ll.base
                for blockY in 0..<subSize {
                    let dstY = (destStartY + blockY)
                    if (dstY < subDyC) {
                        let srcPtr = srcBase.advanced(by: (blockY * size))
                        let limit = min(subSize, (subDxC - destStartX))
                        if (0 < limit) {
                            let dstIdx = (dstY * subDxC + destStartX)
                            sCr.withUnsafeMutableBufferPointer { dstPtr in
                                dstPtr.baseAddress!.advanced(by: dstIdx).update(from: srcPtr, count: limit)
                            }
                        }
                    }
                }
            }
        }
        subPlane = PlaneData420(width: subDxY, height: subDyY, y: sY, cb: sCb, cr: sCr)
    }
    
    return (blocksY, blocksCb, blocksCr, subPlane)
}

@inline(__always)
func subtractCoeffs(currBlocks: inout [Block2D], predBlocks: inout [Block2D], isBase: Bool, size: Int) {
    let half = (size / 2)
    for i in currBlocks.indices {
        currBlocks[i].withView { vC in
            predBlocks[i].withView { vP in
                if isBase {
                    for y in 0..<half {
                        let ptrC = vC.rowPointer(y: y)
                        let ptrP = vP.rowPointer(y: y)
                        for x in 0..<half { ptrC[x] = (ptrC[x] &- ptrP[x]) }
                    }
                }
                for y in 0..<half {
                    let ptrC = vC.rowPointer(y: y).advanced(by: half)
                    let ptrP = vP.rowPointer(y: y).advanced(by: half)
                    for x in 0..<half { ptrC[x] = (ptrC[x] &- ptrP[x]) }
                }
                for y in half..<size {
                    let ptrC = vC.rowPointer(y: y)
                    let ptrP = vP.rowPointer(y: y)
                    for x in 0..<half { ptrC[x] = (ptrC[x] &- ptrP[x]) }
                }
                for y in half..<size {
                    let ptrC = vC.rowPointer(y: y).advanced(by: half)
                    let ptrP = vP.rowPointer(y: y).advanced(by: half)
                    for x in 0..<half { ptrC[x] = (ptrC[x] &- ptrP[x]) }
                }
            }
        }
    }
}

func encodeSpatialLayers(pd: PlaneData420, predictedPd: PlaneData420?, maxbitrate: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, isIFrame: Bool) async throws -> [UInt8] {
    // Extract subbands and residuals for all layers
    // Layer 2
    var (blocks2Y, blocks2Cb, blocks2Cr, sub2) = try await extractTransformBlocks(pd: pd, size: 32, qtY: qtY, qtC: qtC, isBase: false)
    var subPred2: PlaneData420? = nil
    if let pPd = predictedPd {
        var (pY, pCb, pCr, pSub) = try await extractTransformBlocks(pd: pPd, size: 32, qtY: qtY, qtC: qtC, isBase: false)
        subtractCoeffs(currBlocks: &blocks2Y, predBlocks: &pY, isBase: false, size: 32)
        subtractCoeffs(currBlocks: &blocks2Cb, predBlocks: &pCb, isBase: false, size: 32)
        subtractCoeffs(currBlocks: &blocks2Cr, predBlocks: &pCr, isBase: false, size: 32)
        subPred2 = pSub
    }
    for i in blocks2Y.indices { evaluateQuantizeLayer(block: &blocks2Y[i], size: 32, qt: qtY) }
    for i in blocks2Cb.indices { evaluateQuantizeLayer(block: &blocks2Cb[i], size: 32, qt: qtC) }
    for i in blocks2Cr.indices { evaluateQuantizeLayer(block: &blocks2Cr[i], size: 32, qt: qtC) }

    // Layer 1
    var (blocks1Y, blocks1Cb, blocks1Cr, sub1) = try await extractTransformBlocks(pd: sub2!, size: 16, qtY: qtY, qtC: qtC, isBase: false)
    var subPred1: PlaneData420? = nil
    if let pSub2 = subPred2 {
        var (pY, pCb, pCr, pSub) = try await extractTransformBlocks(pd: pSub2, size: 16, qtY: qtY, qtC: qtC, isBase: false)
        subtractCoeffs(currBlocks: &blocks1Y, predBlocks: &pY, isBase: false, size: 16)
        subtractCoeffs(currBlocks: &blocks1Cb, predBlocks: &pCb, isBase: false, size: 16)
        subtractCoeffs(currBlocks: &blocks1Cr, predBlocks: &pCr, isBase: false, size: 16)
        subPred1 = pSub
    }
    for i in blocks1Y.indices { evaluateQuantizeLayer(block: &blocks1Y[i], size: 16, qt: qtY) }
    for i in blocks1Cb.indices { evaluateQuantizeLayer(block: &blocks1Cb[i], size: 16, qt: qtC) }
    for i in blocks1Cr.indices { evaluateQuantizeLayer(block: &blocks1Cr[i], size: 16, qt: qtC) }

    // Layer 0 (Base)
    var (blocks0Y, blocks0Cb, blocks0Cr, _) = try await extractTransformBlocks(pd: sub1!, size: 8, qtY: qtY, qtC: qtC, isBase: true)
    if let pSub1 = subPred1 {
        var (pY, pCb, pCr, _) = try await extractTransformBlocks(pd: pSub1, size: 8, qtY: qtY, qtC: qtC, isBase: true)
        subtractCoeffs(currBlocks: &blocks0Y, predBlocks: &pY, isBase: true, size: 8)
        subtractCoeffs(currBlocks: &blocks0Cb, predBlocks: &pCb, isBase: true, size: 8)
        subtractCoeffs(currBlocks: &blocks0Cr, predBlocks: &pCr, isBase: true, size: 8)
    }
    for i in blocks0Y.indices { evaluateQuantizeBase(block: &blocks0Y[i], size: 8, qt: qtY) }
    for i in blocks0Cb.indices { evaluateQuantizeBase(block: &blocks0Cb[i], size: 8, qt: qtC) }
    for i in blocks0Cr.indices { evaluateQuantizeBase(block: &blocks0Cr[i], size: 8, qt: qtC) }

    // Unified CABAC Streams per Plane
    var ceY = CABACEncoder()
    var ctxsY = PlaneCABACContexts()
    encodePlaneBaseSubbands(ce: &ceY, ctxs: &ctxsY, blocks: &blocks0Y, size: 8, zeroThreshold: zeroThreshold)
    encodePlaneSubbands(ce: &ceY, ctxs: &ctxsY, blocks: &blocks1Y, size: 16, zeroThreshold: zeroThreshold)
    encodePlaneSubbands(ce: &ceY, ctxs: &ctxsY, blocks: &blocks2Y, size: 32, zeroThreshold: zeroThreshold)
    ceY.flush()

    var ceCb = CABACEncoder()
    var ctxsCb = PlaneCABACContexts()
    encodePlaneBaseSubbands(ce: &ceCb, ctxs: &ctxsCb, blocks: &blocks0Cb, size: 8, zeroThreshold: zeroThreshold)
    encodePlaneSubbands(ce: &ceCb, ctxs: &ctxsCb, blocks: &blocks1Cb, size: 16, zeroThreshold: zeroThreshold)
    encodePlaneSubbands(ce: &ceCb, ctxs: &ctxsCb, blocks: &blocks2Cb, size: 32, zeroThreshold: zeroThreshold)
    ceCb.flush()

    var ceCr = CABACEncoder()
    var ctxsCr = PlaneCABACContexts()
    encodePlaneBaseSubbands(ce: &ceCr, ctxs: &ctxsCr, blocks: &blocks0Cr, size: 8, zeroThreshold: zeroThreshold)
    encodePlaneSubbands(ce: &ceCr, ctxs: &ctxsCr, blocks: &blocks1Cr, size: 16, zeroThreshold: zeroThreshold)
    encodePlaneSubbands(ce: &ceCr, ctxs: &ctxsCr, blocks: &blocks2Cr, size: 32, zeroThreshold: zeroThreshold)
    ceCr.flush()

    var out: [UInt8] = []
    out.append(contentsOf: [0x56, 0x45, 0x56, 0x43, 0x03]) // 'VEVC' + unified version 3
    appendUInt16BE(&out, UInt16(pd.width))
    appendUInt16BE(&out, UInt16(pd.height))
    out.append(UInt8(qtY.step))
    out.append(UInt8(qtC.step))

    appendUInt32BE(&out, UInt32(ceY.data.count))
    out.append(contentsOf: ceY.data)
    appendUInt32BE(&out, UInt32(ceCb.data.count))
    out.append(contentsOf: ceCb.data)
    appendUInt32BE(&out, UInt32(ceCr.data.count))
    out.append(contentsOf: ceCr.data)

    return out
}
