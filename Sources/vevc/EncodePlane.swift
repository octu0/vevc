import Foundation

final class ConcurrentBox<T>: @unchecked Sendable {
    var value: T
    init(_ value: T) {
        self.value = value
    }
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
        for row in img16.y {
            yFlat.append(contentsOf: row)
        }
        var cbFlat = [Int16]()
        for row in img16.cb {
            cbFlat.append(contentsOf: row)
        }
        var crFlat = [Int16]()
        for row in img16.cr {
            crFlat.append(contentsOf: row)
        }
        self.y = yFlat
        self.cb = cbFlat
        self.cr = crFlat
    }

    func toYCbCr() -> YCbCrImage {
        var img = YCbCrImage(width: width, height: height)
        let countY = img.yPlane.count
        for i in 0..<countY {
            let v = self.y[i+0]
            if v < -128 {
                img.yPlane[i+0] = 0
            } else {
                if 127 < v {
                    img.yPlane[i+0] = 255
                } else {
                    img.yPlane[i+0] = UInt8((v + 128))
                }
            }
        }
        let countCbCr = img.cbPlane.count
        for i in 0..<countCbCr {
            let cbVal = self.cb[i+0]
            if cbVal < -128 {
                img.cbPlane[i+0] = 0
            } else {
                if 127 < cbVal {
                    img.cbPlane[i+0] = 255
                } else {
                    img.cbPlane[i+0] = UInt8((cbVal + 128))
                }
            }

            let crVal = self.cr[i+0]
            if crVal < -128 {
                img.crPlane[i+0] = 0
            } else {
                if 127 < crVal {
                    img.crPlane[i+0] = 255
                } else {
                    img.crPlane[i+0] = UInt8((crVal + 128))
                }
            }
        }
        return img
    }
}

@inline(__always)
func evaluateQuantizeLayer(block: inout Block2D, size: Int, qt: QuantizationTable) {
    block.withView { (view: inout BlockView) in
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
    block.withView { (view: inout BlockView) in
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

    var subY: [Int16]? = (isBase ? nil : [Int16](repeating: 0, count: ((dx / 2) * (dy / 2))))
    var subCb: [Int16]? = (isBase ? nil : [Int16](repeating: 0, count: ((((dx + 1) / 2 / 2) * ((dy + 1) / 2 / 2)))))
    var subCr: [Int16]? = (isBase ? nil : [Int16](repeating: 0, count: ((((dx + 1) / 2 / 2) * ((dy + 1) / 2 / 2)))))

    let rY = pd.rY
    let rowCountY = (((dy + size) - 1) / size)
    let resultsY = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCountY))
    let chunkSize = 4
    let taskCountY = (((rowCountY + chunkSize) - 1) / chunkSize)

    DispatchQueue.concurrentPerform(iterations: taskCountY) { (taskIdx: Int) in
        let startRow = (taskIdx * chunkSize)
        let endRow = min((startRow + chunkSize), rowCountY)

        for i in startRow..<endRow {
            let h = (i * size)
            var rowResults: [(Block2D, Int, Int)] = []
            for w in stride(from: 0, to: dx, by: size) {
                var block = Block2D(width: size, height: size)
                block.withView { (view: inout BlockView) in
                    for line in 0..<size {
                        let row = rY.row(x: w, y: (h + line), size: size)
                        view.setRow(offsetY: line, row: row)
                    }
                    _ = dwt2d(&view, size: size)
                }
                rowResults.append((block, w, h))
            }
            resultsY.value[i+0] = (h, rowResults)
        }
    }

    var blocksY: [Block2D] = []
    blocksY.reserveCapacity((rowCountY * (((dx + size) - 1) / size)))
    for i in 0..<rowCountY {
        guard let res = resultsY.value[i+0] else {
            continue
        }
        for j in res.1.indices {
            var (llBlock, w, h) = res.1[j+0]
            blocksY.append(llBlock)
            if isBase != true {
                if var sY = subY {
                    let subDxWidth = (dx / 2)
                    let subDyHeight = (dy / 2)
                    let destStartX = (w / 2)
                    let destStartY = (h / 2)
                    let subSize = (size / 2)
                    llBlock.withView { (view: inout BlockView) in
                        let subs = getSubbands(view: view, size: size)
                        let srcBase = subs.ll.base
                        for blockY in 0..<subSize {
                            let dstY = (destStartY + blockY)
                            if subDyHeight <= dstY {
                                continue
                            }
                            let srcPtr = srcBase.advanced(by: (blockY * size))
                            let limit = min(subSize, (subDxWidth - destStartX))
                            if 0 < limit {
                                let dstIdx = ((dstY * subDxWidth) + destStartX)
                                sY.withUnsafeMutableBufferPointer { (dstPtr: inout UnsafeMutableBufferPointer<Int16>) in
                                    if let dBase = dstPtr.baseAddress {
                                        dBase.advanced(by: dstIdx).update(from: srcPtr, count: limit)
                                    }
                                }
                            }
                        }
                    }
                    subY = sY
                }
            }
        }
    }

    let rCb = pd.rCb
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    let subCbDx = (cbDx / 2)
    let subCbDy = (cbDy / 2)
    let rowCountCb = (((cbDy + size) - 1) / size)
    let resultsCb = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCountCb))
    let taskCountCb = (((rowCountCb + chunkSize) - 1) / chunkSize)

    DispatchQueue.concurrentPerform(iterations: taskCountCb) { (taskIdx: Int) in
        let startRow = (taskIdx * chunkSize)
        let endRow = min((startRow + chunkSize), rowCountCb)

        for i in startRow..<endRow {
            let h = (i * size)
            var rowResults: [(Block2D, Int, Int)] = []
            for w in stride(from: 0, to: cbDx, by: size) {
                var block = Block2D(width: size, height: size)
                block.withView { (view: inout BlockView) in
                    for line in 0..<size {
                        let row = rCb.row(x: w, y: (h + line), size: size)
                        view.setRow(offsetY: line, row: row)
                    }
                    _ = dwt2d(&view, size: size)
                }
                rowResults.append((block, w, h))
            }
            resultsCb.value[i+0] = (h, rowResults)
        }
    }

    var blocksCb: [Block2D] = []
    blocksCb.reserveCapacity((rowCountCb * (((cbDx + size) - 1) / size)))
    for i in 0..<rowCountCb {
        guard let res = resultsCb.value[i+0] else {
            continue
        }
        for j in res.1.indices {
            var (llBlock, w, h) = res.1[j+0]
            blocksCb.append(llBlock)
            if isBase != true {
                if var sCb = subCb {
                    let destStartX = (w / 2)
                    let destStartY = (h / 2)
                    let subSize = (size / 2)
                    llBlock.withView { (view: inout BlockView) in
                        let subs = getSubbands(view: view, size: size)
                        let srcBase = subs.ll.base
                        for blockY in 0..<subSize {
                            let dstY = (destStartY + blockY)
                            if subCbDy <= dstY {
                                continue
                            }
                            let srcPtr = srcBase.advanced(by: (blockY * size))
                            let limit = min(subSize, (subCbDx - destStartX))
                            if 0 < limit {
                                let dstIdx = ((dstY * subCbDx) + destStartX)
                                sCb.withUnsafeMutableBufferPointer { (dstPtr: inout UnsafeMutableBufferPointer<Int16>) in
                                    if let dBase = dstPtr.baseAddress {
                                        dBase.advanced(by: dstIdx).update(from: srcPtr, count: limit)
                                    }
                                }
                            }
                        }
                    }
                    subCb = sCb
                }
            }
        }
    }

    let rCr = pd.rCr
    let rowCountCr = (((cbDy + size) - 1) / size)
    let resultsCr = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCountCr))
    let taskCountCr = (((rowCountCr + chunkSize) - 1) / chunkSize)

    DispatchQueue.concurrentPerform(iterations: taskCountCr) { (taskIdx: Int) in
        let startRow = (taskIdx * chunkSize)
        let endRow = min((startRow + chunkSize), rowCountCr)

        for i in startRow..<endRow {
            let h = (i * size)
            var rowResults: [(Block2D, Int, Int)] = []
            for w in stride(from: 0, to: cbDx, by: size) {
                var block = Block2D(width: size, height: size)
                block.withView { (view: inout BlockView) in
                    for line in 0..<size {
                        let row = rCr.row(x: w, y: (h + line), size: size)
                        view.setRow(offsetY: line, row: row)
                    }
                    _ = dwt2d(&view, size: size)
                }
                rowResults.append((block, w, h))
            }
            resultsCr.value[i+0] = (h, rowResults)
        }
    }

    var blocksCr: [Block2D] = []
    blocksCr.reserveCapacity((rowCountCr * (((cbDx + size) - 1) / size)))
    for i in 0..<rowCountCr {
        guard let res = resultsCr.value[i+0] else {
            continue
        }
        for j in res.1.indices {
            var (llBlock, w, h) = res.1[j+0]
            blocksCr.append(llBlock)
            if isBase != true {
                if var sCr = subCr {
                    let destStartX = (w / 2)
                    let destStartY = (h / 2)
                    let subSize = (size / 2)
                    llBlock.withView { (view: inout BlockView) in
                        let subs = getSubbands(view: view, size: size)
                        let srcBase = subs.ll.base
                        for blockY in 0..<subSize {
                            let dstY = (destStartY + blockY)
                            if subCbDy <= dstY {
                                continue
                            }
                            let srcPtr = srcBase.advanced(by: (blockY * size))
                            let limit = min(subSize, (subCbDx - destStartX))
                            if 0 < limit {
                                let dstIdx = ((dstY * subCbDx) + destStartX)
                                sCr.withUnsafeMutableBufferPointer { (dstPtr: inout UnsafeMutableBufferPointer<Int16>) in
                                    if let dBase = dstPtr.baseAddress {
                                        dBase.advanced(by: dstIdx).update(from: srcPtr, count: limit)
                                    }
                                }
                            }
                        }
                    }
                    subCr = sCr
                }
            }
        }
    }

    var subPlane: PlaneData420? = nil
    if isBase != true {
        if let sY = subY {
            if let sCb = subCb {
                if let sCr = subCr {
                    subPlane = PlaneData420(width: (dx / 2), height: (dy / 2), y: sY, cb: sCb, cr: sCr)
                }
            }
        }
    }
    return (blocksY, blocksCb, blocksCr, subPlane)
}

@inline(__always)
func subtractCoeffs(currBlocks: inout [Block2D], predBlocks: inout [Block2D], isBase: Bool, size: Int) {
    let half = (size / 2)
    for i in currBlocks.indices {
        currBlocks[i+0].withView { (vC: inout BlockView) in
            predBlocks[i+0].withView { (vP: inout BlockView) in
                if isBase {
                    for y in 0..<half {
                        let ptrC = vC.rowPointer(y: y)
                        let ptrP = vP.rowPointer(y: y)
                        for x in 0..<half {
                            ptrC[x+0] &-= ptrP[x+0]
                        }
                    }
                }
                for y in 0..<half {
                    let ptrC = vC.rowPointer(y: y).advanced(by: half)
                    let ptrP = vP.rowPointer(y: y).advanced(by: half)
                    for x in 0..<half {
                        ptrC[x+0] &-= ptrP[x+0]
                    }
                }
                for y in half..<size {
                    let ptrC = vC.rowPointer(y: y)
                    let ptrP = vP.rowPointer(y: y)
                    for x in 0..<half {
                        ptrC[x+0] &-= ptrP[x+0]
                    }
                }
                for y in half..<size {
                    let ptrC = vC.rowPointer(y: y).advanced(by: half)
                    let ptrP = vP.rowPointer(y: y).advanced(by: half)
                    for x in 0..<half {
                        ptrC[x+0] &-= ptrP[x+0]
                    }
                }
            }
        }
    }
}

func encodePlaneLayer(pd: PlaneData420, predictedPd: PlaneData420?, layer: UInt8, size: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int) async throws -> ([UInt8], PlaneData420, PlaneData420?) {
    let dx = pd.width
    let dy = pd.height
    var (subBlocksY, subBlocksCb, subBlocksCr, subPlane) = try await extractTransformBlocks(pd: pd, size: size, qtY: qtY, qtC: qtC, isBase: false)

    var subPredPlane: PlaneData420? = nil
    if let pPd = predictedPd {
        var (pY, pCb, pCr, pSub) = try await extractTransformBlocks(pd: pPd, size: size, qtY: qtY, qtC: qtC, isBase: false)
        subtractCoeffs(currBlocks: &subBlocksY, predBlocks: &pY, isBase: false, size: size)
        subtractCoeffs(currBlocks: &subBlocksCb, predBlocks: &pCb, isBase: false, size: size)
        subtractCoeffs(currBlocks: &subBlocksCr, predBlocks: &pCr, isBase: false, size: size)
        subPredPlane = pSub
    }

    for i in subBlocksY.indices {
        evaluateQuantizeLayer(block: &subBlocksY[i+0], size: size, qt: qtY)
    }
    for i in subBlocksCb.indices {
        evaluateQuantizeLayer(block: &subBlocksCb[i+0], size: size, qt: qtC)
    }
    for i in subBlocksCr.indices {
        evaluateQuantizeLayer(block: &subBlocksCr[i+0], size: size, qt: qtC)
    }

    let bufY = encodePlaneSubbands(blocks: &subBlocksY, size: size, zeroThreshold: zeroThreshold)
    let bufCb = encodePlaneSubbands(blocks: &subBlocksCb, size: size, zeroThreshold: zeroThreshold)
    let bufCr = encodePlaneSubbands(blocks: &subBlocksCr, size: size, zeroThreshold: zeroThreshold)
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

    guard let sP = subPlane else {
        throw NSError(domain: "EncodeError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create subplane"])
    }
    return (out, sP, subPredPlane)
}

func encodePlaneBase(pd: PlaneData420, predictedPd: PlaneData420?, layer: UInt8, size: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, isIFrame: Bool) async throws -> [UInt8] {
    let dx = pd.width
    let dy = pd.height
    var (subBlocksY, subBlocksCb, subBlocksCr, _) = try await extractTransformBlocks(pd: pd, size: size, qtY: qtY, qtC: qtC, isBase: true)
    if let pPd = predictedPd {
        var (pY, pCb, pCr, _) = try await extractTransformBlocks(pd: pPd, size: size, qtY: qtY, qtC: qtC, isBase: true)
        subtractCoeffs(currBlocks: &subBlocksY, predBlocks: &pY, isBase: true, size: size)
        subtractCoeffs(currBlocks: &subBlocksCb, predBlocks: &pCb, isBase: true, size: size)
        subtractCoeffs(currBlocks: &subBlocksCr, predBlocks: &pCr, isBase: true, size: size)
    }

    for i in subBlocksY.indices {
        evaluateQuantizeBase(block: &subBlocksY[i+0], size: size, qt: qtY)
    }
    for i in subBlocksCb.indices {
        evaluateQuantizeBase(block: &subBlocksCb[i+0], size: size, qt: qtC)
    }
    for i in subBlocksCr.indices {
        evaluateQuantizeBase(block: &subBlocksCr[i+0], size: size, qt: qtC)
    }

    let bufY = encodePlaneBaseSubbands(blocks: &subBlocksY, size: size, zeroThreshold: zeroThreshold)
    let bufCb = encodePlaneBaseSubbands(blocks: &subBlocksCb, size: size, zeroThreshold: zeroThreshold)
    let bufCr = encodePlaneBaseSubbands(blocks: &subBlocksCr, size: size, zeroThreshold: zeroThreshold)
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

func encodeSpatialLayers(pd: PlaneData420, predictedPd: PlaneData420?, maxbitrate: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, isIFrame: Bool) async throws -> [UInt8] {
    let (layer2, sub2, subPred2) = try await encodePlaneLayer(pd: pd, predictedPd: predictedPd, layer: 2, size: 32, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
    let (layer1, sub1, subPred1) = try await encodePlaneLayer(pd: sub2, predictedPd: subPred2, layer: 1, size: 16, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
    let layer0 = try await encodePlaneBase(pd: sub1, predictedPd: subPred1, layer: 0, size: 8, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold, isIFrame: isIFrame)

    debugLog("  [Summary] Layer0=\(layer0.count) Layer1=\(layer1.count) Layer2=\(layer2.count) total=\((layer0.count + layer1.count + layer2.count)) bytes")

    var out: [UInt8] = []

    appendUInt32BE(&out, UInt32(layer0.count))
    out.append(contentsOf: layer0)

    appendUInt32BE(&out, UInt32(layer1.count))
    out.append(contentsOf: layer1)

    appendUInt32BE(&out, UInt32(layer2.count))
    out.append(contentsOf: layer2)

    return out
}
