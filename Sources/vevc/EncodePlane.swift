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
        // flattened arrays
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

func encodePlaneLayer(pd: PlaneData420, layer: UInt8, size: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int) async throws -> ([UInt8], PlaneData420) {
    let dx = pd.width
    let dy = pd.height
    
    let subDx = dx / 2
    let subDy = dy / 2
    
    var subY = [Int16](repeating: 0, count: subDx * subDy)
    var subCb = [Int16](repeating: 0, count: ((dx + 1) / 2 / 2) * ((dy + 1) / 2 / 2))
    var subCr = [Int16](repeating: 0, count: ((dx + 1) / 2 / 2) * ((dy + 1) / 2 / 2))
    
    let rY = pd.rY
    let rowCountY = (dy + size - 1) / size
    let resultsY = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCountY))
    let errorY = ConcurrentBox<Error?>(nil)

    let chunkSize = 4
    let taskCountY = (rowCountY + chunkSize - 1) / chunkSize
    
    DispatchQueue.concurrentPerform(iterations: taskCountY) { taskIdx in
        let startRow = taskIdx * chunkSize
        let endRow = min(startRow + chunkSize, rowCountY)
        
        for i in startRow..<endRow {
            let h = i * size
            var rowResults: [(Block2D, Int, Int)] = []
            for w in stride(from: 0, to: dx, by: size) {
                var block = Block2D(width: size, height: size)
                block.withView { view in
                    for line in 0..<size {
                        let row = rY.row(x: w, y: (h + line), size: size)
                        view.setRow(offsetY: line, row: row)
                    }
                }
                transformLayer(block: &block, size: size, qt: qtY)
                rowResults.append((block, w, h))
            }
            resultsY.value[i] = (h, rowResults)
        }
    }
    if let err = errorY.value { throw err }

    var subBlocksY: [Block2D] = []
    for i in 0..<rowCountY {
        guard let res = resultsY.value[i] else { continue }
        for j in res.1.indices {
            var (llBlock, w, h) = res.1[j]
            subBlocksY.append(llBlock)
            
            let destStartX = w / 2
            let destStartY = h / 2
            let subSize = size / 2
            llBlock.withView { view in
                let subs = getSubbands(view: view, size: size)
                let srcBase = subs.ll.base
                for blockY in 0..<subSize {
                    let dstY = destStartY + blockY
                    if subDy <= dstY { continue }
                    let srcPtr = srcBase.advanced(by: blockY * size)
                    let limit = min(subSize, subDx - destStartX)
                    if 0 < limit {
                        let dstIdx = dstY * subDx + destStartX
                        subY.withUnsafeMutableBufferPointer { dstPtr in
                            dstPtr.baseAddress!.advanced(by: dstIdx).update(from: srcPtr, count: limit)
                        }
                    }
                }
            }
        }
    }
    
    let rCb = pd.rCb
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let subCbDx = cbDx / 2
    let subCbDy = cbDy / 2
    let rowCountCb = (cbDy + size - 1) / size
    let resultsCb = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCountCb))
    let errorCb = ConcurrentBox<Error?>(nil)
    let taskCountCb = (rowCountCb + chunkSize - 1) / chunkSize
    
    DispatchQueue.concurrentPerform(iterations: taskCountCb) { taskIdx in
        let startRow = taskIdx * chunkSize
        let endRow = min(startRow + chunkSize, rowCountCb)
        
        for i in startRow..<endRow {
            let h = i * size
            var rowResults: [(Block2D, Int, Int)] = []
            for w in stride(from: 0, to: cbDx, by: size) {
                var block = Block2D(width: size, height: size)
                block.withView { view in
                    for line in 0..<size {
                        let row = rCb.row(x: w, y: (h + line), size: size)
                        view.setRow(offsetY: line, row: row)
                    }
                }
                transformLayer(block: &block, size: size, qt: qtC)
                rowResults.append((block, w, h))
            }
            resultsCb.value[i] = (h, rowResults)
        }
    }
    if let err = errorCb.value { throw err }
    
    var subBlocksCb: [Block2D] = []
    for i in 0..<rowCountCb {
        guard let res = resultsCb.value[i] else { continue }
        for j in res.1.indices {
            var (llBlock, w, h) = res.1[j]
            subBlocksCb.append(llBlock)
            let destStartX = w / 2
            let destStartY = h / 2
            let subSize = size / 2
            llBlock.withView { view in
                let subs = getSubbands(view: view, size: size)
                let srcBase = subs.ll.base
                for blockY in 0..<subSize {
                    let dstY = destStartY + blockY
                    if dstY >= subCbDy { continue }
                    let srcPtr = srcBase.advanced(by: blockY * size)
                    let limit = min(subSize, subCbDx - destStartX)
                    if 0 < limit {
                        let dstIdx = dstY * subCbDx + destStartX
                        subCb.withUnsafeMutableBufferPointer { dstPtr in
                            dstPtr.baseAddress!.advanced(by: dstIdx).update(from: srcPtr, count: limit)
                        }
                    }
                }
            }
        }
    }
    
    let rCr = pd.rCr
    let rowCountCr = (cbDy + size - 1) / size
    let resultsCr = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCountCr))
    let errorCr = ConcurrentBox<Error?>(nil)
    let taskCountCr = (rowCountCr + chunkSize - 1) / chunkSize
    
    DispatchQueue.concurrentPerform(iterations: taskCountCr) { taskIdx in
        let startRow = taskIdx * chunkSize
        let endRow = min(startRow + chunkSize, rowCountCr)
        
        for i in startRow..<endRow {
            let h = i * size
            var rowResults: [(Block2D, Int, Int)] = []
            for w in stride(from: 0, to: cbDx, by: size) {
                var block = Block2D(width: size, height: size)
                block.withView { view in
                    for line in 0..<size {
                        let row = rCr.row(x: w, y: (h + line), size: size)
                        view.setRow(offsetY: line, row: row)
                    }
                }
                transformLayer(block: &block, size: size, qt: qtC)
                rowResults.append((block, w, h))
            }
            resultsCr.value[i] = (h, rowResults)
        }
    }
    if let err = errorCr.value { throw err }
    
    var subBlocksCr: [Block2D] = []
    for i in 0..<rowCountCr {
        guard let res = resultsCr.value[i] else { continue }
        for j in res.1.indices {
            var (llBlock, w, h) = res.1[j]
            subBlocksCr.append(llBlock)
            let destStartX = w / 2
            let destStartY = h / 2
            let subSize = size / 2
            llBlock.withView { view in
                let subs = getSubbands(view: view, size: size)
                let srcBase = subs.ll.base
                for blockY in 0..<subSize {
                    let dstY = destStartY + blockY
                    if subCbDy <= dstY { continue }
                    let srcPtr = srcBase.advanced(by: blockY * size)
                    let limit = min(subSize, subCbDx - destStartX)
                    if 0 < limit {
                        let dstIdx = dstY * subCbDx + destStartX
                        subCr.withUnsafeMutableBufferPointer { dstPtr in
                            dstPtr.baseAddress!.advanced(by: dstIdx).update(from: srcPtr, count: limit)
                        }
                    }
                }
            }
        }
    }
    
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
    
    let subPlane = PlaneData420(width: subDx, height: subDy, y: subY, cb: subCb, cr: subCr)
    return (out, subPlane)
}

func encodePlaneBase(pd: PlaneData420, layer: UInt8, size: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int) async throws -> [UInt8] {
    let dx = pd.width
    let dy = pd.height
    let chunkSize = 4
    
    let rY = pd.rY
    let rowCountY = (dy + size - 1) / size
    let resultsY = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCountY))
    let errorY = ConcurrentBox<Error?>(nil)
    let taskCountY = (rowCountY + chunkSize - 1) / chunkSize
    
    DispatchQueue.concurrentPerform(iterations: taskCountY) { taskIdx in
        let startRow = taskIdx * chunkSize
        let endRow = min(startRow + chunkSize, rowCountY)
        
        for i in startRow..<endRow {
            let h = i * size
            var rowResults: [(Block2D, Int, Int)] = []
            for w in stride(from: 0, to: dx, by: size) {
                var block = Block2D(width: size, height: size)
                block.withView { view in
                    for line in 0..<size {
                        let row = rY.row(x: w, y: (h + line), size: size)
                        view.setRow(offsetY: line, row: row)
                    }
                }
                transformBase(block: &block, size: size, qt: qtY)
                rowResults.append((block, w, h))
            }
            resultsY.value[i] = (h, rowResults)
        }
    }
    if let err = errorY.value { throw err }
    
    var subBlocksY: [Block2D] = []
    for i in 0..<rowCountY {
        guard let res = resultsY.value[i] else { continue }
        for j in res.1.indices {
            let (block, _, _) = res.1[j]
            subBlocksY.append(block)
        }
    }
    
    let rCb = pd.rCb
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let rowCountCb = (cbDy + size - 1) / size
    let resultsCb = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCountCb))
    let errorCb = ConcurrentBox<Error?>(nil)
    let taskCountCb = (rowCountCb + chunkSize - 1) / chunkSize
    DispatchQueue.concurrentPerform(iterations: taskCountCb) { taskIdx in
        let startRow = taskIdx * chunkSize
        let endRow = min(startRow + chunkSize, rowCountCb)
        
        for i in startRow..<endRow {
            let h = i * size
            var rowResults: [(Block2D, Int, Int)] = []
            for w in stride(from: 0, to: cbDx, by: size) {
                var block = Block2D(width: size, height: size)
                block.withView { view in
                    for line in 0..<size {
                        let row = rCb.row(x: w, y: (h + line), size: size)
                        view.setRow(offsetY: line, row: row)
                    }
                }
                transformBase(block: &block, size: size, qt: qtC)
                rowResults.append((block, w, h))
            }
            resultsCb.value[i] = (h, rowResults)
        }
    }
    if let err = errorCb.value { throw err }
    
    var subBlocksCb: [Block2D] = []
    for i in 0..<rowCountCb {
        guard let res = resultsCb.value[i] else { continue }
        for j in res.1.indices {
            let (block, _, _) = res.1[j]
            subBlocksCb.append(block)
        }
    }
    
    let rCr = pd.rCr
    let rowCountCr = (cbDy + size - 1) / size
    let resultsCr = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCountCr))
    let errorCr = ConcurrentBox<Error?>(nil)
    let taskCountCr = (rowCountCr + chunkSize - 1) / chunkSize
    DispatchQueue.concurrentPerform(iterations: taskCountCr) { taskIdx in
        let startRow = taskIdx * chunkSize
        let endRow = min(startRow + chunkSize, rowCountCr)
        
        for i in startRow..<endRow {
            let h = i * size
            var rowResults: [(Block2D, Int, Int)] = []
            for w in stride(from: 0, to: cbDx, by: size) {
                var block = Block2D(width: size, height: size)
                block.withView { view in
                    for line in 0..<size {
                        let row = rCr.row(x: w, y: (h + line), size: size)
                        view.setRow(offsetY: line, row: row)
                    }
                }
                transformBase(block: &block, size: size, qt: qtC)
                rowResults.append((block, w, h))
            }
            resultsCr.value[i] = (h, rowResults)
        }
    }
    if let err = errorCr.value { throw err }
    
    var subBlocksCr: [Block2D] = []
    for i in 0..<rowCountCr {
        guard let res = resultsCr.value[i] else { continue }
        for j in res.1.indices {
            let (block, _, _) = res.1[j]
            subBlocksCr.append(block)
        }
    }
    
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

func encodeSpatialLayers(pd: PlaneData420, maxbitrate: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int) async throws -> [UInt8] {
    let (layer2, sub2) = try await encodePlaneLayer(pd: pd, layer: 2, size: 32, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
    let (layer1, sub1) = try await encodePlaneLayer(pd: sub2, layer: 1, size: 16, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
    let layer0 = try await encodePlaneBase(pd: sub1, layer: 0, size: 8, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
    
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
