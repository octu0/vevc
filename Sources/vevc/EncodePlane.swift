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
func extractTransformBlocks32(pd: PlaneData420, qtY: QuantizationTable, qtC: QuantizationTable) async throws -> (blocksY: [Block2D], blocksCb: [Block2D], blocksCr: [Block2D], subPlane: PlaneData420) {
    let dx = pd.width
    let dy = pd.height
    
    var subY: [Int16] = [Int16](repeating: 0, count: ((dx / 2) * (dy / 2)))
    var subCb: [Int16] = [Int16](repeating: 0, count: (((dx + 1) / 2 / 2) * ((dy + 1) / 2 / 2)))
    var subCr: [Int16] = [Int16](repeating: 0, count: (((dx + 1) / 2 / 2) * ((dy + 1) / 2 / 2)))
    
    let rY = pd.rY
    let rowCountY = ((dy + 32 - 1) / 32)
    let resultsY = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCountY))
    let chunkSize = 4
    let taskCountY = ((rowCountY + chunkSize - 1) / chunkSize)
    
    DispatchQueue.concurrentPerform(iterations: taskCountY) { taskIdx in
        let startRow = (taskIdx * chunkSize)
        let endRow = min((startRow + chunkSize), rowCountY)
        
        for i in startRow..<endRow {
            let h = (i * 32)
            var rowResults: [(Block2D, Int, Int)] = []
            for w in stride(from: 0, to: dx, by: 32) {
                var block = Block2D(width: 32, height: 32)
                block.withView { view in
                    for line in 0..<32 {
                        let row = rY.row(x: w, y: (h + line), size: 32)
                        view.setRow(offsetY: line, row: row)
                    }
                    _ = dwt2d_32(&view)
                }
                rowResults.append((block, w, h))
            }
            resultsY.value[i] = (h, rowResults)
        }
    }
    
    var blocksY: [Block2D] = []
    blocksY.reserveCapacity((rowCountY * ((dx + 32 - 1) / 32)))
    for i in 0..<rowCountY {
        guard let res = resultsY.value[i] else { continue }
        for j in res.1.indices {
            var (llBlock, w, h) = res.1[j]
            blocksY.append(llBlock)

            
            let subDxWidth = (dx / 2)
            let subDyHeight = (dy / 2)
            let destStartX = (w / 2)
            let destStartY = (h / 2)
            let subSize = (32 / 2)

            llBlock.withView { view in
                let subs = getSubbands32(view: view)
                let srcBase = subs.ll.base
                for blockY in 0..<subSize {
                    let dstY = (destStartY + blockY)
                    if subDyHeight <= dstY { continue }
                    let srcPtr = srcBase.advanced(by: (blockY * 32))
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
    let rowCountCb = ((cbDy + 32 - 1) / 32)
    let resultsCb = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCountCb))
    let taskCountCb = ((rowCountCb + chunkSize - 1) / chunkSize)
    
    DispatchQueue.concurrentPerform(iterations: taskCountCb) { taskIdx in
        let startRow = (taskIdx * chunkSize)
        let endRow = min((startRow + chunkSize), rowCountCb)
        
        for i in startRow..<endRow {
            let h = (i * 32)
            var rowResults: [(Block2D, Int, Int)] = []
            for w in stride(from: 0, to: cbDx, by: 32) {
                var block = Block2D(width: 32, height: 32)
                block.withView { view in
                    for line in 0..<32 {
                        let row = rCb.row(x: w, y: (h + line), size: 32)
                        view.setRow(offsetY: line, row: row)
                    }
                    _ = dwt2d_32(&view)
                }
                rowResults.append((block, w, h))
            }
            resultsCb.value[i] = (h, rowResults)
        }
    }
    
    var blocksCb: [Block2D] = []
    blocksCb.reserveCapacity((rowCountCb * ((cbDx + 32 - 1) / 32)))
    for i in 0..<rowCountCb {
        guard let res = resultsCb.value[i] else { continue }
        for j in res.1.indices {
            var (llBlock, w, h) = res.1[j]
            blocksCb.append(llBlock)

            let destStartX = (w / 2)
            let destStartY = (h / 2)
            let subSize = (32 / 2)

            llBlock.withView { view in
                let subs = getSubbands32(view: view)
                let srcBase = subs.ll.base
                for blockY in 0..<subSize {
                    let dstY = (destStartY + blockY)
                    if subCbDy <= dstY { continue }
                    let srcPtr = srcBase.advanced(by: (blockY * 32))
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
    let rowCountCr = ((cbDy + 32 - 1) / 32)
    let resultsCr = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCountCr))
    let taskCountCr = ((rowCountCr + chunkSize - 1) / chunkSize)
    
    DispatchQueue.concurrentPerform(iterations: taskCountCr) { taskIdx in
        let startRow = (taskIdx * chunkSize)
        let endRow = min((startRow + chunkSize), rowCountCr)
        
        for i in startRow..<endRow {
            let h = (i * 32)
            var rowResults: [(Block2D, Int, Int)] = []
            for w in stride(from: 0, to: cbDx, by: 32) {
                var block = Block2D(width: 32, height: 32)
                block.withView { view in
                    for line in 0..<32 {
                        let row = rCr.row(x: w, y: (h + line), size: 32)
                        view.setRow(offsetY: line, row: row)
                    }
                    _ = dwt2d_32(&view)
                }
                rowResults.append((block, w, h))
            }
            resultsCr.value[i] = (h, rowResults)
        }
    }
    
    var blocksCr: [Block2D] = []
    blocksCr.reserveCapacity((rowCountCr * ((cbDx + 32 - 1) / 32)))
    for i in 0..<rowCountCr {
        guard let res = resultsCr.value[i] else { continue }
        for j in res.1.indices {
            var (llBlock, w, h) = res.1[j]
            blocksCr.append(llBlock)

            
            let destStartX = (w / 2)
            let destStartY = (h / 2)
            let subSize = (32 / 2)

            llBlock.withView { view in
                let subs = getSubbands32(view: view)
                let srcBase = subs.ll.base
                for blockY in 0..<subSize {
                    let dstY = (destStartY + blockY)
                    if subCbDy <= dstY { continue }
                    let srcPtr = srcBase.advanced(by: (blockY * 32))
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

@inline(__always)
func extractTransformBlocks16(pd: PlaneData420, qtY: QuantizationTable, qtC: QuantizationTable) async throws -> (blocksY: [Block2D], blocksCb: [Block2D], blocksCr: [Block2D], subPlane: PlaneData420) {
    let dx = pd.width
    let dy = pd.height
    
    var subY: [Int16] = [Int16](repeating: 0, count: ((dx / 2) * (dy / 2)))
    var subCb: [Int16] = [Int16](repeating: 0, count: (((dx + 1) / 2 / 2) * ((dy + 1) / 2 / 2)))
    var subCr: [Int16] = [Int16](repeating: 0, count: (((dx + 1) / 2 / 2) * ((dy + 1) / 2 / 2)))
    
    let rY = pd.rY
    let rowCountY = ((dy + 16 - 1) / 16)
    let resultsY = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCountY))
    let chunkSize = 4
    let taskCountY = ((rowCountY + chunkSize - 1) / chunkSize)
    
    DispatchQueue.concurrentPerform(iterations: taskCountY) { taskIdx in
        let startRow = (taskIdx * chunkSize)
        let endRow = min((startRow + chunkSize), rowCountY)
        
        for i in startRow..<endRow {
            let h = (i * 16)
            var rowResults: [(Block2D, Int, Int)] = []
            for w in stride(from: 0, to: dx, by: 16) {
                var block = Block2D(width: 16, height: 16)
                block.withView { view in
                    for line in 0..<16 {
                        let row = rY.row(x: w, y: (h + line), size: 16)
                        view.setRow(offsetY: line, row: row)
                    }
                    _ = dwt2d_16(&view)
                }
                rowResults.append((block, w, h))
            }
            resultsY.value[i] = (h, rowResults)
        }
    }
    
    var blocksY: [Block2D] = []
    blocksY.reserveCapacity((rowCountY * ((dx + 16 - 1) / 16)))
    for i in 0..<rowCountY {
        guard let res = resultsY.value[i] else { continue }
        for j in res.1.indices {
            var (llBlock, w, h) = res.1[j]
            blocksY.append(llBlock)

            
            let subDxWidth = (dx / 2)
            let subDyHeight = (dy / 2)
            let destStartX = (w / 2)
            let destStartY = (h / 2)
            let subSize = (16 / 2)

            llBlock.withView { view in
                let subs = getSubbands16(view: view)
                let srcBase = subs.ll.base
                for blockY in 0..<subSize {
                    let dstY = (destStartY + blockY)
                    if subDyHeight <= dstY { continue }
                    let srcPtr = srcBase.advanced(by: (blockY * 16))
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
    let rowCountCb = ((cbDy + 16 - 1) / 16)
    let resultsCb = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCountCb))
    let taskCountCb = ((rowCountCb + chunkSize - 1) / chunkSize)
    
    DispatchQueue.concurrentPerform(iterations: taskCountCb) { taskIdx in
        let startRow = (taskIdx * chunkSize)
        let endRow = min((startRow + chunkSize), rowCountCb)
        
        for i in startRow..<endRow {
            let h = (i * 16)
            var rowResults: [(Block2D, Int, Int)] = []
            for w in stride(from: 0, to: cbDx, by: 16) {
                var block = Block2D(width: 16, height: 16)
                block.withView { view in
                    for line in 0..<16 {
                        let row = rCb.row(x: w, y: (h + line), size: 16)
                        view.setRow(offsetY: line, row: row)
                    }
                    _ = dwt2d_16(&view)
                }
                rowResults.append((block, w, h))
            }
            resultsCb.value[i] = (h, rowResults)
        }
    }
    
    var blocksCb: [Block2D] = []
    blocksCb.reserveCapacity((rowCountCb * ((cbDx + 16 - 1) / 16)))
    for i in 0..<rowCountCb {
        guard let res = resultsCb.value[i] else { continue }
        for j in res.1.indices {
            var (llBlock, w, h) = res.1[j]
            blocksCb.append(llBlock)

            let destStartX = (w / 2)
            let destStartY = (h / 2)
            let subSize = (16 / 2)

            llBlock.withView { view in
                let subs = getSubbands16(view: view)
                let srcBase = subs.ll.base
                for blockY in 0..<subSize {
                    let dstY = (destStartY + blockY)
                    if subCbDy <= dstY { continue }
                    let srcPtr = srcBase.advanced(by: (blockY * 16))
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
    let rowCountCr = ((cbDy + 16 - 1) / 16)
    let resultsCr = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCountCr))
    let taskCountCr = ((rowCountCr + chunkSize - 1) / chunkSize)
    
    DispatchQueue.concurrentPerform(iterations: taskCountCr) { taskIdx in
        let startRow = (taskIdx * chunkSize)
        let endRow = min((startRow + chunkSize), rowCountCr)
        
        for i in startRow..<endRow {
            let h = (i * 16)
            var rowResults: [(Block2D, Int, Int)] = []
            for w in stride(from: 0, to: cbDx, by: 16) {
                var block = Block2D(width: 16, height: 16)
                block.withView { view in
                    for line in 0..<16 {
                        let row = rCr.row(x: w, y: (h + line), size: 16)
                        view.setRow(offsetY: line, row: row)
                    }
                    _ = dwt2d_16(&view)
                }
                rowResults.append((block, w, h))
            }
            resultsCr.value[i] = (h, rowResults)
        }
    }
    
    var blocksCr: [Block2D] = []
    blocksCr.reserveCapacity((rowCountCr * ((cbDx + 16 - 1) / 16)))
    for i in 0..<rowCountCr {
        guard let res = resultsCr.value[i] else { continue }
        for j in res.1.indices {
            var (llBlock, w, h) = res.1[j]
            blocksCr.append(llBlock)

            
            let destStartX = (w / 2)
            let destStartY = (h / 2)
            let subSize = (16 / 2)

            llBlock.withView { view in
                let subs = getSubbands16(view: view)
                let srcBase = subs.ll.base
                for blockY in 0..<subSize {
                    let dstY = (destStartY + blockY)
                    if subCbDy <= dstY { continue }
                    let srcPtr = srcBase.advanced(by: (blockY * 16))
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

@inline(__always)
func extractTransformBlocksBase8(pd: PlaneData420, qtY: QuantizationTable, qtC: QuantizationTable) async throws -> (blocksY: [Block2D], blocksCb: [Block2D], blocksCr: [Block2D]) {
    let dx = pd.width
    let dy = pd.height
    
    
    let rY = pd.rY
    let rowCountY = ((dy + 8 - 1) / 8)
    let resultsY = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCountY))
    let chunkSize = 4
    let taskCountY = ((rowCountY + chunkSize - 1) / chunkSize)
    
    DispatchQueue.concurrentPerform(iterations: taskCountY) { taskIdx in
        let startRow = (taskIdx * chunkSize)
        let endRow = min((startRow + chunkSize), rowCountY)
        
        for i in startRow..<endRow {
            let h = (i * 8)
            var rowResults: [(Block2D, Int, Int)] = []
            for w in stride(from: 0, to: dx, by: 8) {
                var block = Block2D(width: 8, height: 8)
                block.withView { view in
                    for line in 0..<8 {
                        let row = rY.row(x: w, y: (h + line), size: 8)
                        view.setRow(offsetY: line, row: row)
                    }
                    _ = dwt2d_8(&view)
                }
                rowResults.append((block, w, h))
            }
            resultsY.value[i] = (h, rowResults)
        }
    }
    
    var blocksY: [Block2D] = []
    blocksY.reserveCapacity((rowCountY * ((dx + 8 - 1) / 8)))
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
    let rowCountCb = ((cbDy + 8 - 1) / 8)
    let resultsCb = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCountCb))
    let taskCountCb = ((rowCountCb + chunkSize - 1) / chunkSize)
    
    DispatchQueue.concurrentPerform(iterations: taskCountCb) { taskIdx in
        let startRow = (taskIdx * chunkSize)
        let endRow = min((startRow + chunkSize), rowCountCb)
        
        for i in startRow..<endRow {
            let h = (i * 8)
            var rowResults: [(Block2D, Int, Int)] = []
            for w in stride(from: 0, to: cbDx, by: 8) {
                var block = Block2D(width: 8, height: 8)
                block.withView { view in
                    for line in 0..<8 {
                        let row = rCb.row(x: w, y: (h + line), size: 8)
                        view.setRow(offsetY: line, row: row)
                    }
                    _ = dwt2d_8(&view)
                }
                rowResults.append((block, w, h))
            }
            resultsCb.value[i] = (h, rowResults)
        }
    }
    
    var blocksCb: [Block2D] = []
    blocksCb.reserveCapacity((rowCountCb * ((cbDx + 8 - 1) / 8)))
    for i in 0..<rowCountCb {
        guard let res = resultsCb.value[i] else { continue }
        for j in res.1.indices {
            let (llBlock, _, _) = res.1[j]
            blocksCb.append(llBlock)

        }
    }
    
    let rCr = pd.rCr
    let rowCountCr = ((cbDy + 8 - 1) / 8)
    let resultsCr = ConcurrentBox([(Int, [(Block2D, Int, Int)])?](repeating: nil, count: rowCountCr))
    let taskCountCr = ((rowCountCr + chunkSize - 1) / chunkSize)
    
    DispatchQueue.concurrentPerform(iterations: taskCountCr) { taskIdx in
        let startRow = (taskIdx * chunkSize)
        let endRow = min((startRow + chunkSize), rowCountCr)
        
        for i in startRow..<endRow {
            let h = (i * 8)
            var rowResults: [(Block2D, Int, Int)] = []
            for w in stride(from: 0, to: cbDx, by: 8) {
                var block = Block2D(width: 8, height: 8)
                block.withView { view in
                    for line in 0..<8 {
                        let row = rCr.row(x: w, y: (h + line), size: 8)
                        view.setRow(offsetY: line, row: row)
                    }
                    _ = dwt2d_8(&view)
                }
                rowResults.append((block, w, h))
            }
            resultsCr.value[i] = (h, rowResults)
        }
    }
    
    var blocksCr: [Block2D] = []
    blocksCr.reserveCapacity((rowCountCr * ((cbDx + 8 - 1) / 8)))
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
    var (subBlocksY, subBlocksCb, subBlocksCr, subPlane) = try await extractTransformBlocks32(pd: pd, qtY: qtY, qtC: qtC)
    
    var subPredPlane: PlaneData420? = nil
    if let pPd = predictedPd {
        var (pY, pCb, pCr, pSub) = try await extractTransformBlocks32(pd: pPd, qtY: qtY, qtC: qtC)
        subtractCoeffs32(currBlocks: &subBlocksY, predBlocks: &pY)
        subtractCoeffs32(currBlocks: &subBlocksCb, predBlocks: &pCb)
        subtractCoeffs32(currBlocks: &subBlocksCr, predBlocks: &pCr)
        subPredPlane = pSub
    }
    
    for i in subBlocksY.indices { evaluateQuantizeLayer32(block: &subBlocksY[i], qt: qtY) }
    for i in subBlocksCb.indices { evaluateQuantizeLayer32(block: &subBlocksCb[i], qt: qtC) }
    for i in subBlocksCr.indices { evaluateQuantizeLayer32(block: &subBlocksCr[i], qt: qtC) }
    
    async let taskBufY = encodePlaneSubbands32(blocks: subBlocksY, zeroThreshold: zeroThreshold)
    async let taskBufCb = encodePlaneSubbands32(blocks: subBlocksCb, zeroThreshold: zeroThreshold)
    async let taskBufCr = encodePlaneSubbands32(blocks: subBlocksCr, zeroThreshold: zeroThreshold)

    let bufY = await taskBufY
    let bufCb = await taskBufCb
    let bufCr = await taskBufCr
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
    var (subBlocksY, subBlocksCb, subBlocksCr, subPlane) = try await extractTransformBlocks16(pd: pd, qtY: qtY, qtC: qtC)
    
    var subPredPlane: PlaneData420? = nil
    if let pPd = predictedPd {
        var (pY, pCb, pCr, pSub) = try await extractTransformBlocks16(pd: pPd, qtY: qtY, qtC: qtC)
        subtractCoeffs16(currBlocks: &subBlocksY, predBlocks: &pY)
        subtractCoeffs16(currBlocks: &subBlocksCb, predBlocks: &pCb)
        subtractCoeffs16(currBlocks: &subBlocksCr, predBlocks: &pCr)
        subPredPlane = pSub
    }
    
    for i in subBlocksY.indices { evaluateQuantizeLayer16(block: &subBlocksY[i], qt: qtY) }
    for i in subBlocksCb.indices { evaluateQuantizeLayer16(block: &subBlocksCb[i], qt: qtC) }
    for i in subBlocksCr.indices { evaluateQuantizeLayer16(block: &subBlocksCr[i], qt: qtC) }
    
    async let taskBufY = encodePlaneSubbands16(blocks: subBlocksY, zeroThreshold: zeroThreshold)
    async let taskBufCb = encodePlaneSubbands16(blocks: subBlocksCb, zeroThreshold: zeroThreshold)
    async let taskBufCr = encodePlaneSubbands16(blocks: subBlocksCr, zeroThreshold: zeroThreshold)

    let bufY = await taskBufY
    let bufCb = await taskBufCb
    let bufCr = await taskBufCr
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
    var (subBlocksY, subBlocksCb, subBlocksCr) = try await extractTransformBlocksBase8(pd: pd, qtY: qtY, qtC: qtC)
    if let pPd = predictedPd {
        var (pY, pCb, pCr) = try await extractTransformBlocksBase8(pd: pPd, qtY: qtY, qtC: qtC)
        subtractCoeffsBase8(currBlocks: &subBlocksY, predBlocks: &pY)
        subtractCoeffsBase8(currBlocks: &subBlocksCb, predBlocks: &pCb)
        subtractCoeffsBase8(currBlocks: &subBlocksCr, predBlocks: &pCr)
    }
    
    for i in subBlocksY.indices { evaluateQuantizeBase8(block: &subBlocksY[i], qt: qtY) }
    for i in subBlocksCb.indices { evaluateQuantizeBase8(block: &subBlocksCb[i], qt: qtC) }
    for i in subBlocksCr.indices { evaluateQuantizeBase8(block: &subBlocksCr[i], qt: qtC) }
    
    async let taskBufY = encodePlaneBaseSubbands8(blocks: subBlocksY, zeroThreshold: zeroThreshold)
    async let taskBufCb = encodePlaneBaseSubbands8(blocks: subBlocksCb, zeroThreshold: zeroThreshold)
    async let taskBufCr = encodePlaneBaseSubbands8(blocks: subBlocksCr, zeroThreshold: zeroThreshold)

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
