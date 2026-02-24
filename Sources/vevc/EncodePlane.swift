// MARK: - Encode Plane Arrays

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

func encodePlaneLayer(pd: PlaneData420, layer: UInt8, size: Int, qt: QuantizationTable) async throws -> ([UInt8], PlaneData420) {
    let dx = pd.width
    let dy = pd.height
    
    let subDx = dx / 2
    let subDy = dy / 2
    
    // Next layer LL buffers
    var subY = [Int16](repeating: 0, count: subDx * subDy)
    var subCb = [Int16](repeating: 0, count: ((dx + 1) / 2 / 2) * ((dy + 1) / 2 / 2))
    var subCr = [Int16](repeating: 0, count: ((dx + 1) / 2 / 2) * ((dy + 1) / 2 / 2))
    
    var bufY: [[UInt8]] = []
    var bufCb: [[UInt8]] = []
    var bufCr: [[UInt8]] = []
    
    let rY = pd.rY
    try await withThrowingTaskGroup(of: (Int, [([UInt8], Block2D, Int, Int)]).self) { group in
        for h in stride(from: 0, to: dy, by: size) {
            group.addTask {
                var rowResults: [([UInt8], Block2D, Int, Int)] = []
                for w in stride(from: 0, to: dx, by: size) {
                    let (data, ll) = try transformLayerFunc(rows: rY.row, w: w, h: h, size: size, qt: qt)
                    rowResults.append((data, ll, w, h))
                }
                return (h, rowResults)
            }
        }
        var results: [(Int, [([UInt8], Block2D, Int, Int)])] = []
        for try await res in group {
            results.append(res)
        }
        results.sort { $0.0 < $1.0 }
        
        for i in results.indices {
            for j in results[i].1.indices {
                var (data, llBlock, w, h) = results[i].1[j]
                bufY.append(data)
                
                // Copy LL block to subY
                let destStartX = w / 2
                let destStartY = h / 2
                let subSize = size / 2
                llBlock.withView { view in
                    for blockY in 0..<subSize {
                        let dstY = destStartY + blockY
                        if subDy <= dstY {
                            continue
                        }
                        let srcPtr = view.rowPointer(y: blockY)
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
    }
    
    let rCb = pd.rCb
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let subCbDx = cbDx / 2
    let subCbDy = cbDy / 2
    try await withThrowingTaskGroup(of: (Int, [([UInt8], Block2D, Int, Int)]).self) { group in
        for h in stride(from: 0, to: cbDy, by: size) {
            group.addTask {
                var rowResults: [([UInt8], Block2D, Int, Int)] = []
                for w in stride(from: 0, to: cbDx, by: size) {
                    let (data, ll) = try transformLayerFunc(rows: rCb.row, w: w, h: h, size: size, qt: qt)
                    rowResults.append((data, ll, w, h))
                }
                return (h, rowResults)
            }
        }
        var results: [(Int, [([UInt8], Block2D, Int, Int)])] = []
        for try await res in group { results.append(res) }
        results.sort { $0.0 < $1.0 }
        
        for i in results.indices {
            for j in results[i].1.indices {
                var (data, llBlock, w, h) = results[i].1[j]
                bufCb.append(data)
                let destStartX = w / 2
                let destStartY = h / 2
                let subSize = size / 2
                llBlock.withView { view in
                    for blockY in 0..<subSize {
                        let dstY = destStartY + blockY
                        if dstY >= subCbDy { continue }
                        let srcPtr = view.rowPointer(y: blockY)
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
    }
    
    let rCr = pd.rCr
    try await withThrowingTaskGroup(of: (Int, [([UInt8], Block2D, Int, Int)]).self) { group in
        for h in stride(from: 0, to: cbDy, by: size) {
            group.addTask {
                var rowResults: [([UInt8], Block2D, Int, Int)] = []
                for w in stride(from: 0, to: cbDx, by: size) {
                    let (data, ll) = try transformLayerFunc(rows: rCr.row, w: w, h: h, size: size, qt: qt)
                    rowResults.append((data, ll, w, h))
                }
                return (h, rowResults)
            }
        }
        var results: [(Int, [([UInt8], Block2D, Int, Int)])] = []
        for try await res in group { results.append(res) }
        results.sort { $0.0 < $1.0 }
        
        for i in results.indices {
            for j in results[i].1.indices {
                var (data, llBlock, w, h) = results[i].1[j]
                bufCr.append(data)
                let destStartX = w / 2
                let destStartY = h / 2
                let subSize = size / 2
                llBlock.withView { view in
                    for blockY in 0..<subSize {
                        let dstY = destStartY + blockY
                        if subCbDy <= dstY {
                            continue
                        }
                        let srcPtr = view.rowPointer(y: blockY)
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
    }
    
    var out: [UInt8] = []
    out.append(contentsOf: [0x56, 0x45, 0x56, 0x43, layer]) // 'VEVC' + layer
    appendUInt16BE(&out, UInt16(dx))
    appendUInt16BE(&out, UInt16(dy))
    out.append(UInt8(qt.step))
    
    appendUInt16BE(&out, UInt16(bufY.count))
    for b in bufY { appendUInt16BE(&out, UInt16(b.count)); out.append(contentsOf: b) }
    
    appendUInt16BE(&out, UInt16(bufCb.count))
    for b in bufCb { appendUInt16BE(&out, UInt16(b.count)); out.append(contentsOf: b) }
    
    appendUInt16BE(&out, UInt16(bufCr.count))
    for b in bufCr { appendUInt16BE(&out, UInt16(b.count)); out.append(contentsOf: b) }
    
    let subPlane = PlaneData420(width: subDx, height: subDy, y: subY, cb: subCb, cr: subCr)
    return (out, subPlane)
}

func encodePlaneBase(pd: PlaneData420, layer: UInt8, size: Int, qt: QuantizationTable) async throws -> [UInt8] {
    let dx = pd.width
    let dy = pd.height
    
    var bufY: [[UInt8]] = []
    var bufCb: [[UInt8]] = []
    var bufCr: [[UInt8]] = []
    
    let rY = pd.rY
    try await withThrowingTaskGroup(of: (Int, [([UInt8], Int, Int)]).self) { group in
        for h in stride(from: 0, to: dy, by: size) {
            group.addTask {
                var rowResults: [([UInt8], Int, Int)] = []
                for w in stride(from: 0, to: dx, by: size) {
                    let data = try transformBaseFunc(rows: rY.row, w: w, h: h, size: size, qt: qt)
                    rowResults.append((data, w, h))
                }
                return (h, rowResults)
            }
        }
        var results: [(Int, [([UInt8], Int, Int)])] = []
        for try await res in group {
            results.append(res)
        }
        results.sort { $0.0 < $1.0 }
        for (_, rowBlocks) in results {
            for (data, _, _) in rowBlocks { bufY.append(data) }
        }
    }
    
    let rCb = pd.rCb
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    try await withThrowingTaskGroup(of: (Int, [([UInt8], Int, Int)]).self) { group in
        for h in stride(from: 0, to: cbDy, by: size) {
            group.addTask {
                var rowResults: [([UInt8], Int, Int)] = []
                for w in stride(from: 0, to: cbDx, by: size) {
                    let data = try transformBaseFunc(rows: rCb.row, w: w, h: h, size: size, qt: qt)
                    rowResults.append((data, w, h))
                }
                return (h, rowResults)
            }
        }
        var results: [(Int, [([UInt8], Int, Int)])] = []
        for try await res in group {
            results.append(res)
        }
        results.sort { $0.0 < $1.0 }
        for (_, rowBlocks) in results {
            for (data, _, _) in rowBlocks {
                bufCb.append(data)
            }
        }
    }
    
    let rCr = pd.rCr
    try await withThrowingTaskGroup(of: (Int, [([UInt8], Int, Int)]).self) { group in
        for h in stride(from: 0, to: cbDy, by: size) {
            group.addTask {
                var rowResults: [([UInt8], Int, Int)] = []
                for w in stride(from: 0, to: cbDx, by: size) {
                    let data = try transformBaseFunc(rows: rCr.row, w: w, h: h, size: size, qt: qt)
                    rowResults.append((data, w, h))
                }
                return (h, rowResults)
            }
        }
        var results: [(Int, [([UInt8], Int, Int)])] = []
        for try await res in group {
            results.append(res)
        }
        results.sort { $0.0 < $1.0 }
        for (_, rowBlocks) in results {
            for (data, _, _) in rowBlocks {
                bufCr.append(data)
            }
        }
    }
    
    var out: [UInt8] = []
    out.append(contentsOf: [0x56, 0x45, 0x56, 0x43, layer]) // 'VEVC' + layer
    appendUInt16BE(&out, UInt16(dx))
    appendUInt16BE(&out, UInt16(dy))
    out.append(UInt8(qt.step))
    
    appendUInt16BE(&out, UInt16(bufY.count))
    for b in bufY { appendUInt16BE(&out, UInt16(b.count)); out.append(contentsOf: b) }
    
    appendUInt16BE(&out, UInt16(bufCb.count))
    for b in bufCb { appendUInt16BE(&out, UInt16(b.count)); out.append(contentsOf: b) }
    
    appendUInt16BE(&out, UInt16(bufCr.count))
    for b in bufCr { appendUInt16BE(&out, UInt16(b.count)); out.append(contentsOf: b) }
    
    return out
}

func encodeSpatialLayers(pd: PlaneData420, maxbitrate: Int, qt: QuantizationTable) async throws -> [UInt8] {
    let (layer2, sub2) = try await encodePlaneLayer(pd: pd, layer: 2, size: 32, qt: qt)
    let (layer1, sub1) = try await encodePlaneLayer(pd: sub2, layer: 1, size: 16, qt: qt)
    let layer0 = try await encodePlaneBase(pd: sub1, layer: 0, size: 8, qt: qt)
    
    var out: [UInt8] = []
    
    appendUInt32BE(&out, UInt32(layer0.count))
    out.append(contentsOf: layer0)
    
    appendUInt32BE(&out, UInt32(layer1.count))
    out.append(contentsOf: layer1)
    
    appendUInt32BE(&out, UInt32(layer2.count))
    out.append(contentsOf: layer2)
    
    return out
}
