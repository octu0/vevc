// MARK: - Encode

@inline(__always)
func toUint16(_ n: Int16) -> UInt16 {
    return UInt16(bitPattern: ((n &<< 1) ^ (n >> 15)))
}

@inline(__always)
func blockEncode(rw: inout RiceWriter, block: BlockView, size: Int, k: UInt8) {
    for y in 0..<size {
        let ptr = block.rowPointer(y: y)
        for x in 0..<size {
            rw.write(val: UInt16(bitPattern: ptr[x]), k: k)
        }
    }
}

@inline(__always)
func getSubbands(view: BlockView, size: Int) -> Subbands {
    let half = size / 2
    let base = view.base
    return Subbands(
        ll: BlockView(base: base, width: half, height: half, stride: size),
        hl: BlockView(base: base.advanced(by: half), width: half, height: half, stride: size),
        lh: BlockView(base: base.advanced(by: half * size), width: half, height: half, stride: size),
        hh: BlockView(base: base.advanced(by: half * size + half), width: half, height: half, stride: size),
        size: half
    )
}

@inline(__always)
func blockEncodeDPCM(rw: inout RiceWriter, block: BlockView, size: Int, k: UInt8) {
    var prevVal: Int16 = 0
    for y in 0..<size {
        let ptr = block.rowPointer(y: y)
        for x in 0..<size {
            let val = ptr[x]
            let diff = val - prevVal
            rw.write(val: toUint16(diff), k: k)
            prevVal = val
        }
    }
}


// MARK: - Byte Serialization Helpers

@inline(__always)
func appendUInt16BE(_ out: inout [UInt8], _ val: UInt16) {
    out.append(UInt8(val >> 8))
    out.append(UInt8(val & 0xFF))
}

@inline(__always)
func appendUInt32BE(_ out: inout [UInt8], _ val: UInt32) {
    out.append(UInt8((val >> 24) & 0xFF))
    out.append(UInt8((val >> 16) & 0xFF))
    out.append(UInt8((val >> 8) & 0xFF))
    out.append(UInt8(val & 0xFF))
}

// MARK: - Transform Functions

@inline(__always)
func isEffectivelyZero(hl: inout BlockView, lh: inout BlockView, hh: inout BlockView, size: Int, threshold: Int) -> Bool {
    for y in 0..<size {
        let ptrHL = hl.rowPointer(y: y)
        let ptrLH = lh.rowPointer(y: y)
        let ptrHH = hh.rowPointer(y: y)
        for x in 0..<size {
            if abs(Int(ptrHL[x])) > threshold || abs(Int(ptrLH[x])) > threshold || abs(Int(ptrHH[x])) > threshold {
                return false
            }
        }
    }
    
    // threshold以下のノイズのみであった場合、後段のエントロピー結合やデコーダとの不整合を防ぐため
    // 実際にメモリ上の値をすべて0にクリアする
    for y in 0..<size {
        let ptrHL = hl.rowPointer(y: y)
        let ptrLH = lh.rowPointer(y: y)
        let ptrHH = hh.rowPointer(y: y)
        for x in 0..<size {
            ptrHL[x] = 0
            ptrLH[x] = 0
            ptrHH[x] = 0
        }
    }
    
    return true
}

@inline(__always)
func isEffectivelyZeroBase(ll: BlockView, hl: inout BlockView, lh: inout BlockView, hh: inout BlockView, size: Int, threshold: Int) -> Bool {
    // LL帯域は視覚への影響が大きい（DPCM残差）ため、一切の閾値緩和(Deadzone)を行わない
    for y in 0..<size {
        let ptrLL = ll.rowPointer(y: y)
        for x in 0..<size {
            if ptrLL[x] != 0 {
                return false
            }
        }
    }
    
    for y in 0..<size {
        let ptrHL = hl.rowPointer(y: y)
        let ptrLH = lh.rowPointer(y: y)
        let ptrHH = hh.rowPointer(y: y)
        for x in 0..<size {
            if abs(Int(ptrHL[x])) > threshold || abs(Int(ptrLH[x])) > threshold || abs(Int(ptrHH[x])) > threshold {
                return false
            }
        }
    }
    
    // 高周波帯域のみゼロクリアする
    for y in 0..<size {
        let ptrHL = hl.rowPointer(y: y)
        let ptrLH = lh.rowPointer(y: y)
        let ptrHH = hh.rowPointer(y: y)
        for x in 0..<size {
            ptrHL[x] = 0
            ptrLH[x] = 0
            ptrHH[x] = 0
        }
    }
    
    return true
}

@inline(__always)
func transformLayer(block: inout Block2D, size: Int, qt: QuantizationTable) {
    block.withView { view in
        var sub = dwt2d(&view, size: size)
        quantizeMidSignedMapping(&sub.hl, qt: qt)
        quantizeMidSignedMapping(&sub.lh, qt: qt)
        quantizeHighSignedMapping(&sub.hh, qt: qt)
    }
}

@inline(__always)
func transformBase(block: inout Block2D, size: Int, qt: QuantizationTable) {
    block.withView { view in
        var sub = dwt2d(&view, size: size)
        quantizeLow(&sub.ll, qt: qt)
        quantizeMidSignedMapping(&sub.hl, qt: qt)
        quantizeMidSignedMapping(&sub.lh, qt: qt)
        quantizeHighSignedMapping(&sub.hh, qt: qt)
    }
}

private enum SubbandType { case ll, hl, lh, hh }

private func evaluateK(blocks: inout [Block2D], indices: [Int], size: Int, type: SubbandType, isDPCM: Bool, k: UInt8) -> Int {
    var bitCount = 0
    var zeroCount: UInt16 = 0
    let half = size / 2
    
    @inline(__always)
    func simWrite(_ val: UInt16) {
        if val == 0 {
            zeroCount &+= 1
            return
        }
        if zeroCount > 0 {
            if zeroCount < 255 {
                let q = zeroCount >> k
                bitCount += Int(q) + 1 + Int(k)
            } else {
                let q = UInt16(255) >> k
                bitCount += Int(q) + 1 + Int(k) + 16
            }
            zeroCount = 0
        }
        let q = val >> k
        bitCount += Int(q) + 1 + Int(k)
    }
    
    for i in indices {
        blocks[i].withView { view in
            let subs = getSubbands(view: view, size: size)
            let b: BlockView
            switch type {
            case .ll: b = subs.ll
            case .hl: b = subs.hl
            case .lh: b = subs.lh
            case .hh: b = subs.hh
            }
            if isDPCM {
                var prev: Int16 = 0
                for y in 0..<half {
                    let ptr = b.rowPointer(y: y)
                    for x in 0..<half {
                        let val = ptr[x]
                        let diff = val - prev
                        simWrite(toUint16(diff))
                        prev = val
                    }
                }
            } else {
                for y in 0..<half {
                    let ptr = b.rowPointer(y: y)
                    for x in 0..<half {
                        simWrite(UInt16(bitPattern: ptr[x]))
                    }
                }
            }
        }
    }
    
    if zeroCount > 0 {
        if zeroCount < 255 {
            let q = zeroCount >> k
            bitCount += Int(q) + 1 + Int(k)
        } else {
            let q = UInt16(255) >> k
            bitCount += Int(q) + 1 + Int(k) + 16
        }
    }
    return bitCount
}

private func estimateOptimalK(blocks: inout [Block2D], indices: [Int], size: Int, type: SubbandType, isDPCM: Bool) -> UInt8 {
    if indices.isEmpty { return 0 }
    var bestK: UInt8 = 0
    var minBits = Int.max
    for k in UInt8(0)...UInt8(6) {
        let bits = evaluateK(blocks: &blocks, indices: indices, size: size, type: type, isDPCM: isDPCM, k: k)
        if bits < minBits {
            minBits = bits
            bestK = k
        }
    }
    return bestK
}

func encodePlaneSubbands(blocks: inout [Block2D], size: Int, zeroThreshold: Int) -> [UInt8] {
    var bwFlags = BitWriter()
    var nonZeroIndices: [Int] = []
    
    for i in blocks.indices {
        blocks[i].withView { view in
            let subs = getSubbands(view: view, size: size)
            var hl = subs.hl
            var lh = subs.lh
            var hh = subs.hh
            if isEffectivelyZero(hl: &hl, lh: &lh, hh: &hh, size: subs.size, threshold: zeroThreshold) {
                bwFlags.writeBit(1)
            } else {
                bwFlags.writeBit(0)
                nonZeroIndices.append(i)
            }
        }
    }
    bwFlags.flush()
    
    var bwData = BitWriter()
    
    let kHL = estimateOptimalK(blocks: &blocks, indices: nonZeroIndices, size: size, type: .hl, isDPCM: false)
    bwData.writeBits(val: UInt16(kHL), n: 4)
    RiceWriter.withWriter(&bwData, flushBits: false) { rw in
        for i in nonZeroIndices {
            blocks[i].withView { view in
                let subs = getSubbands(view: view, size: size)
                blockEncode(rw: &rw, block: subs.hl, size: subs.size, k: kHL)
            }
        }
    }
    
    let kLH = estimateOptimalK(blocks: &blocks, indices: nonZeroIndices, size: size, type: .lh, isDPCM: false)
    bwData.writeBits(val: UInt16(kLH), n: 4)
    RiceWriter.withWriter(&bwData, flushBits: false) { rw in
        for i in nonZeroIndices {
            blocks[i].withView { view in
                let subs = getSubbands(view: view, size: size)
                blockEncode(rw: &rw, block: subs.lh, size: subs.size, k: kLH)
            }
        }
    }
    
    let kHH = estimateOptimalK(blocks: &blocks, indices: nonZeroIndices, size: size, type: .hh, isDPCM: false)
    bwData.writeBits(val: UInt16(kHH), n: 4)
    RiceWriter.withWriter(&bwData, flushBits: false) { rw in
        for i in nonZeroIndices {
            blocks[i].withView { view in
                let subs = getSubbands(view: view, size: size)
                blockEncode(rw: &rw, block: subs.hh, size: subs.size, k: kHH)
            }
        }
    }
    
    bwData.flush()
    var out = bwFlags.data
    out.append(contentsOf: bwData.data)
    return out
}

func encodePlaneBaseSubbands(blocks: inout [Block2D], size: Int, zeroThreshold: Int) -> [UInt8] {
    var bwFlags = BitWriter()
    var nonZeroIndices: [Int] = []
    
    for i in blocks.indices {
        blocks[i].withView { view in
            let subs = getSubbands(view: view, size: size)
            var hl = subs.hl
            var lh = subs.lh
            var hh = subs.hh
            if isEffectivelyZeroBase(ll: subs.ll, hl: &hl, lh: &lh, hh: &hh, size: subs.size, threshold: zeroThreshold) {
                bwFlags.writeBit(1)
            } else {
                bwFlags.writeBit(0)
                nonZeroIndices.append(i)
            }
        }
    }
    bwFlags.flush()
    
    var bwData = BitWriter()
    
    let kLL = estimateOptimalK(blocks: &blocks, indices: nonZeroIndices, size: size, type: .ll, isDPCM: true)
    bwData.writeBits(val: UInt16(kLL), n: 4)
    RiceWriter.withWriter(&bwData, flushBits: false) { rw in
        for i in nonZeroIndices {
            blocks[i].withView { view in
                let subs = getSubbands(view: view, size: size)
                blockEncodeDPCM(rw: &rw, block: subs.ll, size: subs.size, k: kLL)
            }
        }
    }
    
    let kHL = estimateOptimalK(blocks: &blocks, indices: nonZeroIndices, size: size, type: .hl, isDPCM: false)
    bwData.writeBits(val: UInt16(kHL), n: 4)
    RiceWriter.withWriter(&bwData, flushBits: false) { rw in
        for i in nonZeroIndices {
            blocks[i].withView { view in
                let subs = getSubbands(view: view, size: size)
                blockEncode(rw: &rw, block: subs.hl, size: subs.size, k: kHL)
            }
        }
    }
    
    let kLH = estimateOptimalK(blocks: &blocks, indices: nonZeroIndices, size: size, type: .lh, isDPCM: false)
    bwData.writeBits(val: UInt16(kLH), n: 4)
    RiceWriter.withWriter(&bwData, flushBits: false) { rw in
        for i in nonZeroIndices {
            blocks[i].withView { view in
                let subs = getSubbands(view: view, size: size)
                blockEncode(rw: &rw, block: subs.lh, size: subs.size, k: kLH)
            }
        }
    }
    
    let kHH = estimateOptimalK(blocks: &blocks, indices: nonZeroIndices, size: size, type: .hh, isDPCM: false)
    bwData.writeBits(val: UInt16(kHH), n: 4)
    RiceWriter.withWriter(&bwData, flushBits: false) { rw in
        for i in nonZeroIndices {
            blocks[i].withView { view in
                let subs = getSubbands(view: view, size: size)
                blockEncode(rw: &rw, block: subs.hh, size: subs.size, k: kHH)
            }
        }
    }
    
    bwData.flush()
    var out = bwFlags.data
    out.append(contentsOf: bwData.data)
    return out
}

private func estimateRiceBitsDPCM(block: BlockView, size: Int) -> Int {
    var sumDiffAbs = 0
    let count = size * size
    var prev: Int16 = 0
    
    for y in 0..<size {
        let ptr = block.rowPointer(y: y)
        for x in 0..<size {
            let val = ptr[x]
            let diff = abs(Int(val - prev))
            sumDiffAbs += diff
            prev = val
        }
    }
    
    if count == 0 { return 0 }
    
    let mean = Double(sumDiffAbs) / Double(count)
    let meanInt = Int(mean)
    let k = (meanInt < 1) ? 0 : (Int.bitWidth - 1 - meanInt.leadingZeroBitCount)
    
    let divisorShift = max(0, k - 1)
    let bodyBits = sumDiffAbs >> divisorShift
    let headerBits = count * (1 + k)
    
    return bodyBits + headerBits
}

private enum PlaneType { case y, cb, cr }

private func fetchBlock(reader: ImageReader, plane: PlaneType, x: Int, y: Int, w: Int, h: Int) -> Block2D {
    var block = Block2D(width: w, height: h)
    block.withView { view in
        for i in 0..<h {
            let row: [Int16]
            switch plane {
            case .y:  row = reader.rowY(x: x, y: y + i, size: w)
            case .cb: row = reader.rowCb(x: x, y: y + i, size: w)
            case .cr: row = reader.rowCr(x: x, y: y + i, size: w)
            }
            view.setRow(offsetY: i, row: row)
        }
    }
    return block
}

private func measureBlockBits(block: inout Block2D, size: Int, qt: QuantizationTable) -> Int {
    var sub = block.withView { view in
        return dwt2d(&view, size: size)
    }
    
    quantizeLow(&sub.ll, qt: qt)
    quantizeMid(&sub.hl, qt: qt)
    quantizeMid(&sub.lh, qt: qt)
    quantizeHigh(&sub.hh, qt: qt)
    
    var hl = sub.hl
    var lh = sub.lh
    var hh = sub.hh
    if isEffectivelyZeroBase(ll: sub.ll, hl: &hl, lh: &lh, hh: &hh, size: sub.size, threshold: 0) {
        return 1
    }
    
    var bits = 1
    bits += estimateRiceBitsDPCM(block: sub.ll, size: sub.size)
    bits += estimateRiceBits(block: sub.hl, size: sub.size)
    bits += estimateRiceBits(block: sub.lh, size: sub.size)
    bits += estimateRiceBits(block: sub.hh, size: sub.size)
    
    return bits
}

private func estimateRiceBits(block: BlockView, size: Int) -> Int {
    var sumAbs = 0
    let count = size * size
    
    for y in 0..<size {
        let ptr = block.rowPointer(y: y)
        for x in 0..<size {
            sumAbs += abs(Int(ptr[x]))
        }
    }
    
    if count == 0 { return 0 }
    
    let mean = Double(sumAbs) / Double(count)
    let meanInt = Int(mean)
    let k = (meanInt < 1) ? 0 : (Int.bitWidth - 1 - meanInt.leadingZeroBitCount)
    
    let divisorShift = max(0, k - 1)
    let bodyBits = sumAbs >> divisorShift
    let headerBits = count * (1 + k)
    
    return bodyBits + headerBits
}

func estimateQuantization(img: YCbCrImage, targetBits: Int) -> QuantizationTable {
    let probeStep = 64
    let qt = QuantizationTable(baseStep: probeStep)
    
    let size = 8
    let w = (img.width / size)
    let h = (img.height / size)
    
    let points: [(Int, Int)] = [
        (0, 0),                                    // Top-Left
        ((img.width - w), 0),                      // Top-Right
        (0, (img.height - h)),                     // Bottom-Left
        ((img.width - w), (img.height - h)),       // Bottom-Right
        (((img.width - w) / 2), 0),                // Top-Center
        ((img.width - w), ((img.height - h) / 2)), // Right-Center
        (((img.width - w) / 2), (img.height - h)), // Bottom-Center
        (0, ((img.height - h) / 2)),               // Left-Center
    ]
    
    var totalSampleBits = 0
    let reader = ImageReader(img: img)

    enum PlaneType { case y, cb, cr }
    
    @inline(__always)
    func fetchBlock(reader: ImageReader, plane: PlaneType, x: Int, y: Int, w: Int, h: Int) -> Block2D {
        var block = Block2D(width: w, height: h)
        block.withView { view in
            for i in 0..<h {
                let row: [Int16]
                switch plane {
                case .y:  row = reader.rowY(x: x, y: y + i, size: w)
                case .cb: row = reader.rowCb(x: x, y: y + i, size: w)
                case .cr: row = reader.rowCr(x: x, y: y + i, size: w)
                }
                view.setRow(offsetY: i, row: row)
            }
        }
        return block
    }
    
    for (sx, sy) in points {
        // Y Plane
        var blockY = fetchBlock(reader: reader, plane: .y, x: sx, y: sy, w: w, h: h)
        totalSampleBits += measureBlockBits(block: &blockY, size: size, qt: qt)
        
        // Cb Plane
        var blockCb = fetchBlock(reader: reader, plane: .cb, x: sx, y: sy, w: w, h: h)
        totalSampleBits += measureBlockBits(block: &blockCb, size: size, qt: qt)
        
        // Cr Plane
        var blockCr = fetchBlock(reader: reader, plane: .cr, x: sx, y: sy, w: w, h: h)
        totalSampleBits += measureBlockBits(block: &blockCr, size: size, qt: qt)
    }
    
    let samplePixels = points.count * (w * h) * 3 // Y+Cb+Cr
    let totalPixels = img.width * img.height * 3
    
    let estimatedTotalBits = Double(totalSampleBits) * (Double(totalPixels) / Double(samplePixels))
        
    let ratio = estimatedTotalBits / Double(targetBits)
    let predictedStep = Double(probeStep) * ratio
    let q = Int(max(1, predictedStep))
    
    return QuantizationTable(baseStep: q)
}

struct Int16Reader {
    let data: [Int16]
    let width: Int
    let height: Int
    
    @inline(__always)
    func row(x: Int, y: Int, size: Int) -> [Int16] {
        var r = [Int16](repeating: 0, count: size)
        if y < height {
            let limit = min(size, width - x)
            if limit > 0 {
                data.withUnsafeBufferPointer { ptr in
                    let base = ptr.baseAddress!.advanced(by: y * width + x)
                    r.withUnsafeMutableBufferPointer { dst in
                        dst.baseAddress!.update(from: base, count: limit)
                    }
                }
            }
        }
        return r
    }
}

@inline(__always)
func toPlaneData420(images: [YCbCrImage]) -> [PlaneData420] {
    return images.map { img in
        let y = img.yPlane.map { Int16($0) - 128 }
        let cWidth = (img.width + 1) / 2
        let cHeight = (img.height + 1) / 2
        var cb = [Int16](repeating: 0, count: cWidth * cHeight)
        var cr = [Int16](repeating: 0, count: cWidth * cHeight)
        
        if img.ratio == .ratio444 {
            for cy in 0..<cHeight {
                let py = cy * 2
                for cx in 0..<cWidth {
                    let px = cx * 2
                    let srcOffset = py * img.width + px
                    let dstOffset = cy * cWidth + cx
                    if srcOffset < img.cbPlane.count {
                        cb[dstOffset] = Int16(img.cbPlane[srcOffset]) - 128
                        cr[dstOffset] = Int16(img.crPlane[srcOffset]) - 128
                    }
                }
            }
        } else {
            cb = img.cbPlane.map { Int16($0) - 128 }
            cr = img.crPlane.map { Int16($0) - 128 }
        }
        
        return PlaneData420(width: img.width, height: img.height, y: y, cb: cb, cr: cr)
    }
}

@inline(__always)
func subtractPlanes(curr: PlaneData420, predicted: PlaneData420) async -> PlaneData420 {
    @Sendable
    func sub(c: [Int16], p: [Int16]) -> [Int16] {
        let count = c.count
        var res = [Int16](repeating: 0, count: count)
        c.withUnsafeBufferPointer { cPtr in
            p.withUnsafeBufferPointer { pPtr in
                for i in 0..<count { res[i] = cPtr[i] - pPtr[i] }
            }
        }
        return res
    }
    
    async let y = sub(c: curr.y, p: predicted.y)
    async let cb = sub(c: curr.cb, p: predicted.cb)
    async let cr = sub(c: curr.cr, p: predicted.cr)
    
    return PlaneData420(width: curr.width, height: curr.height, y: await y, cb: await cb, cr: await cr)
}

@inline(__always)
func addPlanes(residual: PlaneData420, predicted: PlaneData420) async -> PlaneData420 {
    @Sendable
    func add(r: [Int16], p: [Int16]) -> [Int16] {
        let count = r.count
        var curr = [Int16](repeating: 0, count: count)
        r.withUnsafeBufferPointer { rPtr in
            p.withUnsafeBufferPointer { pPtr in
                for i in 0..<count { curr[i] = rPtr[i] + pPtr[i] }
            }
        }
        return curr
    }
    
    async let y = add(r: residual.y, p: predicted.y)
    async let cb = add(r: residual.cb, p: predicted.cb)
    async let cr = add(r: residual.cr, p: predicted.cr)
    
    return PlaneData420(width: residual.width, height: residual.height, y: await y, cb: await cb, cr: await cr)
}

@inline(__always)
func shiftPlane(_ plane: PlaneData420, dx: Int, dy: Int) async -> PlaneData420 {
    if dx == 0 && dy == 0 { return plane }
    
    @Sendable
    func shift(data: [Int16], w: Int, h: Int, sX: Int, sY: Int) -> [Int16] {
        if w == 0 || h == 0 { return data }
        
        var out = [Int16](repeating: 0, count: w * h)
        
        data.withUnsafeBufferPointer { dPtr in
            guard let pData = dPtr.baseAddress else { return }
            out.withUnsafeMutableBufferPointer { oPtr in
                guard let pOut = oPtr.baseAddress else { return }
                
                let eX = ((sX % w) + w) % w
                let eY = ((sY % h) + h) % h
                
                for dstY in 0..<h {
                    let dstRow = dstY * w
                    let srcY = (dstY - eY + h) % h
                    let srcRow = srcY * w
                    
                    if eX == 0 {
                        pOut.advanced(by: dstRow).update(from: pData.advanced(by: srcRow), count: w)
                    } else {
                        let part1Len = eX
                        let part2Len = w - eX
                        
                        pOut.advanced(by: dstRow).update(from: pData.advanced(by: srcRow + w - eX), count: part1Len)
                        pOut.advanced(by: dstRow + eX).update(from: pData.advanced(by: srcRow), count: part2Len)
                    }
                }
            }
        }
        
        return out
    }
    
    async let yTask = shift(data: plane.y, w: plane.width, h: plane.height, sX: dx, sY: dy)
    async let cbTask = shift(data: plane.cb, w: (plane.width + 1) / 2, h: (plane.height + 1) / 2, sX: dx / 2, sY: dy / 2)
    async let crTask = shift(data: plane.cr, w: (plane.width + 1) / 2, h: (plane.height + 1) / 2, sX: dx / 2, sY: dy / 2)
    
    return PlaneData420(width: plane.width, height: plane.height, y: await yTask, cb: await cbTask, cr: await crTask)
}



public func encode(images: [YCbCrImage], maxbitrate: Int, zeroThreshold: Int = 0, gopSize: Int = 8) async throws -> [UInt8] {
    if images.isEmpty { return [] }
    
    let qt = estimateQuantization(img: images[0], targetBits: maxbitrate)
    var out: [UInt8] = []
    
    var prevReconstructed: PlaneData420? = nil
    let planes = toPlaneData420(images: images)
    
    for i in 0..<planes.count {
        let curr = planes[i]
        
        if i % gopSize == 0 {
            // I-Frame
            let qtI = QuantizationTable(baseStep: max(1, Int(qt.step)))
            let bytes = try await encodeSpatialLayers(pd: curr, maxbitrate: maxbitrate, qt: qtI, zeroThreshold: zeroThreshold)
            
            out.append(contentsOf: [0x56, 0x45, 0x56, 0x49]) // 'VEVI'
            appendUInt32BE(&out, UInt32(bytes.count))
            out.append(contentsOf: bytes)
            
            let img16 = try await decodeSpatialLayers(r: bytes, maxLayer: 2)
            prevReconstructed = PlaneData420(img16: img16)
        } else {
            // P-Frame
            guard let prev = prevReconstructed else { continue }
            
            // GMC Estimate
            let gmv = estimateGMV(curr: curr, prev: prev)
            
            // Predict
            // In GMC, estimateGMV_old returns how much curr shifted from prev.
            // If curr shifted by (dx, dy) compared to prev, then predicted = prev shifted by (dx, dy).
            let predictedPlane = await shiftPlane(prev, dx: gmv.dx, dy: gmv.dy)
            
            // Residual
            let residual = await subtractPlanes(curr: curr, predicted: predictedPlane)
            
            // P-Frame uses 4x quantization step (similar to H0/H1 in the old architecture)
            let qtP = QuantizationTable(baseStep: max(1, Int(qt.step) * 4))
            let bytes = try await encodeSpatialLayers(pd: residual, maxbitrate: maxbitrate, qt: qtP, zeroThreshold: zeroThreshold)
            
            out.append(contentsOf: [0x56, 0x45, 0x56, 0x50]) // 'VEVP'
            appendUInt16BE(&out, UInt16(bitPattern: Int16(gmv.dx)))
            appendUInt16BE(&out, UInt16(bitPattern: Int16(gmv.dy)))
            appendUInt32BE(&out, UInt32(bytes.count))
            out.append(contentsOf: bytes)
            
            let img16 = try await decodeSpatialLayers(r: bytes, maxLayer: 2)
            let reconstructedResidual = PlaneData420(img16: img16)
            prevReconstructed = await addPlanes(residual: reconstructedResidual, predicted: predictedPlane)
        }
    }
    
    return out
}
