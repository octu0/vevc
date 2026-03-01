// MARK: - Encode

private let kLL: UInt8 = 4
private let kHL: UInt8 = 4
private let kLH: UInt8 = 4
private let kHH: UInt8 = 4

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
    rw.flushZeros(k: k)
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
    rw.flushZeros(k: k)
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
func isAllZero(hl: BlockView, lh: BlockView, hh: BlockView, size: Int) -> Bool {
    for y in 0..<size {
        let ptrHL = hl.rowPointer(y: y)
        let ptrLH = lh.rowPointer(y: y)
        let ptrHH = hh.rowPointer(y: y)
        for x in 0..<size {
            if ptrHL[x] != 0 || ptrLH[x] != 0 || ptrHH[x] != 0 {
                return false
            }
        }
    }
    return true
}

@inline(__always)
func isAllZeroBase(ll: BlockView, hl: BlockView, lh: BlockView, hh: BlockView, size: Int) -> Bool {
    for y in 0..<size {
        let ptrLL = ll.rowPointer(y: y)
        let ptrHL = hl.rowPointer(y: y)
        let ptrLH = lh.rowPointer(y: y)
        let ptrHH = hh.rowPointer(y: y)
        for x in 0..<size {
            if ptrLL[x] != 0 || ptrHL[x] != 0 || ptrLH[x] != 0 || ptrHH[x] != 0 {
                return false
            }
        }
    }
    return true
}

@inline(__always)
func transformLayer(bw: inout BitWriter, block: inout Block2D, size: Int, qt: QuantizationTable) throws -> Block2D {
    var sub = block.withView { view in
        return dwt2d(&view, size: size)
    }
    
    quantizeMidSignedMapping(&sub.hl, qt: qt)
    quantizeMidSignedMapping(&sub.lh, qt: qt)
    quantizeHighSignedMapping(&sub.hh, qt: qt)
    
    if isAllZero(hl: sub.hl, lh: sub.lh, hh: sub.hh, size: sub.size) {
        bw.writeBit(1)
    } else {
        bw.writeBit(0)
        RiceWriter.withWriter(&bw) { rw in
            blockEncode(rw: &rw, block: sub.hl, size: sub.size, k: kHL)
            blockEncode(rw: &rw, block: sub.lh, size: sub.size, k: kLH)
            blockEncode(rw: &rw, block: sub.hh, size: sub.size, k: kHH)
        }
    }
    
    var llBlock = Block2D(width: sub.size, height: sub.size)
    llBlock.withView { dest in
        let src = sub.ll
        for y in 0..<sub.size {
            let srcPtr = src.rowPointer(y: y)
            let destPtr = dest.rowPointer(y: y)
            destPtr.update(from: srcPtr, count: sub.size)
        }
    }
    return llBlock
}

@inline(__always)
func transformBase(bw: inout BitWriter, block: inout Block2D, size: Int, qt: QuantizationTable) throws {
    var sub = block.withView { view in
        return dwt2d(&view, size: size)
    }
    
    quantizeLow(&sub.ll, qt: qt)
    quantizeMidSignedMapping(&sub.hl, qt: qt)
    quantizeMidSignedMapping(&sub.lh, qt: qt)
    quantizeHighSignedMapping(&sub.hh, qt: qt)
    
    if isAllZeroBase(ll: sub.ll, hl: sub.hl, lh: sub.lh, hh: sub.hh, size: sub.size) {
        bw.writeBit(1)
    } else {
        bw.writeBit(0)
        RiceWriter.withWriter(&bw) { rw in
            blockEncodeDPCM(rw: &rw, block: sub.ll, size: sub.size, k: kLL)
            blockEncode(rw: &rw, block: sub.hl, size: sub.size, k: kHL)
            blockEncode(rw: &rw, block: sub.lh, size: sub.size, k: kLH)
            blockEncode(rw: &rw, block: sub.hh, size: sub.size, k: kHH)
        }
    }
}

@inline(__always)
func transformLayerFunc(rows: RowFunc, w: Int, h: Int, size: Int, qt: QuantizationTable) throws -> ([UInt8], Block2D) {
    var block = Block2D(width: size, height: size)
    block.withView { view in
        for i in 0..<size {
            let row = rows(w, (h + i), size)
            view.setRow(offsetY: i, row: row)
        }
    }
    
    var bw = BitWriter()
    let ll = try transformLayer(bw: &bw, block: &block, size: size, qt: qt)
    bw.flush()
    return (bw.data, ll)
}

@inline(__always)
func transformBaseFunc(rows: RowFunc, w: Int, h: Int, size: Int, qt: QuantizationTable) throws -> [UInt8] {
    var block = Block2D(width: size, height: size)
    block.withView { view in
        for i in 0..<size {
            let row = rows(w, (h + i), size)
            view.setRow(offsetY: i, row: row)
        }
    }
    
    var bw = BitWriter()
    try transformBase(bw: &bw, block: &block, size: size, qt: qt)
    bw.flush()
    return bw.data
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
    
    if isAllZeroBase(ll: sub.ll, hl: sub.hl, lh: sub.lh, hh: sub.hh, size: sub.size) {
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
func applyTemporal(planes: [PlaneData420]) async -> (PlaneData420, PlaneData420, PlaneData420, PlaneData420) {
    let pd0 = planes[0], pd1 = planes[1], pd2 = planes[2], pd3 = planes[3]
    let dx = pd0.width, dy = pd0.height
    
    func transform(p0: [Int16], p1: [Int16], p2: [Int16], p3: [Int16]) -> ([Int16], [Int16], [Int16], [Int16]) {
        let count = p0.count

        // Temporal Dead-zone / Noise reduction
        var mp1 = p1
        var mp2 = p2
        var mp3 = p3
        for i in 0..<count {
            if abs(mp1[i] - p0[i]) <= 5 {
                mp1[i] = p0[i]
            }
            if abs(mp2[i] - mp1[i]) <= 5 {
                mp2[i] = mp1[i]
            }
            if abs(mp3[i] - mp2[i]) <= 5 {
                mp3[i] = mp2[i]
            }
        }

        var ll = [Int16](repeating: 0, count: count)
        var lh = [Int16](repeating: 0, count: count)
        var h0 = [Int16](repeating: 0, count: count)
        var h1 = [Int16](repeating: 0, count: count)
        var t0 = [Int16](repeating: 0, count: count)
        var t1 = [Int16](repeating: 0, count: count)
        
        p0.withUnsafeBufferPointer { ptr0 in
        mp1.withUnsafeBufferPointer { ptr1 in
        mp2.withUnsafeBufferPointer { ptr2 in
        mp3.withUnsafeBufferPointer { ptr3 in
            temporalDWT(
                f0: ptr0.baseAddress!, f1: ptr1.baseAddress!, f2: ptr2.baseAddress!, f3: ptr3.baseAddress!,
                count: count,
                outLL: &ll, outLH: &lh, outH0: &h0, outH1: &h1,
                tempL0: &t0, tempL1: &t1
            )
        }}}}
        return (ll, lh, h0, h1)
    }
    
    // Y, Cb, Cr planes in parallel
    async let yResult = { transform(p0: pd0.y, p1: pd1.y, p2: pd2.y, p3: pd3.y) }()
    async let cbResult = { transform(p0: pd0.cb, p1: pd1.cb, p2: pd2.cb, p3: pd3.cb) }()
    async let crResult = { transform(p0: pd0.cr, p1: pd1.cr, p2: pd2.cr, p3: pd3.cr) }()
    
    let (y, cb, cr) = await (yResult, cbResult, crResult)
    
    return (
        PlaneData420(width: dx, height: dy, y: y.0, cb: cb.0, cr: cr.0),
        PlaneData420(width: dx, height: dy, y: y.1, cb: cb.1, cr: cr.1),
        PlaneData420(width: dx, height: dy, y: y.2, cb: cb.2, cr: cr.2),
        PlaneData420(width: dx, height: dy, y: y.3, cb: cb.3, cr: cr.3)
    )
}

@inline(__always)
func shiftPlane(_ plane: PlaneData420, dx: Int, dy: Int) -> PlaneData420 {
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
    
    let yTask = shift(data: plane.y, w: plane.width, h: plane.height, sX: dx, sY: dy)
    let cbTask = shift(data: plane.cb, w: (plane.width + 1) / 2, h: (plane.height + 1) / 2, sX: dx / 2, sY: dy / 2)
    let crTask = shift(data: plane.cr, w: (plane.width + 1) / 2, h: (plane.height + 1) / 2, sX: dx / 2, sY: dy / 2)
    
    return PlaneData420(width: plane.width, height: plane.height, y: yTask, cb: cbTask, cr: crTask)
}



public func encode(images: [YCbCrImage], maxbitrate: Int) async throws -> [UInt8] {
    if images.isEmpty { return [] }
    let gopSize = 4
    
    let qt = estimateQuantization(img: images[0], targetBits: maxbitrate)
    
    var out: [UInt8] = []
    
    for i in stride(from: 0, to: images.count, by: gopSize) {        
        let endIndex = min(i + gopSize, images.count)
        let chunkImages = Array(images[i..<endIndex])
        
        var chunk4 = chunkImages
        while chunk4.count % 4 != 0 {
            chunk4.append(chunkImages.last!)
        }
        
        let planes = toPlaneData420(images: chunk4)
        
        let blockSize = 16
        
        // HBMA Estimate block-level MVs (Test fallback)
        let gmv1 = estimateGMV_old(curr: planes[1], prev: planes[0])
        let gmv2 = estimateGMV_old(curr: planes[2], prev: planes[0])
        let gmv3 = estimateGMV_old(curr: planes[3], prev: planes[0])
        
        // Build predicted planes using block-level MVs
        let p1_v = shiftPlane(planes[1], dx: -gmv1.dx, dy: -gmv1.dy)
        let p2_v = shiftPlane(planes[2], dx: -gmv2.dx, dy: -gmv2.dy)
        let p3_v = shiftPlane(planes[3], dx: -gmv3.dx, dy: -gmv3.dy)
        
        let (ll, lh, h0, h1) = await applyTemporal(planes: [planes[0], p1_v, p2_v, p3_v])
        
        let qtLL = QuantizationTable(baseStep: max(1, Int(qt.step)))
        let qtLH = QuantizationTable(baseStep: Int(qt.step) * 2)
        let qtH0 = QuantizationTable(baseStep: Int(qt.step) * 4)
        let qtH1 = QuantizationTable(baseStep: Int(qt.step) * 4)
        
        // Encode 4 temporal bands in SERIAL to avoid nested parallelism overhead on M4 Max
        let llBytes = try await encodeSpatialLayers(pd: ll, maxbitrate: maxbitrate, qt: qtLL)
        let lhBytes = try await encodeSpatialLayers(pd: lh, maxbitrate: maxbitrate, qt: qtLH)
        let h0Bytes = try await encodeSpatialLayers(pd: h0, maxbitrate: maxbitrate, qt: qtH0)
        let h1Bytes = try await encodeSpatialLayers(pd: h1, maxbitrate: maxbitrate, qt: qtH1)
        
        out.append(contentsOf: [0x56, 0x45, 0x4C, UInt8(gopSize)]) // 'VEL' + GOP size
        
        // Write Block Size
        out.append(UInt8(blockSize))
        
        // Count should be the same for all MV arrays since they use the same block size
        let mvCount = 1
        appendUInt16BE(&out, UInt16(mvCount))
        
        // Write all MVs (delta coding can be introduced later if size becomes an issue, but standard binary struct for now)
        for _ in 0..<mvCount {
            appendUInt16BE(&out, UInt16(bitPattern: Int16(gmv1.dx)))
            appendUInt16BE(&out, UInt16(bitPattern: Int16(gmv1.dy)))
            appendUInt16BE(&out, UInt16(bitPattern: Int16(gmv2.dx)))
            appendUInt16BE(&out, UInt16(bitPattern: Int16(gmv2.dy)))
            appendUInt16BE(&out, UInt16(bitPattern: Int16(gmv3.dx)))
            appendUInt16BE(&out, UInt16(bitPattern: Int16(gmv3.dy)))
        }
        
        appendUInt32BE(&out, UInt32(llBytes.count))
        out.append(contentsOf: llBytes)
        
        appendUInt32BE(&out, UInt32(lhBytes.count))
        out.append(contentsOf: lhBytes)
        
        appendUInt32BE(&out, UInt32(h0Bytes.count))
        out.append(contentsOf: h0Bytes)
        
        appendUInt32BE(&out, UInt32(h1Bytes.count))
        out.append(contentsOf: h1Bytes)
    }
    
    return out
}
