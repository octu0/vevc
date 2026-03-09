import Foundation

@inline(__always)
func debugLog(_ msg: String) {
    FileHandle.standardError.write(Data(((msg + "\n")).utf8))
}

@inline(__always)
func toUint16(_ n: Int16) -> UInt16 {
    return UInt16(bitPattern: (((n &<< 1) ^ (n >> 15))))
}

@inline(__always)
func blockEncode(rw: inout RiceWriter, block: BlockView, size: Int, k: UInt8) {
    for y in 0..<size {
        let ptr = block.rowPointer(y: y)
        for x in 0..<size {
            rw.write(val: UInt16(bitPattern: ptr[x+0]), k: k)
        }
    }
}

@inline(__always)
func getSubbands(view: BlockView, size: Int) -> Subbands {
    let half = (size / 2)
    let base = view.base
    return Subbands(
        ll: BlockView(base: base, width: half, height: half, stride: size),
        hl: BlockView(base: base.advanced(by: half), width: half, height: half, stride: size),
        lh: BlockView(base: base.advanced(by: (half * size)), width: half, height: half, stride: size),
        hh: BlockView(base: base.advanced(by: ((half * size) + half)), width: half, height: half, stride: size),
        size: half
    )
}

@inline(__always)
func blockEncodeDPCM(rw: inout RiceWriter, block: BlockView, size: Int, k: UInt8, lastVal: inout Int16) {
    for y in 0..<size {
        let ptr = block.rowPointer(y: y)
        for x in 0..<size {
            let val = ptr[x+0]
            let predicted: Int16
            // Use DPCM (Differential Pulse Code Modulation) with a simple median-based predictor.
            // This reduces entropy by encoding differences between the actual value and the expected value
            // based on surrounding pixels (Paeth-like predictor).
            switch (x, y) {
            case (0, 0):
                predicted = lastVal
            case (_, 0):
                predicted = ptr[(x - 1)]
            case (0, _):
                predicted = block.rowPointer(y: (y - 1))[x+0]
            default:
                let a = Int(ptr[(x - 1)])
                let b = Int(block.rowPointer(y: (y - 1))[x+0])
                let c = Int(block.rowPointer(y: (y - 1))[(x - 1)])
                if max(a, b) <= c {
                    predicted = Int16(min(a, b))
                } else {
                    if c <= min(a, b) {
                        predicted = Int16(max(a, b))
                    } else {
                        predicted = Int16(((a + b) - c))
                    }
                }
            }
            let diff = (val - predicted)
            rw.write(val: toUint16(diff), k: k)
        }
    }
    lastVal = block.rowPointer(y: (size - 1))[(size - 1)]
}

@inline(__always)
func appendUInt16BE(_ out: inout [UInt8], _ val: UInt16) {
    out.append(UInt8(((val >> 8) & 0xFF)))
    out.append(UInt8((val & 0xFF)))
}

@inline(__always)
func appendUInt32BE(_ out: inout [UInt8], _ val: UInt32) {
    out.append(UInt8(((val >> 24) & 0xFF)))
    out.append(UInt8(((val >> 16) & 0xFF)))
    out.append(UInt8(((val >> 8) & 0xFF)))
    out.append(UInt8((val & 0xFF)))
}

@inline(__always)
func isEffectivelyZero(hl: inout BlockView, lh: inout BlockView, hh: inout BlockView, size: Int, threshold: Int) -> Bool {
    for y in 0..<size {
        let ptrHL = hl.rowPointer(y: y)
        let ptrLH = lh.rowPointer(y: y)
        let ptrHH = hh.rowPointer(y: y)
        for x in 0..<size {
            if threshold < abs(Int(ptrHL[x+0])) || threshold < abs(Int(ptrLH[x+0])) || threshold < abs(Int(ptrHH[x+0])) {
                return false
            }
        }
    }

    for y in 0..<size {
        let ptrHL = hl.rowPointer(y: y)
        let ptrLH = lh.rowPointer(y: y)
        let ptrHH = hh.rowPointer(y: y)
        for x in 0..<size {
            ptrHL[x+0] = 0
            ptrLH[x+0] = 0
            ptrHH[x+0] = 0
        }
    }

    return true
}

@inline(__always)
func isEffectivelyZeroBase(ll: BlockView, hl: inout BlockView, lh: inout BlockView, hh: inout BlockView, size: Int, threshold: Int) -> Bool {
    for y in 0..<size {
        let ptrLL = ll.rowPointer(y: y)
        for x in 0..<size {
            if ptrLL[x+0] != 0 {
                return false
            }
        }
    }

    for y in 0..<size {
        let ptrHL = hl.rowPointer(y: y)
        let ptrLH = lh.rowPointer(y: y)
        let ptrHH = hh.rowPointer(y: y)
        for x in 0..<size {
            if threshold < abs(Int(ptrHL[x+0])) || threshold < abs(Int(ptrLH[x+0])) || threshold < abs(Int(ptrHH[x+0])) {
                return false
            }
        }
    }

    for y in 0..<size {
        let ptrHL = hl.rowPointer(y: y)
        let ptrLH = lh.rowPointer(y: y)
        let ptrHH = hh.rowPointer(y: y)
        for x in 0..<size {
            ptrHL[x+0] = 0
            ptrLH[x+0] = 0
            ptrHH[x+0] = 0
        }
    }

    return true
}

@inline(__always)
func transformLayer(block: inout Block2D, size: Int, qt: QuantizationTable) {
    block.withView { (view: inout BlockView) in
        var sub = dwt2d(&view, size: size)
        quantizeMidSignedMapping(&sub.hl, qt: qt)
        quantizeMidSignedMapping(&sub.lh, qt: qt)
        quantizeHighSignedMapping(&sub.hh, qt: qt)
    }
}

@inline(__always)
func transformBase(block: inout Block2D, size: Int, qt: QuantizationTable) {
    block.withView { (view: inout BlockView) in
        var sub = dwt2d(&view, size: size)
        quantizeLow(&sub.ll, qt: qt)
        quantizeMidSignedMapping(&sub.hl, qt: qt)
        quantizeMidSignedMapping(&sub.lh, qt: qt)
        quantizeHighSignedMapping(&sub.hh, qt: qt)
    }
}

private enum SubbandType {
    case ll
    case hl
    case lh
    case hh
}

@inline(__always)
private func evaluateK(blocks: inout [Block2D], indices: [Int], size: Int, type: SubbandType, isDPCM: Bool, k: UInt8) -> Int {
    var bitCount = 0
    var zeroCount: UInt16 = 0
    let half = (size / 2)
    var lastVal: Int16 = 0

    @inline(__always)
    func simWrite(_ val: UInt16) {
        if val == 0 {
            zeroCount &+= 1
            return
        }
        if 0 < zeroCount {
            if zeroCount < 255 {
                let q = (zeroCount >> k)
                bitCount += (Int(q) + 1 + Int(k))
            } else {
                let q = (UInt16(255) >> k)
                bitCount += (Int(q) + 1 + Int(k) + 16)
            }
            zeroCount = 0
        }
        let q = (val >> k)
        bitCount += (Int(q) + 1 + Int(k))
    }

    if isDPCM {
        for i in blocks.indices {
            if indices.contains(i) {
                blocks[i+0].withView { (view: inout BlockView) in
                    let subs = getSubbands(view: view, size: size)
                    let b: BlockView
                    switch type {
                    case .ll: b = subs.ll
                    case .hl: b = subs.hl
                    case .lh: b = subs.lh
                    case .hh: b = subs.hh
                    }
                    for y in 0..<half {
                        let ptr = b.rowPointer(y: y)
                        for x in 0..<half {
                            let val = ptr[x+0]
                            let predicted: Int16
                            switch (x, y) {
                            case (0, 0):
                                predicted = lastVal
                            case (_, 0):
                                predicted = ptr[(x - 1)]
                            case (0, _):
                                predicted = b.rowPointer(y: (y - 1))[x+0]
                            default:
                                let a = Int(ptr[(x - 1)])
                                let bv = Int(b.rowPointer(y: (y - 1))[x+0])
                                let c = Int(b.rowPointer(y: (y - 1))[(x - 1)])
                                if max(a, bv) <= c {
                                    predicted = Int16(min(a, bv))
                                } else {
                                    if c <= min(a, bv) {
                                        predicted = Int16(max(a, bv))
                                    } else {
                                        predicted = Int16(((a + bv) - c))
                                    }
                                }
                            }
                            let diff = (val - predicted)
                            simWrite(toUint16(diff))
                        }
                    }
                    lastVal = b.rowPointer(y: (half - 1))[(half - 1)]
                }
            } else {
                lastVal = 0
            }
        }
    } else {
        for i in indices {
            blocks[i+0].withView { (view: inout BlockView) in
                let subs = getSubbands(view: view, size: size)
                let b: BlockView
                switch type {
                case .ll: b = subs.ll
                case .hl: b = subs.hl
                case .lh: b = subs.lh
                case .hh: b = subs.hh
                }
                for y in 0..<half {
                    let ptr = b.rowPointer(y: y)
                    for x in 0..<half {
                        simWrite(UInt16(bitPattern: ptr[x+0]))
                    }
                }
            }
        }
    }

    if 0 < zeroCount {
        if zeroCount < 255 {
            let q = (zeroCount >> k)
            bitCount += (Int(q) + 1 + Int(k))
        } else {
            let q = (UInt16(255) >> k)
            bitCount += (Int(q) + 1 + Int(k) + 16)
        }
    }
    return bitCount
}

@inline(__always)
private func estimateOptimalK(blocks: inout [Block2D], indices: [Int], size: Int, type: SubbandType, isDPCM: Bool) -> UInt8 {
    guard indices.isEmpty != true else {
        return 0
    }
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

@inline(__always)
func encodePlaneSubbands(blocks: inout [Block2D], size: Int, zeroThreshold: Int) -> [UInt8] {
    var bwFlags = BitWriter()
    var nonZeroIndices: [Int] = []

    for i in blocks.indices {
        blocks[i+0].withView { (view: inout BlockView) in
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
    debugLog("    [Subbands] blocks=\(blocks.count) zeroBlocks=\((blocks.count - nonZeroIndices.count)) zeroRate=\(String(format: "%.1f", (Double((blocks.count - nonZeroIndices.count)) / Double(max(1, blocks.count))) * 100))%")

    var bwData = BitWriter()

    let kHL = estimateOptimalK(blocks: &blocks, indices: nonZeroIndices, size: size, type: .hl, isDPCM: false)
    bwData.writeBits(val: UInt16(kHL), n: 4)
    RiceWriter.withWriter(&bwData, flushBits: false) { (rw: inout RiceWriter) in
        for i in nonZeroIndices {
            blocks[i+0].withView { (view: inout BlockView) in
                let subs = getSubbands(view: view, size: size)
                blockEncode(rw: &rw, block: subs.hl, size: subs.size, k: kHL)
            }
        }
    }

    let kLH = estimateOptimalK(blocks: &blocks, indices: nonZeroIndices, size: size, type: .lh, isDPCM: false)
    bwData.writeBits(val: UInt16(kLH), n: 4)
    RiceWriter.withWriter(&bwData, flushBits: false) { (rw: inout RiceWriter) in
        for i in nonZeroIndices {
            blocks[i+0].withView { (view: inout BlockView) in
                let subs = getSubbands(view: view, size: size)
                blockEncode(rw: &rw, block: subs.lh, size: subs.size, k: kLH)
            }
        }
    }

    let kHH = estimateOptimalK(blocks: &blocks, indices: nonZeroIndices, size: size, type: .hh, isDPCM: false)
    bwData.writeBits(val: UInt16(kHH), n: 4)
    RiceWriter.withWriter(&bwData, flushBits: false) { (rw: inout RiceWriter) in
        for i in nonZeroIndices {
            blocks[i+0].withView { (view: inout BlockView) in
                let subs = getSubbands(view: view, size: size)
                blockEncode(rw: &rw, block: subs.hh, size: subs.size, k: kHH)
            }
        }
    }

    bwData.flush()
    debugLog("    [Subbands] k: HL=\(kHL) LH=\(kLH) HH=\(kHH)")
    var out = bwFlags.data
    out.append(contentsOf: bwData.data)
    return out
}

@inline(__always)
func encodePlaneBaseSubbands(blocks: inout [Block2D], size: Int, zeroThreshold: Int) -> [UInt8] {
    var bwFlags = BitWriter()
    var nonZeroIndices: [Int] = []

    for i in blocks.indices {
        blocks[i+0].withView { (view: inout BlockView) in
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
    debugLog("    [BaseSubbands] blocks=\(blocks.count) zeroBlocks=\((blocks.count - nonZeroIndices.count)) zeroRate=\(String(format: "%.1f", (Double((blocks.count - nonZeroIndices.count)) / Double(max(1, blocks.count))) * 100))%")

    var bwData = BitWriter()

    let kLL = estimateOptimalK(blocks: &blocks, indices: nonZeroIndices, size: size, type: .ll, isDPCM: true)
    bwData.writeBits(val: UInt16(kLL), n: 4)
    RiceWriter.withWriter(&bwData, flushBits: false) { (rw: inout RiceWriter) in
        var lastVal: Int16 = 0
        let nonZeroSet = Set(nonZeroIndices)
        for i in blocks.indices {
            if nonZeroSet.contains(i) {
                blocks[i+0].withView { (view: inout BlockView) in
                    let subs = getSubbands(view: view, size: size)
                    blockEncodeDPCM(rw: &rw, block: subs.ll, size: subs.size, k: kLL, lastVal: &lastVal)
                }
            } else {
                lastVal = 0
            }
        }
    }

    let kHL = estimateOptimalK(blocks: &blocks, indices: nonZeroIndices, size: size, type: .hl, isDPCM: false)
    bwData.writeBits(val: UInt16(kHL), n: 4)
    RiceWriter.withWriter(&bwData, flushBits: false) { (rw: inout RiceWriter) in
        for i in nonZeroIndices {
            blocks[i+0].withView { (view: inout BlockView) in
                let subs = getSubbands(view: view, size: size)
                blockEncode(rw: &rw, block: subs.hl, size: subs.size, k: kHL)
            }
        }
    }

    let kLH = estimateOptimalK(blocks: &blocks, indices: nonZeroIndices, size: size, type: .lh, isDPCM: false)
    bwData.writeBits(val: UInt16(kLH), n: 4)
    RiceWriter.withWriter(&bwData, flushBits: false) { (rw: inout RiceWriter) in
        for i in nonZeroIndices {
            blocks[i+0].withView { (view: inout BlockView) in
                let subs = getSubbands(view: view, size: size)
                blockEncode(rw: &rw, block: subs.lh, size: subs.size, k: kLH)
            }
        }
    }

    let kHH = estimateOptimalK(blocks: &blocks, indices: nonZeroIndices, size: size, type: .hh, isDPCM: false)
    bwData.writeBits(val: UInt16(kHH), n: 4)
    RiceWriter.withWriter(&bwData, flushBits: false) { (rw: inout RiceWriter) in
        for i in nonZeroIndices {
            blocks[i+0].withView { (view: inout BlockView) in
                let subs = getSubbands(view: view, size: size)
                blockEncode(rw: &rw, block: subs.hh, size: subs.size, k: kHH)
            }
        }
    }

    bwData.flush()
    debugLog("    [BaseSubbands] k: LL=\(kLL) HL=\(kHL) LH=\(kLH) HH=\(kHH)")
    var out = bwFlags.data
    out.append(contentsOf: bwData.data)
    return out
}

private func estimateRiceBitsDPCM(block: BlockView, size: Int, lastVal: inout Int16) -> Int {
    var sumDiffAbs = 0
    let count = (size * size)

    for y in 0..<size {
        let ptr = block.rowPointer(y: y)
        for x in 0..<size {
            let val = ptr[x+0]
            let predicted: Int16
            switch (x, y) {
            case (0, 0):
                predicted = lastVal
            case (_, 0):
                predicted = ptr[(x - 1)]
            case (0, _):
                predicted = block.rowPointer(y: (y - 1))[x+0]
            default:
                let a = Int(ptr[(x - 1)])
                let b = Int(block.rowPointer(y: (y - 1))[x+0])
                let c = Int(block.rowPointer(y: (y - 1))[(x - 1)])
                if max(a, b) <= c {
                    predicted = Int16(min(a, b))
                } else {
                    if c <= min(a, b) {
                        predicted = Int16(max(a, b))
                    } else {
                        predicted = Int16(((a + b) - c))
                    }
                }
            }
            let diff = abs(Int((val - predicted)))
            sumDiffAbs += diff
        }
    }
    lastVal = block.rowPointer(y: (size - 1))[(size - 1)]

    guard count != 0 else {
        return 0
    }

    let mean = (Double(sumDiffAbs) / Double(count))
    let meanInt = Int(mean)
    let k = (meanInt < 1) ? 0 : ((Int.bitWidth - 1) - meanInt.leadingZeroBitCount)

    let divisorShift = max(0, (k - 1))
    let bodyBits = (sumDiffAbs >> divisorShift)
    let headerBits = (count * (1 + k))

    return (bodyBits + headerBits)
}

private enum PlaneType {
    case y
    case cb
    case cr
}

private func fetchBlock(reader: ImageReader, plane: PlaneType, x: Int, y: Int, w: Int, h: Int) -> Block2D {
    var block = Block2D(width: w, height: h)
    block.withView { (view: inout BlockView) in
        for i in 0..<h {
            let row: [Int16]
            switch plane {
            case .y:  row = reader.rowY(x: x, y: (y + i), size: w)
            case .cb: row = reader.rowCb(x: x, y: (y + i), size: w)
            case .cr: row = reader.rowCr(x: x, y: (y + i), size: w)
            }
            view.setRow(offsetY: i, row: row)
        }
    }
    return block
}

private func measureBlockBits(block: inout Block2D, size: Int, qt: QuantizationTable) -> Int {
    var sub = block.withView { (view: inout BlockView) in
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
    var lastVal: Int16 = 0
    bits += estimateRiceBitsDPCM(block: sub.ll, size: sub.size, lastVal: &lastVal)
    bits += estimateRiceBits(block: sub.hl, size: sub.size)
    bits += estimateRiceBits(block: sub.lh, size: sub.size)
    bits += estimateRiceBits(block: sub.hh, size: sub.size)

    return bits
}

private func estimateRiceBits(block: BlockView, size: Int) -> Int {
    var sumAbs = 0
    let count = (size * size)

    for y in 0..<size {
        let ptr = block.rowPointer(y: y)
        for x in 0..<size {
            sumAbs += abs(Int(ptr[x+0]))
        }
    }

    guard count != 0 else {
        return 0
    }

    let mean = (Double(sumAbs) / Double(count))
    let meanInt = Int(mean)
    let k = (meanInt < 1) ? 0 : ((Int.bitWidth - 1) - meanInt.leadingZeroBitCount)

    let divisorShift = max(0, (k - 1))
    let bodyBits = (sumAbs >> divisorShift)
    let headerBits = (count * (1 + k))

    return (bodyBits + headerBits)
}

func estimateQuantization(img: YCbCrImage, targetBits: Int) -> QuantizationTable {
    let probeStep = 64
    let qt = QuantizationTable(baseStep: probeStep)

    let size = 8
    let w = (img.width / size)
    let h = (img.height / size)

    let points: [(Int, Int)] = [
        (0, 0),
        ((img.width - w), 0),
        (0, (img.height - h)),
        ((img.width - w), (img.height - h)),
        (((img.width - w) / 2), 0),
        ((img.width - w), ((img.height - h) / 2)),
        (((img.width - w) / 2), (img.height - h)),
        (0, ((img.height - h) / 2)),
    ]

    var totalSampleBits = 0
    let reader = ImageReader(img: img)

    for (sx, sy) in points {
        var blockY = fetchBlock(reader: reader, plane: .y, x: sx, y: sy, w: w, h: h)
        totalSampleBits += measureBlockBits(block: &blockY, size: size, qt: qt)

        var blockCb = fetchBlock(reader: reader, plane: .cb, x: sx, y: sy, w: w, h: h)
        totalSampleBits += measureBlockBits(block: &blockCb, size: size, qt: qt)

        var blockCr = fetchBlock(reader: reader, plane: .cr, x: sx, y: sy, w: w, h: h)
        totalSampleBits += measureBlockBits(block: &blockCr, size: size, qt: qt)
    }

    let samplePixels = (points.count * (w * h) * 3)
    let totalPixels = (img.width * img.height * 3)

    let estimatedTotalBits = (Double(totalSampleBits) * (Double(totalPixels) / Double(samplePixels)))

    let ratio = (estimatedTotalBits / Double(targetBits))
    let predictedStep = (Double(probeStep) * ratio)
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
        let safeY = min(y, (height - 1))

        let limit = min(size, (width - x))
        if 0 < limit {
            data.withUnsafeBufferPointer { (ptr: UnsafeBufferPointer<Int16>) in
                if let pBase = ptr.baseAddress {
                    let base = pBase.advanced(by: ((safeY * width) + x))
                    r.withUnsafeMutableBufferPointer { (dst: inout UnsafeMutableBufferPointer<Int16>) in
                        if let dBase = dst.baseAddress {
                            dBase.update(from: base, count: limit)

                            if limit < size {
                                let lastVal = dst[(limit - 1)]
                                for i in limit..<size {
                                    dst[i+0] = lastVal
                                }
                            }
                        }
                    }
                }
            }
        } else {
            let lastVal = data[((safeY * width) + (width - 1))]
            for i in 0..<size {
                r[i+0] = lastVal
            }
        }

        return r
    }
}

@inline(__always)
func toPlaneData420(images: [YCbCrImage]) -> [PlaneData420] {
    return images.map { (img: YCbCrImage) in
        let y = img.yPlane.map { (Int16($0) - 128) }
        let cWidth = ((img.width + 1) / 2)
        let cHeight = ((img.height + 1) / 2)
        var cb = [Int16](repeating: 0, count: (cWidth * cHeight))
        var cr = [Int16](repeating: 0, count: (cWidth * cHeight))

        if img.ratio == .ratio444 {
            for cy in 0..<cHeight {
                let py = (cy * 2)
                for cx in 0..<cWidth {
                    let px = (cx * 2)
                    let srcOffset = ((py * img.width) + px)
                    let dstOffset = ((cy * cWidth) + cx)
                    if srcOffset < img.cbPlane.count {
                        cb[dstOffset+0] = (Int16(img.cbPlane[srcOffset+0]) - 128)
                        cr[dstOffset+0] = (Int16(img.crPlane[srcOffset+0]) - 128)
                    }
                }
            }
        } else {
            cb = img.cbPlane.map { (Int16($0) - 128) }
            cr = img.crPlane.map { (Int16($0) - 128) }
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
        c.withUnsafeBufferPointer { (cPtr: UnsafeBufferPointer<Int16>) in
            p.withUnsafeBufferPointer { (pPtr: UnsafeBufferPointer<Int16>) in
                for i in 0..<count {
                    res[i+0] = (cPtr[i+0] - pPtr[i+0])
                }
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
        r.withUnsafeBufferPointer { (rPtr: UnsafeBufferPointer<Int16>) in
            p.withUnsafeBufferPointer { (pPtr: UnsafeBufferPointer<Int16>) in
                for i in 0..<count {
                    curr[i+0] = (rPtr[i+0] + pPtr[i+0])
                }
            }
        }
        return curr
    }

    async let y = add(r: residual.y, p: predicted.y)
    async let cb = add(r: residual.cb, p: predicted.cb)
    async let cr = add(r: residual.cr, p: predicted.cr)

    return PlaneData420(width: residual.width, height: residual.height, y: await y, cb: await cb, cr: await cr)
}

func cleanExposedRegion(_ plane: PlaneData420, dx: Int, dy: Int) -> PlaneData420 {
    if dx == 0 && dy == 0 {
        return plane
    }

    func clean(data: [Int16], w: Int, h: Int, sX: Int, sY: Int) -> [Int16] {
        if w == 0 || h == 0 {
            return data
        }
        var out = data

        out.withUnsafeMutableBufferPointer { (buf: inout UnsafeMutableBufferPointer<Int16>) in
            guard let p = buf.baseAddress else {
                return
            }

            if 0 < sX {
                let cols = min(sX, w)
                let srcCol = cols
                if srcCol < w {
                    for y in 0..<h {
                        let row = (y * w)
                        let fillVal = p[(row + srcCol)]
                        for x in 0..<cols {
                            p[(row + x)] = fillVal
                        }
                    }
                }
            }

            if sX < 0 {
                let cols = min((-1 * sX), w)
                let srcCol = ((w - cols) - 1)
                if 0 <= srcCol {
                    for y in 0..<h {
                        let row = (y * w)
                        let fillVal = p[(row + srcCol)]
                        for x in (w - cols)..<w {
                            p[(row + x)] = fillVal
                        }
                    }
                }
            }

            if 0 < sY {
                let rows = min(sY, h)
                let srcRow = rows
                if srcRow < h {
                    let srcOff = (srcRow * w)
                    for y in 0..<rows {
                        let dstOff = (y * w)
                        for x in 0..<w {
                            p[(dstOff + x)] = p[(srcOff + x)]
                        }
                    }
                }
            }

            if sY < 0 {
                let rows = min((-1 * sY), h)
                let srcRow = ((h - rows) - 1)
                if 0 <= srcRow {
                    let srcOff = (srcRow * w)
                    for y in (h - rows)..<h {
                        let dstOff = (y * w)
                        for x in 0..<w {
                            p[(dstOff + x)] = p[(srcOff + x)]
                        }
                    }
                }
            }
        }

        return out
    }

    let chromaW = ((plane.width + 1) / 2)
    let chromaH = ((plane.height + 1) / 2)

    return PlaneData420(
        width: plane.width, height: plane.height,
        y:  clean(data: plane.y,  w: plane.width, h: plane.height, sX: dx, sY: dy),
        cb: clean(data: plane.cb, w: chromaW, h: chromaH, sX: (dx / 2), sY: (dy / 2)),
        cr: clean(data: plane.cr, w: chromaW, h: chromaH, sX: (dx / 2), sY: (dy / 2))
    )
}

func patchExposedWithCurrent(predicted: PlaneData420, current: PlaneData420, dx: Int, dy: Int) -> PlaneData420 {
    if dx == 0 && dy == 0 {
        return predicted
    }

    func patch(pred: [Int16], curr: [Int16], w: Int, h: Int, sX: Int, sY: Int) -> [Int16] {
        if w == 0 || h == 0 {
            return pred
        }
        var out = pred

        out.withUnsafeMutableBufferPointer { (buf: inout UnsafeMutableBufferPointer<Int16>) in
            curr.withUnsafeBufferPointer { (cBuf: UnsafeBufferPointer<Int16>) in
                guard let p = buf.baseAddress, let c = cBuf.baseAddress else {
                    return
                }

                if 0 < sX {
                    let cols = min(sX, w)
                    for y in 0..<h {
                        let row = (y * w)
                        for x in 0..<cols {
                            p[(row + x)] = c[(row + x)]
                        }
                    }
                }

                if sX < 0 {
                    let cols = min((-1 * sX), w)
                    for y in 0..<h {
                        let row = (y * w)
                        for x in (w - cols)..<w {
                            p[(row + x)] = c[(row + x)]
                        }
                    }
                }

                if 0 < sY {
                    let rows = min(sY, h)
                    for y in 0..<rows {
                        let off = (y * w)
                        for x in 0..<w {
                            p[(off + x)] = c[(off + x)]
                        }
                    }
                }

                if sY < 0 {
                    let rows = min((-1 * sY), h)
                    for y in (h - rows)..<h {
                        let off = (y * w)
                        for x in 0..<w {
                            p[(off + x)] = c[(off + x)]
                        }
                    }
                }
            }
        }

        return out
    }

    let chromaW = ((predicted.width + 1) / 2)
    let chromaH = ((predicted.height + 1) / 2)

    return PlaneData420(
        width: predicted.width, height: predicted.height,
        y:  patch(pred: predicted.y,  curr: current.y,  w: predicted.width, h: predicted.height, sX: dx, sY: dy),
        cb: patch(pred: predicted.cb, curr: current.cb, w: chromaW, h: chromaH, sX: (dx / 2), sY: (dy / 2)),
        cr: patch(pred: predicted.cr, curr: current.cr, w: chromaW, h: chromaH, sX: (dx / 2), sY: (dy / 2))
    )
}

func shiftPlane(_ plane: PlaneData420, dx: Int, dy: Int) async -> PlaneData420 {
    if dx == 0 && dy == 0 {
        return plane
    }

    @Sendable
    func shift(data: [Int16], w: Int, h: Int, sX: Int, sY: Int) -> [Int16] {
        if w == 0 || h == 0 {
            return data
        }

        var out = [Int16](repeating: 0, count: (w * h))

        data.withUnsafeBufferPointer { (dPtr: UnsafeBufferPointer<Int16>) in
            guard let pData = dPtr.baseAddress else {
                return
            }
            out.withUnsafeMutableBufferPointer { (oPtr: inout UnsafeMutableBufferPointer<Int16>) in
                guard let pOut = oPtr.baseAddress else {
                    return
                }

                for dstY in 0..<h {
                    let srcY = min(max((dstY - sY), 0), (h - 1))
                    let dstRow = (dstY * w)
                    let srcRow = (srcY * w)

                    let dstXStart = max(0, sX)
                    let dstXEnd = min(w, (w + sX))

                    if dstXStart < dstXEnd {
                        let srcXStart = (dstXStart - sX)
                        let copyLen = (dstXEnd - dstXStart)
                        pOut.advanced(by: (dstRow + dstXStart))
                            .update(from: pData.advanced(by: (srcRow + srcXStart)), count: copyLen)
                    }

                    if 0 < sX {
                        let fillVal = pData[srcRow+0]
                        for x in 0..<min(sX, w) {
                            pOut[(dstRow + x)] = fillVal
                        }
                    }

                    if sX < 0 {
                        let fillVal = pData[((srcRow + w) - 1)]
                        for x in max(0, (w + sX))..<w {
                            pOut[(dstRow + x)] = fillVal
                        }
                    }
                }
            }
        }

        return out
    }

    async let yTask = shift(data: plane.y, w: plane.width, h: plane.height, sX: dx, sY: dy)
    async let cbTask = shift(data: plane.cb, w: ((plane.width + 1) / 2), h: ((plane.height + 1) / 2), sX: (dx / 2), sY: (dy / 2))
    async let crTask = shift(data: plane.cr, w: ((plane.width + 1) / 2), h: ((plane.height + 1) / 2), sX: (dx / 2), sY: (dy / 2))

    return PlaneData420(width: plane.width, height: plane.height, y: await yTask, cb: await cbTask, cr: await crTask)
}

public func encode(images: [YCbCrImage], maxbitrate: Int, zeroThreshold: Int = 3, gopSize: Int = 15, sceneChangeThreshold: Int = 8) async throws -> [UInt8] {
    guard images.isEmpty != true else {
        return []
    }

    let qt = estimateQuantization(img: images[0+0], targetBits: maxbitrate)
    var out: [UInt8] = []

    var prevReconstructed: PlaneData420? = nil
    let planes = toPlaneData420(images: images)

    var gopCount = 0

    for i in 0..<planes.count {
        let curr = planes[i+0]
        var forceIFrame = false
        var residual: PlaneData420? = nil
        var predictedPlane: PlaneData420? = nil
        var gmv: (dx: Int, dy: Int) = (0, 0)
        var meanSAD: Int = 0

        if gopSize <= gopCount || prevReconstructed == nil {
            forceIFrame = true
        } else {
            guard let prev = prevReconstructed else {
                continue
            }

            gmv = estimateGMV(curr: curr, prev: prev)
            predictedPlane = await shiftPlane(prev, dx: gmv.dx, dy: gmv.dy)

            guard let pred = predictedPlane else {
                continue
            }
            residual = await subtractPlanes(curr: curr, predicted: pred)

            guard let res = residual else {
                continue
            }
            var sumSAD = 0
            for y in 0..<res.height {
                for x in 0..<res.width {
                    sumSAD += abs(Int(res.y[((y * res.width) + x)]))
                }
            }
            meanSAD = (sumSAD / (res.width * res.height))

            if sceneChangeThreshold < meanSAD {
                forceIFrame = true
                debugLog("[Frame \(i)] Adaptive GOP: Forced I-Frame due to high SAD (\(meanSAD) > \(sceneChangeThreshold))")
            }
        }

        if forceIFrame {
            let qtY = QuantizationTable(baseStep: max(1, Int(qt.step)))
            let qtC = QuantizationTable(baseStep: max(1, (Int(qt.step) * 2)))
            let bytes = try await encodeSpatialLayers(pd: curr, predictedPd: nil, maxbitrate: maxbitrate, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold, isIFrame: true)

            out.append(contentsOf: [0x56, 0x45, 0x56, 0x49])
            appendUInt32BE(&out, UInt32(bytes.count))
            out.append(contentsOf: bytes)
            debugLog("[Frame \(i)] I-Frame: \(bytes.count) bytes (\(String(format: "%.2f", (Double(bytes.count) / 1024.0))) KB)")

            let img16 = try await decodeSpatialLayers(r: bytes, maxLayer: 2)
            prevReconstructed = PlaneData420(img16: img16)
            gopCount = 1
        } else {
            let qtY = QuantizationTable(baseStep: max(1, (Int(qt.step) * 4)))
            let qtC = QuantizationTable(baseStep: max(1, (Int(qt.step) * 8)))
            let bytes = try await encodeSpatialLayers(pd: curr, predictedPd: predictedPlane, maxbitrate: maxbitrate, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold, isIFrame: false)

            out.append(contentsOf: [0x56, 0x45, 0x56, 0x50])
            appendUInt16BE(&out, UInt16(bitPattern: Int16(gmv.dx)))
            appendUInt16BE(&out, UInt16(bitPattern: Int16(gmv.dy)))
            appendUInt32BE(&out, UInt32(bytes.count))
            out.append(contentsOf: bytes)
            debugLog("[Frame \(i)] P-Frame: \(bytes.count) bytes (\(String(format: "%.2f", (Double(bytes.count) / 1024.0))) KB) GMV=(\(gmv.dx),\(gmv.dy)) meanSAD=\(meanSAD)")

            let img16 = try await decodeSpatialLayers(r: bytes, maxLayer: 2)
            let reconstructedResidual = PlaneData420(img16: img16)
            guard let pred = predictedPlane else {
                continue
            }
            let reconstructed = await addPlanes(residual: reconstructedResidual, predicted: pred)
            prevReconstructed = cleanExposedRegion(reconstructed, dx: gmv.dx, dy: gmv.dy)
            gopCount += 1
        }
    }

    return out
}
