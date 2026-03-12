// MARK: - Encode

import Foundation

@inline(__always)
func debugLog(_ msg: String) {
    FileHandle.standardError.write(Data((msg + "\n").utf8))
}

@inline(__always)
func toUint16(_ n: Int16) -> UInt16 {
    return UInt16(bitPattern: ((n &<< 1) ^ (n >> 15)))
}

@inline(__always)
func encodeExpGolomb(val: UInt32, encoder: inout CABACEncoder) {
    var q: Int = 0
    var temp = val &+ 1
    while temp > 1 {
        q += 1
        temp >>= 1
    }
    for _ in 0..<q {
        encoder.encodeBypass(binVal: 1)
    }
    encoder.encodeBypass(binVal: 0)
    if q > 0 {
        for i in stride(from: q - 1, through: 0, by: -1) {
            let bit = UInt8(((val &+ 1) >> i) & 1)
            encoder.encodeBypass(binVal: bit)
        }
    }
}

@inline(__always)
func encodeCoeff(val: Int16, encoder: inout CABACEncoder, ctxSig: inout ContextModel, ctxSign: inout ContextModel, ctxMag: inout [ContextModel]) {
    if val == 0 {
        encoder.encodeBin(binVal: 0, ctx: &ctxSig)
        return
    }
    encoder.encodeBin(binVal: 1, ctx: &ctxSig)

    let signBit: UInt8 = (val < 0) ? 1 : 0
    let absVal = UInt32(abs(Int(val)))

    encoder.encodeBin(binVal: signBit, ctx: &ctxSign)

    let magMinus1 = absVal &- 1
    let numBins = min(magMinus1, 7)
    for i in 0..<numBins {
        encoder.encodeBin(binVal: 1, ctx: &ctxMag[Int(i)])
    }
    if magMinus1 < 7 {
        encoder.encodeBin(binVal: 0, ctx: &ctxMag[Int(numBins)])
    } else {
        let rem = magMinus1 &- 7
        encodeExpGolomb(val: rem, encoder: &encoder)
    }
}

@inline(__always)
func blockEncode(encoder: inout CABACEncoder, block: BlockView, size: Int, ctxSig: inout ContextModel, ctxSign: inout ContextModel, ctxMag: inout [ContextModel]) {
    for y in 0..<size {
        let ptr = block.rowPointer(y: y)
        for x in 0..<size {
            encodeCoeff(val: ptr[x], encoder: &encoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag)
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
func blockEncodeDPCM(encoder: inout CABACEncoder, block: BlockView, size: Int, lastVal: inout Int16, ctxSig: inout ContextModel, ctxSign: inout ContextModel, ctxMag: inout [ContextModel]) {
    let ptr0 = block.rowPointer(y: 0)
    
    let diff00 = ptr0[0] - lastVal
    encodeCoeff(val: diff00, encoder: &encoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag)
    
    for x in 1..<size {
        let diff = ptr0[x] - ptr0[x - 1]
        encodeCoeff(val: diff, encoder: &encoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag)
    }
    
    for y in 1..<size {
        let ptr = block.rowPointer(y: y)
        let ptrPrev = block.rowPointer(y: y - 1)
        
        let diffY0 = ptr[0] - ptrPrev[0]
        encodeCoeff(val: diffY0, encoder: &encoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag)
        
        for x in 1..<size {
            let a = Int(ptr[x - 1])
            let b = Int(ptrPrev[x])
            let c = Int(ptrPrev[x - 1])
            let predicted: Int16
            if c >= a && c >= b {
                predicted = Int16(min(a, b))
            } else if c <= a && c <= b {
                predicted = Int16(max(a, b))
            } else {
                predicted = Int16(a + b - c)
            }
            let diff = ptr[x] - predicted
            encodeCoeff(val: diff, encoder: &encoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag)
        }
    }
    lastVal = block.rowPointer(y: size - 1)[size - 1]
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

@inline(__always)
func encodePlaneSubbands(blocks: inout [Block2D], size: Int, zeroThreshold: Int) -> [UInt8] {
    var bwFlags = CABACBitWriter()
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
    debugLog("    [Subbands] blocks=\(blocks.count) zeroBlocks=\(blocks.count - nonZeroIndices.count) zeroRate=\(String(format: "%.1f", Double(blocks.count - nonZeroIndices.count) / Double(max(1, blocks.count)) * 100))%")
    
    var encoder = CABACEncoder()
    var ctxSigHL = ContextModel()
    var ctxSignHL = ContextModel()
    var ctxMagHL = [ContextModel](repeating: ContextModel(), count: 8)
    
    var ctxSigLH = ContextModel()
    var ctxSignLH = ContextModel()
    var ctxMagLH = [ContextModel](repeating: ContextModel(), count: 8)

    var ctxSigHH = ContextModel()
    var ctxSignHH = ContextModel()
    var ctxMagHH = [ContextModel](repeating: ContextModel(), count: 8)

    for i in nonZeroIndices {
        blocks[i].withView { view in
            let subs = getSubbands(view: view, size: size)
            blockEncode(encoder: &encoder, block: subs.hl, size: subs.size, ctxSig: &ctxSigHL, ctxSign: &ctxSignHL, ctxMag: &ctxMagHL)
            blockEncode(encoder: &encoder, block: subs.lh, size: subs.size, ctxSig: &ctxSigLH, ctxSign: &ctxSignLH, ctxMag: &ctxMagLH)
            blockEncode(encoder: &encoder, block: subs.hh, size: subs.size, ctxSig: &ctxSigHH, ctxSign: &ctxSignHH, ctxMag: &ctxMagHH)
        }
    }
    
    encoder.flush()
    var out = bwFlags.data
    out.append(contentsOf: encoder.getData())
    return out
}

@inline(__always)
func encodePlaneBaseSubbands(blocks: inout [Block2D], size: Int, zeroThreshold: Int) -> [UInt8] {
    var bwFlags = CABACBitWriter()
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
    debugLog("    [BaseSubbands] blocks=\(blocks.count) zeroBlocks=\(blocks.count - nonZeroIndices.count) zeroRate=\(String(format: "%.1f", Double(blocks.count - nonZeroIndices.count) / Double(max(1, blocks.count)) * 100))%")
    
    var encoder = CABACEncoder()
    var ctxSigLL = ContextModel()
    var ctxSignLL = ContextModel()
    var ctxMagLL = [ContextModel](repeating: ContextModel(), count: 8)
    
    var ctxSigHL = ContextModel()
    var ctxSignHL = ContextModel()
    var ctxMagHL = [ContextModel](repeating: ContextModel(), count: 8)

    var ctxSigLH = ContextModel()
    var ctxSignLH = ContextModel()
    var ctxMagLH = [ContextModel](repeating: ContextModel(), count: 8)

    var ctxSigHH = ContextModel()
    var ctxSignHH = ContextModel()
    var ctxMagHH = [ContextModel](repeating: ContextModel(), count: 8)
    
    var lastVal: Int16 = 0
    
    var nzCur = 0
    let nzCount = nonZeroIndices.count
    for i in blocks.indices {
        if nzCur < nzCount && nonZeroIndices[nzCur] == i {
            nzCur += 1

            blocks[i].withView { view in
                let subs = getSubbands(view: view, size: size)
                blockEncodeDPCM(encoder: &encoder, block: subs.ll, size: subs.size, lastVal: &lastVal, ctxSig: &ctxSigLL, ctxSign: &ctxSignLL, ctxMag: &ctxMagLL)
                blockEncode(encoder: &encoder, block: subs.hl, size: subs.size, ctxSig: &ctxSigHL, ctxSign: &ctxSignHL, ctxMag: &ctxMagHL)
                blockEncode(encoder: &encoder, block: subs.lh, size: subs.size, ctxSig: &ctxSigLH, ctxSign: &ctxSignLH, ctxMag: &ctxMagLH)
                blockEncode(encoder: &encoder, block: subs.hh, size: subs.size, ctxSig: &ctxSigHH, ctxSign: &ctxSignHH, ctxMag: &ctxMagHH)
            }
        } else {
            lastVal = 0
        }
    }
    
    encoder.flush()
    var out = bwFlags.data
    out.append(contentsOf: encoder.getData())
    return out
}

private func estimateRiceBitsDPCM(block: BlockView, size: Int, lastVal: inout Int16) -> Int {
    if size < 1 { return 0 }

    var sumDiffAbs = 0
    let count = (size * size)
    
    let ptr0 = block.rowPointer(y: 0)
    
    sumDiffAbs += abs(Int(ptr0[0] - lastVal))
    
    for x in 1..<size {
        sumDiffAbs += abs(Int(ptr0[x] - ptr0[x - 1]))
    }
    
    for y in 1..<size {
        let ptr = block.rowPointer(y: y)
        let ptrPrev = block.rowPointer(y: y - 1)
        
        sumDiffAbs += abs(Int(ptr[0] - ptrPrev[0]))
        
        for x in 1..<size {
            let a = Int(ptr[x - 1])
            let b = Int(ptrPrev[x])
            let c = Int(ptrPrev[x - 1])
            let predicted: Int16
            if c >= a && c >= b {
                predicted = Int16(min(a, b))
            } else if c <= a && c <= b {
                predicted = Int16(max(a, b))
            } else {
                predicted = Int16(a + b - c)
            }
            sumDiffAbs += abs(Int(ptr[x] - predicted))
        }
    }
    lastVal = block.rowPointer(y: size - 1)[size - 1]
    let meanInt = sumDiffAbs / count
    let k: Int
    if meanInt < 1 {
        k = 0
    } else {
        k = (Int.bitWidth - 1) - meanInt.leadingZeroBitCount
    }
    
    let divisorShift = max(0, k - 1)
    let bodyBits = sumDiffAbs >> divisorShift
    let headerBits = count * (1 + k)
    
    return bodyBits + headerBits
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
    var lastVal: Int16 = 0
    bits += estimateRiceBitsDPCM(block: sub.ll, size: sub.size, lastVal: &lastVal)
    bits += estimateRiceBits(block: sub.hl, size: sub.size)
    bits += estimateRiceBits(block: sub.lh, size: sub.size)
    bits += estimateRiceBits(block: sub.hh, size: sub.size)
    
    return bits
}

private func estimateRiceBits(block: BlockView, size: Int) -> Int {
    if size < 1 { return 0 }

    var sumAbs = 0
    let count = (size * size)
    
    for y in 0..<size {
        let ptr = block.rowPointer(y: y)
        for x in 0..<size {
            sumAbs += abs(Int(ptr[x]))
        }
    }
    let meanInt = sumAbs / count
    let k: Int
    if meanInt < 1 {
        k = 0
    } else {
        k = (Int.bitWidth - 1) - meanInt.leadingZeroBitCount
    }
    
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
    @inline(__always)
    func fetchBlockY(reader: ImageReader, x: Int, y: Int, w: Int, h: Int) -> Block2D {
        var block = Block2D(width: w, height: h)
        block.withView { view in
            for i in 0..<h {
                view.setRow(offsetY: i, row: reader.rowY(x: x, y: y + i, size: w))
            }
        }
        return block
    }

    @inline(__always)
    func fetchBlockCb(reader: ImageReader, x: Int, y: Int, w: Int, h: Int) -> Block2D {
        var block = Block2D(width: w, height: h)
        block.withView { view in
            for i in 0..<h {
                view.setRow(offsetY: i, row: reader.rowCb(x: x, y: y + i, size: w))
            }
        }
        return block
    }

    @inline(__always)
    func fetchBlockCr(reader: ImageReader, x: Int, y: Int, w: Int, h: Int) -> Block2D {
        var block = Block2D(width: w, height: h)
        block.withView { view in
            for i in 0..<h {
                view.setRow(offsetY: i, row: reader.rowCr(x: x, y: y + i, size: w))
            }
        }
        return block
    }
    
    for (sx, sy) in points {
        var blockY = fetchBlockY(reader: reader, x: sx, y: sy, w: w, h: h)
        totalSampleBits += measureBlockBits(block: &blockY, size: size, qt: qt)
        
        var blockCb = fetchBlockCb(reader: reader, x: sx, y: sy, w: w, h: h)
        totalSampleBits += measureBlockBits(block: &blockCb, size: size, qt: qt)
        
        var blockCr = fetchBlockCr(reader: reader, x: sx, y: sy, w: w, h: h)
        totalSampleBits += measureBlockBits(block: &blockCr, size: size, qt: qt)
    }
    
    let samplePixels = points.count * (w * h) * 3
    let totalPixels = img.width * img.height * 3
    
    let estimatedTotalBits = Double(totalSampleBits) * (Double(totalPixels) / Double(samplePixels))
        
    let ratio = estimatedTotalBits / Double(targetBits)
    let predictedStep = Double(probeStep) * ratio * 1.5
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
        let safeY = min(y, height - 1)
        
        let limit = min(size, width - x)
        if limit > 0 {
            data.withUnsafeBufferPointer { ptr in
                let base = ptr.baseAddress!.advanced(by: safeY * width + x)
                r.withUnsafeMutableBufferPointer { dst in
                    dst.baseAddress!.update(from: base, count: limit)
                    
                    if limit < size {
                        let lastVal = dst[limit - 1]
                        for i in limit..<size {
                            dst[i] = lastVal
                        }
                    }
                }
            }
        } else {
            let lastVal = data[safeY * width + (width - 1)]
            for i in 0..<size {
                r[i] = lastVal
            }
        }
        
        return r
    }
}

@inline(__always)
func toPlaneData420(images: [YCbCrImage]) -> [PlaneData420] {
    return images.map { (img: YCbCrImage) in
        let y = [Int16](unsafeUninitializedCapacity: img.yPlane.count) { (buffer: inout UnsafeMutableBufferPointer<Int16>, initializedCount: inout Int) in
            img.yPlane.withUnsafeBufferPointer { (src: UnsafeBufferPointer<UInt8>) in
                for i in 0..<src.count {
                    buffer[i] = (Int16(src[i]) - 128)
                }
            }
            initializedCount = img.yPlane.count
        }

        let cWidth = ((img.width + 1) / 2)
        let cHeight = ((img.height + 1) / 2)
        let cCount = (cWidth * cHeight)

        let cb: [Int16]
        let cr: [Int16]
        
        if img.ratio == .ratio444 {
            cb = [Int16](unsafeUninitializedCapacity: cCount) { (buffer: inout UnsafeMutableBufferPointer<Int16>, initializedCount: inout Int) in
                img.cbPlane.withUnsafeBufferPointer { (src: UnsafeBufferPointer<UInt8>) in
                    for cy in 0..<cHeight {
                        let py = (cy * 2)
                        let srcRowOffset = (py * img.width)
                        let dstRowOffset = (cy * cWidth)
                        for cx in 0..<cWidth {
                            let px = (cx * 2)
                            let srcOffset = (srcRowOffset + px)
                            let dstOffset = (dstRowOffset + cx)
                            if srcOffset < src.count {
                                buffer[dstOffset] = (Int16(src[srcOffset]) - 128)
                            } else {
                                buffer[dstOffset] = 0
                            }
                        }
                    }
                }
                initializedCount = cCount
            }
            cr = [Int16](unsafeUninitializedCapacity: cCount) { (buffer: inout UnsafeMutableBufferPointer<Int16>, initializedCount: inout Int) in
                img.crPlane.withUnsafeBufferPointer { (src: UnsafeBufferPointer<UInt8>) in
                    for cy in 0..<cHeight {
                        let py = (cy * 2)
                        let srcRowOffset = (py * img.width)
                        let dstRowOffset = (cy * cWidth)
                        for cx in 0..<cWidth {
                            let px = (cx * 2)
                            let srcOffset = (srcRowOffset + px)
                            let dstOffset = (dstRowOffset + cx)
                            if srcOffset < src.count {
                                buffer[dstOffset] = (Int16(src[srcOffset]) - 128)
                            } else {
                                buffer[dstOffset] = 0
                            }
                        }
                    }
                }
                initializedCount = cCount
            }
        } else {
            cb = [Int16](unsafeUninitializedCapacity: img.cbPlane.count) { (buffer: inout UnsafeMutableBufferPointer<Int16>, initializedCount: inout Int) in
                img.cbPlane.withUnsafeBufferPointer { (src: UnsafeBufferPointer<UInt8>) in
                    for i in 0..<src.count {
                        buffer[i] = (Int16(src[i]) - 128)
                    }
                }
                initializedCount = img.cbPlane.count
            }
            cr = [Int16](unsafeUninitializedCapacity: img.crPlane.count) { (buffer: inout UnsafeMutableBufferPointer<Int16>, initializedCount: inout Int) in
                img.crPlane.withUnsafeBufferPointer { (src: UnsafeBufferPointer<UInt8>) in
                    for i in 0..<src.count {
                        buffer[i] = (Int16(src[i]) - 128)
                    }
                }
                initializedCount = img.crPlane.count
            }
        }
        
        return PlaneData420(width: img.width, height: img.height, y: y, cb: cb, cr: cr)
    }
}

@inline(__always)
func subtractPlanes(curr: PlaneData420, predicted: PlaneData420) async -> PlaneData420 {
    @Sendable
    func sub(c: [Int16], p: [Int16]) -> [Int16] {
        let count = c.count
        if count < 1 { return [] }

        var res = [Int16](repeating: 0, count: count)
        c.withUnsafeBufferPointer { cPtr in
            p.withUnsafeBufferPointer { pPtr in
                for i in 0..<count { res[i] = cPtr[i] &- pPtr[i] }
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
        if count < 1 { return [] }

        var curr = [Int16](repeating: 0, count: count)
        r.withUnsafeBufferPointer { rPtr in
            p.withUnsafeBufferPointer { pPtr in
                for i in 0..<count { curr[i] = rPtr[i] &+ pPtr[i] }
            }
        }
        return curr
    }
    
    async let y = add(r: residual.y, p: predicted.y)
    async let cb = add(r: residual.cb, p: predicted.cb)
    async let cr = add(r: residual.cr, p: predicted.cr)
    
    return PlaneData420(width: residual.width, height: residual.height, y: await y, cb: await cb, cr: await cr)
}





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
                
                for dstY in 0..<h {
                    let srcY = min(max(dstY - sY, 0), h - 1)
                    let dstRow = dstY * w
                    let srcRow = srcY * w
                    
                    let dstXStart = max(0, sX)
                    let dstXEnd = min(w, w + sX)
                    
                    if dstXStart < dstXEnd {
                        let srcXStart = dstXStart - sX
                        let copyLen = dstXEnd - dstXStart
                        pOut.advanced(by: dstRow + dstXStart)
                            .update(from: pData.advanced(by: srcRow + srcXStart), count: copyLen)
                    }
                    
                    if sX > 0 {
                        let fillVal = pData[srcRow]
                        for x in 0..<min(sX, w) {
                            pOut[dstRow + x] = fillVal
                        }
                    }
                    
                    if sX < 0 {
                        let fillVal = pData[srcRow + w - 1]
                        for x in max(0, w + sX)..<w {
                            pOut[dstRow + x] = fillVal
                        }
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

public func encode(images: [YCbCrImage], maxbitrate: Int, zeroThreshold: Int = 3, gopSize: Int = 15, sceneChangeThreshold: Int = 8) async throws -> [UInt8] {
    if images.isEmpty { return [] }
    
    let qt = estimateQuantization(img: images[0], targetBits: maxbitrate)
    var out: [UInt8] = []
    
    var prevReconstructed: PlaneData420? = nil
    let planes = toPlaneData420(images: images)
    
    var gopCount = 0
    
    for i in 0..<planes.count {
        let curr = planes[i]
        var forceIFrame = false
        var residual: PlaneData420? = nil
        var predictedPlane: PlaneData420? = nil
        var mvs: [MotionVector] = []
        var meanSAD: Int = 0
        
        if gopCount >= gopSize || prevReconstructed == nil {
            forceIFrame = true
        } else {
            guard let prev = prevReconstructed else { continue }
            
            mvs = estimateMBME(curr: curr, prev: prev)
            predictedPlane = await applyMBME(prev: prev, mvs: mvs)
            residual = await subtractPlanes(curr: curr, predicted: predictedPlane!)
            
            var sumSAD = 0
            for y in 0..<residual!.height {
                for x in 0..<residual!.width {
                    sumSAD += abs(Int(residual!.y[y * residual!.width + x]))
                }
            }
            meanSAD = sumSAD / (residual!.width * residual!.height)
            
            if meanSAD > sceneChangeThreshold {
                forceIFrame = true
                debugLog("[Frame \(i)] Adaptive GOP: Forced I-Frame due to high SAD (\(meanSAD) > \(sceneChangeThreshold))")
            }
        }
        
        if forceIFrame {
            let qtY = QuantizationTable(baseStep: max(1, Int(qt.step)))
            let qtC = QuantizationTable(baseStep: max(1, Int(qt.step) * 2))
            let bytes = try await encodeSpatialLayers(pd: curr, predictedPd: nil, maxbitrate: maxbitrate, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold, isIFrame: true)
            
            out.append(contentsOf: [0x56, 0x45, 0x56, 0x49]) // 'VEVI'
            appendUInt32BE(&out, UInt32(bytes.count))
            out.append(contentsOf: bytes)
            debugLog("[Frame \(i)] I-Frame: \(bytes.count) bytes (\(String(format: "%.2f", Double(bytes.count) / 1024.0)) KB)")
            
            let img16 = try await decodeSpatialLayers(r: bytes, maxLayer: 2)
            prevReconstructed = PlaneData420(img16: img16)
            gopCount = 1
        } else {
            let qtY = QuantizationTable(baseStep: max(1, Int(qt.step) * 4))
            let qtC = QuantizationTable(baseStep: max(1, Int(qt.step) * 8))
            let bytes = try await encodeSpatialLayers(pd: curr, predictedPd: predictedPlane, maxbitrate: maxbitrate, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold, isIFrame: false)
            
            out.append(contentsOf: [0x56, 0x45, 0x56, 0x50]) // 'VEVP'

            var mvBw = CABACEncoder()
            var ctxDx = ContextModel()

            for mv in mvs {
                if mv.dx == 0 && mv.dy == 0 {
                    mvBw.encodeBin(binVal: 0, ctx: &ctxDx)
                } else {
                    mvBw.encodeBin(binVal: 1, ctx: &ctxDx)
                    let sx: UInt8 = mv.dx < 0 ? 1 : 0
                    mvBw.encodeBypass(binVal: sx)
                    let mx = UInt32(abs(mv.dx))
                    encodeExpGolomb(val: mx, encoder: &mvBw)

                    let sy: UInt8 = mv.dy < 0 ? 1 : 0
                    mvBw.encodeBypass(binVal: sy)
                    let my = UInt32(abs(mv.dy))
                    encodeExpGolomb(val: my, encoder: &mvBw)
                }
            }
            mvBw.flush()
            let mvOut = mvBw.getData()
            appendUInt32BE(&out, UInt32(mvs.count))
            appendUInt32BE(&out, UInt32(mvOut.count))
            out.append(contentsOf: mvOut)

            appendUInt32BE(&out, UInt32(bytes.count))
            out.append(contentsOf: bytes)
            debugLog("[Frame \(i)] P-Frame: \(bytes.count) bytes (\(String(format: "%.2f", Double(bytes.count) / 1024.0)) KB) MVs=\(mvs.count) meanSAD=\(meanSAD)")
            
            let img16 = try await decodeSpatialLayers(r: bytes, maxLayer: 2)
            let reconstructedResidual = PlaneData420(img16: img16)
            let reconstructed = await addPlanes(residual: reconstructedResidual, predicted: predictedPlane!)
            prevReconstructed = reconstructed
            gopCount += 1
        }
    }
    
    return out
}
