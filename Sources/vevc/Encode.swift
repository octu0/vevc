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
func encodeCoeffRun(val: Int16, encoder: inout CABACEncoder, run: Int, ctxRun: inout [ContextModel], ctxMag: inout [ContextModel], band: Int) {
    let rIdx = min(run, 7)
    let ctxBandOffset = min(band, 7) * 8
    ctxRun.withUnsafeMutableBufferPointer { ptr in
        guard let base = ptr.baseAddress else { return }
        for i in 0..<rIdx {
            encoder.encodeBin(binVal: 1, ctx: &base[ctxBandOffset + Int(i)])
        }
        if run < 7 {
            encoder.encodeBin(binVal: 0, ctx: &base[ctxBandOffset + Int(rIdx)])
        }
    }
    if run >= 7 {
        let rem = UInt32(run - 7)
        encodeExpGolomb(val: rem, encoder: &encoder)
    }

    let signBit: UInt8
    if val <= -1 {
        signBit = 1
    } else {
        signBit = 0
    }
    let absVal = UInt32(abs(Int(val)))

    encoder.encodeBypass(binVal: signBit)

    let magMinus1 = absVal &- 1
    let numBins = min(magMinus1, 7)
    ctxMag.withUnsafeMutableBufferPointer { ptr in
        guard let base = ptr.baseAddress else { return }
        for i in 0..<numBins {
            encoder.encodeBin(binVal: 1, ctx: &base[ctxBandOffset + Int(i)])
        }
        if magMinus1 < 7 {
            encoder.encodeBin(binVal: 0, ctx: &base[ctxBandOffset + Int(numBins)])
        }
    }

    if magMinus1 >= 7 {
        let rem = magMinus1 &- 7
        encodeExpGolomb(val: rem, encoder: &encoder)
    }
}

@inline(__always)
func blockEncode32(encoder: inout CABACEncoder, block: BlockView, ctxRun: inout [ContextModel], ctxMag: inout [ContextModel]) {
    var lscpX = -1
    var lscpY = -1
    for y in stride(from: 32 - 1, through: 0, by: -1) {
        let ptr = block.rowPointer(y: y)
        for x in stride(from: 32 - 1, through: 0, by: -1) {
            if ptr[x] != 0 {
                lscpX = x
                lscpY = y
                break
            }
        }
        if lscpX != -1 { break }
    }

    if lscpX == -1 {
        encoder.encodeBypass(binVal: 0)
        return
    }
    encoder.encodeBypass(binVal: 1)

    encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
    encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

    var run = 0
    var currentIdx = 0
    for y in 0...lscpY {
        let ptr = block.rowPointer(y: y)
        let endX = (y == lscpY) ? lscpX : (32 - 1)
        for x in 0...endX {
            let val = ptr[x]
            if val == 0 {
                run += 1
            } else {
                let startY = currentIdx / 32
                let startX = currentIdx % 32
                let band = min(startX + startY, 7)
                encodeCoeffRun(val: val, encoder: &encoder, run: run, ctxRun: &ctxRun, ctxMag: &ctxMag, band: band)
                run = 0
                currentIdx = (y * 32 + x) + 1
            }
        }
    }
}

@inline(__always)
func blockEncode16(encoder: inout CABACEncoder, block: BlockView, ctxRun: inout [ContextModel], ctxMag: inout [ContextModel]) {
    var lscpX = -1
    var lscpY = -1
    for y in stride(from: 16 - 1, through: 0, by: -1) {
        let ptr = block.rowPointer(y: y)
        for x in stride(from: 16 - 1, through: 0, by: -1) {
            if ptr[x] != 0 {
                lscpX = x
                lscpY = y
                break
            }
        }
        if lscpX != -1 { break }
    }

    if lscpX == -1 {
        encoder.encodeBypass(binVal: 0)
        return
    }
    encoder.encodeBypass(binVal: 1)

    encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
    encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

    var run = 0
    var currentIdx = 0
    for y in 0...lscpY {
        let ptr = block.rowPointer(y: y)
        let endX = (y == lscpY) ? lscpX : (16 - 1)
        for x in 0...endX {
            let val = ptr[x]
            if val == 0 {
                run += 1
            } else {
                let startY = currentIdx / 16
                let startX = currentIdx % 16
                let band = min(startX + startY, 7)
                encodeCoeffRun(val: val, encoder: &encoder, run: run, ctxRun: &ctxRun, ctxMag: &ctxMag, band: band)
                run = 0
                currentIdx = (y * 16 + x) + 1
            }
        }
    }
}

@inline(__always)
func blockEncode8(encoder: inout CABACEncoder, block: BlockView, ctxRun: inout [ContextModel], ctxMag: inout [ContextModel]) {
    var lscpX = -1
    var lscpY = -1
    for y in stride(from: 8 - 1, through: 0, by: -1) {
        let ptr = block.rowPointer(y: y)
        for x in stride(from: 8 - 1, through: 0, by: -1) {
            if ptr[x] != 0 {
                lscpX = x
                lscpY = y
                break
            }
        }
        if lscpX != -1 { break }
    }

    if lscpX == -1 {
        encoder.encodeBypass(binVal: 0)
        return
    }
    encoder.encodeBypass(binVal: 1)

    encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
    encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

    var run = 0
    var currentIdx = 0
    for y in 0...lscpY {
        let ptr = block.rowPointer(y: y)
        let endX = (y == lscpY) ? lscpX : (8 - 1)
        for x in 0...endX {
            let val = ptr[x]
            if val == 0 {
                run += 1
            } else {
                let startY = currentIdx / 8
                let startX = currentIdx % 8
                let band = min(startX + startY, 7)
                encodeCoeffRun(val: val, encoder: &encoder, run: run, ctxRun: &ctxRun, ctxMag: &ctxMag, band: band)
                run = 0
                currentIdx = (y * 8 + x) + 1
            }
        }
    }
}

@inline(__always)
func blockEncode4(encoder: inout CABACEncoder, block: BlockView, ctxRun: inout [ContextModel], ctxMag: inout [ContextModel]) {
    var lscpX = -1
    var lscpY = -1
    for y in stride(from: 4 - 1, through: 0, by: -1) {
        let ptr = block.rowPointer(y: y)
        for x in stride(from: 4 - 1, through: 0, by: -1) {
            if ptr[x] != 0 {
                lscpX = x
                lscpY = y
                break
            }
        }
        if lscpX != -1 { break }
    }

    if lscpX == -1 {
        encoder.encodeBypass(binVal: 0)
        return
    }
    encoder.encodeBypass(binVal: 1)

    encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
    encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

    var run = 0
    var currentIdx = 0
    for y in 0...lscpY {
        let ptr = block.rowPointer(y: y)
        let endX = (y == lscpY) ? lscpX : (4 - 1)
        for x in 0...endX {
            let val = ptr[x]
            if val == 0 {
                run += 1
            } else {
                let startY = currentIdx / 4
                let startX = currentIdx % 4
                let band = min(startX + startY, 7)
                encodeCoeffRun(val: val, encoder: &encoder, run: run, ctxRun: &ctxRun, ctxMag: &ctxMag, band: band)
                run = 0
                currentIdx = (y * 4 + x) + 1
            }
        }
    }
}

@inline(__always)
func getSubbands32(view: BlockView) -> Subbands {
    let half = 32 / 2
    let base = view.base
    return Subbands(
        ll: BlockView(base: base, width: half, height: half, stride: 32),
        hl: BlockView(base: base.advanced(by: half), width: half, height: half, stride: 32),
        lh: BlockView(base: base.advanced(by: half * 32), width: half, height: half, stride: 32),
        hh: BlockView(base: base.advanced(by: half * 32 + half), width: half, height: half, stride: 32),
        size: half
    )
}

@inline(__always)
func getSubbands16(view: BlockView) -> Subbands {
    let half = 16 / 2
    let base = view.base
    return Subbands(
        ll: BlockView(base: base, width: half, height: half, stride: 16),
        hl: BlockView(base: base.advanced(by: half), width: half, height: half, stride: 16),
        lh: BlockView(base: base.advanced(by: half * 16), width: half, height: half, stride: 16),
        hh: BlockView(base: base.advanced(by: half * 16 + half), width: half, height: half, stride: 16),
        size: half
    )
}

@inline(__always)
func getSubbands8(view: BlockView) -> Subbands {
    let half = 8 / 2
    let base = view.base
    return Subbands(
        ll: BlockView(base: base, width: half, height: half, stride: 8),
        hl: BlockView(base: base.advanced(by: half), width: half, height: half, stride: 8),
        lh: BlockView(base: base.advanced(by: half * 8), width: half, height: half, stride: 8),
        hh: BlockView(base: base.advanced(by: half * 8 + half), width: half, height: half, stride: 8),
        size: half
    )
}

@inline(__always)
func blockEncodeDPCM4(encoder: inout CABACEncoder, block: BlockView, lastVal: inout Int16, ctxRun: inout [ContextModel], ctxMag: inout [ContextModel]) {
    var lscpIdx = -1
    
    // Pass 1: find LSCP
    let ptr0 = block.rowPointer(y: 0)
    if ptr0[0] - lastVal != 0 { lscpIdx = 0 }
    for x in 1..<4 {
        if ptr0[x] - ptr0[x - 1] != 0 { lscpIdx = max(lscpIdx, x) }
    }
    
    for y in 1..<4 {
        let ptr = block.rowPointer(y: y)
        let ptrPrev = block.rowPointer(y: y - 1)
        if ptr[0] - ptrPrev[0] != 0 { lscpIdx = max(lscpIdx, y * 4 + 0) }
        
        for x in 1..<4 {
            let a = Int(ptr[x - 1])
            let b = Int(ptrPrev[x])
            let c = Int(ptrPrev[x - 1])
            let predicted: Int16
            if a <= c && b <= c {
                predicted = Int16(min(a, b))
            } else if c <= a && c <= b {
                predicted = Int16(max(a, b))
            } else {
                predicted = Int16(a + b - c)
            }
            if ptr[x] - predicted != 0 { lscpIdx = max(lscpIdx, y * 4 + x) }
        }
    }

    if lscpIdx == -1 {
        encoder.encodeBypass(binVal: 0)
    } else {
        encoder.encodeBypass(binVal: 1)
        let lscpX = lscpIdx % 4
        let lscpY = lscpIdx / 4
        encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
        encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

        // Pass 2: encode up to LSCP
        var currentIdx = 0
        var startIdxForRun = 0
        var run = 0
        let diff00 = ptr0[0] - lastVal
        if diff00 == 0 {
            run += 1
        } else {
            encodeCoeffRun(val: diff00, encoder: &encoder, run: run, ctxRun: &ctxRun, ctxMag: &ctxMag, band: 0)
            run = 0
            startIdxForRun = currentIdx + 1
        }

        for x in 1..<4 {
            currentIdx += 1
            if currentIdx > lscpIdx { break }
            let diff = ptr0[x] - ptr0[x - 1]
            if diff == 0 {
                run += 1
            } else {
                let startY = startIdxForRun / 4
                let startX = startIdxForRun % 4
                let band = min(startX + startY, 7)
                encodeCoeffRun(val: diff, encoder: &encoder, run: run, ctxRun: &ctxRun, ctxMag: &ctxMag, band: band)
                run = 0
                startIdxForRun = currentIdx + 1
            }
        }

        for y in 1..<4 {
            if currentIdx >= lscpIdx { break }
            let ptr = block.rowPointer(y: y)
            let ptrPrev = block.rowPointer(y: y - 1)

            currentIdx += 1
            if currentIdx > lscpIdx { break }
            let diffY0 = ptr[0] - ptrPrev[0]
            if diffY0 == 0 {
                run += 1
            } else {
                let startY = startIdxForRun / 4
                let startX = startIdxForRun % 4
                let band = min(startX + startY, 7)
                encodeCoeffRun(val: diffY0, encoder: &encoder, run: run, ctxRun: &ctxRun, ctxMag: &ctxMag, band: band)
                run = 0
                startIdxForRun = currentIdx + 1
            }

            for x in 1..<4 {
                currentIdx += 1
                if currentIdx > lscpIdx { break }
                let a = Int(ptr[x - 1])
                let b = Int(ptrPrev[x])
                let c = Int(ptrPrev[x - 1])
                let predicted: Int16
                if a <= c && b <= c {
                    predicted = Int16(min(a, b))
                } else if c <= a && c <= b {
                    predicted = Int16(max(a, b))
                } else {
                    predicted = Int16(a + b - c)
                }
                let diff = ptr[x] - predicted
                if diff == 0 {
                    run += 1
                } else {
                    let startY = startIdxForRun / 4
                    let startX = startIdxForRun % 4
                    let band = min(startX + startY, 7)
                    encodeCoeffRun(val: diff, encoder: &encoder, run: run, ctxRun: &ctxRun, ctxMag: &ctxMag, band: band)
                    run = 0
                    startIdxForRun = currentIdx + 1
                }
            }
        }
    }

    lastVal = block.rowPointer(y: 4 - 1)[4 - 1]
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
func isEffectivelyZero32(hl: inout BlockView, lh: inout BlockView, hh: inout BlockView, threshold: Int) -> Bool {
    let thPos = threshold
    let thNeg = -1 * threshold
    for y in 0..<32 {
        let ptrHL = hl.rowPointer(y: y)
        let ptrLH = lh.rowPointer(y: y)
        let ptrHH = hh.rowPointer(y: y)
        for x in 0..<32 {
            if thPos < ptrHL[x] || ptrHL[x] < thNeg {
                return false
            }
            if thPos < ptrLH[x] || ptrLH[x] < thNeg {
                return false
            }
            if thPos < ptrHH[x] || ptrHH[x] < thNeg {
                return false
            }
        }
    }
    
    for y in 0..<32 {
        let ptrHL = hl.rowPointer(y: y)
        let ptrLH = lh.rowPointer(y: y)
        let ptrHH = hh.rowPointer(y: y)
        for x in 0..<32 {
            ptrHL[x] = 0
            ptrLH[x] = 0
            ptrHH[x] = 0
        }
    }
    
    return true
}

@inline(__always)
func isEffectivelyZero16(hl: inout BlockView, lh: inout BlockView, hh: inout BlockView, threshold: Int) -> Bool {
    let thPos = threshold
    let thNeg = -1 * threshold
    for y in 0..<16 {
        let ptrHL = hl.rowPointer(y: y)
        let ptrLH = lh.rowPointer(y: y)
        let ptrHH = hh.rowPointer(y: y)
        for x in 0..<16 {
            if thPos < ptrHL[x] || ptrHL[x] < thNeg {
                return false
            }
            if thPos < ptrLH[x] || ptrLH[x] < thNeg {
                return false
            }
            if thPos < ptrHH[x] || ptrHH[x] < thNeg {
                return false
            }
        }
    }
    
    for y in 0..<16 {
        let ptrHL = hl.rowPointer(y: y)
        let ptrLH = lh.rowPointer(y: y)
        let ptrHH = hh.rowPointer(y: y)
        for x in 0..<16 {
            ptrHL[x] = 0
            ptrLH[x] = 0
            ptrHH[x] = 0
        }
    }
    
    return true
}

@inline(__always)
func isEffectivelyZero8(hl: inout BlockView, lh: inout BlockView, hh: inout BlockView, threshold: Int) -> Bool {
    let thPos = threshold
    let thNeg = -1 * threshold
    for y in 0..<8 {
        let ptrHL = hl.rowPointer(y: y)
        let ptrLH = lh.rowPointer(y: y)
        let ptrHH = hh.rowPointer(y: y)
        for x in 0..<8 {
            if thPos < ptrHL[x] || ptrHL[x] < thNeg {
                return false
            }
            if thPos < ptrLH[x] || ptrLH[x] < thNeg {
                return false
            }
            if thPos < ptrHH[x] || ptrHH[x] < thNeg {
                return false
            }
        }
    }
    
    for y in 0..<8 {
        let ptrHL = hl.rowPointer(y: y)
        let ptrLH = lh.rowPointer(y: y)
        let ptrHH = hh.rowPointer(y: y)
        for x in 0..<8 {
            ptrHL[x] = 0
            ptrLH[x] = 0
            ptrHH[x] = 0
        }
    }
    
    return true
}

@inline(__always)
func isEffectivelyZero4(hl: inout BlockView, lh: inout BlockView, hh: inout BlockView, threshold: Int) -> Bool {
    let thPos = threshold
    let thNeg = -1 * threshold
    for y in 0..<4 {
        let ptrHL = hl.rowPointer(y: y)
        let ptrLH = lh.rowPointer(y: y)
        let ptrHH = hh.rowPointer(y: y)
        for x in 0..<4 {
            if thPos < ptrHL[x] || ptrHL[x] < thNeg {
                return false
            }
            if thPos < ptrLH[x] || ptrLH[x] < thNeg {
                return false
            }
            if thPos < ptrHH[x] || ptrHH[x] < thNeg {
                return false
            }
        }
    }
    
    for y in 0..<4 {
        let ptrHL = hl.rowPointer(y: y)
        let ptrLH = lh.rowPointer(y: y)
        let ptrHH = hh.rowPointer(y: y)
        for x in 0..<4 {
            ptrHL[x] = 0
            ptrLH[x] = 0
            ptrHH[x] = 0
        }
    }
    
    return true
}

@inline(__always)
func isEffectivelyZeroBase4(ll: BlockView, hl: inout BlockView, lh: inout BlockView, hh: inout BlockView, threshold: Int) -> Bool {
    for y in 0..<4 {
        let ptrLL = ll.rowPointer(y: y)
        for x in 0..<4 {
            if ptrLL[x] != 0 {
                return false
            }
        }
    }
    
    let thPos = threshold
    let thNeg = -1 * threshold
    for y in 0..<4 {
        let ptrHL = hl.rowPointer(y: y)
        let ptrLH = lh.rowPointer(y: y)
        let ptrHH = hh.rowPointer(y: y)
        for x in 0..<4 {
            if thPos < ptrHL[x] || ptrHL[x] < thNeg {
                return false
            }
            if thPos < ptrLH[x] || ptrLH[x] < thNeg {
                return false
            }
            if thPos < ptrHH[x] || ptrHH[x] < thNeg {
                return false
            }
        }
    }
    
    for y in 0..<4 {
        let ptrHL = hl.rowPointer(y: y)
        let ptrLH = lh.rowPointer(y: y)
        let ptrHH = hh.rowPointer(y: y)
        for x in 0..<4 {
            ptrHL[x] = 0
            ptrLH[x] = 0
            ptrHH[x] = 0
        }
    }
    
    return true
}

@inline(__always)
func transformLayer32(block: inout Block2D, qt: QuantizationTable) {
    block.withView { view in
        var sub = dwt2d_32(&view)
        quantizeMidSignedMapping(&sub.hl, qt: qt)
        quantizeMidSignedMapping(&sub.lh, qt: qt)
        quantizeHighSignedMapping(&sub.hh, qt: qt)
    }
}

@inline(__always)
func transformLayer16(block: inout Block2D, qt: QuantizationTable) {
    block.withView { view in
        var sub = dwt2d_16(&view)
        quantizeMidSignedMapping(&sub.hl, qt: qt)
        quantizeMidSignedMapping(&sub.lh, qt: qt)
        quantizeHighSignedMapping(&sub.hh, qt: qt)
    }
}

@inline(__always)
func transformLayer8(block: inout Block2D, qt: QuantizationTable) {
    block.withView { view in
        var sub = dwt2d_8(&view)
        quantizeMidSignedMapping(&sub.hl, qt: qt)
        quantizeMidSignedMapping(&sub.lh, qt: qt)
        quantizeHighSignedMapping(&sub.hh, qt: qt)
    }
}

@inline(__always)
func transformBase8(block: inout Block2D, qt: QuantizationTable) {
    block.withView { view in
        var sub = dwt2d_8(&view)
        quantizeLow(&sub.ll, qt: qt)
        quantizeMidSignedMapping(&sub.hl, qt: qt)
        quantizeMidSignedMapping(&sub.lh, qt: qt)
        quantizeHighSignedMapping(&sub.hh, qt: qt)
    }
}

@inline(__always)
func encodePlaneSubbands32(blocks: [Block2D], zeroThreshold: Int) -> [UInt8] {
    var blocks = blocks
    var bwFlags = CABACBitWriter(capacity: (blocks.count + 7) / 8)
    var nonZeroIndices: [Int] = []
    
    for i in blocks.indices {
        blocks[i].withView { view in
            let subs = getSubbands32(view: view)
            var hl = subs.hl
            var lh = subs.lh
            var hh = subs.hh
            if isEffectivelyZero16(hl: &hl, lh: &lh, hh: &hh, threshold: zeroThreshold) {
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
    var ctxRunHL = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagHL = [ContextModel](repeating: ContextModel(), count: 64)
    
    var ctxRunLH = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagLH = [ContextModel](repeating: ContextModel(), count: 64)

    var ctxRunHH = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagHH = [ContextModel](repeating: ContextModel(), count: 64)

    for i in nonZeroIndices {
        blocks[i].withView { view in
            let subs = getSubbands32(view: view)
            blockEncode16(encoder: &encoder, block: subs.hl, ctxRun: &ctxRunHL, ctxMag: &ctxMagHL)
            blockEncode16(encoder: &encoder, block: subs.lh, ctxRun: &ctxRunLH, ctxMag: &ctxMagLH)
            blockEncode16(encoder: &encoder, block: subs.hh, ctxRun: &ctxRunHH, ctxMag: &ctxMagHH)
        }
    }
    
    encoder.flush()
    var out = bwFlags.data
    out.append(contentsOf: encoder.getData())
    return out
}

@inline(__always)
func encodePlaneSubbands16(blocks: [Block2D], zeroThreshold: Int) -> [UInt8] {
    var blocks = blocks
    var bwFlags = CABACBitWriter(capacity: (blocks.count + 7) / 8)
    var nonZeroIndices: [Int] = []
    
    for i in blocks.indices {
        blocks[i].withView { view in
            let subs = getSubbands16(view: view)
            var hl = subs.hl
            var lh = subs.lh
            var hh = subs.hh
            if isEffectivelyZero8(hl: &hl, lh: &lh, hh: &hh, threshold: zeroThreshold) {
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
    var ctxRunHL = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagHL = [ContextModel](repeating: ContextModel(), count: 64)
    
    var ctxRunLH = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagLH = [ContextModel](repeating: ContextModel(), count: 64)

    var ctxRunHH = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagHH = [ContextModel](repeating: ContextModel(), count: 64)

    for i in nonZeroIndices {
        blocks[i].withView { view in
            let subs = getSubbands16(view: view)
            blockEncode8(encoder: &encoder, block: subs.hl, ctxRun: &ctxRunHL, ctxMag: &ctxMagHL)
            blockEncode8(encoder: &encoder, block: subs.lh, ctxRun: &ctxRunLH, ctxMag: &ctxMagLH)
            blockEncode8(encoder: &encoder, block: subs.hh, ctxRun: &ctxRunHH, ctxMag: &ctxMagHH)
        }
    }
    
    encoder.flush()
    var out = bwFlags.data
    out.append(contentsOf: encoder.getData())
    return out
}

@inline(__always)
func encodePlaneSubbands8(blocks: [Block2D], zeroThreshold: Int) -> [UInt8] {
    var blocks = blocks
    var bwFlags = CABACBitWriter(capacity: (blocks.count + 7) / 8)
    var nonZeroIndices: [Int] = []
    
    for i in blocks.indices {
        blocks[i].withView { view in
            let subs = getSubbands8(view: view)
            var hl = subs.hl
            var lh = subs.lh
            var hh = subs.hh
            if isEffectivelyZero4(hl: &hl, lh: &lh, hh: &hh, threshold: zeroThreshold) {
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
    var ctxRunHL = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagHL = [ContextModel](repeating: ContextModel(), count: 64)
    
    var ctxRunLH = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagLH = [ContextModel](repeating: ContextModel(), count: 64)

    var ctxRunHH = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagHH = [ContextModel](repeating: ContextModel(), count: 64)

    for i in nonZeroIndices {
        blocks[i].withView { view in
            let subs = getSubbands8(view: view)
            blockEncode4(encoder: &encoder, block: subs.hl, ctxRun: &ctxRunHL, ctxMag: &ctxMagHL)
            blockEncode4(encoder: &encoder, block: subs.lh, ctxRun: &ctxRunLH, ctxMag: &ctxMagLH)
            blockEncode4(encoder: &encoder, block: subs.hh, ctxRun: &ctxRunHH, ctxMag: &ctxMagHH)
        }
    }
    
    encoder.flush()
    var out = bwFlags.data
    out.append(contentsOf: encoder.getData())
    return out
}

@inline(__always)
func encodePlaneBaseSubbands8(blocks: [Block2D], zeroThreshold: Int) -> [UInt8] {
    var blocks = blocks
    var bwFlags = CABACBitWriter(capacity: (blocks.count + 7) / 8)
    var nonZeroIndices: [Int] = []
    
    for i in blocks.indices {
        blocks[i].withView { view in
            let subs = getSubbands8(view: view)
            var hl = subs.hl
            var lh = subs.lh
            var hh = subs.hh
            if isEffectivelyZeroBase4(ll: subs.ll, hl: &hl, lh: &lh, hh: &hh, threshold: zeroThreshold) {
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
    var ctxRunLL = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagLL = [ContextModel](repeating: ContextModel(), count: 64)
    
    var ctxRunHL = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagHL = [ContextModel](repeating: ContextModel(), count: 64)

    var ctxRunLH = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagLH = [ContextModel](repeating: ContextModel(), count: 64)

    var ctxRunHH = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagHH = [ContextModel](repeating: ContextModel(), count: 64)
    
    var lastVal: Int16 = 0
    
    var nzCur = 0
    let nzCount = nonZeroIndices.count
    for i in blocks.indices {
        if nzCur < nzCount && nonZeroIndices[nzCur] == i {
            nzCur += 1

            blocks[i].withView { view in
                let subs = getSubbands8(view: view)
                blockEncodeDPCM4(encoder: &encoder, block: subs.ll, lastVal: &lastVal, ctxRun: &ctxRunLL, ctxMag: &ctxMagLL)
                blockEncode4(encoder: &encoder, block: subs.hl, ctxRun: &ctxRunHL, ctxMag: &ctxMagHL)
                blockEncode4(encoder: &encoder, block: subs.lh, ctxRun: &ctxRunLH, ctxMag: &ctxMagLH)
                blockEncode4(encoder: &encoder, block: subs.hh, ctxRun: &ctxRunHH, ctxMag: &ctxMagHH)
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

private func estimateRiceBitsDPCM4(block: BlockView, lastVal: inout Int16) -> Int {
    let count = 4 * 4
    let ptr0 = block.rowPointer(y: 0)
    
    var sumDiffAbs = abs(Int(ptr0[0] - lastVal))
    for x in 1..<4 {
        sumDiffAbs += abs(Int(ptr0[x] - ptr0[x - 1]))
    }
    
    for y in 1..<4 {
        let ptr = block.rowPointer(y: y)
        let ptrPrev = block.rowPointer(y: y - 1)
        
        sumDiffAbs += abs(Int(ptr[0] - ptrPrev[0]))
        
        for x in 1..<4 {
            let a = Int(ptr[x - 1])
            let b = Int(ptrPrev[x])
            let c = Int(ptrPrev[x - 1])
            let predicted: Int16
            if a <= c && b <= c {
                predicted = Int16(min(a, b))
            } else if c <= a && c <= b {
                predicted = Int16(max(a, b))
            } else {
                predicted = Int16(a + b - c)
            }
            sumDiffAbs += abs(Int(ptr[x] - predicted))
        }
    }
    lastVal = block.rowPointer(y: 4 - 1)[4 - 1]
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

@inline(__always)
private func measureBlockBits8(block: inout Block2D, qt: QuantizationTable) -> Int {
    var sub = block.withView { view in
        return dwt2d_8(&view)
    }
    
    quantizeLow(&sub.ll, qt: qt)
    quantizeMid(&sub.hl, qt: qt)
    quantizeMid(&sub.lh, qt: qt)
    quantizeHigh(&sub.hh, qt: qt)
    
    var hl = sub.hl
    var lh = sub.lh
    var hh = sub.hh
    if isEffectivelyZeroBase4(ll: sub.ll, hl: &hl, lh: &lh, hh: &hh, threshold: 0) {
        return 1
    }
    
    var bits = 1
    var lastVal: Int16 = 0
    bits += estimateRiceBitsDPCM4(block: sub.ll, lastVal: &lastVal)
    bits += estimateRiceBits4(block: sub.hl)
    bits += estimateRiceBits4(block: sub.lh)
    bits += estimateRiceBits4(block: sub.hh)
    
    return bits
}

@inline(__always)
private func estimateRiceBits32(block: BlockView) -> Int {
    var sumAbs = 0
    let count = (32 * 32)
    
    for y in 0..<32 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<32 {
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

@inline(__always)
private func estimateRiceBits16(block: BlockView) -> Int {
    var sumAbs = 0
    let count = (16 * 16)
    
    for y in 0..<16 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<16 {
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

@inline(__always)
private func estimateRiceBits8(block: BlockView) -> Int {
    var sumAbs = 0
    let count = (8 * 8)
    
    for y in 0..<8 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<8 {
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

@inline(__always)
private func estimateRiceBits4(block: BlockView) -> Int {
    var sumAbs = 0
    let count = (4 * 4)
    
    for y in 0..<4 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<4 {
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

@inline(__always)
func estimateQuantization(img: YCbCrImage, targetBits: Int) -> QuantizationTable {
    let probeStep = 64
    let qt = QuantizationTable(baseStep: probeStep)
    
    let w = (img.width / 8)
    let h = (img.height / 8)
    
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
        totalSampleBits += measureBlockBits8(block: &blockY, qt: qt)
        
        var blockCb = fetchBlockCb(reader: reader, x: sx, y: sy, w: w, h: h)
        totalSampleBits += measureBlockBits8(block: &blockCb, qt: qt)
        
        var blockCr = fetchBlockCr(reader: reader, x: sx, y: sy, w: w, h: h)
        totalSampleBits += measureBlockBits8(block: &blockCr, qt: qt)
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
                guard let basePtr = ptr.baseAddress else { return }
                let base = basePtr.advanced(by: safeY * width + x)
                r.withUnsafeMutableBufferPointer { dst in
                    guard let dstBase = dst.baseAddress else { return }
                    dstBase.update(from: base, count: limit)
                    
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
                    
                    if sX <= -1 {
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

@inline(__always)
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
        var predictedPlane: PlaneData420? = nil
        var mvs = MotionVectors(count: 0)
        var meanSAD: Int = 0
        
        if gopSize <= gopCount || prevReconstructed == nil {
            forceIFrame = true
        } else {
            guard let prev = prevReconstructed else { continue }
            
            mvs = estimateMBME(curr: curr, prev: prev)
            let predicted = await applyMBME(prev: prev, mvs: mvs)
            predictedPlane = predicted
            let res = await subtractPlanes(curr: curr, predicted: predicted)
            
            var sumSAD = 0
            for y in 0..<res.height {
                for x in 0..<res.width {
                    sumSAD += abs(Int(res.y[y * res.width + x]))
                }
            }
            meanSAD = sumSAD / (res.width * res.height)
            
            if sceneChangeThreshold < meanSAD {
                forceIFrame = true
                debugLog("[Frame \(i)] Adaptive GOP: Forced I-Frame due to high SAD (\(meanSAD) > \(sceneChangeThreshold))")
            }
        }
        
        if forceIFrame {
            let qtY = QuantizationTable(baseStep: max(1, Int(qt.step)))
            let qtC = QuantizationTable(baseStep: max(1, Int(qt.step) * 2))
            let bytes = try await encodeSpatialLayers(pd: curr, predictedPd: nil, maxbitrate: maxbitrate, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
            
            out.append(contentsOf: [0x56, 0x45, 0x56, 0x49])
            appendUInt32BE(&out, UInt32(bytes.count))
            out.append(contentsOf: bytes)
            debugLog("[Frame \(i)] I-Frame: \(bytes.count) bytes (\(String(format: "%.2f", Double(bytes.count) / 1024.0)) KB)")
            
            let img16 = try await decodeSpatialLayers(r: bytes, maxLayer: 2)
            prevReconstructed = PlaneData420(img16: img16)
            gopCount = 1
        } else {
            let qtY = QuantizationTable(baseStep: max(1, Int(qt.step) * 4))
            let qtC = QuantizationTable(baseStep: max(1, Int(qt.step) * 8))
            let bytes = try await encodeSpatialLayers(pd: curr, predictedPd: predictedPlane, maxbitrate: maxbitrate, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
            
            out.append(contentsOf: [0x56, 0x45, 0x56, 0x50])

            var mvBw = CABACEncoder()
            var ctxDx = ContextModel()

            let mbSize = 32
            let mbCols = (curr.width + mbSize - 1) / mbSize
            for mvIdx in 0..<mvs.dx.count {
                let mbX = mvIdx % mbCols
                let mbY = mvIdx / mbCols
                let pmv = calculatePMV(mvs: mvs, mbX: mbX, mbY: mbY, mbCols: mbCols)
                let mvdX = mvs.dx[mvIdx] - pmv.dx
                let mvdY = mvs.dy[mvIdx] - pmv.dy

                if mvdX == 0 && mvdY == 0 {
                    mvBw.encodeBin(binVal: 0, ctx: &ctxDx)
                } else {
                    mvBw.encodeBin(binVal: 1, ctx: &ctxDx)

                    let sx: UInt8
                    if mvdX <= -1 {
                        sx = 1
                    } else {
                        sx = 0
                    }
                    mvBw.encodeBypass(binVal: sx)
                    let mx = UInt32(abs(mvdX))
                    encodeExpGolomb(val: mx, encoder: &mvBw)

                    let sy: UInt8
                    if mvdY <= -1 {
                        sy = 1
                    } else {
                        sy = 0
                    }
                    mvBw.encodeBypass(binVal: sy)
                    let my = UInt32(abs(mvdY))
                    encodeExpGolomb(val: my, encoder: &mvBw)
                }
            }
            mvBw.flush()
            let mvOut = mvBw.getData()
            appendUInt32BE(&out, UInt32(mvs.dx.count))
            appendUInt32BE(&out, UInt32(mvOut.count))
            out.append(contentsOf: mvOut)

            appendUInt32BE(&out, UInt32(bytes.count))
            out.append(contentsOf: bytes)
            let totalBytes = bytes.count + mvOut.count
            debugLog("[Frame \(i)] P-Frame: \(totalBytes) bytes (MV: \(mvOut.count) bytes, Data: \(bytes.count) bytes) MVs=\(mvs.dx.count) meanSAD=\(meanSAD) [PMV & LSCP applied]")
            
            let img16 = try await decodeSpatialLayers(r: bytes, maxLayer: 2)
            let reconstructedResidual = PlaneData420(img16: img16)
            if let predicted = predictedPlane {
                let reconstructed = await addPlanes(residual: reconstructedResidual, predicted: predicted)
                prevReconstructed = reconstructed
            } else {
                prevReconstructed = reconstructedResidual
            }
            gopCount += 1
        }
    }
    
    return out
}
