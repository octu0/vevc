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
    let ptr0 = block.rowPointer(y: 0)
    let ptr1 = block.rowPointer(y: 1)
    let ptr2 = block.rowPointer(y: 2)
    let ptr3 = block.rowPointer(y: 3)
    
    @inline(__always)
    func diffMED(_ x: Int16, _ a: Int16, _ b: Int16, _ c: Int16) -> Int16 {
        let ia = Int(a), ib = Int(b), ic = Int(c)
        let predicted: Int
        if ia <= ic && ib <= ic {
            predicted = min(ia, ib)
        } else if ic <= ia && ic <= ib {
            predicted = max(ia, ib)
        } else {
            predicted = ia + ib - ic
        }
        return Int16(truncatingIfNeeded: Int(x) - predicted)
    }

    let err00 = Int16(truncatingIfNeeded: Int(ptr0[0]) - Int(lastVal))
    let err01 = Int16(truncatingIfNeeded: Int(ptr0[1]) - Int(ptr0[0]))
    let err02 = Int16(truncatingIfNeeded: Int(ptr0[2]) - Int(ptr0[1]))
    let err03 = Int16(truncatingIfNeeded: Int(ptr0[3]) - Int(ptr0[2]))

    let err10 = Int16(truncatingIfNeeded: Int(ptr1[0]) - Int(ptr0[0]))
    let err11 = diffMED(ptr1[1], ptr1[0], ptr0[1], ptr0[0])
    let err12 = diffMED(ptr1[2], ptr1[1], ptr0[2], ptr0[1])
    let err13 = diffMED(ptr1[3], ptr1[2], ptr0[3], ptr0[2])

    let err20 = Int16(truncatingIfNeeded: Int(ptr2[0]) - Int(ptr1[0]))
    let err21 = diffMED(ptr2[1], ptr2[0], ptr1[1], ptr1[0])
    let err22 = diffMED(ptr2[2], ptr2[1], ptr1[2], ptr1[1])
    let err23 = diffMED(ptr2[3], ptr2[2], ptr1[3], ptr1[2])

    let err30 = Int16(truncatingIfNeeded: Int(ptr3[0]) - Int(ptr2[0]))
    let err31 = diffMED(ptr3[1], ptr3[0], ptr2[1], ptr2[0])
    let err32 = diffMED(ptr3[2], ptr3[1], ptr2[2], ptr2[1])
    let err33 = diffMED(ptr3[3], ptr3[2], ptr2[3], ptr2[2])

    let errors = [
        err00, err01, err02, err03,
        err10, err11, err12, err13,
        err20, err21, err22, err23,
        err30, err31, err32, err33
    ]
    
    var lscpIdx = -1
    for i in 0..<16 {
        if errors[i] != 0 {
            lscpIdx = i
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

        var currentIdx = 0
        var startIdxForRun = 0
        var run = 0

        for i in 0...lscpIdx {
            let diff = errors[i]
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
            currentIdx += 1
        }
    }

    lastVal = ptr3[3]
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

#if arch(arm64) || arch(x86_64) || arch(wasm32)

@inline(__always)
func isEffectivelyZero32_SIMD(data: UnsafeMutableBufferPointer<Int16>, threshold: Int16) -> Bool {
    guard let base = data.baseAddress else { return false }
    let thPos = SIMD16<Int16>(repeating: threshold)
    let thNeg = SIMD16<Int16>(repeating: -threshold)

    let lowerHalfBase = base + 16 * 32
    for i in stride(from: 0, to: 512, by: 16) {
        let vec = SIMD16<Int16>(UnsafeBufferPointer(start: lowerHalfBase + i, count: 16))
        let mask = (vec .> thPos) .| (vec .< thNeg)
        if mask[0] || mask[1] || mask[2] || mask[3] || mask[4] || mask[5] || mask[6] || mask[7] || mask[8] || mask[9] || mask[10] || mask[11] || mask[12] || mask[13] || mask[14] || mask[15] { return false }
    }
    for y in 0..<16 {
        let ptr = base + y * 32 + 16
        let vec = SIMD16<Int16>(UnsafeBufferPointer(start: ptr, count: 16))
        let mask = (vec .> thPos) .| (vec .< thNeg)
        if mask[0] || mask[1] || mask[2] || mask[3] || mask[4] || mask[5] || mask[6] || mask[7] || mask[8] || mask[9] || mask[10] || mask[11] || mask[12] || mask[13] || mask[14] || mask[15] { return false }
    }

    let zeroVec = SIMD16<Int16>(repeating: 0)
    for i in stride(from: 0, to: 512, by: 16) {
        let ptr = UnsafeMutableRawPointer(lowerHalfBase + i).assumingMemoryBound(to: SIMD16<Int16>.self)
        ptr.pointee = zeroVec
    }
    for y in 0..<16 {
        let ptr = UnsafeMutableRawPointer(base + y * 32 + 16).assumingMemoryBound(to: SIMD16<Int16>.self)
        ptr.pointee = zeroVec
    }
    return true
}

@inline(__always)
func isEffectivelyZero16_SIMD(data: UnsafeMutableBufferPointer<Int16>, threshold: Int16) -> Bool {
    guard let base = data.baseAddress else { return false }
    let thPos = SIMD8<Int16>(repeating: threshold)
    let thNeg = SIMD8<Int16>(repeating: -threshold)

    let lowerHalfBase = base + 8 * 16
    for i in stride(from: 0, to: 128, by: 8) {
        let vec = SIMD8<Int16>(UnsafeBufferPointer(start: lowerHalfBase + i, count: 8))
        let mask = (vec .> thPos) .| (vec .< thNeg)
        if mask[0] || mask[1] || mask[2] || mask[3] || mask[4] || mask[5] || mask[6] || mask[7] { return false }
    }
    for y in 0..<8 {
        let ptr = base + y * 16 + 8
        let vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr, count: 8))
        let mask = (vec .> thPos) .| (vec .< thNeg)
        if mask[0] || mask[1] || mask[2] || mask[3] || mask[4] || mask[5] || mask[6] || mask[7] { return false }
    }

    let zeroVec = SIMD8<Int16>(repeating: 0)
    for i in stride(from: 0, to: 128, by: 8) {
        let ptr = UnsafeMutableRawPointer(lowerHalfBase + i).assumingMemoryBound(to: SIMD8<Int16>.self)
        ptr.pointee = zeroVec
    }
    for y in 0..<8 {
        let ptr = UnsafeMutableRawPointer(base + y * 16 + 8).assumingMemoryBound(to: SIMD8<Int16>.self)
        ptr.pointee = zeroVec
    }
    return true
}

@inline(__always)
func isEffectivelyZero8_SIMD(data: UnsafeMutableBufferPointer<Int16>, threshold: Int16) -> Bool {
    guard let base = data.baseAddress else { return false }
    let thPos = SIMD4<Int16>(repeating: threshold)
    let thNeg = SIMD4<Int16>(repeating: -threshold)

    let lowerHalfBase = base + 4 * 8
    for i in stride(from: 0, to: 32, by: 4) {
        let vec = SIMD4<Int16>(UnsafeBufferPointer(start: lowerHalfBase + i, count: 4))
        let mask = (vec .> thPos) .| (vec .< thNeg)
        if mask[0] || mask[1] || mask[2] || mask[3] { return false }
    }
    for y in 0..<4 {
        let ptr = base + y * 8 + 4
        let vec = SIMD4<Int16>(UnsafeBufferPointer(start: ptr, count: 4))
        let mask = (vec .> thPos) .| (vec .< thNeg)
        if mask[0] || mask[1] || mask[2] || mask[3] { return false }
    }

    let zeroVec = SIMD4<Int16>(repeating: 0)
    for i in stride(from: 0, to: 32, by: 4) {
        let ptr = UnsafeMutableRawPointer(lowerHalfBase + i).assumingMemoryBound(to: SIMD4<Int16>.self)
        ptr.pointee = zeroVec
    }
    for y in 0..<4 {
        let ptr = UnsafeMutableRawPointer(base + y * 8 + 4).assumingMemoryBound(to: SIMD4<Int16>.self)
        ptr.pointee = zeroVec
    }
    return true
}

@inline(__always)
func isEffectivelyZero4_SIMD(data: UnsafeMutableBufferPointer<Int16>, threshold: Int16) -> Bool {
    guard let base = data.baseAddress else { return false }
    let thPos = SIMD2<Int16>(repeating: threshold)
    let thNeg = SIMD2<Int16>(repeating: -threshold)

    let lowerHalfBase = base + 2 * 4
    for i in stride(from: 0, to: 8, by: 2) {
        let vec = SIMD2<Int16>(UnsafeBufferPointer(start: lowerHalfBase + i, count: 2))
        let mask = (vec .> thPos) .| (vec .< thNeg)
        if mask[0] || mask[1] { return false }
    }
    for y in 0..<2 {
        let ptr = base + y * 4 + 2
        let vec = SIMD2<Int16>(UnsafeBufferPointer(start: ptr, count: 2))
        let mask = (vec .> thPos) .| (vec .< thNeg)
        if mask[0] || mask[1] { return false }
    }

    let zeroVec = SIMD2<Int16>(repeating: 0)
    for i in stride(from: 0, to: 8, by: 2) {
        let ptr = UnsafeMutableRawPointer(lowerHalfBase + i).assumingMemoryBound(to: SIMD2<Int16>.self)
        ptr.pointee = zeroVec
    }
    for y in 0..<2 {
        let ptr = UnsafeMutableRawPointer(base + y * 4 + 2).assumingMemoryBound(to: SIMD2<Int16>.self)
        ptr.pointee = zeroVec
    }
    return true
}

@inline(__always)
func isEffectivelyZeroBase4_SIMD(data: UnsafeMutableBufferPointer<Int16>, threshold: Int16) -> Bool {
    guard let base = data.baseAddress else { return false }
    // Check LL
    for y in 0..<2 {
        let ptr = base + y * 4
        let vec = SIMD2<Int16>(UnsafeBufferPointer(start: ptr, count: 2))
        if vec[0] != 0 || vec[1] != 0 { return false }
    }
    
    // Check Subbands
    let thPos = SIMD2<Int16>(repeating: threshold)
    let thNeg = SIMD2<Int16>(repeating: -threshold)

    let lowerHalfBase = base + 2 * 4
    for i in stride(from: 0, to: 8, by: 2) {
        let vec = SIMD2<Int16>(UnsafeBufferPointer(start: lowerHalfBase + i, count: 2))
        let mask = (vec .> thPos) .| (vec .< thNeg)
        if mask[0] || mask[1] { return false }
    }
    for y in 0..<2 {
        let ptr = base + y * 4 + 2
        let vec = SIMD2<Int16>(UnsafeBufferPointer(start: ptr, count: 2))
        let mask = (vec .> thPos) .| (vec .< thNeg)
        if mask[0] || mask[1] { return false }
    }

    let zeroVec = SIMD2<Int16>(repeating: 0)
    for i in stride(from: 0, to: 8, by: 2) {
        let ptr = UnsafeMutableRawPointer(lowerHalfBase + i).assumingMemoryBound(to: SIMD2<Int16>.self)
        ptr.pointee = zeroVec
    }
    for y in 0..<2 {
        let ptr = UnsafeMutableRawPointer(base + y * 4 + 2).assumingMemoryBound(to: SIMD2<Int16>.self)
        ptr.pointee = zeroVec
    }
    return true
}

#endif

// Scalar fallbacks (also used directly when SIMD is unavailable)
@inline(__always)
func isEffectivelyZero32_Scalar(data: UnsafeMutableBufferPointer<Int16>, threshold: Int16) -> Bool {
    guard let base = data.baseAddress else { return false }
    let thPos = threshold
    let thNeg = -threshold
    
    let lowerHalfBase = base + 16 * 32
    for i in 0..<512 {
        let val = lowerHalfBase[i]
        if thPos < val || val < thNeg { return false }
    }
    for y in 0..<16 {
        let rowPtr = base + y * 32 + 16
        for x in 0..<16 {
            let val = rowPtr[x]
            if thPos < val || val < thNeg { return false }
        }
    }
    
    for i in 0..<512 { lowerHalfBase[i] = 0 }
    for y in 0..<16 {
        let rowPtr = base + y * 32 + 16
        for x in 0..<16 { rowPtr[x] = 0 }
    }
    return true
}

@inline(__always)
func isEffectivelyZero16_Scalar(data: UnsafeMutableBufferPointer<Int16>, threshold: Int16) -> Bool {
    guard let base = data.baseAddress else { return false }
    let thPos = threshold
    let thNeg = -threshold
    
    let lowerHalfBase = base + 8 * 16
    for i in 0..<128 {
        let val = lowerHalfBase[i]
        if thPos < val || val < thNeg { return false }
    }
    for y in 0..<8 {
        let rowPtr = base + y * 16 + 8
        for x in 0..<8 {
            let val = rowPtr[x]
            if thPos < val || val < thNeg { return false }
        }
    }
    
    for i in 0..<128 { lowerHalfBase[i] = 0 }
    for y in 0..<8 {
        let rowPtr = base + y * 16 + 8
        for x in 0..<8 { rowPtr[x] = 0 }
    }
    return true
}

@inline(__always)
func isEffectivelyZero8_Scalar(data: UnsafeMutableBufferPointer<Int16>, threshold: Int16) -> Bool {
    guard let base = data.baseAddress else { return false }
    let thPos = threshold
    let thNeg = -threshold
    
    let lowerHalfBase = base + 4 * 8
    for i in 0..<32 {
        let val = lowerHalfBase[i]
        if thPos < val || val < thNeg { return false }
    }
    for y in 0..<4 {
        let rowPtr = base + y * 8 + 4
        for x in 0..<4 {
            let val = rowPtr[x]
            if thPos < val || val < thNeg { return false }
        }
    }
    
    for i in 0..<32 { lowerHalfBase[i] = 0 }
    for y in 0..<4 {
        let rowPtr = base + y * 8 + 4
        for x in 0..<4 { rowPtr[x] = 0 }
    }
    return true
}

@inline(__always)
func isEffectivelyZero4_Scalar(data: UnsafeMutableBufferPointer<Int16>, threshold: Int16) -> Bool {
    guard let base = data.baseAddress else { return false }
    let thPos = threshold
    let thNeg = -threshold
    
    let lowerHalfBase = base + 2 * 4
    for i in 0..<8 {
        let val = lowerHalfBase[i]
        if thPos < val || val < thNeg { return false }
    }
    for y in 0..<2 {
        let rowPtr = base + y * 4 + 2
        for x in 0..<2 {
            let val = rowPtr[x]
            if thPos < val || val < thNeg { return false }
        }
    }
    
    for i in 0..<8 { lowerHalfBase[i] = 0 }
    for y in 0..<2 {
        let rowPtr = base + y * 4 + 2
        for x in 0..<2 { rowPtr[x] = 0 }
    }
    return true
}

@inline(__always)
func isEffectivelyZeroBase4_Scalar(data: UnsafeMutableBufferPointer<Int16>, threshold: Int16) -> Bool {
    guard let base = data.baseAddress else { return false }
    for y in 0..<2 {
        let rowPtr = base + y * 4
        for x in 0..<2 {
            if rowPtr[x] != 0 { return false }
        }
    }

    let thPos = threshold
    let thNeg = -threshold
    let lowerHalfBase = base + 2 * 4
    for i in 0..<8 {
        let val = lowerHalfBase[i]
        if thPos < val || val < thNeg { return false }
    }
    for y in 0..<2 {
        let rowPtr = base + y * 4 + 2
        for x in 0..<2 {
            let val = rowPtr[x]
            if thPos < val || val < thNeg { return false }
        }
    }
    
    for i in 0..<8 { lowerHalfBase[i] = 0 }
    for y in 0..<2 {
        let rowPtr = base + y * 4 + 2
        for x in 0..<2 { rowPtr[x] = 0 }
    }
    return true
}

@inline(__always)
func isEffectivelyZero32(data: UnsafeMutableBufferPointer<Int16>, threshold: Int) -> Bool {
    #if arch(arm64) || arch(x86_64) || arch(wasm32)
    return isEffectivelyZero32_SIMD(data: data, threshold: Int16(threshold))
    #else
    return isEffectivelyZero32_Scalar(data: data, threshold: Int16(threshold))
    #endif
}

@inline(__always)
func isEffectivelyZero16(data: UnsafeMutableBufferPointer<Int16>, threshold: Int) -> Bool {
    #if arch(arm64) || arch(x86_64) || arch(wasm32)
    return isEffectivelyZero16_SIMD(data: data, threshold: Int16(threshold))
    #else
    return isEffectivelyZero16_Scalar(data: data, threshold: Int16(threshold))
    #endif
}

@inline(__always)
func isEffectivelyZero8(data: UnsafeMutableBufferPointer<Int16>, threshold: Int) -> Bool {
    #if arch(arm64) || arch(x86_64) || arch(wasm32)
    return isEffectivelyZero8_SIMD(data: data, threshold: Int16(threshold))
    #else
    return isEffectivelyZero8_Scalar(data: data, threshold: Int16(threshold))
    #endif
}

@inline(__always)
func isEffectivelyZero4(data: UnsafeMutableBufferPointer<Int16>, threshold: Int) -> Bool {
    #if arch(arm64) || arch(x86_64) || arch(wasm32)
    return isEffectivelyZero4_SIMD(data: data, threshold: Int16(threshold))
    #else
    return isEffectivelyZero4_Scalar(data: data, threshold: Int16(threshold))
    #endif
}

@inline(__always)
func isEffectivelyZeroBase4(data: UnsafeMutableBufferPointer<Int16>, threshold: Int) -> Bool {
    #if arch(arm64) || arch(x86_64) || arch(wasm32)
    return isEffectivelyZeroBase4_SIMD(data: data, threshold: Int16(threshold))
    #else
    return isEffectivelyZeroBase4_Scalar(data: data, threshold: Int16(threshold))
    #endif
}

@inline(__always)
func transformLayer32(block: inout Block2D, qt: QuantizationTable) {
    block.withView { view in
        var sub = dwt2d_32_sb(&view)
        quantizeMidSignedMapping(&sub.hl, qt: qt)
        quantizeMidSignedMapping(&sub.lh, qt: qt)
        quantizeHighSignedMapping(&sub.hh, qt: qt)
    }
}

@inline(__always)
func transformLayer16(block: inout Block2D, qt: QuantizationTable) {
    block.withView { view in
        var sub = dwt2d_16_sb(&view)
        quantizeMidSignedMapping(&sub.hl, qt: qt)
        quantizeMidSignedMapping(&sub.lh, qt: qt)
        quantizeHighSignedMapping(&sub.hh, qt: qt)
    }
}

@inline(__always)
func transformLayer8(block: inout Block2D, qt: QuantizationTable) {
    block.withView { view in
        var sub = dwt2d_8_sb(&view)
        quantizeMidSignedMapping(&sub.hl, qt: qt)
        quantizeMidSignedMapping(&sub.lh, qt: qt)
        quantizeHighSignedMapping(&sub.hh, qt: qt)
    }
}

@inline(__always)
func transformBase8(block: inout Block2D, qt: QuantizationTable) {
    block.withView { view in
        var sub = dwt2d_8_sb(&view)
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
        let isZero = blocks[i].data.withUnsafeMutableBufferPointer { ptr in
            return isEffectivelyZero32(data: ptr, threshold: zeroThreshold)
        }
        if isZero {
            bwFlags.writeBit(1)
        } else {
            bwFlags.writeBit(0)
            nonZeroIndices.append(i)
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
        let isZero = blocks[i].data.withUnsafeMutableBufferPointer { ptr in
            return isEffectivelyZero16(data: ptr, threshold: zeroThreshold)
        }
        if isZero {
            bwFlags.writeBit(1)
        } else {
            bwFlags.writeBit(0)
            nonZeroIndices.append(i)
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
        let isZero = blocks[i].data.withUnsafeMutableBufferPointer { ptr in
            return isEffectivelyZero8(data: ptr, threshold: zeroThreshold)
        }
        if isZero {
            bwFlags.writeBit(1)
        } else {
            bwFlags.writeBit(0)
            nonZeroIndices.append(i)
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
        let isZero = blocks[i].data.withUnsafeMutableBufferPointer { ptr in
            return isEffectivelyZeroBase4(data: ptr, threshold: zeroThreshold)
        }
        if isZero {
            bwFlags.writeBit(1)
        } else {
            bwFlags.writeBit(0)
            nonZeroIndices.append(i)
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
    let ptr1 = block.rowPointer(y: 1)
    let ptr2 = block.rowPointer(y: 2)
    let ptr3 = block.rowPointer(y: 3)
    
    @inline(__always)
    func errorMED(_ x: Int16, _ a: Int16, _ b: Int16, _ c: Int16) -> Int {
        let ia = Int(a), ib = Int(b), ic = Int(c)
        let predicted: Int
        if ia <= ic && ib <= ic {
            predicted = min(ia, ib)
        } else if ic <= ia && ic <= ib {
            predicted = max(ia, ib)
        } else {
            predicted = ia + ib - ic
        }
        return abs(Int(x) - predicted)
    }

    var sumDiffAbs = abs(Int(ptr0[0]) - Int(lastVal))
    sumDiffAbs += abs(Int(ptr0[1]) - Int(ptr0[0]))
    sumDiffAbs += abs(Int(ptr0[2]) - Int(ptr0[1]))
    sumDiffAbs += abs(Int(ptr0[3]) - Int(ptr0[2]))

    sumDiffAbs += abs(Int(ptr1[0]) - Int(ptr0[0]))
    sumDiffAbs += errorMED(ptr1[1], ptr1[0], ptr0[1], ptr0[0])
    sumDiffAbs += errorMED(ptr1[2], ptr1[1], ptr0[2], ptr0[1])
    sumDiffAbs += errorMED(ptr1[3], ptr1[2], ptr0[3], ptr0[2])
    
    sumDiffAbs += abs(Int(ptr2[0]) - Int(ptr1[0]))
    sumDiffAbs += errorMED(ptr2[1], ptr2[0], ptr1[1], ptr1[0])
    sumDiffAbs += errorMED(ptr2[2], ptr2[1], ptr1[2], ptr1[1])
    sumDiffAbs += errorMED(ptr2[3], ptr2[2], ptr1[3], ptr1[2])

    sumDiffAbs += abs(Int(ptr3[0]) - Int(ptr2[0]))
    sumDiffAbs += errorMED(ptr3[1], ptr3[0], ptr2[1], ptr2[0])
    sumDiffAbs += errorMED(ptr3[2], ptr3[1], ptr2[2], ptr2[1])
    sumDiffAbs += errorMED(ptr3[3], ptr3[2], ptr2[3], ptr2[2])

    lastVal = ptr3[3]
    
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
        return dwt2d_8_sb(&view)
    }
    
    quantizeLow(&sub.ll, qt: qt)
    quantizeMid(&sub.hl, qt: qt)
    quantizeMid(&sub.lh, qt: qt)
    quantizeHigh(&sub.hh, qt: qt)
    
    let isZero = block.data.withUnsafeMutableBufferPointer { ptr in
        return isEffectivelyZeroBase4(data: ptr, threshold: 0)
    }
    if isZero {
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
        if 0 < limit {
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
    
    @inline(__always)
    func readBlock(x: Int, y: Int, width blockWidth: Int, height blockHeight: Int, into view: inout BlockView) {
        data.withUnsafeBufferPointer { srcBuf in
            guard let srcBase = srcBuf.baseAddress else { return }
            
            for line in 0..<blockHeight {
                let currentY = y + line
                let safeY = min(currentY, self.height - 1)
                
                let dstPtr = view.rowPointer(y: line)
                let limit = min(blockWidth, self.width - x)
                
                if 0 < limit {
                    let srcPtr = srcBase.advanced(by: safeY * self.width + x)
                    dstPtr.update(from: srcPtr, count: limit)
                    
                    if limit < blockWidth {
                        let lastVal = dstPtr[limit - 1]
                        for i in limit..<blockWidth {
                            dstPtr[i] = lastVal
                        }
                    }
                } else {
                    let lastVal = srcBuf[safeY * self.width + (self.width - 1)]
                    for i in 0..<blockWidth {
                        dstPtr[i] = lastVal
                    }
                }
            }
        }
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
            print("\t\tUsing full planes for CbCr----------------------------------------------------------")
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
func subPlanes(curr: PlaneData420, predicted: PlaneData420) async -> PlaneData420 {
    @Sendable @inline(__always)
    func sub(c: [Int16], p: [Int16]) -> [Int16] {
        let count = c.count
        if count < 1 { return [] }

        var res = [Int16](repeating: 0, count: count)
        c.withUnsafeBufferPointer { cBuf in
            p.withUnsafeBufferPointer { pBuf in
                res.withUnsafeMutableBufferPointer { resBuf in
                    guard let cPtr = cBuf.baseAddress,
                          let pPtr = pBuf.baseAddress,
                          let resPtr = resBuf.baseAddress else { return }
                    
                    var i = 0
                    #if arch(arm64) || arch(x86_64) || arch(wasm32)
                    let chunk = 16
                    while i + chunk <= count {
                        let cSimd = UnsafeRawPointer(cPtr.advanced(by: i)).load(as: SIMD16<Int16>.self)
                        let pSimd = UnsafeRawPointer(pPtr.advanced(by: i)).load(as: SIMD16<Int16>.self)
                        let diffSimd = cSimd &- pSimd
                        UnsafeMutableRawPointer(resPtr.advanced(by: i)).storeBytes(of: diffSimd, as: SIMD16<Int16>.self)
                        i += chunk
                    }
                    #endif
                    
                    while i < count {
                        resPtr[i] = cPtr[i] &- pPtr[i]
                        i += 1
                    }
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
    @Sendable @inline(__always)
    func add(r: [Int16], p: [Int16]) -> [Int16] {
        let count = r.count
        if count < 1 { return [] }

        var curr = [Int16](repeating: 0, count: count)
        r.withUnsafeBufferPointer { rBuf in
            p.withUnsafeBufferPointer { pBuf in
                curr.withUnsafeMutableBufferPointer { cBuf in
                    guard let rPtr = rBuf.baseAddress,
                          let pPtr = pBuf.baseAddress,
                          let cPtr = cBuf.baseAddress else { return }
                    
                    var i = 0
                    #if arch(arm64) || arch(x86_64) || arch(wasm32)
                    let chunk = 16
                    while i + chunk <= count {
                        let rSimd = UnsafeRawPointer(rPtr.advanced(by: i)).load(as: SIMD16<Int16>.self)
                        let pSimd = UnsafeRawPointer(pPtr.advanced(by: i)).load(as: SIMD16<Int16>.self)
                        let sumSimd = rSimd &+ pSimd
                        UnsafeMutableRawPointer(cPtr.advanced(by: i)).storeBytes(of: sumSimd, as: SIMD16<Int16>.self)
                        i += chunk
                    }
                    #endif
                    
                    while i < count {
                        cPtr[i] = rPtr[i] &+ pPtr[i]
                        i += 1
                    }
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
            let res = await subPlanes(curr: curr, predicted: predicted)
            
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
