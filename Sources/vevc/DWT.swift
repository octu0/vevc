// MARK: - DWT

struct Subbands {
    var ll: BlockView
    var hl: BlockView
    var lh: BlockView
    var hh: BlockView
    let size: Int
}

@inline(__always)
private func makeSubbands(base: UnsafeMutablePointer<Int16>, size: Int, stride: Int) -> Subbands {
    let half = size / 2
    return Subbands(
        ll: BlockView(base: base, width: half, height: half, stride: stride),
        hl: BlockView(base: base.advanced(by: half), width: half, height: half, stride: stride),
        lh: BlockView(base: base.advanced(by: half * stride), width: half, height: half, stride: stride),
        hh: BlockView(base: base.advanced(by: half * stride + half), width: half, height: half, stride: stride),
        size: half
    )
}

// MARK: - LeGall 5/3 Lifting
//
// All lift53/inverseLift53 functions are optimized for stride=1 (contiguous memory).
// Column processing uses transpose->row_lift->transpose_back pattern in dwt2d functions,
// so stride is always 1 when these functions are called.

// stride=1 optimized: contiguous SIMD load/store

@inline(__always)
func lift53Block8(_ buffer: UnsafeMutableBufferPointer<Int16>, stride: Int) {
    let base = buffer.baseAddress!
    let raw = UnsafeRawPointer(base).loadUnaligned(as: SIMD8<Int16>.self)
    var low = raw.evenHalf
    var high = raw.oddHalf

    let lowShifted = SIMD4<Int16>(low[1], low[2], low[3], low[3])
    let ditherPredict = SIMD4<Int16>(0, 1, 0, 1)
    high &-= (low &+ lowShifted &+ ditherPredict) &>> 1

    let highShifted = SIMD4<Int16>(high[0], high[0], high[1], high[2])
    let ditherUpdate = SIMD4<Int16>(1, 2, 1, 2)
    low &+= (highShifted &+ high &+ ditherUpdate) &>> 2

    let result = SIMD8<Int16>(lowHalf: low, highHalf: high)
    UnsafeMutableRawPointer(base).storeBytes(of: result, as: SIMD8<Int16>.self)
}

@inline(__always)
func lift53Block16(_ buffer: UnsafeMutableBufferPointer<Int16>, stride: Int) {
    let base = buffer.baseAddress!
    let raw = UnsafeRawPointer(base).loadUnaligned(as: SIMD16<Int16>.self)
    var low = raw.evenHalf
    var high = raw.oddHalf

    let lowShifted = SIMD8<Int16>(low[1], low[2], low[3], low[4], low[5], low[6], low[7], low[7])
    let ditherPredictVec = SIMD8<Int16>(0, 1, 0, 1, 0, 1, 0, 1)
    high &-= (low &+ lowShifted &+ ditherPredictVec) &>> 1

    let highShifted = SIMD8<Int16>(high[0], high[0], high[1], high[2], high[3], high[4], high[5], high[6])
    let ditherUpdateVec = SIMD8<Int16>(1, 2, 1, 2, 1, 2, 1, 2)
    low &+= (highShifted &+ high &+ ditherUpdateVec) &>> 2

    let result = SIMD16<Int16>(lowHalf: low, highHalf: high)
    UnsafeMutableRawPointer(base).storeBytes(of: result, as: SIMD16<Int16>.self)
}

@inline(__always)
func lift53Block32(_ buffer: UnsafeMutableBufferPointer<Int16>, stride: Int) {
    let base = buffer.baseAddress!
    let raw = UnsafeRawPointer(base).loadUnaligned(as: SIMD32<Int16>.self)
    var low = raw.evenHalf
    var high = raw.oddHalf

    let lowShifted = SIMD16<Int16>(
        low[1], low[2], low[3], low[4], low[5], low[6], low[7], low[8],
        low[9], low[10], low[11], low[12], low[13], low[14], low[15], low[15]
    )
    let ditherPredict = SIMD16<Int16>(
        0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1
    )
    high &-= (low &+ lowShifted &+ ditherPredict) &>> 1

    let highShifted = SIMD16<Int16>(
        high[0], high[0], high[1], high[2], high[3], high[4], high[5], high[6],
        high[7], high[8], high[9], high[10], high[11], high[12], high[13], high[14]
    )
    let ditherUpdate = SIMD16<Int16>(
        1, 2, 1, 2, 1, 2, 1, 2,
        1, 2, 1, 2, 1, 2, 1, 2
    )
    low &+= (highShifted &+ high &+ ditherUpdate) &>> 2

    let result = SIMD32<Int16>(lowHalf: low, highHalf: high)
    UnsafeMutableRawPointer(base).storeBytes(of: result, as: SIMD32<Int16>.self)
}

@inline(__always)
func inverseLift53Block8(base: UnsafeMutablePointer<Int16>) {
    let raw = UnsafeRawPointer(base).loadUnaligned(as: SIMD8<Int16>.self)
    var low = raw.lowHalf
    var high = raw.highHalf

    let highShifted = SIMD4<Int16>(high[0], high[0], high[1], high[2])
    let ditherUpdate = SIMD4<Int16>(1, 2, 1, 2)
    low &-= (highShifted &+ high &+ ditherUpdate) &>> 2

    let lowShifted = SIMD4<Int16>(low[1], low[2], low[3], low[3])
    let ditherPredict = SIMD4<Int16>(0, 1, 0, 1)
    high &+= (low &+ lowShifted &+ ditherPredict) &>> 1

    var result = SIMD8<Int16>()
    result.evenHalf = low
    result.oddHalf = high
    UnsafeMutableRawPointer(base).storeBytes(of: result, as: SIMD8<Int16>.self)
}

@inline(__always)
func inverseLift53Block16(base: UnsafeMutablePointer<Int16>) {
    let raw = UnsafeRawPointer(base).loadUnaligned(as: SIMD16<Int16>.self)
    var low = raw.lowHalf
    var high = raw.highHalf

    let highShifted = SIMD8<Int16>(high[0], high[0], high[1], high[2], high[3], high[4], high[5], high[6])
    let ditherUpdate = SIMD8<Int16>(1, 2, 1, 2, 1, 2, 1, 2)
    low &-= (highShifted &+ high &+ ditherUpdate) &>> 2

    let lowShifted = SIMD8<Int16>(low[1], low[2], low[3], low[4], low[5], low[6], low[7], low[7])
    let ditherPredict = SIMD8<Int16>(0, 1, 0, 1, 0, 1, 0, 1)
    high &+= (low &+ lowShifted &+ ditherPredict) &>> 1

    var result = SIMD16<Int16>()
    result.evenHalf = low
    result.oddHalf = high
    UnsafeMutableRawPointer(base).storeBytes(of: result, as: SIMD16<Int16>.self)
}

@inline(__always)
func inverseLift53Block32(base: UnsafeMutablePointer<Int16>) {
    let raw = UnsafeRawPointer(base).loadUnaligned(as: SIMD32<Int16>.self)
    var low = raw.lowHalf
    var high = raw.highHalf

    let highShifted = SIMD16<Int16>(
        high[0], high[0], high[1], high[2], high[3], high[4], high[5], high[6],
        high[7], high[8], high[9], high[10], high[11], high[12], high[13], high[14]
    )
    let ditherUpdate = SIMD16<Int16>(
        1, 2, 1, 2, 1, 2, 1, 2,
        1, 2, 1, 2, 1, 2, 1, 2
    )
    low &-= (highShifted &+ high &+ ditherUpdate) &>> 2

    let lowShifted = SIMD16<Int16>(
        low[1], low[2], low[3], low[4], low[5], low[6], low[7], low[8],
        low[9], low[10], low[11], low[12], low[13], low[14], low[15], low[15]
    )
    let ditherPredict = SIMD16<Int16>(
        0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1
    )
    high &+= (low &+ lowShifted &+ ditherPredict) &>> 1

    var result = SIMD32<Int16>()
    result.evenHalf = low
    result.oddHalf = high
    UnsafeMutableRawPointer(base).storeBytes(of: result, as: SIMD32<Int16>.self)
}

// MARK: - Vertical Lifting

@inline(__always)
func lift53Block8Vertical(_ base: UnsafeMutablePointer<Int16>, stride: Int) {
    let r0 = UnsafeRawPointer(base + 0 * stride).loadUnaligned(as: SIMD8<Int16>.self)
    var r1 = UnsafeRawPointer(base + 1 * stride).loadUnaligned(as: SIMD8<Int16>.self)
    let r2 = UnsafeRawPointer(base + 2 * stride).loadUnaligned(as: SIMD8<Int16>.self)
    var r3 = UnsafeRawPointer(base + 3 * stride).loadUnaligned(as: SIMD8<Int16>.self)
    let r4 = UnsafeRawPointer(base + 4 * stride).loadUnaligned(as: SIMD8<Int16>.self)
    var r5 = UnsafeRawPointer(base + 5 * stride).loadUnaligned(as: SIMD8<Int16>.self)
    let r6 = UnsafeRawPointer(base + 6 * stride).loadUnaligned(as: SIMD8<Int16>.self)
    var r7 = UnsafeRawPointer(base + 7 * stride).loadUnaligned(as: SIMD8<Int16>.self)

    r1 &-= (r0 &+ r2) &>> 1
    r3 &-= (r2 &+ r4 &+ SIMD8<Int16>(repeating: 1)) &>> 1
    r5 &-= (r4 &+ r6) &>> 1
    r7 &-= (r6 &+ r6 &+ SIMD8<Int16>(repeating: 1)) &>> 1

    var l0 = r0, l2 = r2, l4 = r4, l6 = r6
    l0 &+= (r1 &+ r1 &+ SIMD8<Int16>(repeating: 1)) &>> 2
    l2 &+= (r1 &+ r3 &+ SIMD8<Int16>(repeating: 2)) &>> 2
    l4 &+= (r3 &+ r5 &+ SIMD8<Int16>(repeating: 1)) &>> 2
    l6 &+= (r5 &+ r7 &+ SIMD8<Int16>(repeating: 2)) &>> 2

    UnsafeMutableRawPointer(base + 0 * stride).storeBytes(of: l0, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 1 * stride).storeBytes(of: l2, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 2 * stride).storeBytes(of: l4, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 3 * stride).storeBytes(of: l6, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 4 * stride).storeBytes(of: r1, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 5 * stride).storeBytes(of: r3, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 6 * stride).storeBytes(of: r5, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 7 * stride).storeBytes(of: r7, as: SIMD8<Int16>.self)
}

@inline(__always)
func inverseLift53Block8Vertical(_ base: UnsafeMutablePointer<Int16>, stride: Int) {
    var l0 = UnsafeRawPointer(base + 0 * stride).loadUnaligned(as: SIMD8<Int16>.self)
    var l1 = UnsafeRawPointer(base + 1 * stride).loadUnaligned(as: SIMD8<Int16>.self)
    var l2 = UnsafeRawPointer(base + 2 * stride).loadUnaligned(as: SIMD8<Int16>.self)
    var l3 = UnsafeRawPointer(base + 3 * stride).loadUnaligned(as: SIMD8<Int16>.self)

    var h0 = UnsafeRawPointer(base + 4 * stride).loadUnaligned(as: SIMD8<Int16>.self)
    var h1 = UnsafeRawPointer(base + 5 * stride).loadUnaligned(as: SIMD8<Int16>.self)
    var h2 = UnsafeRawPointer(base + 6 * stride).loadUnaligned(as: SIMD8<Int16>.self)
    var h3 = UnsafeRawPointer(base + 7 * stride).loadUnaligned(as: SIMD8<Int16>.self)

    l0 &-= (h0 &+ h0 &+ SIMD8<Int16>(repeating: 1)) &>> 2
    l1 &-= (h0 &+ h1 &+ SIMD8<Int16>(repeating: 2)) &>> 2
    l2 &-= (h1 &+ h2 &+ SIMD8<Int16>(repeating: 1)) &>> 2
    l3 &-= (h2 &+ h3 &+ SIMD8<Int16>(repeating: 2)) &>> 2

    h0 &+= (l0 &+ l1) &>> 1
    h1 &+= (l1 &+ l2 &+ SIMD8<Int16>(repeating: 1)) &>> 1
    h2 &+= (l2 &+ l3) &>> 1
    h3 &+= (l3 &+ l3 &+ SIMD8<Int16>(repeating: 1)) &>> 1

    UnsafeMutableRawPointer(base + 0 * stride).storeBytes(of: l0, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 1 * stride).storeBytes(of: h0, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 2 * stride).storeBytes(of: l1, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 3 * stride).storeBytes(of: h1, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 4 * stride).storeBytes(of: l2, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 5 * stride).storeBytes(of: h2, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 6 * stride).storeBytes(of: l3, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 7 * stride).storeBytes(of: h3, as: SIMD8<Int16>.self)
}

@inline(__always)
func lift53Block16Vertical(_ base: UnsafeMutablePointer<Int16>, stride s: Int) {
    withUnsafeTemporaryAllocation(of: SIMD16<Int16>.self, capacity: 16) { tmp in
        for y in 0..<16 {
            tmp[y] = UnsafeRawPointer(base + y * s).loadUnaligned(as: SIMD16<Int16>.self)
        }
        for y in Swift.stride(from: 1, to: 15, by: 2) {
            let dither = SIMD16<Int16>(repeating: Int16((y / 2) & 1))
            tmp[y] &-= (tmp[y - 1] &+ tmp[y + 1] &+ dither) &>> 1
        }
        tmp[15] &-= (tmp[14] &+ tmp[14] &+ SIMD16<Int16>(repeating: 1)) &>> 1
        
        tmp[0] &+= (tmp[1] &+ tmp[1] &+ SIMD16<Int16>(repeating: 1)) &>> 2
        for y in Swift.stride(from: 2, to: 16, by: 2) {
            let dither = SIMD16<Int16>(repeating: Int16(((y / 2) & 1) == 0 ? 1 : 2))
            tmp[y] &+= (tmp[y - 1] &+ tmp[y + 1] &+ dither) &>> 2
        }
        
        for y in 0..<8 {
            UnsafeMutableRawPointer(base + y * s).storeBytes(of: tmp[y * 2], as: SIMD16<Int16>.self)
            UnsafeMutableRawPointer(base + (y + 8) * s).storeBytes(of: tmp[y * 2 + 1], as: SIMD16<Int16>.self)
        }
    }
}

@inline(__always)
func inverseLift53Block16Vertical(_ base: UnsafeMutablePointer<Int16>, stride s: Int) {
    withUnsafeTemporaryAllocation(of: SIMD16<Int16>.self, capacity: 16) { tmp in
        for y in 0..<8 {
            tmp[y * 2] = UnsafeRawPointer(base + y * s).loadUnaligned(as: SIMD16<Int16>.self)
            tmp[y * 2 + 1] = UnsafeRawPointer(base + (y + 8) * s).loadUnaligned(as: SIMD16<Int16>.self)
        }
        
        tmp[0] &-= (tmp[1] &+ tmp[1] &+ SIMD16<Int16>(repeating: 1)) &>> 2
        for y in Swift.stride(from: 2, to: 16, by: 2) {
            let dither = SIMD16<Int16>(repeating: Int16(((y / 2) & 1) == 0 ? 1 : 2))
            tmp[y] &-= (tmp[y - 1] &+ tmp[y + 1] &+ dither) &>> 2
        }
        
        for y in Swift.stride(from: 1, to: 15, by: 2) {
            let dither = SIMD16<Int16>(repeating: Int16((y / 2) & 1))
            tmp[y] &+= (tmp[y - 1] &+ tmp[y + 1] &+ dither) &>> 1
        }
        tmp[15] &+= (tmp[14] &+ tmp[14] &+ SIMD16<Int16>(repeating: 1)) &>> 1
        
        for y in 0..<16 {
            UnsafeMutableRawPointer(base + y * s).storeBytes(of: tmp[y], as: SIMD16<Int16>.self)
        }
    }
}

@inline(__always)
func lift53Block32Vertical(_ base: UnsafeMutablePointer<Int16>, stride s: Int) {
    withUnsafeTemporaryAllocation(of: SIMD16<Int16>.self, capacity: 64) { tmp in
        for y in 0..<32 {
            tmp[y * 2] = UnsafeRawPointer(base + y * s).loadUnaligned(as: SIMD16<Int16>.self)
            tmp[y * 2 + 1] = UnsafeRawPointer(base + y * s + 16).loadUnaligned(as: SIMD16<Int16>.self)
        }
        
        for y in Swift.stride(from: 1, to: 31, by: 2) {
            let d = SIMD16<Int16>(repeating: Int16((y / 2) & 1))
            tmp[y * 2] &-= (tmp[(y - 1) * 2] &+ tmp[(y + 1) * 2] &+ d) &>> 1
            tmp[y * 2 + 1] &-= (tmp[(y - 1) * 2 + 1] &+ tmp[(y + 1) * 2 + 1] &+ d) &>> 1
        }
        let d31 = SIMD16<Int16>(repeating: 1)
        tmp[31 * 2] &-= (tmp[30 * 2] &+ tmp[30 * 2] &+ d31) &>> 1
        tmp[31 * 2 + 1] &-= (tmp[30 * 2 + 1] &+ tmp[30 * 2 + 1] &+ d31) &>> 1
        
        let d0 = SIMD16<Int16>(repeating: 1)
        tmp[0] &+= (tmp[1 * 2] &+ tmp[1 * 2] &+ d0) &>> 2
        tmp[1] &+= (tmp[1 * 2 + 1] &+ tmp[1 * 2 + 1] &+ d0) &>> 2
        for y in Swift.stride(from: 2, to: 32, by: 2) {
            let d = SIMD16<Int16>(repeating: Int16(((y / 2) & 1) == 0 ? 1 : 2))
            tmp[y * 2] &+= (tmp[(y - 1) * 2] &+ tmp[(y + 1) * 2] &+ d) &>> 2
            tmp[y * 2 + 1] &+= (tmp[(y - 1) * 2 + 1] &+ tmp[(y + 1) * 2 + 1] &+ d) &>> 2
        }
        
        for y in 0..<16 {
            UnsafeMutableRawPointer(base + y * s).storeBytes(of: tmp[(y * 2) * 2], as: SIMD16<Int16>.self)
            UnsafeMutableRawPointer(base + y * s + 16).storeBytes(of: tmp[(y * 2) * 2 + 1], as: SIMD16<Int16>.self)
            
            UnsafeMutableRawPointer(base + (y + 16) * s).storeBytes(of: tmp[(y * 2 + 1) * 2], as: SIMD16<Int16>.self)
            UnsafeMutableRawPointer(base + (y + 16) * s + 16).storeBytes(of: tmp[(y * 2 + 1) * 2 + 1], as: SIMD16<Int16>.self)
        }
    }
}

@inline(__always)
func inverseLift53Block32Vertical(_ base: UnsafeMutablePointer<Int16>, stride s: Int) {
    withUnsafeTemporaryAllocation(of: SIMD16<Int16>.self, capacity: 64) { tmp in
        for y in 0..<16 {
            tmp[(y * 2) * 2] = UnsafeRawPointer(base + y * s).loadUnaligned(as: SIMD16<Int16>.self)
            tmp[(y * 2) * 2 + 1] = UnsafeRawPointer(base + y * s + 16).loadUnaligned(as: SIMD16<Int16>.self)
            
            tmp[(y * 2 + 1) * 2] = UnsafeRawPointer(base + (y + 16) * s).loadUnaligned(as: SIMD16<Int16>.self)
            tmp[(y * 2 + 1) * 2 + 1] = UnsafeRawPointer(base + (y + 16) * s + 16).loadUnaligned(as: SIMD16<Int16>.self)
        }
        
        let d0 = SIMD16<Int16>(repeating: 1)
        tmp[0] &-= (tmp[1 * 2] &+ tmp[1 * 2] &+ d0) &>> 2
        tmp[1] &-= (tmp[1 * 2 + 1] &+ tmp[1 * 2 + 1] &+ d0) &>> 2
        for y in Swift.stride(from: 2, to: 32, by: 2) {
            let d = SIMD16<Int16>(repeating: Int16(((y / 2) & 1) == 0 ? 1 : 2))
            tmp[y * 2] &-= (tmp[(y - 1) * 2] &+ tmp[(y + 1) * 2] &+ d) &>> 2
            tmp[y * 2 + 1] &-= (tmp[(y - 1) * 2 + 1] &+ tmp[(y + 1) * 2 + 1] &+ d) &>> 2
        }
        
        for y in Swift.stride(from: 1, to: 31, by: 2) {
            let d = SIMD16<Int16>(repeating: Int16((y / 2) & 1))
            tmp[y * 2] &+= (tmp[(y - 1) * 2] &+ tmp[(y + 1) * 2] &+ d) &>> 1
            tmp[y * 2 + 1] &+= (tmp[(y - 1) * 2 + 1] &+ tmp[(y + 1) * 2 + 1] &+ d) &>> 1
        }
        let d31 = SIMD16<Int16>(repeating: 1)
        tmp[31 * 2] &+= (tmp[30 * 2] &+ tmp[30 * 2] &+ d31) &>> 1
        tmp[31 * 2 + 1] &+= (tmp[30 * 2 + 1] &+ tmp[30 * 2 + 1] &+ d31) &>> 1
        
        for y in 0..<32 {
            UnsafeMutableRawPointer(base + y * s).storeBytes(of: tmp[y * 2], as: SIMD16<Int16>.self)
            UnsafeMutableRawPointer(base + y * s + 16).storeBytes(of: tmp[y * 2 + 1], as: SIMD16<Int16>.self)
        }
    }
}

// MARK: - LL-only Vertical Lifting
//
// The L0-loop analysis (llAnalyzeLevel) reads ONLY the LL quadrant, so the
// vertical stage can drop the high-pass column half (never read by LL) and
// the high-row outputs (never stored). The arithmetic on the retained lanes
// is identical to the full vertical lifting — same operands, same dithers —
// so the LL bytes are bit-identical to dwt2DBlock32/16 followed by a gather.

@inline(__always)
func lift53Block32VerticalLLOnly(_ base: UnsafeMutablePointer<Int16>, stride s: Int) {
    withUnsafeTemporaryAllocation(of: SIMD16<Int16>.self, capacity: 32) { tmp in
        for y in 0..<32 {
            tmp[y] = UnsafeRawPointer(base + y * s).loadUnaligned(as: SIMD16<Int16>.self)
        }

        for y in Swift.stride(from: 1, to: 31, by: 2) {
            let d = SIMD16<Int16>(repeating: Int16((y / 2) & 1))
            tmp[y] &-= (tmp[y - 1] &+ tmp[y + 1] &+ d) &>> 1
        }
        let d31 = SIMD16<Int16>(repeating: 1)
        tmp[31] &-= (tmp[30] &+ tmp[30] &+ d31) &>> 1

        let d0 = SIMD16<Int16>(repeating: 1)
        tmp[0] &+= (tmp[1] &+ tmp[1] &+ d0) &>> 2
        for y in Swift.stride(from: 2, to: 32, by: 2) {
            let d = SIMD16<Int16>(repeating: Int16(((y / 2) & 1) == 0 ? 1 : 2))
            tmp[y] &+= (tmp[y - 1] &+ tmp[y + 1] &+ d) &>> 2
        }

        for y in 0..<16 {
            UnsafeMutableRawPointer(base + y * s).storeBytes(of: tmp[y * 2], as: SIMD16<Int16>.self)
        }
    }
}

@inline(__always)
func lift53Block16VerticalLLOnly(_ base: UnsafeMutablePointer<Int16>, stride s: Int) {
    withUnsafeTemporaryAllocation(of: SIMD8<Int16>.self, capacity: 16) { tmp in
        for y in 0..<16 {
            tmp[y] = UnsafeRawPointer(base + y * s).loadUnaligned(as: SIMD8<Int16>.self)
        }

        for y in Swift.stride(from: 1, to: 15, by: 2) {
            let dither = SIMD8<Int16>(repeating: Int16((y / 2) & 1))
            tmp[y] &-= (tmp[y - 1] &+ tmp[y + 1] &+ dither) &>> 1
        }
        tmp[15] &-= (tmp[14] &+ tmp[14] &+ SIMD8<Int16>(repeating: 1)) &>> 1

        tmp[0] &+= (tmp[1] &+ tmp[1] &+ SIMD8<Int16>(repeating: 1)) &>> 2
        for y in Swift.stride(from: 2, to: 16, by: 2) {
            let dither = SIMD8<Int16>(repeating: Int16(((y / 2) & 1) == 0 ? 1 : 2))
            tmp[y] &+= (tmp[y - 1] &+ tmp[y + 1] &+ dither) &>> 2
        }

        for y in 0..<8 {
            UnsafeMutableRawPointer(base + y * s).storeBytes(of: tmp[y * 2], as: SIMD8<Int16>.self)
        }
    }
}

// MARK: - 2D DWT (Transpose-optimized)
//
// Strategy: rows first (contiguous memory, stride=1), then transpose,
// apply column lifting as row lifting (contiguous), then transpose back.
// This eliminates stride-based gather/scatter in column processing,
// converting all SIMD operations to contiguous memory access.

@inline(__always)
func dwt2DBlock8(_ block: BlockView) {
    let base = block.base
    let width = block.stride
    // Row lifting (stride=1, contiguous)
    lift53Block8(UnsafeMutableBufferPointer(start: base + (0 * width), count: 8), stride: 1)
    lift53Block8(UnsafeMutableBufferPointer(start: base + (1 * width), count: 8), stride: 1)
    lift53Block8(UnsafeMutableBufferPointer(start: base + (2 * width), count: 8), stride: 1)
    lift53Block8(UnsafeMutableBufferPointer(start: base + (3 * width), count: 8), stride: 1)
    lift53Block8(UnsafeMutableBufferPointer(start: base + (4 * width), count: 8), stride: 1)
    lift53Block8(UnsafeMutableBufferPointer(start: base + (5 * width), count: 8), stride: 1)
    lift53Block8(UnsafeMutableBufferPointer(start: base + (6 * width), count: 8), stride: 1)
    lift53Block8(UnsafeMutableBufferPointer(start: base + (7 * width), count: 8), stride: 1)
    // Vertical lifting directly without transpose
    lift53Block8Vertical(base, stride: width)
}

@inline(__always)
func dwt2DBlock8Subbands(_ block: BlockView) -> Subbands {
    dwt2DBlock8(block)
    return makeSubbands(base: block.base, size: 8, stride: block.stride)
}

@inline(__always)
func dwt2DBlock16(_ block: BlockView) {
    let base = block.base
    let width = block.stride
    for y in 0..<16 {
        lift53Block16(UnsafeMutableBufferPointer(start: base + (y * width), count: 16), stride: 1)
    }
    // Vertical lifting directly without transpose
    lift53Block16Vertical(base, stride: width)
}

@inline(__always)
func dwt2DBlock32(_ block: BlockView) {
    let base = block.base
    let width = block.stride
    // Row lifting (stride=1, contiguous)
    for y in 0..<32 {
        lift53Block32(UnsafeMutableBufferPointer(start: base + (y * width), count: 32), stride: 1)
    }
    // Vertical lifting directly without transpose
    lift53Block32Vertical(base, stride: width)
}

/// dwt2DBlock32 keeping only the LL quadrant valid (upper-left 16×16, bit-
/// identical to the full transform there). The row stage is unchanged — LL
/// columns depend on the row high-pass — and the vertical stage is LL-only.
/// The other three quadrants are left unspecified.
@inline(__always)
func dwt2DBlock32LL(_ block: BlockView) {
    let base = block.base
    let width = block.stride
    for y in 0..<32 {
        lift53Block32(UnsafeMutableBufferPointer(start: base + (y * width), count: 32), stride: 1)
    }
    lift53Block32VerticalLLOnly(base, stride: width)
}

/// dwt2DBlock16 keeping only the LL quadrant valid (upper-left 8×8, bit-
/// identical to the full transform there).
@inline(__always)
func dwt2DBlock16LL(_ block: BlockView) {
    let base = block.base
    let width = block.stride
    for y in 0..<16 {
        lift53Block16(UnsafeMutableBufferPointer(start: base + (y * width), count: 16), stride: 1)
    }
    lift53Block16VerticalLLOnly(base, stride: width)
}

@inline(__always)
func inverseDWT2DBlock8(ptr base: UnsafeMutablePointer<Int16>, stride width: Int) {
    // Inverse vertical lifting directly
    inverseLift53Block8Vertical(base, stride: width)

    // Inverse row
    inverseLift53Block8(base: base + (0 * width))
    inverseLift53Block8(base: base + (1 * width))
    inverseLift53Block8(base: base + (2 * width))
    inverseLift53Block8(base: base + (3 * width))
    inverseLift53Block8(base: base + (4 * width))
    inverseLift53Block8(base: base + (5 * width))
    inverseLift53Block8(base: base + (6 * width))
    inverseLift53Block8(base: base + (7 * width))
}

@inline(__always)
func inverseDWT2DBlock16(ptr base: UnsafeMutablePointer<Int16>, stride width: Int) {
    // Inverse vertical lifting directly
    inverseLift53Block16Vertical(base, stride: width)

    // Inverse row
    for y in 0..<16 {
        inverseLift53Block16(base: base + (y * width))
    }
}

@inline(__always)
func inverseDWT2DBlock32(ptr base: UnsafeMutablePointer<Int16>, stride width: Int) {
    // Inverse vertical lifting directly
    inverseLift53Block32Vertical(base, stride: width)

    // Inverse row
    for y in 0..<32 {
        inverseLift53Block32(base: base + (y * width))
    }
}