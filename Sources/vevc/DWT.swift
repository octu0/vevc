// MARK: - DWT

// MARK: - DWT Structures

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

@inline(__always)
func lift53Block4(_ buffer: UnsafeMutableBufferPointer<Int16>, stride: Int) {
    var low = SIMD2<Int16>(buffer[0 * stride], buffer[2 * stride])
    var high = SIMD2<Int16>(buffer[1 * stride], buffer[3 * stride])

    let lowShifted = SIMD2<Int16>(low[1], low[1])
    high &-= (low &+ lowShifted) &>> 1

    let highShifted = SIMD2<Int16>(high[0], high[0])
    low &+= (highShifted &+ high &+ 2) &>> 2

    buffer[0 * stride] = low[0]; buffer[1 * stride] = low[1]
    buffer[2 * stride] = high[0]; buffer[3 * stride] = high[1]
}

@inline(__always)
func inverseLift53Block4(_ buffer: UnsafeMutableBufferPointer<Int16>, stride: Int) {
    var low = SIMD2<Int16>(buffer[0 * stride], buffer[1 * stride])
    var high = SIMD2<Int16>(buffer[2 * stride], buffer[3 * stride])

    let highShifted = SIMD2<Int16>(high[0], high[0])
    low &-= (highShifted &+ high &+ 2) &>> 2

    let lowShifted = SIMD2<Int16>(low[1], low[1])
    high &+= (low &+ lowShifted) &>> 1

    buffer[0 * stride] = low[0]; buffer[1 * stride] = high[0]
    buffer[2 * stride] = low[1]; buffer[3 * stride] = high[1]
}

// stride=1 optimized: contiguous SIMD load/store

@inline(__always)
func lift53Block8(_ buffer: UnsafeMutableBufferPointer<Int16>, stride: Int) {
    guard let base = buffer.baseAddress else { return }
    let raw = UnsafeRawPointer(base).loadUnaligned(as: SIMD8<Int16>.self)
    var low = SIMD4<Int16>(raw[0], raw[2], raw[4], raw[6])
    var high = SIMD4<Int16>(raw[1], raw[3], raw[5], raw[7])

    let lowShifted = SIMD4<Int16>(low[1], low[2], low[3], low[3])
    high &-= (low &+ lowShifted) &>> 1

    let highShifted = SIMD4<Int16>(high[0], high[0], high[1], high[2])
    low &+= (highShifted &+ high &+ 2) &>> 2

    let result = SIMD8<Int16>(low[0], low[1], low[2], low[3], high[0], high[1], high[2], high[3])
    UnsafeMutableRawPointer(base).storeBytes(of: result, as: SIMD8<Int16>.self)
}

@inline(__always)
func lift53Block16(_ buffer: UnsafeMutableBufferPointer<Int16>, stride: Int) {
    guard let base = buffer.baseAddress else { return }
    let raw0 = UnsafeRawPointer(base).loadUnaligned(as: SIMD8<Int16>.self)
    let raw1 = UnsafeRawPointer(base + 8).loadUnaligned(as: SIMD8<Int16>.self)
    var low = SIMD8<Int16>(raw0[0], raw0[2], raw0[4], raw0[6], raw1[0], raw1[2], raw1[4], raw1[6])
    var high = SIMD8<Int16>(raw0[1], raw0[3], raw0[5], raw0[7], raw1[1], raw1[3], raw1[5], raw1[7])

    let lowShifted = SIMD8<Int16>(low[1], low[2], low[3], low[4], low[5], low[6], low[7], low[7])
    high &-= (low &+ lowShifted) &>> 1

    let highShifted = SIMD8<Int16>(high[0], high[0], high[1], high[2], high[3], high[4], high[5], high[6])
    low &+= (highShifted &+ high &+ 2) &>> 2

    UnsafeMutableRawPointer(base).storeBytes(of: low, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 8).storeBytes(of: high, as: SIMD8<Int16>.self)
}

@inline(__always)
func lift53Block32(_ buffer: UnsafeMutableBufferPointer<Int16>, stride: Int) {
    guard let base = buffer.baseAddress else { return }
    let raw0 = UnsafeRawPointer(base).loadUnaligned(as: SIMD8<Int16>.self)
    let raw1 = UnsafeRawPointer(base + 8).loadUnaligned(as: SIMD8<Int16>.self)
    let raw2 = UnsafeRawPointer(base + 16).loadUnaligned(as: SIMD8<Int16>.self)
    let raw3 = UnsafeRawPointer(base + 24).loadUnaligned(as: SIMD8<Int16>.self)
    var low = SIMD16<Int16>(
        raw0[0], raw0[2], raw0[4], raw0[6], raw1[0], raw1[2], raw1[4], raw1[6],
        raw2[0], raw2[2], raw2[4], raw2[6], raw3[0], raw3[2], raw3[4], raw3[6]
    )
    var high = SIMD16<Int16>(
        raw0[1], raw0[3], raw0[5], raw0[7], raw1[1], raw1[3], raw1[5], raw1[7],
        raw2[1], raw2[3], raw2[5], raw2[7], raw3[1], raw3[3], raw3[5], raw3[7]
    )

    let lowShifted = SIMD16<Int16>(
        low[1], low[2], low[3], low[4], low[5], low[6], low[7], low[8],
        low[9], low[10], low[11], low[12], low[13], low[14], low[15], low[15]
    )
    high &-= (low &+ lowShifted) &>> 1

    let highShifted = SIMD16<Int16>(
        high[0], high[0], high[1], high[2], high[3], high[4], high[5], high[6],
        high[7], high[8], high[9], high[10], high[11], high[12], high[13], high[14]
    )
    low &+= (highShifted &+ high &+ 2) &>> 2

    UnsafeMutableRawPointer(base).storeBytes(of: low, as: SIMD16<Int16>.self)
    UnsafeMutableRawPointer(base + 16).storeBytes(of: high, as: SIMD16<Int16>.self)
}

@inline(__always)
func inverseLift53Block8(_ buffer: UnsafeMutableBufferPointer<Int16>, stride: Int) {
    guard let base = buffer.baseAddress else { return }
    let raw = UnsafeRawPointer(base).loadUnaligned(as: SIMD8<Int16>.self)
    var low = SIMD4<Int16>(raw[0], raw[1], raw[2], raw[3])
    var high = SIMD4<Int16>(raw[4], raw[5], raw[6], raw[7])

    let highShifted = SIMD4<Int16>(high[0], high[0], high[1], high[2])
    low &-= (highShifted &+ high &+ 2) &>> 2

    let lowShifted = SIMD4<Int16>(low[1], low[2], low[3], low[3])
    high &+= (low &+ lowShifted) &>> 1

    let result = SIMD8<Int16>(low[0], high[0], low[1], high[1], low[2], high[2], low[3], high[3])
    UnsafeMutableRawPointer(base).storeBytes(of: result, as: SIMD8<Int16>.self)
}

@inline(__always)
func inverseLift53Block16(_ buffer: UnsafeMutableBufferPointer<Int16>, stride: Int) {
    guard let base = buffer.baseAddress else { return }
    var low = UnsafeRawPointer(base).loadUnaligned(as: SIMD8<Int16>.self)
    var high = UnsafeRawPointer(base + 8).loadUnaligned(as: SIMD8<Int16>.self)

    let highShifted = SIMD8<Int16>(high[0], high[0], high[1], high[2], high[3], high[4], high[5], high[6])
    low &-= (highShifted &+ high &+ 2) &>> 2

    let lowShifted = SIMD8<Int16>(low[1], low[2], low[3], low[4], low[5], low[6], low[7], low[7])
    high &+= (low &+ lowShifted) &>> 1

    let out0 = SIMD8<Int16>(low[0], high[0], low[1], high[1], low[2], high[2], low[3], high[3])
    let out1 = SIMD8<Int16>(low[4], high[4], low[5], high[5], low[6], high[6], low[7], high[7])
    UnsafeMutableRawPointer(base).storeBytes(of: out0, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 8).storeBytes(of: out1, as: SIMD8<Int16>.self)
}

@inline(__always)
func inverseLift53Block32(_ buffer: UnsafeMutableBufferPointer<Int16>, stride: Int) {
    guard let base = buffer.baseAddress else { return }
    var low = UnsafeRawPointer(base).loadUnaligned(as: SIMD16<Int16>.self)
    var high = UnsafeRawPointer(base + 16).loadUnaligned(as: SIMD16<Int16>.self)

    let highShifted = SIMD16<Int16>(
        high[0], high[0], high[1], high[2], high[3], high[4], high[5], high[6],
        high[7], high[8], high[9], high[10], high[11], high[12], high[13], high[14]
    )
    low &-= (highShifted &+ high &+ 2) &>> 2

    let lowShifted = SIMD16<Int16>(
        low[1], low[2], low[3], low[4], low[5], low[6], low[7], low[8],
        low[9], low[10], low[11], low[12], low[13], low[14], low[15], low[15]
    )
    high &+= (low &+ lowShifted) &>> 1

    let out0 = SIMD8<Int16>(low[0], high[0], low[1], high[1], low[2], high[2], low[3], high[3])
    let out1 = SIMD8<Int16>(low[4], high[4], low[5], high[5], low[6], high[6], low[7], high[7])
    let out2 = SIMD8<Int16>(low[8], high[8], low[9], high[9], low[10], high[10], low[11], high[11])
    let out3 = SIMD8<Int16>(low[12], high[12], low[13], high[13], low[14], high[14], low[15], high[15])
    UnsafeMutableRawPointer(base).storeBytes(of: out0, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 8).storeBytes(of: out1, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 16).storeBytes(of: out2, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 24).storeBytes(of: out3, as: SIMD8<Int16>.self)
}

// MARK: - In-place Transpose

/// 8x8 in-place transpose using SIMD.
/// Transposes an 8x8 block stored row-major with the given stride.
/// After transpose, row[i] becomes column[i] and vice versa.
@inline(__always)
private func transpose8x8InPlace(_ base: UnsafeMutablePointer<Int16>, stride s: Int) {
    // Load all 8 rows
    var r0 = UnsafeRawPointer(base + 0 * s).loadUnaligned(as: SIMD8<Int16>.self)
    var r1 = UnsafeRawPointer(base + 1 * s).loadUnaligned(as: SIMD8<Int16>.self)
    var r2 = UnsafeRawPointer(base + 2 * s).loadUnaligned(as: SIMD8<Int16>.self)
    var r3 = UnsafeRawPointer(base + 3 * s).loadUnaligned(as: SIMD8<Int16>.self)
    var r4 = UnsafeRawPointer(base + 4 * s).loadUnaligned(as: SIMD8<Int16>.self)
    var r5 = UnsafeRawPointer(base + 5 * s).loadUnaligned(as: SIMD8<Int16>.self)
    var r6 = UnsafeRawPointer(base + 6 * s).loadUnaligned(as: SIMD8<Int16>.self)
    var r7 = UnsafeRawPointer(base + 7 * s).loadUnaligned(as: SIMD8<Int16>.self)

    // Phase 1: Interleave pairs of rows (2x2 blocks)
    // swap(r0[1], r1[0]), swap(r0[3], r1[2]), etc.
    var t0: SIMD8<Int16>, t1: SIMD8<Int16>

    // Interleave r0,r1 -> columns 0-1 fixed
    t0 = SIMD8(r0[0], r1[0], r0[2], r1[2], r0[4], r1[4], r0[6], r1[6])
    t1 = SIMD8(r0[1], r1[1], r0[3], r1[3], r0[5], r1[5], r0[7], r1[7])
    r0 = t0; r1 = t1

    t0 = SIMD8(r2[0], r3[0], r2[2], r3[2], r2[4], r3[4], r2[6], r3[6])
    t1 = SIMD8(r2[1], r3[1], r2[3], r3[3], r2[5], r3[5], r2[7], r3[7])
    r2 = t0; r3 = t1

    t0 = SIMD8(r4[0], r5[0], r4[2], r5[2], r4[4], r5[4], r4[6], r5[6])
    t1 = SIMD8(r4[1], r5[1], r4[3], r5[3], r4[5], r5[5], r4[7], r5[7])
    r4 = t0; r5 = t1

    t0 = SIMD8(r6[0], r7[0], r6[2], r7[2], r6[4], r7[4], r6[6], r7[6])
    t1 = SIMD8(r6[1], r7[1], r6[3], r7[3], r6[5], r7[5], r6[7], r7[7])
    r6 = t0; r7 = t1

    // Phase 2: Interleave quads (4x4 blocks)
    t0 = SIMD8(r0[0], r0[1], r2[0], r2[1], r0[4], r0[5], r2[4], r2[5])
    t1 = SIMD8(r0[2], r0[3], r2[2], r2[3], r0[6], r0[7], r2[6], r2[7])
    let q0 = t0; let q2 = t1

    t0 = SIMD8(r1[0], r1[1], r3[0], r3[1], r1[4], r1[5], r3[4], r3[5])
    t1 = SIMD8(r1[2], r1[3], r3[2], r3[3], r1[6], r1[7], r3[6], r3[7])
    let q1 = t0; let q3 = t1

    t0 = SIMD8(r4[0], r4[1], r6[0], r6[1], r4[4], r4[5], r6[4], r6[5])
    t1 = SIMD8(r4[2], r4[3], r6[2], r6[3], r4[6], r4[7], r6[6], r6[7])
    let q4 = t0; let q6 = t1

    t0 = SIMD8(r5[0], r5[1], r7[0], r7[1], r5[4], r5[5], r7[4], r7[5])
    t1 = SIMD8(r5[2], r5[3], r7[2], r7[3], r5[6], r5[7], r7[6], r7[7])
    let q5 = t0; let q7 = t1

    // Phase 3: Interleave octets (8x8 final)
    let f0 = SIMD8(q0[0], q0[1], q0[2], q0[3], q4[0], q4[1], q4[2], q4[3])
    let f1 = SIMD8(q1[0], q1[1], q1[2], q1[3], q5[0], q5[1], q5[2], q5[3])
    let f2 = SIMD8(q2[0], q2[1], q2[2], q2[3], q6[0], q6[1], q6[2], q6[3])
    let f3 = SIMD8(q3[0], q3[1], q3[2], q3[3], q7[0], q7[1], q7[2], q7[3])
    let f4 = SIMD8(q0[4], q0[5], q0[6], q0[7], q4[4], q4[5], q4[6], q4[7])
    let f5 = SIMD8(q1[4], q1[5], q1[6], q1[7], q5[4], q5[5], q5[6], q5[7])
    let f6 = SIMD8(q2[4], q2[5], q2[6], q2[7], q6[4], q6[5], q6[6], q6[7])
    let f7 = SIMD8(q3[4], q3[5], q3[6], q3[7], q7[4], q7[5], q7[6], q7[7])

    // Store
    UnsafeMutableRawPointer(base + 0 * s).storeBytes(of: f0, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 1 * s).storeBytes(of: f1, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 2 * s).storeBytes(of: f2, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 3 * s).storeBytes(of: f3, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 4 * s).storeBytes(of: f4, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 5 * s).storeBytes(of: f5, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 6 * s).storeBytes(of: f6, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 7 * s).storeBytes(of: f7, as: SIMD8<Int16>.self)
}

/// 16x16 in-place transpose using 8x8 sub-block transposes and swaps.
/// Decomposes into four 8x8 quadrants: TL, TR, BL, BR
/// 1. Transpose each 8x8 quadrant in-place
/// 2. Swap TR and BL quadrants
@inline(__always)
private func transpose16x16InPlace(_ base: UnsafeMutablePointer<Int16>, stride s: Int) {
    // Transpose 4 quadrants independently
    transpose8x8InPlace(base, stride: s)                          // TL
    transpose8x8InPlace(base + 8, stride: s)                      // TR
    transpose8x8InPlace(base + 8 * s, stride: s)                  // BL
    transpose8x8InPlace(base + 8 * s + 8, stride: s)              // BR

    // Swap TR (base+8) and BL (base+8*s)
    for y in 0..<8 {
        let ptrTR = base + y * s + 8
        let ptrBL = base + (y + 8) * s
        let vTR = UnsafeRawPointer(ptrTR).loadUnaligned(as: SIMD8<Int16>.self)
        let vBL = UnsafeRawPointer(ptrBL).loadUnaligned(as: SIMD8<Int16>.self)
        UnsafeMutableRawPointer(ptrTR).storeBytes(of: vBL, as: SIMD8<Int16>.self)
        UnsafeMutableRawPointer(ptrBL).storeBytes(of: vTR, as: SIMD8<Int16>.self)
    }
}

/// 32x32 in-place transpose using 16x16 sub-block transposes and swaps.
/// Decomposes into four 16x16 quadrants: TL, TR, BL, BR
/// 1. Transpose each 16x16 quadrant in-place
/// 2. Swap TR and BL quadrants
@inline(__always)
private func transpose32x32InPlace(_ base: UnsafeMutablePointer<Int16>, stride s: Int) {
    // Transpose 4 quadrants independently
    transpose16x16InPlace(base, stride: s)                         // TL
    transpose16x16InPlace(base + 16, stride: s)                    // TR
    transpose16x16InPlace(base + 16 * s, stride: s)                // BL
    transpose16x16InPlace(base + 16 * s + 16, stride: s)           // BR

    // Swap TR (base+16) and BL (base+16*s) using SIMD16 for full-width operations
    for y in 0..<16 {
        let ptrTR = base + y * s + 16
        let ptrBL = base + (y + 16) * s
        let vTR = UnsafeRawPointer(ptrTR).loadUnaligned(as: SIMD16<Int16>.self)
        let vBL = UnsafeRawPointer(ptrBL).loadUnaligned(as: SIMD16<Int16>.self)
        UnsafeMutableRawPointer(ptrTR).storeBytes(of: vBL, as: SIMD16<Int16>.self)
        UnsafeMutableRawPointer(ptrBL).storeBytes(of: vTR, as: SIMD16<Int16>.self)
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
    // Transpose -> row lifting (was column) -> transpose back
    transpose8x8InPlace(base, stride: width)
    lift53Block8(UnsafeMutableBufferPointer(start: base + (0 * width), count: 8), stride: 1)
    lift53Block8(UnsafeMutableBufferPointer(start: base + (1 * width), count: 8), stride: 1)
    lift53Block8(UnsafeMutableBufferPointer(start: base + (2 * width), count: 8), stride: 1)
    lift53Block8(UnsafeMutableBufferPointer(start: base + (3 * width), count: 8), stride: 1)
    lift53Block8(UnsafeMutableBufferPointer(start: base + (4 * width), count: 8), stride: 1)
    lift53Block8(UnsafeMutableBufferPointer(start: base + (5 * width), count: 8), stride: 1)
    lift53Block8(UnsafeMutableBufferPointer(start: base + (6 * width), count: 8), stride: 1)
    lift53Block8(UnsafeMutableBufferPointer(start: base + (7 * width), count: 8), stride: 1)
    transpose8x8InPlace(base, stride: width)
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
    // Row lifting (stride=1, contiguous)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (0 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (1 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (2 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (3 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (4 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (5 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (6 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (7 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (8 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (9 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (10 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (11 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (12 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (13 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (14 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (15 * width), count: 16), stride: 1)
    // Transpose -> row lifting (was column) -> transpose back
    transpose16x16InPlace(base, stride: width)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (0 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (1 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (2 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (3 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (4 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (5 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (6 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (7 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (8 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (9 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (10 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (11 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (12 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (13 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (14 * width), count: 16), stride: 1)
    lift53Block16(UnsafeMutableBufferPointer(start: base + (15 * width), count: 16), stride: 1)
    transpose16x16InPlace(base, stride: width)
}

@inline(__always)
func dwt2DBlock16Subbands(_ block: BlockView) -> Subbands {
    dwt2DBlock16(block)
    return makeSubbands(base: block.base, size: 16, stride: block.stride)
}

@inline(__always)
func dwt2DBlock32(_ block: BlockView) {
    let base = block.base
    let width = block.stride
    // Row lifting (stride=1, contiguous)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (0 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (1 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (2 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (3 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (4 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (5 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (6 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (7 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (8 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (9 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (10 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (11 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (12 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (13 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (14 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (15 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (16 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (17 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (18 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (19 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (20 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (21 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (22 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (23 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (24 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (25 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (26 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (27 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (28 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (29 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (30 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (31 * width), count: 32), stride: 1)
    // Transpose -> row lifting (was column) -> transpose back
    transpose32x32InPlace(base, stride: width)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (0 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (1 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (2 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (3 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (4 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (5 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (6 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (7 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (8 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (9 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (10 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (11 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (12 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (13 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (14 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (15 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (16 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (17 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (18 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (19 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (20 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (21 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (22 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (23 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (24 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (25 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (26 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (27 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (28 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (29 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (30 * width), count: 32), stride: 1)
    lift53Block32(UnsafeMutableBufferPointer(start: base + (31 * width), count: 32), stride: 1)
    transpose32x32InPlace(base, stride: width)
}

@inline(__always)
func dwt2DBlock32Subbands(_ block: BlockView) -> Subbands {
    dwt2DBlock32(block)
    return makeSubbands(base: block.base, size: 32, stride: block.stride)
}

@inline(__always)
func inverseDWT2DBlock8(_ block: BlockView) {
    let base = block.base
    let width = block.stride
    // Inverse column lifting via transpose: transpose -> inverseLift rows -> transpose back
    transpose8x8InPlace(base, stride: width)
    inverseLift53Block8(UnsafeMutableBufferPointer(start: base + (0 * width), count: 8), stride: 1)
    inverseLift53Block8(UnsafeMutableBufferPointer(start: base + (1 * width), count: 8), stride: 1)
    inverseLift53Block8(UnsafeMutableBufferPointer(start: base + (2 * width), count: 8), stride: 1)
    inverseLift53Block8(UnsafeMutableBufferPointer(start: base + (3 * width), count: 8), stride: 1)
    inverseLift53Block8(UnsafeMutableBufferPointer(start: base + (4 * width), count: 8), stride: 1)
    inverseLift53Block8(UnsafeMutableBufferPointer(start: base + (5 * width), count: 8), stride: 1)
    inverseLift53Block8(UnsafeMutableBufferPointer(start: base + (6 * width), count: 8), stride: 1)
    inverseLift53Block8(UnsafeMutableBufferPointer(start: base + (7 * width), count: 8), stride: 1)
    transpose8x8InPlace(base, stride: width)
    // Inverse row lifting (stride=1, contiguous)
    inverseLift53Block8(UnsafeMutableBufferPointer(start: base + (0 * width), count: 8), stride: 1)
    inverseLift53Block8(UnsafeMutableBufferPointer(start: base + (1 * width), count: 8), stride: 1)
    inverseLift53Block8(UnsafeMutableBufferPointer(start: base + (2 * width), count: 8), stride: 1)
    inverseLift53Block8(UnsafeMutableBufferPointer(start: base + (3 * width), count: 8), stride: 1)
    inverseLift53Block8(UnsafeMutableBufferPointer(start: base + (4 * width), count: 8), stride: 1)
    inverseLift53Block8(UnsafeMutableBufferPointer(start: base + (5 * width), count: 8), stride: 1)
    inverseLift53Block8(UnsafeMutableBufferPointer(start: base + (6 * width), count: 8), stride: 1)
    inverseLift53Block8(UnsafeMutableBufferPointer(start: base + (7 * width), count: 8), stride: 1)
}

@inline(__always)
func inverseDWT2DBlock16(_ block: BlockView) {
    let base = block.base
    let width = block.stride
    // Inverse column lifting via transpose
    transpose16x16InPlace(base, stride: width)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (0 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (1 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (2 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (3 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (4 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (5 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (6 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (7 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (8 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (9 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (10 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (11 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (12 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (13 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (14 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (15 * width), count: 16), stride: 1)
    transpose16x16InPlace(base, stride: width)
    // Inverse row lifting (stride=1, contiguous)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (0 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (1 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (2 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (3 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (4 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (5 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (6 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (7 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (8 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (9 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (10 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (11 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (12 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (13 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (14 * width), count: 16), stride: 1)
    inverseLift53Block16(UnsafeMutableBufferPointer(start: base + (15 * width), count: 16), stride: 1)
}

@inline(__always)
func inverseDWT2DBlock32(_ block: BlockView) {
    let base = block.base
    let width = block.stride
    // Inverse column lifting via transpose
    transpose32x32InPlace(base, stride: width)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (0 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (1 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (2 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (3 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (4 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (5 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (6 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (7 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (8 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (9 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (10 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (11 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (12 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (13 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (14 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (15 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (16 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (17 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (18 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (19 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (20 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (21 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (22 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (23 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (24 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (25 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (26 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (27 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (28 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (29 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (30 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (31 * width), count: 32), stride: 1)
    transpose32x32InPlace(base, stride: width)
    // Inverse row lifting (stride=1, contiguous)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (0 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (1 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (2 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (3 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (4 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (5 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (6 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (7 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (8 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (9 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (10 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (11 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (12 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (13 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (14 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (15 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (16 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (17 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (18 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (19 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (20 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (21 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (22 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (23 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (24 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (25 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (26 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (27 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (28 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (29 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (30 * width), count: 32), stride: 1)
    inverseLift53Block32(UnsafeMutableBufferPointer(start: base + (31 * width), count: 32), stride: 1)
}
