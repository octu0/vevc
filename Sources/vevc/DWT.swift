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

    // MARK: - Overflow-Safe Int16 Arithmetic
    // In DWT Lifting, input values bounded at ~16382 can double at every cascade column/row dimension,
    // easily wrapping around Int16 boundaries during regular SIMD addition `(A &+ B &+ 2) &>> 2`.
    // We rewrite the operation strictly avoiding the explicit sum to prevent the diamond ringing artifacts.

    extension SIMD where Scalar == Int16 {
        @inline(__always)
        internal func addShift1(_ other: Self) -> Self {
            let halfA = self &>> 1
            let halfB = other &>> 1
            let sumHalf = halfA &+ halfB
            let remainder = (self & 1) &+ (other & 1)
            return sumHalf &+ (remainder &>> 1)
        }

        @inline(__always)
        internal func addShift2(_ other: Self) -> Self {
            let halfA = self &>> 1
            let halfB = other &>> 1
            let sumHalf = halfA &+ halfB
            let remainder = (self & 1) &+ (other & 1) &+ 2
            return (sumHalf &+ (remainder &>> 1)) &>> 1
        }
    }

// MARK: - LeGall 5/3 Lifting

@inline(__always)
func lift53_4(_ buffer: UnsafeMutableBufferPointer<Int16>, stride: Int) {
    // even samples = low, odd samples = high
    var low = SIMD2<Int16>(buffer[0 * stride], buffer[2 * stride])
    var high = SIMD2<Int16>(buffer[1 * stride], buffer[3 * stride])

    // Predict: H[n] -= (L[n] + L[n+1]) >> 1
    // Boundary mirror: L[n+1] for last element uses L[n] itself
    let lowShifted = SIMD2<Int16>(low[1], low[1])
    high &-= low.addShift1(lowShifted)
    let highShifted = SIMD2<Int16>(high[0], high[0])
    low &+= highShifted.addShift2(high)

    // Output: [L0, L1, H0, H1]
    buffer[0 * stride] = low[0]; buffer[1 * stride] = low[1]
    buffer[2 * stride] = high[0]; buffer[3 * stride] = high[1]
}

@inline(__always)
func invLift53_4(_ buffer: UnsafeMutableBufferPointer<Int16>, stride: Int) {
    // Input layout: [L0, L1, H0, H1]
    var low = SIMD2<Int16>(buffer[0 * stride], buffer[1 * stride])
    var high = SIMD2<Int16>(buffer[2 * stride], buffer[3 * stride])

    // Inverse Update: L[n] -= (H[n-1] + H[n] + 2) >> 2
    let highShifted = SIMD2<Int16>(high[0], high[0])
    low &-= highShifted.addShift2(high)
    let lowShifted = SIMD2<Int16>(low[1], low[1])
    high &+= low.addShift1(lowShifted)

    // Interleave back: [L0, H0, L1, H1]
    buffer[0 * stride] = low[0]; buffer[1 * stride] = high[0]
    buffer[2 * stride] = low[1]; buffer[3 * stride] = high[1]
}

@inline(__always)
func lift53_8(_ buffer: UnsafeMutableBufferPointer<Int16>, stride: Int) {
    var low = SIMD4<Int16>(buffer[0 * stride], buffer[2 * stride], buffer[4 * stride], buffer[6 * stride])
    var high = SIMD4<Int16>(buffer[1 * stride], buffer[3 * stride], buffer[5 * stride], buffer[7 * stride])

    let lowShifted = SIMD4<Int16>(low[1], low[2], low[3], low[3])
    high &-= low.addShift1(lowShifted)
    let highShifted = SIMD4<Int16>(high[0], high[0], high[1], high[2])
    low &+= highShifted.addShift2(high)

    buffer[0 * stride] = low[0]; buffer[1 * stride] = low[1]; buffer[2 * stride] = low[2]; buffer[3 * stride] = low[3]
    buffer[4 * stride] = high[0]; buffer[5 * stride] = high[1]; buffer[6 * stride] = high[2]; buffer[7 * stride] = high[3]
}

@inline(__always)
func lift53_16(_ buffer: UnsafeMutableBufferPointer<Int16>, stride: Int) {
    var low = SIMD8<Int16>(
        buffer[0 * stride], buffer[2 * stride], buffer[4 * stride], buffer[6 * stride],
        buffer[8 * stride], buffer[10 * stride], buffer[12 * stride], buffer[14 * stride]
    )
    var high = SIMD8<Int16>(
        buffer[1 * stride], buffer[3 * stride], buffer[5 * stride], buffer[7 * stride],
        buffer[9 * stride], buffer[11 * stride], buffer[13 * stride], buffer[15 * stride]
    )

    let lowShifted = SIMD8<Int16>(low[1], low[2], low[3], low[4], low[5], low[6], low[7], low[7])
    high &-= low.addShift1(lowShifted)
    let highShifted = SIMD8<Int16>(high[0], high[0], high[1], high[2], high[3], high[4], high[5], high[6])
    low &+= highShifted.addShift2(high)

    buffer[0 * stride] = low[0]; buffer[1 * stride] = low[1]; buffer[2 * stride] = low[2]; buffer[3 * stride] = low[3]
    buffer[4 * stride] = low[4]; buffer[5 * stride] = low[5]; buffer[6 * stride] = low[6]; buffer[7 * stride] = low[7]
    buffer[8 * stride] = high[0]; buffer[9 * stride] = high[1]; buffer[10 * stride] = high[2]; buffer[11 * stride] = high[3]
    buffer[12 * stride] = high[4]; buffer[13 * stride] = high[5]; buffer[14 * stride] = high[6]; buffer[15 * stride] = high[7]
}

@inline(__always)
func lift53_32(_ buffer: UnsafeMutableBufferPointer<Int16>, stride: Int) {
    var low = SIMD16<Int16>(
        buffer[0 * stride], buffer[2 * stride], buffer[4 * stride], buffer[6 * stride],
        buffer[8 * stride], buffer[10 * stride], buffer[12 * stride], buffer[14 * stride],
        buffer[16 * stride], buffer[18 * stride], buffer[20 * stride], buffer[22 * stride],
        buffer[24 * stride], buffer[26 * stride], buffer[28 * stride], buffer[30 * stride]
    )
    var high = SIMD16<Int16>(
        buffer[1 * stride], buffer[3 * stride], buffer[5 * stride], buffer[7 * stride],
        buffer[9 * stride], buffer[11 * stride], buffer[13 * stride], buffer[15 * stride],
        buffer[17 * stride], buffer[19 * stride], buffer[21 * stride], buffer[23 * stride],
        buffer[25 * stride], buffer[27 * stride], buffer[29 * stride], buffer[31 * stride]
    )

    let lowShifted = SIMD16<Int16>(
        low[1], low[2], low[3], low[4], low[5], low[6], low[7], low[8],
        low[9], low[10], low[11], low[12], low[13], low[14], low[15], low[15]
    )
    high &-= low.addShift1(lowShifted)
    let highShifted = SIMD16<Int16>(
        high[0], high[0], high[1], high[2], high[3], high[4], high[5], high[6],
        high[7], high[8], high[9], high[10], high[11], high[12], high[13], high[14]
    )
    low &+= highShifted.addShift2(high)

    buffer[0 * stride] = low[0]; buffer[1 * stride] = low[1]; buffer[2 * stride] = low[2]; buffer[3 * stride] = low[3]
    buffer[4 * stride] = low[4]; buffer[5 * stride] = low[5]; buffer[6 * stride] = low[6]; buffer[7 * stride] = low[7]
    buffer[8 * stride] = low[8]; buffer[9 * stride] = low[9]; buffer[10 * stride] = low[10]; buffer[11 * stride] = low[11]
    buffer[12 * stride] = low[12]; buffer[13 * stride] = low[13]; buffer[14 * stride] = low[14]; buffer[15 * stride] = low[15]
    buffer[16 * stride] = high[0]; buffer[17 * stride] = high[1]; buffer[18 * stride] = high[2]; buffer[19 * stride] = high[3]
    buffer[20 * stride] = high[4]; buffer[21 * stride] = high[5]; buffer[22 * stride] = high[6]; buffer[23 * stride] = high[7]
    buffer[24 * stride] = high[8]; buffer[25 * stride] = high[9]; buffer[26 * stride] = high[10]; buffer[27 * stride] = high[11]
    buffer[28 * stride] = high[12]; buffer[29 * stride] = high[13]; buffer[30 * stride] = high[14]; buffer[31 * stride] = high[15]
}

@inline(__always)
func invLift53_8(_ buffer: UnsafeMutableBufferPointer<Int16>, stride: Int) {
    var low = SIMD4<Int16>(buffer[0 * stride], buffer[1 * stride], buffer[2 * stride], buffer[3 * stride])
    var high = SIMD4<Int16>(buffer[4 * stride], buffer[5 * stride], buffer[6 * stride], buffer[7 * stride])

    let highShifted = SIMD4<Int16>(high[0], high[0], high[1], high[2])
    low &-= highShifted.addShift2(high)
    let lowShifted = SIMD4<Int16>(low[1], low[2], low[3], low[3])
    high &+= low.addShift1(lowShifted)

    buffer[0 * stride] = low[0]; buffer[1 * stride] = high[0]
    buffer[2 * stride] = low[1]; buffer[3 * stride] = high[1]
    buffer[4 * stride] = low[2]; buffer[5 * stride] = high[2]
    buffer[6 * stride] = low[3]; buffer[7 * stride] = high[3]
}

@inline(__always)
func invLift53_16(_ buffer: UnsafeMutableBufferPointer<Int16>, stride: Int) {
    var low = SIMD8<Int16>(
        buffer[0 * stride], buffer[1 * stride], buffer[2 * stride], buffer[3 * stride],
        buffer[4 * stride], buffer[5 * stride], buffer[6 * stride], buffer[7 * stride]
    )
    var high = SIMD8<Int16>(
        buffer[8 * stride], buffer[9 * stride], buffer[10 * stride], buffer[11 * stride],
        buffer[12 * stride], buffer[13 * stride], buffer[14 * stride], buffer[15 * stride]
    )

    let highShifted = SIMD8<Int16>(high[0], high[0], high[1], high[2], high[3], high[4], high[5], high[6])
    low &-= highShifted.addShift2(high)
    let lowShifted = SIMD8<Int16>(low[1], low[2], low[3], low[4], low[5], low[6], low[7], low[7])
    high &+= low.addShift1(lowShifted)

    buffer[0 * stride] = low[0]; buffer[1 * stride] = high[0]
    buffer[2 * stride] = low[1]; buffer[3 * stride] = high[1]
    buffer[4 * stride] = low[2]; buffer[5 * stride] = high[2]
    buffer[6 * stride] = low[3]; buffer[7 * stride] = high[3]
    buffer[8 * stride] = low[4]; buffer[9 * stride] = high[4]
    buffer[10 * stride] = low[5]; buffer[11 * stride] = high[5]
    buffer[12 * stride] = low[6]; buffer[13 * stride] = high[6]
    buffer[14 * stride] = low[7]; buffer[15 * stride] = high[7]
}

@inline(__always)
func invLift53_32(_ buffer: UnsafeMutableBufferPointer<Int16>, stride: Int) {
    var low = SIMD16<Int16>(
        buffer[0 * stride], buffer[1 * stride], buffer[2 * stride], buffer[3 * stride],
        buffer[4 * stride], buffer[5 * stride], buffer[6 * stride], buffer[7 * stride],
        buffer[8 * stride], buffer[9 * stride], buffer[10 * stride], buffer[11 * stride],
        buffer[12 * stride], buffer[13 * stride], buffer[14 * stride], buffer[15 * stride]
    )
    var high = SIMD16<Int16>(
        buffer[16 * stride], buffer[17 * stride], buffer[18 * stride], buffer[19 * stride],
        buffer[20 * stride], buffer[21 * stride], buffer[22 * stride], buffer[23 * stride],
        buffer[24 * stride], buffer[25 * stride], buffer[26 * stride], buffer[27 * stride],
        buffer[28 * stride], buffer[29 * stride], buffer[30 * stride], buffer[31 * stride]
    )

    let highShifted = SIMD16<Int16>(
        high[0], high[0], high[1], high[2], high[3], high[4], high[5], high[6],
        high[7], high[8], high[9], high[10], high[11], high[12], high[13], high[14]
    )
    low &-= highShifted.addShift2(high)
    let lowShifted = SIMD16<Int16>(
        low[1], low[2], low[3], low[4], low[5], low[6], low[7], low[8],
        low[9], low[10], low[11], low[12], low[13], low[14], low[15], low[15]
    )
    high &+= low.addShift1(lowShifted)

    buffer[0 * stride] = low[0]; buffer[1 * stride] = high[0]
    buffer[2 * stride] = low[1]; buffer[3 * stride] = high[1]
    buffer[4 * stride] = low[2]; buffer[5 * stride] = high[2]
    buffer[6 * stride] = low[3]; buffer[7 * stride] = high[3]
    buffer[8 * stride] = low[4]; buffer[9 * stride] = high[4]
    buffer[10 * stride] = low[5]; buffer[11 * stride] = high[5]
    buffer[12 * stride] = low[6]; buffer[13 * stride] = high[6]
    buffer[14 * stride] = low[7]; buffer[15 * stride] = high[7]
    buffer[16 * stride] = low[8]; buffer[17 * stride] = high[8]
    buffer[18 * stride] = low[9]; buffer[19 * stride] = high[9]
    buffer[20 * stride] = low[10]; buffer[21 * stride] = high[10]
    buffer[22 * stride] = low[11]; buffer[23 * stride] = high[11]
    buffer[24 * stride] = low[12]; buffer[25 * stride] = high[12]
    buffer[26 * stride] = low[13]; buffer[27 * stride] = high[13]
    buffer[28 * stride] = low[14]; buffer[29 * stride] = high[14]
    buffer[30 * stride] = low[15]; buffer[31 * stride] = high[15]
}

// MARK: - 2D DWT

@inline(__always)
func dwt2d_8(_ block: inout BlockView) {
    let size = 8
    let base = block.base
    let width = block.stride
    for y in 0..<size {
        let rowBuffer = UnsafeMutableBufferPointer(start: base + (y * width), count: size)
        lift53_8(rowBuffer, stride: 1)
    }
    let colCount = ((size - 1) * width) + 1
    for x in 0..<size {
        let colBuffer = UnsafeMutableBufferPointer(start: base + x, count: colCount)
        lift53_8(colBuffer, stride: width)
    }
}

@inline(__always)
func dwt2d_8_sb(_ block: inout BlockView) -> Subbands {
    dwt2d_8(&block)
    return makeSubbands(base: block.base, size: 8, stride: block.stride)
}

@inline(__always)
func dwt2d_16(_ block: inout BlockView) {
    let size = 16
    let base = block.base
    let width = block.stride
    for y in 0..<size {
        let rowBuffer = UnsafeMutableBufferPointer(start: base + (y * width), count: size)
        lift53_16(rowBuffer, stride: 1)
    }
    let colCount = ((size - 1) * width) + 1
    for x in 0..<size {
        let colBuffer = UnsafeMutableBufferPointer(start: base + x, count: colCount)
        lift53_16(colBuffer, stride: width)
    }
}

@inline(__always)
func dwt2d_16_sb(_ block: inout BlockView) -> Subbands {
    dwt2d_16(&block)
    return makeSubbands(base: block.base, size: 16, stride: block.stride)
}

@inline(__always)
func dwt2d_32(_ block: inout BlockView) {
    let size = 32
    let base = block.base
    let width = block.stride
    for y in 0..<size {
        let rowBuffer = UnsafeMutableBufferPointer(start: base + (y * width), count: size)
        lift53_32(rowBuffer, stride: 1)
    }
    let colCount = ((size - 1) * width) + 1
    for x in 0..<size {
        let colBuffer = UnsafeMutableBufferPointer(start: base + x, count: colCount)
        lift53_32(colBuffer, stride: width)
    }
}

@inline(__always)
func dwt2d_32_sb(_ block: inout BlockView) -> Subbands {
    dwt2d_32(&block)
    return makeSubbands(base: block.base, size: 32, stride: block.stride)
}

@inline(__always)
func invDwt2d_8(_ block: inout BlockView) {
    let size = 8
    let base = block.base
    let width = block.stride
    let colCount = ((size - 1) * width) + 1
    for x in 0..<size {
        let colBuffer = UnsafeMutableBufferPointer(start: base + x, count: colCount)
        invLift53_8(colBuffer, stride: width)
    }
    for y in 0..<size {
        let rowBuffer = UnsafeMutableBufferPointer(start: base + (y * width), count: size)
        invLift53_8(rowBuffer, stride: 1)
    }
}

@inline(__always)
func invDwt2d_16(_ block: inout BlockView) {
    let size = 16
    let base = block.base
    let width = block.stride
    let colCount = ((size - 1) * width) + 1
    for x in 0..<size {
        let colBuffer = UnsafeMutableBufferPointer(start: base + x, count: colCount)
        invLift53_16(colBuffer, stride: width)
    }
    for y in 0..<size {
        let rowBuffer = UnsafeMutableBufferPointer(start: base + (y * width), count: size)
        invLift53_16(rowBuffer, stride: 1)
    }
}

@inline(__always)
func invDwt2d_32(_ block: inout BlockView) {
    let size = 32
    let base = block.base
    let width = block.stride
    let colCount = ((size - 1) * width) + 1
    for x in 0..<size {
        let colBuffer = UnsafeMutableBufferPointer(start: base + x, count: colCount)
        invLift53_32(colBuffer, stride: width)
    }
    for y in 0..<size {
        let rowBuffer = UnsafeMutableBufferPointer(start: base + (y * width), count: size)
        invLift53_32(rowBuffer, stride: 1)
    }
}
