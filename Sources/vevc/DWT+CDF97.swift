import simd

public enum DWTKernel: UInt8 {
    case leGall53 = 0x01
    case cdf97 = 0x02
}

// cA: -25987, cB: -868, cG: 14466, cD: 7266, cIK: 13318, cK: 20155
private let cA: Int32 = -25987
private let cB: Int32 = -868
private let cG: Int32 = 14466
private let cD: Int32 = 7266
private let cIK: Int32 = 13318
private let cK: Int32 = 20155

@inline(__always)
private func mulShift(_ c: Int32, _ v: SIMD8<Int32>) -> SIMD8<Int32> {
    return (v &* c &+ 8192) &>> 14
}

@inline(__always)
private func mulShift16(_ c: Int32, _ v: SIMD16<Int32>) -> SIMD16<Int32> {
    return (v &* c &+ 8192) &>> 14
}

// MARK: - CDF 9/7 SIMD 16

@inline(__always)
func cdf97LiftBlock16(_ buffer: UnsafeMutableBufferPointer<Int16>, stride: Int) {
    guard let base = buffer.baseAddress else { return }
    let raw0 = UnsafeRawPointer(base).loadUnaligned(as: SIMD8<Int16>.self)
    let raw1 = UnsafeRawPointer(base + 8).loadUnaligned(as: SIMD8<Int16>.self)
    
    // Split interleaved to L and H, and convert to Int32
    var low = SIMD8<Int32>(
        Int32(raw0[0]), Int32(raw0[2]), Int32(raw0[4]), Int32(raw0[6]),
        Int32(raw1[0]), Int32(raw1[2]), Int32(raw1[4]), Int32(raw1[6])
    )
    var high = SIMD8<Int32>(
        Int32(raw0[1]), Int32(raw0[3]), Int32(raw0[5]), Int32(raw0[7]),
        Int32(raw1[1]), Int32(raw1[3]), Int32(raw1[5]), Int32(raw1[7])
    )
    
    // H += cA * (cl(i) + cl(i+1))
    var lowShifted = SIMD8<Int32>(low[1], low[2], low[3], low[4], low[5], low[6], low[7], low[7])
    high &+= mulShift(cA, low &+ lowShifted)
    
    // L += cB * (ch(i-1) + H[i])
    var highShifted = SIMD8<Int32>(high[0], high[0], high[1], high[2], high[3], high[4], high[5], high[6])
    low &+= mulShift(cB, highShifted &+ high)
    
    // H += cG * (cl(i) + cl(i+1))
    lowShifted = SIMD8<Int32>(low[1], low[2], low[3], low[4], low[5], low[6], low[7], low[7])
    high &+= mulShift(cG, low &+ lowShifted)
    
    // L += cD * (ch(i-1) + H[i])
    highShifted = SIMD8<Int32>(high[0], high[0], high[1], high[2], high[3], high[4], high[5], high[6])
    low &+= mulShift(cD, highShifted &+ high)
    // Convert back to Int16
    let outL = SIMD8<Int16>(truncatingIfNeeded: low)
    let outH = SIMD8<Int16>(truncatingIfNeeded: high)
    
    UnsafeMutableRawPointer(base).storeBytes(of: outL, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 8).storeBytes(of: outH, as: SIMD8<Int16>.self)
}

@inline(__always)
func inverseCdf97LiftBlock16(_ buffer: UnsafeMutableBufferPointer<Int16>, stride: Int) {
    guard let base = buffer.baseAddress else { return }
    let rawL = UnsafeRawPointer(base).loadUnaligned(as: SIMD8<Int16>.self)
    let rawH = UnsafeRawPointer(base + 8).loadUnaligned(as: SIMD8<Int16>.self)
    
    var low = SIMD8<Int32>(
        Int32(rawL[0]), Int32(rawL[1]), Int32(rawL[2]), Int32(rawL[3]),
        Int32(rawL[4]), Int32(rawL[5]), Int32(rawL[6]), Int32(rawL[7])
    )
    var high = SIMD8<Int32>(
        Int32(rawH[0]), Int32(rawH[1]), Int32(rawH[2]), Int32(rawH[3]),
        Int32(rawH[4]), Int32(rawH[5]), Int32(rawH[6]), Int32(rawH[7])
    )
    // L -= cD * (ch(i-1) + H[i])
    var highShifted = SIMD8<Int32>(high[0], high[0], high[1], high[2], high[3], high[4], high[5], high[6])
    low &-= mulShift(cD, highShifted &+ high)
    
    // H -= cG * (cl(i) + cl(i+1))
    var lowShifted = SIMD8<Int32>(low[1], low[2], low[3], low[4], low[5], low[6], low[7], low[7])
    high &-= mulShift(cG, low &+ lowShifted)
    
    // L -= cB * (ch(i-1) + H[i])
    highShifted = SIMD8<Int32>(high[0], high[0], high[1], high[2], high[3], high[4], high[5], high[6])
    low &-= mulShift(cB, highShifted &+ high)
    
    // H -= cA * (cl(i) + cl(i+1))
    lowShifted = SIMD8<Int32>(low[1], low[2], low[3], low[4], low[5], low[6], low[7], low[7])
    high &-= mulShift(cA, low &+ lowShifted)
    
    let outL = SIMD8<Int16>(truncatingIfNeeded: low)
    let outH = SIMD8<Int16>(truncatingIfNeeded: high)
    
    let out0 = SIMD8<Int16>(outL[0], outH[0], outL[1], outH[1], outL[2], outH[2], outL[3], outH[3])
    let out1 = SIMD8<Int16>(outL[4], outH[4], outL[5], outH[5], outL[6], outH[6], outL[7], outH[7])
    
    UnsafeMutableRawPointer(base).storeBytes(of: out0, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 8).storeBytes(of: out1, as: SIMD8<Int16>.self)
}

// MARK: - CDF 9/7 SIMD 32

@inline(__always)
func cdf97LiftBlock32(_ buffer: UnsafeMutableBufferPointer<Int16>, stride: Int) {
    guard let base = buffer.baseAddress else { return }
    let raw0 = UnsafeRawPointer(base).loadUnaligned(as: SIMD8<Int16>.self)
    let raw1 = UnsafeRawPointer(base + 8).loadUnaligned(as: SIMD8<Int16>.self)
    let raw2 = UnsafeRawPointer(base + 16).loadUnaligned(as: SIMD8<Int16>.self)
    let raw3 = UnsafeRawPointer(base + 24).loadUnaligned(as: SIMD8<Int16>.self)
    
    var low = SIMD16<Int32>(
        Int32(raw0[0]), Int32(raw0[2]), Int32(raw0[4]), Int32(raw0[6]),
        Int32(raw1[0]), Int32(raw1[2]), Int32(raw1[4]), Int32(raw1[6]),
        Int32(raw2[0]), Int32(raw2[2]), Int32(raw2[4]), Int32(raw2[6]),
        Int32(raw3[0]), Int32(raw3[2]), Int32(raw3[4]), Int32(raw3[6])
    )
    var high = SIMD16<Int32>(
        Int32(raw0[1]), Int32(raw0[3]), Int32(raw0[5]), Int32(raw0[7]),
        Int32(raw1[1]), Int32(raw1[3]), Int32(raw1[5]), Int32(raw1[7]),
        Int32(raw2[1]), Int32(raw2[3]), Int32(raw2[5]), Int32(raw2[7]),
        Int32(raw3[1]), Int32(raw3[3]), Int32(raw3[5]), Int32(raw3[7])
    )
    
    var lowShifted = SIMD16<Int32>(
        low[1], low[2], low[3], low[4], low[5], low[6], low[7], low[8],
        low[9], low[10], low[11], low[12], low[13], low[14], low[15], low[15]
    )
    high &+= mulShift16(cA, low &+ lowShifted)
    
    var highShifted = SIMD16<Int32>(
        high[0], high[0], high[1], high[2], high[3], high[4], high[5], high[6],
        high[7], high[8], high[9], high[10], high[11], high[12], high[13], high[14]
    )
    low &+= mulShift16(cB, highShifted &+ high)
    
    lowShifted = SIMD16<Int32>(
        low[1], low[2], low[3], low[4], low[5], low[6], low[7], low[8],
        low[9], low[10], low[11], low[12], low[13], low[14], low[15], low[15]
    )
    high &+= mulShift16(cG, low &+ lowShifted)
    
    highShifted = SIMD16<Int32>(
        high[0], high[0], high[1], high[2], high[3], high[4], high[5], high[6],
        high[7], high[8], high[9], high[10], high[11], high[12], high[13], high[14]
    )
    low &+= mulShift16(cD, highShifted &+ high)
    
    let outL0 = SIMD8<Int16>(truncatingIfNeeded: low.lowHalf)
    let outL1 = SIMD8<Int16>(truncatingIfNeeded: low.highHalf)
    let outH0 = SIMD8<Int16>(truncatingIfNeeded: high.lowHalf)
    let outH1 = SIMD8<Int16>(truncatingIfNeeded: high.highHalf)
    
    UnsafeMutableRawPointer(base).storeBytes(of: outL0, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 8).storeBytes(of: outL1, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 16).storeBytes(of: outH0, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 24).storeBytes(of: outH1, as: SIMD8<Int16>.self)
}

@inline(__always)
func inverseCdf97LiftBlock32(_ buffer: UnsafeMutableBufferPointer<Int16>, stride: Int) {
    guard let base = buffer.baseAddress else { return }
    let rawL0 = UnsafeRawPointer(base).loadUnaligned(as: SIMD8<Int16>.self)
    let rawL1 = UnsafeRawPointer(base + 8).loadUnaligned(as: SIMD8<Int16>.self)
    let rawH0 = UnsafeRawPointer(base + 16).loadUnaligned(as: SIMD8<Int16>.self)
    let rawH1 = UnsafeRawPointer(base + 24).loadUnaligned(as: SIMD8<Int16>.self)
    
    var low = SIMD16<Int32>(
        Int32(rawL0[0]), Int32(rawL0[1]), Int32(rawL0[2]), Int32(rawL0[3]),
        Int32(rawL0[4]), Int32(rawL0[5]), Int32(rawL0[6]), Int32(rawL0[7]),
        Int32(rawL1[0]), Int32(rawL1[1]), Int32(rawL1[2]), Int32(rawL1[3]),
        Int32(rawL1[4]), Int32(rawL1[5]), Int32(rawL1[6]), Int32(rawL1[7])
    )
    var high = SIMD16<Int32>(
        Int32(rawH0[0]), Int32(rawH0[1]), Int32(rawH0[2]), Int32(rawH0[3]),
        Int32(rawH0[4]), Int32(rawH0[5]), Int32(rawH0[6]), Int32(rawH0[7]),
        Int32(rawH1[0]), Int32(rawH1[1]), Int32(rawH1[2]), Int32(rawH1[3]),
        Int32(rawH1[4]), Int32(rawH1[5]), Int32(rawH1[6]), Int32(rawH1[7])
    )
    
    var highShifted = SIMD16<Int32>(
        high[0], high[0], high[1], high[2], high[3], high[4], high[5], high[6],
        high[7], high[8], high[9], high[10], high[11], high[12], high[13], high[14]
    )
    low &-= mulShift16(cD, highShifted &+ high)
    
    var lowShifted = SIMD16<Int32>(
        low[1], low[2], low[3], low[4], low[5], low[6], low[7], low[8],
        low[9], low[10], low[11], low[12], low[13], low[14], low[15], low[15]
    )
    high &-= mulShift16(cG, low &+ lowShifted)
    
    highShifted = SIMD16<Int32>(
        high[0], high[0], high[1], high[2], high[3], high[4], high[5], high[6],
        high[7], high[8], high[9], high[10], high[11], high[12], high[13], high[14]
    )
    low &-= mulShift16(cB, highShifted &+ high)
    
    lowShifted = SIMD16<Int32>(
        low[1], low[2], low[3], low[4], low[5], low[6], low[7], low[8],
        low[9], low[10], low[11], low[12], low[13], low[14], low[15], low[15]
    )
    high &-= mulShift16(cA, low &+ lowShifted)
    
    let outL0 = SIMD8<Int16>(truncatingIfNeeded: low.lowHalf)
    let outL1 = SIMD8<Int16>(truncatingIfNeeded: low.highHalf)
    let outH0 = SIMD8<Int16>(truncatingIfNeeded: high.lowHalf)
    let outH1 = SIMD8<Int16>(truncatingIfNeeded: high.highHalf)
    
    let out0 = SIMD8<Int16>(outL0[0], outH0[0], outL0[1], outH0[1], outL0[2], outH0[2], outL0[3], outH0[3])
    let out1 = SIMD8<Int16>(outL0[4], outH0[4], outL0[5], outH0[5], outL0[6], outH0[6], outL0[7], outH0[7])
    let out2 = SIMD8<Int16>(outL1[0], outH1[0], outL1[1], outH1[1], outL1[2], outH1[2], outL1[3], outH1[3])
    let out3 = SIMD8<Int16>(outL1[4], outH1[4], outL1[5], outH1[5], outL1[6], outH1[6], outL1[7], outH1[7])
    
    UnsafeMutableRawPointer(base).storeBytes(of: out0, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 8).storeBytes(of: out1, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 16).storeBytes(of: out2, as: SIMD8<Int16>.self)
    UnsafeMutableRawPointer(base + 24).storeBytes(of: out3, as: SIMD8<Int16>.self)
}

// MARK: - 2D DWT

@inline(__always)
func cdf97Dwt2DBlock16(_ block: BlockView) {
    let base = block.base
    let width = block.stride
    // Row lifting
    for i in 0..<16 { cdf97LiftBlock16(UnsafeMutableBufferPointer(start: base + (i * width), count: 16), stride: 1) }
    transpose16x16InPlace(base, stride: width)
    for i in 0..<16 { cdf97LiftBlock16(UnsafeMutableBufferPointer(start: base + (i * width), count: 16), stride: 1) }
    transpose16x16InPlace(base, stride: width)
}

@inline(__always)
func inverseCdf97Dwt2DBlock16(_ block: BlockView) {
    let base = block.base
    let width = block.stride
    transpose16x16InPlace(base, stride: width)
    for i in 0..<16 { inverseCdf97LiftBlock16(UnsafeMutableBufferPointer(start: base + (i * width), count: 16), stride: 1) }
    transpose16x16InPlace(base, stride: width)
    for i in 0..<16 { inverseCdf97LiftBlock16(UnsafeMutableBufferPointer(start: base + (i * width), count: 16), stride: 1) }
}

@inline(__always)
func cdf97Dwt2DBlock32(_ block: BlockView) {
    let base = block.base
    let width = block.stride
    for i in 0..<32 { cdf97LiftBlock32(UnsafeMutableBufferPointer(start: base + (i * width), count: 32), stride: 1) }
    transpose32x32InPlace(base, stride: width)
    for i in 0..<32 { cdf97LiftBlock32(UnsafeMutableBufferPointer(start: base + (i * width), count: 32), stride: 1) }
    transpose32x32InPlace(base, stride: width)
}

@inline(__always)
func inverseCdf97Dwt2DBlock32(_ block: BlockView) {
    let base = block.base
    let width = block.stride
    transpose32x32InPlace(base, stride: width)
    for i in 0..<32 { inverseCdf97LiftBlock32(UnsafeMutableBufferPointer(start: base + (i * width), count: 32), stride: 1) }
    transpose32x32InPlace(base, stride: width)
    for i in 0..<32 { inverseCdf97LiftBlock32(UnsafeMutableBufferPointer(start: base + (i * width), count: 32), stride: 1) }
}

