// MARK: - Encode Context

import Foundation

public enum EncodeError: Error {
    case unsupportedArchitecture
}

@inline(__always)
func encodeExpGolomb<M: EntropyModelProvider>(val: UInt32, encoder: inout EntropyEncoder<M>) {
    var q = val
    var bits = 0
    while 0 < q {
        bits += 1
        q >>= 1
    }
    for _ in 0..<bits {
        encoder.encodeBypass(binVal: 0)
    }
    encoder.encodeBypass(binVal: 1)
    for i in stride(from: bits - 1, through: 0, by: -1) {
        encoder.encodeBypass(binVal: UInt8((val >> i) & 1))
    }
}

@inline(__always)
func encodeCoeffRun<M: EntropyModelProvider>(val: Int16, encoder: inout EntropyEncoder<M>, run: Int, isParentZero: Bool = false) {
    encoder.addPair(run: UInt32(run), val: val, isParentZero: isParentZero)
}

@inline(__always)
func blockEncode32V<M: EntropyModelProvider>(encoder: inout EntropyEncoder<M>, block: BlockView, parentBlock: BlockView?) {
    var lscpX = -1
    var lscpY = -1

    for x in stride(from: 32 - 1, through: 0, by: -1) {
        for y in stride(from: 32 - 1, through: 0, by: -1) {
            if block.rowPointer(y: y)[x] != 0 {
                lscpX = x
                lscpY = y
                break
            }
        }
        if lscpX != -1 { break }
    }

    if lscpX == -1 {
        encoder.encodeBypass(binVal: 0)
        for y in 0..<32 {
            let ptr = block.rowPointer(y: y)
            for x in 0..<32 { ptr[x] = 0 }
        }
        return
    }
    encoder.encodeBypass(binVal: 1)

    encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
    encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

    var run = 0
    var currentIdx = 0
    var startIdx = 0
    
    for x in 0...lscpX {
        let endY = if x == lscpX { lscpY } else { 32 - 1 }
        for y in 0...endY {
            let val = block.rowPointer(y: y)[x]
            if run == 0 { startIdx = currentIdx }
            if val == 0 {
                run += 1
            } else {
                let startX = startIdx / 32
                let startY = startIdx % 32
                let isParentZ: Bool
                if let pb = parentBlock {
                    isParentZ = pb.rowPointer(y: startY >> 1)[startX >> 1] == 0
                } else { isParentZ = false }
                encodeCoeffRun(val: val, encoder: &encoder, run: run, isParentZero: isParentZ)
                run = 0
            }
            currentIdx += 1
        }
    }
    for y in (lscpY + 1)..<32 { block.rowPointer(y: y)[lscpX] = 0 }
    for x in (lscpX + 1)..<32 {
        for y in 0..<32 { block.rowPointer(y: y)[x] = 0 }
    }
}

@inline(__always)
func blockEncode32H<M: EntropyModelProvider>(encoder: inout EntropyEncoder<M>, block: BlockView, parentBlock: BlockView?) {
    var lscpX = -1
    var lscpY = -1
    let zero16 = SIMD16<Int16>(repeating: 0)
    
    for y in stride(from: 32 - 1, through: 0, by: -1) {
        let ptr = block.rowPointer(y: y)
        let v1 = UnsafeRawPointer(ptr.advanced(by: 16)).loadUnaligned(as: SIMD16<Int16>.self)
        if any(v1 .!= zero16) {
            for x in stride(from: 31, through: 16, by: -1) {
                if ptr[x] != 0 {
                    lscpX = x
                    lscpY = y
                    break
                }
            }
        }
        if lscpX != -1 { break }
        
        let v0 = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD16<Int16>.self)
        if any(v0 .!= zero16) {
            for x in stride(from: 15, through: 0, by: -1) {
                if ptr[x] != 0 {
                    lscpX = x
                    lscpY = y
                    break
                }
            }
        }
        if lscpX != -1 { break }
    }
    
    if lscpX == -1 {
        encoder.encodeBypass(binVal: 0)
        for y in 0..<32 {
            let ptr = block.rowPointer(y: y)
            for x in 0..<32 { ptr[x] = 0 }
        }
        return
    }
    encoder.encodeBypass(binVal: 1)

    encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
    encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

    var run = 0
    var currentIdx = 0
    var startIdx = 0
    
    for y in 0...lscpY {
        let ptr = block.rowPointer(y: y)
        let endX = if y == lscpY { lscpX } else { 32 - 1 }
        for x in 0...endX {
            let val = ptr[x]
            if run == 0 { startIdx = currentIdx }
            if val == 0 {
                run += 1
            } else {
                let startY = startIdx / 32
                let startX = startIdx % 32
                let isParentZ: Bool
                if let pb = parentBlock {
                    isParentZ = pb.rowPointer(y: startY >> 1)[startX >> 1] == 0
                } else { isParentZ = false }
                encodeCoeffRun(val: val, encoder: &encoder, run: run, isParentZero: isParentZ)
                run = 0
            }
            currentIdx += 1
        }
    }
    let lscpPtr = block.rowPointer(y: lscpY)
    for x in (lscpX + 1)..<32 { lscpPtr[x] = 0 }
    for y in (lscpY + 1)..<32 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<32 { ptr[x] = 0 }
    }
}

@inline(__always)
func blockEncode16V<M: EntropyModelProvider>(encoder: inout EntropyEncoder<M>, block: BlockView, parentBlock: BlockView?) {
    var lscpX = -1
    var lscpY = -1
    
    for x in stride(from: 16 - 1, through: 0, by: -1) {
        for y in stride(from: 16 - 1, through: 0, by: -1) {
            if block.rowPointer(y: y)[x] != 0 {
                lscpX = x
                lscpY = y
                break
            }
        }
        if lscpX != -1 { break }
    }

    if lscpX == -1 {
        encoder.encodeBypass(binVal: 0)
        for y in 0..<16 {
            let ptr = block.rowPointer(y: y)
            for x in 0..<16 { ptr[x] = 0 }
        }
        return
    }
    encoder.encodeBypass(binVal: 1)

    encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
    encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

    var run = 0
    var currentIdx = 0
    var startIdx = 0
    
    for x in 0...lscpX {
        let endY = if x == lscpX { lscpY } else { 16 - 1 }
        for y in 0...endY {
            let val = block.rowPointer(y: y)[x]
            if run == 0 { startIdx = currentIdx }
            if val == 0 {
                run += 1
            } else {
                let startX = startIdx / 16
                let startY = startIdx % 16
                let isParentZ: Bool
                if let pb = parentBlock {
                    isParentZ = pb.rowPointer(y: startY >> 1)[startX >> 1] == 0
                } else { isParentZ = false }
                encodeCoeffRun(val: val, encoder: &encoder, run: run, isParentZero: isParentZ)
                run = 0
            }
            currentIdx += 1
        }
    }
    for y in (lscpY + 1)..<16 { block.rowPointer(y: y)[lscpX] = 0 }
    for x in (lscpX + 1)..<16 {
        for y in 0..<16 { block.rowPointer(y: y)[x] = 0 }
    }
}

@inline(__always)
func blockEncode16H<M: EntropyModelProvider>(encoder: inout EntropyEncoder<M>, block: BlockView, parentBlock: BlockView?) {
    var lscpX = -1
    var lscpY = -1
    let zero8 = SIMD8<Int16>(repeating: 0)
    
    for y in stride(from: 16 - 1, through: 0, by: -1) {
        let ptr = block.rowPointer(y: y)
        let v1 = UnsafeRawPointer(ptr.advanced(by: 8)).loadUnaligned(as: SIMD8<Int16>.self)
        if any(v1 .!= zero8) {
            for x in stride(from: 15, through: 8, by: -1) {
                if ptr[x] != 0 {
                    lscpX = x
                    lscpY = y
                    break
                }
            }
        }
        if lscpX != -1 { break }
        
        let v0 = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
        if any(v0 .!= zero8) {
            for x in stride(from: 7, through: 0, by: -1) {
                if ptr[x] != 0 {
                    lscpX = x
                    lscpY = y
                    break
                }
            }
        }
        if lscpX != -1 { break }
    }

    if lscpX == -1 {
        encoder.encodeBypass(binVal: 0)
        for y in 0..<16 {
            let ptr = block.rowPointer(y: y)
            for x in 0..<16 { ptr[x] = 0 }
        }
        return
    }
    encoder.encodeBypass(binVal: 1)

    encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
    encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

    var run = 0
    var currentIdx = 0
    var startIdx = 0
    
    for y in 0...lscpY {
        let ptr = block.rowPointer(y: y)
        let endX = if y == lscpY { lscpX } else { 16 - 1 }
        for x in 0...endX {
            let val = ptr[x]
            if run == 0 { startIdx = currentIdx }
            if val == 0 {
                run += 1
            } else {
                let startY = startIdx / 16
                let startX = startIdx % 16
                let isParentZ: Bool
                if let pb = parentBlock {
                    isParentZ = pb.rowPointer(y: startY >> 1)[startX >> 1] == 0
                } else { isParentZ = false }
                encodeCoeffRun(val: val, encoder: &encoder, run: run, isParentZero: isParentZ)
                run = 0
            }
            currentIdx += 1
        }
    }
    let lscpPtr = block.rowPointer(y: lscpY)
    for x in (lscpX + 1)..<16 { lscpPtr[x] = 0 }
    for y in (lscpY + 1)..<16 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<16 { ptr[x] = 0 }
    }
}

@inline(__always)
func blockEncode8V<M: EntropyModelProvider>(encoder: inout EntropyEncoder<M>, block: BlockView, parentBlock: BlockView?) {
    var lscpX = -1
    var lscpY = -1
    
    for x in stride(from: 8 - 1, through: 0, by: -1) {
        for y in stride(from: 8 - 1, through: 0, by: -1) {
            if block.rowPointer(y: y)[x] != 0 {
                lscpX = x
                lscpY = y
                break
            }
        }
        if lscpX != -1 { break }
    }
    

    if lscpX == -1 {
        encoder.encodeBypass(binVal: 0)
        for y in 0..<8 {
            let ptr = block.rowPointer(y: y)
            for x in 0..<8 { ptr[x] = 0 }
        }
        return
    }
    encoder.encodeBypass(binVal: 1)

    encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
    encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

    var run = 0
    var currentIdx = 0
    var startIdx = 0
    
    
    for x in 0...lscpX {
        let endY = if x == lscpX { lscpY } else { 8 - 1 }
        for y in 0...endY {
            let val = block.rowPointer(y: y)[x]
            if run == 0 { startIdx = currentIdx }
            if val == 0 {
                run += 1
            } else {
                let startX = startIdx / 8
                let startY = startIdx % 8
                let isParentZ: Bool
                if let pb = parentBlock {
                    isParentZ = pb.rowPointer(y: startY >> 1)[startX >> 1] == 0
                } else { isParentZ = false }
                encodeCoeffRun(val: val, encoder: &encoder, run: run, isParentZero: isParentZ)
                run = 0
            }
            currentIdx += 1
        }
    }
    for y in (lscpY + 1)..<8 { block.rowPointer(y: y)[lscpX] = 0 }
    for x in (lscpX + 1)..<8 {
        for y in 0..<8 { block.rowPointer(y: y)[x] = 0 }
    }
}

@inline(__always)
func blockEncode8H<M: EntropyModelProvider>(encoder: inout EntropyEncoder<M>, block: BlockView, parentBlock: BlockView?) {
    var lscpX = -1
    var lscpY = -1
    let zero4 = SIMD4<Int16>(repeating: 0)
    
    for y in stride(from: 8 - 1, through: 0, by: -1) {
        let ptr = block.rowPointer(y: y)
        let v1 = UnsafeRawPointer(ptr.advanced(by: 4)).loadUnaligned(as: SIMD4<Int16>.self)
        if any(v1 .!= zero4) {
            for x in stride(from: 7, through: 4, by: -1) {
                if ptr[x] != 0 {
                    lscpX = x
                    lscpY = y
                    break
                }
            }
        }
        if lscpX != -1 { break }
        
        let v0 = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD4<Int16>.self)
        if any(v0 .!= zero4) {
            for x in stride(from: 3, through: 0, by: -1) {
                if ptr[x] != 0 {
                    lscpX = x
                    lscpY = y
                    break
                }
            }
        }
        if lscpX != -1 { break }
    }

    if lscpX == -1 {
        encoder.encodeBypass(binVal: 0)
        for y in 0..<8 {
            let ptr = block.rowPointer(y: y)
            for x in 0..<8 { ptr[x] = 0 }
        }
        return
    }
    encoder.encodeBypass(binVal: 1)

    encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
    encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

    var run = 0
    var currentIdx = 0
    var startIdx = 0
    
    for y in 0...lscpY {
        let ptr = block.rowPointer(y: y)
        let endX = if y == lscpY { lscpX } else { 8 - 1 }
        for x in 0...endX {
            let val = ptr[x]
            if run == 0 { startIdx = currentIdx }
            if val == 0 {
                run += 1
            } else {
                let startY = startIdx / 8
                let startX = startIdx % 8
                let isParentZ: Bool
                if let pb = parentBlock {
                    isParentZ = pb.rowPointer(y: startY >> 1)[startX >> 1] == 0
                } else { isParentZ = false }
                encodeCoeffRun(val: val, encoder: &encoder, run: run, isParentZero: isParentZ)
                run = 0
            }
            currentIdx += 1
        }
    }
    let lscpPtr = block.rowPointer(y: lscpY)
    for x in (lscpX + 1)..<8 { lscpPtr[x] = 0 }
    for y in (lscpY + 1)..<8 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<8 { ptr[x] = 0 }
    }
}

@inline(__always)
func blockEncode4V<M: EntropyModelProvider>(encoder: inout EntropyEncoder<M>, block: BlockView, parentBlock: BlockView?) {
    var lscpX = -1
    var lscpY = -1
    
    for x in stride(from: 4 - 1, through: 0, by: -1) {
        for y in stride(from: 4 - 1, through: 0, by: -1) {
            if block.rowPointer(y: y)[x] != 0 {
                lscpX = x
                lscpY = y
                break
            }
        }
        if lscpX != -1 { break }
    }

    if lscpX == -1 {
        encoder.encodeBypass(binVal: 0)
        for y in 0..<4 {
            let ptr = block.rowPointer(y: y)
            for x in 0..<4 { ptr[x] = 0 }
        }
        return
    }
    encoder.encodeBypass(binVal: 1)

    encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
    encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

    var run = 0
    var currentIdx = 0
    var startIdx = 0
    
    for x in 0...lscpX {
        let endY = if x == lscpX { lscpY } else { 4 - 1 }
        for y in 0...endY {
            let val = block.rowPointer(y: y)[x]
            if run == 0 { startIdx = currentIdx }
            if val == 0 {
                run += 1
            } else {
                let startX = startIdx / 4
                let startY = startIdx % 4
                let isParentZ: Bool
                if let pb = parentBlock {
                    isParentZ = pb.rowPointer(y: startY >> 1)[startX >> 1] == 0
                } else { isParentZ = false }
                encodeCoeffRun(val: val, encoder: &encoder, run: run, isParentZero: isParentZ)
                run = 0
            }
            currentIdx += 1
        }
    }
    for y in (lscpY + 1)..<4 { block.rowPointer(y: y)[lscpX] = 0 }
    for x in (lscpX + 1)..<4 {
        for y in 0..<4 { block.rowPointer(y: y)[x] = 0 }
    }
}

@inline(__always)
func blockEncode4H<M: EntropyModelProvider>(encoder: inout EntropyEncoder<M>, block: BlockView, parentBlock: BlockView?) {
    var lscpX = -1
    var lscpY = -1
    let zero4 = SIMD4<Int16>(repeating: 0)
    
    for y in stride(from: 4 - 1, through: 0, by: -1) {
        let ptr = block.rowPointer(y: y)
        let v0 = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD4<Int16>.self)
        if any(v0 .!= zero4) {
            for x in stride(from: 3, through: 0, by: -1) {
                if ptr[x] != 0 {
                    lscpX = x
                    lscpY = y
                    break
                }
            }
        }
        if lscpX != -1 { break }
    }

    if lscpX == -1 {
        encoder.encodeBypass(binVal: 0)
        for y in 0..<4 {
            let ptr = block.rowPointer(y: y)
            for x in 0..<4 { ptr[x] = 0 }
        }
        return
    }
    encoder.encodeBypass(binVal: 1)

    encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
    encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

    var run = 0
    var currentIdx = 0
    var startIdx = 0
    
    for y in 0...lscpY {
        let ptr = block.rowPointer(y: y)
        let endX = if y == lscpY { lscpX } else { 4 - 1 }
        for x in 0...endX {
            let val = ptr[x]
            if run == 0 { startIdx = currentIdx }
            if val == 0 {
                run += 1
            } else {
                let startY = startIdx / 4
                let startX = startIdx % 4
                let isParentZ: Bool
                if let pb = parentBlock {
                    isParentZ = pb.rowPointer(y: startY >> 1)[startX >> 1] == 0
                } else { isParentZ = false }
                encodeCoeffRun(val: val, encoder: &encoder, run: run, isParentZero: isParentZ)
                run = 0
            }
            currentIdx += 1
        }
    }
    let lscpPtr = block.rowPointer(y: lscpY)
    for x in (lscpX + 1)..<4 { lscpPtr[x] = 0 }
    for y in (lscpY + 1)..<4 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<4 { ptr[x] = 0 }
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
func blockEncodeDPCM4<M: EntropyModelProvider>(encoder: inout EntropyEncoder<M>, block: BlockView, lastVal: inout Int16) {
    let ptr0 = block.rowPointer(y: 0)
    let ptr1 = block.rowPointer(y: 1)
    let ptr2 = block.rowPointer(y: 2)
    let ptr3 = block.rowPointer(y: 3)
    
    @inline(__always)
    func diffMED(_ x: SIMD4<Int16>, _ a: SIMD4<Int16>, _ b: SIMD4<Int16>, _ c: SIMD4<Int16>) -> SIMD4<Int16> {
        let pMin = SIMD4<Int16>(min(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
        let pMax = SIMD4<Int16>(max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
        let rawPred = a &+ b &- c
        let pred = SIMD4<Int16>(
            min(max(rawPred[0], pMin[0]), pMax[0]),
            min(max(rawPred[1], pMin[1]), pMax[1]),
            min(max(rawPred[2], pMin[2]), pMax[2]),
            min(max(rawPred[3], pMin[3]), pMax[3])
        )
        return x &- pred
    }

    let x0 = SIMD4<Int16>(ptr0[0], ptr0[1], ptr0[2], ptr0[3])
    let b0 = SIMD4<Int16>(lastVal, x0[0], x0[1], x0[2])
    let errSIMD0 = x0 &- b0

    let x1 = SIMD4<Int16>(ptr1[0], ptr1[1], ptr1[2], ptr1[3])
    let a1 = SIMD4<Int16>(x0[0], x1[0], x1[1], x1[2])
    let b1 = x0
    let c1 = SIMD4<Int16>(lastVal, b1[0], b1[1], b1[2])
    let errSIMD1 = diffMED(x1, a1, b1, c1)

    let x2 = SIMD4<Int16>(ptr2[0], ptr2[1], ptr2[2], ptr2[3])
    let a2 = SIMD4<Int16>(x1[0], x2[0], x2[1], x2[2])
    let b2 = x1
    let c2 = SIMD4<Int16>(x0[0], b2[0], b2[1], b2[2])
    let errSIMD2 = diffMED(x2, a2, b2, c2)

    let x3 = SIMD4<Int16>(ptr3[0], ptr3[1], ptr3[2], ptr3[3])
    let a3 = SIMD4<Int16>(x2[0], x3[0], x3[1], x3[2])
    let b3 = x2
    let c3 = SIMD4<Int16>(x1[0], b3[0], b3[1], b3[2])
    let errSIMD3 = diffMED(x3, a3, b3, c3)

    let errors = [
        errSIMD0[0], errSIMD0[1], errSIMD0[2], errSIMD0[3],
        errSIMD1[0], errSIMD1[1], errSIMD1[2], errSIMD1[3],
        errSIMD2[0], errSIMD2[1], errSIMD2[2], errSIMD2[3],
        errSIMD3[0], errSIMD3[1], errSIMD3[2], errSIMD3[3]
    ]
    
    var lscpIdx = -1
    let zero4 = SIMD4<Int16>(repeating: 0)
    switch true {
    case errSIMD3[3] != 0: lscpIdx = 15
    case errSIMD3[2] != 0: lscpIdx = 14
    case errSIMD3[1] != 0: lscpIdx = 13
    case errSIMD3[0] != 0: lscpIdx = 12
    case errSIMD2[3] != 0: lscpIdx = 11
    case errSIMD2[2] != 0: lscpIdx = 10
    case errSIMD2[1] != 0: lscpIdx = 9
    case errSIMD2[0] != 0: lscpIdx = 8
    case errSIMD1[3] != 0: lscpIdx = 7
    case errSIMD1[2] != 0: lscpIdx = 6
    case errSIMD1[1] != 0: lscpIdx = 5
    case errSIMD1[0] != 0: lscpIdx = 4
    case errSIMD0[3] != 0: lscpIdx = 3
    case errSIMD0[2] != 0: lscpIdx = 2
    case errSIMD0[1] != 0: lscpIdx = 1
    case errSIMD0[0] != 0: lscpIdx = 0
    default: lscpIdx = -1
    }

    if lscpIdx == -1 {
        encoder.encodeBypass(binVal: 0)
    } else {
        encoder.encodeBypass(binVal: 1)
        let lscpX = lscpIdx % 4
        let lscpY = lscpIdx / 4
        encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
        encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

        var run = 0
        for i in 0...lscpIdx {
            let diff = errors[i]
            if diff == 0 {
                run += 1
            } else {
                encodeCoeffRun(val: diff, encoder: &encoder, run: run)
                run = 0
            }
        }
        
        for i in 0..<16 {
            let y = i / 4
            let x = i % 4
            let val: Int16 = if i <= lscpIdx { errors[i] } else { 0 }
            block.rowPointer(y: y)[x] = val
        }
        
        let ptrY0 = block.rowPointer(y: 0)
        ptrY0[0] = ptrY0[0] &+ lastVal
        for x in 1..<4 {
            ptrY0[x] = ptrY0[x] &+ ptrY0[x - 1]
        }
        
        var reconLast = ptrY0[3]
        for y in 1..<4 {
            let ptrY = block.rowPointer(y: y)
            let ptrPrevY = block.rowPointer(y: y - 1)
            ptrY[0] = ptrY[0] &+ ptrPrevY[0]
            for x in 1..<4 {
                ptrY[x] = ptrY[x] &+ predictMED(ptrY[x - 1], ptrPrevY[x], ptrPrevY[x - 1])
            }
            reconLast = ptrY[3]
        }
        lastVal = reconLast
        return
    }

    lastVal = ptr3[3]
}

// why: extracted from inner function to avoid closure capture overhead on hot path
@inline(__always)
func blockEncodeDPCMErrorMED(_ x: Int16, _ a: Int16, _ b: Int16, _ c: Int16) -> Int16 {
    let ia = Int(a), ib = Int(b), ic = Int(c)
    let predicted: Int16
    switch true {
    case ia <= ic && ib <= ic:
        predicted = Int16(truncatingIfNeeded: min(ia, ib))
    case ic <= ia && ic <= ib:
        predicted = Int16(truncatingIfNeeded: max(ia, ib))
    default:
        predicted = Int16(truncatingIfNeeded: ia + ib - ic)
    }
    return x &- predicted
}

@inline(__always)
func blockEncodeDPCM8<M: EntropyModelProvider>(encoder: inout EntropyEncoder<M>, block: BlockView, lastVal: inout Int16) {
    withUnsafeTemporaryAllocation(of: Int16.self, capacity: 64) { ptrErr in
        guard let baseErr = ptrErr.baseAddress else { return }
        
        var last: Int16 = lastVal
        
        let ptrY0 = block.rowPointer(y: 0)
        baseErr[0] = ptrY0[0] &- last
        for x in 1..<8 {
            baseErr[x] = ptrY0[x] &- ptrY0[x - 1]
        }
        last = ptrY0[7]
        
        for y in 1..<8 {
            let ptrY = block.rowPointer(y: y)
            let ptrPrevY = block.rowPointer(y: y - 1)
            let rowOffset = y * 8
            
            baseErr[rowOffset] = ptrY[0] &- ptrPrevY[0]
            for x in 1..<8 {
                baseErr[rowOffset + x] = blockEncodeDPCMErrorMED(ptrY[x], ptrY[x - 1], ptrPrevY[x], ptrPrevY[x - 1])
            }
            last = ptrY[7]
        }
        
        var lscpIdx = -1
        for i in stride(from: 63, through: 0, by: -1) {
            if baseErr[i] != 0 {
                lscpIdx = i
                break
            }
        }
        
        if lscpIdx == -1 {
            encoder.encodeBypass(binVal: 0)
            let ptrY = block.rowPointer(y: 7)
            lastVal = ptrY[7]
            return
        }

        encoder.encodeBypass(binVal: 1)
        let lscpX = UInt32(lscpIdx % 8)
        let lscpY = UInt32(lscpIdx / 8)
        encodeExpGolomb(val: lscpX, encoder: &encoder)
        encodeExpGolomb(val: lscpY, encoder: &encoder)
        
        lastVal = last
        
        var run = 0
        for i in 0...lscpIdx {
            let diff = baseErr[i]
            if diff == 0 {
                run += 1
            } else {
                encodeCoeffRun(val: diff, encoder: &encoder, run: run)
                run = 0
            }
        }
        
        for i in 0..<64 {
            let y = i / 8
            let x = i % 8
            let val: Int16 = if i <= lscpIdx { baseErr[i] } else { 0 }
            block.rowPointer(y: y)[x] = val
        }
        
        let reconPtrY0 = block.rowPointer(y: 0)
        reconPtrY0[0] = reconPtrY0[0] &+ lastVal
        for x in 1..<8 {
            reconPtrY0[x] = reconPtrY0[x] &+ reconPtrY0[x - 1]
        }
        
        var reconLast = reconPtrY0[7]
        for y in 1..<8 {
            let ptrY = block.rowPointer(y: y)
            let ptrPrevY = block.rowPointer(y: y - 1)
            ptrY[0] = ptrY[0] &+ ptrPrevY[0]
            for x in 1..<8 {
                ptrY[x] = ptrY[x] &+ predictMED(ptrY[x - 1], ptrPrevY[x], ptrPrevY[x - 1])
            }
            reconLast = ptrY[7]
        }
        lastVal = reconLast
    }
}

@inline(__always)
func blockEncodeDPCM16<M: EntropyModelProvider>(encoder: inout EntropyEncoder<M>, block: BlockView, lastVal: inout Int16) {
    let originalLastVal = lastVal
    var lscpIdx = -1
    var last: Int16 = originalLastVal

    let ptrY0 = block.rowPointer(y: 0)
    let diffFirst = ptrY0[0] &- last
    if diffFirst != 0 {
        lscpIdx = 0
    }
    for x in 1..<16 {
        let diff = ptrY0[x] &- ptrY0[x - 1]
        if diff != 0 {
            lscpIdx = x
        }
    }
    last = ptrY0[15]

    for y in 1..<16 {
        let ptrY = block.rowPointer(y: y)
        let ptrPrevY = block.rowPointer(y: y - 1)
        
        let diffY0 = ptrY[0] &- ptrPrevY[0]
        if diffY0 != 0 {
            lscpIdx = y * 16
        }
        for x in 1..<16 {
            let diff = blockEncodeDPCMErrorMED(ptrY[x], ptrY[x - 1], ptrPrevY[x], ptrPrevY[x - 1])
            if diff != 0 {
                lscpIdx = y * 16 + x
            }
        }
        last = ptrY[15]
    }

    if lscpIdx == -1 {
        encoder.encodeBypass(binVal: 0)
        let ptrY = block.rowPointer(y: 15)
        lastVal = ptrY[15]
        return
    }

    encoder.encodeBypass(binVal: 1)
    let lscpX = UInt32(lscpIdx % 16)
    let lscpY = UInt32(lscpIdx / 16)
    encodeExpGolomb(val: lscpX, encoder: &encoder)
    encodeExpGolomb(val: lscpY, encoder: &encoder)

    lastVal = last

    var run = 0
    var currentIdx = 0
    last = originalLastVal

    let ptrY0_run = block.rowPointer(y: 0)
    let diffRunFirst = ptrY0_run[0] &- last
    if diffRunFirst == 0 {
        run += 1
    } else {
        encodeCoeffRun(val: diffRunFirst, encoder: &encoder, run: run)
        run = 0
    }
    currentIdx += 1

    if currentIdx <= lscpIdx {
        for x in 1..<16 {
            let diff = ptrY0_run[x] &- ptrY0_run[x - 1]
            if diff == 0 {
                run += 1
            } else {
                encodeCoeffRun(val: diff, encoder: &encoder, run: run)
                run = 0
            }
            currentIdx += 1
            if lscpIdx < currentIdx { break }
        }
    }
    last = ptrY0_run[15]

    if currentIdx <= lscpIdx {
        outerLoop: for y in 1..<16 {
            let ptrY = block.rowPointer(y: y)
            let ptrPrevY = block.rowPointer(y: y - 1)
            
            let diffY0 = ptrY[0] &- ptrPrevY[0]
            if diffY0 == 0 {
                run += 1
            } else {
                encodeCoeffRun(val: diffY0, encoder: &encoder, run: run)
                run = 0
            }
            currentIdx += 1
            if lscpIdx < currentIdx { break outerLoop }
            
            for x in 1..<16 {
                let diff = blockEncodeDPCMErrorMED(ptrY[x], ptrY[x - 1], ptrPrevY[x], ptrPrevY[x - 1])
                if diff == 0 {
                    run += 1
                } else {
                    encodeCoeffRun(val: diff, encoder: &encoder, run: run)
                    run = 0
                }
                currentIdx += 1
                if lscpIdx < currentIdx { break outerLoop }
            }
            last = ptrY[15]
        }
    }
    lastVal = last
}

// MARK: - Transform Functions

@inline(__always)
func isEffectivelyZero32(data base: UnsafeMutablePointer<Int16>, threshold: Int) -> Bool {
    let th = Int16(threshold)
    let thPos = SIMD16<Int16>(repeating: th)
    let thNeg = SIMD16<Int16>(repeating: -th)

    let lowerHalfBase = base + 16 * 32
    for i in stride(from: 0, to: 512, by: 16) {
        let vec: SIMD16<Int16> = UnsafeRawPointer(lowerHalfBase + i).loadUnaligned(as: SIMD16<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
    }
    for y in 0..<16 {
        let ptr = base + y * 32 + 16
        let vec: SIMD16<Int16> = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD16<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
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
func isEffectivelyZero16(data base: UnsafeMutablePointer<Int16>, threshold: Int) -> Bool {
    let th = Int16(threshold)
    let thPos = SIMD8<Int16>(repeating: th)
    let thNeg = SIMD8<Int16>(repeating: -th)

    let lowerHalfBase = base + 8 * 16
    for i in stride(from: 0, to: 128, by: 8) {
        let vec: SIMD8<Int16> = UnsafeRawPointer(lowerHalfBase + i).loadUnaligned(as: SIMD8<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
    }
    for y in 0..<8 {
        let ptr = base + y * 16 + 8
        let vec: SIMD8<Int16> = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
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
func isEffectivelyZero8(data base: UnsafeMutablePointer<Int16>, threshold: Int) -> Bool {
    let th = Int16(threshold)
    let thPos = SIMD4<Int16>(repeating: th)
    let thNeg = SIMD4<Int16>(repeating: -th)

    let lowerHalfBase = base + 4 * 8
    for i in stride(from: 0, to: 32, by: 4) {
        let vec: SIMD4<Int16> = UnsafeRawPointer(lowerHalfBase + i).loadUnaligned(as: SIMD4<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
    }
    for y in 0..<4 {
        let ptr = base + y * 8 + 4
        let vec: SIMD4<Int16> = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD4<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
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
func checkQuadrants16x16(base: UnsafeMutablePointer<Int16>, stride: Int, q0: inout Bool, q1: inout Bool, q2: inout Bool, q3: inout Bool) {
    let zero8 = SIMD8<Int16>(repeating: 0)
    for y in 0..<8 {
        let ptr = base.advanced(by: y * stride)
        if q0 != true {
            let v = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
            if any(v .!= zero8) { q0 = true }
        }
        if q1 != true {
            let v = UnsafeRawPointer(ptr.advanced(by: 8)).loadUnaligned(as: SIMD8<Int16>.self)
            if any(v .!= zero8) { q1 = true }
        }
    }
    for y in 8..<16 {
        let ptr = base.advanced(by: y * stride)
        if q2 != true {
            let v = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
            if any(v .!= zero8) { q2 = true }
        }
        if q3 != true {
            let v = UnsafeRawPointer(ptr.advanced(by: 8)).loadUnaligned(as: SIMD8<Int16>.self)
            if any(v .!= zero8) { q3 = true }
        }
    }
}

@inline(__always)
func shouldSplit32WithLL(data base: UnsafeMutablePointer<Int16>) -> Bool {
    var q0 = false, q1 = false, q2 = false, q3 = false
    
    checkQuadrants16x16(base: base, stride: 32, q0: &q0, q1: &q1, q2: &q2, q3: &q3)
    if (q0 && q1 && q2 && q3) != true {
        checkQuadrants16x16(base: base + 16, stride: 32, q0: &q0, q1: &q1, q2: &q2, q3: &q3)
    }
    if (q0 && q1 && q2 && q3) != true {
        checkQuadrants16x16(base: base + 16 * 32, stride: 32, q0: &q0, q1: &q1, q2: &q2, q3: &q3)
    }
    if (q0 && q1 && q2 && q3) != true {
        checkQuadrants16x16(base: base + 16 * 32 + 16, stride: 32, q0: &q0, q1: &q1, q2: &q2, q3: &q3)
    }
    
    // If not all 4 quadrants are busy, splitting avoids encoding zeros.
    return (q0 && q1 && q2 && q3) != true
}

@inline(__always)
func shouldSplit32WithoutLL(data base: UnsafeMutablePointer<Int16>) -> Bool {
    // LL quadrant is skipped because it is encoded separately (DPCM path)
    var q0 = false, q1 = false, q2 = false, q3 = false
    
    if (q0 && q1 && q2 && q3) != true {
        checkQuadrants16x16(base: base + 16, stride: 32, q0: &q0, q1: &q1, q2: &q2, q3: &q3)
    }
    if (q0 && q1 && q2 && q3) != true {
        checkQuadrants16x16(base: base + 16 * 32, stride: 32, q0: &q0, q1: &q1, q2: &q2, q3: &q3)
    }
    if (q0 && q1 && q2 && q3) != true {
        checkQuadrants16x16(base: base + 16 * 32 + 16, stride: 32, q0: &q0, q1: &q1, q2: &q2, q3: &q3)
    }
    
    return (q0 && q1 && q2 && q3) != true
}

@inline(__always)
func checkQuadrants8x8(base: UnsafeMutablePointer<Int16>, stride: Int, q0: inout Bool, q1: inout Bool, q2: inout Bool, q3: inout Bool) {
    let zero4 = SIMD4<Int16>(repeating: 0)
    for y in 0..<4 {
        let ptr = base.advanced(by: y * stride)
        if q0 != true {
            let v = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD4<Int16>.self)
            if any(v .!= zero4) { q0 = true }
        }
        if q1 != true {
            let v = UnsafeRawPointer(ptr.advanced(by: 4)).loadUnaligned(as: SIMD4<Int16>.self)
            if any(v .!= zero4) { q1 = true }
        }
    }
    for y in 4..<8 {
        let ptr = base.advanced(by: y * stride)
        if q2 != true {
            let v = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD4<Int16>.self)
            if any(v .!= zero4) { q2 = true }
        }
        if q3 != true {
            let v = UnsafeRawPointer(ptr.advanced(by: 4)).loadUnaligned(as: SIMD4<Int16>.self)
            if any(v .!= zero4) { q3 = true }
        }
    }
}

@inline(__always)
func shouldSplit16(data base: UnsafeMutablePointer<Int16>) -> Bool {
    // LL quadrant is skipped because it is encoded separately (DPCM path)
    var q0 = false, q1 = false, q2 = false, q3 = false
    
    if (q0 && q1 && q2 && q3) != true {
        checkQuadrants8x8(base: base + 8, stride: 16, q0: &q0, q1: &q1, q2: &q2, q3: &q3)
    }
    if (q0 && q1 && q2 && q3) != true {
        checkQuadrants8x8(base: base + 8 * 16, stride: 16, q0: &q0, q1: &q1, q2: &q2, q3: &q3)
    }
    if (q0 && q1 && q2 && q3) != true {
        checkQuadrants8x8(base: base + 8 * 16 + 8, stride: 16, q0: &q0, q1: &q1, q2: &q2, q3: &q3)
    }
    
    return (q0 && q1 && q2 && q3) != true
}

@inline(__always)
func isEffectivelyZeroBase4(data base: UnsafeMutablePointer<Int16>, threshold: Int) -> Bool {
    // I-frame path: LL must be exactly zero (DPCM requires exact values)
    for y in 0..<2 {
        let ptr = base + y * 4
        let vec: SIMD2<Int16> = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD2<Int16>.self)
        if vec[0] != 0 || vec[1] != 0 { return false }
    }
    
    let th = Int16(threshold)
    let thPos = SIMD2<Int16>(repeating: th)
    let thNeg = SIMD2<Int16>(repeating: -th)
    
    // Check Subbands
    let lowerHalfBase = base + 2 * 4
    for i in stride(from: 0, to: 8, by: 2) {
        let vec: SIMD2<Int16> = UnsafeRawPointer(lowerHalfBase + i).loadUnaligned(as: SIMD2<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
    }
    for y in 0..<2 {
        let ptr = base + y * 4 + 2
        let vec: SIMD2<Int16> = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD2<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
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
func isEffectivelyZeroBase4PFrame(data base: UnsafeMutablePointer<Int16>, threshold: Int) -> Bool {
    // P-frame path: LL is threshold-checked (residual values after motion compensation)
    let th = Int16(threshold)
    let thPos = SIMD2<Int16>(repeating: th)
    let thNeg = SIMD2<Int16>(repeating: -th)
    
    // Check LL with threshold
    for y in 0..<2 {
        let ptr = base + y * 4
        let vec: SIMD2<Int16> = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD2<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
    }
    
    // Check Subbands
    let lowerHalfBase = base + 2 * 4
    for i in stride(from: 0, to: 8, by: 2) {
        let vec: SIMD2<Int16> = UnsafeRawPointer(lowerHalfBase + i).loadUnaligned(as: SIMD2<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
    }
    for y in 0..<2 {
        let ptr = base + (y * 4) + 2
        let vec: SIMD2<Int16> = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD2<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
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
func isEffectivelyZeroBase32(data base: UnsafeMutablePointer<Int16>, threshold: Int) -> Bool {
    // Check LL
    let zeroVec16 = SIMD16<Int16>(repeating: 0)
    for y in 0..<16 {
        let ptr = base + y * 32
        let vec: SIMD16<Int16> = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD16<Int16>.self)
        let mask = vec .!= zeroVec16
        if any(mask) { return false }
    }
    
    // Check Subbands
    let th = Int16(threshold)
    let thPos = SIMD16<Int16>(repeating: th)
    let thNeg = SIMD16<Int16>(repeating: -th)

    let lowerHalfBase = base + 16 * 32
    for i in stride(from: 0, to: 512, by: 16) {
        let vec: SIMD16<Int16> = UnsafeRawPointer(lowerHalfBase + i).loadUnaligned(as: SIMD16<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
    }
    for y in 0..<16 {
        let ptr = base + y * 32 + 16
        let vec: SIMD16<Int16> = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD16<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
    }

    for i in stride(from: 0, to: 512, by: 16) {
        let ptr = UnsafeMutableRawPointer(lowerHalfBase + i).assumingMemoryBound(to: SIMD16<Int16>.self)
        ptr.pointee = zeroVec16
    }
    for y in 0..<16 {
        let ptr = UnsafeMutableRawPointer(base + y * 32 + 16).assumingMemoryBound(to: SIMD16<Int16>.self)
        ptr.pointee = zeroVec16
    }
    return true
}

enum EncodeTask32 {
    case encode16
    case split8(Bool, Bool, Bool, Bool)
}

@inline(__always)
func encodePlaneSubbands32(blocks: inout [BlockView], zeroThreshold: Int, parentBlocks: [BlockView]?, sads: [Int]? = nil, colCount: Int = 0, rowCount: Int = 0) -> [UInt8] {
    var bwFlags = BypassWriter()
    var tasks: [(Int, EncodeTask32)] = []
    tasks.reserveCapacity(blocks.count)
    
    // Spatial adaptive threshold: when colCount/rowCount are provided,
    // apply higher zero-thresholds to peripheral blocks where human
    // visual attention is lower, increasing zero-block rate at edges.
    let useSpatialWeight = 1 < colCount && 1 < rowCount
    
    var zeroCount = 0
    for i in blocks.indices {
        let blockThreshold: Int
        if useSpatialWeight {
            let col = i % colCount
            let row = i / colCount
            let weight = spatialWeight(blockCol: col, blockRow: row, colCount: colCount, rowCount: rowCount)
            blockThreshold = (max(1, zeroThreshold) * weight) / 1024
        } else {
            blockThreshold = zeroThreshold
        }
        let isZero = isEffectivelyZero32(data: blocks[i].base, threshold: blockThreshold)
        if isZero {
            bwFlags.writeBit(true)
            let view = blocks[i]
            let half = 32 / 2
            let base = view.base
            let hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 32)
            let lhView = BlockView(base: base.advanced(by: half * 32), width: half, height: half, stride: 32)
            let hhView = BlockView(base: base.advanced(by: half * 32 + half), width: half, height: half, stride: 32)
            hlView.clearAll()
            lhView.clearAll()
            hhView.clearAll()
            zeroCount += 1
        } else {
            bwFlags.writeBit(false)
            
            let forceSplit = shouldSplit32WithoutLL(data: blocks[i].base)
            if forceSplit {
                bwFlags.writeBit(true)
                
                bwFlags.writeBit(false) // TL isZero = false
                bwFlags.writeBit(false) // TL MB_Type = false (No further split)
                
                bwFlags.writeBit(false) // TR isZero = false
                bwFlags.writeBit(false) // TR MB_Type = false
                
                bwFlags.writeBit(false) // BL isZero = false
                bwFlags.writeBit(false) // BL MB_Type = false
                
                bwFlags.writeBit(false) // BR isZero = false
                bwFlags.writeBit(false) // BR MB_Type = false
                
                tasks.append((i, .split8(true, true, true, true)))
            } else {
                bwFlags.writeBit(false) // MB_Type = false
                tasks.append((i, .encode16))
            }
        }
    }
    bwFlags.flush()
    debugLog({
        let zeroPermyriad = (zeroCount * 10000) / max(1, blocks.count)
        let rateStr = "\(zeroPermyriad / 100).\(zeroPermyriad / 10 % 10)"
        return "    [Subbands] blocks=\(blocks.count) zeroBlocks=\(zeroCount) zeroRate=\(rateStr)%"
    }())
    
    var encoder = EntropyEncoder<DynamicEntropyModel>()
    
    if let pb = parentBlocks {
        for (i, task) in tasks {
            if i < pb.count {
                let pBlock = pb[i]
                let pView = pBlock
                let pSubs = getSubbands16(view: pView)
                let view = blocks[i]
                let subs = getSubbands32(view: view)
                encodeSubbands32WithParent(task: task, encoder: &encoder, subs: subs, parentHL: pSubs.hl, parentLH: pSubs.lh, parentHH: pSubs.hh)
            } else {
                let view = blocks[i]
                let subs = getSubbands32(view: view)
                encodeSubbands32WithoutParent(task: task, encoder: &encoder, subs: subs)
            }
        }
    } else {
        for (i, task) in tasks {
            let view = blocks[i]
            let subs = getSubbands32(view: view)
            encodeSubbands32WithoutParent(task: task, encoder: &encoder, subs: subs)
        }
    }
    
    encoder.flush()
    var out = bwFlags.bytes
    out.append(contentsOf: encoder.getData())
    return out
}

enum EncodeTask16 {
    case encode8
    case split4(Bool, Bool, Bool, Bool)
}

@inline(__always)
func encodePlaneSubbands16(blocks: inout [BlockView], zeroThreshold: Int, parentBlocks: [BlockView]?, sads: [Int]? = nil, colCount: Int = 0, rowCount: Int = 0) -> [UInt8] {
    var bwFlags = BypassWriter()
    var tasks: [(Int, EncodeTask16)] = []
    tasks.reserveCapacity(blocks.count)
    
    let useSpatialWeight = 1 < colCount && 1 < rowCount
    
    var zeroCount = 0
    for i in blocks.indices {
        let safeCol = if 0 < colCount { colCount } else { 1 }
        let col = i % safeCol
        let row = i / safeCol
        let colCount32 = (colCount + 1) / 2
        let sadIdx = ((row / 2) * colCount32) + (col / 2)
        
        let isHighError = if let sads = sads, sadIdx < sads.count, 1500 <= sads[sadIdx] { true } else { false }
        let blockThreshold: Int
        switch true {
        case isHighError:
            // Adaptive AC Preservation (緩和版): if the prediction error is significant,
            // half the zero thresholds to preserve edge details and suppress ghosts,
            // while still discarding the ±1 mosquito noise.
            blockThreshold = max(1, zeroThreshold / 2)
        case useSpatialWeight:
            let weight = spatialWeight(blockCol: col, blockRow: row, colCount: colCount, rowCount: rowCount)
            blockThreshold = (max(1, zeroThreshold) * weight) / 1024
        default:
            blockThreshold = zeroThreshold
        }
        if isEffectivelyZero16(data: blocks[i].base, threshold: blockThreshold) {
            bwFlags.writeBit(true)
            let view = blocks[i]
            let half = 16 / 2
            let base = view.base
            let hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 16)
            let lhView = BlockView(base: base.advanced(by: half * 16), width: half, height: half, stride: 16)
            let hhView = BlockView(base: base.advanced(by: half * 16 + half), width: half, height: half, stride: 16)
            hlView.clearAll()
            lhView.clearAll()
            hhView.clearAll()
            zeroCount += 1
        } else {
            bwFlags.writeBit(false)
            let forceSplit = shouldSplit16(data: blocks[i].base)
            if forceSplit {
                bwFlags.writeBit(true)
                
                bwFlags.writeBit(false)
                bwFlags.writeBit(false)
                
                bwFlags.writeBit(false)
                bwFlags.writeBit(false)
                
                bwFlags.writeBit(false)
                bwFlags.writeBit(false)
                
                bwFlags.writeBit(false)
                bwFlags.writeBit(false)
                
                tasks.append((i, .split4(true, true, true, true)))
            } else {
                bwFlags.writeBit(false)
                tasks.append((i, .encode8))
            }
        }
    }
    bwFlags.flush()
    debugLog({
        let zeroPermyriad = (zeroCount * 10000) / max(1, blocks.count)
        let rateStr = "\(zeroPermyriad / 100).\(zeroPermyriad / 10 % 10)"
        return "    [Subbands] blocks=\(blocks.count) zeroBlocks=\(zeroCount) zeroRate=\(rateStr)%"
    }())
    
    var encoder = EntropyEncoder<DynamicEntropyModel>()
    
    if let pb = parentBlocks {
        for (i, task) in tasks {
            if i < pb.count {
                let pBlock = pb[i]
                let pView = pBlock
                let pSubs = getSubbands8(view: pView)
                let view = blocks[i]
                let subs = getSubbands16(view: view)
                encodeSubbands16WithParent(task: task, encoder: &encoder, subs: subs, parentHL: pSubs.hl, parentLH: pSubs.lh, parentHH: pSubs.hh)
            } else {
                let view = blocks[i]
                let subs = getSubbands16(view: view)
                encodeSubbands16WithoutParent(task: task, encoder: &encoder, subs: subs)
            }
        }
    } else {
        for (i, task) in tasks {
            let view = blocks[i]
            let subs = getSubbands16(view: view)
            encodeSubbands16WithoutParent(task: task, encoder: &encoder, subs: subs)
        }
    }
    
    encoder.flush()
    var out = bwFlags.bytes
    out.append(contentsOf: encoder.getData())
    return out
}

@inline(__always)
func encodePlaneBaseSubbands8(blocks: inout [BlockView], zeroThreshold: Int) -> [UInt8] {
    var bwFlags = BypassWriter()
    var nonZeroIndices: [Int] = []
    
    for i in blocks.indices {
        let isZero = isEffectivelyZeroBase4(data: blocks[i].base, threshold: zeroThreshold)
        if isZero {
            bwFlags.writeBit(true)
            bwFlags.writeBit(false)
            blocks[i].clearAll()
        } else {
            bwFlags.writeBit(false)
            bwFlags.writeBit(false)
            nonZeroIndices.append(i)
        }
    }
    bwFlags.flush()
    let zeroCount = blocks.count - nonZeroIndices.count
    debugLog({
        let zeroPermyriad = (zeroCount * 10000) / max(1, blocks.count)
        let rateStr = "\(zeroPermyriad / 100).\(zeroPermyriad / 10 % 10)"
        return "    [BaseSubbands] blocks=\(blocks.count) zeroBlocks=\(zeroCount) zeroRate=\(rateStr)%"
    }())
    
    var encoder = EntropyEncoder<DynamicEntropyModel>()
    var lastVal: Int16 = 0
    
    var nzCur = 0
    let nzCount = nonZeroIndices.count
    for i in blocks.indices {
        if nzCur < nzCount && nonZeroIndices[nzCur] == i {
            nzCur += 1

            let view = blocks[i]
            let subs = getSubbands8(view: view)
            blockEncodeDPCM4(encoder: &encoder, block: subs.ll, lastVal: &lastVal)
            blockEncode4V(encoder: &encoder, block: subs.hl, parentBlock: nil)
            blockEncode4H(encoder: &encoder, block: subs.lh, parentBlock: nil)
            blockEncode4H(encoder: &encoder, block: subs.hh, parentBlock: nil)
        } else {
            lastVal = 0
        }
    }
    
    encoder.flush()
    var out = bwFlags.bytes
    out.append(contentsOf: encoder.getData())
    return out
}

@inline(__always)
func encodePlaneBaseSubbands8PFrame(blocks: inout [BlockView], zeroThreshold: Int) -> [UInt8] {
    var bwFlags = BypassWriter()
    var nonZeroIndices: [Int] = []
    
    for i in blocks.indices {
        let isZero = isEffectivelyZeroBase4PFrame(data: blocks[i].base, threshold: zeroThreshold)
        if isZero {
            bwFlags.writeBit(true)
            bwFlags.writeBit(false)
            blocks[i].clearAll()
        } else {
            bwFlags.writeBit(false)
            bwFlags.writeBit(false)
            nonZeroIndices.append(i)
        }
    }
    bwFlags.flush()
    let zeroCount = blocks.count - nonZeroIndices.count
    debugLog({
        let zeroPermyriad = (zeroCount * 10000) / max(1, blocks.count)
        let rateStr = "\(zeroPermyriad / 100).\(zeroPermyriad / 10 % 10)"
        return "    [BaseSubbands] blocks=\(blocks.count) zeroBlocks=\(zeroCount) zeroRate=\(rateStr)%"
    }())
    
    var encoder = EntropyEncoder<DynamicEntropyModel>()
    
    var nzCur = 0
    let nzCount = nonZeroIndices.count
    for i in blocks.indices {
        if nzCur < nzCount && nonZeroIndices[nzCur] == i {
            nzCur += 1

            let view = blocks[i]
            let subs = getSubbands8(view: view)
            blockEncode4H(encoder: &encoder, block: subs.ll, parentBlock: nil)
            blockEncode4V(encoder: &encoder, block: subs.hl, parentBlock: nil)
            blockEncode4H(encoder: &encoder, block: subs.lh, parentBlock: nil)
            blockEncode4H(encoder: &encoder, block: subs.hh, parentBlock: nil)
        }
    }
    
    encoder.flush()
    var out = bwFlags.bytes
    out.append(contentsOf: encoder.getData())
    return out
}

enum EncodeTaskBase32 {
    case skip
    case encode16
    case split8(Bool, Bool, Bool, Bool)
}

@inline(__always)
func encodePlaneBaseSubbands32(blocks: inout [BlockView], zeroThreshold: Int) -> [UInt8] {
    var bwFlags = BypassWriter()
    var tasks: [(Int, EncodeTaskBase32)] = []
    tasks.reserveCapacity(blocks.count)
    
    var zeroCount = 0
    for i in blocks.indices {
        let isZero = isEffectivelyZeroBase32(data: blocks[i].base, threshold: zeroThreshold)
        if isZero {
            bwFlags.writeBit(true)
            blocks[i].clearAll()
            tasks.append((i, .skip))
            zeroCount += 1
        } else {
            bwFlags.writeBit(false)
            let forceSplit = shouldSplit32WithLL(data: blocks[i].base)
            if forceSplit {
                bwFlags.writeBit(true)
                
                bwFlags.writeBit(false) // TL
                bwFlags.writeBit(false)
                
                bwFlags.writeBit(false) // TR
                bwFlags.writeBit(false)
                
                bwFlags.writeBit(false) // BL
                bwFlags.writeBit(false)
                
                bwFlags.writeBit(false) // BR
                bwFlags.writeBit(false)
                
                tasks.append((i, .split8(true, true, true, true)))
            } else {
                bwFlags.writeBit(false)
                tasks.append((i, .encode16))
            }
        }
    }
    bwFlags.flush()
    debugLog({
        let zeroPermyriad32 = (zeroCount * 10000) / max(1, blocks.count)
        return "    [BaseSubbands32] blocks=\(blocks.count) zeroBlocks=\(zeroCount) zeroRate=\(zeroPermyriad32 / 100).\(zeroPermyriad32 / 10 % 10)%"
    }())
    
    var encoder = EntropyEncoder<StaticDPCMEntropyModel>()
    var lastVal: Int16 = 0
    
    for (i, task) in tasks {
        switch task {
        case .skip:
            lastVal = 0
            
        case .encode16:
            let view = blocks[i]
            let subs = getSubbands32(view: view)
            blockEncodeDPCM16(encoder: &encoder, block: subs.ll, lastVal: &lastVal)
            blockEncode16V(encoder: &encoder, block: subs.hl, parentBlock: nil)
            blockEncode16H(encoder: &encoder, block: subs.lh, parentBlock: nil)
            blockEncode16H(encoder: &encoder, block: subs.hh, parentBlock: nil)
        
        case .split8(let tl, let tr, let bl, let br):
            let view = blocks[i]
            let subs = getSubbands32(view: view)
            if tl {
                let ll = BlockView(base: subs.ll.base, width: 8, height: 8, stride: 32)
                let hl = BlockView(base: subs.hl.base, width: 8, height: 8, stride: 32)
                let lh = BlockView(base: subs.lh.base, width: 8, height: 8, stride: 32)
                let hh = BlockView(base: subs.hh.base, width: 8, height: 8, stride: 32)
                blockEncodeDPCM8(encoder: &encoder, block: ll, lastVal: &lastVal)
                blockEncode8H(encoder: &encoder, block: hl, parentBlock: nil)
                blockEncode8H(encoder: &encoder, block: lh, parentBlock: nil)
                blockEncode8H(encoder: &encoder, block: hh, parentBlock: nil)
            }
            if tr {
                let ll = BlockView(base: subs.ll.base.advanced(by: 8), width: 8, height: 8, stride: 32)
                let hl = BlockView(base: subs.hl.base.advanced(by: 8), width: 8, height: 8, stride: 32)
                let lh = BlockView(base: subs.lh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
                let hh = BlockView(base: subs.hh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
                blockEncodeDPCM8(encoder: &encoder, block: ll, lastVal: &lastVal)
                blockEncode8H(encoder: &encoder, block: hl, parentBlock: nil)
                blockEncode8H(encoder: &encoder, block: lh, parentBlock: nil)
                blockEncode8H(encoder: &encoder, block: hh, parentBlock: nil)
            }
            if bl {
                let ll = BlockView(base: subs.ll.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                let hl = BlockView(base: subs.hl.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                let lh = BlockView(base: subs.lh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                let hh = BlockView(base: subs.hh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                blockEncodeDPCM8(encoder: &encoder, block: ll, lastVal: &lastVal)
                blockEncode8H(encoder: &encoder, block: hl, parentBlock: nil)
                blockEncode8H(encoder: &encoder, block: lh, parentBlock: nil)
                blockEncode8H(encoder: &encoder, block: hh, parentBlock: nil)
            }
            if br {
                let ll = BlockView(base: subs.ll.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                let hl = BlockView(base: subs.hl.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                let lh = BlockView(base: subs.lh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                let hh = BlockView(base: subs.hh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                blockEncodeDPCM8(encoder: &encoder, block: ll, lastVal: &lastVal)
                blockEncode8H(encoder: &encoder, block: hl, parentBlock: nil)
                blockEncode8H(encoder: &encoder, block: lh, parentBlock: nil)
                blockEncode8H(encoder: &encoder, block: hh, parentBlock: nil)
            }
        }
    }
    
    encoder.flush()
    var out = bwFlags.bytes
    out.append(contentsOf: encoder.getData())
    return out
}

// MARK: - Dedicated Subband Process Functions

@inline(__always)
func encodeSubbands32WithParent<M: EntropyModelProvider>(
    task: EncodeTask32,
    encoder: inout EntropyEncoder<M>,
    subs: Subbands,
    parentHL: BlockView,
    parentLH: BlockView,
    parentHH: BlockView
) {
    switch task {
    case .encode16:
        blockEncode16V(encoder: &encoder, block: subs.hl, parentBlock: parentHL)
        blockEncode16H(encoder: &encoder, block: subs.lh, parentBlock: parentLH)
        blockEncode16H(encoder: &encoder, block: subs.hh, parentBlock: parentHH)
    case .split8(let tl, let tr, let bl, let br):
        if tl {
            let pbHL = BlockView(base: parentHL.base, width: 4, height: 4, stride: parentHL.stride)
            let pbLH = BlockView(base: parentLH.base, width: 4, height: 4, stride: parentLH.stride)
            let pbHH = BlockView(base: parentHH.base, width: 4, height: 4, stride: parentHH.stride)
            let hl = BlockView(base: subs.hl.base, width: 8, height: 8, stride: 32)
            let lh = BlockView(base: subs.lh.base, width: 8, height: 8, stride: 32)
            let hh = BlockView(base: subs.hh.base, width: 8, height: 8, stride: 32)
            blockEncode8H(encoder: &encoder, block: hl, parentBlock: pbHL)
            blockEncode8H(encoder: &encoder, block: lh, parentBlock: pbLH)
            blockEncode8H(encoder: &encoder, block: hh, parentBlock: pbHH)
        } else {
            BlockView(base: subs.hl.base, width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.lh.base, width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.hh.base, width: 8, height: 8, stride: 32).clearAll()
        }
        if tr {
            let pbHL = BlockView(base: parentHL.base.advanced(by: 4), width: 4, height: 4, stride: parentHL.stride)
            let pbLH = BlockView(base: parentLH.base.advanced(by: 4), width: 4, height: 4, stride: parentLH.stride)
            let pbHH = BlockView(base: parentHH.base.advanced(by: 4), width: 4, height: 4, stride: parentHH.stride)
            let hl = BlockView(base: subs.hl.base.advanced(by: 8), width: 8, height: 8, stride: 32)
            let lh = BlockView(base: subs.lh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
            let hh = BlockView(base: subs.hh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
            blockEncode8H(encoder: &encoder, block: hl, parentBlock: pbHL)
            blockEncode8H(encoder: &encoder, block: lh, parentBlock: pbLH)
            blockEncode8H(encoder: &encoder, block: hh, parentBlock: pbHH)
        } else {
            BlockView(base: subs.hl.base.advanced(by: 8), width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.lh.base.advanced(by: 8), width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.hh.base.advanced(by: 8), width: 8, height: 8, stride: 32).clearAll()
        }
        if bl {
            let pbHL = BlockView(base: parentHL.base.advanced(by: 4 * parentHL.stride), width: 4, height: 4, stride: parentHL.stride)
            let pbLH = BlockView(base: parentLH.base.advanced(by: 4 * parentLH.stride), width: 4, height: 4, stride: parentLH.stride)
            let pbHH = BlockView(base: parentHH.base.advanced(by: 4 * parentHH.stride), width: 4, height: 4, stride: parentHH.stride)
            let hl = BlockView(base: subs.hl.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
            let lh = BlockView(base: subs.lh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
            let hh = BlockView(base: subs.hh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
            blockEncode8H(encoder: &encoder, block: hl, parentBlock: pbHL)
            blockEncode8H(encoder: &encoder, block: lh, parentBlock: pbLH)
            blockEncode8H(encoder: &encoder, block: hh, parentBlock: pbHH)
        } else {
            BlockView(base: subs.hl.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.lh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.hh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32).clearAll()
        }
        if br {
            let pbHL = BlockView(base: parentHL.base.advanced(by: 4 * parentHL.stride + 4), width: 4, height: 4, stride: parentHL.stride)
            let pbLH = BlockView(base: parentLH.base.advanced(by: 4 * parentLH.stride + 4), width: 4, height: 4, stride: parentLH.stride)
            let pbHH = BlockView(base: parentHH.base.advanced(by: 4 * parentHH.stride + 4), width: 4, height: 4, stride: parentHH.stride)
            let hl = BlockView(base: subs.hl.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
            let lh = BlockView(base: subs.lh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
            let hh = BlockView(base: subs.hh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
            blockEncode8H(encoder: &encoder, block: hl, parentBlock: pbHL)
            blockEncode8H(encoder: &encoder, block: lh, parentBlock: pbLH)
            blockEncode8H(encoder: &encoder, block: hh, parentBlock: pbHH)
        } else {
            BlockView(base: subs.hl.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.lh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.hh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32).clearAll()
        }
    }
}

@inline(__always)
func encodeSubbands32WithoutParent<M: EntropyModelProvider>(
    task: EncodeTask32,
    encoder: inout EntropyEncoder<M>,
    subs: Subbands
) {
    switch task {
    case .encode16:
        blockEncode16V(encoder: &encoder, block: subs.hl, parentBlock: nil)
        blockEncode16H(encoder: &encoder, block: subs.lh, parentBlock: nil)
        blockEncode16H(encoder: &encoder, block: subs.hh, parentBlock: nil)
    case .split8(let tl, let tr, let bl, let br):
        if tl {
            let hl = BlockView(base: subs.hl.base, width: 8, height: 8, stride: 32)
            let lh = BlockView(base: subs.lh.base, width: 8, height: 8, stride: 32)
            let hh = BlockView(base: subs.hh.base, width: 8, height: 8, stride: 32)
            blockEncode8H(encoder: &encoder, block: hl, parentBlock: nil)
            blockEncode8H(encoder: &encoder, block: lh, parentBlock: nil)
            blockEncode8H(encoder: &encoder, block: hh, parentBlock: nil)
        } else {
            BlockView(base: subs.hl.base, width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.lh.base, width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.hh.base, width: 8, height: 8, stride: 32).clearAll()
        }
        if tr {
            let hl = BlockView(base: subs.hl.base.advanced(by: 8), width: 8, height: 8, stride: 32)
            let lh = BlockView(base: subs.lh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
            let hh = BlockView(base: subs.hh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
            blockEncode8H(encoder: &encoder, block: hl, parentBlock: nil)
            blockEncode8H(encoder: &encoder, block: lh, parentBlock: nil)
            blockEncode8H(encoder: &encoder, block: hh, parentBlock: nil)
        } else {
            BlockView(base: subs.hl.base.advanced(by: 8), width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.lh.base.advanced(by: 8), width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.hh.base.advanced(by: 8), width: 8, height: 8, stride: 32).clearAll()
        }
        if bl {
            let hl = BlockView(base: subs.hl.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
            let lh = BlockView(base: subs.lh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
            let hh = BlockView(base: subs.hh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
            blockEncode8H(encoder: &encoder, block: hl, parentBlock: nil)
            blockEncode8H(encoder: &encoder, block: lh, parentBlock: nil)
            blockEncode8H(encoder: &encoder, block: hh, parentBlock: nil)
        } else {
            BlockView(base: subs.hl.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.lh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.hh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32).clearAll()
        }
        if br {
            let hl = BlockView(base: subs.hl.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
            let lh = BlockView(base: subs.lh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
            let hh = BlockView(base: subs.hh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
            blockEncode8H(encoder: &encoder, block: hl, parentBlock: nil)
            blockEncode8H(encoder: &encoder, block: lh, parentBlock: nil)
            blockEncode8H(encoder: &encoder, block: hh, parentBlock: nil)
        } else {
            BlockView(base: subs.hl.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.lh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.hh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32).clearAll()
        }
    }
}

@inline(__always)
func encodeSubbands16WithParent<M: EntropyModelProvider>(
    task: EncodeTask16,
    encoder: inout EntropyEncoder<M>,
    subs: Subbands,
    parentHL: BlockView,
    parentLH: BlockView,
    parentHH: BlockView
) {
    switch task {
    case .encode8:
        blockEncode8V(encoder: &encoder, block: subs.hl, parentBlock: parentHL)
        blockEncode8H(encoder: &encoder, block: subs.lh, parentBlock: parentLH)
        blockEncode8H(encoder: &encoder, block: subs.hh, parentBlock: parentHH)
    case .split4(let tl, let tr, let bl, let br):
        if tl {
            let pbHL = BlockView(base: parentHL.base, width: 2, height: 2, stride: parentHL.stride)
            let pbLH = BlockView(base: parentLH.base, width: 2, height: 2, stride: parentLH.stride)
            let pbHH = BlockView(base: parentHH.base, width: 2, height: 2, stride: parentHH.stride)
            let hl = BlockView(base: subs.hl.base, width: 4, height: 4, stride: 16)
            let lh = BlockView(base: subs.lh.base, width: 4, height: 4, stride: 16)
            let hh = BlockView(base: subs.hh.base, width: 4, height: 4, stride: 16)
            blockEncode4H(encoder: &encoder, block: hl, parentBlock: pbHL)
            blockEncode4H(encoder: &encoder, block: lh, parentBlock: pbLH)
            blockEncode4H(encoder: &encoder, block: hh, parentBlock: pbHH)
        }
        if tr {
            let pbHL = BlockView(base: parentHL.base.advanced(by: 2), width: 2, height: 2, stride: parentHL.stride)
            let pbLH = BlockView(base: parentLH.base.advanced(by: 2), width: 2, height: 2, stride: parentLH.stride)
            let pbHH = BlockView(base: parentHH.base.advanced(by: 2), width: 2, height: 2, stride: parentHH.stride)
            let hl = BlockView(base: subs.hl.base.advanced(by: 4), width: 4, height: 4, stride: 16)
            let lh = BlockView(base: subs.lh.base.advanced(by: 4), width: 4, height: 4, stride: 16)
            let hh = BlockView(base: subs.hh.base.advanced(by: 4), width: 4, height: 4, stride: 16)
            blockEncode4H(encoder: &encoder, block: hl, parentBlock: pbHL)
            blockEncode4H(encoder: &encoder, block: lh, parentBlock: pbLH)
            blockEncode4H(encoder: &encoder, block: hh, parentBlock: pbHH)
        }
        if bl {
            let pbHL = BlockView(base: parentHL.base.advanced(by: 2 * parentHL.stride), width: 2, height: 2, stride: parentHL.stride)
            let pbLH = BlockView(base: parentLH.base.advanced(by: 2 * parentLH.stride), width: 2, height: 2, stride: parentLH.stride)
            let pbHH = BlockView(base: parentHH.base.advanced(by: 2 * parentHH.stride), width: 2, height: 2, stride: parentHH.stride)
            let hl = BlockView(base: subs.hl.base.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
            let lh = BlockView(base: subs.lh.base.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
            let hh = BlockView(base: subs.hh.base.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
            blockEncode4H(encoder: &encoder, block: hl, parentBlock: pbHL)
            blockEncode4H(encoder: &encoder, block: lh, parentBlock: pbLH)
            blockEncode4H(encoder: &encoder, block: hh, parentBlock: pbHH)
        }
        if br {
            let pbHL = BlockView(base: parentHL.base.advanced(by: 2 * parentHL.stride + 2), width: 2, height: 2, stride: parentHL.stride)
            let pbLH = BlockView(base: parentLH.base.advanced(by: 2 * parentLH.stride + 2), width: 2, height: 2, stride: parentLH.stride)
            let pbHH = BlockView(base: parentHH.base.advanced(by: 2 * parentHH.stride + 2), width: 2, height: 2, stride: parentHH.stride)
            let hl = BlockView(base: subs.hl.base.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
            let lh = BlockView(base: subs.lh.base.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
            let hh = BlockView(base: subs.hh.base.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
            blockEncode4H(encoder: &encoder, block: hl, parentBlock: pbHL)
            blockEncode4H(encoder: &encoder, block: lh, parentBlock: pbLH)
            blockEncode4H(encoder: &encoder, block: hh, parentBlock: pbHH)
        }
    }
}

@inline(__always)
func encodeSubbands16WithoutParent<M: EntropyModelProvider>(
    task: EncodeTask16,
    encoder: inout EntropyEncoder<M>,
    subs: Subbands
) {
    switch task {
    case .encode8:
        blockEncode8V(encoder: &encoder, block: subs.hl, parentBlock: nil)
        blockEncode8H(encoder: &encoder, block: subs.lh, parentBlock: nil)
        blockEncode8H(encoder: &encoder, block: subs.hh, parentBlock: nil)
    case .split4(let tl, let tr, let bl, let br):
        if tl {
            let hl = BlockView(base: subs.hl.base, width: 4, height: 4, stride: 16)
            let lh = BlockView(base: subs.lh.base, width: 4, height: 4, stride: 16)
            let hh = BlockView(base: subs.hh.base, width: 4, height: 4, stride: 16)
            blockEncode4H(encoder: &encoder, block: hl, parentBlock: nil)
            blockEncode4H(encoder: &encoder, block: lh, parentBlock: nil)
            blockEncode4H(encoder: &encoder, block: hh, parentBlock: nil)
        }
        if tr {
            let hl = BlockView(base: subs.hl.base.advanced(by: 4), width: 4, height: 4, stride: 16)
            let lh = BlockView(base: subs.lh.base.advanced(by: 4), width: 4, height: 4, stride: 16)
            let hh = BlockView(base: subs.hh.base.advanced(by: 4), width: 4, height: 4, stride: 16)
            blockEncode4H(encoder: &encoder, block: hl, parentBlock: nil)
            blockEncode4H(encoder: &encoder, block: lh, parentBlock: nil)
            blockEncode4H(encoder: &encoder, block: hh, parentBlock: nil)
        }
        if bl {
            let hl = BlockView(base: subs.hl.base.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
            let lh = BlockView(base: subs.lh.base.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
            let hh = BlockView(base: subs.hh.base.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
            blockEncode4H(encoder: &encoder, block: hl, parentBlock: nil)
            blockEncode4H(encoder: &encoder, block: lh, parentBlock: nil)
            blockEncode4H(encoder: &encoder, block: hh, parentBlock: nil)
        }
        if br {
            let hl = BlockView(base: subs.hl.base.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
            let lh = BlockView(base: subs.lh.base.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
            let hh = BlockView(base: subs.hh.base.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
            blockEncode4H(encoder: &encoder, block: hl, parentBlock: nil)
            blockEncode4H(encoder: &encoder, block: lh, parentBlock: nil)
            blockEncode4H(encoder: &encoder, block: hh, parentBlock: nil)
        }
    }
}