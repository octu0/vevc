// MARK: - Encode Context


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

// why: blockEncode32V/H were dead code (no call sites in EncodeTransform.swift or tests) — removed

@inline(__always)
func blockEncode16V<M: EntropyModelProvider>(encoder: inout EntropyEncoder<M>, block: BlockView) {
    var lscpX = -1
    var lscpY = -1
    let zero8 = SIMD8<Int16>(repeating: 0)

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
            UnsafeMutableRawPointer(ptr).storeBytes(of: zero8, as: SIMD8<Int16>.self)
            UnsafeMutableRawPointer(ptr.advanced(by: 8)).storeBytes(of: zero8, as: SIMD8<Int16>.self)
        }
        return
    }
    encoder.encodeBypass(binVal: 1)

    encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
    encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

    var run = 0

    for x in 0...lscpX {
        let endY = if x == lscpX { lscpY } else { 16 - 1 }
        for y in 0...endY {
            let val = block.rowPointer(y: y)[x]
            if val == 0 {
                run += 1
            }
            if val != 0 {
                encodeCoeffRun(val: val, encoder: &encoder, run: run)
                run = 0
            }
        }
    }

    for y in (lscpY + 1)..<16 {
        block.rowPointer(y: y)[lscpX] = 0
    }
    for x in (lscpX + 1)..<16 {
        for y in 0..<16 {
            block.rowPointer(y: y)[x] = 0
        }
    }
}

@inline(__always)
func blockEncode16VWithParent<M: EntropyModelProvider>(encoder: inout EntropyEncoder<M>, block: BlockView, parentBlock: BlockView) {
    var lscpX = -1
    var lscpY = -1
    let zero8 = SIMD8<Int16>(repeating: 0)

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
            UnsafeMutableRawPointer(ptr).storeBytes(of: zero8, as: SIMD8<Int16>.self)
            UnsafeMutableRawPointer(ptr.advanced(by: 8)).storeBytes(of: zero8, as: SIMD8<Int16>.self)
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
            if run == 0 {
                startIdx = currentIdx
            }
            if val == 0 {
                run += 1
            }
            if val != 0 {
                let startX = startIdx / 16
                let startY = startIdx % 16
                let isParentZ = parentBlock.rowPointer(y: startY >> 1)[startX >> 1] == 0
                encodeCoeffRun(val: val, encoder: &encoder, run: run, isParentZero: isParentZ)
                run = 0
            }
            currentIdx += 1
        }
    }

    for y in (lscpY + 1)..<16 {
        block.rowPointer(y: y)[lscpX] = 0
    }
    for x in (lscpX + 1)..<16 {
        for y in 0..<16 {
            block.rowPointer(y: y)[x] = 0
        }
    }
}

@inline(__always)
func blockEncode16H<M: EntropyModelProvider>(encoder: inout EntropyEncoder<M>, block: BlockView) {
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
            UnsafeMutableRawPointer(ptr).storeBytes(of: zero8, as: SIMD8<Int16>.self)
            UnsafeMutableRawPointer(ptr.advanced(by: 8)).storeBytes(of: zero8, as: SIMD8<Int16>.self)
        }
        return
    }
    encoder.encodeBypass(binVal: 1)

    encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
    encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

    var run = 0
    
    for y in 0...lscpY {
        let ptr = block.rowPointer(y: y)
        let endX = if y == lscpY { lscpX } else { 16 - 1 }
        for x in 0...endX {
            let val = ptr[x]
            if val == 0 {
                run += 1
            }
            if val != 0 {
                encodeCoeffRun(val: val, encoder: &encoder, run: run)
                run = 0
            }
        }
    }
    let lscpPtr = block.rowPointer(y: lscpY)
    for x in (lscpX + 1)..<16 {
        lscpPtr[x] = 0
    }
    for y in (lscpY + 1)..<16 {
        let ptr = block.rowPointer(y: y)
        UnsafeMutableRawPointer(ptr).storeBytes(of: zero8, as: SIMD8<Int16>.self)
        UnsafeMutableRawPointer(ptr.advanced(by: 8)).storeBytes(of: zero8, as: SIMD8<Int16>.self)
    }
}

@inline(__always)
func blockEncode16HWithParent<M: EntropyModelProvider>(encoder: inout EntropyEncoder<M>, block: BlockView, parentBlock: BlockView) {
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
            UnsafeMutableRawPointer(ptr).storeBytes(of: zero8, as: SIMD8<Int16>.self)
            UnsafeMutableRawPointer(ptr.advanced(by: 8)).storeBytes(of: zero8, as: SIMD8<Int16>.self)
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
            if run == 0 {
                startIdx = currentIdx
            }
            if val == 0 {
                run += 1
            }
            if val != 0 {
                let startY = startIdx / 16
                let startX = startIdx % 16
                let isParentZ = parentBlock.rowPointer(y: startY >> 1)[startX >> 1] == 0
                encodeCoeffRun(val: val, encoder: &encoder, run: run, isParentZero: isParentZ)
                run = 0
            }
            currentIdx += 1
        }
    }
    let lscpPtr = block.rowPointer(y: lscpY)
    for x in (lscpX + 1)..<16 {
        lscpPtr[x] = 0
    }
    for y in (lscpY + 1)..<16 {
        let ptr = block.rowPointer(y: y)
        UnsafeMutableRawPointer(ptr).storeBytes(of: zero8, as: SIMD8<Int16>.self)
        UnsafeMutableRawPointer(ptr.advanced(by: 8)).storeBytes(of: zero8, as: SIMD8<Int16>.self)
    }
}

@inline(__always)
func blockEncode8V<M: EntropyModelProvider>(encoder: inout EntropyEncoder<M>, block: BlockView) {
    var lscpX = -1
    var lscpY = -1
    let zero4 = SIMD4<Int16>(repeating: 0)

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
            UnsafeMutableRawPointer(ptr).storeBytes(of: zero4, as: SIMD4<Int16>.self)
            UnsafeMutableRawPointer(ptr.advanced(by: 4)).storeBytes(of: zero4, as: SIMD4<Int16>.self)
        }
        return
    }
    encoder.encodeBypass(binVal: 1)

    encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
    encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

    var run = 0
    
    for x in 0...lscpX {
        let endY = if x == lscpX { lscpY } else { 8 - 1 }
        for y in 0...endY {
            let val = block.rowPointer(y: y)[x]
            if val == 0 {
                run += 1
            }
            if val != 0 {
                encodeCoeffRun(val: val, encoder: &encoder, run: run)
                run = 0
            }
        }
    }
    for y in (lscpY + 1)..<8 {
        block.rowPointer(y: y)[lscpX] = 0
    }
    for x in (lscpX + 1)..<8 {
        for y in 0..<8 {
            block.rowPointer(y: y)[x] = 0
        }
    }
}

@inline(__always)
func blockEncode8VWithParent<M: EntropyModelProvider>(encoder: inout EntropyEncoder<M>, block: BlockView, parentBlock: BlockView) {
    var lscpX = -1
    var lscpY = -1
    let zero4 = SIMD4<Int16>(repeating: 0)

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
            UnsafeMutableRawPointer(ptr).storeBytes(of: zero4, as: SIMD4<Int16>.self)
            UnsafeMutableRawPointer(ptr.advanced(by: 4)).storeBytes(of: zero4, as: SIMD4<Int16>.self)
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
            }
            if val != 0 {
                let startX = startIdx / 8
                let startY = startIdx % 8
                let isParentZ = parentBlock.rowPointer(y: startY >> 1)[startX >> 1] == 0
                encodeCoeffRun(val: val, encoder: &encoder, run: run, isParentZero: isParentZ)
                run = 0
            }
            currentIdx += 1
        }
    }
    for y in (lscpY + 1)..<8 {
        block.rowPointer(y: y)[lscpX] = 0
    }
    for x in (lscpX + 1)..<8 {
        for y in 0..<8 {
            block.rowPointer(y: y)[x] = 0
        }
    }
}

@inline(__always)
func blockEncode8H<M: EntropyModelProvider>(encoder: inout EntropyEncoder<M>, block: BlockView) {
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
            UnsafeMutableRawPointer(ptr).storeBytes(of: zero4, as: SIMD4<Int16>.self)
            UnsafeMutableRawPointer(ptr.advanced(by: 4)).storeBytes(of: zero4, as: SIMD4<Int16>.self)
        }
        return
    }
    encoder.encodeBypass(binVal: 1)

    encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
    encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

    var run = 0

    for y in 0...lscpY {
        let ptr = block.rowPointer(y: y)
        let endX = if y == lscpY { lscpX } else { 8 - 1 }
        for x in 0...endX {
            let val = ptr[x]
            if val == 0 {
                run += 1
            }
            if val != 0 {
                encodeCoeffRun(val: val, encoder: &encoder, run: run)
                run = 0
            }
        }
    }
    let lscpPtr = block.rowPointer(y: lscpY)
    for x in (lscpX + 1)..<8 {
        lscpPtr[x] = 0
    }
    for y in (lscpY + 1)..<8 {
        let ptr = block.rowPointer(y: y)
        UnsafeMutableRawPointer(ptr).storeBytes(of: zero4, as: SIMD4<Int16>.self)
        UnsafeMutableRawPointer(ptr.advanced(by: 4)).storeBytes(of: zero4, as: SIMD4<Int16>.self)
    }
}

@inline(__always)
func blockEncode8HWithParent<M: EntropyModelProvider>(encoder: inout EntropyEncoder<M>, block: BlockView, parentBlock: BlockView) {
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
            UnsafeMutableRawPointer(ptr).storeBytes(of: zero4, as: SIMD4<Int16>.self)
            UnsafeMutableRawPointer(ptr.advanced(by: 4)).storeBytes(of: zero4, as: SIMD4<Int16>.self)
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
            }
            if val != 0 {
                let startY = startIdx / 8
                let startX = startIdx % 8
                let isParentZ = parentBlock.rowPointer(y: startY >> 1)[startX >> 1] == 0
                encodeCoeffRun(val: val, encoder: &encoder, run: run, isParentZero: isParentZ)
                run = 0
            }
            currentIdx += 1
        }
    }
    let lscpPtr = block.rowPointer(y: lscpY)
    for x in (lscpX + 1)..<8 {
        lscpPtr[x] = 0
    }
    for y in (lscpY + 1)..<8 {
        let ptr = block.rowPointer(y: y)
        UnsafeMutableRawPointer(ptr).storeBytes(of: zero4, as: SIMD4<Int16>.self)
        UnsafeMutableRawPointer(ptr.advanced(by: 4)).storeBytes(of: zero4, as: SIMD4<Int16>.self)
    }
}

// why: blockEncode4V is only called with parentBlock: nil, so parentBlock parameter removed entirely
@inline(__always)
func blockEncode4V<M: EntropyModelProvider>(encoder: inout EntropyEncoder<M>, block: BlockView) {
    var lscpX = -1
    var lscpY = -1
    let zero4 = SIMD4<Int16>(repeating: 0)

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
            UnsafeMutableRawPointer(ptr).storeBytes(of: zero4, as: SIMD4<Int16>.self)
        }
        return
    }
    encoder.encodeBypass(binVal: 1)

    encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
    encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

    var run = 0
    
    for x in 0...lscpX {
        let endY = if x == lscpX { lscpY } else { 4 - 1 }
        for y in 0...endY {
            let val = block.rowPointer(y: y)[x]
            if val == 0 {
                run += 1
            }
            if val != 0 {
                encodeCoeffRun(val: val, encoder: &encoder, run: run)
                run = 0
            }
        }
    }
    for y in (lscpY + 1)..<4 {
        block.rowPointer(y: y)[lscpX] = 0
    }
    for x in (lscpX + 1)..<4 {
        for y in 0..<4 {
            block.rowPointer(y: y)[x] = 0
        }
    }
}

@inline(__always)
func blockEncode4H<M: EntropyModelProvider>(encoder: inout EntropyEncoder<M>, block: BlockView) {
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
            UnsafeMutableRawPointer(ptr).storeBytes(of: zero4, as: SIMD4<Int16>.self)
        }
        return
    }
    encoder.encodeBypass(binVal: 1)

    encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
    encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

    var run = 0

    for y in 0...lscpY {
        let ptr = block.rowPointer(y: y)
        let endX = if y == lscpY { lscpX } else { 4 - 1 }
        for x in 0...endX {
            let val = ptr[x]
            if val == 0 {
                run += 1
            }
            if val != 0 {
                encodeCoeffRun(val: val, encoder: &encoder, run: run)
                run = 0
            }
        }
    }
    let lscpPtr = block.rowPointer(y: lscpY)
    for x in (lscpX + 1)..<4 {
        lscpPtr[x] = 0
    }
    for y in (lscpY + 1)..<4 {
        let ptr = block.rowPointer(y: y)
        UnsafeMutableRawPointer(ptr).storeBytes(of: zero4, as: SIMD4<Int16>.self)
    }
}

@inline(__always)
func blockEncode4HWithParent<M: EntropyModelProvider>(encoder: inout EntropyEncoder<M>, block: BlockView, parentBlock: BlockView) {
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
            UnsafeMutableRawPointer(ptr).storeBytes(of: zero4, as: SIMD4<Int16>.self)
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
            if run == 0 {
                startIdx = currentIdx
            }
            if val == 0 {
                run += 1
            }
            if val != 0 {
                let startY = startIdx / 4
                let startX = startIdx % 4
                let isParentZ = parentBlock.rowPointer(y: startY >> 1)[startX >> 1] == 0
                encodeCoeffRun(val: val, encoder: &encoder, run: run, isParentZero: isParentZ)
                run = 0
            }
            currentIdx += 1
        }
    }
    let lscpPtr = block.rowPointer(y: lscpY)
    for x in (lscpX + 1)..<4 {
        lscpPtr[x] = 0
    }
    for y in (lscpY + 1)..<4 {
        let ptr = block.rowPointer(y: y)
        UnsafeMutableRawPointer(ptr).storeBytes(of: zero4, as: SIMD4<Int16>.self)
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

@inline(__always)
func blockEncodeDPCM4<M: EntropyModelProvider>(encoder: inout EntropyEncoder<M>, block: BlockView, lastVal: inout Int16) {
    let ptr0 = block.rowPointer(y: 0)
    let ptr1 = block.rowPointer(y: 1)
    let ptr2 = block.rowPointer(y: 2)
    let ptr3 = block.rowPointer(y: 3)
    
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

