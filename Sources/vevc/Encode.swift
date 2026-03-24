// MARK: - Encode Context

import Foundation

public enum EncodeError: Error {
    case unsupportedArchitecture
}

@inline(__always)
func debugLog(_ message: String) {
    if ProcessInfo.processInfo.environment["VEVC_DEBUG"] != nil {
        fputs(message + "\n", stderr)
    }
}

@inline(__always)
func encodeExpGolomb(val: UInt32, encoder: inout EntropyEncoder) {
    var q = val
    var bits = 0
    while q > 0 {
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
func encodeCoeffRun(val: Int16, encoder: inout EntropyEncoder, run: Int, isParentZero: Bool = false) {
    encoder.addPair(run: UInt32(run), val: val, isParentZero: isParentZero)
}

@inline(__always)
func blockEncode32(encoder: inout EntropyEncoder, block: BlockView, parentBlock: BlockView?) {
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
        return
    }
    encoder.encodeBypass(binVal: 1)

    encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
    encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

    var run = 0
    var isParentZero = false
    for y in 0...lscpY {
        let ptr = block.rowPointer(y: y)
        let endX = (y == lscpY) ? lscpX : (32 - 1)
        for x in 0...endX {
            if run == 0 {
                if let pb = parentBlock {
                    isParentZero = pb.rowPointer(y: y / 2)[x / 2] == 0
                }
            }
            let val = ptr[x]
            if val == 0 {
                run += 1
            } else {
                encodeCoeffRun(val: val, encoder: &encoder, run: run, isParentZero: isParentZero)
                run = 0
            }
        }
    }
}

@inline(__always)
func blockEncode16(encoder: inout EntropyEncoder, block: BlockView, parentBlock: BlockView?) {
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
        // Decoder invokes clearAll(), so encoder side must also zero clear to guarantee match during reconstruction
        for y in 0..<16 {
            let ptr = block.rowPointer(y: y)
            for x in 0..<16 {
                ptr[x] = 0
            }
        }
        return
    }
    encoder.encodeBypass(binVal: 1)

    encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
    encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

    var run = 0
    var isParentZero = false
    for y in 0...lscpY {
        let ptr = block.rowPointer(y: y)
        let endX = (y == lscpY) ? lscpX : (16 - 1)
        for x in 0...endX {
            if run == 0 {
                if let pb = parentBlock {
                    isParentZero = pb.rowPointer(y: y / 2)[x / 2] == 0
                }
            }
            let val = ptr[x]
            if val == 0 {
                run += 1
            } else {
                encodeCoeffRun(val: val, encoder: &encoder, run: run, isParentZero: isParentZero)
                run = 0
            }
        }
    }
    
    // Zero clear positions beyond LSCP (Decoder clearAll -> decodes up to lscp, so beyond lscp is 0)
    // Encoder side also zeros beyond lscp to guarantee match with decoder during reconstruction
    let lscpPtr = block.rowPointer(y: lscpY)
    for x in (lscpX + 1)..<16 {
        lscpPtr[x] = 0
    }
    for y in (lscpY + 1)..<16 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<16 {
            ptr[x] = 0
        }
    }
}

@inline(__always)
func blockEncode8(encoder: inout EntropyEncoder, block: BlockView, parentBlock: BlockView?) {
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
        // Decoder invokes clearAll(), so encoder side must also zero clear to guarantee match during reconstruction
        for y in 0..<8 {
            let ptr = block.rowPointer(y: y)
            for x in 0..<8 {
                ptr[x] = 0
            }
        }
        return
    }
    encoder.encodeBypass(binVal: 1)

    encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
    encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

    var run = 0
    var isParentZero = false
    for y in 0...lscpY {
        let ptr = block.rowPointer(y: y)
        let endX = (y == lscpY) ? lscpX : (8 - 1)
        for x in 0...endX {
            if run == 0 {
                if let pb = parentBlock {
                    isParentZero = pb.rowPointer(y: y / 2)[x / 2] == 0
                }
            }
            let val = ptr[x]
            if val == 0 {
                run += 1
            } else {
                encodeCoeffRun(val: val, encoder: &encoder, run: run, isParentZero: isParentZero)
                run = 0
            }
        }
    }
    
    // Zero clear positions beyond LSCP (Decoder clearAll -> decodes up to lscp, so beyond lscp is 0)
    let lscpPtr = block.rowPointer(y: lscpY)
    for x in (lscpX + 1)..<8 {
        lscpPtr[x] = 0
    }
    for y in (lscpY + 1)..<8 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<8 {
            ptr[x] = 0
        }
    }
}

@inline(__always)
func blockEncode4(encoder: inout EntropyEncoder, block: BlockView, parentBlock: BlockView?) {
    var lscpX = -1
    var lscpY = -1
    let zero2 = SIMD2<Int16>(repeating: 0)
    for y in stride(from: 4 - 1, through: 0, by: -1) {
        let ptr = block.rowPointer(y: y)
        let v1 = UnsafeRawPointer(ptr.advanced(by: 2)).loadUnaligned(as: SIMD2<Int16>.self)
        if any(v1 .!= zero2) {
            for x in stride(from: 3, through: 2, by: -1) {
                if ptr[x] != 0 {
                    lscpX = x
                    lscpY = y
                    break
                }
            }
        }
        if lscpX != -1 { break }
        
        let v0 = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD2<Int16>.self)
        if any(v0 .!= zero2) {
            for x in stride(from: 1, through: 0, by: -1) {
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
            for x in 0..<4 {
                ptr[x] = 0
            }
        }
        return
    }
    encoder.encodeBypass(binVal: 1)

    encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
    encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

    var run = 0
    var isParentZero = false
    for y in 0...lscpY {
        let ptr = block.rowPointer(y: y)
        let endX = (y == lscpY) ? lscpX : (4 - 1)
        for x in 0...endX {
            if run == 0 {
                if let pb = parentBlock {
                    isParentZero = pb.rowPointer(y: y / 2)[x / 2] == 0
                }
            }
            let val = ptr[x]
            if val == 0 {
                run += 1
            } else {
                encodeCoeffRun(val: val, encoder: &encoder, run: run, isParentZero: isParentZero)
                run = 0
            }
        }
    }
    
    // Zero clear beyond LSCP
    let lscpPtr = block.rowPointer(y: lscpY)
    for x in (lscpX + 1)..<4 {
        lscpPtr[x] = 0
    }
    for y in (lscpY + 1)..<4 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<4 {
            ptr[x] = 0
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
func blockEncodeDPCM4(encoder: inout EntropyEncoder, block: BlockView, lastVal: inout Int16) {
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
    if any(errSIMD3 .!= zero4) {
        lscpIdx = errSIMD3[3] != 0 ? 15 : (errSIMD3[2] != 0 ? 14 : (errSIMD3[1] != 0 ? 13 : 12))
    } else if any(errSIMD2 .!= zero4) {
        lscpIdx = errSIMD2[3] != 0 ? 11 : (errSIMD2[2] != 0 ? 10 : (errSIMD2[1] != 0 ? 9 : 8))
    } else if any(errSIMD1 .!= zero4) {
        lscpIdx = errSIMD1[3] != 0 ? 7 : (errSIMD1[2] != 0 ? 6 : (errSIMD1[1] != 0 ? 5 : 4))
    } else if any(errSIMD0 .!= zero4) {
        lscpIdx = errSIMD0[3] != 0 ? 3 : (errSIMD0[2] != 0 ? 2 : (errSIMD0[1] != 0 ? 1 : (errSIMD0[0] != 0 ? 0 : -1)))
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
    }

    lastVal = ptr3[3]
}

@inline(__always)
func blockEncodeDPCM8(encoder: inout EntropyEncoder, block: BlockView, lastVal: inout Int16) {
    @inline(__always)
    func errorMED(_ x: Int16, _ a: Int16, _ b: Int16, _ c: Int16) -> Int16 {
        let ia = Int(a), ib = Int(b), ic = Int(c)
        let predicted: Int16
        if ia <= ic && ib <= ic {
            predicted = Int16(truncatingIfNeeded: min(ia, ib))
        } else if ic <= ia && ic <= ib {
            predicted = Int16(truncatingIfNeeded: max(ia, ib))
        } else {
            predicted = Int16(truncatingIfNeeded: ia + ib - ic)
        }
        return x &- predicted
    }

    withUnsafeTemporaryAllocation(of: Int16.self, capacity: 64) { ptrErr in
        guard let baseErr = ptrErr.baseAddress else { return }
        
        var last: Int16 = lastVal
        for y in 0..<8 {
            let ptrY = block.rowPointer(y: y)
            let rowOffset = y * 8
            if y == 0 {
                for x in 0..<8 {
                    if x == 0 {
                        baseErr[rowOffset + 0] = ptrY[0] &- last
                    } else {
                        baseErr[rowOffset + x] = ptrY[x] &- ptrY[x - 1]
                    }
                }
            } else {
                let ptrPrevY = block.rowPointer(y: y - 1)
                for x in 0..<8 {
                    if x == 0 {
                        baseErr[rowOffset + 0] = ptrY[0] &- ptrPrevY[0]
                    } else {
                        baseErr[rowOffset + x] = errorMED(ptrY[x], ptrY[x - 1], ptrPrevY[x], ptrPrevY[x - 1])
                    }
                }
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
    }
}

@inline(__always)
func blockEncodeDPCM16(encoder: inout EntropyEncoder, block: BlockView, lastVal: inout Int16) {
    @inline(__always)
    func errorMED(_ x: Int16, _ a: Int16, _ b: Int16, _ c: Int16) -> Int16 {
        let ia = Int(a), ib = Int(b), ic = Int(c)
        let predicted: Int16
        if ia <= ic && ib <= ic {
            predicted = Int16(truncatingIfNeeded: min(ia, ib))
        } else if ic <= ia && ic <= ib {
            predicted = Int16(truncatingIfNeeded: max(ia, ib))
        } else {
            predicted = Int16(truncatingIfNeeded: ia + ib - ic)
        }
        return x &- predicted
    }

    let originalLastVal = lastVal
    var lscpIdx = -1
    var last: Int16 = originalLastVal

    for y in 0..<16 {
        let ptrY = block.rowPointer(y: y)
        if y == 0 {
            for x in 0..<16 {
                let diff: Int16
                if x == 0 {
                    diff = ptrY[0] &- last
                } else {
                    diff = ptrY[x] &- ptrY[x - 1]
                }
                if diff != 0 {
                    lscpIdx = y * 16 + x
                }
            }
        } else {
            let ptrPrevY = block.rowPointer(y: y - 1)
            for x in 0..<16 {
                let diff: Int16
                if x == 0 {
                    diff = ptrY[0] &- ptrPrevY[0]
                } else {
                    diff = errorMED(ptrY[x], ptrY[x - 1], ptrPrevY[x], ptrPrevY[x - 1])
                }
                if diff != 0 {
                    lscpIdx = y * 16 + x
                }
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

    for y in 0..<16 {
        let ptrY = block.rowPointer(y: y)
        if y == 0 {
            for x in 0..<16 {
                let diff: Int16
                if x == 0 {
                    diff = ptrY[0] &- last
                } else {
                    diff = ptrY[x] &- ptrY[x - 1]
                }
                if diff == 0 {
                    run += 1
                } else {
                    encodeCoeffRun(val: diff, encoder: &encoder, run: run)
                    run = 0
                }
                currentIdx += 1
                if currentIdx > lscpIdx { break }
            }
        } else {
            let ptrPrevY = block.rowPointer(y: y - 1)
            for x in 0..<16 {
                let diff: Int16
                if x == 0 {
                    diff = ptrY[0] &- ptrPrevY[0]
                } else {
                    diff = errorMED(ptrY[x], ptrY[x - 1], ptrPrevY[x], ptrPrevY[x - 1])
                }
                if diff == 0 {
                    run += 1
                } else {
                    encodeCoeffRun(val: diff, encoder: &encoder, run: run)
                    run = 0
                }
                currentIdx += 1
                if currentIdx > lscpIdx { break }
            }
        }
        if currentIdx > lscpIdx { break }
        last = ptrY[15]
    }
}

// MARK: - Transform Functions

@inline(__always)
func isEffectivelyZero32(data: UnsafeMutableBufferPointer<Int16>, threshold: Int) -> Bool {
    guard let base = data.baseAddress else { return false }
    let th = Int16(threshold)
    let thPos = SIMD16<Int16>(repeating: th)
    let thNeg = SIMD16<Int16>(repeating: -th)

    let lowerHalfBase = base + 16 * 32
    for i in stride(from: 0, to: 512, by: 16) {
        let vec = SIMD16<Int16>(UnsafeBufferPointer(start: lowerHalfBase + i, count: 16))
        let mask = (vec .> thPos) .| (vec .< thNeg)
        if any(mask) { return false }
    }
    for y in 0..<16 {
        let ptr = base + y * 32 + 16
        let vec = SIMD16<Int16>(UnsafeBufferPointer(start: ptr, count: 16))
        let mask = (vec .> thPos) .| (vec .< thNeg)
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
func isEffectivelyZero16(data: UnsafeMutableBufferPointer<Int16>, threshold: Int) -> Bool {
    guard let base = data.baseAddress else { return false }
    let th = Int16(threshold)
    let thPos = SIMD8<Int16>(repeating: th)
    let thNeg = SIMD8<Int16>(repeating: -th)

    let lowerHalfBase = base + 8 * 16
    for i in stride(from: 0, to: 128, by: 8) {
        let vec = SIMD8<Int16>(UnsafeBufferPointer(start: lowerHalfBase + i, count: 8))
        let mask = (vec .> thPos) .| (vec .< thNeg)
        if any(mask) { return false }
    }
    for y in 0..<8 {
        let ptr = base + y * 16 + 8
        let vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr, count: 8))
        let mask = (vec .> thPos) .| (vec .< thNeg)
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
func isEffectivelyZero8(data: UnsafeMutableBufferPointer<Int16>, threshold: Int) -> Bool {
    guard let base = data.baseAddress else { return false }
    let th = Int16(threshold)
    let thPos = SIMD4<Int16>(repeating: th)
    let thNeg = SIMD4<Int16>(repeating: -th)

    let lowerHalfBase = base + 4 * 8
    for i in stride(from: 0, to: 32, by: 4) {
        let vec = SIMD4<Int16>(UnsafeBufferPointer(start: lowerHalfBase + i, count: 4))
        let mask = (vec .> thPos) .| (vec .< thNeg)
        if any(mask) { return false }
    }
    for y in 0..<4 {
        let ptr = base + y * 8 + 4
        let vec = SIMD4<Int16>(UnsafeBufferPointer(start: ptr, count: 4))
        let mask = (vec .> thPos) .| (vec .< thNeg)
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
func isEffectivelyZero4(data: UnsafeMutableBufferPointer<Int16>, threshold: Int) -> Bool {
    guard let base = data.baseAddress else { return false }
    let th = Int16(threshold)
    let thPos = SIMD2<Int16>(repeating: th)
    let thNeg = SIMD2<Int16>(repeating: -th)

    let lowerHalfBase = base + 2 * 4
    for i in stride(from: 0, to: 8, by: 2) {
        let vec = SIMD2<Int16>(UnsafeBufferPointer(start: lowerHalfBase + i, count: 2))
        let mask = (vec .> thPos) .| (vec .< thNeg)
        if any(mask) { return false }
    }
    for y in 0..<2 {
        let ptr = base + y * 4 + 2
        let vec = SIMD2<Int16>(UnsafeBufferPointer(start: ptr, count: 2))
        let mask = (vec .> thPos) .| (vec .< thNeg)
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
func shouldSplit32(data: UnsafeMutableBufferPointer<Int16>, skipLL: Bool) -> Bool {
    guard let base = data.baseAddress else { return false }
    var q0 = false, q1 = false, q2 = false, q3 = false
    
    if skipLL != true {
        checkQuadrants16x16(base: base, stride: 32, q0: &q0, q1: &q1, q2: &q2, q3: &q3)
    }
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
func shouldSplit16(data: UnsafeMutableBufferPointer<Int16>, skipLL: Bool) -> Bool {
    guard let base = data.baseAddress else { return false }
    var q0 = false, q1 = false, q2 = false, q3 = false
    
    if skipLL != true {
        checkQuadrants8x8(base: base, stride: 16, q0: &q0, q1: &q1, q2: &q2, q3: &q3)
    }
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
func isEffectivelyZeroBase4(data: UnsafeMutableBufferPointer<Int16>, threshold: Int) -> Bool {
    guard let base = data.baseAddress else { return false }
    // Check LL
    for y in 0..<2 {
        let ptr = base + y * 4
        let vec = SIMD2<Int16>(UnsafeBufferPointer(start: ptr, count: 2))
        if vec[0] != 0 || vec[1] != 0 { return false }
    }
    
    // Check Subbands
    let th = Int16(threshold)
    let thPos = SIMD2<Int16>(repeating: th)
    let thNeg = SIMD2<Int16>(repeating: -th)

    let lowerHalfBase = base + 2 * 4
    for i in stride(from: 0, to: 8, by: 2) {
        let vec = SIMD2<Int16>(UnsafeBufferPointer(start: lowerHalfBase + i, count: 2))
        let mask = (vec .> thPos) .| (vec .< thNeg)
        if any(mask) { return false }
    }
    for y in 0..<2 {
        let ptr = base + y * 4 + 2
        let vec = SIMD2<Int16>(UnsafeBufferPointer(start: ptr, count: 2))
        let mask = (vec .> thPos) .| (vec .< thNeg)
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
func isEffectivelyZeroBase32(data: UnsafeMutableBufferPointer<Int16>, threshold: Int) -> Bool {
    guard let base = data.baseAddress else { return false }
    // Check LL
    for y in 0..<16 {
        let ptr = base + y * 32
        let vec = SIMD16<Int16>(UnsafeBufferPointer(start: ptr, count: 16))
        let mask = vec .!= 0
        if any(mask) { return false }
    }
    
    // Check Subbands
    let th = Int16(threshold)
    let thPos = SIMD16<Int16>(repeating: th)
    let thNeg = SIMD16<Int16>(repeating: -th)

    let lowerHalfBase = base + 16 * 32
    for i in stride(from: 0, to: 512, by: 16) {
        let vec = SIMD16<Int16>(UnsafeBufferPointer(start: lowerHalfBase + i, count: 16))
        let mask = (vec .> thPos) .| (vec .< thNeg)
        if any(mask) { return false }
    }
    for y in 0..<16 {
        let ptr = base + y * 32 + 16
        let vec = SIMD16<Int16>(UnsafeBufferPointer(start: ptr, count: 16))
        let mask = (vec .> thPos) .| (vec .< thNeg)
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

enum EncodeTask32 {
    case encode16
    case split8(Bool, Bool, Bool, Bool)
}

@inline(__always)
func encodePlaneSubbands32(blocks: inout [Block2D], zeroThreshold: Int, parentBlocks: [Block2D]?) -> [UInt8] {
    var bwFlags = BypassWriter()
    var tasks: [(Int, EncodeTask32)] = []
    tasks.reserveCapacity(blocks.count)
    
    var zeroCount = 0
    for i in blocks.indices {
        let isZero = blocks[i].data.withUnsafeMutableBufferPointer { ptr in
            return isEffectivelyZero32(data: ptr, threshold: zeroThreshold)
        }
        if isZero {
            bwFlags.writeBit(true)
            blocks[i].withView { view in
                let half = 32 / 2
                let base = view.base
                let hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 32)
                let lhView = BlockView(base: base.advanced(by: half * 32), width: half, height: half, stride: 32)
                let hhView = BlockView(base: base.advanced(by: half * 32 + half), width: half, height: half, stride: 32)
                hlView.clearAll()
                lhView.clearAll()
                hhView.clearAll()
            }
            zeroCount += 1
        } else {
            bwFlags.writeBit(false)
            
            let forceSplit = blocks[i].data.withUnsafeMutableBufferPointer { ptr in
                return shouldSplit32(data: ptr, skipLL: true)
            }
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
    let zeroRate = Double(zeroCount) / Double(max(1, blocks.count)) * 100.0
    let rateStr = String(format: "%.1f", zeroRate)
    debugLog("    [Subbands] blocks=\(blocks.count) zeroBlocks=\(zeroCount) zeroRate=\(rateStr)%")
    
    var encoder = EntropyEncoder()
    
    for (i, task) in tasks {
        func encodeAction(_ parentHL: BlockView?, _ parentLH: BlockView?, _ parentHH: BlockView?) {
            blocks[i].withView { view in
                let subs = getSubbands32(view: view)
                switch task {
                case .encode16:
                    blockEncode16(encoder: &encoder, block: subs.hl, parentBlock: parentHL)
                    blockEncode16(encoder: &encoder, block: subs.lh, parentBlock: parentLH)
                    blockEncode16(encoder: &encoder, block: subs.hh, parentBlock: parentHH)
                case .split8(let tl, let tr, let bl, let br):
                    if tl {
                        let pbHL = parentHL.map { BlockView(base: $0.base, width: 4, height: 4, stride: $0.stride) }
                        let pbLH = parentLH.map { BlockView(base: $0.base, width: 4, height: 4, stride: $0.stride) }
                        let pbHH = parentHH.map { BlockView(base: $0.base, width: 4, height: 4, stride: $0.stride) }
                        let hl = BlockView(base: subs.hl.base, width: 8, height: 8, stride: 32)
                        let lh = BlockView(base: subs.lh.base, width: 8, height: 8, stride: 32)
                        let hh = BlockView(base: subs.hh.base, width: 8, height: 8, stride: 32)
                        blockEncode8(encoder: &encoder, block: hl, parentBlock: pbHL)
                        blockEncode8(encoder: &encoder, block: lh, parentBlock: pbLH)
                        blockEncode8(encoder: &encoder, block: hh, parentBlock: pbHH)
                    }
                    if tr {
                        let pbHL = parentHL.map { BlockView(base: $0.base.advanced(by: 4), width: 4, height: 4, stride: $0.stride) }
                        let pbLH = parentLH.map { BlockView(base: $0.base.advanced(by: 4), width: 4, height: 4, stride: $0.stride) }
                        let pbHH = parentHH.map { BlockView(base: $0.base.advanced(by: 4), width: 4, height: 4, stride: $0.stride) }
                        let hl = BlockView(base: subs.hl.base.advanced(by: 8), width: 8, height: 8, stride: 32)
                        let lh = BlockView(base: subs.lh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
                        let hh = BlockView(base: subs.hh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
                        blockEncode8(encoder: &encoder, block: hl, parentBlock: pbHL)
                        blockEncode8(encoder: &encoder, block: lh, parentBlock: pbLH)
                        blockEncode8(encoder: &encoder, block: hh, parentBlock: pbHH)
                    }
                    if bl {
                        let pbHL = parentHL.map { BlockView(base: $0.base.advanced(by: 4 * $0.stride), width: 4, height: 4, stride: $0.stride) }
                        let pbLH = parentLH.map { BlockView(base: $0.base.advanced(by: 4 * $0.stride), width: 4, height: 4, stride: $0.stride) }
                        let pbHH = parentHH.map { BlockView(base: $0.base.advanced(by: 4 * $0.stride), width: 4, height: 4, stride: $0.stride) }
                        let hl = BlockView(base: subs.hl.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                        let lh = BlockView(base: subs.lh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                        let hh = BlockView(base: subs.hh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                        blockEncode8(encoder: &encoder, block: hl, parentBlock: pbHL)
                        blockEncode8(encoder: &encoder, block: lh, parentBlock: pbLH)
                        blockEncode8(encoder: &encoder, block: hh, parentBlock: pbHH)
                    }
                    if br {
                        let pbHL = parentHL.map { BlockView(base: $0.base.advanced(by: 4 * $0.stride + 4), width: 4, height: 4, stride: $0.stride) }
                        let pbLH = parentLH.map { BlockView(base: $0.base.advanced(by: 4 * $0.stride + 4), width: 4, height: 4, stride: $0.stride) }
                        let pbHH = parentHH.map { BlockView(base: $0.base.advanced(by: 4 * $0.stride + 4), width: 4, height: 4, stride: $0.stride) }
                        let hl = BlockView(base: subs.hl.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                        let lh = BlockView(base: subs.lh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                        let hh = BlockView(base: subs.hh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                        blockEncode8(encoder: &encoder, block: hl, parentBlock: pbHL)
                        blockEncode8(encoder: &encoder, block: lh, parentBlock: pbLH)
                        blockEncode8(encoder: &encoder, block: hh, parentBlock: pbHH)
                    }
                }
            }
        }
        
        if let pBlocks = parentBlocks, i < pBlocks.count {
            var pBlock = pBlocks[i]
            pBlock.withView { pView in
                let pSubs = getSubbands16(view: pView)
                encodeAction(pSubs.hl, pSubs.lh, pSubs.hh)
            }
        } else {
            encodeAction(nil, nil, nil)
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
func encodePlaneSubbands16(blocks: inout [Block2D], zeroThreshold: Int, parentBlocks: [Block2D]?) -> [UInt8] {
    var bwFlags = BypassWriter()
    var tasks: [(Int, EncodeTask16)] = []
    tasks.reserveCapacity(blocks.count)
    
    var zeroCount = 0
    for i in blocks.indices {
        let isZero = blocks[i].data.withUnsafeMutableBufferPointer { ptr in
            return isEffectivelyZero16(data: ptr, threshold: zeroThreshold)
        }
        if isZero {
            bwFlags.writeBit(true)
            blocks[i].withView { view in
                let half = 16 / 2
                let base = view.base
                let hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 16)
                let lhView = BlockView(base: base.advanced(by: half * 16), width: half, height: half, stride: 16)
                let hhView = BlockView(base: base.advanced(by: half * 16 + half), width: half, height: half, stride: 16)
                hlView.clearAll()
                lhView.clearAll()
                hhView.clearAll()
            }
            zeroCount += 1
        } else {
            bwFlags.writeBit(false)
            let forceSplit = blocks[i].data.withUnsafeMutableBufferPointer { ptr in
                return shouldSplit16(data: ptr, skipLL: true)
            }
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
    let zeroRate = Double(zeroCount) / Double(max(1, blocks.count)) * 100.0
    let rateStr = String(format: "%.1f", zeroRate)
    debugLog("    [Subbands] blocks=\(blocks.count) zeroBlocks=\(zeroCount) zeroRate=\(rateStr)%")
    
    var encoder = EntropyEncoder()
    
    for (i, task) in tasks {
        func encodeAction(_ parentHL: BlockView?, _ parentLH: BlockView?, _ parentHH: BlockView?) {
            blocks[i].withView { view in
                let subs = getSubbands16(view: view)
                switch task {
                case .encode8:
                    blockEncode8(encoder: &encoder, block: subs.hl, parentBlock: parentHL)
                    blockEncode8(encoder: &encoder, block: subs.lh, parentBlock: parentLH)
                    blockEncode8(encoder: &encoder, block: subs.hh, parentBlock: parentHH)
                case .split4(let tl, let tr, let bl, let br):
                    if tl {
                        let pbHL = parentHL.map { BlockView(base: $0.base, width: 2, height: 2, stride: $0.stride) }
                        let pbLH = parentLH.map { BlockView(base: $0.base, width: 2, height: 2, stride: $0.stride) }
                        let pbHH = parentHH.map { BlockView(base: $0.base, width: 2, height: 2, stride: $0.stride) }
                        let hl = BlockView(base: subs.hl.base, width: 4, height: 4, stride: 16)
                        let lh = BlockView(base: subs.lh.base, width: 4, height: 4, stride: 16)
                        let hh = BlockView(base: subs.hh.base, width: 4, height: 4, stride: 16)
                        blockEncode4(encoder: &encoder, block: hl, parentBlock: pbHL)
                        blockEncode4(encoder: &encoder, block: lh, parentBlock: pbLH)
                        blockEncode4(encoder: &encoder, block: hh, parentBlock: pbHH)
                    }
                    if tr {
                        let pbHL = parentHL.map { BlockView(base: $0.base.advanced(by: 2), width: 2, height: 2, stride: $0.stride) }
                        let pbLH = parentLH.map { BlockView(base: $0.base.advanced(by: 2), width: 2, height: 2, stride: $0.stride) }
                        let pbHH = parentHH.map { BlockView(base: $0.base.advanced(by: 2), width: 2, height: 2, stride: $0.stride) }
                        let hl = BlockView(base: subs.hl.base.advanced(by: 4), width: 4, height: 4, stride: 16)
                        let lh = BlockView(base: subs.lh.base.advanced(by: 4), width: 4, height: 4, stride: 16)
                        let hh = BlockView(base: subs.hh.base.advanced(by: 4), width: 4, height: 4, stride: 16)
                        blockEncode4(encoder: &encoder, block: hl, parentBlock: pbHL)
                        blockEncode4(encoder: &encoder, block: lh, parentBlock: pbLH)
                        blockEncode4(encoder: &encoder, block: hh, parentBlock: pbHH)
                    }
                    if bl {
                        let pbHL = parentHL.map { BlockView(base: $0.base.advanced(by: 2 * $0.stride), width: 2, height: 2, stride: $0.stride) }
                        let pbLH = parentLH.map { BlockView(base: $0.base.advanced(by: 2 * $0.stride), width: 2, height: 2, stride: $0.stride) }
                        let pbHH = parentHH.map { BlockView(base: $0.base.advanced(by: 2 * $0.stride), width: 2, height: 2, stride: $0.stride) }
                        let hl = BlockView(base: subs.hl.base.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
                        let lh = BlockView(base: subs.lh.base.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
                        let hh = BlockView(base: subs.hh.base.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
                        blockEncode4(encoder: &encoder, block: hl, parentBlock: pbHL)
                        blockEncode4(encoder: &encoder, block: lh, parentBlock: pbLH)
                        blockEncode4(encoder: &encoder, block: hh, parentBlock: pbHH)
                    }
                    if br {
                        let pbHL = parentHL.map { BlockView(base: $0.base.advanced(by: 2 * $0.stride + 2), width: 2, height: 2, stride: $0.stride) }
                        let pbLH = parentLH.map { BlockView(base: $0.base.advanced(by: 2 * $0.stride + 2), width: 2, height: 2, stride: $0.stride) }
                        let pbHH = parentHH.map { BlockView(base: $0.base.advanced(by: 2 * $0.stride + 2), width: 2, height: 2, stride: $0.stride) }
                        let hl = BlockView(base: subs.hl.base.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
                        let lh = BlockView(base: subs.lh.base.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
                        let hh = BlockView(base: subs.hh.base.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
                        blockEncode4(encoder: &encoder, block: hl, parentBlock: pbHL)
                        blockEncode4(encoder: &encoder, block: lh, parentBlock: pbLH)
                        blockEncode4(encoder: &encoder, block: hh, parentBlock: pbHH)
                    }
                }
            }
        }
        
        if let pBlocks = parentBlocks, i < pBlocks.count {
            var pBlock = pBlocks[i]
            pBlock.withView { pView in
                let pSubs = getSubbands8(view: pView)
                encodeAction(pSubs.hl, pSubs.lh, pSubs.hh)
            }
        } else {
            encodeAction(nil, nil, nil)
        }
    }
    
    encoder.flush()
    var out = bwFlags.bytes
    out.append(contentsOf: encoder.getData())
    return out
}

@inline(__always)
func encodePlaneSubbands8(blocks: inout [Block2D], zeroThreshold: Int) -> [UInt8] {
    var bwFlags = BypassWriter()
    var nonZeroIndices: [Int] = []
    
    for i in blocks.indices {
        let isZero = blocks[i].data.withUnsafeMutableBufferPointer { ptr in
            return isEffectivelyZero8(data: ptr, threshold: zeroThreshold)
        }
        if isZero {
            bwFlags.writeBit(true)
            bwFlags.writeBit(false)
        } else {
            bwFlags.writeBit(false)
            bwFlags.writeBit(false)
            nonZeroIndices.append(i)
        }
    }
    bwFlags.flush()
    let zeroCount = blocks.count - nonZeroIndices.count
    let zeroRate = Double(zeroCount) / Double(max(1, blocks.count)) * 100.0
    let rateStr = String(format: "%.1f", zeroRate)
    debugLog("    [Subbands] blocks=\(blocks.count) zeroBlocks=\(zeroCount) zeroRate=\(rateStr)%")
    
    var encoder = EntropyEncoder()
    
    for i in nonZeroIndices {
        blocks[i].withView { view in
            let subs = getSubbands8(view: view)
            blockEncode4(encoder: &encoder, block: subs.hl, parentBlock: nil)
            blockEncode4(encoder: &encoder, block: subs.lh, parentBlock: nil)
            blockEncode4(encoder: &encoder, block: subs.hh, parentBlock: nil)
        }
    }
    
    encoder.flush()
    var out = bwFlags.bytes
    out.append(contentsOf: encoder.getData())
    return out
}

@inline(__always)
func encodePlaneBaseSubbands8(blocks: inout [Block2D], zeroThreshold: Int) -> [UInt8] {
    var bwFlags = BypassWriter()
    var nonZeroIndices: [Int] = []
    
    for i in blocks.indices {
        let isZero = blocks[i].data.withUnsafeMutableBufferPointer { ptr in
            return isEffectivelyZeroBase4(data: ptr, threshold: zeroThreshold)
        }
        if isZero {
            bwFlags.writeBit(true)
            bwFlags.writeBit(false)
            blocks[i].withView { $0.clearAll() }
        } else {
            bwFlags.writeBit(false)
            bwFlags.writeBit(false)
            nonZeroIndices.append(i)
        }
    }
    bwFlags.flush()
    let zeroCount = blocks.count - nonZeroIndices.count
    let zeroRate = Double(zeroCount) / Double(max(1, blocks.count)) * 100.0
    let rateStr = String(format: "%.1f", zeroRate)
    debugLog("    [BaseSubbands] blocks=\(blocks.count) zeroBlocks=\(zeroCount) zeroRate=\(rateStr)%")
    
    var encoder = EntropyEncoder()
    var lastVal: Int16 = 0
    
    var nzCur = 0
    let nzCount = nonZeroIndices.count
    for i in blocks.indices {
        if nzCur < nzCount && nonZeroIndices[nzCur] == i {
            nzCur += 1

            blocks[i].withView { view in
                let subs = getSubbands8(view: view)
                blockEncodeDPCM4(encoder: &encoder, block: subs.ll, lastVal: &lastVal)
                blockEncode4(encoder: &encoder, block: subs.hl, parentBlock: nil)
                blockEncode4(encoder: &encoder, block: subs.lh, parentBlock: nil)
                blockEncode4(encoder: &encoder, block: subs.hh, parentBlock: nil)
            }
        } else {
            lastVal = 0
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
func encodePlaneBaseSubbands32(blocks: inout [Block2D], zeroThreshold: Int) -> [UInt8] {
    var bwFlags = BypassWriter()
    var tasks: [(Int, EncodeTaskBase32)] = []
    tasks.reserveCapacity(blocks.count)
    
    var zeroCount = 0
    for i in blocks.indices {
        let isZero = blocks[i].data.withUnsafeMutableBufferPointer { ptr in
            return isEffectivelyZeroBase32(data: ptr, threshold: zeroThreshold)
        }
        if isZero {
            bwFlags.writeBit(true)
            blocks[i].withView { $0.clearAll() }
            tasks.append((i, .skip))
            zeroCount += 1
        } else {
            bwFlags.writeBit(false)
            let forceSplit = blocks[i].data.withUnsafeMutableBufferPointer { ptr in
                return shouldSplit32(data: ptr, skipLL: false)
            }
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
    debugLog("    [BaseSubbands32] blocks=\(blocks.count) zeroBlocks=\(zeroCount) zeroRate=\(String(format: "%.1f", Double(zeroCount) / Double(max(1, blocks.count)) * 100))%")
    
    var encoder = EntropyEncoder(useStaticTable: false)
    var lastVal: Int16 = 0
    
    for (i, task) in tasks {
        switch task {
        case .skip:
            lastVal = 0
            
        case .encode16:
            blocks[i].withView { view in
                let subs = getSubbands32(view: view)
                blockEncodeDPCM16(encoder: &encoder, block: subs.ll, lastVal: &lastVal)
                blockEncode16(encoder: &encoder, block: subs.hl, parentBlock: nil)
                blockEncode16(encoder: &encoder, block: subs.lh, parentBlock: nil)
                blockEncode16(encoder: &encoder, block: subs.hh, parentBlock: nil)
            }

        case .split8(let tl, let tr, let bl, let br):
            blocks[i].withView { view in
                let subs = getSubbands32(view: view)
                if tl {
                    let ll = BlockView(base: subs.ll.base, width: 8, height: 8, stride: 32)
                    let hl = BlockView(base: subs.hl.base, width: 8, height: 8, stride: 32)
                    let lh = BlockView(base: subs.lh.base, width: 8, height: 8, stride: 32)
                    let hh = BlockView(base: subs.hh.base, width: 8, height: 8, stride: 32)
                    blockEncodeDPCM8(encoder: &encoder, block: ll, lastVal: &lastVal)
                    blockEncode8(encoder: &encoder, block: hl, parentBlock: nil)
                    blockEncode8(encoder: &encoder, block: lh, parentBlock: nil)
                    blockEncode8(encoder: &encoder, block: hh, parentBlock: nil)
                }
                if tr {
                    let ll = BlockView(base: subs.ll.base.advanced(by: 8), width: 8, height: 8, stride: 32)
                    let hl = BlockView(base: subs.hl.base.advanced(by: 8), width: 8, height: 8, stride: 32)
                    let lh = BlockView(base: subs.lh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
                    let hh = BlockView(base: subs.hh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
                    blockEncodeDPCM8(encoder: &encoder, block: ll, lastVal: &lastVal)
                    blockEncode8(encoder: &encoder, block: hl, parentBlock: nil)
                    blockEncode8(encoder: &encoder, block: lh, parentBlock: nil)
                    blockEncode8(encoder: &encoder, block: hh, parentBlock: nil)
                }
                if bl {
                    let ll = BlockView(base: subs.ll.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    let hl = BlockView(base: subs.hl.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    let lh = BlockView(base: subs.lh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    let hh = BlockView(base: subs.hh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    blockEncodeDPCM8(encoder: &encoder, block: ll, lastVal: &lastVal)
                    blockEncode8(encoder: &encoder, block: hl, parentBlock: nil)
                    blockEncode8(encoder: &encoder, block: lh, parentBlock: nil)
                    blockEncode8(encoder: &encoder, block: hh, parentBlock: nil)
                }
                if br {
                    let ll = BlockView(base: subs.ll.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    let hl = BlockView(base: subs.hl.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    let lh = BlockView(base: subs.lh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    let hh = BlockView(base: subs.hh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    blockEncodeDPCM8(encoder: &encoder, block: ll, lastVal: &lastVal)
                    blockEncode8(encoder: &encoder, block: hl, parentBlock: nil)
                    blockEncode8(encoder: &encoder, block: lh, parentBlock: nil)
                    blockEncode8(encoder: &encoder, block: hh, parentBlock: nil)
                }
            }
        }
    }
    
    encoder.flush()
    var out = bwFlags.bytes
    out.append(contentsOf: encoder.getData())
    return out
}

@inline(__always)
func calculateSADAndMaxBlockSAD(res: PlaneData420, mbSize: Int) -> (meanSAD: Int, maxBlockSAD: Int) {
    let resMbCols = res.width / mbSize
    let resMbRows = res.height / mbSize
    let stride = res.width
    
    var totalSAD = 0
    var maxBlockSAD = 0
    
    res.y.withUnsafeBufferPointer { yBuf in
        guard let yPtr = yBuf.baseAddress else { return }
        
        for mbY in 0..<resMbRows {
            let startY = mbY * mbSize
            for mbX in 0..<resMbCols {
                let startX = mbX * mbSize
                var blockSAD = 0
                
                if mbSize == 64 {
                    var sumVec0 = SIMD16<Int16>()
                    var sumVec1 = SIMD16<Int16>()
                    var sumVec2 = SIMD16<Int16>()
                    var sumVec3 = SIMD16<Int16>()
                    
                    for by in 0..<64 {
                        let rowPtr = yPtr.advanced(by: (startY + by) * stride + startX)
                        let c0 = UnsafeRawPointer(rowPtr).loadUnaligned(as: SIMD16<Int16>.self)
                        let c1 = UnsafeRawPointer(rowPtr.advanced(by: 16)).loadUnaligned(as: SIMD16<Int16>.self)
                        let c2 = UnsafeRawPointer(rowPtr.advanced(by: 32)).loadUnaligned(as: SIMD16<Int16>.self)
                        let c3 = UnsafeRawPointer(rowPtr.advanced(by: 48)).loadUnaligned(as: SIMD16<Int16>.self)
                        
                        let mask0 = c0 &>> 15
                        let abs0 = (c0 ^ mask0) &- mask0
                        let mask1 = c1 &>> 15
                        let abs1 = (c1 ^ mask1) &- mask1
                        let mask2 = c2 &>> 15
                        let abs2 = (c2 ^ mask2) &- mask2
                        let mask3 = c3 &>> 15
                        let abs3 = (c3 ^ mask3) &- mask3
                        
                        sumVec0 &+= abs0
                        sumVec1 &+= abs1
                        sumVec2 &+= abs2
                        sumVec3 &+= abs3
                    }
                    
                    let total0 = SIMD16<Int32>(clamping: sumVec0).wrappedSum()
                    let total1 = SIMD16<Int32>(clamping: sumVec1).wrappedSum()
                    let total2 = SIMD16<Int32>(clamping: sumVec2).wrappedSum()
                    let total3 = SIMD16<Int32>(clamping: sumVec3).wrappedSum()
                    blockSAD = Int(total0 &+ total1 &+ total2 &+ total3)
                } else {
                    for by in 0..<mbSize {
                        let rowOffset = (startY + by) * stride + startX
                        for bx in 0..<mbSize {
                            blockSAD += abs(Int(yPtr[rowOffset + bx]))
                        }
                    }
                }
                
                totalSAD += blockSAD
                if blockSAD > maxBlockSAD {
                    maxBlockSAD = blockSAD
                }
            }
        }
    }
    
    let meanSAD = totalSAD / (res.width * res.height)
    return (meanSAD, maxBlockSAD)
}

// MARK: - Cascaded Encoding

@inline(__always)
func encodeCascadedPlaneSubbands32(blocks: inout [Block2D], zeroThreshold: Int) -> [UInt8] {
    var bwFlags = BypassWriter()
    var tasks: [(Int, Bool)] = []
    tasks.reserveCapacity(blocks.count)
    
    var zeroCount = 0
    for i in blocks.indices {
        let isZero = blocks[i].data.withUnsafeMutableBufferPointer { ptr in
            return isEffectivelyZeroBase32(data: ptr, threshold: zeroThreshold)
        }
        if isZero {
            bwFlags.writeBit(true)
            blocks[i].withView { $0.clearAll() }
            tasks.append((i, true))
            zeroCount += 1
        } else {
            bwFlags.writeBit(false)
            tasks.append((i, false))
        }
    }
    bwFlags.flush()
    debugLog("    [CascadedSubbands32] blocks=\(blocks.count) zeroBlocks=\(zeroCount) zeroRate=\(String(format: "%.1f", Double(zeroCount) / Double(max(1, blocks.count)) * 100))%")
    
    var encoder = EntropyEncoder(useStaticTable: false)
    var lastVal: Int16 = 0
    
    for (i, skip) in tasks {
        if skip {
            lastVal = 0
            continue
        }
        
        blocks[i].withView { view in
            let hl1 = BlockView(base: view.base.advanced(by: 16), width: 16, height: 16, stride: view.stride)
            let lh1 = BlockView(base: view.base.advanced(by: 16 * view.stride), width: 16, height: 16, stride: view.stride)
            let hh1 = BlockView(base: view.base.advanced(by: 16 * view.stride + 16), width: 16, height: 16, stride: view.stride)
            
            let hl2 = BlockView(base: view.base.advanced(by: 8), width: 8, height: 8, stride: view.stride)
            let lh2 = BlockView(base: view.base.advanced(by: 8 * view.stride), width: 8, height: 8, stride: view.stride)
            let hh2 = BlockView(base: view.base.advanced(by: 8 * view.stride + 8), width: 8, height: 8, stride: view.stride)
            
            let ll3 = BlockView(base: view.base, width: 4, height: 4, stride: view.stride)
            let hl3 = BlockView(base: view.base.advanced(by: 4), width: 4, height: 4, stride: view.stride)
            let lh3 = BlockView(base: view.base.advanced(by: 4 * view.stride), width: 4, height: 4, stride: view.stride)
            let hh3 = BlockView(base: view.base.advanced(by: 4 * view.stride + 4), width: 4, height: 4, stride: view.stride)
            
            blockEncodeDPCM4(encoder: &encoder, block: ll3, lastVal: &lastVal)
            blockEncode4(encoder: &encoder, block: hl3, parentBlock: nil)
            blockEncode4(encoder: &encoder, block: lh3, parentBlock: nil)
            blockEncode4(encoder: &encoder, block: hh3, parentBlock: nil)
            
            blockEncode8(encoder: &encoder, block: hl2, parentBlock: nil)
            blockEncode8(encoder: &encoder, block: lh2, parentBlock: nil)
            blockEncode8(encoder: &encoder, block: hh2, parentBlock: nil)
            
            blockEncode16(encoder: &encoder, block: hl1, parentBlock: nil)
            blockEncode16(encoder: &encoder, block: lh1, parentBlock: nil)
            blockEncode16(encoder: &encoder, block: hh1, parentBlock: nil)
        }
    }
    
    encoder.flush()
    var out = bwFlags.bytes
    out.append(contentsOf: encoder.getData())
    return out
}

#if (arch(arm64) || arch(x86_64) || arch(wasm32))
public func encode(images: [YCbCrImage], maxbitrate: Int, framerate: Int = 30, zeroThreshold: Int = 3, keyint: Int = 60, sceneChangeThreshold: Int = 32) async throws -> [UInt8] {    
    if images.isEmpty { return [] }
    guard let first = images.first else { return [] }
    
    let encoder = Encoder(
        width: first.width,
        height: first.height,
        maxbitrate: maxbitrate,
        framerate: framerate,
        zeroThreshold: zeroThreshold,
        keyint: keyint,
        sceneChangeThreshold: sceneChangeThreshold,
        isOne: false
    )
    
    var out: [UInt8] = []
    for img in images {
        let chunk = try await encoder.encode(image: img)
        out.append(contentsOf: chunk)
    }
    return out
}

@inline(__always)
public func encodeOne(images: [YCbCrImage], maxbitrate: Int, framerate: Int = 30, zeroThreshold: Int = 3, keyint: Int = 60, sceneChangeThreshold: Int = 32) async throws -> [UInt8] {
    if images.isEmpty { return [] }
    guard let first = images.first else { return [] }
    
    let encoder = Encoder(
        width: first.width,
        height: first.height,
        maxbitrate: maxbitrate,
        framerate: framerate,
        zeroThreshold: zeroThreshold,
        keyint: keyint,
        sceneChangeThreshold: sceneChangeThreshold,
        isOne: true
    )
    
    var out: [UInt8] = []
    for img in images {
        let chunk = try await encoder.encode(image: img)
        out.append(contentsOf: chunk)
    }
    return out
}
            
#else
public func encode(images: [YCbCrImage], maxbitrate: Int, framerate: Int = 30, zeroThreshold: Int = 3, gopSize: Int = 15, sceneChangeThreshold: Int = 32) async throws -> [UInt8] {
    throw EncodeError.unsupportedArchitecture
}
public func encodeOne(images: [YCbCrImage], maxbitrate: Int, framerate: Int = 30, zeroThreshold: Int = 3, gopSize: Int = 15, sceneChangeThreshold: Int = 32) async throws -> [UInt8] {
    throw EncodeError.unsupportedArchitecture
}
#endif
