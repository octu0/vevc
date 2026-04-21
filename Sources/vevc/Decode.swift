// MARK: - Decode Error

public enum DecodeError: Error, CustomStringConvertible {
    case eof
    case insufficientData
    case insufficientDataContext(String)
    case invalidBlockData
    case invalidBlockDataContext(String)
    case invalidHeader
    case invalidLayerNumber
    case noDataProvided
    case unsupportedArchitecture
    case outOfBits
    
    public var description: String {
        switch self {
        case .eof: return "DecodeError.eof"
        case .insufficientData: return "DecodeError.insufficientData"
        case .insufficientDataContext(let ctx): return "DecodeError.insufficientData: \(ctx)"
        case .invalidBlockData: return "DecodeError.invalidBlockData"
        case .invalidBlockDataContext(let ctx): return "DecodeError.invalidBlockData: \(ctx)"
        case .invalidHeader: return "DecodeError.invalidHeader"
        case .invalidLayerNumber: return "DecodeError.invalidLayerNumber"
        case .noDataProvided: return "DecodeError.noDataProvided"
        case .unsupportedArchitecture: return "DecodeError.unsupportedArchitecture"
        case .outOfBits: return "DecodeError.outOfBits"
        }
    }
}



// Adaptive predictor: selects prediction based on edge direction.
// vertical edge -> min(a,b), horizontal edge -> max(a,b), flat -> a+b-c

@inline(__always)
func predictMED(_ a: Int16, _ b: Int16, _ c: Int16) -> Int16 {
    let ia = Int(a), ib = Int(b), ic = Int(c)
    if ia <= ic && ib <= ic {
        return Int16(truncatingIfNeeded: min(ia, ib))
    }
    if ic <= ia && ic <= ib {
        return Int16(truncatingIfNeeded: max(ia, ib))
    }
    return Int16(truncatingIfNeeded: ia + ib - ic)
}

@inline(__always)
func decodeExpGolomb(decoder: inout EntropyDecoder) throws -> UInt32 {
    var bits = 0
    while try decoder.decodeBypass() == 0 {
        bits += 1
    }
    guard 0 < bits else { return 0 }
    if 31 < bits { return 0 } // guard
    var val: UInt32 = 0
    for i in stride(from: bits - 1, through: 0, by: -1) {
        val |= UInt32(try decoder.decodeBypass()) << i
    }
    return val
}

@inline(__always)
func decodeCoeffRun(decoder: inout EntropyDecoder, isParentZero: Bool) throws -> (Int, Int16) {
    let pair = decoder.readPair(isParentZero: isParentZero)
    return (pair.run, pair.val)
}

@inline(__always)
func blockDecode32V(decoder: inout EntropyDecoder, block: BlockView, parentBlock: BlockView?) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        // blocks from pool are guaranteed zero (cleared on put), no explicit zeroing needed
        return
    }

    let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
    let lscpY = Int(try decodeExpGolomb(decoder: &decoder))

    var currentIdx = 0
    let lscpIdx = lscpX * 32 + lscpY
    while currentIdx <= lscpIdx {
        let startX = currentIdx / 32
        let startY = currentIdx % 32
        let isParentZero: Bool
        if let pb = parentBlock {
            isParentZero = pb.rowPointer(y: startY >> 1)[startX >> 1] == 0
        } else {
            isParentZero = false
        }
        let (run, val) = try decodeCoeffRun(decoder: &decoder, isParentZero: isParentZero)

        currentIdx += run
        if currentIdx <= lscpIdx {
            let x = currentIdx / 32
            let y = currentIdx % 32
            let ptr = block.rowPointer(y: y)
            ptr[x] = val
        }
        currentIdx += 1
    }
}

@inline(__always)
func blockDecode32H(decoder: inout EntropyDecoder, block: BlockView, parentBlock: BlockView?) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        // blocks from pool are guaranteed zero (cleared on put), no explicit zeroing needed
        return
    }

    let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
    let lscpY = Int(try decodeExpGolomb(decoder: &decoder))

    var currentIdx = 0
    let lscpIdx = lscpY * 32 + lscpX
    while currentIdx <= lscpIdx {
        let startY = currentIdx / 32
        let startX = currentIdx % 32
        let isParentZero: Bool
        if let pb = parentBlock {
            isParentZero = pb.rowPointer(y: startY >> 1)[startX >> 1] == 0
        } else {
            isParentZero = false
        }
        let (run, val) = try decodeCoeffRun(decoder: &decoder, isParentZero: isParentZero)

        currentIdx += run
        if currentIdx <= lscpIdx {
            let y = currentIdx / 32
            let x = currentIdx % 32
            let ptr = block.rowPointer(y: y)
            ptr[x] = val
        }
        currentIdx += 1
    }
}

@inline(__always)
func blockDecode16V(decoder: inout EntropyDecoder, block: BlockView) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        return
    }

    let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
    let lscpY = Int(try decodeExpGolomb(decoder: &decoder))
    guard lscpX < 16 && lscpY < 16 else { throw DecodeError.invalidBlockData }

    var currentIdx = 0
    let lscpIdx = lscpX * 16 + lscpY
    while currentIdx <= lscpIdx {
        let isParentZero = false
        let (run, val) = try decodeCoeffRun(decoder: &decoder, isParentZero: isParentZero)

        currentIdx += run
        if currentIdx <= lscpIdx {
            let x = currentIdx / 16
            let y = currentIdx % 16
            let ptr = block.rowPointer(y: y)
            ptr[x] = val
        }
        currentIdx += 1
    }
}

@inline(__always)
func blockDecode16VWithParentBlock(decoder: inout EntropyDecoder, block: BlockView, parentBlock: BlockView) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        return
    }

    let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
    let lscpY = Int(try decodeExpGolomb(decoder: &decoder))
    guard lscpX < 16 && lscpY < 16 else { throw DecodeError.invalidBlockData }

    var currentIdx = 0
    let lscpIdx = lscpX * 16 + lscpY
    while currentIdx <= lscpIdx {
        let startX = currentIdx / 16
        let startY = currentIdx % 16
        let isParentZero = parentBlock.rowPointer(y: startY >> 1)[startX >> 1] == 0
        let (run, val) = try decodeCoeffRun(decoder: &decoder, isParentZero: isParentZero)

        currentIdx += run
        if currentIdx <= lscpIdx {
            let x = currentIdx / 16
            let y = currentIdx % 16
            let ptr = block.rowPointer(y: y)
            ptr[x] = val
        }
        currentIdx += 1
    }
}

@inline(__always)
func blockDecode16H(decoder: inout EntropyDecoder, block: BlockView) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        return
    }

    let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
    let lscpY = Int(try decodeExpGolomb(decoder: &decoder))
    guard lscpX < 16 && lscpY < 16 else { throw DecodeError.invalidBlockData }

    var currentIdx = 0
    let lscpIdx = lscpY * 16 + lscpX
    while currentIdx <= lscpIdx {
        let isParentZero = false
        let (run, val) = try decodeCoeffRun(decoder: &decoder, isParentZero: isParentZero)

        currentIdx += run
        if currentIdx <= lscpIdx {
            let y = currentIdx / 16
            let x = currentIdx % 16
            let ptr = block.rowPointer(y: y)
            ptr[x] = val
        }
        currentIdx += 1
    }
}

@inline(__always)
func blockDecode16HWithParentBlock(decoder: inout EntropyDecoder, block: BlockView, parentBlock: BlockView) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        return
    }

    let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
    let lscpY = Int(try decodeExpGolomb(decoder: &decoder))
    guard lscpX < 16 && lscpY < 16 else { throw DecodeError.invalidBlockData }

    var currentIdx = 0
    let lscpIdx = lscpY * 16 + lscpX
    while currentIdx <= lscpIdx {
        let startY = currentIdx / 16
        let startX = currentIdx % 16
        let isParentZero = parentBlock.rowPointer(y: startY >> 1)[startX >> 1] == 0
        let (run, val) = try decodeCoeffRun(decoder: &decoder, isParentZero: isParentZero)

        currentIdx += run
        if currentIdx <= lscpIdx {
            let y = currentIdx / 16
            let x = currentIdx % 16
            let ptr = block.rowPointer(y: y)
            ptr[x] = val
        }
        currentIdx += 1
    }
}

@inline(__always)
func blockDecode8V(decoder: inout EntropyDecoder, block: BlockView) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        return
    }

    let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
    let lscpY = Int(try decodeExpGolomb(decoder: &decoder))

    var currentIdx = 0
    let lscpIdx = lscpX * 8 + lscpY
    while currentIdx <= lscpIdx {
        let isParentZero = false
        let (run, val) = try decodeCoeffRun(decoder: &decoder, isParentZero: isParentZero)

        currentIdx += run
        if currentIdx <= lscpIdx {
            let x = currentIdx / 8
            let y = currentIdx % 8
            let ptr = block.rowPointer(y: y)
            ptr[x] = val
        }
        currentIdx += 1
    }
}

@inline(__always)
func blockDecode8VWithParentBlock(decoder: inout EntropyDecoder, block: BlockView, parentBlock: BlockView) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        return
    }

    let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
    let lscpY = Int(try decodeExpGolomb(decoder: &decoder))

    var currentIdx = 0
    let lscpIdx = lscpX * 8 + lscpY
    while currentIdx <= lscpIdx {
        let startX = currentIdx / 8
        let startY = currentIdx % 8
        let isParentZero = parentBlock.rowPointer(y: startY >> 1)[startX >> 1] == 0
        let (run, val) = try decodeCoeffRun(decoder: &decoder, isParentZero: isParentZero)

        currentIdx += run
        if currentIdx <= lscpIdx {
            let x = currentIdx / 8
            let y = currentIdx % 8
            let ptr = block.rowPointer(y: y)
            ptr[x] = val
        }
        currentIdx += 1
    }
}

@inline(__always)
func blockDecode8H(decoder: inout EntropyDecoder, block: BlockView) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        return
    }

    let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
    let lscpY = Int(try decodeExpGolomb(decoder: &decoder))

    var currentIdx = 0
    let lscpIdx = lscpY * 8 + lscpX
    while currentIdx <= lscpIdx {
        let isParentZero = false
        let (run, val) = try decodeCoeffRun(decoder: &decoder, isParentZero: isParentZero)

        currentIdx += run
        if currentIdx <= lscpIdx {
            let y = currentIdx / 8
            let x = currentIdx % 8
            let ptr = block.rowPointer(y: y)
            ptr[x] = val
        }
        currentIdx += 1
    }
}

@inline(__always)
func blockDecode8HWithParentBlock(decoder: inout EntropyDecoder, block: BlockView, parentBlock: BlockView) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        return
    }

    let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
    let lscpY = Int(try decodeExpGolomb(decoder: &decoder))

    var currentIdx = 0
    let lscpIdx = lscpY * 8 + lscpX
    while currentIdx <= lscpIdx {
        let startY = currentIdx / 8
        let startX = currentIdx % 8
        let isParentZero = parentBlock.rowPointer(y: startY >> 1)[startX >> 1] == 0
        let (run, val) = try decodeCoeffRun(decoder: &decoder, isParentZero: isParentZero)

        currentIdx += run
        if currentIdx <= lscpIdx {
            let y = currentIdx / 8
            let x = currentIdx % 8
            let ptr = block.rowPointer(y: y)
            ptr[x] = val
        }
        currentIdx += 1
    }
}

@inline(__always)
func blockDecode4V(decoder: inout EntropyDecoder, block: BlockView) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        return
    }

    let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
    let lscpY = Int(try decodeExpGolomb(decoder: &decoder))
    guard lscpX < 4 && lscpY < 4 else { throw DecodeError.invalidBlockData }

    var currentIdx = 0
    let lscpIdx = lscpX * 4 + lscpY
    while currentIdx <= lscpIdx {
        let isParentZero = false
        let (run, val) = try decodeCoeffRun(decoder: &decoder, isParentZero: isParentZero)

        currentIdx += run
        if currentIdx <= lscpIdx {
            let x = currentIdx / 4
            let y = currentIdx % 4
            let ptr = block.rowPointer(y: y)
            ptr[x] = val
        }
        currentIdx += 1
    }
}

@inline(__always)
func blockDecode4VWithParentBlock(decoder: inout EntropyDecoder, block: BlockView, parentBlock: BlockView) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        return
    }

    let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
    let lscpY = Int(try decodeExpGolomb(decoder: &decoder))
    guard lscpX < 4 && lscpY < 4 else { throw DecodeError.invalidBlockData }

    var currentIdx = 0
    let lscpIdx = lscpX * 4 + lscpY
    while currentIdx <= lscpIdx {
        let startX = currentIdx / 4
        let startY = currentIdx % 4
        let isParentZero = parentBlock.rowPointer(y: startY >> 1)[startX >> 1] == 0
        let (run, val) = try decodeCoeffRun(decoder: &decoder, isParentZero: isParentZero)

        currentIdx += run
        if currentIdx <= lscpIdx {
            let x = currentIdx / 4
            let y = currentIdx % 4
            let ptr = block.rowPointer(y: y)
            ptr[x] = val
        }
        currentIdx += 1
    }
}

@inline(__always)
func blockDecode4H(decoder: inout EntropyDecoder, block: BlockView) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        return
    }

    let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
    let lscpY = Int(try decodeExpGolomb(decoder: &decoder))
    guard lscpX < 4 && lscpY < 4 else { throw DecodeError.invalidBlockData }

    var currentIdx = 0
    let lscpIdx = lscpY * 4 + lscpX
    while currentIdx <= lscpIdx {
        let isParentZero = false
        let (run, val) = try decodeCoeffRun(decoder: &decoder, isParentZero: isParentZero)

        currentIdx += run
        if currentIdx <= lscpIdx {
            let y = currentIdx / 4
            let x = currentIdx % 4
            let ptr = block.rowPointer(y: y)
            ptr[x] = val
        }
        currentIdx += 1
    }
}

@inline(__always)
func blockDecode4HWithParentBlock(decoder: inout EntropyDecoder, block: BlockView, parentBlock: BlockView) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        return
    }

    let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
    let lscpY = Int(try decodeExpGolomb(decoder: &decoder))
    guard lscpX < 4 && lscpY < 4 else { throw DecodeError.invalidBlockData }

    var currentIdx = 0
    let lscpIdx = lscpY * 4 + lscpX
    while currentIdx <= lscpIdx {
        let startY = currentIdx / 4
        let startX = currentIdx % 4
        let isParentZero = parentBlock.rowPointer(y: startY >> 1)[startX >> 1] == 0
        let (run, val) = try decodeCoeffRun(decoder: &decoder, isParentZero: isParentZero)

        currentIdx += run
        if currentIdx <= lscpIdx {
            let y = currentIdx / 4
            let x = currentIdx % 4
            let ptr = block.rowPointer(y: y)
            ptr[x] = val
        }
        currentIdx += 1
    }
}

@inline(__always)
func blockDecodeDPCM4(decoder: inout EntropyDecoder, block: BlockView, lastVal: inout Int16) throws {
    let hasNonZero = try decoder.decodeBypass()
    var lscpIdx = -1
    if hasNonZero == 1 {
        let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
        let lscpY = Int(try decodeExpGolomb(decoder: &decoder))
        guard lscpX < 4 && lscpY < 4 else { throw DecodeError.invalidBlockDataContext("DPCM4 lscp out of range: (\(lscpX), \(lscpY))") }
        lscpIdx = lscpY * 4 + lscpX
    }

    var currentIdx = 0
    while currentIdx <= lscpIdx {
        let (run, val) = try decodeCoeffRun(decoder: &decoder, isParentZero: false)

        currentIdx += run
        if currentIdx <= lscpIdx {
            let y = currentIdx / 4
            let x = currentIdx % 4
            let ptr = block.rowPointer(y: y)
            ptr[x] = val
        }
        currentIdx += 1
    }

    let ptr0 = block.rowPointer(y: 0)
    let ptr1 = block.rowPointer(y: 1)
    let ptr2 = block.rowPointer(y: 2)
    let ptr3 = block.rowPointer(y: 3)

    ptr0[0] = ptr0[0] &+ lastVal
    ptr0[1] = ptr0[1] &+ ptr0[0]
    ptr0[2] = ptr0[2] &+ ptr0[1]
    ptr0[3] = ptr0[3] &+ ptr0[2]

    ptr1[0] = ptr1[0] &+ ptr0[0]
    ptr1[1] = ptr1[1] &+ predictMED(ptr1[0], ptr0[1], ptr0[0])
    ptr1[2] = ptr1[2] &+ predictMED(ptr1[1], ptr0[2], ptr0[1])
    ptr1[3] = ptr1[3] &+ predictMED(ptr1[2], ptr0[3], ptr0[2])

    ptr2[0] = ptr2[0] &+ ptr1[0]
    ptr2[1] = ptr2[1] &+ predictMED(ptr2[0], ptr1[1], ptr1[0])
    ptr2[2] = ptr2[2] &+ predictMED(ptr2[1], ptr1[2], ptr1[1])
    ptr2[3] = ptr2[3] &+ predictMED(ptr2[2], ptr1[3], ptr1[2])

    ptr3[0] = ptr3[0] &+ ptr2[0]
    ptr3[1] = ptr3[1] &+ predictMED(ptr3[0], ptr2[1], ptr2[0])
    ptr3[2] = ptr3[2] &+ predictMED(ptr3[1], ptr2[2], ptr2[1])
    ptr3[3] = ptr3[3] &+ predictMED(ptr3[2], ptr2[3], ptr2[2])

    lastVal = ptr3[3]
}

@inline(__always)
func blockDecodeDPCM8(decoder: inout EntropyDecoder, block: BlockView, lastVal: inout Int16) throws {
    let hasNonZero = try decoder.decodeBypass()
    var lscpIdx = -1
    if hasNonZero == 1 {
        let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
        let lscpY = Int(try decodeExpGolomb(decoder: &decoder))
        guard lscpX < 8 && lscpY < 8 else { throw DecodeError.invalidBlockDataContext("DPCM8 lscp out of range: (\(lscpX), \(lscpY))") }
        lscpIdx = lscpY * 8 + lscpX
    }

    var currentIdx = 0
    while currentIdx <= lscpIdx {
        let (run, val) = try decodeCoeffRun(decoder: &decoder, isParentZero: false)

        currentIdx += run
        if currentIdx <= lscpIdx {
            let y = currentIdx / 8
            let x = currentIdx % 8
            let ptr = block.rowPointer(y: y)
            ptr[x] = val
        }
        currentIdx += 1
    }

    let ptrY0 = block.rowPointer(y: 0)
    ptrY0[0] = ptrY0[0] &+ lastVal
    for x in 1..<8 {
        ptrY0[x] = ptrY0[x] &+ ptrY0[x - 1]
    }
    
    var last = ptrY0[7]
    for y in 1..<8 {
        let ptrY = block.rowPointer(y: y)
        let ptrPrevY = block.rowPointer(y: y - 1)
        
        ptrY[0] = ptrY[0] &+ ptrPrevY[0]
        for x in 1..<8 {
            ptrY[x] = ptrY[x] &+ predictMED(ptrY[x - 1], ptrPrevY[x], ptrPrevY[x - 1])
        }
        last = ptrY[7]
    }
    lastVal = last
}

@inline(__always)
func blockDecodeDPCM16(decoder: inout EntropyDecoder, block: BlockView, lastVal: inout Int16) throws {
    let hasNonZero = try decoder.decodeBypass()
    var lscpIdx = -1
    if hasNonZero == 1 {
        let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
        let lscpY = Int(try decodeExpGolomb(decoder: &decoder))
        guard lscpX < 16 && lscpY < 16 else { throw DecodeError.invalidBlockData }
        lscpIdx = lscpY * 16 + lscpX
    }

    var currentIdx = 0
    while currentIdx <= lscpIdx {
        let (run, val) = try decodeCoeffRun(decoder: &decoder, isParentZero: false)

        currentIdx += run
        if currentIdx <= lscpIdx {
            let y = currentIdx / 16
            let x = currentIdx % 16
            let ptr = block.rowPointer(y: y)
            ptr[x] = val
        }
        currentIdx += 1
    }

    let ptrY0 = block.rowPointer(y: 0)
    ptrY0[0] = ptrY0[0] &+ lastVal
    for x in 1..<16 {
        ptrY0[x] = ptrY0[x] &+ ptrY0[x - 1]
    }
    
    var last = ptrY0[15]
    for y in 1..<16 {
        let ptrY = block.rowPointer(y: y)
        let ptrPrevY = block.rowPointer(y: y - 1)
        
        ptrY[0] = ptrY[0] &+ ptrPrevY[0]
        for x in 1..<16 {
            ptrY[x] = ptrY[x] &+ predictMED(ptrY[x - 1], ptrPrevY[x], ptrPrevY[x - 1])
        }
        last = ptrY[15]
    }
    lastVal = last
}

// MARK: - Internal Decode Functions

@inline(__always)
func decodeLayer32(r: [UInt8], pool: BlockViewPool, layer: UInt8, dx: Int, dy: Int, prev: Image16, parentYBlocks: [BlockView]?, parentCbBlocks: [BlockView]?, parentCrBlocks: [BlockView]?, predictedPd: PlaneData420? = nil, nextPd: PlaneData420? = nil, mvs: [MotionVector]? = nil, refDirs: [Bool]? = nil, roundOffset: Int) async throws -> Image16 {
    var offset = 0
    let qtY = QuantizationTable(baseStep: Int(try readUInt16BEFromBytes(r, offset: &offset)), isChroma: false, layerIndex: Int(layer))
    let qtC = QuantizationTable(baseStep: Int(try readUInt16BEFromBytes(r, offset: &offset)), isChroma: true, layerIndex: Int(layer))
    
    let bufYLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufYLen) <= r.count else { throw DecodeError.invalidBlockData }
    let bufY = Array(r[offset..<(offset + bufYLen)])
    offset += bufYLen
    
    let bufCbLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufCbLen) <= r.count else { throw DecodeError.invalidBlockDataContext("Layer32 Cb overflow: offset=\(offset) len=\(bufCbLen) total=\(r.count)") }
    let bufCb = Array(r[offset..<(offset + bufCbLen)])
    offset += bufCbLen
    
    let bufCrLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufCrLen) <= r.count else { throw DecodeError.invalidBlockDataContext("Layer32 Cr overflow: offset=\(offset) len=\(bufCrLen) total=\(r.count)") }
    let bufCr = Array(r[offset..<(offset + bufCrLen)])
    offset += bufCrLen
    
    var sub = Image16(width: dx, height: dy, pool: pool)
    
    let rowCountY = (dy + 32 - 1) / 32
    let colCountY = (dx + 32 - 1) / 32
    let yBlocks: [BlockView]
    if let p = parentYBlocks {
        yBlocks = try decodePlaneSubbands32WithParentBlocks(data: bufY, pool: pool, blockCount: rowCountY * colCountY, parentBlocks: p)
    } else {
        yBlocks = try decodePlaneSubbands32(data: bufY, pool: pool, blockCount: rowCountY * colCountY)
    }
    
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let rowCountCb = (cbDy + 32 - 1) / 32
    let colCountCb = (cbDx + 32 - 1) / 32
    let cbBlocks: [BlockView]
    if let p = parentCbBlocks {
        cbBlocks = try decodePlaneSubbands32WithParentBlocks(data: bufCb, pool: pool, blockCount: rowCountCb * colCountCb, parentBlocks: p)
    } else {
        cbBlocks = try decodePlaneSubbands32(data: bufCb, pool: pool, blockCount: rowCountCb * colCountCb)
    }
    
    let rowCountCr = (cbDy + 32 - 1) / 32
    let colCountCr = (cbDx + 32 - 1) / 32
    let crBlocks: [BlockView]
    if let p = parentCrBlocks {
        crBlocks = try decodePlaneSubbands32WithParentBlocks(data: bufCr, pool: pool, blockCount: rowCountCr * colCountCr, parentBlocks: p)
    } else {
        crBlocks = try decodePlaneSubbands32(data: bufCr, pool: pool, blockCount: rowCountCr * colCountCr)
    }
    
    let chunkSize = 16
    let taskCountY = (rowCountY + chunkSize - 1) / chunkSize
    try await withThrowingTaskGroup(of: [(BlockView, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountY {
            group.addTask { return decodeLayer32ProcessY(pool: pool, taskIdx: taskIdx, chunkSize: chunkSize, rowCount: rowCountY, dx: dx, colCount: colCountY, blocks: yBlocks, prev: prev, qt: qtY) }
        }
        for try await res in group {
            for j in res.indices {
                var blk = res[j].0
                let w = res[j].1
                let h = res[j].2
                sub.updateY(data: &blk, startX: w, startY: h, size: 32)
            }
        }
    }
    
    let taskCountCb = (rowCountCb + chunkSize - 1) / chunkSize
    try await withThrowingTaskGroup(of: [(BlockView, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountCb {
            group.addTask { return decodeLayer32ProcessCb(pool: pool, taskIdx: taskIdx, chunkSize: chunkSize, rowCount: rowCountCb, dx: cbDx, colCount: colCountCb, blocks: cbBlocks, prev: prev, qt: qtC) }
        }
        for try await res in group {
            for j in res.indices {
                var blk = res[j].0
                let w = res[j].1
                let h = res[j].2
                sub.updateCb(data: &blk, startX: w, startY: h, size: 32)
            }
        }
    }
    
    let taskCountCr = (rowCountCr + chunkSize - 1) / chunkSize
    try await withThrowingTaskGroup(of: [(BlockView, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountCr {
            group.addTask { return decodeLayer32ProcessCr(pool: pool, taskIdx: taskIdx, chunkSize: chunkSize, rowCount: rowCountCr, dx: cbDx, colCount: colCountCr, blocks: crBlocks, prev: prev, qt: qtC) }
        }
        for try await res in group {
            for j in res.indices {
                var blk = res[j].0
                let w = res[j].1
                let h = res[j].2
                sub.updateCr(data: &blk, startX: w, startY: h, size: 32)
            }
        }
    }
    
    if let tPrev = predictedPd, let mvs = mvs {
        if let tNext = nextPd, let dirs = refDirs {
            // bidirectional prediction: use refDirs to select forward/backward prediction for each block
            applyBidirectionalMotionCompensationPixelsLuma32(plane: &sub.y, prevPlane: tPrev.y, nextPlane: tNext.y, mvs: mvs, refDirs: dirs, width: dx, height: dy, roundOffset: roundOffset)
            applyBidirectionalMotionCompensationPixelsChroma16(plane: &sub.cb, prevPlane: tPrev.cb, nextPlane: tNext.cb, mvs: mvs, refDirs: dirs, width: cbDx, height: cbDy, roundOffset: roundOffset)
            applyBidirectionalMotionCompensationPixelsChroma16(plane: &sub.cr, prevPlane: tPrev.cr, nextPlane: tNext.cr, mvs: mvs, refDirs: dirs, width: cbDx, height: cbDy, roundOffset: roundOffset)
        } else {
            // forward prediction only
            applyMotionCompensationPixelsLuma32(plane: &sub.y, prevPlane: tPrev.y, mvs: mvs, width: dx, height: dy, roundOffset: roundOffset)
            applyMotionCompensationPixelsChroma16(plane: &sub.cb, prevPlane: tPrev.cb, mvs: mvs, width: cbDx, height: cbDy, roundOffset: roundOffset)
            applyMotionCompensationPixelsChroma16(plane: &sub.cr, prevPlane: tPrev.cr, mvs: mvs, width: cbDx, height: cbDy, roundOffset: roundOffset)
        }
    }

    applyDeblockingFilter(plane: &sub.y, width: dx, height: dy, blockSize: 32, qStep: Int(qtY.step))
    applyDeblockingFilter(plane: &sub.cb, width: cbDx, height: cbDy, blockSize: 32, qStep: Int(qtC.step))
    applyDeblockingFilter(plane: &sub.cr, width: cbDx, height: cbDy, blockSize: 32, qStep: Int(qtC.step))
    
    return sub
}

@inline(__always)
func decodeLayer16(r: [UInt8], pool: BlockViewPool, layer: UInt8, dx: Int, dy: Int, prev: Image16, parentYBlocks: [BlockView]?, parentCbBlocks: [BlockView]?, parentCrBlocks: [BlockView]?) async throws -> (Image16, [BlockView], [BlockView], [BlockView]) {
    var offset = 0
    let qtY = QuantizationTable(baseStep: Int(try readUInt16BEFromBytes(r, offset: &offset)), isChroma: false, layerIndex: Int(layer))
    let qtC = QuantizationTable(baseStep: Int(try readUInt16BEFromBytes(r, offset: &offset)), isChroma: true, layerIndex: Int(layer))
    
    let bufYLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufYLen) <= r.count else { throw DecodeError.invalidBlockDataContext("Layer16 Y overflow: offset=\(offset) len=\(bufYLen) total=\(r.count)") }
    let bufY = Array(r[offset..<(offset + bufYLen)])
    offset += bufYLen
    
    let bufCbLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufCbLen) <= r.count else { throw DecodeError.invalidBlockDataContext("Layer16 Cb overflow: offset=\(offset) len=\(bufCbLen) total=\(r.count)") }
    let bufCb = Array(r[offset..<(offset + bufCbLen)])
    offset += bufCbLen
    
    let bufCrLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufCrLen) <= r.count else { throw DecodeError.invalidBlockDataContext("Layer16 Cr overflow: offset=\(offset) len=\(bufCrLen) total=\(r.count)") }
    let bufCr = Array(r[offset..<(offset + bufCrLen)])
    offset += bufCrLen
    
    var sub = Image16(width: dx, height: dy, pool: pool)
    
    let rowCountY = (dy + 16 - 1) / 16
    let colCountY = (dx + 16 - 1) / 16
    let yBlocks: [BlockView]
    if let p = parentYBlocks {
        yBlocks = try decodePlaneSubbands16WithParentBlocks(data: bufY, pool: pool, blockCount: rowCountY * colCountY, parentBlocks: p)
    } else {
        yBlocks = try decodePlaneSubbands16(data: bufY, pool: pool, blockCount: rowCountY * colCountY)
    }
    
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let rowCountCb = (cbDy + 16 - 1) / 16
    let colCountCb = (cbDx + 16 - 1) / 16
    let cbBlocks: [BlockView]
    if let p = parentCbBlocks {
        cbBlocks = try decodePlaneSubbands16WithParentBlocks(data: bufCb, pool: pool, blockCount: rowCountCb * colCountCb, parentBlocks: p)
    } else {
        cbBlocks = try decodePlaneSubbands16(data: bufCb, pool: pool, blockCount: rowCountCb * colCountCb)
    }
    
    let rowCountCr = (cbDy + 16 - 1) / 16
    let colCountCr = (cbDx + 16 - 1) / 16
    let crBlocks: [BlockView]
    if let p = parentCrBlocks {
        crBlocks = try decodePlaneSubbands16WithParentBlocks(data: bufCr, pool: pool, blockCount: rowCountCr * colCountCr, parentBlocks: p)
    } else {
        crBlocks = try decodePlaneSubbands16(data: bufCr, pool: pool, blockCount: rowCountCr * colCountCr)
    }
    
    let chunkSize = 16
    let taskCountY = (rowCountY + chunkSize - 1) / chunkSize
    let taskCountCb = (rowCountCb + chunkSize - 1) / chunkSize
    let taskCountCr = (rowCountCr + chunkSize - 1) / chunkSize

    try await withThrowingTaskGroup(of: [(BlockView, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountY {
            group.addTask { return decodeLayer16ProcessY(pool: pool, taskIdx: taskIdx, chunkSize: chunkSize, rowCount: rowCountY, dx: dx, colCount: colCountY, blocks: yBlocks, prev: prev, qt: qtY) }
        }
        for try await res in group {
            for j in res.indices {
                var blk = res[j].0
                let w = res[j].1
                let h = res[j].2
                sub.updateY(data: &blk, startX: w, startY: h, size: 16)
            }
        }
    }

    try await withThrowingTaskGroup(of: [(BlockView, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountCb {
            group.addTask { return decodeLayer16ProcessCb(pool: pool, taskIdx: taskIdx, chunkSize: chunkSize, rowCount: rowCountCb, dx: cbDx, colCount: colCountCb, blocks: cbBlocks, prev: prev, qt: qtC) }
        }
        for try await res in group {
            for j in res.indices {
                var blk = res[j].0
                let w = res[j].1
                let h = res[j].2
                sub.updateCb(data: &blk, startX: w, startY: h, size: 16)
            }
        }
    }
    
    try await withThrowingTaskGroup(of: [(BlockView, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountCr {
            group.addTask { return decodeLayer16ProcessCr(pool: pool, taskIdx: taskIdx, chunkSize: chunkSize, rowCount: rowCountCr, dx: cbDx, colCount: colCountCr, blocks: crBlocks, prev: prev, qt: qtC) }
        }
        for try await res in group {
            for j in res.indices {
                var blk = res[j].0
                let w = res[j].1
                let h = res[j].2
                sub.updateCr(data: &blk, startX: w, startY: h, size: 16)
            }
        }
    }
    
    return (sub, yBlocks, cbBlocks, crBlocks)
}

@inline(__always)
func decodeBase8(r: [UInt8], pool: BlockViewPool, layer: UInt8, dx: Int, dy: Int, isIFrame: Bool) async throws -> (Image16, [BlockView], [BlockView], [BlockView]) {
    var offset = 0
    let qtY = QuantizationTable(baseStep: Int(try readUInt16BEFromBytes(r, offset: &offset)), isChroma: false, layerIndex: Int(layer))
    let qtC = QuantizationTable(baseStep: Int(try readUInt16BEFromBytes(r, offset: &offset)), isChroma: true, layerIndex: Int(layer))
    
    let bufYLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufYLen) <= r.count else { throw DecodeError.invalidBlockDataContext("Base8 Y overflow: offset=\(offset) len=\(bufYLen) total=\(r.count)") }
    let bufY = Array(r[offset..<(offset + bufYLen)])
    offset += bufYLen
    
    let bufCbLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufCbLen) <= r.count else { throw DecodeError.invalidBlockDataContext("Base8 Cb overflow: offset=\(offset) len=\(bufCbLen) total=\(r.count)") }
    let bufCb = Array(r[offset..<(offset + bufCbLen)])
    offset += bufCbLen
    
    let bufCrLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufCrLen) <= r.count else { throw DecodeError.invalidBlockDataContext("Base8 Cr overflow: offset=\(offset) len=\(bufCrLen) total=\(r.count)") }
    let bufCr = Array(r[offset..<(offset + bufCrLen)])
    offset += bufCrLen
    
    var sub = Image16(width: dx, height: dy, pool: pool)
    
    let rowCountY = (dy + 8 - 1) / 8
    let colCountY = (dx + 8 - 1) / 8
    let yBlocks = try decodePlaneBaseSubbands8(data: bufY, pool: pool, blockCount: rowCountY * colCountY, isIFrame: isIFrame)
    
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let rowCountCb = (cbDy + 8 - 1) / 8
    let colCountCb = (cbDx + 8 - 1) / 8
    let cbBlocks = try decodePlaneBaseSubbands8(data: bufCb, pool: pool, blockCount: rowCountCb * colCountCb, isIFrame: isIFrame)
    
    let rowCountCr = (cbDy + 8 - 1) / 8
    let colCountCr = (cbDx + 8 - 1) / 8
    let crBlocks = try decodePlaneBaseSubbands8(data: bufCr, pool: pool, blockCount: rowCountCr * colCountCr, isIFrame: isIFrame)
    
    let chunkSize = 16
    let taskCountY = (rowCountY + chunkSize - 1) / chunkSize
    let taskCountCb = (rowCountCb + chunkSize - 1) / chunkSize
    let taskCountCr = (rowCountCr + chunkSize - 1) / chunkSize

    try await withThrowingTaskGroup(of: [(BlockView, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountY {
            group.addTask { return decodeBase8ProcessY(pool: pool, taskIdx: taskIdx, chunkSize: chunkSize, rowCount: rowCountY, dx: dx, colCount: colCountY, blocks: yBlocks, qt: qtY) }
        }
        for try await res in group {
            for j in res.indices {
                var blk = res[j].0
                let w = res[j].1
                let h = res[j].2
                sub.updateY(data: &blk, startX: w, startY: h, size: 8)
            }
        }
    }

    try await withThrowingTaskGroup(of: [(BlockView, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountCb {
            group.addTask { return decodeBase8ProcessCb(pool: pool, taskIdx: taskIdx, chunkSize: chunkSize, rowCount: rowCountCb, dx: cbDx, colCount: colCountCb, blocks: cbBlocks, qt: qtC) }
        }
        for try await res in group {
            for j in res.indices {
                var blk = res[j].0
                let w = res[j].1
                let h = res[j].2
                sub.updateCb(data: &blk, startX: w, startY: h, size: 8)
            }
        }
    }
    
    try await withThrowingTaskGroup(of: [(BlockView, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountCr {
            group.addTask { return decodeBase8ProcessCr(pool: pool, taskIdx: taskIdx, chunkSize: chunkSize, rowCount: rowCountCr, dx: cbDx, colCount: colCountCr, blocks: crBlocks, qt: qtC) }
        }
        for try await res in group {
            for j in res.indices {
                var blk = res[j].0
                let w = res[j].1
                let h = res[j].2
                sub.updateCr(data: &blk, startX: w, startY: h, size: 8)
            }
        }
    }
        
    return (sub, yBlocks, cbBlocks, crBlocks)
}

@Sendable @inline(__always)
func decodeLayer32ProcessY(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable) -> [(BlockView, Int, Int)] {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return [] }
    var rowResults: [(BlockView, Int, Int)] = []
    for i in startRow..<endRow {
        let h: Int = i * 32
        for (xIdx, w) in stride(from: 0, to: dx, by: 32).enumerated() {
            let blockIndex: Int = i * colCount + xIdx
            let block: BlockView = blocks[blockIndex]
            let half: Int = 32 / 2
            prev.readY(x: w / 2, y: h / 2, size: half, into: block)
            let view = block
            let base = view.base
            let hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 32)
            let lhView = BlockView(base: base.advanced(by: half * 32), width: half, height: half, stride: 32)
            let hhView = BlockView(base: base.advanced(by: half * 32 + half), width: half, height: half, stride: 32)
            dequantizeSIMDSignedMapping(hlView, q: qt.qMid)
            dequantizeSIMDSignedMapping(lhView, q: qt.qMid)
            dequantizeSIMDSignedMapping(hhView, q: qt.qHigh)
            inverseDWT2DBlock32(view)
            rowResults.append((block, w, h))
        }
    }
    return rowResults
}

@Sendable @inline(__always)
func decodeLayer32ProcessCb(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable) -> [(BlockView, Int, Int)] {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return [] }
    var rowResults: [(BlockView, Int, Int)] = []
    for i in startRow..<endRow {
        let h: Int = i * 32
        for (xIdx, w) in stride(from: 0, to: dx, by: 32).enumerated() {
            let blockIndex: Int = i * colCount + xIdx
            let block: BlockView = blocks[blockIndex]
            let half: Int = 32 / 2
            prev.readCb(x: w / 2, y: h / 2, size: half, into: block)
            let view = block
            let base = view.base
            let hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 32)
            let lhView = BlockView(base: base.advanced(by: half * 32), width: half, height: half, stride: 32)
            let hhView = BlockView(base: base.advanced(by: half * 32 + half), width: half, height: half, stride: 32)
            dequantizeSIMDSignedMapping(hlView, q: qt.qMid)
            dequantizeSIMDSignedMapping(lhView, q: qt.qMid)
            dequantizeSIMDSignedMapping(hhView, q: qt.qHigh)
            inverseDWT2DBlock32(view)
            rowResults.append((block, w, h))
        }
    }
    return rowResults
}

@Sendable @inline(__always)
func decodeLayer32ProcessCr(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable) -> [(BlockView, Int, Int)] {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return [] }
    var rowResults: [(BlockView, Int, Int)] = []
    for i in startRow..<endRow {
        let h: Int = i * 32
        for (xIdx, w) in stride(from: 0, to: dx, by: 32).enumerated() {
            let blockIndex: Int = i * colCount + xIdx
            let block: BlockView = blocks[blockIndex]
            let half: Int = 32 / 2
            prev.readCr(x: w / 2, y: h / 2, size: half, into: block)
            let view = block
            let base = view.base
            let hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 32)
            let lhView = BlockView(base: base.advanced(by: half * 32), width: half, height: half, stride: 32)
            let hhView = BlockView(base: base.advanced(by: half * 32 + half), width: half, height: half, stride: 32)
            dequantizeSIMDSignedMapping(hlView, q: qt.qMid)
            dequantizeSIMDSignedMapping(lhView, q: qt.qMid)
            dequantizeSIMDSignedMapping(hhView, q: qt.qHigh)
            inverseDWT2DBlock32(view)
            rowResults.append((block, w, h))
        }
    }
    return rowResults
}

@Sendable @inline(__always)
func decodeLayer16ProcessY(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable) -> [(BlockView, Int, Int)] {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return [] }
    var rowResults: [(BlockView, Int, Int)] = []
    for i in startRow..<endRow {
        let h: Int = i * 16
        for (xIdx, w) in stride(from: 0, to: dx, by: 16).enumerated() {
            let blockIndex: Int = i * colCount + xIdx
            let block: BlockView = blocks[blockIndex]
            let half: Int = 16 / 2
            prev.readY(x: w / 2, y: h / 2, size: half, into: block)
            let view = block
            let base = view.base
            let hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 16)
            let lhView = BlockView(base: base.advanced(by: half * 16), width: half, height: half, stride: 16)
            let hhView = BlockView(base: base.advanced(by: half * 16 + half), width: half, height: half, stride: 16)
            dequantizeSIMDSignedMapping(hlView, q: qt.qMid)
            dequantizeSIMDSignedMapping(lhView, q: qt.qMid)
            dequantizeSIMDSignedMapping(hhView, q: qt.qHigh)
            inverseDWT2DBlock16(view)
            rowResults.append((block, w, h))
        }
    }
    return rowResults
}

@Sendable @inline(__always)
func decodeLayer16ProcessCb(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable) -> [(BlockView, Int, Int)] {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return [] }
    var rowResults: [(BlockView, Int, Int)] = []
    for i in startRow..<endRow {
        let h: Int = i * 16
        for (xIdx, w) in stride(from: 0, to: dx, by: 16).enumerated() {
            let blockIndex: Int = i * colCount + xIdx
            let block: BlockView = blocks[blockIndex]
            let half: Int = 16 / 2
            prev.readCb(x: w / 2, y: h / 2, size: half, into: block)
            let view = block
            let base = view.base
            let hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 16)
            let lhView = BlockView(base: base.advanced(by: half * 16), width: half, height: half, stride: 16)
            let hhView = BlockView(base: base.advanced(by: half * 16 + half), width: half, height: half, stride: 16)
            dequantizeSIMDSignedMapping8(hlView, q: qt.qMid)
            dequantizeSIMDSignedMapping8(lhView, q: qt.qMid)
            dequantizeSIMDSignedMapping8(hhView, q: qt.qHigh)
            inverseDWT2DBlock16(view)
            rowResults.append((block, w, h))
        }
    }
    return rowResults
}

@Sendable @inline(__always)
func decodeLayer16ProcessCr(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable) -> [(BlockView, Int, Int)] {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return [] }
    var rowResults: [(BlockView, Int, Int)] = []
    for i in startRow..<endRow {
        let h: Int = i * 16
        for (xIdx, w) in stride(from: 0, to: dx, by: 16).enumerated() {
            let blockIndex: Int = i * colCount + xIdx
            let block: BlockView = blocks[blockIndex]
            let half: Int = 16 / 2
            prev.readCr(x: w / 2, y: h / 2, size: half, into: block)
            let view = block
            let base = view.base
            let hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 16)
            let lhView = BlockView(base: base.advanced(by: half * 16), width: half, height: half, stride: 16)
            let hhView = BlockView(base: base.advanced(by: half * 16 + half), width: half, height: half, stride: 16)
            dequantizeSIMDSignedMapping8(hlView, q: qt.qMid)
            dequantizeSIMDSignedMapping8(lhView, q: qt.qMid)
            dequantizeSIMDSignedMapping8(hhView, q: qt.qHigh)
            inverseDWT2DBlock16(view)
            rowResults.append((block, w, h))
        }
    }
    return rowResults
}

@Sendable @inline(__always)
func decodeBase8ProcessY(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], qt: QuantizationTable) -> [(BlockView, Int, Int)] {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return [] }
    var rowResults: [(BlockView, Int, Int)] = []
    for i in startRow..<endRow {
        let h: Int = i * 8
        for (xIdx, w) in stride(from: 0, to: dx, by: 8).enumerated() {
            let blockIndex: Int = i * colCount + xIdx
            let block: BlockView = blocks[blockIndex]
            let half: Int = 8 / 2
            let view = block
            let base = view.base
            let llView = BlockView(base: base, width: half, height: half, stride: 8)
            let hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 8)
            let lhView = BlockView(base: base.advanced(by: half * 8), width: half, height: half, stride: 8)
            let hhView = BlockView(base: base.advanced(by: half * 8 + half), width: half, height: half, stride: 8)
            dequantizeSIMD(llView, q: qt.qLow)
            dequantizeSIMDSignedMapping(hlView, q: qt.qMid)
            dequantizeSIMDSignedMapping(lhView, q: qt.qMid)
            dequantizeSIMDSignedMapping(hhView, q: qt.qHigh)
            inverseDWT2DBlock8(view)
            rowResults.append((block, w, h))
        }
    }
    return rowResults
}

@Sendable @inline(__always)
func decodeBase8ProcessCb(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], qt: QuantizationTable) -> [(BlockView, Int, Int)] {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return [] }
    var rowResults: [(BlockView, Int, Int)] = []
    for i in startRow..<endRow {
        let h: Int = i * 8
        for (xIdx, w) in stride(from: 0, to: dx, by: 8).enumerated() {
            let blockIndex: Int = i * colCount + xIdx
            let block: BlockView = blocks[blockIndex]
            let half: Int = 8 / 2
            let view = block
            let base = view.base
            let llView = BlockView(base: base, width: half, height: half, stride: 8)
            let hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 8)
            let lhView = BlockView(base: base.advanced(by: half * 8), width: half, height: half, stride: 8)
            let hhView = BlockView(base: base.advanced(by: half * 8 + half), width: half, height: half, stride: 8)
            dequantizeSIMD(llView, q: qt.qLow)
            dequantizeSIMDSignedMapping(hlView, q: qt.qMid)
            dequantizeSIMDSignedMapping(lhView, q: qt.qMid)
            dequantizeSIMDSignedMapping(hhView, q: qt.qHigh)
            inverseDWT2DBlock8(view)
            rowResults.append((block, w, h))
        }
    }
    return rowResults
}

@Sendable @inline(__always)
func decodeBase8ProcessCr(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], qt: QuantizationTable) -> [(BlockView, Int, Int)] {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return [] }
    var rowResults: [(BlockView, Int, Int)] = []
    for i in startRow..<endRow {
        let h: Int = i * 8
        for (xIdx, w) in stride(from: 0, to: dx, by: 8).enumerated() {
            let blockIndex: Int = i * colCount + xIdx
            let block: BlockView = blocks[blockIndex]
            let half: Int = 8 / 2
            let view = block
            let base = view.base
            let llView = BlockView(base: base, width: half, height: half, stride: 8)
            let hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 8)
            let lhView = BlockView(base: base.advanced(by: half * 8), width: half, height: half, stride: 8)
            let hhView = BlockView(base: base.advanced(by: half * 8 + half), width: half, height: half, stride: 8)
            dequantizeSIMD(llView, q: qt.qLow)
            dequantizeSIMDSignedMapping(hlView, q: qt.qMid)
            dequantizeSIMDSignedMapping(lhView, q: qt.qMid)
            dequantizeSIMDSignedMapping(hhView, q: qt.qHigh)
            inverseDWT2DBlock8(view)
            rowResults.append((block, w, h))
        }
    }
    return rowResults
}