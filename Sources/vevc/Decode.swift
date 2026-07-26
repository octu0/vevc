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
func decodeCoeffRun(decoder: inout EntropyDecoder, context: UInt8) throws -> (Int, Int16) {
    let pair = decoder.readPair(context: context)
    return (pair.run, pair.val)
}

@inline(__always)
func blockDecode16V(decoder: inout EntropyDecoder, block: BlockView) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        return
    }

    let lscpX = Int(decoder.readPair(context: 5).run)
    let lscpY = Int(decoder.readPair(context: 5).run)
    guard lscpX < 16 && lscpY < 16 else { throw DecodeError.invalidBlockData }

    var currentIdx = 0
    var prevVal: Int16 = 0
    let lscpIdx = lscpX * 16 + lscpY
    while currentIdx <= lscpIdx {
        let isParentZero = false
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getContext(prevVal: prevVal, isParentZero: isParentZero))
        prevVal = val

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

    let lscpX = Int(decoder.readPair(context: 5).run)
    let lscpY = Int(decoder.readPair(context: 5).run)
    guard lscpX < 16 && lscpY < 16 else { throw DecodeError.invalidBlockData }

    var currentIdx = 0
    var prevVal: Int16 = 0
    let lscpIdx = lscpX * 16 + lscpY
    while currentIdx <= lscpIdx {
        let startX = currentIdx / 16
        let startY = currentIdx % 16
        let isParentZero = parentBlock.rowPointer(y: startY >> 1)[startX >> 1] == 0
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getContext(prevVal: prevVal, isParentZero: isParentZero))
        prevVal = val

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

    let lscpX = Int(decoder.readPair(context: 5).run)
    let lscpY = Int(decoder.readPair(context: 5).run)
    guard lscpX < 16 && lscpY < 16 else { throw DecodeError.invalidBlockData }

    var currentIdx = 0
    var prevVal: Int16 = 0
    let lscpIdx = lscpY * 16 + lscpX
    while currentIdx <= lscpIdx {
        let isParentZero = false
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getContext(prevVal: prevVal, isParentZero: isParentZero))
        prevVal = val

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

    let lscpX = Int(decoder.readPair(context: 5).run)
    let lscpY = Int(decoder.readPair(context: 5).run)
    guard lscpX < 16 && lscpY < 16 else { throw DecodeError.invalidBlockData }

    var currentIdx = 0
    var prevVal: Int16 = 0
    let lscpIdx = lscpY * 16 + lscpX
    while currentIdx <= lscpIdx {
        let startY = currentIdx / 16
        let startX = currentIdx % 16
        let isParentZero = parentBlock.rowPointer(y: startY >> 1)[startX >> 1] == 0
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getContext(prevVal: prevVal, isParentZero: isParentZero))
        prevVal = val

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

    let lscpX = Int(decoder.readPair(context: 5).run)
    let lscpY = Int(decoder.readPair(context: 5).run)

    var currentIdx = 0
    var prevVal: Int16 = 0
    let lscpIdx = lscpX * 8 + lscpY
    while currentIdx <= lscpIdx {
        let isParentZero = false
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getContext(prevVal: prevVal, isParentZero: isParentZero))
        prevVal = val

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

    let lscpX = Int(decoder.readPair(context: 5).run)
    let lscpY = Int(decoder.readPair(context: 5).run)

    var currentIdx = 0
    var prevVal: Int16 = 0
    let lscpIdx = lscpX * 8 + lscpY
    while currentIdx <= lscpIdx {
        let startX = currentIdx / 8
        let startY = currentIdx % 8
        let isParentZero = parentBlock.rowPointer(y: startY >> 1)[startX >> 1] == 0
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getContext(prevVal: prevVal, isParentZero: isParentZero))
        prevVal = val

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

    let lscpX = Int(decoder.readPair(context: 5).run)
    let lscpY = Int(decoder.readPair(context: 5).run)

    var currentIdx = 0
    var prevVal: Int16 = 0
    let lscpIdx = lscpY * 8 + lscpX
    while currentIdx <= lscpIdx {
        let isParentZero = false
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getContext(prevVal: prevVal, isParentZero: isParentZero))
        prevVal = val

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

    let lscpX = Int(decoder.readPair(context: 5).run)
    let lscpY = Int(decoder.readPair(context: 5).run)

    var currentIdx = 0
    var prevVal: Int16 = 0
    let lscpIdx = lscpY * 8 + lscpX
    while currentIdx <= lscpIdx {
        let startY = currentIdx / 8
        let startX = currentIdx % 8
        let isParentZero = parentBlock.rowPointer(y: startY >> 1)[startX >> 1] == 0
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getContext(prevVal: prevVal, isParentZero: isParentZero))
        prevVal = val

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

    let lscpX = Int(decoder.readPair(context: 5).run)
    let lscpY = Int(decoder.readPair(context: 5).run)
    guard lscpX < 4 && lscpY < 4 else { throw DecodeError.invalidBlockData }

    var currentIdx = 0
    var prevVal: Int16 = 0
    let lscpIdx = lscpX * 4 + lscpY
    while currentIdx <= lscpIdx {
        let isParentZero = false
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getContext(prevVal: prevVal, isParentZero: isParentZero))
        prevVal = val

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

    let lscpX = Int(decoder.readPair(context: 5).run)
    let lscpY = Int(decoder.readPair(context: 5).run)
    guard lscpX < 4 && lscpY < 4 else { throw DecodeError.invalidBlockData }

    var currentIdx = 0
    var prevVal: Int16 = 0
    let lscpIdx = lscpY * 4 + lscpX
    while currentIdx <= lscpIdx {
        let isParentZero = false
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getContext(prevVal: prevVal, isParentZero: isParentZero))
        prevVal = val

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

    let lscpX = Int(decoder.readPair(context: 5).run)
    let lscpY = Int(decoder.readPair(context: 5).run)
    guard lscpX < 4 && lscpY < 4 else { throw DecodeError.invalidBlockData }

    var currentIdx = 0
    var prevVal: Int16 = 0
    let lscpIdx = lscpY * 4 + lscpX
    while currentIdx <= lscpIdx {
        let startY = currentIdx / 4
        let startX = currentIdx % 4
        let isParentZero = parentBlock.rowPointer(y: startY >> 1)[startX >> 1] == 0
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getContext(prevVal: prevVal, isParentZero: isParentZero))
        prevVal = val

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
        let lscpX = Int(decoder.readPair(context: 5).run)
        let lscpY = Int(decoder.readPair(context: 5).run)
        guard lscpX < 4 && lscpY < 4 else { throw DecodeError.invalidBlockDataContext("DPCM4 lscp out of range: (\(lscpX), \(lscpY))") }
        lscpIdx = lscpY * 4 + lscpX
    }

    var currentIdx = 0
    var prevVal: Int16 = 0
    while currentIdx <= lscpIdx {
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getDPCMContext(prevVal: prevVal))
        prevVal = val

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
        let lscpX = Int(decoder.readPair(context: 5).run)
        let lscpY = Int(decoder.readPair(context: 5).run)
        guard lscpX < 8 && lscpY < 8 else { throw DecodeError.invalidBlockDataContext("DPCM8 lscp out of range: (\(lscpX), \(lscpY))") }
        lscpIdx = lscpY * 8 + lscpX
    }

    var currentIdx = 0
    var prevVal: Int16 = 0
    while currentIdx <= lscpIdx {
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getDPCMContext(prevVal: prevVal))
        prevVal = val

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
        let lscpX = Int(decoder.readPair(context: 5).run)
        let lscpY = Int(decoder.readPair(context: 5).run)
        guard lscpX < 16 && lscpY < 16 else { throw DecodeError.invalidBlockData }
        lscpIdx = lscpY * 16 + lscpX
    }

    var currentIdx = 0
    var prevVal: Int16 = 0
    while currentIdx <= lscpIdx {
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getDPCMContext(prevVal: prevVal))
        prevVal = val

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
func decodeLayer32(r: [UInt8], pool: BlockViewPool, layer: UInt8, dx: Int, dy: Int, prev: Image16, parentYBlocks: [BlockView]?, parentCbBlocks: [BlockView]?, parentCrBlocks: [BlockView]?, predictedPd: PlaneData420? = nil, nextPd: PlaneData420? = nil, mvs: MotionVectors? = nil, refDirs: [Bool]? = nil, roundOffset: Int, skipMap: [BlockMode]? = nil) async throws -> Image16 {
    let (qtY, qtC, bufY, bufCb, bufCr) = try VEVCLayerData.deserialize(from: r, layer: layer, layerLabel: "Layer32")
    
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
    defer {
        pool.putBlockViewArray1024(yBlocks)
        pool.putBlockViewArray1024(cbBlocks)
        pool.putBlockViewArray1024(crBlocks)
    }
    
    if let sMap = skipMap {
        decodeLayer32ProcessYWithSkipMap(pool: pool, taskIdx: 0, chunkSize: rowCountY, rowCount: rowCountY, dx: dx, colCount: colCountY, blocks: yBlocks, prev: prev, qt: qtY, skipMap: sMap, sub: &sub)
    } else {
        decodeLayer32ProcessY(pool: pool, taskIdx: 0, chunkSize: rowCountY, rowCount: rowCountY, dx: dx, colCount: colCountY, blocks: yBlocks, prev: prev, qt: qtY, sub: &sub)
    }
    
    decodeLayer32ProcessCb(pool: pool, taskIdx: 0, chunkSize: rowCountCb, rowCount: rowCountCb, dx: cbDx, colCount: colCountCb, blocks: cbBlocks, prev: prev, qt: qtC, sub: &sub)
        
    decodeLayer32ProcessCr(pool: pool, taskIdx: 0, chunkSize: rowCountCr, rowCount: rowCountCr, dx: cbDx, colCount: colCountCr, blocks: crBlocks, prev: prev, qt: qtC, sub: &sub)
    
    // MC: MV is layer0 precision -> layer2 (full resolution) mvScale=4
    if let tPrev = predictedPd, let mvs = mvs {
        if let tNext = nextPd, let dirs = refDirs {
            await applyScaledBidirectionalMotionCompensationLuma(plane: &sub.y, prevPlane: tPrev.y, nextPlane: tNext.y, mvs: mvs, refDirs: dirs, skipMap: skipMap, width: dx, height: dy, lumaBlockSize: 32, mvShift: 0, roundOffset: roundOffset)
            await applyScaledBidirectionalMotionCompensationChroma(plane: &sub.cb, prevPlane: tPrev.cb, nextPlane: tNext.cb, mvs: mvs, refDirs: dirs, skipMap: skipMap, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
            await applyScaledBidirectionalMotionCompensationChroma(plane: &sub.cr, prevPlane: tPrev.cr, nextPlane: tNext.cr, mvs: mvs, refDirs: dirs, skipMap: skipMap, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
        } else {
            await applyScaledMotionCompensationLuma(plane: &sub.y, prevPlane: tPrev.y, mvs: mvs, skipMap: skipMap, width: dx, height: dy, lumaBlockSize: 32, mvShift: 0, roundOffset: roundOffset)
            await applyScaledMotionCompensationChroma(plane: &sub.cb, prevPlane: tPrev.cb, mvs: mvs, skipMap: skipMap, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
            await applyScaledMotionCompensationChroma(plane: &sub.cr, prevPlane: tPrev.cr, mvs: mvs, skipMap: skipMap, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
        }
    }

    if let mvs = mvs, mvs.isEmpty != true {
        applyDeblockingFilter32(plane: &sub.y, width: dx, height: dy, qStep: (Int(qtY.step) + 8) >> 4, mvs: mvs, skipMap: skipMap)
        applyDeblockingFilterChroma16(plane: &sub.cb, width: cbDx, height: cbDy, qStep: (Int(qtC.step) + 8) >> 4, mvs: mvs, skipMap: skipMap)
        applyDeblockingFilterChroma16(plane: &sub.cr, width: cbDx, height: cbDy, qStep: (Int(qtC.step) + 8) >> 4, mvs: mvs, skipMap: skipMap)
    } else {
        applyDeblockingFilter32(plane: &sub.y, width: dx, height: dy, qStep: (Int(qtY.step) + 8) >> 4)
        applyDeblockingFilter16(plane: &sub.cb, width: cbDx, height: cbDy, qStep: (Int(qtC.step) + 8) >> 4)
        applyDeblockingFilter16(plane: &sub.cr, width: cbDx, height: cbDy, qStep: (Int(qtC.step) + 8) >> 4)
    }
    
    return sub
}

@inline(__always)
func decodeLayer16(r: [UInt8], pool: BlockViewPool, layer: UInt8, dx: Int, dy: Int, prev: Image16, parentYBlocks: [BlockView]?, parentCbBlocks: [BlockView]?, parentCrBlocks: [BlockView]?) async throws -> (Image16, [BlockView], [BlockView], [BlockView]) {
    let (qtY, qtC, bufY, bufCb, bufCr) = try VEVCLayerData.deserialize(from: r, layer: layer, layerLabel: "Layer16")
    
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
    
    decodeLayer16ProcessY(pool: pool, taskIdx: 0, chunkSize: rowCountY, rowCount: rowCountY, dx: dx, colCount: colCountY, blocks: yBlocks, prev: prev, qt: qtY, sub: &sub)
    decodeLayer16ProcessCb(pool: pool, taskIdx: 0, chunkSize: rowCountCb, rowCount: rowCountCb, dx: cbDx, colCount: colCountCb, blocks: cbBlocks, prev: prev, qt: qtC, sub: &sub)
    decodeLayer16ProcessCr(pool: pool, taskIdx: 0, chunkSize: rowCountCr, rowCount: rowCountCr, dx: cbDx, colCount: colCountCr, blocks: crBlocks, prev: prev, qt: qtC, sub: &sub)
    
    return (sub, yBlocks, cbBlocks, crBlocks)
}

@inline(__always)
func decodeBase8(r: [UInt8], pool: BlockViewPool, layer: UInt8, dx: Int, dy: Int, isIFrame: Bool) async throws -> (Image16, [BlockView], [BlockView], [BlockView], qtYStep: Int, qtCStep: Int) {
    let (qtY, qtC, bufY, bufCb, bufCr) = try VEVCLayerData.deserialize(from: r, layer: layer, layerLabel: "Base8")
    
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
    
    decodeBase8ProcessY(pool: pool, taskIdx: 0, chunkSize: rowCountY, rowCount: rowCountY, dx: dx, colCount: colCountY, blocks: yBlocks, qt: qtY, sub: &sub)
    decodeBase8ProcessCb(pool: pool, taskIdx: 0, chunkSize: rowCountCb, rowCount: rowCountCb, dx: cbDx, colCount: colCountCb, blocks: cbBlocks, qt: qtC, sub: &sub)
    decodeBase8ProcessCr(pool: pool, taskIdx: 0, chunkSize: rowCountCr, rowCount: rowCountCr, dx: cbDx, colCount: colCountCr, blocks: crBlocks, qt: qtC, sub: &sub)
        
    return (sub, yBlocks, cbBlocks, crBlocks, qtYStep: Int(qtY.step), qtCStep: Int(qtC.step))
}

@Sendable @inline(__always)
func decodeLayer32ProcessYWithSkipMap(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable, skipMap: [BlockMode], sub: inout Image16) {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return }
    let subConst = sub
    let sCount = skipMap.count
    sub.withUnsafeY { destBase in
        for i in startRow..<endRow {
            let h: Int = i * 32
            let rowOffset = i * colCount
            for (xIdx, w) in stride(from: 0, to: dx, by: 32).enumerated() {
                let blockIndex: Int = rowOffset &+ xIdx
                let isSkip = blockIndex < sCount && skipMap[blockIndex] != .inter
                if isSkip {
                    continue
                }
                let block: BlockView = blocks[blockIndex]
                let half: Int = 16
                let view = block
                let base = view.base
                let hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 32)
                let lhView = BlockView(base: base.advanced(by: half * 32), width: half, height: half, stride: 32)
                let hhView = BlockView(base: base.advanced(by: half * 32 + half), width: half, height: half, stride: 32)
                
                prev.readY(x: w / 2, y: h / 2, size: half, into: block)
                dequantizeSIMDSignedMapping16(hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping16(lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping16(hhView, q: qt.qHigh)
                inverseDWT2DBlock32(view)
                var blk = block
                subConst.updateY(destBase: destBase, data: &blk, startX: w, startY: h, size: 32)
            }
        }
    }
}

@Sendable @inline(__always)
func decodeLayer32ProcessY(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable, sub: inout Image16) {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return }
    let subConst = sub
    sub.withUnsafeY { destBase in
        for i in startRow..<endRow {
            let h: Int = i * 32
            let rowOffset = i * colCount
            for (xIdx, w) in stride(from: 0, to: dx, by: 32).enumerated() {
                let blockIndex: Int = rowOffset &+ xIdx
                let block: BlockView = blocks[blockIndex]
                let half: Int = 16
                let view = block
                let base = view.base
                let hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 32)
                let lhView = BlockView(base: base.advanced(by: half * 32), width: half, height: half, stride: 32)
                let hhView = BlockView(base: base.advanced(by: half * 32 + half), width: half, height: half, stride: 32)
                
                prev.readY(x: w / 2, y: h / 2, size: half, into: block)
                dequantizeSIMDSignedMapping16(hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping16(lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping16(hhView, q: qt.qHigh)
                inverseDWT2DBlock32(view)
                var blk = block
                subConst.updateY(destBase: destBase, data: &blk, startX: w, startY: h, size: 32)
            }
        }
    }
}

@Sendable @inline(__always)
func decodeLayer32ProcessCb(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable, sub: inout Image16) {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return }
    let subConst = sub
    sub.withUnsafeCb { destBase in
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
                dequantizeSIMDSignedMapping16(hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping16(lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping16(hhView, q: qt.qHigh)
                inverseDWT2DBlock32(view)
                var blk = block
                subConst.updateCb(destBase: destBase, data: &blk, startX: w, startY: h, size: 32)
            }
        }
    }
}

@Sendable @inline(__always)
func decodeLayer32ProcessCr(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable, sub: inout Image16) {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return }
    let subConst = sub
    sub.withUnsafeCr { destBase in
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
                dequantizeSIMDSignedMapping16(hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping16(lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping16(hhView, q: qt.qHigh)
                inverseDWT2DBlock32(view)
                var blk = block
                subConst.updateCr(destBase: destBase, data: &blk, startX: w, startY: h, size: 32)
            }
        }
    }
}

@Sendable @inline(__always)
func decodeLayer16ProcessY(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable, sub: inout Image16) {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return }
    let subConst = sub
    sub.withUnsafeY { destBase in
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
                dequantizeSIMDSignedMapping8(hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping8(lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping8(hhView, q: qt.qHigh)
                inverseDWT2DBlock16(view)
                var blk = block
                subConst.updateY(destBase: destBase, data: &blk, startX: w, startY: h, size: 16)
            }
        }
    }
}

@Sendable @inline(__always)
func decodeLayer16ProcessCb(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable, sub: inout Image16) {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return }
    let subConst = sub
    sub.withUnsafeCb { destBase in
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
                var blk = block
                subConst.updateCb(destBase: destBase, data: &blk, startX: w, startY: h, size: 16)
            }
        }
    }
}

@Sendable @inline(__always)
func decodeLayer16ProcessCr(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable, sub: inout Image16) {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return }
    let subConst = sub
    sub.withUnsafeCr { destBase in
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
                var blk = block
                subConst.updateCr(destBase: destBase, data: &blk, startX: w, startY: h, size: 16)
            }
        }
    }
}

@Sendable @inline(__always)
func decodeBase8ProcessY(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], qt: QuantizationTable, sub: inout Image16) {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return }
    let subConst = sub
    sub.withUnsafeY { destBase in
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
                dequantizeSIMD4(llView, q: qt.qLow)
                dequantizeSIMDSignedMapping4(hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping4(lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping4(hhView, q: qt.qHigh)
                inverseDWT2DBlock8(view)
                var blk = block
                subConst.updateY(destBase: destBase, data: &blk, startX: w, startY: h, size: 8)
            }
        }
    }
}

@Sendable @inline(__always)
func decodeBase8ProcessCb(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], qt: QuantizationTable, sub: inout Image16) {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return }
    let subConst = sub
    sub.withUnsafeCb { destBase in
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
                dequantizeSIMD4(llView, q: qt.qLow)
                dequantizeSIMDSignedMapping4(hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping4(lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping4(hhView, q: qt.qHigh)
                inverseDWT2DBlock8(view)
                var blk = block
                subConst.updateCb(destBase: destBase, data: &blk, startX: w, startY: h, size: 8)
            }
        }
    }
}

@Sendable @inline(__always)
func decodeBase8ProcessCr(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], qt: QuantizationTable, sub: inout Image16) {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return }
    let subConst = sub
    sub.withUnsafeCr { destBase in
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
                dequantizeSIMD4(llView, q: qt.qLow)
                dequantizeSIMDSignedMapping4(hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping4(lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping4(hhView, q: qt.qHigh)
                inverseDWT2DBlock8(view)
                var blk = block
                subConst.updateCr(destBase: destBase, data: &blk, startX: w, startY: h, size: 8)
            }
        }
    }
}