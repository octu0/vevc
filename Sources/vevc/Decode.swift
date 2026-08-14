import Foundation

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
func blockDecode16V(decoder: inout EntropyDecoder, ptr base: UnsafeMutablePointer<Int16>, stride: Int) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        return
    }

    let lscpX = Int(decoder.readPair(context: 5).run)
    let lscpY = Int(decoder.readPair(context: 5).run)
    guard lscpX < 16 && lscpY < 16 else { throw DecodeError.invalidBlockDataContext("blockDecode16V lscp out of range: (\(lscpX), \(lscpY))") }

    var currentIdx = 0
    var prevVal: Int16 = 0
    let lscpIdx = (lscpX << 4) + lscpY
    while currentIdx <= lscpIdx {
        let isParentZero = false
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getContext(prevVal: prevVal, isParentZero: isParentZero))
        prevVal = val

        currentIdx += run
        if currentIdx <= lscpIdx {
            let x = currentIdx >> 4
            let y = currentIdx & 15
            base[y * stride + x] = val
        }
        currentIdx += 1
    }
}

@inline(__always)
func blockDecode16VWithParentBlock(decoder: inout EntropyDecoder, ptr base: UnsafeMutablePointer<Int16>, stride: Int, parentPtr: UnsafePointer<Int16>, parentStride: Int) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        return
    }

    let lscpX = Int(decoder.readPair(context: 5).run)
    let lscpY = Int(decoder.readPair(context: 5).run)
    guard lscpX < 16 && lscpY < 16 else { throw DecodeError.invalidBlockDataContext("blockDecode16VParent lscp out of range: (\(lscpX), \(lscpY))") }

    var currentIdx = 0
    var prevVal: Int16 = 0
    let lscpIdx = (lscpX << 4) + lscpY
    while currentIdx <= lscpIdx {
        let startX = currentIdx >> 4
        let startY = currentIdx & 15
        let isParentZero = parentPtr[(startY >> 1) * parentStride + (startX >> 1)] == 0
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getContext(prevVal: prevVal, isParentZero: isParentZero))
        prevVal = val

        currentIdx += run
        if currentIdx <= lscpIdx {
            let x = currentIdx >> 4
            let y = currentIdx & 15
            base[y * stride + x] = val
        }
        currentIdx += 1
    }
}

@inline(__always)
func blockDecode16H(decoder: inout EntropyDecoder, ptr base: UnsafeMutablePointer<Int16>, stride: Int) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        return
    }

    let lscpX = Int(decoder.readPair(context: 5).run)
    let lscpY = Int(decoder.readPair(context: 5).run)
    guard lscpX < 16 && lscpY < 16 else { throw DecodeError.invalidBlockDataContext("blockDecode16H lscp out of range: (\(lscpX), \(lscpY))") }

    var currentIdx = 0
    var prevVal: Int16 = 0
    let lscpIdx = (lscpY << 4) + lscpX
    while currentIdx <= lscpIdx {
        let isParentZero = false
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getContext(prevVal: prevVal, isParentZero: isParentZero))
        prevVal = val

        currentIdx += run
        if currentIdx <= lscpIdx {
            let y = currentIdx >> 4
            let x = currentIdx & 15
            base[y * stride + x] = val
        }
        currentIdx += 1
    }
}

@inline(__always)
func blockDecode16HWithParentBlock(decoder: inout EntropyDecoder, ptr base: UnsafeMutablePointer<Int16>, stride: Int, parentPtr: UnsafePointer<Int16>, parentStride: Int) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        return
    }

    let lscpX = Int(decoder.readPair(context: 5).run)
    let lscpY = Int(decoder.readPair(context: 5).run)
    guard lscpX < 16 && lscpY < 16 else { throw DecodeError.invalidBlockDataContext("blockDecode16HParent lscp out of range: (\(lscpX), \(lscpY))") }

    var currentIdx = 0
    var prevVal: Int16 = 0
    let lscpIdx = (lscpY << 4) + lscpX
    while currentIdx <= lscpIdx {
        let startY = currentIdx >> 4
        let startX = currentIdx & 15
        let isParentZero = parentPtr[(startY >> 1) * parentStride + (startX >> 1)] == 0
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getContext(prevVal: prevVal, isParentZero: isParentZero))
        prevVal = val

        currentIdx += run
        if currentIdx <= lscpIdx {
            let y = currentIdx >> 4
            let x = currentIdx & 15
            base[y * stride + x] = val
        }
        currentIdx += 1
    }
}

@inline(__always)
func blockDecode8V(decoder: inout EntropyDecoder, ptr base: UnsafeMutablePointer<Int16>, stride: Int) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        return
    }

    let lscpX = Int(decoder.readPair(context: 5).run)
    let lscpY = Int(decoder.readPair(context: 5).run)

    var currentIdx = 0
    var prevVal: Int16 = 0
    let lscpIdx = (lscpX << 3) + lscpY
    while currentIdx <= lscpIdx {
        let isParentZero = false
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getContext(prevVal: prevVal, isParentZero: isParentZero))
        prevVal = val

        currentIdx += run
        if currentIdx <= lscpIdx {
            let x = currentIdx >> 3
            let y = currentIdx & 7
            base[y * stride + x] = val
        }
        currentIdx += 1
    }
}

@inline(__always)
func blockDecode8VWithParentBlock(decoder: inout EntropyDecoder, ptr base: UnsafeMutablePointer<Int16>, stride: Int, parentPtr: UnsafePointer<Int16>, parentStride: Int) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        return
    }

    let lscpX = Int(decoder.readPair(context: 5).run)
    let lscpY = Int(decoder.readPair(context: 5).run)

    var currentIdx = 0
    var prevVal: Int16 = 0
    let lscpIdx = (lscpX << 3) + lscpY
    while currentIdx <= lscpIdx {
        let startX = currentIdx >> 3
        let startY = currentIdx & 7
        let isParentZero = parentPtr[(startY >> 1) * parentStride + (startX >> 1)] == 0
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getContext(prevVal: prevVal, isParentZero: isParentZero))
        prevVal = val

        currentIdx += run
        if currentIdx <= lscpIdx {
            let x = currentIdx >> 3
            let y = currentIdx & 7
            base[y * stride + x] = val
        }
        currentIdx += 1
    }
}


@inline(__always)
func blockDecode8H(decoder: inout EntropyDecoder, ptr base: UnsafeMutablePointer<Int16>, stride: Int) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        return
    }

    let lscpX = Int(decoder.readPair(context: 5).run)
    let lscpY = Int(decoder.readPair(context: 5).run)

    var currentIdx = 0
    var prevVal: Int16 = 0
    let lscpIdx = (lscpY << 3) + lscpX
    while currentIdx <= lscpIdx {
        let isParentZero = false
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getContext(prevVal: prevVal, isParentZero: isParentZero))
        prevVal = val

        currentIdx += run
        if currentIdx <= lscpIdx {
            let y = currentIdx >> 3
            let x = currentIdx & 7
            base[y * stride + x] = val
        }
        currentIdx += 1
    }
}


@inline(__always)
func blockDecode8HWithParentBlock(decoder: inout EntropyDecoder, ptr base: UnsafeMutablePointer<Int16>, stride: Int, parentPtr: UnsafePointer<Int16>, parentStride: Int) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        return
    }

    let lscpX = Int(decoder.readPair(context: 5).run)
    let lscpY = Int(decoder.readPair(context: 5).run)

    var currentIdx = 0
    var prevVal: Int16 = 0
    let lscpIdx = (lscpY << 3) + lscpX
    while currentIdx <= lscpIdx {
        let startY = currentIdx >> 3
        let startX = currentIdx & 7
        let isParentZero = parentPtr[(startY >> 1) * parentStride + (startX >> 1)] == 0
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getContext(prevVal: prevVal, isParentZero: isParentZero))
        prevVal = val

        currentIdx += run
        if currentIdx <= lscpIdx {
            let y = currentIdx >> 3
            let x = currentIdx & 7
            base[y * stride + x] = val
        }
        currentIdx += 1
    }
}


@inline(__always)
func blockDecode4V(decoder: inout EntropyDecoder, ptr base: UnsafeMutablePointer<Int16>, stride: Int) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        return
    }

    let lscpX = Int(decoder.readPair(context: 5).run)
    let lscpY = Int(decoder.readPair(context: 5).run)
    guard lscpX < 4 && lscpY < 4 else { throw DecodeError.invalidBlockDataContext("blockDecode4V lscp out of range: (\(lscpX), \(lscpY))") }

    var currentIdx = 0
    var prevVal: Int16 = 0
    let lscpIdx = (lscpX << 2) + lscpY
    while currentIdx <= lscpIdx {
        let isParentZero = false
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getContext(prevVal: prevVal, isParentZero: isParentZero))
        prevVal = val

        currentIdx += run
        if currentIdx <= lscpIdx {
            let x = currentIdx >> 2
            let y = currentIdx & 3
            base[y * stride + x] = val
        }
        currentIdx += 1
    }
}


@inline(__always)
func blockDecode4H(decoder: inout EntropyDecoder, ptr base: UnsafeMutablePointer<Int16>, stride: Int) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        return
    }

    let lscpX = Int(decoder.readPair(context: 5).run)
    let lscpY = Int(decoder.readPair(context: 5).run)
    guard lscpX < 4 && lscpY < 4 else { throw DecodeError.invalidBlockDataContext("blockDecode4H lscp out of range: (\(lscpX), \(lscpY))") }

    var currentIdx = 0
    var prevVal: Int16 = 0
    let lscpIdx = (lscpY << 2) + lscpX
    while currentIdx <= lscpIdx {
        let isParentZero = false
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getContext(prevVal: prevVal, isParentZero: isParentZero))
        prevVal = val

        currentIdx += run
        if currentIdx <= lscpIdx {
            let y = currentIdx >> 2
            let x = currentIdx & 3
            base[y * stride + x] = val
        }
        currentIdx += 1
    }
}


@inline(__always)
func blockDecode4HWithParentBlock(decoder: inout EntropyDecoder, ptr base: UnsafeMutablePointer<Int16>, stride: Int, parentPtr: UnsafePointer<Int16>, parentStride: Int) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        return
    }

    let lscpX = Int(decoder.readPair(context: 5).run)
    let lscpY = Int(decoder.readPair(context: 5).run)
    guard lscpX < 4 && lscpY < 4 else { throw DecodeError.invalidBlockDataContext("blockDecode4HParent lscp out of range: (\(lscpX), \(lscpY))") }

    var currentIdx = 0
    var prevVal: Int16 = 0
    let lscpIdx = (lscpY << 2) + lscpX
    while currentIdx <= lscpIdx {
        let startY = currentIdx >> 2
        let startX = currentIdx & 3
        let isParentZero = parentPtr[(startY >> 1) * parentStride + (startX >> 1)] == 0
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getContext(prevVal: prevVal, isParentZero: isParentZero))
        prevVal = val

        currentIdx += run
        if currentIdx <= lscpIdx {
            let y = currentIdx >> 2
            let x = currentIdx & 3
            base[y * stride + x] = val
        }
        currentIdx += 1
    }
}


@inline(__always)
func blockDecodeDPCM4(decoder: inout EntropyDecoder, ptr base: UnsafeMutablePointer<Int16>, stride: Int, lastVal: inout Int16) throws {
    let hasNonZero = try decoder.decodeBypass()
    var lscpIdx = -1
    if hasNonZero == 1 {
        let lscpX = Int(decoder.readPair(context: 5).run)
        let lscpY = Int(decoder.readPair(context: 5).run)
        guard lscpX < 4 && lscpY < 4 else { throw DecodeError.invalidBlockDataContext("DPCM4 lscp out of range: (\(lscpX), \(lscpY))") }
        lscpIdx = (lscpY << 2) + lscpX
    }

    var currentIdx = 0
    var prevVal: Int16 = 0
    while currentIdx <= lscpIdx {
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getDPCMContext(prevVal: prevVal))
        prevVal = val

        currentIdx += run
        if currentIdx <= lscpIdx {
            let y = currentIdx >> 2
            let x = currentIdx & 3
            base[y * stride + x] = val
        }
        currentIdx += 1
    }

    let ptr1 = base + stride
    let ptr2 = base + 2 * stride
    let ptr3 = base + 3 * stride

    base[0] = base[0] &+ lastVal
    base[1] = base[1] &+ base[0]
    base[2] = base[2] &+ base[1]
    base[3] = base[3] &+ base[2]

    ptr1[0] = ptr1[0] &+ base[0]
    ptr1[1] = ptr1[1] &+ predictMED(ptr1[0], base[1], base[0])
    ptr1[2] = ptr1[2] &+ predictMED(ptr1[1], base[2], base[1])
    ptr1[3] = ptr1[3] &+ predictMED(ptr1[2], base[3], base[2])

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
func blockDecodeDPCM8(decoder: inout EntropyDecoder, ptr base: UnsafeMutablePointer<Int16>, stride: Int, lastVal: inout Int16) throws {
    let hasNonZero = try decoder.decodeBypass()
    var lscpIdx = -1
    if hasNonZero == 1 {
        let lscpX = Int(decoder.readPair(context: 5).run)
        let lscpY = Int(decoder.readPair(context: 5).run)
        guard lscpX < 8 && lscpY < 8 else { throw DecodeError.invalidBlockDataContext("DPCM8 lscp out of range: (\(lscpX), \(lscpY))") }
        lscpIdx = (lscpY << 3) + lscpX
    }

    var currentIdx = 0
    var prevVal: Int16 = 0
    while currentIdx <= lscpIdx {
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getDPCMContext(prevVal: prevVal))
        prevVal = val

        currentIdx += run
        if currentIdx <= lscpIdx {
            let y = currentIdx >> 3
            let x = currentIdx & 7
            base[y * stride + x] = val
        }
        currentIdx += 1
    }

    base[0] = base[0] &+ lastVal
    for x in 1..<8 {
        base[x] = base[x] &+ base[x - 1]
    }
    
    var last = base[7]
    for y in 1..<8 {
        let ptrY = base + y * stride
        let ptrPrevY = base + (y - 1) * stride
        
        ptrY[0] = ptrY[0] &+ ptrPrevY[0]
        for x in 1..<8 {
            ptrY[x] = ptrY[x] &+ predictMED(ptrY[x - 1], ptrPrevY[x], ptrPrevY[x - 1])
        }
        last = ptrY[7]
    }
    lastVal = last
}


@inline(__always)
func blockDecodeDPCM16(decoder: inout EntropyDecoder, ptr base: UnsafeMutablePointer<Int16>, stride: Int, lastVal: inout Int16) throws {
    let hasNonZero = try decoder.decodeBypass()
    var lscpIdx = -1
    if hasNonZero == 1 {
        let lscpX = Int(decoder.readPair(context: 5).run)
        let lscpY = Int(decoder.readPair(context: 5).run)
        guard lscpX < 16 && lscpY < 16 else { throw DecodeError.invalidBlockData }
        lscpIdx = (lscpY << 4) + lscpX
    }

    var currentIdx = 0
    var prevVal: Int16 = 0
    while currentIdx <= lscpIdx {
        let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getDPCMContext(prevVal: prevVal))
        prevVal = val

        currentIdx += run
        if currentIdx <= lscpIdx {
            let y = currentIdx >> 4
            let x = currentIdx & 15
            base[y * stride + x] = val
        }
        currentIdx += 1
    }

    base[0] = base[0] &+ lastVal
    for x in 1..<16 {
        base[x] = base[x] &+ base[x - 1]
    }
    
    var last = base[15]
    for y in 1..<16 {
        let ptrY = base + y * stride
        let ptrPrevY = base + (y - 1) * stride
        
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
func decodeLayer32(r: [UInt8], pool: BlockViewPool, layer: UInt8, dx: Int, dy: Int, prev: Image16, parentYBlocks: [BlockView]?, parentCbBlocks: [BlockView]?, parentCrBlocks: [BlockView]?, predictedPd: PlaneData420? = nil, nextPd: PlaneData420? = nil, mvs: MotionVectors? = nil, refDirs: [Bool]? = nil, roundOffset: Int, skipMap: [BlockMode]? = nil, histories: [EntropyHistoryState]? = nil, parentFreeStatics: Bool = false) async throws -> Image16 {
    let (qtY, qtC, bufY, bufCb, bufCr) = try VEVCLayerData.deserialize(from: r, layer: layer, layerLabel: "Layer32")
    
    var sub = Image16(width: dx, height: dy, pool: pool)
    
    let rowCountY = (dy + 32 - 1) / 32
    let colCountY = (dx + 32 - 1) / 32
    
    let yBlocks: [BlockView]
    if let p = parentYBlocks {
        yBlocks = try decodePlaneSubbands32WithParentBlocks(data: bufY, pool: pool, blockCount: rowCountY * colCountY, parentBlocks: p, history: histories?[0], parentFreeStatics: parentFreeStatics)
    } else {
        yBlocks = try decodePlaneSubbands32(data: bufY, pool: pool, blockCount: rowCountY * colCountY, history: histories?[0], parentFreeStatics: parentFreeStatics)
    }
    
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let rowCountCb = (cbDy + 32 - 1) / 32
    let colCountCb = (cbDx + 32 - 1) / 32
    let cbBlocks: [BlockView]
    if let p = parentCbBlocks {
        cbBlocks = try decodePlaneSubbands32WithParentBlocks(data: bufCb, pool: pool, blockCount: rowCountCb * colCountCb, parentBlocks: p, history: histories?[1], parentFreeStatics: parentFreeStatics)
    } else {
        cbBlocks = try decodePlaneSubbands32(data: bufCb, pool: pool, blockCount: rowCountCb * colCountCb, history: histories?[1], parentFreeStatics: parentFreeStatics)
    }
    
    let rowCountCr = (cbDy + 32 - 1) / 32
    let colCountCr = (cbDx + 32 - 1) / 32
    let crBlocks: [BlockView]
    if let p = parentCrBlocks {
        crBlocks = try decodePlaneSubbands32WithParentBlocks(data: bufCr, pool: pool, blockCount: rowCountCr * colCountCr, parentBlocks: p, history: histories?[2], parentFreeStatics: parentFreeStatics)
    } else {
        crBlocks = try decodePlaneSubbands32(data: bufCr, pool: pool, blockCount: rowCountCr * colCountCr, history: histories?[2], parentFreeStatics: parentFreeStatics)
    }
    defer {
        pool.putBlockViewArray1024(yBlocks)
        pool.putBlockViewArray1024(cbBlocks)
        pool.putBlockViewArray1024(crBlocks)
    }
    
    if let sMap = skipMap {
        await decodeLayer32ProcessYWithSkipMap(pool: pool, taskIdx: 0, chunkSize: rowCountY, rowCount: rowCountY, dx: dx, colCount: colCountY, blocks: yBlocks, prev: prev, qt: qtY, skipMap: sMap, sub: &sub)
        await decodeLayer32ProcessCbWithSkipMap(pool: pool, taskIdx: 0, chunkSize: rowCountCb, rowCount: rowCountCb, dx: cbDx, colCount: colCountCb, blocks: cbBlocks, prev: prev, qt: qtC, skipMap: sMap, sub: &sub)
        await decodeLayer32ProcessCrWithSkipMap(pool: pool, taskIdx: 0, chunkSize: rowCountCr, rowCount: rowCountCr, dx: cbDx, colCount: colCountCr, blocks: crBlocks, prev: prev, qt: qtC, skipMap: sMap, sub: &sub)
    } else {
        await decodeLayer32ProcessY(pool: pool, taskIdx: 0, chunkSize: rowCountY, rowCount: rowCountY, dx: dx, colCount: colCountY, blocks: yBlocks, prev: prev, qt: qtY, sub: &sub)
        await decodeLayer32ProcessCb(pool: pool, taskIdx: 0, chunkSize: rowCountCb, rowCount: rowCountCb, dx: cbDx, colCount: colCountCb, blocks: cbBlocks, prev: prev, qt: qtC, sub: &sub)
        await decodeLayer32ProcessCr(pool: pool, taskIdx: 0, chunkSize: rowCountCr, rowCount: rowCountCr, dx: cbDx, colCount: colCountCr, blocks: crBlocks, prev: prev, qt: qtC, sub: &sub)
    }
    
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
func decodeLayer16(r: [UInt8], pool: BlockViewPool, layer: UInt8, dx: Int, dy: Int, prev: Image16, parentYBlocks: [BlockView]?, parentCbBlocks: [BlockView]?, parentCrBlocks: [BlockView]?, histories: [EntropyHistoryState]? = nil, parentFreeStatics: Bool = false) async throws -> (Image16, [BlockView], [BlockView], [BlockView]) {
    let (qtY, qtC, bufY, bufCb, bufCr) = try VEVCLayerData.deserialize(from: r, layer: layer, layerLabel: "Layer16")
    
    var sub = Image16(width: dx, height: dy, pool: pool, zeroed: false)
    
    let rowCountY = (dy + 16 - 1) / 16
    let colCountY = (dx + 16 - 1) / 16
    let yBlocks: [BlockView]
    if let p = parentYBlocks {
        yBlocks = try decodePlaneSubbands16WithParentBlocks(data: bufY, pool: pool, blockCount: rowCountY * colCountY, parentBlocks: p, history: histories?[0], parentFreeStatics: parentFreeStatics)
    } else {
        yBlocks = try decodePlaneSubbands16(data: bufY, pool: pool, blockCount: rowCountY * colCountY, history: histories?[0], parentFreeStatics: parentFreeStatics)
    }
    
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let rowCountCb = (cbDy + 16 - 1) / 16
    let colCountCb = (cbDx + 16 - 1) / 16
    let cbBlocks: [BlockView]
    if let p = parentCbBlocks {
        cbBlocks = try decodePlaneSubbands16WithParentBlocks(data: bufCb, pool: pool, blockCount: rowCountCb * colCountCb, parentBlocks: p, history: histories?[1], parentFreeStatics: parentFreeStatics)
    } else {
        cbBlocks = try decodePlaneSubbands16(data: bufCb, pool: pool, blockCount: rowCountCb * colCountCb, history: histories?[1], parentFreeStatics: parentFreeStatics)
    }
    
    let rowCountCr = (cbDy + 16 - 1) / 16
    let colCountCr = (cbDx + 16 - 1) / 16
    let crBlocks: [BlockView]
    if let p = parentCrBlocks {
        crBlocks = try decodePlaneSubbands16WithParentBlocks(data: bufCr, pool: pool, blockCount: rowCountCr * colCountCr, parentBlocks: p, history: histories?[2], parentFreeStatics: parentFreeStatics)
    } else {
        crBlocks = try decodePlaneSubbands16(data: bufCr, pool: pool, blockCount: rowCountCr * colCountCr, history: histories?[2], parentFreeStatics: parentFreeStatics)
    }
    
    await decodeLayer16ProcessY(pool: pool, taskIdx: 0, chunkSize: rowCountY, rowCount: rowCountY, dx: dx, colCount: colCountY, blocks: yBlocks, prev: prev, qt: qtY, sub: &sub)
    await decodeLayer16ProcessCb(pool: pool, taskIdx: 0, chunkSize: rowCountCb, rowCount: rowCountCb, dx: cbDx, colCount: colCountCb, blocks: cbBlocks, prev: prev, qt: qtC, sub: &sub)
    await decodeLayer16ProcessCr(pool: pool, taskIdx: 0, chunkSize: rowCountCr, rowCount: rowCountCr, dx: cbDx, colCount: colCountCr, blocks: crBlocks, prev: prev, qt: qtC, sub: &sub)
    
    return (sub, yBlocks, cbBlocks, crBlocks)
}

@inline(__always)
func decodeBase8(r: [UInt8], pool: BlockViewPool, layer: UInt8, dx: Int, dy: Int, isIFrame: Bool, histories: [EntropyHistoryState]? = nil, parentFreeStatics: Bool = false) async throws -> (Image16, [BlockView], [BlockView], [BlockView], qtYStep: Int, qtCStep: Int) {
    let (qtY, qtC, bufY, bufCb, bufCr) = try VEVCLayerData.deserialize(from: r, layer: layer, layerLabel: "Base8")
    
    var sub = Image16(width: dx, height: dy, pool: pool, zeroed: false)
    
    let rowCountY = (dy + 8 - 1) / 8
    let colCountY = (dx + 8 - 1) / 8
    let yBlocks = try decodePlaneBaseSubbands8(data: bufY, pool: pool, blockCount: rowCountY * colCountY, isIFrame: isIFrame, history: histories?[0], parentFreeStatics: parentFreeStatics)
    
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let rowCountCb = (cbDy + 8 - 1) / 8
    let colCountCb = (cbDx + 8 - 1) / 8
    let cbBlocks = try decodePlaneBaseSubbands8(data: bufCb, pool: pool, blockCount: rowCountCb * colCountCb, isIFrame: isIFrame, history: histories?[1], parentFreeStatics: parentFreeStatics)
    
    let rowCountCr = (cbDy + 8 - 1) / 8
    let colCountCr = (cbDx + 8 - 1) / 8
    let crBlocks = try decodePlaneBaseSubbands8(data: bufCr, pool: pool, blockCount: rowCountCr * colCountCr, isIFrame: isIFrame, history: histories?[2], parentFreeStatics: parentFreeStatics)
    
    await decodeBase8ProcessY(pool: pool, taskIdx: 0, chunkSize: rowCountY, rowCount: rowCountY, dx: dx, colCount: colCountY, blocks: yBlocks, qt: qtY, sub: &sub)
    await decodeBase8ProcessCb(pool: pool, taskIdx: 0, chunkSize: rowCountCb, rowCount: rowCountCb, dx: cbDx, colCount: colCountCb, blocks: cbBlocks, qt: qtC, sub: &sub)
    await decodeBase8ProcessCr(pool: pool, taskIdx: 0, chunkSize: rowCountCr, rowCount: rowCountCr, dx: cbDx, colCount: colCountCr, blocks: crBlocks, qt: qtC, sub: &sub)
        
    return (sub, yBlocks, cbBlocks, crBlocks, qtYStep: Int(qtY.step), qtCStep: Int(qtC.step))
}

@Sendable @inline(__always)
func decodeLayer32ProcessYWithSkipMap(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable, skipMap: [BlockMode], sub: inout Image16) async {
    let concurrency = min(rowCount, 4)
    let chunkSizeSlice = (rowCount + concurrency - 1) / concurrency
    let subConst = sub
    let sCount = skipMap.count
    let sSkip = skipMap.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let pWidth = prev.width
    let pHeight = prev.height
    let sWidth = sub.width
    let sHeight = sub.height
    
    let sPrev = prev.withUnsafeYReadOnly { UnsafeSendablePointer(ptr: $0) }
    let sDest = sub.withUnsafeY { UnsafeSendableMutablePointer(ptr: $0) }

    // Structured concurrency instead of DispatchQueue.concurrentPerform:
    // blocking dispatch_apply on the Swift Concurrency cooperative pool can
    // trip libdispatch's deadlock detection (EXC_BREAKPOINT in _dlock_wait)
    // when many GOPs decode in parallel.
    let sBlocks = blocks.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let work: @Sendable (Int) -> Void = { tIdx in
                let startRow: Int = tIdx * chunkSizeSlice
                let endRow: Int = min(startRow + chunkSizeSlice, rowCount)
                guard startRow < endRow else { return }
                let prevPtr = sPrev.ptr
                let destPtr = sDest.ptr
                for i in startRow..<endRow {
                    let h: Int = i * 32
                    let rowOffset = i * colCount
                    let py = h / 2
                    for (xIdx, w) in stride(from: 0, to: dx, by: 32).enumerated() {
                        let blockIndex: Int = rowOffset &+ xIdx
                        let isSkip = blockIndex < sCount && sSkip.ptr[blockIndex] != .inter
                        if isSkip {
                            continue
                        }
                        let block: BlockView = sBlocks.ptr[blockIndex]
                        let base = block.base
                        let px = w / 2
                        
                        if 0 <= px && 0 <= py && px + 16 <= pWidth && py + 16 <= pHeight && 0 <= w && 0 <= h && w + 32 <= sWidth && h + 32 <= sHeight {
                            copy16x16ContiguousDirect(srcBase: prevPtr, srcWidth: pWidth, x: px, y: py, dstBase: base, dstStride: 32)
                            dequantize16(ptr: base.advanced(by: 16), stride: 32, q: qt.qMid)
                            dequantize16(ptr: base.advanced(by: 512), stride: 32, q: qt.qMid)
                            dequantize16(ptr: base.advanced(by: 528), stride: 32, q: qt.qHigh)
                            inverseDWT2DBlock32(ptr: base, stride: 32)
                            copy32x32ContiguousDirect(srcBase: base, srcStride: 32, destBase: destPtr, destWidth: sWidth, x: w, y: h)
                        } else {
                            prev.readYDirect(srcBase: prevPtr, x: px, y: py, size: 16, into: block)
                            dequantize16(ptr: base.advanced(by: 16), stride: 32, q: qt.qMid)
                            dequantize16(ptr: base.advanced(by: 512), stride: 32, q: qt.qMid)
                            dequantize16(ptr: base.advanced(by: 528), stride: 32, q: qt.qHigh)
                            inverseDWT2DBlock32(ptr: base, stride: 32)
                            var blk = block
                            subConst.updateY(destBase: destPtr, data: &blk, startX: w, startY: h, size: 32)
                        }
                    }
                }
    }
    // Only the dominant full-res luma pass benefits from intra-plane fan-out;
    // for everything smaller the task-spawn overhead outweighs the gain
    // (and under GOP-parallel decode all cores are busy anyway). Large passes
    // run chunk 0 inline and spawn only the remainder.
    if rowCount * colCount < 1500 {
        for tIdx in 0..<concurrency {
            work(tIdx)
        }
        return
    }
    await withTaskGroup(of: Void.self) { group in
        for tIdx in 1..<concurrency {
            group.addTask { work(tIdx) }
        }
        work(0)
    }
}

@inline(__always)
private func copy16x16ContiguousDirect(srcBase: UnsafePointer<Int16>, srcWidth: Int, x: Int, y: Int, dstBase: UnsafeMutablePointer<Int16>, dstStride: Int) {
    let srcStart = srcBase.advanced(by: y * srcWidth + x)
    for h in 0..<16 {
        let srcPtr = srcStart.advanced(by: h * srcWidth)
        let dstPtr = dstBase.advanced(by: h * dstStride)
        let s0 = UnsafeRawPointer(srcPtr).loadUnaligned(as: SIMD16<Int16>.self)
        UnsafeMutableRawPointer(dstPtr).storeBytes(of: s0, as: SIMD16<Int16>.self)
    }
}

@inline(__always)
private func copy32x32ContiguousDirect(srcBase: UnsafePointer<Int16>, srcStride: Int, destBase: UnsafeMutablePointer<Int16>, destWidth: Int, x: Int, y: Int) {
    let destStart = destBase.advanced(by: y * destWidth + x)
    for h in 0..<32 {
        let srcPtr = srcBase.advanced(by: h * srcStride)
        let destPtr = destStart.advanced(by: h * destWidth)
        let s0 = UnsafeRawPointer(srcPtr).loadUnaligned(as: SIMD16<Int16>.self)
        let s1 = UnsafeRawPointer(srcPtr.advanced(by: 16)).loadUnaligned(as: SIMD16<Int16>.self)
        UnsafeMutableRawPointer(destPtr).storeBytes(of: s0, as: SIMD16<Int16>.self)
        UnsafeMutableRawPointer(destPtr.advanced(by: 16)).storeBytes(of: s1, as: SIMD16<Int16>.self)
    }
}

@Sendable @inline(__always)
func decodeLayer32ProcessY(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable, sub: inout Image16) async {
    let concurrency = min(rowCount, 4)
    let chunkSizeSlice = (rowCount + concurrency - 1) / concurrency
    let subConst = sub
    let pWidth = prev.width
    let pHeight = prev.height
    let sWidth = sub.width
    let sHeight = sub.height
    
    let sPrev = prev.withUnsafeYReadOnly { UnsafeSendablePointer(ptr: $0) }
    let sDest = sub.withUnsafeY { UnsafeSendableMutablePointer(ptr: $0) }

    // Structured concurrency instead of DispatchQueue.concurrentPerform:
    // blocking dispatch_apply on the Swift Concurrency cooperative pool can
    // trip libdispatch's deadlock detection (EXC_BREAKPOINT in _dlock_wait)
    // when many GOPs decode in parallel.
    let sBlocks = blocks.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let work: @Sendable (Int) -> Void = { tIdx in
                let startRow: Int = tIdx * chunkSizeSlice
                let endRow: Int = min(startRow + chunkSizeSlice, rowCount)
                guard startRow < endRow else { return }
                let prevPtr = sPrev.ptr
                let destPtr = sDest.ptr
                for i in startRow..<endRow {
                    let h: Int = i * 32
                    let rowOffset = i * colCount
                    let py = h / 2
                    for (xIdx, w) in stride(from: 0, to: dx, by: 32).enumerated() {
                        let blockIndex: Int = rowOffset &+ xIdx
                        let block: BlockView = sBlocks.ptr[blockIndex]
                        let base = block.base
                        let px = w / 2
                        
                        if 0 <= px && 0 <= py && px + 16 <= pWidth && py + 16 <= pHeight && 0 <= w && 0 <= h && w + 32 <= sWidth && h + 32 <= sHeight {
                            copy16x16ContiguousDirect(srcBase: prevPtr, srcWidth: pWidth, x: px, y: py, dstBase: base, dstStride: 32)
                            dequantize16(ptr: base.advanced(by: 16), stride: 32, q: qt.qMid)
                            dequantize16(ptr: base.advanced(by: 512), stride: 32, q: qt.qMid)
                            dequantize16(ptr: base.advanced(by: 528), stride: 32, q: qt.qHigh)
                            inverseDWT2DBlock32(ptr: base, stride: 32)
                            copy32x32ContiguousDirect(srcBase: base, srcStride: 32, destBase: destPtr, destWidth: sWidth, x: w, y: h)
                        } else {
                            prev.readYDirect(srcBase: prevPtr, x: px, y: py, size: 16, into: block)
                            dequantize16(ptr: base.advanced(by: 16), stride: 32, q: qt.qMid)
                            dequantize16(ptr: base.advanced(by: 512), stride: 32, q: qt.qMid)
                            dequantize16(ptr: base.advanced(by: 528), stride: 32, q: qt.qHigh)
                            inverseDWT2DBlock32(ptr: base, stride: 32)
                            var blk = block
                            subConst.updateY(destBase: destPtr, data: &blk, startX: w, startY: h, size: 32)
                        }
                    }
                }
    }
    // Only the dominant full-res luma pass benefits from intra-plane fan-out;
    // for everything smaller the task-spawn overhead outweighs the gain
    // (and under GOP-parallel decode all cores are busy anyway). Large passes
    // run chunk 0 inline and spawn only the remainder.
    if rowCount * colCount < 1500 {
        for tIdx in 0..<concurrency {
            work(tIdx)
        }
        return
    }
    await withTaskGroup(of: Void.self) { group in
        for tIdx in 1..<concurrency {
            group.addTask { work(tIdx) }
        }
        work(0)
    }
}

@Sendable @inline(__always)
func decodeLayer32ProcessCbWithSkipMap(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable, skipMap: [BlockMode], sub: inout Image16) async {
    let concurrency = min(rowCount, 4)
    let chunkSizeSlice = (rowCount + concurrency - 1) / concurrency
    let subConst = sub
    let sCount = skipMap.count
    let sSkip = skipMap.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let sPrev = prev.withUnsafeCbReadOnly { UnsafeSendablePointer(ptr: $0) }
    let sDest = sub.withUnsafeCb { UnsafeSendableMutablePointer(ptr: $0) }

    // Structured concurrency instead of DispatchQueue.concurrentPerform (see ProcessY note).
    let sBlocks = blocks.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let work: @Sendable (Int) -> Void = { tIdx in
                let startRow: Int = tIdx * chunkSizeSlice
                let endRow: Int = min(startRow + chunkSizeSlice, rowCount)
                guard startRow < endRow else { return }
                let prevPtr = sPrev.ptr
                let destPtr = sDest.ptr
                for i in startRow..<endRow {
                    let h: Int = i * 32
                    let rowOffset = i * colCount
                    for (xIdx, w) in stride(from: 0, to: dx, by: 32).enumerated() {
                        let blockIndex: Int = rowOffset &+ xIdx
                        if blockIndex < sCount && sSkip.ptr[blockIndex] != .inter {
                            continue
                        }
                        let block: BlockView = sBlocks.ptr[blockIndex]
                        let base = block.base
                        prev.readCbDirect(srcBase: prevPtr, x: w / 2, y: h / 2, size: 16, into: block)
                        dequantize16(ptr: base.advanced(by: 16), stride: 32, q: qt.qMid)
                        dequantize16(ptr: base.advanced(by: 512), stride: 32, q: qt.qMid)
                        dequantize16(ptr: base.advanced(by: 528), stride: 32, q: qt.qHigh)
                        inverseDWT2DBlock32(ptr: base, stride: 32)
                        var blk = block
                        subConst.updateCb(destBase: destPtr, data: &blk, startX: w, startY: h, size: 32)
                    }
                }
    }
    // Only the dominant full-res luma pass benefits from intra-plane fan-out;
    // for everything smaller the task-spawn overhead outweighs the gain
    // (and under GOP-parallel decode all cores are busy anyway). Large passes
    // run chunk 0 inline and spawn only the remainder.
    if rowCount * colCount < 1500 {
        for tIdx in 0..<concurrency {
            work(tIdx)
        }
        return
    }
    await withTaskGroup(of: Void.self) { group in
        for tIdx in 1..<concurrency {
            group.addTask { work(tIdx) }
        }
        work(0)
    }
}

@Sendable @inline(__always)
func decodeLayer32ProcessCb(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable, sub: inout Image16) async {
    let concurrency = min(rowCount, 4)
    let chunkSizeSlice = (rowCount + concurrency - 1) / concurrency
    let subConst = sub
    let sPrev = prev.withUnsafeCbReadOnly { UnsafeSendablePointer(ptr: $0) }
    let sDest = sub.withUnsafeCb { UnsafeSendableMutablePointer(ptr: $0) }

    // Structured concurrency instead of DispatchQueue.concurrentPerform (see ProcessY note).
    let sBlocks = blocks.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let work: @Sendable (Int) -> Void = { tIdx in
                let startRow: Int = tIdx * chunkSizeSlice
                let endRow: Int = min(startRow + chunkSizeSlice, rowCount)
                guard startRow < endRow else { return }
                let prevPtr = sPrev.ptr
                let destPtr = sDest.ptr
                for i in startRow..<endRow {
                    let h: Int = i * 32
                    let rowOffset = i * colCount
                    for (xIdx, w) in stride(from: 0, to: dx, by: 32).enumerated() {
                        let blockIndex: Int = rowOffset &+ xIdx
                        let block: BlockView = sBlocks.ptr[blockIndex]
                        let base = block.base
                        prev.readCbDirect(srcBase: prevPtr, x: w / 2, y: h / 2, size: 16, into: block)
                        dequantize16(ptr: base.advanced(by: 16), stride: 32, q: qt.qMid)
                        dequantize16(ptr: base.advanced(by: 512), stride: 32, q: qt.qMid)
                        dequantize16(ptr: base.advanced(by: 528), stride: 32, q: qt.qHigh)
                        inverseDWT2DBlock32(ptr: base, stride: 32)
                        var blk = block
                        subConst.updateCb(destBase: destPtr, data: &blk, startX: w, startY: h, size: 32)
                    }
                }
    }
    // Only the dominant full-res luma pass benefits from intra-plane fan-out;
    // for everything smaller the task-spawn overhead outweighs the gain
    // (and under GOP-parallel decode all cores are busy anyway). Large passes
    // run chunk 0 inline and spawn only the remainder.
    if rowCount * colCount < 1500 {
        for tIdx in 0..<concurrency {
            work(tIdx)
        }
        return
    }
    await withTaskGroup(of: Void.self) { group in
        for tIdx in 1..<concurrency {
            group.addTask { work(tIdx) }
        }
        work(0)
    }
}

@Sendable @inline(__always)
func decodeLayer32ProcessCrWithSkipMap(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable, skipMap: [BlockMode], sub: inout Image16) async {
    let concurrency = min(rowCount, 4)
    let chunkSizeSlice = (rowCount + concurrency - 1) / concurrency
    let subConst = sub
    let sCount = skipMap.count
    let sSkip = skipMap.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let sPrev = prev.withUnsafeCrReadOnly { UnsafeSendablePointer(ptr: $0) }
    let sDest = sub.withUnsafeCr { UnsafeSendableMutablePointer(ptr: $0) }

    // Structured concurrency instead of DispatchQueue.concurrentPerform (see ProcessY note).
    let sBlocks = blocks.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let work: @Sendable (Int) -> Void = { tIdx in
                let startRow: Int = tIdx * chunkSizeSlice
                let endRow: Int = min(startRow + chunkSizeSlice, rowCount)
                guard startRow < endRow else { return }
                let prevPtr = sPrev.ptr
                let destPtr = sDest.ptr
                for i in startRow..<endRow {
                    let h: Int = i * 32
                    let rowOffset = i * colCount
                    for (xIdx, w) in stride(from: 0, to: dx, by: 32).enumerated() {
                        let blockIndex: Int = rowOffset &+ xIdx
                        if blockIndex < sCount && sSkip.ptr[blockIndex] != .inter {
                            continue
                        }
                        let block: BlockView = sBlocks.ptr[blockIndex]
                        let base = block.base
                        prev.readCrDirect(srcBase: prevPtr, x: w / 2, y: h / 2, size: 16, into: block)
                        dequantize16(ptr: base.advanced(by: 16), stride: 32, q: qt.qMid)
                        dequantize16(ptr: base.advanced(by: 512), stride: 32, q: qt.qMid)
                        dequantize16(ptr: base.advanced(by: 528), stride: 32, q: qt.qHigh)
                        inverseDWT2DBlock32(ptr: base, stride: 32)
                        var blk = block
                        subConst.updateCr(destBase: destPtr, data: &blk, startX: w, startY: h, size: 32)
                    }
                }
    }
    // Only the dominant full-res luma pass benefits from intra-plane fan-out;
    // for everything smaller the task-spawn overhead outweighs the gain
    // (and under GOP-parallel decode all cores are busy anyway). Large passes
    // run chunk 0 inline and spawn only the remainder.
    if rowCount * colCount < 1500 {
        for tIdx in 0..<concurrency {
            work(tIdx)
        }
        return
    }
    await withTaskGroup(of: Void.self) { group in
        for tIdx in 1..<concurrency {
            group.addTask { work(tIdx) }
        }
        work(0)
    }
}

@Sendable @inline(__always)
func decodeLayer32ProcessCr(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable, sub: inout Image16) async {
    let concurrency = min(rowCount, 4)
    let chunkSizeSlice = (rowCount + concurrency - 1) / concurrency
    let subConst = sub
    let sPrev = prev.withUnsafeCrReadOnly { UnsafeSendablePointer(ptr: $0) }
    let sDest = sub.withUnsafeCr { UnsafeSendableMutablePointer(ptr: $0) }

    // Structured concurrency instead of DispatchQueue.concurrentPerform (see ProcessY note).
    let sBlocks = blocks.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let work: @Sendable (Int) -> Void = { tIdx in
                let startRow: Int = tIdx * chunkSizeSlice
                let endRow: Int = min(startRow + chunkSizeSlice, rowCount)
                guard startRow < endRow else { return }
                let prevPtr = sPrev.ptr
                let destPtr = sDest.ptr
                for i in startRow..<endRow {
                    let h: Int = i * 32
                    let rowOffset = i * colCount
                    for (xIdx, w) in stride(from: 0, to: dx, by: 32).enumerated() {
                        let blockIndex: Int = rowOffset &+ xIdx
                        let block: BlockView = sBlocks.ptr[blockIndex]
                        let base = block.base
                        prev.readCrDirect(srcBase: prevPtr, x: w / 2, y: h / 2, size: 16, into: block)
                        dequantize16(ptr: base.advanced(by: 16), stride: 32, q: qt.qMid)
                        dequantize16(ptr: base.advanced(by: 512), stride: 32, q: qt.qMid)
                        dequantize16(ptr: base.advanced(by: 528), stride: 32, q: qt.qHigh)
                        inverseDWT2DBlock32(ptr: base, stride: 32)
                        var blk = block
                        subConst.updateCr(destBase: destPtr, data: &blk, startX: w, startY: h, size: 32)
                    }
                }
    }
    // Only the dominant full-res luma pass benefits from intra-plane fan-out;
    // for everything smaller the task-spawn overhead outweighs the gain
    // (and under GOP-parallel decode all cores are busy anyway). Large passes
    // run chunk 0 inline and spawn only the remainder.
    if rowCount * colCount < 1500 {
        for tIdx in 0..<concurrency {
            work(tIdx)
        }
        return
    }
    await withTaskGroup(of: Void.self) { group in
        for tIdx in 1..<concurrency {
            group.addTask { work(tIdx) }
        }
        work(0)
    }
}

@Sendable @inline(__always)
func decodeLayer16ProcessY(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable, sub: inout Image16) async {
    let concurrency = min(rowCount, 4)
    let chunkSizeSlice = (rowCount + concurrency - 1) / concurrency
    let subConst = sub
    let sPrev = prev.withUnsafeYReadOnly { UnsafeSendablePointer(ptr: $0) }
    let sDest = sub.withUnsafeY { UnsafeSendableMutablePointer(ptr: $0) }

    // Structured concurrency instead of DispatchQueue.concurrentPerform:
    // blocking dispatch_apply on the Swift Concurrency cooperative pool can
    // trip libdispatch's deadlock detection (EXC_BREAKPOINT in _dlock_wait)
    // when many GOPs decode in parallel.
    let sBlocks = blocks.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let work: @Sendable (Int) -> Void = { tIdx in
                let startRow: Int = tIdx * chunkSizeSlice
                let endRow: Int = min(startRow + chunkSizeSlice, rowCount)
                guard startRow < endRow else { return }
                let prevPtr = sPrev.ptr
                let destPtr = sDest.ptr
                for i in startRow..<endRow {
                    let h: Int = i * 16
                    let rowOffset = i * colCount
                    for xIdx in 0..<colCount {
                        let w = xIdx * 16
                        let blockIndex: Int = rowOffset &+ xIdx
                        let block: BlockView = sBlocks.ptr[blockIndex]
                        let base = block.base
                        prev.readYDirect(srcBase: prevPtr, x: w / 2, y: h / 2, size: 8, into: block)
                        dequantize8(ptr: base.advanced(by: 8), stride: 16, q: qt.qMid)
                        dequantize8(ptr: base.advanced(by: 128), stride: 16, q: qt.qMid)
                        dequantize8(ptr: base.advanced(by: 136), stride: 16, q: qt.qHigh)
                        inverseDWT2DBlock16(ptr: base, stride: 16)
                        var blk = block
                        subConst.updateY(destBase: destPtr, data: &blk, startX: w, startY: h, size: 16)
                    }
                }
    }
    // Only the dominant full-res luma pass benefits from intra-plane fan-out;
    // for everything smaller the task-spawn overhead outweighs the gain
    // (and under GOP-parallel decode all cores are busy anyway). Large passes
    // run chunk 0 inline and spawn only the remainder.
    if rowCount * colCount < 1500 {
        for tIdx in 0..<concurrency {
            work(tIdx)
        }
        return
    }
    await withTaskGroup(of: Void.self) { group in
        for tIdx in 1..<concurrency {
            group.addTask { work(tIdx) }
        }
        work(0)
    }
}

@Sendable @inline(__always)
func decodeLayer16ProcessCb(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable, sub: inout Image16) async {
    let concurrency = min(rowCount, 4)
    let chunkSizeSlice = (rowCount + concurrency - 1) / concurrency
    let subConst = sub
    let sPrev = prev.withUnsafeCbReadOnly { UnsafeSendablePointer(ptr: $0) }
    let sDest = sub.withUnsafeCb { UnsafeSendableMutablePointer(ptr: $0) }

    // Structured concurrency instead of DispatchQueue.concurrentPerform (see ProcessY note).
    let sBlocks = blocks.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let work: @Sendable (Int) -> Void = { tIdx in
                let startRow: Int = tIdx * chunkSizeSlice
                let endRow: Int = min(startRow + chunkSizeSlice, rowCount)
                guard startRow < endRow else { return }
                let prevPtr = sPrev.ptr
                let destPtr = sDest.ptr
                for i in startRow..<endRow {
                    let h: Int = i * 16
                    let rowOffset = i * colCount
                    for xIdx in 0..<colCount {
                        let w = xIdx * 16
                        let blockIndex: Int = rowOffset &+ xIdx
                        let block: BlockView = sBlocks.ptr[blockIndex]
                        let base = block.base
                        prev.readCbDirect(srcBase: prevPtr, x: w / 2, y: h / 2, size: 8, into: block)
                        dequantize8(ptr: base.advanced(by: 8), stride: 16, q: qt.qMid)
                        dequantize8(ptr: base.advanced(by: 128), stride: 16, q: qt.qMid)
                        dequantize8(ptr: base.advanced(by: 136), stride: 16, q: qt.qHigh)
                        inverseDWT2DBlock16(ptr: base, stride: 16)
                        var blk = block
                        subConst.updateCb(destBase: destPtr, data: &blk, startX: w, startY: h, size: 16)
                    }
                }
    }
    // Only the dominant full-res luma pass benefits from intra-plane fan-out;
    // for everything smaller the task-spawn overhead outweighs the gain
    // (and under GOP-parallel decode all cores are busy anyway). Large passes
    // run chunk 0 inline and spawn only the remainder.
    if rowCount * colCount < 1500 {
        for tIdx in 0..<concurrency {
            work(tIdx)
        }
        return
    }
    await withTaskGroup(of: Void.self) { group in
        for tIdx in 1..<concurrency {
            group.addTask { work(tIdx) }
        }
        work(0)
    }
}

@Sendable @inline(__always)
func decodeLayer16ProcessCr(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], prev: Image16, qt: QuantizationTable, sub: inout Image16) async {
    let concurrency = min(rowCount, 4)
    let chunkSizeSlice = (rowCount + concurrency - 1) / concurrency
    let subConst = sub
    let sPrev = prev.withUnsafeCrReadOnly { UnsafeSendablePointer(ptr: $0) }
    let sDest = sub.withUnsafeCr { UnsafeSendableMutablePointer(ptr: $0) }

    // Structured concurrency instead of DispatchQueue.concurrentPerform (see ProcessY note).
    let sBlocks = blocks.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let work: @Sendable (Int) -> Void = { tIdx in
                let startRow: Int = tIdx * chunkSizeSlice
                let endRow: Int = min(startRow + chunkSizeSlice, rowCount)
                guard startRow < endRow else { return }
                let prevPtr = sPrev.ptr
                let destPtr = sDest.ptr
                for i in startRow..<endRow {
                    let h: Int = i * 16
                    let rowOffset = i * colCount
                    for xIdx in 0..<colCount {
                        let w = xIdx * 16
                        let blockIndex: Int = rowOffset &+ xIdx
                        let block: BlockView = sBlocks.ptr[blockIndex]
                        let base = block.base
                        prev.readCrDirect(srcBase: prevPtr, x: w / 2, y: h / 2, size: 8, into: block)
                        dequantize8(ptr: base.advanced(by: 8), stride: 16, q: qt.qMid)
                        dequantize8(ptr: base.advanced(by: 128), stride: 16, q: qt.qMid)
                        dequantize8(ptr: base.advanced(by: 136), stride: 16, q: qt.qHigh)
                        inverseDWT2DBlock16(ptr: base, stride: 16)
                        var blk = block
                        subConst.updateCr(destBase: destPtr, data: &blk, startX: w, startY: h, size: 16)
                    }
                }
    }
    // Only the dominant full-res luma pass benefits from intra-plane fan-out;
    // for everything smaller the task-spawn overhead outweighs the gain
    // (and under GOP-parallel decode all cores are busy anyway). Large passes
    // run chunk 0 inline and spawn only the remainder.
    if rowCount * colCount < 1500 {
        for tIdx in 0..<concurrency {
            work(tIdx)
        }
        return
    }
    await withTaskGroup(of: Void.self) { group in
        for tIdx in 1..<concurrency {
            group.addTask { work(tIdx) }
        }
        work(0)
    }
}

@Sendable @inline(__always)
func decodeBase8ProcessY(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], qt: QuantizationTable, sub: inout Image16) async {
    let concurrency = min(rowCount, 4)
    let chunkSizeSlice = (rowCount + concurrency - 1) / concurrency
    let subConst = sub
    let sDest = sub.withUnsafeY { UnsafeSendableMutablePointer(ptr: $0) }

    // Structured concurrency instead of DispatchQueue.concurrentPerform (see ProcessY note).
    let sBlocks = blocks.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let work: @Sendable (Int) -> Void = { tIdx in
                let startRow: Int = tIdx * chunkSizeSlice
                let endRow: Int = min(startRow + chunkSizeSlice, rowCount)
                guard startRow < endRow else { return }
                let destPtr = sDest.ptr
                for i in startRow..<endRow {
                    let h: Int = i * 8
                    let rowOffset = i * colCount
                    for xIdx in 0..<colCount {
                        let w = xIdx * 8
                        let blockIndex: Int = rowOffset &+ xIdx
                        let block: BlockView = sBlocks.ptr[blockIndex]
                        let base = block.base
                        dequantizeDPCM(ptr: base, stride: 8, q: qt.qLow)
                        dequantize4(ptr: base.advanced(by: 4), stride: 8, q: qt.qMid)
                        dequantize4(ptr: base.advanced(by: 32), stride: 8, q: qt.qMid)
                        dequantize4(ptr: base.advanced(by: 36), stride: 8, q: qt.qHigh)
                        inverseDWT2DBlock8(ptr: base, stride: 8)
                        var blk = block
                        subConst.updateY(destBase: destPtr, data: &blk, startX: w, startY: h, size: 8)
                    }
                }
    }
    // Only the dominant full-res luma pass benefits from intra-plane fan-out;
    // for everything smaller the task-spawn overhead outweighs the gain
    // (and under GOP-parallel decode all cores are busy anyway). Large passes
    // run chunk 0 inline and spawn only the remainder.
    if rowCount * colCount < 1500 {
        for tIdx in 0..<concurrency {
            work(tIdx)
        }
        return
    }
    await withTaskGroup(of: Void.self) { group in
        for tIdx in 1..<concurrency {
            group.addTask { work(tIdx) }
        }
        work(0)
    }
}

@Sendable @inline(__always)
func decodeBase8ProcessCb(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], qt: QuantizationTable, sub: inout Image16) async {
    let concurrency = min(rowCount, 4)
    let chunkSizeSlice = (rowCount + concurrency - 1) / concurrency
    let subConst = sub
    let sDest = sub.withUnsafeCb { UnsafeSendableMutablePointer(ptr: $0) }

    // Structured concurrency instead of DispatchQueue.concurrentPerform (see ProcessY note).
    let sBlocks = blocks.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let work: @Sendable (Int) -> Void = { tIdx in
                let startRow: Int = tIdx * chunkSizeSlice
                let endRow: Int = min(startRow + chunkSizeSlice, rowCount)
                guard startRow < endRow else { return }
                let destPtr = sDest.ptr
                for i in startRow..<endRow {
                    let h: Int = i * 8
                    let rowOffset = i * colCount
                    for xIdx in 0..<colCount {
                        let w = xIdx * 8
                        let blockIndex: Int = rowOffset &+ xIdx
                        let block: BlockView = sBlocks.ptr[blockIndex]
                        let base = block.base
                        dequantizeDPCM(ptr: base, stride: 8, q: qt.qLow)
                        dequantize4(ptr: base.advanced(by: 4), stride: 8, q: qt.qMid)
                        dequantize4(ptr: base.advanced(by: 32), stride: 8, q: qt.qMid)
                        dequantize4(ptr: base.advanced(by: 36), stride: 8, q: qt.qHigh)
                        inverseDWT2DBlock8(ptr: base, stride: 8)
                        var blk = block
                        subConst.updateCb(destBase: destPtr, data: &blk, startX: w, startY: h, size: 8)
                    }
                }
    }
    // Only the dominant full-res luma pass benefits from intra-plane fan-out;
    // for everything smaller the task-spawn overhead outweighs the gain
    // (and under GOP-parallel decode all cores are busy anyway). Large passes
    // run chunk 0 inline and spawn only the remainder.
    if rowCount * colCount < 1500 {
        for tIdx in 0..<concurrency {
            work(tIdx)
        }
        return
    }
    await withTaskGroup(of: Void.self) { group in
        for tIdx in 1..<concurrency {
            group.addTask { work(tIdx) }
        }
        work(0)
    }
}

@Sendable @inline(__always)
func decodeBase8ProcessCr(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], qt: QuantizationTable, sub: inout Image16) async {
    let concurrency = min(rowCount, 4)
    let chunkSizeSlice = (rowCount + concurrency - 1) / concurrency
    let subConst = sub
    let sDest = sub.withUnsafeCr { UnsafeSendableMutablePointer(ptr: $0) }

    // Structured concurrency instead of DispatchQueue.concurrentPerform (see ProcessY note).
    let sBlocks = blocks.withUnsafeBufferPointer { UnsafeSendableBufferPointer(ptr: $0) }
    let work: @Sendable (Int) -> Void = { tIdx in
                let startRow: Int = tIdx * chunkSizeSlice
                let endRow: Int = min(startRow + chunkSizeSlice, rowCount)
                guard startRow < endRow else { return }
                let destPtr = sDest.ptr
                for i in startRow..<endRow {
                    let h: Int = i * 8
                    let rowOffset = i * colCount
                    for xIdx in 0..<colCount {
                        let w = xIdx * 8
                        let blockIndex: Int = rowOffset &+ xIdx
                        let block: BlockView = sBlocks.ptr[blockIndex]
                        let base = block.base
                        dequantizeDPCM(ptr: base, stride: 8, q: qt.qLow)
                        dequantize4(ptr: base.advanced(by: 4), stride: 8, q: qt.qMid)
                        dequantize4(ptr: base.advanced(by: 32), stride: 8, q: qt.qMid)
                        dequantize4(ptr: base.advanced(by: 36), stride: 8, q: qt.qHigh)
                        inverseDWT2DBlock8(ptr: base, stride: 8)
                        var blk = block
                        subConst.updateCr(destBase: destPtr, data: &blk, startX: w, startY: h, size: 8)
                    }
                }
    }
    // Only the dominant full-res luma pass benefits from intra-plane fan-out;
    // for everything smaller the task-spawn overhead outweighs the gain
    // (and under GOP-parallel decode all cores are busy anyway). Large passes
    // run chunk 0 inline and spawn only the remainder.
    if rowCount * colCount < 1500 {
        for tIdx in 0..<concurrency {
            work(tIdx)
        }
        return
    }
    await withTaskGroup(of: Void.self) { group in
        for tIdx in 1..<concurrency {
            group.addTask { work(tIdx) }
        }
        work(0)
    }
}
