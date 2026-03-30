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

@inline(__always)
func decodeSpatialLayers(r: [UInt8], maxLayer: Int, dx: Int, dy: Int, predictedPd: PlaneData420? = nil) async throws -> Image16 {
    var offset = 0

    // Compute per-layer dimensions matching encoder DWT subband sizes:
    // Layer2 (32x32): original size
    // Layer1 (16x16): DWT LL subband of Layer2 = (dx+1)/2 × (dy+1)/2
    // Layer0 (Base8): DWT LL subband of Layer1 = ((dx+1)/2+1)/2 × ((dy+1)/2+1)/2
    let l2dx = dx
    let l2dy = dy
    let l1dx = (dx + 1) / 2
    let l1dy = (dy + 1) / 2
    let l0dx = (l1dx + 1) / 2
    let l0dy = (l1dy + 1) / 2

    // encodeSpatialLayers appends layer0, then layer1, then layer2.
    var mvs: [MotionVector]? = nil
    let mvCount = Int(try readUInt32BEFromBytes(r, offset: &offset))
    let mvDataLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    
    if mvCount > 0 && mvDataLen > 0 {
        guard (offset + mvDataLen) <= r.count else { throw DecodeError.insufficientData }
        
        mvs = try decodeMVs(data: Array(r[offset..<(offset + mvDataLen)]), count: mvCount)
        offset += mvDataLen
    }
    
    let len0 = try readUInt32BEFromBytes(r, offset: &offset)
    guard (offset + Int(len0)) <= r.count else { throw DecodeError.insufficientData }
    let layer0Data = Array(r[offset..<(offset + Int(len0))])
    offset += Int(len0)
    
    // Base layer (layer 0) is always Base8
    let (baseImg, base8YBlocks, base8CbBlocks, base8CrBlocks) = try await decodeBase8(r: layer0Data, layer: 0, dx: l0dx, dy: l0dy, isIFrame: (mvCount == 0))
    var current = baseImg
    var parentYBlocks: [Block2D]? = base8YBlocks
    var parentCbBlocks: [Block2D]? = base8CbBlocks
    var parentCrBlocks: [Block2D]? = base8CrBlocks
    
    if 1 <= maxLayer {
        let len1 = try readUInt32BEFromBytes(r, offset: &offset)
        guard (offset + Int(len1)) <= r.count else { throw DecodeError.insufficientData }
        let layer1Data = Array(r[offset..<(offset + Int(len1))])
        offset += Int(len1)
        
        let (l16Img, l16YBlocks, l16CbBlocks, l16CrBlocks) = try await decodeLayer16(r: layer1Data, layer: 1, dx: l1dx, dy: l1dy, prev: current, parentYBlocks: parentYBlocks, parentCbBlocks: parentCbBlocks, parentCrBlocks: parentCrBlocks)
        current = l16Img
        parentYBlocks = l16YBlocks
        parentCbBlocks = l16CbBlocks
        parentCrBlocks = l16CrBlocks
    }
    
    if 2 <= maxLayer {
        let len2 = try readUInt32BEFromBytes(r, offset: &offset)
        guard (offset + Int(len2)) <= r.count else { throw DecodeError.insufficientData }
        let layer2Data = Array(r[offset..<(offset + Int(len2))])
        offset += Int(len2)
        
        current = try await decodeLayer32(r: layer2Data, layer: 2, dx: l2dx, dy: l2dy, prev: current, parentYBlocks: parentYBlocks, parentCbBlocks: parentCbBlocks, parentCrBlocks: parentCrBlocks, predictedPd: predictedPd, mvs: mvs)
    }
    
    return current
}

// MARK: - Decode Logic

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

// getContextIdx is defined in Encode.swift and available module-wide.

@inline(__always)
func decodeCoeffRun(decoder: inout EntropyDecoder, contextIdx: Int = 0) throws -> (Int, Int16) {
    let pair = decoder.readPair(contextIdx: contextIdx)
    return (pair.run, pair.val)
}

@inline(__always)
func blockDecode32(decoder: inout EntropyDecoder, block: inout BlockView, parentBlock: BlockView?) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        block.clearAll()
        return
    }

    let lsci = Int(try decoder.decodeLSCI())
    guard lsci < 1024 else { throw DecodeError.invalidBlockData }

    block.clearAll()

    var currentIdx = 0
    let isParentZero = isBlockAllZero(block: parentBlock)

    while currentIdx <= lsci {
        let ctx = getContextIdx(idx: currentIdx, isParentZero: isParentZero)
        let (run, val) = try decodeCoeffRun(decoder: &decoder, contextIdx: ctx)

        currentIdx += run
        if currentIdx <= lsci {
            let (x, y) = ZOrder.coords32[currentIdx]
            let ptr = block.rowPointer(y: y)
            ptr[x] = val
        }
        currentIdx += 1
    }
}

@inline(__always)
func blockDecode16(decoder: inout EntropyDecoder, block: inout BlockView, parentBlock: BlockView?) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        block.clearAll()
        return
    }

    let lsci = Int(try decoder.decodeLSCI())
    guard lsci < 256 else { throw DecodeError.invalidBlockData }

    block.clearAll()

    var currentIdx = 0
    let isParentZero = isBlockAllZero(block: parentBlock)

    while currentIdx <= lsci {
        let ctx = getContextIdx(idx: currentIdx, isParentZero: isParentZero)
        let (run, val) = try decodeCoeffRun(decoder: &decoder, contextIdx: ctx)

        currentIdx += run
        if currentIdx <= lsci {
            let (x, y) = ZOrder.coords16[currentIdx]
            let ptr = block.rowPointer(y: y)
            ptr[x] = val
        }
        currentIdx += 1
    }
}

@inline(__always)
func blockDecode8(decoder: inout EntropyDecoder, block: inout BlockView, parentBlock: BlockView?) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        block.clearAll()
        return
    }

    let lsci = Int(try decoder.decodeLSCI())
    guard lsci < 64 else { throw DecodeError.invalidBlockData }

    block.clearAll()

    var currentIdx = 0
    let isParentZero = isBlockAllZero(block: parentBlock)

    while currentIdx <= lsci {
        let ctx = getContextIdx(idx: currentIdx, isParentZero: isParentZero)
        let (run, val) = try decodeCoeffRun(decoder: &decoder, contextIdx: ctx)

        currentIdx += run
        if currentIdx <= lsci {
            let (x, y) = ZOrder.coords8[currentIdx]
            let ptr = block.rowPointer(y: y)
            ptr[x] = val
        }
        currentIdx += 1
    }
}

@inline(__always)
func blockDecode4(decoder: inout EntropyDecoder, block: inout BlockView, parentBlock: BlockView?) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        block.clearAll()
        return
    }

    let lsci = Int(try decoder.decodeLSCI())
    guard lsci < 16 else { throw DecodeError.invalidBlockData }

    block.clearAll()

    var currentIdx = 0
    let isParentZero = isBlockAllZero(block: parentBlock)

    while currentIdx <= lsci {
        let ctx = getContextIdx(idx: currentIdx, isParentZero: isParentZero)
        let (run, val) = try decodeCoeffRun(decoder: &decoder, contextIdx: ctx)

        currentIdx += run
        if currentIdx <= lsci {
            let (x, y) = ZOrder.coords4[currentIdx]
            let ptr = block.rowPointer(y: y)
            ptr[x] = val
        }
        currentIdx += 1
    }
}

@inline(__always)
func blockDecodeDPCM4(decoder: inout EntropyDecoder, block: inout BlockView, lastVal: inout Int16) throws {
    let hasNonZero = try decoder.decodeBypass()
    var lscpIdx = -1
    if hasNonZero == 1 {
        let lscpX = Int(try decoder.decodeLSCI())
        let lscpY = Int(try decoder.decodeLSCI())
        guard lscpX < 4 && lscpY < 4 else { throw DecodeError.invalidBlockDataContext("DPCM4 lscp out of range: (\(lscpX), \(lscpY))") }
        lscpIdx = lscpY * 4 + lscpX
    }

    block.clearAll()

    var currentIdx = 0
    
    while currentIdx <= lscpIdx {
        let (run, val) = try decodeCoeffRun(decoder: &decoder)

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
func blockDecodeDPCM8(decoder: inout EntropyDecoder, block: inout BlockView, lastVal: inout Int16) throws {
    let hasNonZero = try decoder.decodeBypass()
    var lscpIdx = -1
    if hasNonZero == 1 {
        let lscpX = Int(try decoder.decodeLSCI())
        let lscpY = Int(try decoder.decodeLSCI())
        guard lscpX < 8 && lscpY < 8 else { throw DecodeError.invalidBlockDataContext("DPCM8 lscp out of range: (\(lscpX), \(lscpY))") }
        lscpIdx = lscpY * 8 + lscpX
    }

    block.clearAll()

    var currentIdx = 0
    
    while currentIdx <= lscpIdx {
        let (run, val) = try decodeCoeffRun(decoder: &decoder)

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
func blockDecodeDPCM16(decoder: inout EntropyDecoder, block: inout BlockView, lastVal: inout Int16) throws {
    let hasNonZero = try decoder.decodeBypass()
    var lscpIdx = -1
    if hasNonZero == 1 {
        let lscpX = Int(try decoder.decodeLSCI())
        let lscpY = Int(try decoder.decodeLSCI())
        guard lscpX < 16 && lscpY < 16 else { throw DecodeError.invalidBlockData }
        lscpIdx = lscpY * 16 + lscpX
    }

    block.clearAll()

    var currentIdx = 0
    
    while currentIdx <= lscpIdx {
        let (run, val) = try decodeCoeffRun(decoder: &decoder)

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
func decodeLayer32(r: [UInt8], layer: UInt8, dx: Int, dy: Int, prev: Image16, parentYBlocks: [Block2D]?, parentCbBlocks: [Block2D]?, parentCrBlocks: [Block2D]?, predictedPd: PlaneData420? = nil, mvs: [MotionVector]? = nil) async throws -> Image16 {
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
    
    var sub = Image16(width: dx, height: dy)
    
    let rowCountY = (dy + 32 - 1) / 32
    let colCountY = (dx + 32 - 1) / 32
    let yBlocks = try decodePlaneSubbands32(data: bufY, blockCount: rowCountY * colCountY, parentBlocks: parentYBlocks)
    
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let rowCountCb = (cbDy + 32 - 1) / 32
    let colCountCb = (cbDx + 32 - 1) / 32
    let cbBlocks = try decodePlaneSubbands32(data: bufCb, blockCount: rowCountCb * colCountCb, parentBlocks: parentCbBlocks)
    
    let rowCountCr = (cbDy + 32 - 1) / 32
    let colCountCr = (cbDx + 32 - 1) / 32
    let crBlocks = try decodePlaneSubbands32(data: bufCr, blockCount: rowCountCr * colCountCr, parentBlocks: parentCrBlocks)
    
    let chunkSize = 4
    let taskCountY = (rowCountY + chunkSize - 1) / chunkSize

    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountY {
            group.addTask { return decodeLayer32ProcessY(taskIdx: taskIdx, chunkSize: chunkSize, rowCount: rowCountY, dx: dx, colCount: colCountY, blocks: yBlocks, prev: prev, qt: qtY) }
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

    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountCb {
            group.addTask { return decodeLayer32ProcessCb(taskIdx: taskIdx, chunkSize: chunkSize, rowCount: rowCountCb, dx: cbDx, colCount: colCountCb, blocks: cbBlocks, prev: prev, qt: qtC) }
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

    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountCr {
            group.addTask { return decodeLayer32ProcessCr(taskIdx: taskIdx, chunkSize: chunkSize, rowCount: rowCountCr, dx: cbDx, colCount: colCountCr, blocks: crBlocks, prev: prev, qt: qtC) }
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
        applyMotionCompensationPixels(plane: &sub.y, prevPlane: tPrev.y, mvs: mvs, width: dx, height: dy, blockSize: 32, shiftMultiplierX2: 8)
        applyMotionCompensationPixels(plane: &sub.cb, prevPlane: tPrev.cb, mvs: mvs, width: cbDx, height: cbDy, blockSize: 16, shiftMultiplierX2: 4)
        applyMotionCompensationPixels(plane: &sub.cr, prevPlane: tPrev.cr, mvs: mvs, width: cbDx, height: cbDy, blockSize: 16, shiftMultiplierX2: 4)
    }

    applyDeblockingFilter(plane: &sub.y, width: dx, height: dy, blockSize: 32, qStep: Int(qtY.step))
    applyDeblockingFilter(plane: &sub.cb, width: cbDx, height: cbDy, blockSize: 16, qStep: Int(qtC.step))
    applyDeblockingFilter(plane: &sub.cr, width: cbDx, height: cbDy, blockSize: 16, qStep: Int(qtC.step))
    return sub
}

@inline(__always)
func decodeLayer16(r: [UInt8], layer: UInt8, dx: Int, dy: Int, prev: Image16, parentYBlocks: [Block2D]?, parentCbBlocks: [Block2D]?, parentCrBlocks: [Block2D]?) async throws -> (Image16, [Block2D], [Block2D], [Block2D]) {
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
    
    var sub = Image16(width: dx, height: dy)
    
    let rowCountY = (dy + 16 - 1) / 16
    let colCountY = (dx + 16 - 1) / 16
    let yBlocks = try decodePlaneSubbands16(data: bufY, blockCount: rowCountY * colCountY, parentBlocks: parentYBlocks)
    
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let rowCountCb = (cbDy + 16 - 1) / 16
    let colCountCb = (cbDx + 16 - 1) / 16
    let cbBlocks = try decodePlaneSubbands16(data: bufCb, blockCount: rowCountCb * colCountCb, parentBlocks: parentCbBlocks)
    
    let rowCountCr = (cbDy + 16 - 1) / 16
    let colCountCr = (cbDx + 16 - 1) / 16
    let crBlocks = try decodePlaneSubbands16(data: bufCr, blockCount: rowCountCr * colCountCr, parentBlocks: parentCrBlocks)
    
    let chunkSize = 4
    let taskCountY = (rowCountY + chunkSize - 1) / chunkSize
    let taskCountCb = (rowCountCb + chunkSize - 1) / chunkSize
    let taskCountCr = (rowCountCr + chunkSize - 1) / chunkSize

    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountY {
            group.addTask { return decodeLayer16ProcessY(taskIdx: taskIdx, chunkSize: chunkSize, rowCount: rowCountY, dx: dx, colCount: colCountY, blocks: yBlocks, prev: prev, qt: qtY) }
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

    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountCb {
            group.addTask { return decodeLayer16ProcessCb(taskIdx: taskIdx, chunkSize: chunkSize, rowCount: rowCountCb, dx: cbDx, colCount: colCountCb, blocks: cbBlocks, prev: prev, qt: qtC) }
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
    
    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountCr {
            group.addTask { return decodeLayer16ProcessCr(taskIdx: taskIdx, chunkSize: chunkSize, rowCount: rowCountCr, dx: cbDx, colCount: colCountCr, blocks: crBlocks, prev: prev, qt: qtC) }
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
func decodeBase8(r: [UInt8], layer: UInt8, dx: Int, dy: Int, isIFrame: Bool) async throws -> (Image16, [Block2D], [Block2D], [Block2D]) {
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
    
    var sub = Image16(width: dx, height: dy)
    
    let rowCountY = (dy + 8 - 1) / 8
    let colCountY = (dx + 8 - 1) / 8
    var tmpYBlocks = try decodePlaneBaseSubbands8(data: bufY, blockCount: rowCountY * colCountY)
    
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let rowCountCb = (cbDy + 8 - 1) / 8
    let colCountCb = (cbDx + 8 - 1) / 8
    var tmpCbBlocks = try decodePlaneBaseSubbands8(data: bufCb, blockCount: rowCountCb * colCountCb)
    
    let rowCountCr = (cbDy + 8 - 1) / 8
    let colCountCr = (cbDx + 8 - 1) / 8
    var tmpCrBlocks = try decodePlaneBaseSubbands8(data: bufCr, blockCount: rowCountCr * colCountCr)
    
    if isIFrame {
        var predDCY: Int16 = 0
        for i in tmpYBlocks.indices {
            let diff = tmpYBlocks[i].data[0]
            let qDC = diff + predDCY
            tmpYBlocks[i].data[0] = qDC
            predDCY = qDC
        }
        var predDCCb: Int16 = 0
        for i in tmpCbBlocks.indices {
            let diff = tmpCbBlocks[i].data[0]
            let qDC = diff + predDCCb
            tmpCbBlocks[i].data[0] = qDC
            predDCCb = qDC
        }
        var predDCCr: Int16 = 0
        for i in tmpCrBlocks.indices {
            let diff = tmpCrBlocks[i].data[0]
            let qDC = diff + predDCCr
            tmpCrBlocks[i].data[0] = qDC
            predDCCr = qDC
        }
    }
    
    let yBlocks = tmpYBlocks
    let cbBlocks = tmpCbBlocks
    let crBlocks = tmpCrBlocks
    
    let chunkSize = 4
    let taskCountY = (rowCountY + chunkSize - 1) / chunkSize
    let taskCountCb = (rowCountCb + chunkSize - 1) / chunkSize
    let taskCountCr = (rowCountCr + chunkSize - 1) / chunkSize

    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountY {
            group.addTask { return decodeBase8ProcessY(taskIdx: taskIdx, chunkSize: chunkSize, rowCount: rowCountY, dx: dx, colCount: colCountY, blocks: yBlocks, qt: qtY) }
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

    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountCb {
            group.addTask { return decodeBase8ProcessCb(taskIdx: taskIdx, chunkSize: chunkSize, rowCount: rowCountCb, dx: cbDx, colCount: colCountCb, blocks: cbBlocks, qt: qtC) }
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
    
    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountCr {
            group.addTask { return decodeBase8ProcessCr(taskIdx: taskIdx, chunkSize: chunkSize, rowCount: rowCountCr, dx: cbDx, colCount: colCountCr, blocks: crBlocks, qt: qtC) }
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
func decodeLayer32ProcessY(taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [Block2D], prev: Image16, qt: QuantizationTable) -> [(Block2D, Int, Int)] {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return [] }
    var rowResults: [(Block2D, Int, Int)] = []
    for i in startRow..<endRow {
        let h: Int = i * 32
        for (xIdx, w) in stride(from: 0, to: dx, by: 32).enumerated() {
            let blockIndex: Int = i * colCount + xIdx
            var block: Block2D = blocks[blockIndex]
            let half: Int = 32 / 2
            var ll: Block2D = prev.getY(x: w / 2, y: h / 2, size: half)
            ll.withView { srcView in
                block.withView { destView in
                    for yi in 0..<half {
                        let srcPtr = srcView.rowPointer(y: yi)
                        let destPtr = destView.rowPointer(y: yi)
                        destPtr.update(from: srcPtr, count: half)
                    }
                }
            }
            block.withView { view in
                let base = view.base
                var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 32)
                var lhView = BlockView(base: base.advanced(by: half * 32), width: half, height: half, stride: 32)
                var hhView = BlockView(base: base.advanced(by: half * 32 + half), width: half, height: half, stride: 32)
                dequantizeSIMDSignedMapping(&hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping(&lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping(&hhView, q: qt.qHigh)
                invDwt2d_32(&view)
            }
            rowResults.append((block, w, h))
        }
    }
    return rowResults
}

@Sendable @inline(__always)
func decodeLayer32ProcessCb(taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [Block2D], prev: Image16, qt: QuantizationTable) -> [(Block2D, Int, Int)] {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return [] }
    var rowResults: [(Block2D, Int, Int)] = []
    for i in startRow..<endRow {
        let h: Int = i * 32
        for (xIdx, w) in stride(from: 0, to: dx, by: 32).enumerated() {
            let blockIndex: Int = i * colCount + xIdx
            var block: Block2D = blocks[blockIndex]
            let half: Int = 32 / 2
            var ll: Block2D = prev.getCb(x: w / 2, y: h / 2, size: half)
            ll.withView { srcView in
                block.withView { destView in
                    for yi in 0..<half {
                        let srcPtr = srcView.rowPointer(y: yi)
                        let destPtr = destView.rowPointer(y: yi)
                        destPtr.update(from: srcPtr, count: half)
                    }
                }
            }
            block.withView { view in
                let base = view.base
                var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 32)
                var lhView = BlockView(base: base.advanced(by: half * 32), width: half, height: half, stride: 32)
                var hhView = BlockView(base: base.advanced(by: half * 32 + half), width: half, height: half, stride: 32)
                dequantizeSIMDSignedMapping(&hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping(&lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping(&hhView, q: qt.qHigh)
                invDwt2d_32(&view)
            }
            rowResults.append((block, w, h))
        }
    }
    return rowResults
}

@Sendable @inline(__always)
func decodeLayer32ProcessCr(taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [Block2D], prev: Image16, qt: QuantizationTable) -> [(Block2D, Int, Int)] {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return [] }
    var rowResults: [(Block2D, Int, Int)] = []
    for i in startRow..<endRow {
        let h: Int = i * 32
        for (xIdx, w) in stride(from: 0, to: dx, by: 32).enumerated() {
            let blockIndex: Int = i * colCount + xIdx
            var block: Block2D = blocks[blockIndex]
            let half: Int = 32 / 2
            var ll: Block2D = prev.getCr(x: w / 2, y: h / 2, size: half)
            ll.withView { srcView in
                block.withView { destView in
                    for yi in 0..<half {
                        let srcPtr = srcView.rowPointer(y: yi)
                        let destPtr = destView.rowPointer(y: yi)
                        destPtr.update(from: srcPtr, count: half)
                    }
                }
            }
            block.withView { view in
                let base = view.base
                var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 32)
                var lhView = BlockView(base: base.advanced(by: half * 32), width: half, height: half, stride: 32)
                var hhView = BlockView(base: base.advanced(by: half * 32 + half), width: half, height: half, stride: 32)
                dequantizeSIMDSignedMapping(&hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping(&lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping(&hhView, q: qt.qHigh)
                invDwt2d_32(&view)
            }
            rowResults.append((block, w, h))
        }
    }
    return rowResults
}

@Sendable @inline(__always)
func decodeLayer16ProcessY(taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [Block2D], prev: Image16, qt: QuantizationTable) -> [(Block2D, Int, Int)] {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return [] }
    var rowResults: [(Block2D, Int, Int)] = []
    for i in startRow..<endRow {
        let h: Int = i * 16
        for (xIdx, w) in stride(from: 0, to: dx, by: 16).enumerated() {
            let blockIndex: Int = i * colCount + xIdx
            var block: Block2D = blocks[blockIndex]
            let half: Int = 16 / 2
            var ll: Block2D = prev.getY(x: w / 2, y: h / 2, size: half)
            ll.withView { srcView in
                block.withView { destView in
                    for yi in 0..<half {
                        let srcPtr = srcView.rowPointer(y: yi)
                        let destPtr = destView.rowPointer(y: yi)
                        destPtr.update(from: srcPtr, count: half)
                    }
                }
            }
            block.withView { view in
                let base = view.base
                var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 16)
                var lhView = BlockView(base: base.advanced(by: half * 16), width: half, height: half, stride: 16)
                var hhView = BlockView(base: base.advanced(by: half * 16 + half), width: half, height: half, stride: 16)
                dequantizeSIMDSignedMapping(&hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping(&lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping(&hhView, q: qt.qHigh)
                invDwt2d_16(&view)
            }
            rowResults.append((block, w, h))
        }
    }
    return rowResults
}

@Sendable @inline(__always)
func decodeLayer16ProcessCb(taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [Block2D], prev: Image16, qt: QuantizationTable) -> [(Block2D, Int, Int)] {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return [] }
    var rowResults: [(Block2D, Int, Int)] = []
    for i in startRow..<endRow {
        let h: Int = i * 16
        for (xIdx, w) in stride(from: 0, to: dx, by: 16).enumerated() {
            let blockIndex: Int = i * colCount + xIdx
            var block: Block2D = blocks[blockIndex]
            let half: Int = 16 / 2
            var ll: Block2D = prev.getCb(x: w / 2, y: h / 2, size: half)
            ll.withView { srcView in
                block.withView { destView in
                    for yi in 0..<half {
                        let srcPtr = srcView.rowPointer(y: yi)
                        let destPtr = destView.rowPointer(y: yi)
                        destPtr.update(from: srcPtr, count: half)
                    }
                }
            }
            block.withView { view in
                let base = view.base
                var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 16)
                var lhView = BlockView(base: base.advanced(by: half * 16), width: half, height: half, stride: 16)
                var hhView = BlockView(base: base.advanced(by: half * 16 + half), width: half, height: half, stride: 16)
                dequantizeSIMDSignedMapping(&hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping(&lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping(&hhView, q: qt.qHigh)
                invDwt2d_16(&view)
            }
            rowResults.append((block, w, h))
        }
    }
    return rowResults
}

@Sendable @inline(__always)
func decodeLayer16ProcessCr(taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [Block2D], prev: Image16, qt: QuantizationTable) -> [(Block2D, Int, Int)] {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return [] }
    var rowResults: [(Block2D, Int, Int)] = []
    for i in startRow..<endRow {
        let h: Int = i * 16
        for (xIdx, w) in stride(from: 0, to: dx, by: 16).enumerated() {
            let blockIndex: Int = i * colCount + xIdx
            var block: Block2D = blocks[blockIndex]
            let half: Int = 16 / 2
            var ll: Block2D = prev.getCr(x: w / 2, y: h / 2, size: half)
            ll.withView { srcView in
                block.withView { destView in
                    for yi in 0..<half {
                        let srcPtr = srcView.rowPointer(y: yi)
                        let destPtr = destView.rowPointer(y: yi)
                        destPtr.update(from: srcPtr, count: half)
                    }
                }
            }
            block.withView { view in
                let base = view.base
                var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 16)
                var lhView = BlockView(base: base.advanced(by: half * 16), width: half, height: half, stride: 16)
                var hhView = BlockView(base: base.advanced(by: half * 16 + half), width: half, height: half, stride: 16)
                dequantizeSIMDSignedMapping(&hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping(&lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping(&hhView, q: qt.qHigh)
                invDwt2d_16(&view)
            }
            rowResults.append((block, w, h))
        }
    }
    return rowResults
}

@Sendable @inline(__always)
func decodeBase8ProcessY(taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [Block2D], qt: QuantizationTable) -> [(Block2D, Int, Int)] {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return [] }
    var rowResults: [(Block2D, Int, Int)] = []
    for i in startRow..<endRow {
        let h: Int = i * 8
        for (xIdx, w) in stride(from: 0, to: dx, by: 8).enumerated() {
            let blockIndex: Int = i * colCount + xIdx
            var block: Block2D = blocks[blockIndex]
            let half: Int = 8 / 2
            block.withView { view in
                let base = view.base
                var llView = BlockView(base: base, width: half, height: half, stride: 8)
                var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 8)
                var lhView = BlockView(base: base.advanced(by: half * 8), width: half, height: half, stride: 8)
                var hhView = BlockView(base: base.advanced(by: half * 8 + half), width: half, height: half, stride: 8)
                dequantizeSIMD(&llView, q: qt.qLow)
                dequantizeSIMDSignedMapping(&hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping(&lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping(&hhView, q: qt.qHigh)
                invDwt2d_8(&view)
            }
            rowResults.append((block, w, h))
        }
    }
    return rowResults
}

@Sendable @inline(__always)
func decodeBase8ProcessCb(taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [Block2D], qt: QuantizationTable) -> [(Block2D, Int, Int)] {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return [] }
    var rowResults: [(Block2D, Int, Int)] = []
    for i in startRow..<endRow {
        let h: Int = i * 8
        for (xIdx, w) in stride(from: 0, to: dx, by: 8).enumerated() {
            let blockIndex: Int = i * colCount + xIdx
            var block: Block2D = blocks[blockIndex]
            let half: Int = 8 / 2
            block.withView { view in
                let base = view.base
                var llView = BlockView(base: base, width: half, height: half, stride: 8)
                var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 8)
                var lhView = BlockView(base: base.advanced(by: half * 8), width: half, height: half, stride: 8)
                var hhView = BlockView(base: base.advanced(by: half * 8 + half), width: half, height: half, stride: 8)
                dequantizeSIMD(&llView, q: qt.qLow)
                dequantizeSIMDSignedMapping(&hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping(&lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping(&hhView, q: qt.qHigh)
                invDwt2d_8(&view)
            }
            rowResults.append((block, w, h))
        }
    }
    return rowResults
}

@Sendable @inline(__always)
func decodeBase8ProcessCr(taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [Block2D], qt: QuantizationTable) -> [(Block2D, Int, Int)] {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return [] }
    var rowResults: [(Block2D, Int, Int)] = []
    for i in startRow..<endRow {
        let h: Int = i * 8
        for (xIdx, w) in stride(from: 0, to: dx, by: 8).enumerated() {
            let blockIndex: Int = i * colCount + xIdx
            var block: Block2D = blocks[blockIndex]
            let half: Int = 8 / 2
            block.withView { view in
                let base = view.base
                var llView = BlockView(base: base, width: half, height: half, stride: 8)
                var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 8)
                var lhView = BlockView(base: base.advanced(by: half * 8), width: half, height: half, stride: 8)
                var hhView = BlockView(base: base.advanced(by: half * 8 + half), width: half, height: half, stride: 8)
                dequantizeSIMD(&llView, q: qt.qLow)
                dequantizeSIMDSignedMapping(&hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping(&lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping(&hhView, q: qt.qHigh)
                invDwt2d_8(&view)
            }
            rowResults.append((block, w, h))
        }
    }
    return rowResults
}