// MARK: - Decode Error

public enum DecodeError: Error {
    case eof
    case insufficientData
    case invalidBlockData
    case invalidHeader
    case invalidLayerNumber
    case noDataProvided
    case unsupportedArchitecture
    case outOfBits
}

@inline(__always)
func decodeSpatialLayers(r: [UInt8], maxLayer: Int, predictedPd: PlaneData420? = nil) async throws -> Image16 {
    var offset = 0

    // encodeSpatialLayers appends layer0, then layer1, then layer2.
    // So chunks are ordered [layer0, layer1, layer2].
    
    let len0 = try readUInt32BEFromBytes(r, offset: &offset)
    guard (offset + Int(len0)) <= r.count else { throw DecodeError.insufficientData }
    let layer0Data = Array(r[offset..<(offset + Int(len0))])
    offset += Int(len0)
    
    // Base layer (layer 0) is always Base8
    var current = try await decodeBase8(r: layer0Data, layer: 0)
    
    if 1 <= maxLayer {
        let len1 = try readUInt32BEFromBytes(r, offset: &offset)
        guard (offset + Int(len1)) <= r.count else { throw DecodeError.insufficientData }
        let layer1Data = Array(r[offset..<(offset + Int(len1))])
        offset += Int(len1)
        
        current = try await decodeLayer16(r: layer1Data, layer: 1, prev: current)
    }
    
    if 2 <= maxLayer {
        let len2 = try readUInt32BEFromBytes(r, offset: &offset)
        guard (offset + Int(len2)) <= r.count else { throw DecodeError.insufficientData }
        let layer2Data = Array(r[offset..<(offset + Int(len2))])
        offset += Int(len2)
        
        current = try await decodeLayer32(r: layer2Data, layer: 2, prev: current)
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

@inline(__always)
func decodeCoeffRun(decoder: inout EntropyDecoder, isParentZero: Bool) throws -> (Int, Int16) {
    let pair = try decoder.readPair(isParentZero: isParentZero)
    return (pair.run, pair.val)
}

@inline(__always)
func blockDecode32(decoder: inout EntropyDecoder, block: inout BlockView, parentBlock: BlockView?) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        block.clearAll()
        return
    }

    let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
    let lscpY = Int(try decodeExpGolomb(decoder: &decoder))

    block.clearAll()

    var currentIdx = 0
    let lscpIdx = lscpY * 32 + lscpX

    while currentIdx <= lscpIdx {
        let isParentZero: Bool
        if let pb = parentBlock {
            let y = currentIdx / 32
            let x = currentIdx % 32
            isParentZero = (pb.rowPointer(y: y)[x] == 0)
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
func blockDecode16(decoder: inout EntropyDecoder, block: inout BlockView, parentBlock: BlockView?) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        block.clearAll()
        return
    }

    let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
    let lscpY = Int(try decodeExpGolomb(decoder: &decoder))
    guard lscpX < 16 && lscpY < 16 else { throw DecodeError.invalidBlockData }

    block.clearAll()

    var currentIdx = 0
    let lscpIdx = lscpY * 16 + lscpX

    while currentIdx <= lscpIdx {
        let isParentZero: Bool
        if let pb = parentBlock {
            let y = currentIdx / 16
            let x = currentIdx % 16
            isParentZero = (pb.rowPointer(y: y)[x] == 0)
        } else {
            isParentZero = false
        }
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
func blockDecode8(decoder: inout EntropyDecoder, block: inout BlockView, parentBlock: BlockView?) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        block.clearAll()
        return
    }

    let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
    let lscpY = Int(try decodeExpGolomb(decoder: &decoder))

    block.clearAll()

    var currentIdx = 0
    let lscpIdx = lscpY * 8 + lscpX

    while currentIdx <= lscpIdx {
        let isParentZero: Bool
        if let pb = parentBlock {
            let y = currentIdx / 8
            let x = currentIdx % 8
            isParentZero = (pb.rowPointer(y: y)[x] == 0)
        } else {
            isParentZero = false
        }
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
func blockDecode4(decoder: inout EntropyDecoder, block: inout BlockView, parentBlock: BlockView?) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        block.clearAll()
        return
    }

    let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
    let lscpY = Int(try decodeExpGolomb(decoder: &decoder))
    guard lscpX < 4 && lscpY < 4 else { throw DecodeError.invalidBlockData }

    block.clearAll()

    var currentIdx = 0
    let lscpIdx = lscpY * 4 + lscpX

    while currentIdx <= lscpIdx {
        let isParentZero: Bool
        if let pb = parentBlock {
            let y = currentIdx / 4
            let x = currentIdx % 4
            isParentZero = (pb.rowPointer(y: y)[x] == 0)
        } else {
            isParentZero = false
        }
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
func blockDecodeDPCM4(decoder: inout EntropyDecoder, block: inout BlockView, lastVal: inout Int16) throws {
    let hasNonZero = try decoder.decodeBypass()
    var lscpIdx = -1
    if hasNonZero == 1 {
        let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
        let lscpY = Int(try decodeExpGolomb(decoder: &decoder))
        guard lscpX < 4 && lscpY < 4 else { throw DecodeError.invalidBlockData }
        lscpIdx = lscpY * 4 + lscpX
    }

    block.clearAll()

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
func blockDecodeDPCM8(decoder: inout EntropyDecoder, block: inout BlockView, lastVal: inout Int16) throws {
    let hasNonZero = try decoder.decodeBypass()
    var lscpIdx = -1
    if hasNonZero == 1 {
        let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
        let lscpY = Int(try decodeExpGolomb(decoder: &decoder))
        guard lscpX < 8 && lscpY < 8 else { throw DecodeError.invalidBlockData }
        lscpIdx = lscpY * 8 + lscpX
    }

    block.clearAll()

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

    var last: Int16 = lastVal
    for y in 0..<8 {
        let ptrY = block.rowPointer(y: y)
        if y == 0 {
            for x in 0..<8 {
                if x == 0 {
                    ptrY[0] = ptrY[0] &+ last
                } else {
                    ptrY[x] = ptrY[x] &+ ptrY[x - 1]
                }
            }
        } else {
            let ptrPrevY = block.rowPointer(y: y - 1)
            for x in 0..<8 {
                if x == 0 {
                    ptrY[0] = ptrY[0] &+ ptrPrevY[0]
                } else {
                    ptrY[x] = ptrY[x] &+ predictMED(ptrY[x - 1], ptrPrevY[x], ptrPrevY[x - 1])
                }
            }
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
        let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
        let lscpY = Int(try decodeExpGolomb(decoder: &decoder))
        guard lscpX < 16 && lscpY < 16 else { throw DecodeError.invalidBlockData }
        lscpIdx = lscpY * 16 + lscpX
    }

    block.clearAll()

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

    var last: Int16 = lastVal
    for y in 0..<16 {
        let ptrY = block.rowPointer(y: y)
        if y == 0 {
            for x in 0..<16 {
                if x == 0 {
                    ptrY[0] = ptrY[0] &+ last
                } else {
                    ptrY[x] = ptrY[x] &+ ptrY[x - 1]
                }
            }
        } else {
            let ptrPrevY = block.rowPointer(y: y - 1)
            for x in 0..<16 {
                if x == 0 {
                    ptrY[0] = ptrY[0] &+ ptrPrevY[0]
                } else {
                    ptrY[x] = ptrY[x] &+ predictMED(ptrY[x - 1], ptrPrevY[x], ptrPrevY[x - 1])
                }
            }
        }
        last = ptrY[15]
    }
    lastVal = last
}

// MARK: - Internal Decode Functions

@inline(__always)
func decodeLayer32(r: [UInt8], layer: UInt8, prev: Image16) async throws -> Image16 {
    var offset = 0
    
    guard (offset + 5) <= r.count else { throw DecodeError.insufficientData }
    let header = Array(r[offset..<(offset + 5)])
    offset += 5
    
    guard header[0] == 0x56 && header[1] == 0x45 && header[2] == 0x56 && header[3] == 0x43 else {
         throw DecodeError.invalidHeader
    }
    guard header[4] == layer else {
        throw DecodeError.invalidLayerNumber
    }
    
    let dx = Int(try readUInt16BEFromBytes(r, offset: &offset))
    let dy = Int(try readUInt16BEFromBytes(r, offset: &offset))
    let qtY = QuantizationTable(baseStep: Int(try readUInt16BEFromBytes(r, offset: &offset)), isChroma: false, layerIndex: Int(layer), isOne: false)
    let qtC = QuantizationTable(baseStep: Int(try readUInt16BEFromBytes(r, offset: &offset)), isChroma: true, layerIndex: Int(layer), isOne: false)
    
    let bufYLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufYLen) <= r.count else { throw DecodeError.invalidBlockData }
    let bufY = Array(r[offset..<(offset + bufYLen)])
    offset += bufYLen
    
    let bufCbLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufCbLen) <= r.count else { throw DecodeError.invalidBlockData }
    let bufCb = Array(r[offset..<(offset + bufCbLen)])
    offset += bufCbLen
    
    let bufCrLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufCrLen) <= r.count else { throw DecodeError.invalidBlockData }
    let bufCr = Array(r[offset..<(offset + bufCrLen)])
    offset += bufCrLen
    
    var sub = Image16(width: dx, height: dy)
    
    let rowCountY = (dy + 32 - 1) / 32
    let colCountY = (dx + 32 - 1) / 32
    let yBlocks = try decodePlaneSubbands32(data: bufY, blockCount: rowCountY * colCountY, parentImage: prev, dx: dx, planeType: 0)
    
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let rowCountCb = (cbDy + 32 - 1) / 32
    let colCountCb = (cbDx + 32 - 1) / 32
    let cbBlocks = try decodePlaneSubbands32(data: bufCb, blockCount: rowCountCb * colCountCb, parentImage: prev, dx: cbDx, planeType: 1)
    
    let rowCountCr = (cbDy + 32 - 1) / 32
    let colCountCr = (cbDx + 32 - 1) / 32
    let crBlocks = try decodePlaneSubbands32(data: bufCr, blockCount: rowCountCr * colCountCr, parentImage: prev, dx: cbDx, planeType: 2)
    
    let chunkSize = 4
    
    let taskCountY = (rowCountY + chunkSize - 1) / chunkSize
    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountY {
            group.addTask {
                let startRow = taskIdx * chunkSize
                let endRow = min(startRow + chunkSize, rowCountY)
                var rowResults: [(Block2D, Int, Int)] = []
                for i in startRow..<endRow {
                    let h = i * 32
                    for (xIdx, w) in stride(from: 0, to: dx, by: 32).enumerated() {
                        let blockIndex = i * colCountY + xIdx
                        var block = yBlocks[blockIndex]
                        let half = 32 / 2
                        var ll = prev.getY(x: w / 2, y: h / 2, size: half)
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
                            dequantizeSignedMapping(&hlView, q: qtY.qMid)
                            dequantizeSignedMapping(&lhView, q: qtY.qMid)
                            dequantizeSignedMapping(&hhView, q: qtY.qHigh)
                            invDwt2d_32(&view)
                        }
                        rowResults.append((block, w, h))
                    }
                }
                return rowResults
            }
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
            group.addTask {
                let startRow = taskIdx * chunkSize
                let endRow = min(startRow + chunkSize, rowCountCb)
                var rowResults: [(Block2D, Int, Int)] = []
                for i in startRow..<endRow {
                    let h = i * 32
                    for (xIdx, w) in stride(from: 0, to: cbDx, by: 32).enumerated() {
                        let blockIndex = i * colCountCb + xIdx
                        var block = cbBlocks[blockIndex]
                        let half = 32 / 2
                        var ll = prev.getCb(x: w / 2, y: h / 2, size: half)
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
                            dequantizeSignedMapping(&hlView, q: qtC.qMid)
                            dequantizeSignedMapping(&lhView, q: qtC.qMid)
                            dequantizeSignedMapping(&hhView, q: qtC.qHigh)
                            invDwt2d_32(&view)
                        }
                        rowResults.append((block, w, h))
                    }
                }
                return rowResults
            }
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
            group.addTask {
                let startRow = taskIdx * chunkSize
                let endRow = min(startRow + chunkSize, rowCountCr)
                var rowResults: [(Block2D, Int, Int)] = []
                for i in startRow..<endRow {
                    let h = i * 32
                    for (xIdx, w) in stride(from: 0, to: cbDx, by: 32).enumerated() {
                        let blockIndex = i * colCountCr + xIdx
                        var block = crBlocks[blockIndex]
                        let half = 32 / 2
                        var ll = prev.getCr(x: w / 2, y: h / 2, size: half)
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
                            dequantizeSignedMapping(&hlView, q: qtC.qMid)
                            dequantizeSignedMapping(&lhView, q: qtC.qMid)
                            dequantizeSignedMapping(&hhView, q: qtC.qHigh)
                            invDwt2d_32(&view)
                        }
                        rowResults.append((block, w, h))
                    }
                }
                return rowResults
            }
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
    
    applyDeblockingFilter(plane: &sub.y, width: dx, height: dy, blockSize: 32, qStep: Int(qtY.step))
    applyDeblockingFilter(plane: &sub.cb, width: cbDx, height: cbDy, blockSize: 16, qStep: Int(qtC.step))
    applyDeblockingFilter(plane: &sub.cr, width: cbDx, height: cbDy, blockSize: 16, qStep: Int(qtC.step))
    return sub
}

@inline(__always)
func decodeLayer16(r: [UInt8], layer: UInt8, prev: Image16) async throws -> Image16 {
    var offset = 0
    
    guard (offset + 5) <= r.count else { throw DecodeError.insufficientData }
    let header = Array(r[offset..<(offset + 5)])
    offset += 5
    
    guard header[0] == 0x56 && header[1] == 0x45 && header[2] == 0x56 && header[3] == 0x43 else {
         throw DecodeError.invalidHeader
    }
    guard header[4] == layer else {
        throw DecodeError.invalidLayerNumber
    }
    
    let dx = Int(try readUInt16BEFromBytes(r, offset: &offset))
    let dy = Int(try readUInt16BEFromBytes(r, offset: &offset))
    let qtY = QuantizationTable(baseStep: Int(try readUInt16BEFromBytes(r, offset: &offset)), isChroma: false, layerIndex: Int(layer), isOne: false)
    let qtC = QuantizationTable(baseStep: Int(try readUInt16BEFromBytes(r, offset: &offset)), isChroma: true, layerIndex: Int(layer), isOne: false)
    
    let bufYLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufYLen) <= r.count else { throw DecodeError.invalidBlockData }
    let bufY = Array(r[offset..<(offset + bufYLen)])
    offset += bufYLen
    
    let bufCbLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufCbLen) <= r.count else { throw DecodeError.invalidBlockData }
    let bufCb = Array(r[offset..<(offset + bufCbLen)])
    offset += bufCbLen
    
    let bufCrLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufCrLen) <= r.count else { throw DecodeError.invalidBlockData }
    let bufCr = Array(r[offset..<(offset + bufCrLen)])
    offset += bufCrLen
    
    var sub = Image16(width: dx, height: dy)
    
    let rowCountY = (dy + 16 - 1) / 16
    let colCountY = (dx + 16 - 1) / 16
    let yBlocks = try decodePlaneSubbands16(data: bufY, blockCount: rowCountY * colCountY, parentImage: prev, dx: dx, planeType: 0)
    
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let rowCountCb = (cbDy + 16 - 1) / 16
    let colCountCb = (cbDx + 16 - 1) / 16
    let cbBlocks = try decodePlaneSubbands16(data: bufCb, blockCount: rowCountCb * colCountCb, parentImage: prev, dx: cbDx, planeType: 1)
    
    let rowCountCr = (cbDy + 16 - 1) / 16
    let colCountCr = (cbDx + 16 - 1) / 16
    let crBlocks = try decodePlaneSubbands16(data: bufCr, blockCount: rowCountCr * colCountCr, parentImage: prev, dx: cbDx, planeType: 2)
    
    let chunkSize = 4
    
    let taskCountY = (rowCountY + chunkSize - 1) / chunkSize
    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountY {
            group.addTask {
                let startRow = taskIdx * chunkSize
                let endRow = min(startRow + chunkSize, rowCountY)
                var rowResults: [(Block2D, Int, Int)] = []
                for i in startRow..<endRow {
                    let h = i * 16
                    for (xIdx, w) in stride(from: 0, to: dx, by: 16).enumerated() {
                        let blockIndex = i * colCountY + xIdx
                        var block = yBlocks[blockIndex]
                        let half = 16 / 2
                        var ll = prev.getY(x: w / 2, y: h / 2, size: half)
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
                            dequantizeSignedMapping(&hlView, q: qtY.qMid)
                            dequantizeSignedMapping(&lhView, q: qtY.qMid)
                            dequantizeSignedMapping(&hhView, q: qtY.qHigh)
                            invDwt2d_16(&view)
                        }
                        rowResults.append((block, w, h))
                    }
                }
                return rowResults
            }
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
    
    let taskCountCb = (rowCountCb + chunkSize - 1) / chunkSize
    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountCb {
            group.addTask {
                let startRow = taskIdx * chunkSize
                let endRow = min(startRow + chunkSize, rowCountCb)
                var rowResults: [(Block2D, Int, Int)] = []
                for i in startRow..<endRow {
                    let h = i * 16
                    for (xIdx, w) in stride(from: 0, to: cbDx, by: 16).enumerated() {
                        let blockIndex = i * colCountCb + xIdx
                        var block = cbBlocks[blockIndex]
                        let half = 16 / 2
                        var ll = prev.getCb(x: w / 2, y: h / 2, size: half)
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
                            dequantizeSignedMapping(&hlView, q: qtC.qMid)
                            dequantizeSignedMapping(&lhView, q: qtC.qMid)
                            dequantizeSignedMapping(&hhView, q: qtC.qHigh)
                            invDwt2d_16(&view)
                        }
                        rowResults.append((block, w, h))
                    }
                }
                return rowResults
            }
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
    
    let taskCountCr = (rowCountCr + chunkSize - 1) / chunkSize
    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountCr {
            group.addTask {
                let startRow = taskIdx * chunkSize
                let endRow = min(startRow + chunkSize, rowCountCr)
                var rowResults: [(Block2D, Int, Int)] = []
                for i in startRow..<endRow {
                    let h = i * 16
                    for (xIdx, w) in stride(from: 0, to: cbDx, by: 16).enumerated() {
                        let blockIndex = i * colCountCr + xIdx
                        var block = crBlocks[blockIndex]
                        let half = 16 / 2
                        var ll = prev.getCr(x: w / 2, y: h / 2, size: half)
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
                            dequantizeSignedMapping(&hlView, q: qtC.qMid)
                            dequantizeSignedMapping(&lhView, q: qtC.qMid)
                            dequantizeSignedMapping(&hhView, q: qtC.qHigh)
                            invDwt2d_16(&view)
                        }
                        rowResults.append((block, w, h))
                    }
                }
                return rowResults
            }
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
    
    return sub
}

@inline(__always)
func decodeBase8(r: [UInt8], layer: UInt8) async throws -> Image16 {
    var offset = 0
    
    guard (offset + 5) <= r.count else { throw DecodeError.insufficientData }
    let header = Array(r[offset..<(offset + 5)])
    offset += 5
    
    guard header[0] == 0x56 && header[1] == 0x45 && header[2] == 0x56 && header[3] == 0x43 else {
         throw DecodeError.invalidHeader
    }
    guard header[4] == layer else {
        throw DecodeError.invalidLayerNumber
    }
    
    let dx = Int(try readUInt16BEFromBytes(r, offset: &offset))
    let dy = Int(try readUInt16BEFromBytes(r, offset: &offset))
    let qtY = QuantizationTable(baseStep: Int(try readUInt16BEFromBytes(r, offset: &offset)), isChroma: false, layerIndex: Int(layer), isOne: false)
    let qtC = QuantizationTable(baseStep: Int(try readUInt16BEFromBytes(r, offset: &offset)), isChroma: true, layerIndex: Int(layer), isOne: false)
    
    let bufYLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufYLen) <= r.count else { throw DecodeError.invalidBlockData }
    let bufY = Array(r[offset..<(offset + bufYLen)])
    offset += bufYLen
    
    let bufCbLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufCbLen) <= r.count else { throw DecodeError.invalidBlockData }
    let bufCb = Array(r[offset..<(offset + bufCbLen)])
    offset += bufCbLen
    
    let bufCrLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufCrLen) <= r.count else { throw DecodeError.invalidBlockData }
    let bufCr = Array(r[offset..<(offset + bufCrLen)])
    offset += bufCrLen
    
    var sub = Image16(width: dx, height: dy)
    
    let rowCountY = (dy + 8 - 1) / 8
    let colCountY = (dx + 8 - 1) / 8
    let yBlocks = try decodePlaneBaseSubbands8(data: bufY, blockCount: rowCountY * colCountY)
    
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let rowCountCb = (cbDy + 8 - 1) / 8
    let colCountCb = (cbDx + 8 - 1) / 8
    let cbBlocks = try decodePlaneBaseSubbands8(data: bufCb, blockCount: rowCountCb * colCountCb)
    
    let rowCountCr = (cbDy + 8 - 1) / 8
    let colCountCr = (cbDx + 8 - 1) / 8
    let crBlocks = try decodePlaneBaseSubbands8(data: bufCr, blockCount: rowCountCr * colCountCr)
    
    let chunkSize = 4
    
    let taskCountY = (rowCountY + chunkSize - 1) / chunkSize
    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountY {
            group.addTask {
                let startRow = taskIdx * chunkSize
                let endRow = min(startRow + chunkSize, rowCountY)
                var rowResults: [(Block2D, Int, Int)] = []
                for i in startRow..<endRow {
                    let h = i * 8
                    for (xIdx, w) in stride(from: 0, to: dx, by: 8).enumerated() {
                        let blockIndex = i * colCountY + xIdx
                        var block = yBlocks[blockIndex]
                        let half = 8 / 2
                        block.withView { view in
                            let base = view.base
                            var llView = BlockView(base: base, width: half, height: half, stride: 8)
                            var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 8)
                            var lhView = BlockView(base: base.advanced(by: half * 8), width: half, height: half, stride: 8)
                            var hhView = BlockView(base: base.advanced(by: half * 8 + half), width: half, height: half, stride: 8)
                            dequantize(&llView, q: qtY.qLow)
                            dequantizeSignedMapping(&hlView, q: qtY.qMid)
                            dequantizeSignedMapping(&lhView, q: qtY.qMid)
                            dequantizeSignedMapping(&hhView, q: qtY.qHigh)
                            invDwt2d_8(&view)
                        }
                        rowResults.append((block, w, h))
                    }
                }
                return rowResults
            }
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
    
    let taskCountCb = (rowCountCb + chunkSize - 1) / chunkSize
    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountCb {
            group.addTask {
                let startRow = taskIdx * chunkSize
                let endRow = min(startRow + chunkSize, rowCountCb)
                var rowResults: [(Block2D, Int, Int)] = []
                for i in startRow..<endRow {
                    let h = i * 8
                    for (xIdx, w) in stride(from: 0, to: cbDx, by: 8).enumerated() {
                        let blockIndex = i * colCountCb + xIdx
                        var block = cbBlocks[blockIndex]
                        let half = 8 / 2
                        block.withView { view in
                            let base = view.base
                            var llView = BlockView(base: base, width: half, height: half, stride: 8)
                            var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 8)
                            var lhView = BlockView(base: base.advanced(by: half * 8), width: half, height: half, stride: 8)
                            var hhView = BlockView(base: base.advanced(by: half * 8 + half), width: half, height: half, stride: 8)
                            dequantize(&llView, q: qtC.qLow)
                            dequantizeSignedMapping(&hlView, q: qtC.qMid)
                            dequantizeSignedMapping(&lhView, q: qtC.qMid)
                            dequantizeSignedMapping(&hhView, q: qtC.qHigh)
                            invDwt2d_8(&view)
                        }
                        rowResults.append((block, w, h))
                    }
                }
                return rowResults
            }
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
    
    let taskCountCr = (rowCountCr + chunkSize - 1) / chunkSize
    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountCr {
            group.addTask {
                let startRow = taskIdx * chunkSize
                let endRow = min(startRow + chunkSize, rowCountCr)
                var rowResults: [(Block2D, Int, Int)] = []
                for i in startRow..<endRow {
                    let h = i * 8
                    for (xIdx, w) in stride(from: 0, to: cbDx, by: 8).enumerated() {
                        let blockIndex = i * colCountCr + xIdx
                        var block = crBlocks[blockIndex]
                        let half = 8 / 2
                        block.withView { view in
                            let base = view.base
                            var llView = BlockView(base: base, width: half, height: half, stride: 8)
                            var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 8)
                            var lhView = BlockView(base: base.advanced(by: half * 8), width: half, height: half, stride: 8)
                            var hhView = BlockView(base: base.advanced(by: half * 8 + half), width: half, height: half, stride: 8)
                            dequantize(&llView, q: qtC.qLow)
                            dequantizeSignedMapping(&hlView, q: qtC.qMid)
                            dequantizeSignedMapping(&lhView, q: qtC.qMid)
                            dequantizeSignedMapping(&hhView, q: qtC.qHigh)
                            invDwt2d_8(&view)
                        }
                        rowResults.append((block, w, h))
                    }
                }
                return rowResults
            }
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
    
    return sub
}

@inline(__always)
func decodeBase32(r: [UInt8], layer: UInt8) async throws -> Image16 {
    var offset = 0
    guard (offset + 5) <= r.count else { throw DecodeError.insufficientData }
    let header = Array(r[offset..<(offset + 5)])
    offset += 5
    
    guard header[0] == 0x56 && header[1] == 0x45 && header[2] == 0x56 && header[3] == 0x43 else {
         throw DecodeError.invalidHeader
    }
    guard header[4] == layer else {
        throw DecodeError.invalidLayerNumber
    }
    
    let dx = Int(try readUInt16BEFromBytes(r, offset: &offset))
    let dy = Int(try readUInt16BEFromBytes(r, offset: &offset))
    let qtY = QuantizationTable(baseStep: Int(try readUInt16BEFromBytes(r, offset: &offset)), isChroma: false, layerIndex: Int(layer), isOne: true)
    let qtC = QuantizationTable(baseStep: Int(try readUInt16BEFromBytes(r, offset: &offset)), isChroma: true, layerIndex: Int(layer), isOne: true)
    
    let bufYLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufYLen) <= r.count else { throw DecodeError.invalidBlockData }
    let bufY = Array(r[offset..<(offset + bufYLen)])
    offset += bufYLen
    
    let bufCbLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufCbLen) <= r.count else { throw DecodeError.invalidBlockData }
    let bufCb = Array(r[offset..<(offset + bufCbLen)])
    offset += bufCbLen
    
    let bufCrLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufCrLen) <= r.count else { throw DecodeError.invalidBlockData }
    let bufCr = Array(r[offset..<(offset + bufCrLen)])
    offset += bufCrLen
    
    var sub = Image16(width: dx, height: dy)
    
    let rowCountY = (dy + 32 - 1) / 32
    let colCountY = (dx + 32 - 1) / 32
    let yBlocks = try decodePlaneBaseSubbands32(data: bufY, blockCount: rowCountY * colCountY)
    
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let rowCountCb = (cbDy + 32 - 1) / 32
    let colCountCb = (cbDx + 32 - 1) / 32
    let cbBlocks = try decodePlaneBaseSubbands32(data: bufCb, blockCount: rowCountCb * colCountCb)
    
    let rowCountCr = (cbDy + 32 - 1) / 32
    let colCountCr = (cbDx + 32 - 1) / 32
    let crBlocks = try decodePlaneBaseSubbands32(data: bufCr, blockCount: rowCountCr * colCountCr)
    
    let chunkSize = 4
    
    let taskCountY = (rowCountY + chunkSize - 1) / chunkSize
    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountY {
            group.addTask {
                let startRow = taskIdx * chunkSize
                let endRow = min(startRow + chunkSize, rowCountY)
                var rowResults: [(Block2D, Int, Int)] = []
                for i in startRow..<endRow {
                    let h = i * 32
                    for (xIdx, w) in stride(from: 0, to: dx, by: 32).enumerated() {
                        let blockIndex = i * colCountY + xIdx
                        var block = yBlocks[blockIndex]
                        let half = 32 / 2
                        block.withView { view in
                            let base = view.base
                            var llView = BlockView(base: base, width: half, height: half, stride: 32)
                            var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 32)
                            var lhView = BlockView(base: base.advanced(by: half * 32), width: half, height: half, stride: 32)
                            var hhView = BlockView(base: base.advanced(by: half * 32 + half), width: half, height: half, stride: 32)
                            dequantize(&llView, q: qtY.qLow)
                            dequantizeSignedMapping(&hlView, q: qtY.qMid)
                            dequantizeSignedMapping(&lhView, q: qtY.qMid)
                            dequantizeSignedMapping(&hhView, q: qtY.qHigh)
                            invDwt2d_32(&view)
                        }
                        rowResults.append((block, w, h))
                    }
                }
                return rowResults
            }
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
            group.addTask {
                let startRow = taskIdx * chunkSize
                let endRow = min(startRow + chunkSize, rowCountCb)
                var rowResults: [(Block2D, Int, Int)] = []
                for i in startRow..<endRow {
                    let h = i * 32
                    for (xIdx, w) in stride(from: 0, to: cbDx, by: 32).enumerated() {
                        let blockIndex = i * colCountCb + xIdx
                        var block = cbBlocks[blockIndex]
                        let half = 32 / 2
                        block.withView { view in
                            let base = view.base
                            var llView = BlockView(base: base, width: half, height: half, stride: 32)
                            var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 32)
                            var lhView = BlockView(base: base.advanced(by: half * 32), width: half, height: half, stride: 32)
                            var hhView = BlockView(base: base.advanced(by: half * 32 + half), width: half, height: half, stride: 32)
                            dequantize(&llView, q: qtC.qLow)
                            dequantizeSignedMapping(&hlView, q: qtC.qMid)
                            dequantizeSignedMapping(&lhView, q: qtC.qMid)
                            dequantizeSignedMapping(&hhView, q: qtC.qHigh)
                            invDwt2d_32(&view)
                        }
                        rowResults.append((block, w, h))
                    }
                }
                return rowResults
            }
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
            group.addTask {
                let startRow = taskIdx * chunkSize
                let endRow = min(startRow + chunkSize, rowCountCr)
                var rowResults: [(Block2D, Int, Int)] = []
                for i in startRow..<endRow {
                    let h = i * 32
                    for (xIdx, w) in stride(from: 0, to: cbDx, by: 32).enumerated() {
                        let blockIndex = i * colCountCr + xIdx
                        var block = crBlocks[blockIndex]
                        let half = 32 / 2
                        block.withView { view in
                            let base = view.base
                            var llView = BlockView(base: base, width: half, height: half, stride: 32)
                            var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 32)
                            var lhView = BlockView(base: base.advanced(by: half * 32), width: half, height: half, stride: 32)
                            var hhView = BlockView(base: base.advanced(by: half * 32 + half), width: half, height: half, stride: 32)
                            dequantize(&llView, q: qtC.qLow)
                            dequantizeSignedMapping(&hlView, q: qtC.qMid)
                            dequantizeSignedMapping(&lhView, q: qtC.qMid)
                            dequantizeSignedMapping(&hhView, q: qtC.qHigh)
                            invDwt2d_32(&view)
                        }
                        rowResults.append((block, w, h))
                    }
                }
                return rowResults
            }
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
    
    applyDeblockingFilter(plane: &sub.y, width: dx, height: dy, blockSize: 32, qStep: Int(qtY.step))
    applyDeblockingFilter(plane: &sub.cb, width: cbDx, height: cbDy, blockSize: 16, qStep: Int(qtC.step))
    applyDeblockingFilter(plane: &sub.cr, width: cbDx, height: cbDy, blockSize: 16, qStep: Int(qtC.step))
    return sub
}

public struct DecodeOptions: Sendable {
    public var maxLayer: Int
    public var maxFrames: Int
    public var isOne: Bool
    
    public init(maxLayer: Int = 2, maxFrames: Int = 4, isOne: Bool = false) {
        self.maxLayer = maxLayer
        self.maxFrames = maxFrames
        self.isOne = isOne
    }
}

#if (arch(arm64) || arch(x86_64) || arch(wasm32))
@inline(__always)
public func decode(data: [UInt8], opts: DecodeOptions = DecodeOptions()) async throws -> [YCbCrImage] {
    if data.isEmpty { return [] }
    var out: [YCbCrImage] = []
    var offset = 0
    var prevReconstructed: PlaneData420? = nil
    
    while offset + 4 <= data.count {
        let magic = Array(data[offset..<(offset + 4)])
        offset += 4
        
        switch magic {
        case [0x56, 0x45, 0x56, 0x49], [0x56, 0x45, 0x4F, 0x49]: // VEVI, VEOI
            let len = Int(try readUInt32BEFromBytes(data, offset: &offset))
            guard (offset + len) <= data.count else { throw DecodeError.insufficientData }
            let chunk = Array(data[offset..<(offset + len)])
            offset += len
            
            let img16: Image16
            let isOneBlock = opts.isOne || magic == [0x56, 0x45, 0x4F, 0x49]
            if isOneBlock {
                img16 = try await decodeBase32(r: chunk, layer: 0)
            } else {
                img16 = try await decodeSpatialLayers(r: chunk, maxLayer: opts.maxLayer)
            }
            let pd = PlaneData420(img16: img16)
            out.append(pd.toYCbCr())
            prevReconstructed = pd

        case [0x56, 0x45, 0x56, 0x48]: // VEVH
            let len = Int(try readUInt32BEFromBytes(data, offset: &offset))
            guard (offset + len) <= data.count else { throw DecodeError.insufficientData }
            offset += len

        case [0x56, 0x45, 0x56, 0x50], [0x56, 0x45, 0x4F, 0x50]: // VEVP, VEOP
            let mvsCount = Int(try readUInt32BEFromBytes(data, offset: &offset))
            let mvDataLen = Int(try readUInt32BEFromBytes(data, offset: &offset))
            var mvs = MotionVectors(count: mvsCount)
            guard (offset + mvDataLen) <= data.count else { throw DecodeError.insufficientData }

            let mvData = Array(data[offset..<(offset + mvDataLen)])
            offset += mvDataLen
            var mvBr = try EntropyDecoder(data: mvData)

            let mbSize = 64
            // We need width to compute mbCols. We can infer width from previous frame.
            guard let prevWidth = prevReconstructed?.width else { throw DecodeError.invalidHeader }
            let mbCols = (prevWidth + mbSize - 1) / mbSize

            for i in 0..<mvsCount {
                let mbX = i % mbCols
                let mbY = i / mbCols
                let pmv = calculatePMV(mvs: mvs, mbX: mbX, mbY: mbY, mbCols: mbCols)

                let isSig = try mvBr.decodeBypass()
                if isSig == 0 {
                    mvs.vectors[i] = SIMD2(Int16(clamping: pmv.dx), Int16(clamping: pmv.dy))
                } else {
                    let sx = try mvBr.decodeBypass()
                    let mx = try decodeExpGolomb(decoder: &mvBr)

                    let mvdX: Int
                    if sx == 1 {
                        mvdX = -1 * Int(mx)
                    } else {
                        mvdX = Int(mx)
                    }

                    let sy = try mvBr.decodeBypass()
                    let my = try decodeExpGolomb(decoder: &mvBr)

                    let mvdY: Int
                    if sy == 1 {
                        mvdY = -1 * Int(my)
                    } else {
                        mvdY = Int(my)
                    }

                    mvs.vectors[i] = SIMD2(Int16(clamping: mvdX + pmv.dx), Int16(clamping: mvdY + pmv.dy))
                }
            }
            
            let len = Int(try readUInt32BEFromBytes(data, offset: &offset))
            guard (offset + len) <= data.count else { throw DecodeError.insufficientData }
            let chunk = Array(data[offset..<(offset + len)])
            offset += len
            
            let img16: Image16
            let isOneBlock = opts.isOne || magic == [0x56, 0x45, 0x4F, 0x50] // VEOI/VEOP
            if isOneBlock {
                img16 = try await decodeBase32(r: chunk, layer: 0)
            } else {
                img16 = try await decodeSpatialLayers(r: chunk, maxLayer: opts.maxLayer)
            }
            let residual = PlaneData420(img16: img16)
            
            if let prev = prevReconstructed {
                let predicted = await applyMBME(prev: prev, mvs: mvs)
                let curr = await addPlanes(residual: residual, predicted: predicted)
                out.append(curr.toYCbCr())
                prevReconstructed = curr
            } else {
                out.append(residual.toYCbCr())
            }
            

            
        default: 
             throw DecodeError.invalidHeader
        }
    }
    
    return out
}

@inline(__always)
public func decodeOne(data: [UInt8]) async throws -> [YCbCrImage] {
    return try await decode(data: data, opts: DecodeOptions(maxLayer: 0, maxFrames: 4, isOne: true))
}

#else
public func decode(data: [UInt8], opts: DecodeOptions = DecodeOptions()) async throws -> [YCbCrImage] {
    throw DecodeError.unsupportedArchitecture
}
public func decodeOne(data: [UInt8]) async throws -> [YCbCrImage] {
    throw DecodeError.unsupportedArchitecture
}
#endif
