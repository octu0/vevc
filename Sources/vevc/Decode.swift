// MARK: - Decode Error

public enum DecodeError: Error {
    case eof
    case insufficientData
    case invalidBlockData
    case invalidHeader
    case invalidLayerNumber
    case noDataProvided
    case unsupportedArchitecture
}

@inline(__always)
func decodeSpatialLayers(r: [UInt8], maxLayer: Int, predictedPd: PlaneData420? = nil) async throws -> Image16 {
    var offset = 0
    let len0 = try readUInt32BEFromBytes(r, offset: &offset)
    guard (offset + Int(len0)) <= r.count else { throw DecodeError.insufficientData }

    let layer0Data = Array(r[offset..<(offset + Int(len0))])
    offset += Int(len0)
    
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
func toInt16(_ u: UInt16) -> Int16 {
    let s = Int16(bitPattern: (u >> 1))
    let m = (-1 * Int16(bitPattern: (u & 1)))
    return (s ^ m)
}

@inline(__always)
func decodeExpGolomb(decoder: inout CABACDecoder) throws -> UInt32 {
    var q: UInt32 = 0
    while try decoder.decodeBypass() == 1 {
        q += 1
    }
    var val: UInt32 = 0
    for _ in 0..<q {
        let bit = try decoder.decodeBypass()
        val = (val << 1) | UInt32(bit)
    }
    return ((1 << q) + val - 1)
}

@inline(__always)
func decodeCoeffRun(decoder: inout CABACDecoder, ctxRun: inout [ContextModel], ctxMag: inout [ContextModel], band: Int) throws -> (Int, Int16) {
    var run = 0
    let ctxBandOffset = min(band, 7) * 8
    try ctxRun.withUnsafeMutableBufferPointer { ptr in
        guard let base = ptr.baseAddress else { return }
        for i in 0..<7 {
            let bit = try decoder.decodeBin(ctx: &base[ctxBandOffset + Int(i)])
            if bit == 0 {
                break
            }
            run += 1
        }
    }
    if run == 7 {
        let rem = try decodeExpGolomb(decoder: &decoder)
        run += Int(rem)
    }

    let signBit = try decoder.decodeBypass()

    var mag: UInt32 = 1
    try ctxMag.withUnsafeMutableBufferPointer { ptr in
        guard let base = ptr.baseAddress else { return }
        for i in 0..<7 {
            let bit = try decoder.decodeBin(ctx: &base[ctxBandOffset + Int(i)])
            if bit == 0 {
                break
            }
            mag += 1
        }
    }

    if mag == 8 {
        let rem = try decodeExpGolomb(decoder: &decoder)
        mag += rem
    }

    let sVal = Int16(mag)
    let finalVal: Int16
    if signBit == 1 {
        finalVal = -1 * sVal
    } else {
        finalVal = sVal
    }

    return (run, finalVal)
}

@inline(__always)
func blockDecode32(decoder: inout CABACDecoder, block: inout BlockView, ctxRun: inout [ContextModel], ctxMag: inout [ContextModel]) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        for y in 0..<32 {
            let ptr = block.rowPointer(y: y)
            for x in 0..<32 {
                ptr[x] = 0
            }
        }
        return
    }

    let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
    let lscpY = Int(try decodeExpGolomb(decoder: &decoder))

    for y in 0..<32 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<32 { ptr[x] = 0 }
    }

    var currentIdx = 0
    let lscpIdx = lscpY * 32 + lscpX

    while currentIdx <= lscpIdx {
        let startY = currentIdx / 32
        let startX = currentIdx % 32
        let band = min(startX + startY, 7)

        let (run, val) = try decodeCoeffRun(decoder: &decoder, ctxRun: &ctxRun, ctxMag: &ctxMag, band: band)

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
func blockDecode16(decoder: inout CABACDecoder, block: inout BlockView, ctxRun: inout [ContextModel], ctxMag: inout [ContextModel]) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        for y in 0..<16 {
            let ptr = block.rowPointer(y: y)
            for x in 0..<16 {
                ptr[x] = 0
            }
        }
        return
    }

    let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
    let lscpY = Int(try decodeExpGolomb(decoder: &decoder))
    guard lscpX < 16 && lscpY < 16 else { throw DecodeError.invalidBlockData }

    for y in 0..<16 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<16 { ptr[x] = 0 }
    }

    var currentIdx = 0
    let lscpIdx = lscpY * 16 + lscpX

    while currentIdx <= lscpIdx {
        let startY = currentIdx / 16
        let startX = currentIdx % 16
        let band = min(startX + startY, 7)

        let (run, val) = try decodeCoeffRun(decoder: &decoder, ctxRun: &ctxRun, ctxMag: &ctxMag, band: band)

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
func blockDecode8(decoder: inout CABACDecoder, block: inout BlockView, ctxRun: inout [ContextModel], ctxMag: inout [ContextModel]) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        for y in 0..<8 {
            let ptr = block.rowPointer(y: y)
            for x in 0..<8 {
                ptr[x] = 0
            }
        }
        return
    }

    let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
    let lscpY = Int(try decodeExpGolomb(decoder: &decoder))

    for y in 0..<8 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<8 { ptr[x] = 0 }
    }

    var currentIdx = 0
    let lscpIdx = lscpY * 8 + lscpX

    while currentIdx <= lscpIdx {
        let startY = currentIdx / 8
        let startX = currentIdx % 8
        let band = min(startX + startY, 7)

        let (run, val) = try decodeCoeffRun(decoder: &decoder, ctxRun: &ctxRun, ctxMag: &ctxMag, band: band)

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
func blockDecode4(decoder: inout CABACDecoder, block: inout BlockView, ctxRun: inout [ContextModel], ctxMag: inout [ContextModel]) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        for y in 0..<4 {
            let ptr = block.rowPointer(y: y)
            for x in 0..<4 {
                ptr[x] = 0
            }
        }
        return
    }

    let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
    let lscpY = Int(try decodeExpGolomb(decoder: &decoder))

    for y in 0..<4 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<4 { ptr[x] = 0 }
    }

    var currentIdx = 0
    let lscpIdx = lscpY * 4 + lscpX

    while currentIdx <= lscpIdx {
        let startY = currentIdx / 4
        let startX = currentIdx % 4
        let band = min(startX + startY, 7)

        let (run, val) = try decodeCoeffRun(decoder: &decoder, ctxRun: &ctxRun, ctxMag: &ctxMag, band: band)

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
func blockDecodeDPCM4(decoder: inout CABACDecoder, block: inout BlockView, lastVal: inout Int16, ctxRun: inout [ContextModel], ctxMag: inout [ContextModel]) throws {
    let hasNonZero = try decoder.decodeBypass()
    var lscpIdx = -1
    if hasNonZero == 1 {
        let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
        let lscpY = Int(try decodeExpGolomb(decoder: &decoder))
        lscpIdx = lscpY * 4 + lscpX
    }

    for y in 0..<4 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<4 { ptr[x] = 0 }
    }

    var currentIdx = 0
    
    while currentIdx <= lscpIdx {
        let startY = currentIdx / 4
        let startX = currentIdx % 4
        let band = min(startX + startY, 7)

        let (run, val) = try decodeCoeffRun(decoder: &decoder, ctxRun: &ctxRun, ctxMag: &ctxMag, band: band)

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
func blockDecodeDPCM8(decoder: inout CABACDecoder, block: inout BlockView, lastVal: inout Int16, ctxRun: inout [ContextModel], ctxMag: inout [ContextModel]) throws {
    let hasNonZero = try decoder.decodeBypass()
    var lscpIdx = -1
    if hasNonZero == 1 {
        let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
        let lscpY = Int(try decodeExpGolomb(decoder: &decoder))
        guard lscpX < 8 && lscpY < 8 else { throw DecodeError.invalidBlockData }
        lscpIdx = lscpY * 8 + lscpX
    }

    for y in 0..<8 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<8 { ptr[x] = 0 }
    }

    var currentIdx = 0
    
    while currentIdx <= lscpIdx {
        let startY = currentIdx / 8
        let startX = currentIdx % 8
        let band = min(startX + startY, 7)

        let (run, val) = try decodeCoeffRun(decoder: &decoder, ctxRun: &ctxRun, ctxMag: &ctxMag, band: band)

        currentIdx += run
        if currentIdx <= lscpIdx {
            let y = currentIdx / 8
            let x = currentIdx % 8
            let ptr = block.rowPointer(y: y)
            ptr[x] = val
        }
        currentIdx += 1
    }

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
func blockDecodeDPCM16(decoder: inout CABACDecoder, block: inout BlockView, lastVal: inout Int16, ctxRun: inout [ContextModel], ctxMag: inout [ContextModel]) throws {
    let hasNonZero = try decoder.decodeBypass()
    var lscpIdx = -1
    if hasNonZero == 1 {
        let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
        let lscpY = Int(try decodeExpGolomb(decoder: &decoder))
        guard lscpX < 16 && lscpY < 16 else { throw DecodeError.invalidBlockData }
        lscpIdx = lscpY * 16 + lscpX
    }

    for y in 0..<16 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<16 { ptr[x] = 0 }
    }

    var currentIdx = 0
    
    while currentIdx <= lscpIdx {
        let startY = currentIdx / 16
        let startX = currentIdx % 16
        let band = min(startX + startY, 7)

        let (run, val) = try decodeCoeffRun(decoder: &decoder, ctxRun: &ctxRun, ctxMag: &ctxMag, band: band)

        currentIdx += run
        if currentIdx <= lscpIdx {
            let y = currentIdx / 16
            let x = currentIdx % 16
            let ptr = block.rowPointer(y: y)
            ptr[x] = val
        }
        currentIdx += 1
    }

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

enum DecodeTask32 {
    case skip
    case decode16
    case split8(Bool, Bool, Bool, Bool)
}

@inline(__always)
func decodePlaneSubbands32(data: [UInt8], blockCount: Int) throws -> [Block2D] {
    var blocks: [Block2D] = []
    blocks.reserveCapacity(blockCount)
    for _ in 0..<blockCount {
        blocks.append(Block2D(width: 32, height: 32))
    }
    
    var brFlags = CABACBitReader(data: data)
    var tasks: [(Int, DecodeTask32)] = []
    tasks.reserveCapacity(blockCount)
    for i in 0..<blockCount {
        let isZero = try brFlags.readBit()
        if isZero == 1 {
            tasks.append((i, .skip))
        } else {
            let mbType = try brFlags.readBit()
            if mbType == 1 {
                let tlZero = try brFlags.readBit()
                if tlZero == 0 { let _ = try brFlags.readBit() }
                
                let trZero = try brFlags.readBit()
                if trZero == 0 { let _ = try brFlags.readBit() }
                
                let blZero = try brFlags.readBit()
                if blZero == 0 { let _ = try brFlags.readBit() }
                
                let brZero = try brFlags.readBit()
                if brZero == 0 { let _ = try brFlags.readBit() }
                
                tasks.append((i, .split8(tlZero == 0, trZero == 0, blZero == 0, brZero == 0)))
            } else {
                tasks.append((i, .decode16))
            }
        }
    }
    
    let consumed = brFlags.consumedBytes
    guard consumed <= data.count else { throw DecodeError.insufficientData }
    let dataSlice = Array(data[consumed...])
    
    var decoder = try CABACDecoder(data: dataSlice)
    
    let half = 32 / 2
    
    var ctxRunHL = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagHL = [ContextModel](repeating: ContextModel(), count: 64)
    
    var ctxRunLH = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagLH = [ContextModel](repeating: ContextModel(), count: 64)
    
    var ctxRunHH = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagHH = [ContextModel](repeating: ContextModel(), count: 64)

    for (i, task) in tasks {
        try blocks[i].withView { view in
            let hlBase = view.base.advanced(by: half)
            let lhBase = view.base.advanced(by: half * 32)
            let hhBase = view.base.advanced(by: half * 32 + half)
            
            switch task {
            case .skip:
                break
            case .decode16:
                var hlView = BlockView(base: hlBase, width: half, height: half, stride: 32)
                try blockDecode16(decoder: &decoder, block: &hlView, ctxRun: &ctxRunHL, ctxMag: &ctxMagHL)
                
                var lhView = BlockView(base: lhBase, width: half, height: half, stride: 32)
                try blockDecode16(decoder: &decoder, block: &lhView, ctxRun: &ctxRunLH, ctxMag: &ctxMagLH)
                
                var hhView = BlockView(base: hhBase, width: half, height: half, stride: 32)
                try blockDecode16(decoder: &decoder, block: &hhView, ctxRun: &ctxRunHH, ctxMag: &ctxMagHH)
            case .split8(let tl, let tr, let bl, let br):
                if tl {
                    var hl = BlockView(base: hlBase, width: 8, height: 8, stride: 32)
                    var lh = BlockView(base: lhBase, width: 8, height: 8, stride: 32)
                    var hh = BlockView(base: hhBase, width: 8, height: 8, stride: 32)
                    try blockDecode8(decoder: &decoder, block: &hl, ctxRun: &ctxRunHL, ctxMag: &ctxMagHL)
                    try blockDecode8(decoder: &decoder, block: &lh, ctxRun: &ctxRunLH, ctxMag: &ctxMagLH)
                    try blockDecode8(decoder: &decoder, block: &hh, ctxRun: &ctxRunHH, ctxMag: &ctxMagHH)
                }
                if tr {
                    var hl = BlockView(base: hlBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    var lh = BlockView(base: lhBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    var hh = BlockView(base: hhBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    try blockDecode8(decoder: &decoder, block: &hl, ctxRun: &ctxRunHL, ctxMag: &ctxMagHL)
                    try blockDecode8(decoder: &decoder, block: &lh, ctxRun: &ctxRunLH, ctxMag: &ctxMagLH)
                    try blockDecode8(decoder: &decoder, block: &hh, ctxRun: &ctxRunHH, ctxMag: &ctxMagHH)
                }
                if bl {
                    var hl = BlockView(base: hlBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    var lh = BlockView(base: lhBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    var hh = BlockView(base: hhBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    try blockDecode8(decoder: &decoder, block: &hl, ctxRun: &ctxRunHL, ctxMag: &ctxMagHL)
                    try blockDecode8(decoder: &decoder, block: &lh, ctxRun: &ctxRunLH, ctxMag: &ctxMagLH)
                    try blockDecode8(decoder: &decoder, block: &hh, ctxRun: &ctxRunHH, ctxMag: &ctxMagHH)
                }
                if br {
                    var hl = BlockView(base: hlBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    var lh = BlockView(base: lhBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    var hh = BlockView(base: hhBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    try blockDecode8(decoder: &decoder, block: &hl, ctxRun: &ctxRunHL, ctxMag: &ctxMagHL)
                    try blockDecode8(decoder: &decoder, block: &lh, ctxRun: &ctxRunLH, ctxMag: &ctxMagLH)
                    try blockDecode8(decoder: &decoder, block: &hh, ctxRun: &ctxRunHH, ctxMag: &ctxMagHH)
                }
            }
        }
    }

    return blocks
}

enum DecodeTask16 {
    case skip
    case decode8
    case split4(Bool, Bool, Bool, Bool)
}

@inline(__always)
func decodePlaneSubbands16(data: [UInt8], blockCount: Int) throws -> [Block2D] {
    var blocks: [Block2D] = []
    blocks.reserveCapacity(blockCount)
    for _ in 0..<blockCount {
        blocks.append(Block2D(width: 16, height: 16))
    }
    
    var brFlags = CABACBitReader(data: data)
    var tasks: [(Int, DecodeTask16)] = []
    tasks.reserveCapacity(blockCount)
    for i in 0..<blockCount {
        let isZero = try brFlags.readBit()
        if isZero == 1 {
            tasks.append((i, .skip))
        } else {
            let mbType = try brFlags.readBit()
            if mbType == 1 {
                let tlZero = try brFlags.readBit()
                if tlZero == 0 { let _ = try brFlags.readBit() }
                
                let trZero = try brFlags.readBit()
                if trZero == 0 { let _ = try brFlags.readBit() }
                
                let blZero = try brFlags.readBit()
                if blZero == 0 { let _ = try brFlags.readBit() }
                
                let brZero = try brFlags.readBit()
                if brZero == 0 { let _ = try brFlags.readBit() }
                
                tasks.append((i, .split4(tlZero == 0, trZero == 0, blZero == 0, brZero == 0)))
            } else {
                tasks.append((i, .decode8))
            }
        }
    }
    
    let consumed = brFlags.consumedBytes
    guard consumed <= data.count else { throw DecodeError.insufficientData }
    let dataSlice = Array(data[consumed...])
    
    var decoder = try CABACDecoder(data: dataSlice)
    
    let half = 16 / 2
    
    var ctxRunHL = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagHL = [ContextModel](repeating: ContextModel(), count: 64)
    
    var ctxRunLH = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagLH = [ContextModel](repeating: ContextModel(), count: 64)
    
    var ctxRunHH = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagHH = [ContextModel](repeating: ContextModel(), count: 64)

    for (i, task) in tasks {
        try blocks[i].withView { view in
            let hlBase = view.base.advanced(by: half)
            let lhBase = view.base.advanced(by: half * 16)
            let hhBase = view.base.advanced(by: half * 16 + half)
            
            switch task {
            case .skip:
                break
            case .decode8:
                var hlView = BlockView(base: hlBase, width: half, height: half, stride: 16)
                try blockDecode8(decoder: &decoder, block: &hlView, ctxRun: &ctxRunHL, ctxMag: &ctxMagHL)
                
                var lhView = BlockView(base: lhBase, width: half, height: half, stride: 16)
                try blockDecode8(decoder: &decoder, block: &lhView, ctxRun: &ctxRunLH, ctxMag: &ctxMagLH)
                
                var hhView = BlockView(base: hhBase, width: half, height: half, stride: 16)
                try blockDecode8(decoder: &decoder, block: &hhView, ctxRun: &ctxRunHH, ctxMag: &ctxMagHH)
            case .split4(let tl, let tr, let bl, let br):
                if tl {
                    var hl = BlockView(base: hlBase, width: 4, height: 4, stride: 16)
                    var lh = BlockView(base: lhBase, width: 4, height: 4, stride: 16)
                    var hh = BlockView(base: hhBase, width: 4, height: 4, stride: 16)
                    try blockDecode4(decoder: &decoder, block: &hl, ctxRun: &ctxRunHL, ctxMag: &ctxMagHL)
                    try blockDecode4(decoder: &decoder, block: &lh, ctxRun: &ctxRunLH, ctxMag: &ctxMagLH)
                    try blockDecode4(decoder: &decoder, block: &hh, ctxRun: &ctxRunHH, ctxMag: &ctxMagHH)
                }
                if tr {
                    var hl = BlockView(base: hlBase.advanced(by: 4), width: 4, height: 4, stride: 16)
                    var lh = BlockView(base: lhBase.advanced(by: 4), width: 4, height: 4, stride: 16)
                    var hh = BlockView(base: hhBase.advanced(by: 4), width: 4, height: 4, stride: 16)
                    try blockDecode4(decoder: &decoder, block: &hl, ctxRun: &ctxRunHL, ctxMag: &ctxMagHL)
                    try blockDecode4(decoder: &decoder, block: &lh, ctxRun: &ctxRunLH, ctxMag: &ctxMagLH)
                    try blockDecode4(decoder: &decoder, block: &hh, ctxRun: &ctxRunHH, ctxMag: &ctxMagHH)
                }
                if bl {
                    var hl = BlockView(base: hlBase.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
                    var lh = BlockView(base: lhBase.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
                    var hh = BlockView(base: hhBase.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
                    try blockDecode4(decoder: &decoder, block: &hl, ctxRun: &ctxRunHL, ctxMag: &ctxMagHL)
                    try blockDecode4(decoder: &decoder, block: &lh, ctxRun: &ctxRunLH, ctxMag: &ctxMagLH)
                    try blockDecode4(decoder: &decoder, block: &hh, ctxRun: &ctxRunHH, ctxMag: &ctxMagHH)
                }
                if br {
                    var hl = BlockView(base: hlBase.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
                    var lh = BlockView(base: lhBase.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
                    var hh = BlockView(base: hhBase.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
                    try blockDecode4(decoder: &decoder, block: &hl, ctxRun: &ctxRunHL, ctxMag: &ctxMagHL)
                    try blockDecode4(decoder: &decoder, block: &lh, ctxRun: &ctxRunLH, ctxMag: &ctxMagLH)
                    try blockDecode4(decoder: &decoder, block: &hh, ctxRun: &ctxRunHH, ctxMag: &ctxMagHH)
                }
            }
        }
    }

    return blocks
}

@inline(__always)
func decodePlaneSubbands8(data: [UInt8], blockCount: Int) throws -> [Block2D] {
    var blocks: [Block2D] = []
    blocks.reserveCapacity(blockCount)
    for _ in 0..<blockCount {
        blocks.append(Block2D(width: 8, height: 8))
    }
    
    let flagsByteCount = (blockCount * 2 + 7) / 8
    guard flagsByteCount <= data.count else { throw DecodeError.insufficientData }
    let flagsData = Array(data[0..<flagsByteCount])
    let dataSlice = Array(data[flagsByteCount...])
    
    var brFlags = CABACBitReader(data: flagsData)
    var nonZeroIndices: [Int] = []
    nonZeroIndices.reserveCapacity(blockCount)
    for i in 0..<blockCount {
        let isZero = try brFlags.readBit()
        let _ = try brFlags.readBit()
        if isZero == 0 {
            nonZeroIndices.append(i)
        }
    }
    
    var decoder = try CABACDecoder(data: dataSlice)
    
    let half = 8 / 2
    
    var ctxRunHL = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagHL = [ContextModel](repeating: ContextModel(), count: 64)
    
    var ctxRunLH = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagLH = [ContextModel](repeating: ContextModel(), count: 64)
    
    var ctxRunHH = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagHH = [ContextModel](repeating: ContextModel(), count: 64)

    for i in nonZeroIndices {
        try blocks[i].withView { view in
            var hlView = BlockView(base: view.base.advanced(by: half), width: half, height: half, stride: 8)
            try blockDecode4(decoder: &decoder, block: &hlView, ctxRun: &ctxRunHL, ctxMag: &ctxMagHL)
            
            var lhView = BlockView(base: view.base.advanced(by: half * 8), width: half, height: half, stride: 8)
            try blockDecode4(decoder: &decoder, block: &lhView, ctxRun: &ctxRunLH, ctxMag: &ctxMagLH)
            
            var hhView = BlockView(base: view.base.advanced(by: half * 8 + half), width: half, height: half, stride: 8)
            try blockDecode4(decoder: &decoder, block: &hhView, ctxRun: &ctxRunHH, ctxMag: &ctxMagHH)
        }
    }

    return blocks
}

@inline(__always)
func decodePlaneBaseSubbands8(data: [UInt8], blockCount: Int) throws -> [Block2D] {
    var blocks: [Block2D] = []
    blocks.reserveCapacity(blockCount)
    for _ in 0..<blockCount {
        blocks.append(Block2D(width: 8, height: 8))
    }
    
    let flagsByteCount = (blockCount * 2 + 7) / 8
    guard flagsByteCount <= data.count else { throw DecodeError.insufficientData }
    let flagsData = Array(data[0..<flagsByteCount])
    let dataSlice = Array(data[flagsByteCount...])
    
    var brFlags = CABACBitReader(data: flagsData)
    var nonZeroIndices: [Int] = []
    nonZeroIndices.reserveCapacity(blockCount)
    for i in 0..<blockCount {
        let isZero = try brFlags.readBit()
        let _ = try brFlags.readBit()
        if isZero == 0 {
            nonZeroIndices.append(i)
        }
    }
    
    var decoder = try CABACDecoder(data: dataSlice)
    
    let half = 8 / 2
    
    var ctxRunLL = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagLL = [ContextModel](repeating: ContextModel(), count: 64)

    var ctxRunHL = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagHL = [ContextModel](repeating: ContextModel(), count: 64)
    
    var ctxRunLH = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagLH = [ContextModel](repeating: ContextModel(), count: 64)
    
    var ctxRunHH = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagHH = [ContextModel](repeating: ContextModel(), count: 64)

    var lastVal: Int16 = 0
    var nzCur = 0
    let nzCount = nonZeroIndices.count
    for i in 0..<blockCount {
        if nzCur < nzCount && nonZeroIndices[nzCur] == i {
            nzCur += 1
            try blocks[i].withView { view in
                var llView = BlockView(base: view.base, width: half, height: half, stride: 8)
                try blockDecodeDPCM4(decoder: &decoder, block: &llView, lastVal: &lastVal, ctxRun: &ctxRunLL, ctxMag: &ctxMagLL)
                
                var hlView = BlockView(base: view.base.advanced(by: half), width: half, height: half, stride: 8)
                try blockDecode4(decoder: &decoder, block: &hlView, ctxRun: &ctxRunHL, ctxMag: &ctxMagHL)
                
                var lhView = BlockView(base: view.base.advanced(by: half * 8), width: half, height: half, stride: 8)
                try blockDecode4(decoder: &decoder, block: &lhView, ctxRun: &ctxRunLH, ctxMag: &ctxMagLH)
                
                var hhView = BlockView(base: view.base.advanced(by: half * 8 + half), width: half, height: half, stride: 8)
                try blockDecode4(decoder: &decoder, block: &hhView, ctxRun: &ctxRunHH, ctxMag: &ctxMagHH)
            }
        } else {
            lastVal = 0
        }
    }

    return blocks
}

enum DecodeTaskBase32 {
    case skip
    case decode16
    case split8(Bool, Bool, Bool, Bool)
}

@inline(__always)
func decodePlaneBaseSubbands32(data: [UInt8], blockCount: Int) throws -> [Block2D] {
    var blocks: [Block2D] = []
    blocks.reserveCapacity(blockCount)
    for _ in 0..<blockCount {
        blocks.append(Block2D(width: 32, height: 32))
    }
    
    var brFlags = CABACBitReader(data: data)
    var tasks: [(Int, DecodeTaskBase32)] = []
    tasks.reserveCapacity(blockCount)
    for i in 0..<blockCount {
        let isZero = try brFlags.readBit()
        if isZero == 1 {
            tasks.append((i, .skip))
        } else {
            let mbType = try brFlags.readBit()
            if mbType == 1 {
                let tlZero = try brFlags.readBit()
                if tlZero == 0 { let _ = try brFlags.readBit() }
                
                let trZero = try brFlags.readBit()
                if trZero == 0 { let _ = try brFlags.readBit() }
                
                let blZero = try brFlags.readBit()
                if blZero == 0 { let _ = try brFlags.readBit() }
                
                let brZero = try brFlags.readBit()
                if brZero == 0 { let _ = try brFlags.readBit() }
                
                tasks.append((i, .split8(tlZero == 0, trZero == 0, blZero == 0, brZero == 0)))
            } else {
                tasks.append((i, .decode16))
            }
        }
    }
    
    let consumed = brFlags.consumedBytes
    guard consumed <= data.count else { throw DecodeError.insufficientData }
    let dataSlice = Array(data[consumed...])
    
    var decoder = try CABACDecoder(data: dataSlice)
    
    let half = 32 / 2
    
    var ctxRunLL = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagLL = [ContextModel](repeating: ContextModel(), count: 64)

    var ctxRunHL = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagHL = [ContextModel](repeating: ContextModel(), count: 64)
    
    var ctxRunLH = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagLH = [ContextModel](repeating: ContextModel(), count: 64)
    
    var ctxRunHH = [ContextModel](repeating: ContextModel(), count: 64)
    var ctxMagHH = [ContextModel](repeating: ContextModel(), count: 64)

    var lastVal: Int16 = 0
    for (i, task) in tasks {
        try blocks[i].withView { view in
            let llBase = view.base
            let hlBase = view.base.advanced(by: half)
            let lhBase = view.base.advanced(by: half * 32)
            let hhBase = view.base.advanced(by: half * 32 + half)
            
            switch task {
            case .skip:
                lastVal = 0
            case .decode16:
                var llView = BlockView(base: llBase, width: half, height: half, stride: 32)
                try blockDecodeDPCM16(decoder: &decoder, block: &llView, lastVal: &lastVal, ctxRun: &ctxRunLL, ctxMag: &ctxMagLL)
                
                var hlView = BlockView(base: hlBase, width: half, height: half, stride: 32)
                try blockDecode16(decoder: &decoder, block: &hlView, ctxRun: &ctxRunHL, ctxMag: &ctxMagHL)
                
                var lhView = BlockView(base: lhBase, width: half, height: half, stride: 32)
                try blockDecode16(decoder: &decoder, block: &lhView, ctxRun: &ctxRunLH, ctxMag: &ctxMagLH)
                
                var hhView = BlockView(base: hhBase, width: half, height: half, stride: 32)
                try blockDecode16(decoder: &decoder, block: &hhView, ctxRun: &ctxRunHH, ctxMag: &ctxMagHH)
            case .split8(let tl, let tr, let bl, let br):
                if tl {
                    var ll = BlockView(base: llBase, width: 8, height: 8, stride: 32)
                    var hl = BlockView(base: hlBase, width: 8, height: 8, stride: 32)
                    var lh = BlockView(base: lhBase, width: 8, height: 8, stride: 32)
                    var hh = BlockView(base: hhBase, width: 8, height: 8, stride: 32)
                    try blockDecodeDPCM8(decoder: &decoder, block: &ll, lastVal: &lastVal, ctxRun: &ctxRunLL, ctxMag: &ctxMagLL)
                    try blockDecode8(decoder: &decoder, block: &hl, ctxRun: &ctxRunHL, ctxMag: &ctxMagHL)
                    try blockDecode8(decoder: &decoder, block: &lh, ctxRun: &ctxRunLH, ctxMag: &ctxMagLH)
                    try blockDecode8(decoder: &decoder, block: &hh, ctxRun: &ctxRunHH, ctxMag: &ctxMagHH)
                }
                if tr {
                    var ll = BlockView(base: llBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    var hl = BlockView(base: hlBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    var lh = BlockView(base: lhBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    var hh = BlockView(base: hhBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    try blockDecodeDPCM8(decoder: &decoder, block: &ll, lastVal: &lastVal, ctxRun: &ctxRunLL, ctxMag: &ctxMagLL)
                    try blockDecode8(decoder: &decoder, block: &hl, ctxRun: &ctxRunHL, ctxMag: &ctxMagHL)
                    try blockDecode8(decoder: &decoder, block: &lh, ctxRun: &ctxRunLH, ctxMag: &ctxMagLH)
                    try blockDecode8(decoder: &decoder, block: &hh, ctxRun: &ctxRunHH, ctxMag: &ctxMagHH)
                }
                if bl {
                    var ll = BlockView(base: llBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    var hl = BlockView(base: hlBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    var lh = BlockView(base: lhBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    var hh = BlockView(base: hhBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    try blockDecodeDPCM8(decoder: &decoder, block: &ll, lastVal: &lastVal, ctxRun: &ctxRunLL, ctxMag: &ctxMagLL)
                    try blockDecode8(decoder: &decoder, block: &hl, ctxRun: &ctxRunHL, ctxMag: &ctxMagHL)
                    try blockDecode8(decoder: &decoder, block: &lh, ctxRun: &ctxRunLH, ctxMag: &ctxMagLH)
                    try blockDecode8(decoder: &decoder, block: &hh, ctxRun: &ctxRunHH, ctxMag: &ctxMagHH)
                }
                if br {
                    var ll = BlockView(base: llBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    var hl = BlockView(base: hlBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    var lh = BlockView(base: lhBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    var hh = BlockView(base: hhBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    try blockDecodeDPCM8(decoder: &decoder, block: &ll, lastVal: &lastVal, ctxRun: &ctxRunLL, ctxMag: &ctxMagLL)
                    try blockDecode8(decoder: &decoder, block: &hl, ctxRun: &ctxRunHL, ctxMag: &ctxMagHL)
                    try blockDecode8(decoder: &decoder, block: &lh, ctxRun: &ctxRunLH, ctxMag: &ctxMagLH)
                    try blockDecode8(decoder: &decoder, block: &hh, ctxRun: &ctxRunHH, ctxMag: &ctxMagHH)
                }
            }
        }
    }

    return blocks
}

@inline(__always)
func readUInt8FromBytes(_ r: [UInt8], offset: inout Int) throws -> UInt8 {
    guard (offset + 1) <= r.count else { throw DecodeError.insufficientData }
    let val = r[offset]
    offset += 1
    return val
}

@inline(__always)
func readUInt16BEFromBytes(_ r: [UInt8], offset: inout Int) throws -> UInt16 {
    guard (offset + 2) <= r.count else { throw DecodeError.insufficientData }
    let val = (UInt16(r[offset]) << 8) | UInt16(r[offset + 1])
    offset += 2
    return val
}

@inline(__always)
func readUInt32BEFromBytes(_ r: [UInt8], offset: inout Int) throws -> UInt32 {
    guard (offset + 4) <= r.count else { throw DecodeError.insufficientData }
    let val = (UInt32(r[offset]) << 24) | (UInt32(r[offset + 1]) << 16) | (UInt32(r[offset + 2]) << 8) | UInt32(r[offset + 3])
    offset += 4
    return val
}

@inline(__always)
func readBlockFromBytes(_ r: [UInt8], offset: inout Int) throws -> [UInt8] {
    let len = try readUInt16BEFromBytes(r, offset: &offset)
    let intLen = Int(len)
    guard (offset + intLen) <= r.count else { throw DecodeError.invalidBlockData }
    let block = Array(r[offset..<(offset + intLen)])
    offset += intLen
    return block
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
    let qtY = QuantizationTable(baseStep: Int(try readUInt8FromBytes(r, offset: &offset)))
    let qtC = QuantizationTable(baseStep: Int(try readUInt8FromBytes(r, offset: &offset)))
    
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
    let yBlocks = try decodePlaneSubbands32(data: bufY, blockCount: rowCountY * colCountY)
    
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let rowCountCb = (cbDy + 32 - 1) / 32
    let colCountCb = (cbDx + 32 - 1) / 32
    let cbBlocks = try decodePlaneSubbands32(data: bufCb, blockCount: rowCountCb * colCountCb)
    
    let rowCountCr = (cbDy + 32 - 1) / 32
    let colCountCr = (cbDx + 32 - 1) / 32
    let crBlocks = try decodePlaneSubbands32(data: bufCr, blockCount: rowCountCr * colCountCr)
    
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
                            dequantizeMidSignedMapping(&hlView, qt: qtY)
                            dequantizeMidSignedMapping(&lhView, qt: qtY)
                            dequantizeHighSignedMapping(&hhView, qt: qtY)
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
                            dequantizeMidSignedMapping(&hlView, qt: qtC)
                            dequantizeMidSignedMapping(&lhView, qt: qtC)
                            dequantizeHighSignedMapping(&hhView, qt: qtC)
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
                            dequantizeMidSignedMapping(&hlView, qt: qtC)
                            dequantizeMidSignedMapping(&lhView, qt: qtC)
                            dequantizeHighSignedMapping(&hhView, qt: qtC)
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
    let qtY = QuantizationTable(baseStep: Int(try readUInt8FromBytes(r, offset: &offset)))
    let qtC = QuantizationTable(baseStep: Int(try readUInt8FromBytes(r, offset: &offset)))
    
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
    let yBlocks = try decodePlaneSubbands16(data: bufY, blockCount: rowCountY * colCountY)
    
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let rowCountCb = (cbDy + 16 - 1) / 16
    let colCountCb = (cbDx + 16 - 1) / 16
    let cbBlocks = try decodePlaneSubbands16(data: bufCb, blockCount: rowCountCb * colCountCb)
    
    let rowCountCr = (cbDy + 16 - 1) / 16
    let colCountCr = (cbDx + 16 - 1) / 16
    let crBlocks = try decodePlaneSubbands16(data: bufCr, blockCount: rowCountCr * colCountCr)
    
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
                            dequantizeMidSignedMapping(&hlView, qt: qtY)
                            dequantizeMidSignedMapping(&lhView, qt: qtY)
                            dequantizeHighSignedMapping(&hhView, qt: qtY)
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
                            dequantizeMidSignedMapping(&hlView, qt: qtC)
                            dequantizeMidSignedMapping(&lhView, qt: qtC)
                            dequantizeHighSignedMapping(&hhView, qt: qtC)
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
                            dequantizeMidSignedMapping(&hlView, qt: qtC)
                            dequantizeMidSignedMapping(&lhView, qt: qtC)
                            dequantizeHighSignedMapping(&hhView, qt: qtC)
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
    let qtY = QuantizationTable(baseStep: Int(try readUInt8FromBytes(r, offset: &offset)))
    let qtC = QuantizationTable(baseStep: Int(try readUInt8FromBytes(r, offset: &offset)))
    
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
                            dequantizeLow(&llView, qt: qtY)
                            dequantizeMidSignedMapping(&hlView, qt: qtY)
                            dequantizeMidSignedMapping(&lhView, qt: qtY)
                            dequantizeHighSignedMapping(&hhView, qt: qtY)
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
                            dequantizeLow(&llView, qt: qtC)
                            dequantizeMidSignedMapping(&hlView, qt: qtC)
                            dequantizeMidSignedMapping(&lhView, qt: qtC)
                            dequantizeHighSignedMapping(&hhView, qt: qtC)
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
                            dequantizeLow(&llView, qt: qtC)
                            dequantizeMidSignedMapping(&hlView, qt: qtC)
                            dequantizeMidSignedMapping(&lhView, qt: qtC)
                            dequantizeHighSignedMapping(&hhView, qt: qtC)
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

public struct DecodeOptions: Sendable {
    public var maxLayer: Int
    public var maxFrames: Int
    
    public init(maxLayer: Int = 2, maxFrames: Int = 4) {
        self.maxLayer = maxLayer
        self.maxFrames = maxFrames
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
        case [0x56, 0x45, 0x56, 0x49]:
            let len = Int(try readUInt32BEFromBytes(data, offset: &offset))
            guard (offset + len) <= data.count else { throw DecodeError.insufficientData }
            let chunk = Array(data[offset..<(offset + len)])
            offset += len
            
            let img16 = try await decodeSpatialLayers(r: chunk, maxLayer: opts.maxLayer)
            let pd = PlaneData420(img16: img16)
            out.append(pd.toYCbCr())
            prevReconstructed = pd

        case [0x56, 0x45, 0x56, 0x50]:
            let mvsCount = Int(try readUInt32BEFromBytes(data, offset: &offset))
            let mvDataLen = Int(try readUInt32BEFromBytes(data, offset: &offset))
            var mvs = MotionVectors(count: mvsCount)
            guard (offset + mvDataLen) <= data.count else { throw DecodeError.insufficientData }

            let mvData = Array(data[offset..<(offset + mvDataLen)])
            offset += mvDataLen
            var mvBr = try CABACDecoder(data: mvData)
            var ctxDx = ContextModel()

            let mbSize = 64
            // We need width to compute mbCols. We can infer width from previous frame.
            guard let prevWidth = prevReconstructed?.width else { throw DecodeError.invalidHeader }
            let mbCols = (prevWidth + mbSize - 1) / mbSize

            for i in 0..<mvsCount {
                let mbX = i % mbCols
                let mbY = i / mbCols
                let pmv = calculatePMV(mvs: mvs, mbX: mbX, mbY: mbY, mbCols: mbCols)

                let isSig = try mvBr.decodeBin(ctx: &ctxDx)
                if isSig == 0 {
                    mvs.vectors[i] = SIMD2(Int16(pmv.dx), Int16(pmv.dy))
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

                    mvs.vectors[i] = SIMD2(Int16(mvdX + pmv.dx), Int16(mvdY + pmv.dy))
                }
            }
            
            let len = Int(try readUInt32BEFromBytes(data, offset: &offset))
            guard (offset + len) <= data.count else { throw DecodeError.insufficientData }
            let chunk = Array(data[offset..<(offset + len)])
            offset += len
            
            let img16 = try await decodeSpatialLayers(r: chunk, maxLayer: opts.maxLayer)
            let residual = PlaneData420(img16: img16)
            
            if let prev = prevReconstructed {
                let predicted = await applyMBME(prev: prev, mvs: mvs)
                let curr = await addPlanes(residual: residual, predicted: predicted)
                out.append(curr.toYCbCr())
                prevReconstructed = curr
            } else {
                out.append(residual.toYCbCr())
            }
            
        case [0x56, 0x45, 0x4F, 0x49]: // VEOI
            let len = Int(try readUInt32BEFromBytes(data, offset: &offset))
            guard (offset + len) <= data.count else { throw DecodeError.insufficientData }
            let chunk = Array(data[offset..<(offset + len)])
            offset += len
            
            let img16 = try await decodeBase32(r: chunk, layer: 0)
            let pd = PlaneData420(img16: img16)
            out.append(pd.toYCbCr())
            prevReconstructed = pd

        case [0x56, 0x45, 0x4F, 0x50]: // VEOP
            let mvsCount = Int(try readUInt32BEFromBytes(data, offset: &offset))
            let mvDataLen = Int(try readUInt32BEFromBytes(data, offset: &offset))
            var mvs = MotionVectors(count: mvsCount)
            guard (offset + mvDataLen) <= data.count else { throw DecodeError.insufficientData }

            let mvData = Array(data[offset..<(offset + mvDataLen)])
            offset += mvDataLen
            var mvBr = try CABACDecoder(data: mvData)
            var ctxDx = ContextModel()

            let mbSize = 64
            // We need width to compute mbCols. We can infer width from previous frame.
            guard let prevWidth = prevReconstructed?.width else { throw DecodeError.invalidHeader }
            let mbCols = (prevWidth + mbSize - 1) / mbSize

            for i in 0..<mvsCount {
                let mbX = i % mbCols
                let mbY = i / mbCols
                let pmv = calculatePMV(mvs: mvs, mbX: mbX, mbY: mbY, mbCols: mbCols)

                let isSig = try mvBr.decodeBin(ctx: &ctxDx)
                if isSig == 0 {
                    mvs.vectors[i] = SIMD2(Int16(pmv.dx), Int16(pmv.dy))
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

                    mvs.vectors[i] = SIMD2(Int16(mvdX + pmv.dx), Int16(mvdY + pmv.dy))
                }
            }
            
            let len = Int(try readUInt32BEFromBytes(data, offset: &offset))
            guard (offset + len) <= data.count else { throw DecodeError.insufficientData }
            let chunk = Array(data[offset..<(offset + len)])
            offset += len
            
            let img16 = try await decodeBase32(r: chunk, layer: 0)
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
    let qtY = QuantizationTable(baseStep: Int(try readUInt8FromBytes(r, offset: &offset)))
    let qtC = QuantizationTable(baseStep: Int(try readUInt8FromBytes(r, offset: &offset)))
    
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
                            dequantizeLow(&llView, qt: qtY)
                            dequantizeMidSignedMapping(&hlView, qt: qtY)
                            dequantizeMidSignedMapping(&lhView, qt: qtY)
                            dequantizeHighSignedMapping(&hhView, qt: qtY)
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
                            dequantizeLow(&llView, qt: qtC)
                            dequantizeMidSignedMapping(&hlView, qt: qtC)
                            dequantizeMidSignedMapping(&lhView, qt: qtC)
                            dequantizeHighSignedMapping(&hhView, qt: qtC)
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
                            dequantizeLow(&llView, qt: qtC)
                            dequantizeMidSignedMapping(&hlView, qt: qtC)
                            dequantizeMidSignedMapping(&lhView, qt: qtC)
                            dequantizeHighSignedMapping(&hhView, qt: qtC)
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
    
    return sub
}

@inline(__always)
public func decodeOne(data: [UInt8]) async throws -> [YCbCrImage] {
    return try await decode(data: data, opts: DecodeOptions(maxLayer: 0))
}

#else
public func decode(data: [UInt8], opts: DecodeOptions = DecodeOptions()) async throws -> [YCbCrImage] {
    throw DecodeError.unsupportedArchitecture
}
public func decodeOne(data: [UInt8]) async throws -> [YCbCrImage] {
    throw DecodeError.unsupportedArchitecture
}
#endif
