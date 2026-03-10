// MARK: - Decode Error

public enum DecodeError: Error {
    case eof
    case insufficientData
    case invalidBlockData
    case invalidHeader
    case invalidLayerNumber
    case noDataProvided
}

@inline(__always)
func decodeSpatialLayers(r: [UInt8], maxLayer: Int, predictedPd: PlaneData420? = nil) async throws -> Image16 {
    var offset = 0
    guard (offset + 5) <= r.count else { throw DecodeError.insufficientData }
    let magic = Array(r[offset..<(offset + 4)])
    let version = r[offset + 4]
    offset += 5

    guard (magic == [0x56, 0x45, 0x56, 0x43] && version == 0x03) else {
        throw DecodeError.invalidHeader
    }

    let dx = Int(try readUInt16BEFromBytes(r, offset: &offset))
    let dy = Int(try readUInt16BEFromBytes(r, offset: &offset))
    let qtY = QuantizationTable(baseStep: Int(try readUInt8FromBytes(r, offset: &offset)))
    let qtC = QuantizationTable(baseStep: Int(try readUInt8FromBytes(r, offset: &offset)))

    let bufYLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    let bufY = Array(r[offset..<(offset + bufYLen)])
    offset += bufYLen

    let bufCbLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    let bufCb = Array(r[offset..<(offset + bufCbLen)])
    offset += bufCbLen

    let bufCrLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    let bufCr = Array(r[offset..<(offset + bufCrLen)])
    offset += bufCrLen

    var cdY = CABACDecoder(data: bufY)
    var ctxsY = PlaneCABACContexts()
    var cdCb = CABACDecoder(data: bufCb)
    var ctxsCb = PlaneCABACContexts()
    var cdCr = CABACDecoder(data: bufCr)
    var ctxsCr = PlaneCABACContexts()

    // Layer 0
    let row0Y = ((dy / 4) + 7) / 8
    let col0Y = ((dx / 4) + 7) / 8
    let blocks0Y = try decodePlaneBaseSubbands(cd: &cdY, ctxFlags: &ctxsY.ctxFlagsL0, ctxZeroLL: &ctxsY.ctxZeroLL0, ctxG1LL: &ctxsY.ctxG1LL0, ctxZeroHL: &ctxsY.ctxZeroHL0, ctxG1HL: &ctxsY.ctxG1HL0, ctxZeroLH: &ctxsY.ctxZeroLH0, ctxG1LH: &ctxsY.ctxG1LH0, ctxZeroHH: &ctxsY.ctxZeroHH0, ctxG1HH: &ctxsY.ctxG1HH0, size: 8, blockCount: (row0Y * col0Y))
    let row0C = ((((dy + 1) / 2 / 4)) + 7) / 8
    let col0C = ((((dx + 1) / 2 / 4)) + 7) / 8
    let blocks0Cb = try decodePlaneBaseSubbands(cd: &cdCb, ctxFlags: &ctxsCb.ctxFlagsL0, ctxZeroLL: &ctxsCb.ctxZeroLL0, ctxG1LL: &ctxsCb.ctxG1LL0, ctxZeroHL: &ctxsCb.ctxZeroHL0, ctxG1HL: &ctxsCb.ctxG1HL0, ctxZeroLH: &ctxsCb.ctxZeroLH0, ctxG1LH: &ctxsCb.ctxG1LH0, ctxZeroHH: &ctxsCb.ctxZeroHH0, ctxG1HH: &ctxsCb.ctxG1HH0, size: 8, blockCount: (row0C * col0C))
    let blocks0Cr = try decodePlaneBaseSubbands(cd: &cdCr, ctxFlags: &ctxsCr.ctxFlagsL0, ctxZeroLL: &ctxsCr.ctxZeroLL0, ctxG1LL: &ctxsCr.ctxG1LL0, ctxZeroHL: &ctxsCr.ctxZeroHL0, ctxG1HL: &ctxsCr.ctxG1HL0, ctxZeroLH: &ctxsCr.ctxZeroLH0, ctxG1LH: &ctxsCr.ctxG1LH0, ctxZeroHH: &ctxsCr.ctxZeroHH0, ctxG1HH: &ctxsCr.ctxG1HH0, size: 8, blockCount: (row0C * col0C))

    // Layer 1
    let row1Y = ((dy / 2) + 15) / 16
    let col1Y = ((dx / 2) + 15) / 16
    let blocks1Y = try decodePlaneSubbands(cd: &cdY, ctxFlags: &ctxsY.ctxFlagsL1, ctxZeroHL: &ctxsY.ctxZeroHL1, ctxG1HL: &ctxsY.ctxG1HL1, ctxZeroLH: &ctxsY.ctxZeroLH1, ctxG1LH: &ctxsY.ctxG1LH1, ctxZeroHH: &ctxsY.ctxZeroHH1, ctxG1HH: &ctxsY.ctxG1HH1, size: 16, blockCount: (row1Y * col1Y))
    let row1C = ((((dy + 1) / 2 / 2)) + 15) / 16
    let col1C = ((((dx + 1) / 2 / 2)) + 15) / 16
    let blocks1Cb = try decodePlaneSubbands(cd: &cdCb, ctxFlags: &ctxsCb.ctxFlagsL1, ctxZeroHL: &ctxsCb.ctxZeroHL1, ctxG1HL: &ctxsCb.ctxG1HL1, ctxZeroLH: &ctxsCb.ctxZeroLH1, ctxG1LH: &ctxsCb.ctxG1LH1, ctxZeroHH: &ctxsCb.ctxZeroHH1, ctxG1HH: &ctxsCb.ctxG1HH1, size: 16, blockCount: (row1C * col1C))
    let blocks1Cr = try decodePlaneSubbands(cd: &cdCr, ctxFlags: &ctxsCr.ctxFlagsL1, ctxZeroHL: &ctxsCr.ctxZeroHL1, ctxG1HL: &ctxsCr.ctxG1HL1, ctxZeroLH: &ctxsCr.ctxZeroLH1, ctxG1LH: &ctxsCr.ctxG1LH1, ctxZeroHH: &ctxsCr.ctxZeroHH1, ctxG1HH: &ctxsCr.ctxG1HH1, size: 16, blockCount: (row1C * col1C))

    // Layer 2
    let row2Y = (dy + 31) / 32
    let col2Y = (dx + 31) / 32
    let blocks2Y = try decodePlaneSubbands(cd: &cdY, ctxFlags: &ctxsY.ctxFlagsL2, ctxZeroHL: &ctxsY.ctxZeroHL2, ctxG1HL: &ctxsY.ctxG1HL2, ctxZeroLH: &ctxsY.ctxZeroLH2, ctxG1LH: &ctxsY.ctxG1LH2, ctxZeroHH: &ctxsY.ctxZeroHH2, ctxG1HH: &ctxsY.ctxG1HH2, size: 32, blockCount: (row2Y * col2Y))
    let row2C = (((dy + 1) / 2) + 31) / 32
    let col2C = (((dx + 1) / 2) + 31) / 32
    let blocks2Cb = try decodePlaneSubbands(cd: &cdCb, ctxFlags: &ctxsCb.ctxFlagsL2, ctxZeroHL: &ctxsCb.ctxZeroHL2, ctxG1HL: &ctxsCb.ctxG1HL2, ctxZeroLH: &ctxsCb.ctxZeroLH2, ctxG1LH: &ctxsCb.ctxG1LH2, ctxZeroHH: &ctxsCb.ctxZeroHH2, ctxG1HH: &ctxsCb.ctxG1HH2, size: 32, blockCount: (row2C * col2C))
    let blocks2Cr = try decodePlaneSubbands(cd: &cdCr, ctxFlags: &ctxsCr.ctxFlagsL2, ctxZeroHL: &ctxsCr.ctxZeroHL2, ctxG1HL: &ctxsCr.ctxG1HL2, ctxZeroLH: &ctxsCr.ctxZeroLH2, ctxG1LH: &ctxsCr.ctxG1LH2, ctxZeroHH: &ctxsCr.ctxZeroHH2, ctxG1HH: &ctxsCr.ctxG1HH2, size: 32, blockCount: (row2C * col2C))

    // L0
    var img0 = Image16(width: (dx / 4), height: (dy / 4))
    await reconstructPlaneParallel(&img0, blocks: blocks0Y, blocksCb: blocks0Cb, blocksCr: blocks0Cr, size: 8, qtY: qtY, qtC: qtC, isBase: true, prev: nil)
    
    // L1
    var img1 = Image16(width: (dx / 2), height: (dy / 2))
    await reconstructPlaneParallel(&img1, blocks: blocks1Y, blocksCb: blocks1Cb, blocksCr: blocks1Cr, size: 16, qtY: qtY, qtC: qtC, isBase: false, prev: img0)
    
    // L2
    var img2 = Image16(width: dx, height: dy)
    await reconstructPlaneParallel(&img2, blocks: blocks2Y, blocksCb: blocks2Cb, blocksCr: blocks2Cr, size: 32, qtY: qtY, qtC: qtC, isBase: false, prev: img1)

    return img2
}

func reconstructPlaneParallel(_ img: inout Image16, blocks: [Block2D], blocksCb: [Block2D], blocksCr: [Block2D], size: Int, qtY: QuantizationTable, qtC: QuantizationTable, isBase: Bool, prev: Image16?) async {
    let dx = img.width
    let colCount = ((dx + size - 1) / size)
    let half = (size / 2)

    // Process planes sequentially to avoid complex captures and copying
    // But within each plane, we could potentially parallelize (not implemented here for simplicity/safety)

    // Y
    for i in blocks.indices {
        let r = (i / colCount)
        let c = (i % colCount)
        var block = blocks[i]
        if let p = prev {
            var ll = p.getY(x: (c * half), y: (r * half), size: half)
            ll.withView { srcView in
                block.withView { destView in
                    for yi in 0..<half {
                        destView.rowPointer(y: yi).update(from: srcView.rowPointer(y: yi), count: half)
                    }
                }
            }
        }
        block.withView { view in
            if isBase { dequantizeLow(&view, qt: qtY) }
            let base = view.base
            var hl = BlockView(base: base.advanced(by: half), width: half, height: half, stride: size)
            var lh = BlockView(base: base.advanced(by: (half * size)), width: half, height: half, stride: size)
            var hh = BlockView(base: base.advanced(by: (half * size + half)), width: half, height: half, stride: size)
            dequantizeMidSignedMapping(&hl, qt: qtY)
            dequantizeMidSignedMapping(&lh, qt: qtY)
            dequantizeHighSignedMapping(&hh, qt: qtY)
            invDwt2d(&view, size: size)
        }
        img.updateY(data: &block, startX: (c * size), startY: (r * size), size: size)
    }

    let cWidth = ((dx + 1) / 2)
    let colCountC = ((cWidth + size - 1) / size)

    // Cb
    for i in blocksCb.indices {
        let r = (i / colCountC)
        let c = (i % colCountC)
        var block = blocksCb[i]
        if let p = prev {
            var ll = p.getCb(x: (c * half), y: (r * half), size: half)
            ll.withView { srcView in
                block.withView { destView in
                    for yi in 0..<half {
                        destView.rowPointer(y: yi).update(from: srcView.rowPointer(y: yi), count: half)
                    }
                }
            }
        }
        block.withView { view in
            if isBase { dequantizeLow(&view, qt: qtC) }
            let base = view.base
            var hl = BlockView(base: base.advanced(by: half), width: half, height: half, stride: size)
            var lh = BlockView(base: base.advanced(by: (half * size)), width: half, height: half, stride: size)
            var hh = BlockView(base: base.advanced(by: (half * size + half)), width: half, height: half, stride: size)
            dequantizeMidSignedMapping(&hl, qt: qtC)
            dequantizeMidSignedMapping(&lh, qt: qtC)
            dequantizeHighSignedMapping(&hh, qt: qtC)
            invDwt2d(&view, size: size)
        }
        img.updateCb(data: &block, startX: (c * size), startY: (r * size), size: size)
    }

    // Cr
    for i in blocksCr.indices {
        let r = (i / colCountC)
        let c = (i % colCountC)
        var block = blocksCr[i]
        if let p = prev {
            var ll = p.getCr(x: (c * half), y: (r * half), size: half)
            ll.withView { srcView in
                block.withView { destView in
                    for yi in 0..<half {
                        destView.rowPointer(y: yi).update(from: srcView.rowPointer(y: yi), count: half)
                    }
                }
            }
        }
        block.withView { view in
            if isBase { dequantizeLow(&view, qt: qtC) }
            let base = view.base
            var hl = BlockView(base: base.advanced(by: half), width: half, height: half, stride: size)
            var lh = BlockView(base: base.advanced(by: (half * size)), width: half, height: half, stride: size)
            var hh = BlockView(base: base.advanced(by: (half * size + half)), width: half, height: half, stride: size)
            dequantizeMidSignedMapping(&hl, qt: qtC)
            dequantizeMidSignedMapping(&lh, qt: qtC)
            dequantizeHighSignedMapping(&hh, qt: qtC)
            invDwt2d(&view, size: size)
        }
        img.updateCr(data: &block, startX: (c * size), startY: (r * size), size: size)
    }
}

// MARK: - Decode Logic

@inline(__always)
func toInt16(_ u: UInt16) -> Int16 {
    return Int16(bitPattern: (u >> 1) ^ (0 &- (u & 1)))
}

@inline(__always)
func blockDecode(cd: inout CABACDecoder, ctxZero: inout [ContextModel], ctxG1: inout [ContextModel], block: inout BlockView, size: Int) throws {
    let half = size
    for y in 0..<half {
        let ptr = block.rowPointer(y: y)
        for x in 0..<half {
            let pos = min(15, (y + x))
            if (cd.decode(context: &ctxZero[pos]) == 1) {
                ptr[x] = 0
            } else {
                let v = cd.decodeVal(ctxG1: &ctxG1)
                ptr[x] = Int16(bitPattern: UInt16(v + 1))
            }
        }
    }
}

@inline(__always)
func blockDecodeDPCM(cd: inout CABACDecoder, ctxZero: inout [ContextModel], ctxG1: inout [ContextModel], block: inout BlockView, size: Int, lastVal: inout Int16) throws {
    let half = size
    for y in 0..<half {
        let ptr = block.rowPointer(y: y)
        for x in 0..<half {
            let pos = min(15, (y + x))
            let uVal: UInt16
            if (cd.decode(context: &ctxZero[pos]) == 1) {
                uVal = 0
            } else {
                let v = cd.decodeVal(ctxG1: &ctxG1)
                uVal = UInt16(v + 1)
            }

            let diff = toInt16(uVal)
            let predicted: Int16
            if (x == 0 && y == 0) {
                predicted = lastVal
            } else if (y == 0) {
                predicted = ptr[x - 1]
            } else if (x == 0) {
                predicted = block.rowPointer(y: y - 1)[x]
            } else {
                let a = Int(ptr[x - 1])
                let b = Int(block.rowPointer(y: y - 1)[x])
                let c = Int(block.rowPointer(y: y - 1)[x - 1])
                if (c >= max(a, b)) {
                    predicted = Int16(min(a, b))
                } else if (c <= min(a, b)) {
                    predicted = Int16(max(a, b))
                } else {
                    predicted = Int16(a + b - c)
                }
            }
            let val = (diff + predicted)
            ptr[x] = val
        }
    }
    lastVal = block.rowPointer(y: half - 1)[half - 1]
}

@inline(__always)
func decodePlaneSubbands(cd: inout CABACDecoder, ctxFlags: inout ContextModel, ctxZeroHL: inout [ContextModel], ctxG1HL: inout [ContextModel], ctxZeroLH: inout [ContextModel], ctxG1LH: inout [ContextModel], ctxZeroHH: inout [ContextModel], ctxG1HH: inout [ContextModel], size: Int, blockCount: Int) throws -> [Block2D] {
    var blocks: [Block2D] = []
    blocks.reserveCapacity(blockCount)
    for _ in 0..<blockCount {
        blocks.append(Block2D(width: size, height: size))
    }
    
    var nonZeroIndices: [Int] = []
    for i in 0..<blockCount {
        if (cd.decode(context: &ctxFlags) == 0) {
            nonZeroIndices.append(i)
        }
    }
    
    let half = (size / 2)
    for i in nonZeroIndices {
        try blocks[i].withView { view in
            var hlView = BlockView(base: view.base.advanced(by: half), width: half, height: half, stride: size)
            try blockDecode(cd: &cd, ctxZero: &ctxZeroHL, ctxG1: &ctxG1HL, block: &hlView, size: half)
        }
    }
    
    for i in nonZeroIndices {
        try blocks[i].withView { view in
            var lhView = BlockView(base: view.base.advanced(by: (half * size)), width: half, height: half, stride: size)
            try blockDecode(cd: &cd, ctxZero: &ctxZeroLH, ctxG1: &ctxG1LH, block: &lhView, size: half)
        }
    }
    
    for i in nonZeroIndices {
        try blocks[i].withView { view in
            var hhView = BlockView(base: view.base.advanced(by: (half * size + half)), width: half, height: half, stride: size)
            try blockDecode(cd: &cd, ctxZero: &ctxZeroHH, ctxG1: &ctxG1HH, block: &hhView, size: half)
        }
    }

    return blocks
}

@inline(__always)
func decodePlaneBaseSubbands(cd: inout CABACDecoder, ctxFlags: inout ContextModel, ctxZeroLL: inout [ContextModel], ctxG1LL: inout [ContextModel], ctxZeroHL: inout [ContextModel], ctxG1HL: inout [ContextModel], ctxZeroLH: inout [ContextModel], ctxG1LH: inout [ContextModel], ctxZeroHH: inout [ContextModel], ctxG1HH: inout [ContextModel], size: Int, blockCount: Int) throws -> [Block2D] {
    var blocks: [Block2D] = []
    blocks.reserveCapacity(blockCount)
    for _ in 0..<blockCount {
        blocks.append(Block2D(width: size, height: size))
    }
    
    var nonZeroIndices: [Int] = []
    for i in 0..<blockCount {
        if (cd.decode(context: &ctxFlags) == 0) {
            nonZeroIndices.append(i)
        }
    }
    
    let half = (size / 2)
    var lastVal: Int16 = 0
    let nonZeroSet = Set(nonZeroIndices)
    for i in 0..<blockCount {
        if (nonZeroSet.contains(i)) {
            try blocks[i].withView { view in
                var llView = BlockView(base: view.base, width: half, height: half, stride: size)
                try blockDecodeDPCM(cd: &cd, ctxZero: &ctxZeroLL, ctxG1: &ctxG1LL, block: &llView, size: half, lastVal: &lastVal)
            }
        } else {
            lastVal = 0
        }
    }
    
    for i in nonZeroIndices {
        try blocks[i].withView { view in
            var hlView = BlockView(base: view.base.advanced(by: half), width: half, height: half, stride: size)
            try blockDecode(cd: &cd, ctxZero: &ctxZeroHL, ctxG1: &ctxG1HL, block: &hlView, size: half)
        }
    }
    
    for i in nonZeroIndices {
        try blocks[i].withView { view in
            var lhView = BlockView(base: view.base.advanced(by: (half * size)), width: half, height: half, stride: size)
            try blockDecode(cd: &cd, ctxZero: &ctxZeroLH, ctxG1: &ctxG1LH, block: &lhView, size: half)
        }
    }
    
    for i in nonZeroIndices {
        try blocks[i].withView { view in
            var hhView = BlockView(base: view.base.advanced(by: (half * size + half)), width: half, height: half, stride: size)
            try blockDecode(cd: &cd, ctxZero: &ctxZeroHH, ctxG1: &ctxG1HH, block: &hhView, size: half)
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


public struct DecodeOptions: Sendable {
    public var maxLayer: Int = 2
    public var maxFrames: Int = 4
    
    public init(maxLayer: Int = 2, maxFrames: Int = 4) {
        self.maxLayer = maxLayer
        self.maxFrames = maxFrames
    }
}

public func decode(data: [UInt8], opts: DecodeOptions = DecodeOptions()) async throws -> [YCbCrImage] {
    var out: [YCbCrImage] = []
    var offset = 0
    var prevReconstructed: PlaneData420? = nil
    
    while (offset + 4) <= data.count {
        let magic = Array(data[offset..<(offset + 4)])
        offset = (offset + 4)
        
        if (magic == [0x56, 0x45, 0x56, 0x49]) {
            let len = Int(try readUInt32BEFromBytes(data, offset: &offset))
            let chunk = Array(data[offset..<(offset + len)])
            offset = (offset + len)
            
            let img16 = try await decodeSpatialLayers(r: chunk, maxLayer: opts.maxLayer)
            let pd = PlaneData420(img16: img16)
            out.append(pd.toYCbCr())
            prevReconstructed = pd
            
        } else if (magic == [0x56, 0x45, 0x56, 0x50]) {
            let dx = Int(Int16(bitPattern: try readUInt16BEFromBytes(data, offset: &offset)))
            let dy = Int(Int16(bitPattern: try readUInt16BEFromBytes(data, offset: &offset)))
            
            let len = Int(try readUInt32BEFromBytes(data, offset: &offset))
            let chunk = Array(data[offset..<(offset + len)])
            offset = (offset + len)
            
            let img16 = try await decodeSpatialLayers(r: chunk, maxLayer: opts.maxLayer)
            let residual = PlaneData420(img16: img16)
            
            if let prev = prevReconstructed {
                let predicted = await shiftPlane(prev, dx: dx, dy: dy)
                let curr = await addPlanes(residual: residual, predicted: predicted)
                out.append(curr.toYCbCr())
                prevReconstructed = cleanExposedRegion(curr, dx: dx, dy: dy)
            } else {
                out.append(residual.toYCbCr())
            }
        } else {
             throw DecodeError.invalidHeader
        }
    }
    
    return out
}
