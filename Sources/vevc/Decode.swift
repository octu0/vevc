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
func decodeSpatialLayers(r: [UInt8], maxLayer: Int) async throws -> Image16 {
    var offset = 0
    let len0 = try readUInt32BEFromBytes(r, offset: &offset)
    let layer0Data = Array(r[offset..<(offset + Int(len0))])
    offset += Int(len0)
    var current = try await decodeBase(r: layer0Data, layer: 0, size: 8)
    
    if maxLayer >= 1 {
        let len1 = try readUInt32BEFromBytes(r, offset: &offset)
        let layer1Data = Array(r[offset..<(offset + Int(len1))])
        offset += Int(len1)
        current = try await decodeLayer(r: layer1Data, layer: 1, prev: current, size: 16)
    }
    if maxLayer >= 2 {
        let len2 = try readUInt32BEFromBytes(r, offset: &offset)
        let layer2Data = Array(r[offset..<(offset + Int(len2))])
        offset += Int(len2)
        current = try await decodeLayer(r: layer2Data, layer: 2, prev: current, size: 32)
    }
    return current
}



// MARK: - Decode Logic

private let k: UInt8 = 2

@inline(__always)
func toInt16(_ u: UInt16) -> Int16 {
    let s = Int16(bitPattern: (u >> 1))
    let m = (-1 * Int16(bitPattern: (u & 1)))
    return (s ^ m)
}

@inline(__always)
func blockDecode(rr: inout RiceReader, block: inout BlockView, size: Int) throws {
    for y in 0..<size {
        let ptr = block.rowPointer(y: y)
        for x in 0..<size {
            let v = try rr.read(k: k)
            ptr[x] = Int16(bitPattern: v)
        }
    }
}

@inline(__always)
func blockDecodeDPCM(rr: inout RiceReader, block: inout BlockView, size: Int) throws {
    var prevVal: Int16 = 0
    for y in 0..<size {
        let ptr = block.rowPointer(y: y)
        for x in 0..<size {
            let v = try rr.read(k: k)
            let diff = toInt16(v)
            let val = diff + prevVal
            ptr[x] = val
            prevVal = val
        }
    }
}

func decodePlaneSubbands(data: [UInt8], size: Int, blockCount: Int) throws -> [Block2D] {
    var blocks: [Block2D] = []
    blocks.reserveCapacity(blockCount)
    for _ in 0..<blockCount {
        blocks.append(Block2D(width: size, height: size))
    }
    
    let flagsByteCount = (blockCount + 7) / 8
    guard flagsByteCount <= data.count else { throw DecodeError.insufficientData }
    let flagsData = Array(data[0..<flagsByteCount])
    let dataSlice = Array(data[flagsByteCount...])
    
    var brFlags = BitReader(data: flagsData)
    var nonZeroIndices: [Int] = []
    for i in 0..<blockCount {
        if try brFlags.readBit() == 0 {
            nonZeroIndices.append(i)
        }
    }
    
    let brData = BitReader(data: dataSlice)
    var rr = RiceReader(br: brData)
    
    let half = size / 2
    for i in nonZeroIndices {
        try blocks[i].withView { view in
            var hlView = BlockView(base: view.base.advanced(by: half), width: half, height: half, stride: size)
            try blockDecode(rr: &rr, block: &hlView, size: half)
        }
    }
    for i in nonZeroIndices {
        try blocks[i].withView { view in
            var lhView = BlockView(base: view.base.advanced(by: half * size), width: half, height: half, stride: size)
            try blockDecode(rr: &rr, block: &lhView, size: half)
        }
    }
    for i in nonZeroIndices {
        try blocks[i].withView { view in
            var hhView = BlockView(base: view.base.advanced(by: half * size + half), width: half, height: half, stride: size)
            try blockDecode(rr: &rr, block: &hhView, size: half)
        }
    }
    return blocks
}

func decodePlaneBaseSubbands(data: [UInt8], size: Int, blockCount: Int) throws -> [Block2D] {
    var blocks: [Block2D] = []
    blocks.reserveCapacity(blockCount)
    for _ in 0..<blockCount {
        blocks.append(Block2D(width: size, height: size))
    }
    
    let flagsByteCount = (blockCount + 7) / 8
    guard flagsByteCount <= data.count else { throw DecodeError.insufficientData }
    let flagsData = Array(data[0..<flagsByteCount])
    let dataSlice = Array(data[flagsByteCount...])
    
    var brFlags = BitReader(data: flagsData)
    var nonZeroIndices: [Int] = []
    for i in 0..<blockCount {
        if try brFlags.readBit() == 0 {
            nonZeroIndices.append(i)
        }
    }
    
    let brData = BitReader(data: dataSlice)
    var rr = RiceReader(br: brData)
    
    let half = size / 2
    for i in nonZeroIndices {
        try blocks[i].withView { view in
            var llView = BlockView(base: view.base, width: half, height: half, stride: size)
            try blockDecodeDPCM(rr: &rr, block: &llView, size: half)
        }
    }
    for i in nonZeroIndices {
        try blocks[i].withView { view in
            var hlView = BlockView(base: view.base.advanced(by: half), width: half, height: half, stride: size)
            try blockDecode(rr: &rr, block: &hlView, size: half)
        }
    }
    for i in nonZeroIndices {
        try blocks[i].withView { view in
            var lhView = BlockView(base: view.base.advanced(by: half * size), width: half, height: half, stride: size)
            try blockDecode(rr: &rr, block: &lhView, size: half)
        }
    }
    for i in nonZeroIndices {
        try blocks[i].withView { view in
            var hhView = BlockView(base: view.base.advanced(by: half * size + half), width: half, height: half, stride: size)
            try blockDecode(rr: &rr, block: &hhView, size: half)
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
func decodeLayer(r: [UInt8], layer: UInt8, prev: Image16, size: Int) async throws -> Image16 {
    var offset = 0
    
    guard (offset + 5) <= r.count else { throw DecodeError.insufficientData }
    let header = Array(r[offset..<(offset + 5)])
    offset += 5
    
    guard header[0] == 0x56 && header[1] == 0x45 && header[2] == 0x56 && header[3] == 0x43 else { // check 'VEVC'
         throw DecodeError.invalidHeader
    }
    guard header[4] == layer else { // check layer
        throw DecodeError.invalidLayerNumber
    }
    
    let dx = Int(try readUInt16BEFromBytes(r, offset: &offset))
    let dy = Int(try readUInt16BEFromBytes(r, offset: &offset))
    let qt = QuantizationTable(baseStep: Int(try readUInt8FromBytes(r, offset: &offset)))
    
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
    
    let rowCountY = (dy + size - 1) / size
    let colCountY = (dx + size - 1) / size
    let yBlocks = try decodePlaneSubbands(data: bufY, size: size, blockCount: rowCountY * colCountY)
    
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let rowCountCb = (cbDy + size - 1) / size
    let colCountCb = (cbDx + size - 1) / size
    let cbBlocks = try decodePlaneSubbands(data: bufCb, size: size, blockCount: rowCountCb * colCountCb)
    
    let rowCountCr = (cbDy + size - 1) / size
    let colCountCr = (cbDx + size - 1) / size
    let crBlocks = try decodePlaneSubbands(data: bufCr, size: size, blockCount: rowCountCr * colCountCr)
    
    let chunkSize = 4
    
    // Y
    let taskCountY = (rowCountY + chunkSize - 1) / chunkSize
    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountY {
            group.addTask {
                let startRow = taskIdx * chunkSize
                let endRow = min(startRow + chunkSize, rowCountY)
                var rowResults: [(Block2D, Int, Int)] = []
                for i in startRow..<endRow {
                    let h = i * size
                    for (xIdx, w) in stride(from: 0, to: dx, by: size).enumerated() {
                        let blockIndex = i * colCountY + xIdx
                        var block = yBlocks[blockIndex]
                        let half = size / 2
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
                            var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: size)
                            var lhView = BlockView(base: base.advanced(by: half * size), width: half, height: half, stride: size)
                            var hhView = BlockView(base: base.advanced(by: half * size + half), width: half, height: half, stride: size)
                            dequantizeMidSignedMapping(&hlView, qt: qt)
                            dequantizeMidSignedMapping(&lhView, qt: qt)
                            dequantizeHighSignedMapping(&hhView, qt: qt)
                            invDwt2d(&view, size: size)
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
                sub.updateY(data: &blk, startX: w, startY: h, size: size)
            }
        }
    }
    
    // Cb
    let taskCountCb = (rowCountCb + chunkSize - 1) / chunkSize
    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountCb {
            group.addTask {
                let startRow = taskIdx * chunkSize
                let endRow = min(startRow + chunkSize, rowCountCb)
                var rowResults: [(Block2D, Int, Int)] = []
                for i in startRow..<endRow {
                    let h = i * size
                    for (xIdx, w) in stride(from: 0, to: cbDx, by: size).enumerated() {
                        let blockIndex = i * colCountCb + xIdx
                        var block = cbBlocks[blockIndex]
                        let half = size / 2
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
                            var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: size)
                            var lhView = BlockView(base: base.advanced(by: half * size), width: half, height: half, stride: size)
                            var hhView = BlockView(base: base.advanced(by: half * size + half), width: half, height: half, stride: size)
                            dequantizeMidSignedMapping(&hlView, qt: qt)
                            dequantizeMidSignedMapping(&lhView, qt: qt)
                            dequantizeHighSignedMapping(&hhView, qt: qt)
                            invDwt2d(&view, size: size)
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
                sub.updateCb(data: &blk, startX: w, startY: h, size: size)
            }
        }
    }
    
    // Cr
    let taskCountCr = (rowCountCr + chunkSize - 1) / chunkSize
    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountCr {
            group.addTask {
                let startRow = taskIdx * chunkSize
                let endRow = min(startRow + chunkSize, rowCountCr)
                var rowResults: [(Block2D, Int, Int)] = []
                for i in startRow..<endRow {
                    let h = i * size
                    for (xIdx, w) in stride(from: 0, to: cbDx, by: size).enumerated() {
                        let blockIndex = i * colCountCr + xIdx
                        var block = crBlocks[blockIndex]
                        let half = size / 2
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
                            var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: size)
                            var lhView = BlockView(base: base.advanced(by: half * size), width: half, height: half, stride: size)
                            var hhView = BlockView(base: base.advanced(by: half * size + half), width: half, height: half, stride: size)
                            dequantizeMidSignedMapping(&hlView, qt: qt)
                            dequantizeMidSignedMapping(&lhView, qt: qt)
                            dequantizeHighSignedMapping(&hhView, qt: qt)
                            invDwt2d(&view, size: size)
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
                sub.updateCr(data: &blk, startX: w, startY: h, size: size)
            }
        }
    }
    
    return sub
}

@inline(__always)
func decodeBase(r: [UInt8], layer: UInt8, size: Int) async throws -> Image16 {
    var offset = 0
    
    guard (offset + 5) <= r.count else { throw DecodeError.insufficientData }
    let header = Array(r[offset..<(offset + 5)])
    offset += 5
    
    guard header[0] == 0x56 && header[1] == 0x45 && header[2] == 0x56 && header[3] == 0x43 else { // check 'VEVC'
         throw DecodeError.invalidHeader
    }
    guard header[4] == layer else { // check layer
        throw DecodeError.invalidLayerNumber
    }
    
    let dx = Int(try readUInt16BEFromBytes(r, offset: &offset))
    let dy = Int(try readUInt16BEFromBytes(r, offset: &offset))
    let qt = QuantizationTable(baseStep: Int(try readUInt8FromBytes(r, offset: &offset)))
    
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
    
    let rowCountY = (dy + size - 1) / size
    let colCountY = (dx + size - 1) / size
    let yBlocks = try decodePlaneBaseSubbands(data: bufY, size: size, blockCount: rowCountY * colCountY)
    
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let rowCountCb = (cbDy + size - 1) / size
    let colCountCb = (cbDx + size - 1) / size
    let cbBlocks = try decodePlaneBaseSubbands(data: bufCb, size: size, blockCount: rowCountCb * colCountCb)
    
    let rowCountCr = (cbDy + size - 1) / size
    let colCountCr = (cbDx + size - 1) / size
    let crBlocks = try decodePlaneBaseSubbands(data: bufCr, size: size, blockCount: rowCountCr * colCountCr)
    
    let chunkSize = 4
    
    // Y
    let taskCountY = (rowCountY + chunkSize - 1) / chunkSize
    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountY {
            group.addTask {
                let startRow = taskIdx * chunkSize
                let endRow = min(startRow + chunkSize, rowCountY)
                var rowResults: [(Block2D, Int, Int)] = []
                for i in startRow..<endRow {
                    let h = i * size
                    for (xIdx, w) in stride(from: 0, to: dx, by: size).enumerated() {
                        let blockIndex = i * colCountY + xIdx
                        var block = yBlocks[blockIndex]
                        let half = size / 2
                        block.withView { view in
                            let base = view.base
                            var llView = BlockView(base: base, width: half, height: half, stride: size)
                            var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: size)
                            var lhView = BlockView(base: base.advanced(by: half * size), width: half, height: half, stride: size)
                            var hhView = BlockView(base: base.advanced(by: half * size + half), width: half, height: half, stride: size)
                            dequantizeLow(&llView, qt: qt)
                            dequantizeMidSignedMapping(&hlView, qt: qt)
                            dequantizeMidSignedMapping(&lhView, qt: qt)
                            dequantizeHighSignedMapping(&hhView, qt: qt)
                            invDwt2d(&view, size: size)
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
                sub.updateY(data: &blk, startX: w, startY: h, size: size)
            }
        }
    }
    
    // Cb
    let taskCountCb = (rowCountCb + chunkSize - 1) / chunkSize
    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountCb {
            group.addTask {
                let startRow = taskIdx * chunkSize
                let endRow = min(startRow + chunkSize, rowCountCb)
                var rowResults: [(Block2D, Int, Int)] = []
                for i in startRow..<endRow {
                    let h = i * size
                    for (xIdx, w) in stride(from: 0, to: cbDx, by: size).enumerated() {
                        let blockIndex = i * colCountCb + xIdx
                        var block = cbBlocks[blockIndex]
                        let half = size / 2
                        block.withView { view in
                            let base = view.base
                            var llView = BlockView(base: base, width: half, height: half, stride: size)
                            var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: size)
                            var lhView = BlockView(base: base.advanced(by: half * size), width: half, height: half, stride: size)
                            var hhView = BlockView(base: base.advanced(by: half * size + half), width: half, height: half, stride: size)
                            dequantizeLow(&llView, qt: qt)
                            dequantizeMidSignedMapping(&hlView, qt: qt)
                            dequantizeMidSignedMapping(&lhView, qt: qt)
                            dequantizeHighSignedMapping(&hhView, qt: qt)
                            invDwt2d(&view, size: size)
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
                sub.updateCb(data: &blk, startX: w, startY: h, size: size)
            }
        }
    }
    
    // Cr
    let taskCountCr = (rowCountCr + chunkSize - 1) / chunkSize
    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { group in
        for taskIdx in 0..<taskCountCr {
            group.addTask {
                let startRow = taskIdx * chunkSize
                let endRow = min(startRow + chunkSize, rowCountCr)
                var rowResults: [(Block2D, Int, Int)] = []
                for i in startRow..<endRow {
                    let h = i * size
                    for (xIdx, w) in stride(from: 0, to: cbDx, by: size).enumerated() {
                        let blockIndex = i * colCountCr + xIdx
                        var block = crBlocks[blockIndex]
                        let half = size / 2
                        block.withView { view in
                            let base = view.base
                            var llView = BlockView(base: base, width: half, height: half, stride: size)
                            var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: size)
                            var lhView = BlockView(base: base.advanced(by: half * size), width: half, height: half, stride: size)
                            var hhView = BlockView(base: base.advanced(by: half * size + half), width: half, height: half, stride: size)
                            dequantizeLow(&llView, qt: qt)
                            dequantizeMidSignedMapping(&hlView, qt: qt)
                            dequantizeMidSignedMapping(&lhView, qt: qt)
                            dequantizeHighSignedMapping(&hhView, qt: qt)
                            invDwt2d(&view, size: size)
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
                sub.updateCr(data: &blk, startX: w, startY: h, size: size)
            }
        }
    }
    
    return sub
}

public struct DecodeOptions: Sendable {
    public var maxLayer: Int = 2
    public var maxFrames: Int = 4 // 1, 2, or 4
    
    public init(maxLayer: Int = 2, maxFrames: Int = 4) {
        self.maxLayer = maxLayer
        self.maxFrames = maxFrames
    }
}

public func decode(data: [UInt8], opts: DecodeOptions = DecodeOptions()) async throws -> [YCbCrImage] {
    var out: [YCbCrImage] = []
    var offset = 0
    var prevReconstructed: PlaneData420? = nil
    
    while offset + 4 <= data.count {
        let magic = Array(data[offset..<(offset + 4)])
        offset += 4
        
        if magic == [0x56, 0x45, 0x56, 0x49] { // 'VEVI'
            let len = Int(try readUInt32BEFromBytes(data, offset: &offset))
            let chunk = Array(data[offset..<(offset + len)])
            offset += len
            
            let img16 = try await decodeSpatialLayers(r: chunk, maxLayer: opts.maxLayer)
            let pd = PlaneData420(img16: img16)
            out.append(pd.toYCbCr())
            prevReconstructed = pd
            
        } else if magic == [0x56, 0x45, 0x56, 0x50] { // 'VEVP'
            let dx = Int(Int16(bitPattern: try readUInt16BEFromBytes(data, offset: &offset)))
            let dy = Int(Int16(bitPattern: try readUInt16BEFromBytes(data, offset: &offset)))
            
            let len = Int(try readUInt32BEFromBytes(data, offset: &offset))
            let chunk = Array(data[offset..<(offset + len)])
            offset += len
            
            let img16 = try await decodeSpatialLayers(r: chunk, maxLayer: opts.maxLayer)
            let residual = PlaneData420(img16: img16)
            
            if let prev = prevReconstructed {
                let predicted = await shiftPlane(prev, dx: dx, dy: dy)
                let curr = await addPlanes(residual: residual, predicted: predicted)
                out.append(curr.toYCbCr())
                prevReconstructed = curr
            } else {
                // If there's no previous frame, we can't decode a P-frame properly.
                // Just use residual as is or fail.
                out.append(residual.toYCbCr())
            }
        } else {
             throw DecodeError.invalidHeader
        }
    }
    
    return out
}
