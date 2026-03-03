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

private let k: UInt8 = 4

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

@inline(__always)
func invertLayer(br: BitReader, ll: Block2D, size: Int, qt: QuantizationTable) throws -> Block2D {
    var ll = ll
    var block = Block2D(width: size, height: size)
    let half = size / 2
    
    // Copy LL to top-left
    ll.withView { srcView in
        block.withView { destView in
            for y in 0..<half {
                let srcPtr = srcView.rowPointer(y: y)
                let destPtr = destView.rowPointer(y: y)
                destPtr.update(from: srcPtr, count: half)
            }
        }
    }
    
    var br = br
    let isZero = try br.readBit() == 1
    
    try block.withView { view in
        let base = view.base
        var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: size)
        var lhView = BlockView(base: base.advanced(by: half * size), width: half, height: half, stride: size)
        var hhView = BlockView(base: base.advanced(by: half * size + half), width: half, height: half, stride: size)
        
        if isZero != true {
            var rr = RiceReader(br: br)
            try blockDecode(rr: &rr, block: &hlView, size: half)
            try blockDecode(rr: &rr, block: &lhView, size: half)
            try blockDecode(rr: &rr, block: &hhView, size: half)
            
            dequantizeMidSignedMapping(&hlView, qt: qt)
            dequantizeMidSignedMapping(&lhView, qt: qt)
            dequantizeHighSignedMapping(&hhView, qt: qt)
        }
        
        invDwt2d(&view, size: size)
    }
    
    return block
}

@inline(__always)
func invertBase(br: BitReader, size: Int, qt: QuantizationTable) throws -> Block2D {
    var block = Block2D(width: size, height: size)
    let half = size / 2
    
    var br = br
    let isZero = try br.readBit() == 1
    
    try block.withView { view in
        let base = view.base
        var llView = BlockView(base: base, width: half, height: half, stride: size)
        var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: size)
        var lhView = BlockView(base: base.advanced(by: half * size), width: half, height: half, stride: size)
        var hhView = BlockView(base: base.advanced(by: half * size + half), width: half, height: half, stride: size)
        
        if !isZero {
            var rr = RiceReader(br: br)
            try blockDecodeDPCM(rr: &rr, block: &llView, size: half)
            try blockDecode(rr: &rr, block: &hlView, size: half)
            try blockDecode(rr: &rr, block: &lhView, size: half)
            try blockDecode(rr: &rr, block: &hhView, size: half)
            
            dequantizeLow(&llView, qt: qt)
            dequantizeMidSignedMapping(&hlView, qt: qt)
            dequantizeMidSignedMapping(&lhView, qt: qt)
            dequantizeHighSignedMapping(&hhView, qt: qt)
        }
        
        invDwt2d(&view, size: size)
    }
    
    return block
}

public typealias GetLLFunc = (_ x: Int, _ y: Int, _ size: Int) -> Block2D

@inline(__always)
func invertLayerFunc(br: BitReader, w: Int, h: Int, size: Int, qt: QuantizationTable, getLL: GetLLFunc) throws -> Block2D {
    let ll = getLL(w/2, h/2, size/2)
    let planes = try invertLayer(br: br, ll: ll, size: size, qt: qt)
    return planes
}

@inline(__always)
func invertBaseFunc(br: BitReader, w: Int, h: Int, size: Int, qt: QuantizationTable) throws -> Block2D {
    let planes = try invertBase(br: br, size: size, qt: qt)
    return planes
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
    
    let bufYLen = Int(try readUInt16BEFromBytes(r, offset: &offset))
    var yBufs: [[UInt8]] = []
    for _ in 0..<bufYLen {
        yBufs.append(try readBlockFromBytes(r, offset: &offset))
    }
    
    let bufCbLen = Int(try readUInt16BEFromBytes(r, offset: &offset))
    var cbBufs: [[UInt8]] = []
    for _ in 0..<bufCbLen {
        cbBufs.append(try readBlockFromBytes(r, offset: &offset))
    }
    
    let bufCrLen = Int(try readUInt16BEFromBytes(r, offset: &offset))
    var crBufs: [[UInt8]] = []
    for _ in 0..<bufCrLen {
        crBufs.append(try readBlockFromBytes(r, offset: &offset))
    }
    
    var sub = Image16(width: dx, height: dy)
    
    // Y
    try await withThrowingTaskGroup(of: (Int, [(Block2D, Int, Int)]).self) { group in
        var bufIndex = 0
        for h in stride(from: 0, to: dy, by: size) {
            let wStride = Array(stride(from: 0, to: dx, by: size))
            let rowBufs = Array(yBufs[bufIndex..<(bufIndex + wStride.count)])
            bufIndex += wStride.count
            
            group.addTask {
                var rowResults: [(Block2D, Int, Int)] = []
                for (i, w) in wStride.enumerated() {
                    let data = rowBufs[i]
                    let br = BitReader(data: data)
                    let ll = try invertLayerFunc(br: br, w: w, h: h, size: size, qt: qt, getLL: prev.getY)
                    rowResults.append((ll, w, h))
                }
                return (h, rowResults)
            }
        }
        
        var results: [(Int, [(Block2D, Int, Int)])] = []
        for try await res in group {
            results.append(res)
        }
        results.sort { $0.0 < $1.0 }
        
        for i in results.indices {
            for j in results[i].1.indices {
                let w = results[i].1[j].1
                let h = results[i].1[j].2
                sub.updateY(data: &results[i].1[j].0, startX: w, startY: h, size: size)
            }
        }
    }
    
    // Cb
    try await withThrowingTaskGroup(of: (Int, [(Block2D, Int, Int)]).self) { group in
        var bufIndex = 0
        for h in stride(from: 0, to: ((dy + 1) / 2), by: size) {
            let wStride = Array(stride(from: 0, to: ((dx + 1) / 2), by: size))
            let rowBufs = Array(cbBufs[bufIndex..<(bufIndex + wStride.count)])
            bufIndex += wStride.count
            
            group.addTask {
                var rowResults: [(Block2D, Int, Int)] = []
                for (i, w) in wStride.enumerated() {
                    let data = rowBufs[i]
                    let br = BitReader(data: data)
                    let ll = try invertLayerFunc(br: br, w: w, h: h, size: size, qt: qt, getLL: prev.getCb)
                    rowResults.append((ll, w, h))
                }
                return (h, rowResults)
            }
        }
        
        var results: [(Int, [(Block2D, Int, Int)])] = []
        for try await res in group {
            results.append(res)
        }
        results.sort { $0.0 < $1.0 }
        
        for i in results.indices {
            for j in results[i].1.indices {
                let w = results[i].1[j].1
                let h = results[i].1[j].2
                sub.updateCb(data: &results[i].1[j].0, startX: w, startY: h, size: size)
            }
        }
    }
    
    // Cr
    try await withThrowingTaskGroup(of: (Int, [(Block2D, Int, Int)]).self) { group in
        var bufIndex = 0
        for h in stride(from: 0, to: ((dy + 1) / 2), by: size) {
            let wStride = Array(stride(from: 0, to: ((dx + 1) / 2), by: size))
            let rowBufs = Array(crBufs[bufIndex..<(bufIndex + wStride.count)])
            bufIndex += wStride.count
            
            group.addTask {
                var rowResults: [(Block2D, Int, Int)] = []
                for (i, w) in wStride.enumerated() {
                    let data = rowBufs[i]
                    let br = BitReader(data: data)
                    let ll = try invertLayerFunc(br: br, w: w, h: h, size: size, qt: qt, getLL: prev.getCr)
                    rowResults.append((ll, w, h))
                }
                return (h, rowResults)
            }
        }
        
        var results: [(Int, [(Block2D, Int, Int)])] = []
        for try await res in group {
            results.append(res)
        }
        results.sort { $0.0 < $1.0 }
        
        for i in results.indices {
            for j in results[i].1.indices {
                let w = results[i].1[j].1
                let h = results[i].1[j].2
                sub.updateCr(data: &results[i].1[j].0, startX: w, startY: h, size: size)
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
    
    let bufYLen = Int(try readUInt16BEFromBytes(r, offset: &offset))
    var yBufs: [[UInt8]] = []
    for _ in 0..<bufYLen {
        yBufs.append(try readBlockFromBytes(r, offset: &offset))
    }
    
    let bufCbLen = Int(try readUInt16BEFromBytes(r, offset: &offset))
    var cbBufs: [[UInt8]] = []
    for _ in 0..<bufCbLen {
        cbBufs.append(try readBlockFromBytes(r, offset: &offset))
    }
    
    let bufCrLen = Int(try readUInt16BEFromBytes(r, offset: &offset))
    var crBufs: [[UInt8]] = []
    for _ in 0..<bufCrLen {
        crBufs.append(try readBlockFromBytes(r, offset: &offset))
    }
    
    var sub = Image16(width: dx, height: dy)
    
    // Y
    try await withThrowingTaskGroup(of: (Int, [(Block2D, Int, Int)]).self) { group in
        var bufIndex = 0
        for h in stride(from: 0, to: dy, by: size) {
            let wStride = Array(stride(from: 0, to: dx, by: size))
            let rowBufs = Array(yBufs[bufIndex..<(bufIndex + wStride.count)])
            bufIndex += wStride.count
            
            group.addTask {
                var rowResults: [(Block2D, Int, Int)] = []
                for (i, w) in wStride.enumerated() {
                    let data = rowBufs[i]
                    let br = BitReader(data: data)
                    let ll = try invertBaseFunc(br: br, w: w, h: h, size: size, qt: qt)
                    rowResults.append((ll, w, h))
                }
                return (h, rowResults)
            }
        }
        
        var results: [(Int, [(Block2D, Int, Int)])] = []
        for try await res in group {
            results.append(res)
        }
        results.sort { $0.0 < $1.0 }
        
        for i in results.indices {
            for j in results[i].1.indices {
                let w = results[i].1[j].1
                let h = results[i].1[j].2
                sub.updateY(data: &results[i].1[j].0, startX: w, startY: h, size: size)
            }
        }
    }
    
    // Cb
    try await withThrowingTaskGroup(of: (Int, [(Block2D, Int, Int)]).self) { group in
        var bufIndex = 0
        for h in stride(from: 0, to: ((dy + 1) / 2), by: size) {
            let wStride = Array(stride(from: 0, to: ((dx + 1) / 2), by: size))
            let rowBufs = Array(cbBufs[bufIndex..<(bufIndex + wStride.count)])
            bufIndex += wStride.count
            
            group.addTask {
                var rowResults: [(Block2D, Int, Int)] = []
                for (i, w) in wStride.enumerated() {
                    let data = rowBufs[i]
                    let br = BitReader(data: data)
                    let ll = try invertBaseFunc(br: br, w: w, h: h, size: size, qt: qt)
                    rowResults.append((ll, w, h))
                }
                return (h, rowResults)
            }
        }
        
        var results: [(Int, [(Block2D, Int, Int)])] = []
        for try await res in group {
            results.append(res)
        }
        results.sort { $0.0 < $1.0 }
        
        for i in results.indices {
            for j in results[i].1.indices {
                let w = results[i].1[j].1
                let h = results[i].1[j].2
                sub.updateCb(data: &results[i].1[j].0, startX: w, startY: h, size: size)
            }
        }
    }
    
    // Cr
    try await withThrowingTaskGroup(of: (Int, [(Block2D, Int, Int)]).self) { group in
        var bufIndex = 0
        for h in stride(from: 0, to: ((dy + 1) / 2), by: size) {
            let wStride = Array(stride(from: 0, to: ((dx + 1) / 2), by: size))
            let rowBufs = Array(crBufs[bufIndex..<(bufIndex + wStride.count)])
            bufIndex += wStride.count
            
            group.addTask {
                var rowResults: [(Block2D, Int, Int)] = []
                for (i, w) in wStride.enumerated() {
                    let data = rowBufs[i]
                    let br = BitReader(data: data)
                    let ll = try invertBaseFunc(br: br, w: w, h: h, size: size, qt: qt)
                    rowResults.append((ll, w, h))
                }
                return (h, rowResults)
            }
        }
        
        var results: [(Int, [(Block2D, Int, Int)])] = []
        for try await res in group {
            results.append(res)
        }
        results.sort { $0.0 < $1.0 }
        
        for i in results.indices {
            for j in results[i].1.indices {
                let w = results[i].1[j].1
                let h = results[i].1[j].2
                sub.updateCr(data: &results[i].1[j].0, startX: w, startY: h, size: size)
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
