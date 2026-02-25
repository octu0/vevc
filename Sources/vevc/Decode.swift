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

@inline(__always)
func applyInverseTemporal(ll: PlaneData420, lh: PlaneData420, h0: PlaneData420, h1: PlaneData420, countY: Int, countC: Int) -> (PlaneData420, PlaneData420, PlaneData420, PlaneData420) {
    let dx = ll.width, dy = ll.height
    
    func transform(l: [Int16], h: [Int16], hh0: [Int16], hh1: [Int16], count: Int) -> ([Int16], [Int16], [Int16], [Int16]) {
        var f0 = [Int16](repeating: 0, count: count)
        var f1 = [Int16](repeating: 0, count: count)
        var f2 = [Int16](repeating: 0, count: count)
        var f3 = [Int16](repeating: 0, count: count)
        var t0 = [Int16](repeating: 0, count: count)
        var t1 = [Int16](repeating: 0, count: count)
        
        l.withUnsafeBufferPointer { ptrL in
        h.withUnsafeBufferPointer { ptrH in
        hh0.withUnsafeBufferPointer { ptrH0 in
        hh1.withUnsafeBufferPointer { ptrH1 in
            invTemporalDWT(
                inLL: ptrL.baseAddress!,
                inLH: ptrH.baseAddress!,
                inH0: ptrH0.baseAddress!,
                inH1: ptrH1.baseAddress!,
                count: count,
                outF0: &f0,
                outF1: &f1,
                outF2: &f2,
                outF3: &f3,
                tempL0: &t0,
                tempL1: &t1
            )
        }}}}
        return (f0, f1, f2, f3)
    }
    
    let y = transform(l: ll.y, h: lh.y, hh0: h0.y, hh1: h1.y, count: countY)
    let cb = transform(l: ll.cb, h: lh.cb, hh0: h0.cb, hh1: h1.cb, count: countC)
    let cr = transform(l: ll.cr, h: lh.cr, hh0: h0.cr, hh1: h1.cr, count: countC)
    
    return (
        PlaneData420(width: dx, height: dy, y: y.0, cb: cb.0, cr: cr.0),
        PlaneData420(width: dx, height: dy, y: y.1, cb: cb.1, cr: cr.1),
        PlaneData420(width: dx, height: dy, y: y.2, cb: cb.2, cr: cr.2),
        PlaneData420(width: dx, height: dy, y: y.3, cb: cb.3, cr: cr.3)
    )
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
    
    while offset + 4 <= data.count {
        let magic = Array(data[offset..<(offset + 3)])
        offset += 3
        guard magic == [0x56, 0x45, 0x4C] else {
            throw DecodeError.invalidHeader
        }
        
        let _ = Int(data[offset]) // GOP size (not used yet)
        offset += 1
        
        let gmv1_dx = Int(Int16(bitPattern: try readUInt16BEFromBytes(data, offset: &offset)))
        let gmv1_dy = Int(Int16(bitPattern: try readUInt16BEFromBytes(data, offset: &offset)))
        let gmv2_dx = Int(Int16(bitPattern: try readUInt16BEFromBytes(data, offset: &offset)))
        let gmv2_dy = Int(Int16(bitPattern: try readUInt16BEFromBytes(data, offset: &offset)))
        let gmv3_dx = Int(Int16(bitPattern: try readUInt16BEFromBytes(data, offset: &offset)))
        let gmv3_dy = Int(Int16(bitPattern: try readUInt16BEFromBytes(data, offset: &offset)))
        
        func readPlane() async throws -> PlaneData420 {
            let len = try readUInt32BEFromBytes(data, offset: &offset)
            let chunk = Array(data[offset..<(offset + Int(len))])
            offset += Int(len)
            let img = try await decodeSpatialLayers(r: chunk, maxLayer: opts.maxLayer)
            return PlaneData420(img16: img)
        }
        
        let ll = try await readPlane()
        let lh = try await readPlane()
        let h0 = try await readPlane()
        let h1 = try await readPlane()
        
        let countY = ll.y.count
        let countC = ll.cb.count
        
        let emptyPlane = PlaneData420(width: ll.width, height: ll.height, y: [Int16](repeating: 0, count: countY), cb: [Int16](repeating: 0, count: countC), cr: [Int16](repeating: 0, count: countC))
        
        let actualLH = opts.maxFrames >= 2 ? lh : emptyPlane
        let actualH0 = opts.maxFrames >= 4 ? h0 : emptyPlane
        let actualH1 = opts.maxFrames >= 4 ? h1 : emptyPlane

        // temporal inverse
        let (f0_s, f1_s, f2_s, f3_s) = applyInverseTemporal(ll: ll, lh: actualLH, h0: actualH0, h1: actualH1, countY: countY, countC: countC)
        
        let f0 = f0_s
        let f1 = shiftPlane(f1_s, dx: gmv1_dx, dy: gmv1_dy)
        let f2 = shiftPlane(f2_s, dx: gmv2_dx, dy: gmv2_dy)
        let f3 = shiftPlane(f3_s, dx: gmv3_dx, dy: gmv3_dy)
        
        out.append(f0.toYCbCr())
        if opts.maxFrames >= 2 {
            out.append(f1.toYCbCr())
            if opts.maxFrames >= 4 {
                out.append(f2.toYCbCr())
                out.append(f3.toYCbCr())
            }
        }
    }
    
    return out
}
