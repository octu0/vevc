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
    let len0 = try readUInt32BEFromBytes(r, offset: &offset)
    let layer0Data = Array(r[offset..<(offset + Int(len0))])
    offset += Int(len0)

    var current = try await decodeBase(r: layer0Data, layer: 0, size: 8)

    if 1 <= maxLayer {
        let len1 = try readUInt32BEFromBytes(r, offset: &offset)
        let layer1Data = Array(r[offset..<(offset + Int(len1))])
        offset += Int(len1)
        current = try await decodeLayer(r: layer1Data, layer: 1, prev: current, size: 16)
    }
    if 2 <= maxLayer {
        let len2 = try readUInt32BEFromBytes(r, offset: &offset)
        let layer2Data = Array(r[offset..<(offset + Int(len2))])
        offset += Int(len2)
        current = try await decodeLayer(r: layer2Data, layer: 2, prev: current, size: 32)
    }
    return current
}

@inline(__always)
func toInt16(_ u: UInt16) -> Int16 {
    let s = Int16(bitPattern: (u >> 1))
    let m = (-1 * Int16(bitPattern: (u & 1)))
    return (s ^ m)
}

@inline(__always)
func blockDecode(rr: inout RiceReader, block: inout BlockView, size: Int, k: UInt8) throws {
    for y in 0..<size {
        let ptr = block.rowPointer(y: y)
        for x in 0..<size {
            let v = try rr.read(k: k)
            ptr[x+0] = Int16(bitPattern: v)
        }
    }
}

@inline(__always)
func blockDecodeDPCM(rr: inout RiceReader, block: inout BlockView, size: Int, k: UInt8, lastVal: inout Int16) throws {
    for y in 0..<size {
        let ptr = block.rowPointer(y: y)
        for x in 0..<size {
            let v = try rr.read(k: k)
            let diff = toInt16(v)
            let predicted: Int16
            switch (x, y) {
            case (0, 0):
                predicted = lastVal
            case (_, 0):
                predicted = ptr[(x - 1)]
            case (0, _):
                predicted = block.rowPointer(y: (y - 1))[x+0]
            default:
                let a = Int(ptr[(x - 1)])
                let b = Int(block.rowPointer(y: (y - 1))[x+0])
                let c = Int(block.rowPointer(y: (y - 1))[(x - 1)])
                if max(a, b) <= c {
                    predicted = Int16(min(a, b))
                } else {
                    if c <= min(a, b) {
                        predicted = Int16(max(a, b))
                    } else {
                        predicted = Int16(((a + b) - c))
                    }
                }
            }
            let val = (diff + predicted)
            ptr[x+0] = val
        }
    }
    lastVal = block.rowPointer(y: (size - 1))[(size - 1)]
}

@inline(__always)
func decodePlaneSubbands(data: [UInt8], size: Int, blockCount: Int) throws -> [Block2D] {
    var blocks: [Block2D] = []
    blocks.reserveCapacity(blockCount)
    for _ in 0..<blockCount {
        blocks.append(Block2D(width: size, height: size))
    }

    let flagsByteCount = ((blockCount + 7) / 8)
    guard flagsByteCount <= data.count else {
        throw DecodeError.insufficientData
    }
    let flagsData = Array(data[0..<flagsByteCount])
    let dataSlice = Array(data[flagsByteCount...])

    var brFlags = BitReader(data: flagsData)
    var nonZeroIndices: [Int] = []
    for i in 0..<blockCount {
        if try brFlags.readBit() == 0 {
            nonZeroIndices.append(i)
        }
    }

    var brData = BitReader(data: dataSlice)

    let half = (size / 2)

    let kHL = UInt8(try brData.readBits(n: 4))
    try RiceReader.withReader(&brData) { (rr: inout RiceReader) in
        for i in nonZeroIndices {
            try blocks[i+0].withView { (view: inout BlockView) in
                var hlView = BlockView(base: view.base.advanced(by: half), width: half, height: half, stride: size)
                try blockDecode(rr: &rr, block: &hlView, size: half, k: kHL)
            }
        }
    }

    let kLH = UInt8(try brData.readBits(n: 4))
    try RiceReader.withReader(&brData) { (rr: inout RiceReader) in
        for i in nonZeroIndices {
            try blocks[i+0].withView { (view: inout BlockView) in
                var lhView = BlockView(base: view.base.advanced(by: (half * size)), width: half, height: half, stride: size)
                try blockDecode(rr: &rr, block: &lhView, size: half, k: kLH)
            }
        }
    }

    let kHH = UInt8(try brData.readBits(n: 4))
    try RiceReader.withReader(&brData) { (rr: inout RiceReader) in
        for i in nonZeroIndices {
            try blocks[i+0].withView { (view: inout BlockView) in
                var hhView = BlockView(base: view.base.advanced(by: ((half * size) + half)), width: half, height: half, stride: size)
                try blockDecode(rr: &rr, block: &hhView, size: half, k: kHH)
            }
        }
    }
    return blocks
}

@inline(__always)
func decodePlaneBaseSubbands(data: [UInt8], size: Int, blockCount: Int) throws -> [Block2D] {
    var blocks: [Block2D] = []
    blocks.reserveCapacity(blockCount)
    for _ in 0..<blockCount {
        blocks.append(Block2D(width: size, height: size))
    }

    let flagsByteCount = ((blockCount + 7) / 8)
    guard flagsByteCount <= data.count else {
        throw DecodeError.insufficientData
    }
    let flagsData = Array(data[0..<flagsByteCount])
    let dataSlice = Array(data[flagsByteCount...])

    var brFlags = BitReader(data: flagsData)
    var nonZeroIndices: [Int] = []
    for i in 0..<blockCount {
        if try brFlags.readBit() == 0 {
            nonZeroIndices.append(i)
        }
    }

    var brData = BitReader(data: dataSlice)

    let half = (size / 2)

    let kLL = UInt8(try brData.readBits(n: 4))
    try RiceReader.withReader(&brData) { (rr: inout RiceReader) in
        var lastVal: Int16 = 0
        let nonZeroSet = Set(nonZeroIndices)
        for i in 0..<blockCount {
            if nonZeroSet.contains(i) {
                try blocks[i+0].withView { (view: inout BlockView) in
                    var llView = BlockView(base: view.base, width: half, height: half, stride: size)
                    try blockDecodeDPCM(rr: &rr, block: &llView, size: half, k: kLL, lastVal: &lastVal)
                }
            } else {
                lastVal = 0
            }
        }
    }

    let kHL = UInt8(try brData.readBits(n: 4))
    try RiceReader.withReader(&brData) { (rr: inout RiceReader) in
        for i in nonZeroIndices {
            try blocks[i+0].withView { (view: inout BlockView) in
                var hlView = BlockView(base: view.base.advanced(by: half), width: half, height: half, stride: size)
                try blockDecode(rr: &rr, block: &hlView, size: half, k: kHL)
            }
        }
    }

    let kLH = UInt8(try brData.readBits(n: 4))
    try RiceReader.withReader(&brData) { (rr: inout RiceReader) in
        for i in nonZeroIndices {
            try blocks[i+0].withView { (view: inout BlockView) in
                var lhView = BlockView(base: view.base.advanced(by: (half * size)), width: half, height: half, stride: size)
                try blockDecode(rr: &rr, block: &lhView, size: half, k: kLH)
            }
        }
    }

    let kHH = UInt8(try brData.readBits(n: 4))
    try RiceReader.withReader(&brData) { (rr: inout RiceReader) in
        for i in nonZeroIndices {
            try blocks[i+0].withView { (view: inout BlockView) in
                var hhView = BlockView(base: view.base.advanced(by: ((half * size) + half)), width: half, height: half, stride: size)
                try blockDecode(rr: &rr, block: &hhView, size: half, k: kHH)
            }
        }
    }
    return blocks
}

@inline(__always)
func readUInt8FromBytes(_ r: [UInt8], offset: inout Int) throws -> UInt8 {
    guard (offset + 1) <= r.count else {
        throw DecodeError.insufficientData
    }
    let val = r[offset+0]
    offset += 1
    return val
}

@inline(__always)
func readUInt16BEFromBytes(_ r: [UInt8], offset: inout Int) throws -> UInt16 {
    guard (offset + 2) <= r.count else {
        throw DecodeError.insufficientData
    }
    let val = ((UInt16(r[offset+0]) << 8) | UInt16(r[offset+1]))
    offset += 2
    return val
}

@inline(__always)
func readUInt32BEFromBytes(_ r: [UInt8], offset: inout Int) throws -> UInt32 {
    guard (offset + 4) <= r.count else {
        throw DecodeError.insufficientData
    }
    let val = ((UInt32(r[offset+0]) << 24) | (UInt32(r[offset+1]) << 16) | (UInt32(r[offset+2]) << 8) | UInt32(r[offset+3]))
    offset += 4
    return val
}

@inline(__always)
func readBlockFromBytes(_ r: [UInt8], offset: inout Int) throws -> [UInt8] {
    let len = try readUInt16BEFromBytes(r, offset: &offset)
    let intLen = Int(len)
    guard (offset + intLen) <= r.count else {
        throw DecodeError.invalidBlockData
    }
    let block = Array(r[offset..<(offset + intLen)])
    offset += intLen
    return block
}

@inline(__always)
func decodeLayer(r: [UInt8], layer: UInt8, prev: Image16, size: Int) async throws -> Image16 {
    var offset = 0

    guard (offset + 5) <= r.count else {
        throw DecodeError.insufficientData
    }
    let header = Array(r[offset..<(offset + 5)])
    offset += 5

    guard header[0+0] == 0x56 && header[1+0] == 0x45 && header[2+0] == 0x56 && header[3+0] == 0x43 else {
         throw DecodeError.invalidHeader
    }
    guard header[4+0] == layer else {
        throw DecodeError.invalidLayerNumber
    }

    let dx = Int(try readUInt16BEFromBytes(r, offset: &offset))
    let dy = Int(try readUInt16BEFromBytes(r, offset: &offset))
    let qtY = QuantizationTable(baseStep: Int(try readUInt8FromBytes(r, offset: &offset)))
    let qtC = QuantizationTable(baseStep: Int(try readUInt8FromBytes(r, offset: &offset)))

    let bufYLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufYLen) <= r.count else {
        throw DecodeError.invalidBlockData
    }
    let bufY = Array(r[offset..<(offset + bufYLen)])
    offset += bufYLen

    let bufCbLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufCbLen) <= r.count else {
        throw DecodeError.invalidBlockData
    }
    let bufCb = Array(r[offset..<(offset + bufCbLen)])
    offset += bufCbLen

    let bufCrLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufCrLen) <= r.count else {
        throw DecodeError.invalidBlockData
    }
    let bufCr = Array(r[offset..<(offset + bufCrLen)])
    offset += bufCrLen

    var sub = Image16(width: dx, height: dy)

    let rowCountY = (((dy + size) - 1) / size)
    let colCountY = (((dx + size) - 1) / size)
    let yBlocks = try decodePlaneSubbands(data: bufY, size: size, blockCount: (rowCountY * colCountY))

    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    let rowCountCb = (((cbDy + size) - 1) / size)
    let colCountCb = (((cbDx + size) - 1) / size)
    let cbBlocks = try decodePlaneSubbands(data: bufCb, size: size, blockCount: (rowCountCb * colCountCb))

    let rowCountCr = (((cbDy + size) - 1) / size)
    let colCountCr = (((cbDx + size) - 1) / size)
    let crBlocks = try decodePlaneSubbands(data: bufCr, size: size, blockCount: (rowCountCr * colCountCr))

    let chunkSize = 4

    let taskCountY = (((rowCountY + chunkSize) - 1) / chunkSize)
    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { (group: inout ThrowingTaskGroup<[(Block2D, Int, Int)], any Error>) in
        for taskIdx in 0..<taskCountY {
            group.addTask {
                let startRow = (taskIdx * chunkSize)
                let endRow = min((startRow + chunkSize), rowCountY)
                var rowResults: [(Block2D, Int, Int)] = []
                for i in startRow..<endRow {
                    let h = (i * size)
                    for (xIdx, w) in stride(from: 0, to: dx, by: size).enumerated() {
                        let blockIndex = ((i * colCountY) + xIdx)
                        var block = yBlocks[blockIndex+0]
                        let half = (size / 2)
                        var ll = prev.getY(x: (w / 2), y: (h / 2), size: half)
                        ll.withView { (srcView: inout BlockView) in
                            block.withView { (destView: inout BlockView) in
                                for yi in 0..<half {
                                    let srcPtr = srcView.rowPointer(y: yi)
                                    let destPtr = destView.rowPointer(y: yi)
                                    destPtr.update(from: srcPtr, count: half)
                                }
                            }
                        }
                        block.withView { (view: inout BlockView) in
                            let base = view.base
                            var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: size)
                            var lhView = BlockView(base: base.advanced(by: (half * size)), width: half, height: half, stride: size)
                            var hhView = BlockView(base: base.advanced(by: ((half * size) + half)), width: half, height: half, stride: size)
                            dequantizeMidSignedMapping(&hlView, qt: qtY)
                            dequantizeMidSignedMapping(&lhView, qt: qtY)
                            dequantizeHighSignedMapping(&hhView, qt: qtY)
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
                var blk = res[j+0].0
                let w = res[j+0].1
                let h = res[j+0].2
                sub.updateY(data: &blk, startX: w, startY: h, size: size)
            }
        }
    }

    let taskCountCb = (((rowCountCb + chunkSize) - 1) / chunkSize)
    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { (group: inout ThrowingTaskGroup<[(Block2D, Int, Int)], any Error>) in
        for taskIdx in 0..<taskCountCb {
            group.addTask {
                let startRow = (taskIdx * chunkSize)
                let endRow = min((startRow + chunkSize), rowCountCb)
                var rowResults: [(Block2D, Int, Int)] = []
                for i in startRow..<endRow {
                    let h = (i * size)
                    for (xIdx, w) in stride(from: 0, to: cbDx, by: size).enumerated() {
                        let blockIndex = ((i * colCountCb) + xIdx)
                        var block = cbBlocks[blockIndex+0]
                        let half = (size / 2)
                        var ll = prev.getCb(x: (w / 2), y: (h / 2), size: half)
                        ll.withView { (srcView: inout BlockView) in
                            block.withView { (destView: inout BlockView) in
                                for yi in 0..<half {
                                    let srcPtr = srcView.rowPointer(y: yi)
                                    let destPtr = destView.rowPointer(y: yi)
                                    destPtr.update(from: srcPtr, count: half)
                                }
                            }
                        }
                        block.withView { (view: inout BlockView) in
                            let base = view.base
                            var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: size)
                            var lhView = BlockView(base: base.advanced(by: (half * size)), width: half, height: half, stride: size)
                            var hhView = BlockView(base: base.advanced(by: ((half * size) + half)), width: half, height: half, stride: size)
                            dequantizeMidSignedMapping(&hlView, qt: qtC)
                            dequantizeMidSignedMapping(&lhView, qt: qtC)
                            dequantizeHighSignedMapping(&hhView, qt: qtC)
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
                var blk = res[j+0].0
                let w = res[j+0].1
                let h = res[j+0].2
                sub.updateCb(data: &blk, startX: w, startY: h, size: size)
            }
        }
    }

    let taskCountCr = (((rowCountCr + chunkSize) - 1) / chunkSize)
    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { (group: inout ThrowingTaskGroup<[(Block2D, Int, Int)], any Error>) in
        for taskIdx in 0..<taskCountCr {
            group.addTask {
                let startRow = (taskIdx * chunkSize)
                let endRow = min((startRow + chunkSize), rowCountCr)
                var rowResults: [(Block2D, Int, Int)] = []
                for i in startRow..<endRow {
                    let h = (i * size)
                    for (xIdx, w) in stride(from: 0, to: cbDx, by: size).enumerated() {
                        let blockIndex = ((i * colCountCr) + xIdx)
                        var block = crBlocks[blockIndex+0]
                        let half = (size / 2)
                        var ll = prev.getCr(x: (w / 2), y: (h / 2), size: half)
                        ll.withView { (srcView: inout BlockView) in
                            block.withView { (destView: inout BlockView) in
                                for yi in 0..<half {
                                    let srcPtr = srcView.rowPointer(y: yi)
                                    let destPtr = destView.rowPointer(y: yi)
                                    destPtr.update(from: srcPtr, count: half)
                                }
                            }
                        }
                        block.withView { (view: inout BlockView) in
                            let base = view.base
                            var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: size)
                            var lhView = BlockView(base: base.advanced(by: (half * size)), width: half, height: half, stride: size)
                            var hhView = BlockView(base: base.advanced(by: ((half * size) + half)), width: half, height: half, stride: size)
                            dequantizeMidSignedMapping(&hlView, qt: qtC)
                            dequantizeMidSignedMapping(&lhView, qt: qtC)
                            dequantizeHighSignedMapping(&hhView, qt: qtC)
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
                var blk = res[j+0].0
                let w = res[j+0].1
                let h = res[j+0].2
                sub.updateCr(data: &blk, startX: w, startY: h, size: size)
            }
        }
    }

    return sub
}

@inline(__always)
func decodeBase(r: [UInt8], layer: UInt8, size: Int) async throws -> Image16 {
    var offset = 0

    guard (offset + 5) <= r.count else {
        throw DecodeError.insufficientData
    }
    let header = Array(r[offset..<(offset + 5)])
    offset += 5

    guard header[0+0] == 0x56 && header[1+0] == 0x45 && header[2+0] == 0x56 && header[3+0] == 0x43 else {
         throw DecodeError.invalidHeader
    }
    guard header[4+0] == layer else {
        throw DecodeError.invalidLayerNumber
    }

    let dx = Int(try readUInt16BEFromBytes(r, offset: &offset))
    let dy = Int(try readUInt16BEFromBytes(r, offset: &offset))
    let qtY = QuantizationTable(baseStep: Int(try readUInt8FromBytes(r, offset: &offset)))
    let qtC = QuantizationTable(baseStep: Int(try readUInt8FromBytes(r, offset: &offset)))

    let bufYLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufYLen) <= r.count else {
        throw DecodeError.invalidBlockData
    }
    let bufY = Array(r[offset..<(offset + bufYLen)])
    offset += bufYLen

    let bufCbLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufCbLen) <= r.count else {
        throw DecodeError.invalidBlockData
    }
    let bufCb = Array(r[offset..<(offset + bufCbLen)])
    offset += bufCbLen

    let bufCrLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    guard (offset + bufCrLen) <= r.count else {
        throw DecodeError.invalidBlockData
    }
    let bufCr = Array(r[offset..<(offset + bufCrLen)])
    offset += bufCrLen

    var sub = Image16(width: dx, height: dy)

    let rowCountY = (((dy + size) - 1) / size)
    let colCountY = (((dx + size) - 1) / size)
    let yBlocks = try decodePlaneBaseSubbands(data: bufY, size: size, blockCount: (rowCountY * colCountY))

    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    let rowCountCb = (((cbDy + size) - 1) / size)
    let colCountCb = (((cbDx + size) - 1) / size)
    let cbBlocks = try decodePlaneBaseSubbands(data: bufCb, size: size, blockCount: (rowCountCb * colCountCb))

    let rowCountCr = (((cbDy + size) - 1) / size)
    let colCountCr = (((cbDx + size) - 1) / size)
    let crBlocks = try decodePlaneBaseSubbands(data: bufCr, size: size, blockCount: (rowCountCr * colCountCr))

    let chunkSize = 4

    let taskCountY = (((rowCountY + chunkSize) - 1) / chunkSize)
    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { (group: inout ThrowingTaskGroup<[(Block2D, Int, Int)], any Error>) in
        for taskIdx in 0..<taskCountY {
            group.addTask {
                let startRow = (taskIdx * chunkSize)
                let endRow = min((startRow + chunkSize), rowCountY)
                var rowResults: [(Block2D, Int, Int)] = []
                for i in startRow..<endRow {
                    let h = (i * size)
                    for (xIdx, w) in stride(from: 0, to: dx, by: size).enumerated() {
                        let blockIndex = ((i * colCountY) + xIdx)
                        var block = yBlocks[blockIndex+0]
                        let half = (size / 2)
                        block.withView { (view: inout BlockView) in
                            let base = view.base
                            var llView = BlockView(base: base, width: half, height: half, stride: size)
                            var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: size)
                            var lhView = BlockView(base: base.advanced(by: (half * size)), width: half, height: half, stride: size)
                            var hhView = BlockView(base: base.advanced(by: ((half * size) + half)), width: half, height: half, stride: size)
                            dequantizeLow(&llView, qt: qtY)
                            dequantizeMidSignedMapping(&hlView, qt: qtY)
                            dequantizeMidSignedMapping(&lhView, qt: qtY)
                            dequantizeHighSignedMapping(&hhView, qt: qtY)
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
                var blk = res[j+0].0
                let w = res[j+0].1
                let h = res[j+0].2
                sub.updateY(data: &blk, startX: w, startY: h, size: size)
            }
        }
    }

    let taskCountCb = (((rowCountCb + chunkSize) - 1) / chunkSize)
    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { (group: inout ThrowingTaskGroup<[(Block2D, Int, Int)], any Error>) in
        for taskIdx in 0..<taskCountCb {
            group.addTask {
                let startRow = (taskIdx * chunkSize)
                let endRow = min((startRow + chunkSize), rowCountCb)
                var rowResults: [(Block2D, Int, Int)] = []
                for i in startRow..<endRow {
                    let h = (i * size)
                    for (xIdx, w) in stride(from: 0, to: cbDx, by: size).enumerated() {
                        let blockIndex = ((i * colCountCb) + xIdx)
                        var block = cbBlocks[blockIndex+0]
                        let half = (size / 2)
                        block.withView { (view: inout BlockView) in
                            let base = view.base
                            var llView = BlockView(base: base, width: half, height: half, stride: size)
                            var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: size)
                            var lhView = BlockView(base: base.advanced(by: (half * size)), width: half, height: half, stride: size)
                            var hhView = BlockView(base: base.advanced(by: ((half * size) + half)), width: half, height: half, stride: size)
                            dequantizeLow(&llView, qt: qtC)
                            dequantizeMidSignedMapping(&hlView, qt: qtC)
                            dequantizeMidSignedMapping(&lhView, qt: qtC)
                            dequantizeHighSignedMapping(&hhView, qt: qtC)
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
                var blk = res[j+0].0
                let w = res[j+0].1
                let h = res[j+0].2
                sub.updateCb(data: &blk, startX: w, startY: h, size: size)
            }
        }
    }

    let taskCountCr = (((rowCountCr + chunkSize) - 1) / chunkSize)
    try await withThrowingTaskGroup(of: [(Block2D, Int, Int)].self) { (group: inout ThrowingTaskGroup<[(Block2D, Int, Int)], any Error>) in
        for taskIdx in 0..<taskCountCr {
            group.addTask {
                let startRow = (taskIdx * chunkSize)
                let endRow = min((startRow + chunkSize), rowCountCr)
                var rowResults: [(Block2D, Int, Int)] = []
                for i in startRow..<endRow {
                    let h = (i * size)
                    for (xIdx, w) in stride(from: 0, to: cbDx, by: size).enumerated() {
                        let blockIndex = ((i * colCountCr) + xIdx)
                        var block = crBlocks[blockIndex+0]
                        let half = (size / 2)
                        block.withView { (view: inout BlockView) in
                            let base = view.base
                            var llView = BlockView(base: base, width: half, height: half, stride: size)
                            var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: size)
                            var lhView = BlockView(base: base.advanced(by: (half * size)), width: half, height: half, stride: size)
                            var hhView = BlockView(base: base.advanced(by: ((half * size) + half)), width: half, height: half, stride: size)
                            dequantizeLow(&llView, qt: qtC)
                            dequantizeMidSignedMapping(&hlView, qt: qtC)
                            dequantizeMidSignedMapping(&lhView, qt: qtC)
                            dequantizeHighSignedMapping(&hhView, qt: qtC)
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
                var blk = res[j+0].0
                let w = res[j+0].1
                let h = res[j+0].2
                sub.updateCr(data: &blk, startX: w, startY: h, size: size)
            }
        }
    }

    return sub
}

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
        offset += 4

        if magic == [0x56, 0x45, 0x56, 0x49] {
            let len = Int(try readUInt32BEFromBytes(data, offset: &offset))
            let chunk = Array(data[offset..<(offset + len)])
            offset += len

            let img16 = try await decodeSpatialLayers(r: chunk, maxLayer: opts.maxLayer)
            let pd = PlaneData420(img16: img16)
            out.append(pd.toYCbCr())
            prevReconstructed = pd

        } else {
            guard magic == [0x56, 0x45, 0x56, 0x50] else {
                 throw DecodeError.invalidHeader
            }
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
                prevReconstructed = cleanExposedRegion(curr, dx: dx, dy: dy)
            } else {
                out.append(residual.toYCbCr())
            }
        }
    }

    return out
}
