import Foundation

public enum TrainingError: Error {
    case badFormat(String)
    case emptyDump
    case invalidInput(String)
}

struct DumpSubPlane: @unchecked Sendable {
    let width: Int
    let height: Int
    var data: [Int16]
}

struct DumpEntry: @unchecked Sendable {
    let subbands: [DumpSubPlane]
}

struct DumpFrame: @unchecked Sendable {
    let gop: Int
    let width: Int
    let height: Int
    let qsteps: [[Int]]
    let layerBytes: [Int]
    let coded: [DumpEntry]
    let parents: [DumpEntry]
    let pyr: [DumpEntry]
}

// MARK: - Coefficient Dump Writer

public final class CoeffDumpWriter: @unchecked Sendable {
    private let handle: FileHandle
    private let lock = NSLock()
    private var pending: [String: [DumpSubPlane]] = [:]

    public init?(path: String) {
        FileManager.default.createFile(atPath: path, contents: nil, attributes: nil)
        guard let h = FileHandle(forWritingAtPath: path) else {
            return nil
        }
        self.handle = h
        h.write(Data("VSD1".utf8))
    }

    public func close() {
        lock.lock()
        try? handle.close()
        lock.unlock()
    }

    static func assembleQuadrants(blocks: [BlockView], planeW: Int, planeH: Int, blockSize: Int, includeLL: Bool) -> [DumpSubPlane] {
        let q = blockSize / 2
        let colCount = (planeW + blockSize - 1) / blockSize
        let rowCount = (planeH + blockSize - 1) / blockSize
        let gw = colCount * q
        let gh = rowCount * q
        let offsets: [(x: Int, y: Int)]
        if includeLL {
            offsets = [(0, 0), (q, 0), (0, q), (q, q)]
        } else {
            offsets = [(q, 0), (0, q), (q, q)]
        }
        var planes = offsets.map { _ in DumpSubPlane(width: gw, height: gh, data: [Int16](repeating: 0, count: gw * gh)) }
        for r in 0..<rowCount {
            for c in 0..<colCount {
                let blk = blocks[r * colCount + c]
                for (pi, off) in offsets.enumerated() {
                    planes[pi].data.withUnsafeMutableBufferPointer { dst in
                        let dBase = dst.baseAddress!
                        for y in 0..<q {
                            let src = blk.base.advanced(by: (off.y + y) * blk.stride + off.x)
                            dBase.advanced(by: (r * q + y) * gw + c * q).update(from: src, count: q)
                        }
                    }
                }
            }
        }
        return planes
    }

    private static func dwtLevelPlanes(_ plane: [Int16], w: Int, h: Int, blockSize: Int, includeLL: Bool) -> (subs: [DumpSubPlane], ll: [Int16], llW: Int, llH: Int) {
        let q = blockSize / 2
        let colCount = (w + blockSize - 1) / blockSize
        let rowCount = (h + blockSize - 1) / blockSize
        let gw = colCount * q
        let gh = rowCount * q
        let offsets: [(x: Int, y: Int)]
        if includeLL {
            offsets = [(0, 0), (q, 0), (0, q), (q, q)]
        } else {
            offsets = [(q, 0), (0, q), (q, q)]
        }
        var subs = offsets.map { _ in DumpSubPlane(width: gw, height: gh, data: [Int16](repeating: 0, count: gw * gh)) }
        let llW = (w + 1) / 2
        let llH = (h + 1) / 2
        var ll = [Int16](repeating: 0, count: llW * llH)
        var scratch = [Int16](repeating: 0, count: blockSize * blockSize)
        let reader = Int16Reader(data: plane, width: w, height: h)
        for r in 0..<rowCount {
            for c in 0..<colCount {
                scratch.withUnsafeMutableBufferPointer { sb in
                    let sBase = sb.baseAddress!
                    let view = BlockView(base: sBase, width: blockSize, height: blockSize, stride: blockSize)
                    reader.readBlock(x: c * blockSize, y: r * blockSize, width: blockSize, height: blockSize, into: view)
                    switch blockSize {
                    case 32:
                        dwt2DBlock32(ptr: sBase, stride: blockSize)
                    case 16:
                        dwt2DBlock16(ptr: sBase, stride: blockSize)
                    default:
                        dwt2DBlock8(ptr: sBase, stride: blockSize)
                    }
                    for (pi, off) in offsets.enumerated() {
                        subs[pi].data.withUnsafeMutableBufferPointer { dst in
                            let dBase = dst.baseAddress!
                            for y in 0..<q {
                                let src = sBase.advanced(by: (off.y + y) * blockSize + off.x)
                                dBase.advanced(by: (r * q + y) * gw + c * q).update(from: src, count: q)
                            }
                        }
                    }
                    ll.withUnsafeMutableBufferPointer { dst in
                        let dBase = dst.baseAddress!
                        let dx0 = c * q
                        let dy0 = r * q
                        let copyW = min(q, llW - dx0)
                        if 0 < copyW {
                            for y in 0..<q {
                                let dy = dy0 + y
                                if dy < llH {
                                    let src = sBase.advanced(by: y * blockSize)
                                    dBase.advanced(by: dy * llW + dx0).update(from: src, count: copyW)
                                }
                            }
                        }
                    }
                }
            }
        }
        return (subs, ll, llW, llH)
    }

    private static func pyramid(_ plane: [Int16], w: Int, h: Int) -> [[DumpSubPlane]] {
        let l2 = dwtLevelPlanes(plane, w: w, h: h, blockSize: 32, includeLL: false)
        let l1 = dwtLevelPlanes(l2.ll, w: l2.llW, h: l2.llH, blockSize: 16, includeLL: false)
        let l0 = dwtLevelPlanes(l1.ll, w: l1.llW, h: l1.llH, blockSize: 8, includeLL: true)
        return [l2.subs, l1.subs, l0.subs]
    }

    func stash(_ key: String, blocks: [BlockView], planeW: Int, planeH: Int, blockSize: Int, includeLL: Bool) {
        let planes = Self.assembleQuadrants(blocks: blocks, planeW: planeW, planeH: planeH, blockSize: blockSize, includeLL: includeLL)
        lock.lock()
        pending[key] = planes
        lock.unlock()
    }

    func finalizePFrame(
        gopPosition: Int, width: Int, height: Int,
        predictedPd: PlaneData420,
        l1yBlocks: [BlockView], l1cbBlocks: [BlockView], l1crBlocks: [BlockView],
        b8yBlocks: [BlockView], b8cbBlocks: [BlockView], b8crBlocks: [BlockView],
        sub2W: Int, sub2H: Int, sub1W: Int, sub1H: Int,
        qtY2: QuantizationTable, qtC2: QuantizationTable,
        qtY1: QuantizationTable, qtC1: QuantizationTable,
        qtY0: QuantizationTable, qtC0: QuantizationTable,
        layer0Bytes: Int, layer1Bytes: Int, layer2Bytes: Int
    ) {
        let l1cbW = (sub2W + 1) / 2
        let l1cbH = (sub2H + 1) / 2
        let l0cbW = (sub1W + 1) / 2
        let l0cbH = (sub1H + 1) / 2
        let parents: [[DumpSubPlane]] = [
            Self.assembleQuadrants(blocks: l1yBlocks, planeW: sub2W, planeH: sub2H, blockSize: 16, includeLL: false),
            Self.assembleQuadrants(blocks: l1cbBlocks, planeW: l1cbW, planeH: l1cbH, blockSize: 16, includeLL: false),
            Self.assembleQuadrants(blocks: l1crBlocks, planeW: l1cbW, planeH: l1cbH, blockSize: 16, includeLL: false),
            Self.assembleQuadrants(blocks: b8yBlocks, planeW: sub1W, planeH: sub1H, blockSize: 8, includeLL: false),
            Self.assembleQuadrants(blocks: b8cbBlocks, planeW: l0cbW, planeH: l0cbH, blockSize: 8, includeLL: false),
            Self.assembleQuadrants(blocks: b8crBlocks, planeW: l0cbW, planeH: l0cbH, blockSize: 8, includeLL: false),
        ]

        let cbW = (width + 1) / 2
        let cbH = (height + 1) / 2
        let pyrY = Self.pyramid(predictedPd.y, w: width, h: height)
        let pyrCb = Self.pyramid(predictedPd.cb, w: cbW, h: cbH)
        let pyrCr = Self.pyramid(predictedPd.cr, w: cbW, h: cbH)
        let pyr: [[DumpSubPlane]] = [
            pyrY[0], pyrCb[0], pyrCr[0],
            pyrY[1], pyrCb[1], pyrCr[1],
            pyrY[2], pyrCb[2], pyrCr[2],
        ]

        lock.lock()
        defer { lock.unlock() }
        let codedKeys = ["L2Y", "L2Cb", "L2Cr", "L1Y", "L1Cb", "L1Cr", "L0Y", "L0Cb", "L0Cr"]
        var coded: [[DumpSubPlane]] = []
        coded.reserveCapacity(codedKeys.count)
        for key in codedKeys {
            guard let planes = pending[key] else {
                return
            }
            coded.append(planes)
        }
        pending.removeAll(keepingCapacity: true)

        var buf = [UInt8]()
        buf.reserveCapacity(1 << 23)
        buf.append(contentsOf: Array("FRAM".utf8))
        put32(gopPosition, &buf)
        put32(width, &buf)
        put32(height, &buf)
        for qt in [qtY2, qtC2, qtY1, qtC1, qtY0, qtC0] {
            put32(Int(qt.qLow.step), &buf)
            put32(Int(qt.qMid.step), &buf)
            put32(Int(qt.qHigh.step), &buf)
        }
        put32(layer0Bytes, &buf)
        put32(layer1Bytes, &buf)
        put32(layer2Bytes, &buf)
        for group in [coded, parents, pyr] {
            for entry in group {
                put32(entry.count, &buf)
                for p in entry {
                    put32(p.width, &buf)
                    put32(p.height, &buf)
                    p.data.withUnsafeBufferPointer { src in
                        buf.append(contentsOf: UnsafeRawBufferPointer(src))
                    }
                }
            }
        }
        handle.write(Data(buf))
    }

    @inline(__always)
    private func put32(_ v: Int, _ buf: inout [UInt8]) {
        let u = UInt32(truncatingIfNeeded: v)
        buf.append(UInt8(u & 0xff))
        buf.append(UInt8((u >> 8) & 0xff))
        buf.append(UInt8((u >> 16) & 0xff))
        buf.append(UInt8((u >> 24) & 0xff))
    }
}

// MARK: - Coefficient Dump Reader

public final class DumpReader {
    private let data: Data
    private var off: Int

    public init(path: String) throws {
        self.data = try Data(contentsOf: URL(fileURLWithPath: path), options: .mappedIfSafe)
        self.off = 0
        guard try magic() == "VSD1" else {
            throw TrainingError.badFormat("bad magic")
        }
    }

    public var atEnd: Bool {
        data.count <= off
    }

    private func magic() throws -> String {
        guard off + 4 <= data.count else {
            throw TrainingError.badFormat("truncated magic")
        }
        let s = String(decoding: data[data.startIndex + off ..< data.startIndex + off + 4], as: UTF8.self)
        off += 4
        return s
    }

    private func i32() throws -> Int {
        guard off + 4 <= data.count else {
            throw TrainingError.badFormat("truncated i32")
        }
        var u: UInt32 = 0
        _ = withUnsafeMutableBytes(of: &u) { dst in
            data.copyBytes(to: dst, from: data.startIndex + off ..< data.startIndex + off + 4)
        }
        off += 4
        return Int(Int32(bitPattern: UInt32(littleEndian: u)))
    }

    private func int16Array(_ count: Int) throws -> [Int16] {
        let byteCount = count * 2
        guard off + byteCount <= data.count else {
            throw TrainingError.badFormat("truncated plane")
        }
        var arr = [Int16](repeating: 0, count: count)
        _ = arr.withUnsafeMutableBytes { dst in
            data.copyBytes(to: dst, from: data.startIndex + off ..< data.startIndex + off + byteCount)
        }
        off += byteCount
        return arr
    }

    private func entry() throws -> DumpEntry {
        let nSub = try i32()
        guard 1 <= nSub && nSub <= 4 else {
            throw TrainingError.badFormat("bad subband count \(nSub)")
        }
        var subs: [DumpSubPlane] = []
        subs.reserveCapacity(nSub)
        for _ in 0..<nSub {
            let w = try i32()
            let h = try i32()
            subs.append(DumpSubPlane(width: w, height: h, data: try int16Array(w * h)))
        }
        return DumpEntry(subbands: subs)
    }

    func nextFrame() throws -> DumpFrame? {
        if atEnd {
            return nil
        }
        guard try magic() == "FRAM" else {
            throw TrainingError.badFormat("bad frame marker")
        }
        let gop = try i32()
        let width = try i32()
        let height = try i32()
        var qsteps: [[Int]] = []
        for _ in 0..<6 {
            qsteps.append([try i32(), try i32(), try i32()])
        }
        let layerBytes = [try i32(), try i32(), try i32()]
        var coded: [DumpEntry] = []
        for _ in 0..<9 {
            coded.append(try entry())
        }
        var parents: [DumpEntry] = []
        for _ in 0..<6 {
            parents.append(try entry())
        }
        var pyr: [DumpEntry] = []
        for _ in 0..<9 {
            pyr.append(try entry())
        }
        return DumpFrame(gop: gop, width: width, height: height, qsteps: qsteps, layerBytes: layerBytes, coded: coded, parents: parents, pyr: pyr)
    }
}

// MARK: - Token Counting and Walking

private struct TrainingFeatures {
    let isLscp: Bool
    let baseCtx: Int
    let prevNonZero: Bool
    let parentZero: Bool
}

private final class TrainingStreamCounter {
    var run: [[Int]]
    var val: [[Int]]
    var sharedBits: Int = 0

    init(contextCount: Int) {
        run = [[Int]](repeating: [Int](repeating: 0, count: 64), count: contextCount)
        val = [[Int]](repeating: [Int](repeating: 0, count: 64), count: contextCount)
    }

    @inline(__always)
    func addPair(runLen: Int, value: Int16, features: TrainingFeatures) {
        let rt = valueTokenizeUnsigned(UInt32(runLen))
        let vt = valueTokenize(value)
        sharedBits += rt.bypassLen + vt.bypassLen
        let ctx = features.isLscp ? 5 : features.baseCtx
        run[ctx][Int(rt.token)] += 1
        val[ctx][Int(vt.token)] += 1
    }
}

@inline(__always)
private func tileAllZero(_ p: DumpSubPlane, _ ox: Int, _ oy: Int, _ n: Int) -> Bool {
    for y in 0..<n {
        let ro = (oy + y) * p.width + ox
        for x in 0..<n {
            if p.data[ro + x] != 0 {
                return false
            }
        }
    }
    return true
}

@inline(__always)
private func walkTile(
    plane: DumpSubPlane, ox: Int, oy: Int, n: Int, scanV: Bool,
    parent: DumpSubPlane?, pox: Int, poy: Int,
    counter: TrainingStreamCounter
) {
    counter.sharedBits += 1  // hasNonZero flag

    var lscpIdx = -1 * 1
    for idx in 0..<(n * n) {
        let x: Int
        let y: Int
        if scanV {
            x = idx / n
            y = idx % n
        } else {
            x = idx % n
            y = idx / n
        }
        if plane.data[(oy + y) * plane.width + ox + x] != 0 {
            lscpIdx = idx
        }
    }
    if lscpIdx < 0 {
        return
    }

    let lx: Int
    let ly: Int
    if scanV {
        lx = lscpIdx / n
        ly = lscpIdx % n
    } else {
        lx = lscpIdx % n
        ly = lscpIdx / n
    }
    let lscpFeatures = TrainingFeatures(isLscp: true, baseCtx: 5, prevNonZero: false, parentZero: false)
    counter.addPair(runLen: lx, value: 0, features: lscpFeatures)
    counter.addPair(runLen: ly, value: 0, features: lscpFeatures)

    var run = 0
    var prevVal: Int16 = 0
    var startIdx = 0
    for idx in 0...lscpIdx {
        let x: Int
        let y: Int
        if scanV {
            x = idx / n
            y = idx % n
        } else {
            x = idx % n
            y = idx / n
        }
        let v = plane.data[(oy + y) * plane.width + ox + x]
        if run == 0 {
            startIdx = idx
        }
        if v == 0 {
            run += 1
        } else {
            let sx: Int
            let sy: Int
            if scanV {
                sx = startIdx / n
                sy = startIdx % n
            } else {
                sx = startIdx % n
                sy = startIdx / n
            }
            var parentZ = false
            if let par = parent {
                parentZ = par.data[(poy + (sy >> 1)) * par.width + pox + (sx >> 1)] == 0
            }
            var baseCtx = 0
            if parent != nil && parentZ {
                baseCtx += 2
            }
            if prevVal != 0 {
                baseCtx += 1
            }
            counter.addPair(
                runLen: run,
                value: v,
                features: TrainingFeatures(
                    isLscp: false,
                    baseCtx: baseCtx,
                    prevNonZero: prevVal != 0,
                    parentZero: parentZ
                )
            )
            prevVal = v
            run = 0
        }
    }
}

private func walkStreamUpper(entry: DumpEntry, parent: DumpEntry, tile: Int, counter: TrainingStreamCounter) {
    let half = tile / 2
    let hl = entry.subbands[0]
    let lh = entry.subbands[1]
    let hh = entry.subbands[2]
    let cols = hl.width / tile
    let rows = hl.height / tile
    for r in 0..<rows {
        for c in 0..<cols {
            let ox = c * tile
            let oy = r * tile
            if tileAllZero(hl, ox, oy, tile) && tileAllZero(lh, ox, oy, tile) && tileAllZero(hh, ox, oy, tile) {
                counter.sharedBits += 1
                continue
            }
            var allQuadrantsHaveContent = true
            for q in 0..<4 {
                let qx = (q & 1) * half
                let qy = (q >> 1) * half
                if tileAllZero(hl, ox + qx, oy + qy, half) && tileAllZero(lh, ox + qx, oy + qy, half) && tileAllZero(hh, ox + qx, oy + qy, half) {
                    allQuadrantsHaveContent = false
                }
            }
            if allQuadrantsHaveContent != true {
                counter.sharedBits += 10
                for q in 0..<4 {
                    let qx = (q & 1) * half
                    let qy = (q >> 1) * half
                    for s in 0..<3 {
                        walkTile(
                            plane: entry.subbands[s], ox: ox + qx, oy: oy + qy, n: half, scanV: false,
                            parent: parent.subbands[s], pox: c * half + qx / 2, poy: r * half + qy / 2,
                            counter: counter
                        )
                    }
                }
            } else {
                counter.sharedBits += 2
                walkTile(plane: hl, ox: ox, oy: oy, n: tile, scanV: true, parent: parent.subbands[0], pox: c * half, poy: r * half, counter: counter)
                walkTile(plane: lh, ox: ox, oy: oy, n: tile, scanV: false, parent: parent.subbands[1], pox: c * half, poy: r * half, counter: counter)
                walkTile(plane: hh, ox: ox, oy: oy, n: tile, scanV: false, parent: parent.subbands[2], pox: c * half, poy: r * half, counter: counter)
            }
        }
    }
}

private func walkStreamBase(entry: DumpEntry, counter: TrainingStreamCounter) {
    let ll = entry.subbands[0]
    let hl = entry.subbands[1]
    let lh = entry.subbands[2]
    let hh = entry.subbands[3]
    let cols = ll.width / 4
    let rows = ll.height / 4
    for r in 0..<rows {
        for c in 0..<cols {
            let ox = c * 4
            let oy = r * 4
            counter.sharedBits += 2  // zero flag + reserved flag
            if tileAllZero(ll, ox, oy, 4) && tileAllZero(hl, ox, oy, 4) && tileAllZero(lh, ox, oy, 4) && tileAllZero(hh, ox, oy, 4) {
                continue
            }
            walkTile(plane: ll, ox: ox, oy: oy, n: 4, scanV: false, parent: nil, pox: 0, poy: 0, counter: counter)
            walkTile(plane: hl, ox: ox, oy: oy, n: 4, scanV: true, parent: nil, pox: 0, poy: 0, counter: counter)
            walkTile(plane: lh, ox: ox, oy: oy, n: 4, scanV: false, parent: nil, pox: 0, poy: 0, counter: counter)
            walkTile(plane: hh, ox: ox, oy: oy, n: 4, scanV: false, parent: nil, pox: 0, poy: 0, counter: counter)
        }
    }
}

private func walkBase6(frame: DumpFrame, sink: (_ e: Int, _ run: [[Int]], _ val: [[Int]], _ shared: Int) -> Void) {
    for e in 0..<9 {
        let entry = frame.coded[e]
        let counter = TrainingStreamCounter(contextCount: 6)
        switch e / 3 {
        case 0:
            walkStreamUpper(entry: entry, parent: frame.parents[e], tile: 16, counter: counter)
        case 1:
            walkStreamUpper(entry: entry, parent: frame.parents[e], tile: 8, counter: counter)
        default:
            walkStreamBase(entry: entry, counter: counter)
        }
        sink(e, counter.run, counter.val, counter.sharedBits)
    }
}

// MARK: - Cost Simulation and Training

private final class TrainingBackwardState {
    var run: [[[Int]]]
    var val: [[[Int]]]
    var primed: [Bool]

    init(streams: Int) {
        run = (0..<streams).map { _ in [[Int]](repeating: [Int](repeating: 0, count: 64), count: entropyContextCount) }
        val = run
        primed = [Bool](repeating: false, count: streams)
    }
}

private func costBackwardQ8(curRun: [[Int]], curVal: [[Int]], accRun: [[Int]], accVal: [[Int]]) -> Int {
    var mr = [Int](repeating: 0, count: 64)
    var mv = [Int](repeating: 0, count: 64)
    for c in accRun.indices {
        for t in 0..<64 {
            mr[t] += accRun[c][t]
            mv[t] += accVal[c][t]
        }
    }
    var mergedRun = rANSModel(buildLUT: false)
    var mergedVal = rANSModel(buildLUT: false)
    mergedRun.normalize(tokenCounts: mr)
    mergedVal.normalize(tokenCounts: mv)

    var q8 = 0
    for c in curRun.indices {
        let curSum = curRun[c].reduce(0, +) + curVal[c].reduce(0, +)
        if curSum == 0 {
            continue
        }
        let accSum = accRun[c].reduce(0, +) + accVal[c].reduce(0, +)
        if 0 < accSum {
            var rm = rANSModel(buildLUT: false)
            var vm = rANSModel(buildLUT: false)
            rm.normalize(tokenCounts: accRun[c])
            vm.normalize(tokenCounts: accVal[c])
            q8 += estimateBitCostQ8(tokenCounts: curRun[c], model: rm)
            q8 += estimateBitCostQ8(tokenCounts: curVal[c], model: vm)
        } else {
            q8 += estimateBitCostQ8(tokenCounts: curRun[c], model: mergedRun)
            q8 += estimateBitCostQ8(tokenCounts: curVal[c], model: mergedVal)
        }
    }
    return q8
}

private func staticCostQ8(run: [[Int]], val: [[Int]], statRun: [rANSModel], statVal: [rANSModel]) -> Int {
    var q8 = 0
    for c in 0..<entropyContextCount {
        q8 += estimateBitCostQ8(tokenCounts: run[c], model: statRun[c])
        q8 += estimateBitCostQ8(tokenCounts: val[c], model: statVal[c])
    }
    return q8
}

private func dynAndMergedCostQ8(run: [[Int]], val: [[Int]]) -> (dyn: Int, merged: Int) {
    var dynQ8 = 0
    var dynHdrBits = 0
    var mr = [Int](repeating: 0, count: 64)
    var mv = [Int](repeating: 0, count: 64)
    for c in 0..<entropyContextCount {
        for t in 0..<64 {
            mr[t] += run[c][t]
            mv[t] += val[c][t]
        }
        var rm = rANSModel(buildLUT: false)
        var vm = rANSModel(buildLUT: false)
        rm.normalize(tokenCounts: run[c])
        vm.normalize(tokenCounts: val[c])
        dynQ8 += estimateBitCostQ8(tokenCounts: run[c], model: rm)
        dynQ8 += estimateBitCostQ8(tokenCounts: val[c], model: vm)
        dynHdrBits += headerCostBits(model: rm) + headerCostBits(model: vm)
    }
    dynQ8 += dynHdrBits << 8

    var rmMerged = rANSModel(buildLUT: false)
    var vmMerged = rANSModel(buildLUT: false)
    rmMerged.normalize(tokenCounts: mr)
    vmMerged.normalize(tokenCounts: mv)
    var mergedQ8 = estimateBitCostQ8(tokenCounts: mr, model: rmMerged) + estimateBitCostQ8(tokenCounts: mv, model: vmMerged)
    mergedQ8 += (headerCostBits(model: rmMerged) + headerCostBits(model: vmMerged)) << 8

    return (dynQ8, mergedQ8)
}

private func emitTable(_ name: String, _ freqs: [UInt32]) -> String {
    var s = "    var \(name) = buildStaticModel(rawFreqs: [\n"
    for r in 0..<4 {
        let slice = freqs[r * 16 ..< (r + 1) * 16]
        let line = slice.map { String(format: "%4d", $0) }.joined(separator: ", ")
        s += "        \(line),\n"
    }
    s += "    ])\n\n"
    return s
}

public func runTableTraining(trainPath: String, testPath: String) throws -> String {
    try runTableTraining(trainPath: trainPath, testPath: testPath, parentFree: false)
}

public func runTableTraining(trainPath: String, testPath: String, parentFree: Bool) throws -> String {
    var gRun = [[Int]](repeating: [Int](repeating: 0, count: 64), count: entropyContextCount)
    var gVal = [[Int]](repeating: [Int](repeating: 0, count: 64), count: entropyContextCount)
    var trainFrames = 0
    for path in trainPath.split(separator: ",").map(String.init) {
        let trainReader = try DumpReader(path: path)
        while let frame = try trainReader.nextFrame() {
            walkBase6(frame: frame) { _, run, val, _ in
                for c in 0..<entropyContextCount {
                    for t in 0..<64 {
                        gRun[c][t] += run[c][t]
                        gVal[c][t] += val[c][t]
                    }
                }
            }
            trainFrames += 1
        }
    }
    guard 0 < trainFrames else {
        throw TrainingError.emptyDump
    }

    func remapParentFree(_ counts: inout [[Int]]) {
        for t in 0..<64 {
            counts[0][t] += counts[2][t]
            counts[1][t] += counts[3][t]
            counts[2][t] = 0
            counts[3][t] = 0
        }
    }
    if parentFree {
        remapParentFree(&gRun)
        remapParentFree(&gVal)
    }

    func trained(_ counts: [Int]) -> rANSModel {
        var m = rANSModel(buildLUT: false)
        m.normalize(tokenCounts: counts)
        return m
    }

    let shipped = StaticRANSModels.shared
    var newRun = (0..<entropyContextCount).map { trained(gRun[$0]) }
    var newVal = (0..<entropyContextCount).map { trained(gVal[$0]) }
    newRun[4] = shipped.dpcmRunModel
    newVal[4] = shipped.dpcmValModel
    newVal[5] = shipped.dpcmValModel
    if parentFree {
        newRun[2] = shipped.runModel2
        newRun[3] = shipped.runModel3
        newVal[2] = shipped.valModel2
        newVal[3] = shipped.valModel3
    }

    let s = StaticRANSModels.shared
    let oldRun: [rANSModel]
    let oldVal: [rANSModel]
    if parentFree {
        oldRun = [s.pfRunModel0, s.pfRunModel1, s.runModel2, s.runModel3, s.dpcmRunModel, s.lscpRunModel]
        oldVal = [s.pfValModel0, s.pfValModel1, s.valModel2, s.valModel3, s.dpcmValModel, s.dpcmValModel]
    } else {
        oldRun = [s.runModel0, s.runModel1, s.runModel2, s.runModel3, s.dpcmRunModel, s.lscpRunModel]
        oldVal = [s.valModel0, s.valModel1, s.valModel2, s.valModel3, s.dpcmValModel, s.dpcmValModel]
    }

    let testReader = try DumpReader(path: testPath)
    let backward = TrainingBackwardState(streams: 9)
    var bitsOld = 0.0
    var bitsNew = 0.0
    var staticPickedOld = 0
    var staticPickedNew = 0
    var streams = 0
    var testFrames = 0
    while let frame = try testReader.nextFrame() {
        walkBase6(frame: frame) { e, rawRun, rawVal, _ in
            var run = rawRun
            var val = rawVal
            if parentFree {
                remapParentFree(&run)
                remapParentFree(&val)
            }
            let stOld = staticCostQ8(run: run, val: val, statRun: oldRun, statVal: oldVal)
            let stNew = staticCostQ8(run: run, val: val, statRun: newRun, statVal: newVal)
            let (dyn, merged) = dynAndMergedCostQ8(run: run, val: val)
            let hist: Int
            if backward.primed[e] {
                hist = costBackwardQ8(curRun: run, curVal: val, accRun: backward.run[e], accVal: backward.val[e])
            } else {
                hist = Int.max
            }
            let bestOld = min(stOld, dyn, merged, hist)
            let bestNew = min(stNew, dyn, merged, hist)
            bitsOld += Double(bestOld) / 256.0
            bitsNew += Double(bestNew) / 256.0
            if bestOld == stOld {
                staticPickedOld += 1
            }
            if bestNew == stNew {
                staticPickedNew += 1
            }
            streams += 1
            for cc in 0..<entropyContextCount {
                for t in 0..<64 {
                    backward.run[e][cc][t] = backward.run[e][cc][t] / 2 + run[cc][t]
                    backward.val[e][cc][t] = backward.val[e][cc][t] / 2 + val[cc][t]
                }
            }
            backward.primed[e] = true
        }
        testFrames += 1
    }
    guard 0 < testFrames else {
        throw TrainingError.emptyDump
    }

    var out = "=== static table retraining ===\n"
    out += "train: \(trainPath) (\(trainFrames) frames)\ntest:  \(testPath) (\(testFrames) frames)\n\n"
    out += String(format: "coded bits under real selection (static/dyn/merged/history):\n")
    out += String(format: "  shipped tables:   %10.1f KB (static picked %d/%d streams)\n", bitsOld / 8.0 / 1024.0, staticPickedOld, streams)
    out += String(format: "  retrained tables: %10.1f KB (static picked %d/%d streams)\n", bitsNew / 8.0 / 1024.0, staticPickedNew, streams)
    out += String(format: "  delta: %+.2f%%\n\n", (bitsOld - bitsNew) / bitsOld * 100.0)

    out += "drop-in Swift (StaticRANSModels; dpcm models intentionally kept as shipped):\n\n"
    if parentFree {
        for (n, c) in [("pfRunModel0", 0), ("pfRunModel1", 1)] {
            out += emitTable(n, newRun[c].tokenFreqs)
        }
        for (n, c) in [("pfValModel0", 0), ("pfValModel1", 1)] {
            out += emitTable(n, newVal[c].tokenFreqs)
        }
    } else {
        let names = [("runModel0", 0), ("runModel1", 1), ("runModel2", 2), ("runModel3", 3), ("lscpRunModel", 5)]
        for (n, c) in names {
            out += emitTable(n, newRun[c].tokenFreqs)
        }
        for (n, c) in [("valModel0", 0), ("valModel1", 1), ("valModel2", 2), ("valModel3", 3)] {
            out += emitTable(n, newVal[c].tokenFreqs)
        }
    }
    return out
}

// MARK: - Training Dump Encoder Pipeline

public final class TrainingDumpEncoder: @unchecked Sendable {
    public init() {}

    public func dump(
        inputPath: String,
        outputPath: String,
        profile: UInt8 = 0x01,
        maxbitrate: Int = 500_000,
        qstep: Int? = nil
    ) async throws {
        guard let dumper = CoeffDumpWriter(path: outputPath) else {
            throw TrainingError.invalidInput("Failed to open output dump file: \(outputPath)")
        }
        defer {
            dumper.close()
        }

        guard let fh = FileHandle(forReadingAtPath: inputPath) else {
            throw TrainingError.invalidInput("Failed to open input y4m file: \(inputPath)")
        }
        defer {
            try? fh.close()
        }

        let y4m = try Y4MReader(fileHandle: fh)
        var fps = 30
        if y4m.fpsHeader.starts(with: "F") {
            let parts = y4m.fpsHeader.dropFirst().split(separator: ":")
            if parts.count == 2, let num = Int(parts[0]), let den = Int(parts[1]), 0 < den {
                let parsed = num / den
                if parsed != 0 {
                    fps = parsed
                }
            }
        }
        let encoder = VEVCEncoder(
            width: y4m.width,
            height: y4m.height,
            qstep: qstep,
            maxbitrate: maxbitrate,
            framerate: fps,
            profile: profile,
            dumpWriter: dumper
        )
        while let frame = try y4m.readFrame() {
            _ = try await encoder.encode(image: frame)
        }
    }
}
