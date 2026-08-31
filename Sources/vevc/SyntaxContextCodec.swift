import Foundation

// MARK: - Context-conditioned adaptive rANS for the non-coefficient syntax
//
// skipMap / refDir / treeMap used to be a raw bitmap (refDir, treeMap) or an
// RLE with per-frame frequency tables (skipMap). All three are now coded with
// context-conditioned adaptive rANS: no frequency table is ever transmitted,
// the models adapt backwards from symbols both sides have already processed,
// and every context is built only from already-decoded data.
//
// Frequencies always sum to exactly `rANSScale`, which is what
// rANSEncoder/rANSDecoder require. Adaptation moves mass between symbols of
// the same context so the sum is preserved exactly, with no division.
//
// Patent lineage: context template + adaptive frequency is the JBIG / PPM
// lineage (expired), rANS is public domain. No CABAC-style LPS/MPS state
// machine and no SAO-style structure is used.

private let adaptiveModelFrequencyTotal: UInt32 = rANSScale
private let adaptiveModelMinFrequency: UInt32 = 24
private let adaptiveModelAdaptationShift: UInt32 = 5

/// Two-symbol adaptive model. Symbol 0 owns `[0, f0)`, symbol 1 owns
/// `[f0, total)`.
struct AdaptiveBinModel {
    var f0: UInt32

    /// `oneIn` seeds p(1) as 1/oneIn. refDir is 0.25-2% ones, so starting at
    /// 1/2 would waste the first few hundred symbols of every frame.
    init(oneIn: UInt32 = 2) {
        let f1 = max(adaptiveModelMinFrequency, adaptiveModelFrequencyTotal / max(1, oneIn))
        self.f0 = adaptiveModelFrequencyTotal - f1
    }

    @inline(__always)
    func interval(_ sym: Int) -> (cum: UInt32, freq: UInt32) {
        if sym == 0 {
            return (0, f0)
        }
        return (f0, adaptiveModelFrequencyTotal - f0)
    }

    @inline(__always)
    func symbol(forCumFreq cf: UInt32) -> Int {
        if cf < f0 {
            return 0
        }
        return 1
    }

    @inline(__always)
    mutating func update(_ sym: Int) {
        if sym == 0 {
            let f1 = adaptiveModelFrequencyTotal - f0
            var d: UInt32 = 0
            if adaptiveModelMinFrequency < f1 {
                d = (f1 - adaptiveModelMinFrequency) >> adaptiveModelAdaptationShift
            }
            f0 += d
        } else {
            var d: UInt32 = 0
            if adaptiveModelMinFrequency < f0 {
                d = (f0 - adaptiveModelMinFrequency) >> adaptiveModelAdaptationShift
            }
            f0 -= d
        }
    }
}

/// Three-symbol adaptive model, frequencies summing to `rANSScale`.
struct AdaptiveTriModel {
    var f: (UInt32, UInt32, UInt32)

    init() {
        let third = adaptiveModelFrequencyTotal / 3
        self.f = (third, third, adaptiveModelFrequencyTotal - (2 * third))
    }

    @inline(__always)
    func freq(_ i: Int) -> UInt32 {
        switch i {
        case 0: return f.0
        case 1: return f.1
        default: return f.2
        }
    }

    @inline(__always)
    mutating func setFreq(_ i: Int, _ v: UInt32) {
        switch i {
        case 0: f.0 = v
        case 1: f.1 = v
        default: f.2 = v
        }
    }

    @inline(__always)
    func interval(_ sym: Int) -> (cum: UInt32, freq: UInt32) {
        switch sym {
        case 0: return (0, f.0)
        case 1: return (f.0, f.1)
        default: return (f.0 + f.1, f.2)
        }
    }

    @inline(__always)
    func symbol(forCumFreq cf: UInt32) -> Int {
        if cf < f.0 {
            return 0
        }
        if cf < f.0 + f.1 {
            return 1
        }
        return 2
    }

    /// Moves mass from the two other symbols to `sym`, preserving the sum.
    @inline(__always)
    mutating func update(_ sym: Int) {
        var gained: UInt32 = 0
        for t in 0..<3 where t != sym {
            let v = freq(t)
            var d: UInt32 = 0
            if adaptiveModelMinFrequency < v {
                d = (v - adaptiveModelMinFrequency) >> adaptiveModelAdaptationShift
            }
            if 0 < d {
                setFreq(t, v - d)
                gained += d
            }
        }
        if 0 < gained {
            setFreq(sym, freq(sym) + gained)
        }
    }
}

/// Per-GOP syntax coding state. Reset at every coded I frame (periodic,
/// scene-change and floor-fired alike), so a decoder started at any I frame
/// — which is what GOP-parallel decoding does — reproduces the same states.
final class SyntaxContextModels: @unchecked Sendable {
    /// skipMap: spatial (left, up, upleft) x temporal (previous frame's mode
    /// at the same position, or "absent"). 27 * 4 = 108 contexts. Carried
    /// across frames within a GOP: the decoder always decodes the skipMap of
    /// a P frame, so both sides update in lockstep.
    var skipModels = [AdaptiveTriModel](repeating: AdaptiveTriModel(), count: 108)
    /// Previous frame's skipMap, for the temporal context. nil at a GOP start.
    var prevSkipMap: [BlockMode]? = nil

    func reset() {
        for i in 0..<skipModels.count {
            skipModels[i] = AdaptiveTriModel()
        }
        prevSkipMap = nil
    }

    /// Spatial context index, absent neighbours counted as `.inter` — the same
    /// convention the syntax-opportunity oracle was measured with.
    @inline(__always)
    static func skipSpatialContext(map: [BlockMode], i: Int, cols: Int) -> Int {
        let col = i % cols
        var l = 0
        if 0 < col {
            l = Int(map[i - 1].rawValue)
        }
        var u = 0
        if cols <= i {
            u = Int(map[i - cols].rawValue)
        }
        var ul = 0
        if cols <= i, 0 < col {
            ul = Int(map[(i - cols) - 1].rawValue)
        }
        return ((l * 9) + (u * 3)) + ul
    }

    /// Full context: spatial * 4 + temporal, temporal 3 = no previous frame.
    @inline(__always)
    static func skipContext(map: [BlockMode], i: Int, cols: Int, prev: [BlockMode]?) -> Int {
        let s = skipSpatialContext(map: map, i: i, cols: cols)
        var t = 3
        if let p = prev, i < p.count {
            t = Int(p[i].rawValue)
        }
        return (s * 4) + t
    }
}

// MARK: - Stream tags
//
// skipMap keeps its leading mode byte so a decoder built before this change
// rejects the new stream instead of misreading it: decodeSkipMap only accepts
// 0 (raw RLE) and 1 (rANS RLE) and throws invalidBlockData otherwise.
let skipMapModeContext: UInt8 = 2

// MARK: - skipMap

/// Context-coded skipMap. Encodes forward to snapshot each symbol's interval
/// under the model state the decoder will hold at that symbol, then emits the
/// rANS stream in reverse (rANS decodes in the opposite order it encodes).
@inline(__always)
func encodeSkipMapContext(map: [BlockMode], cols: Int, state: SyntaxContextModels) -> [UInt8] {
    var out = [UInt8]()
    out.append(skipMapModeContext)
    guard 0 < map.count, 0 < cols else {
        return out
    }

    let prev = state.prevSkipMap
    var cums = [UInt32](repeating: 0, count: map.count)
    var freqs = [UInt32](repeating: 0, count: map.count)
    for i in 0..<map.count {
        let context = SyntaxContextModels.skipContext(map: map, i: i, cols: cols, prev: prev)
        let sym = Int(map[i].rawValue)
        let iv = state.skipModels[context].interval(sym)
        cums[i] = iv.cum
        freqs[i] = iv.freq
        state.skipModels[context].update(sym)
    }

    var enc = rANSEncoder()
    for i in stride(from: map.count - 1, through: 0, by: -1) {
        enc.encodeSymbol(cumFreq: cums[i], freq: freqs[i])
    }
    enc.flush()
    out.append(contentsOf: enc.getBitstream())
    state.prevSkipMap = map
    return out
}

@inline(__always)
func decodeSkipMapContext(data: [UInt8], count: Int, cols: Int, state: SyntaxContextModels) throws -> [BlockMode] {
    guard 0 < data.count, data[0] == skipMapModeContext else {
        throw DecodeError.invalidBlockData
    }
    guard 0 < count, 0 < cols else {
        return []
    }
    guard 5 <= data.count else {
        throw DecodeError.insufficientData
    }

    var map = [BlockMode](repeating: .inter, count: count)
    let prev = state.prevSkipMap
    try data.withUnsafeBufferPointer { buf in
        let base = buf.baseAddress!
        var dec = rANSDecoder(base: base.advanced(by: 1), count: data.count - 1)
        for i in 0..<count {
            let context = SyntaxContextModels.skipContext(map: map, i: i, cols: cols, prev: prev)
            let sym = state.skipModels[context].symbol(forCumFreq: dec.getCumulativeFreq())
            guard let mode = BlockMode(rawValue: UInt8(sym)) else {
                throw DecodeError.invalidBlockData
            }
            map[i] = mode
            let iv = state.skipModels[context].interval(sym)
            dec.advanceSymbol(cumFreq: iv.cum, freq: iv.freq)
            state.skipModels[context].update(sym)
        }
    }
    state.prevSkipMap = map
    return map
}

// MARK: - refDir
//
// Frame-local models: the decoder only decodes refDir when it needs it
// (`nextPd != nil` in parseProfile2Frame), so a model carried across frames
// would diverge from the encoder's. Adaptation therefore restarts every
// frame, seeded at p(1) = 1/16 because refDir is 0.25-2% ones.

@inline(__always)
func encodeRefDirsContextProfile2(refDirs: [Bool], skipMap: [BlockMode]) -> [UInt8] {
    var syms = [Int]()
    syms.reserveCapacity(skipMap.count)
    for i in 0..<skipMap.count where skipMap[i] == .inter {
        var sym = 0
        if refDirs[i] {
            sym = 1
        }
        syms.append(sym)
    }
    if syms.isEmpty {
        return []
    }

    var model = AdaptiveBinModel(oneIn: 16)
    var cums = [UInt32](repeating: 0, count: syms.count)
    var freqs = [UInt32](repeating: 0, count: syms.count)
    for i in 0..<syms.count {
        let iv = model.interval(syms[i])
        cums[i] = iv.cum
        freqs[i] = iv.freq
        model.update(syms[i])
    }
    var enc = rANSEncoder()
    for i in stride(from: syms.count - 1, through: 0, by: -1) {
        enc.encodeSymbol(cumFreq: cums[i], freq: freqs[i])
    }
    enc.flush()
    return enc.getBitstream()
}

@inline(__always)
func decodeRefDirsContextProfile2(buf: [UInt8], count: Int, skipMap: [BlockMode]?) -> [Bool] {
    var refDirs = [Bool](repeating: false, count: count)
    guard let sm = skipMap else {
        return refDirs
    }
    // Skip blocks carry no transmitted bit: their reference direction follows
    // from the mode itself (skip_ltr reads the long-term reference, skip_prev
    // the previous frame). Only inter blocks are coded.
    for i in 0..<min(count, sm.count) {
        switch sm[i] {
        case .skip_ltr: refDirs[i] = true
        case .skip_prev: refDirs[i] = false
        case .inter: break
        }
    }
    guard 4 <= buf.count else {
        return refDirs
    }
    var model = AdaptiveBinModel(oneIn: 16)
    buf.withUnsafeBufferPointer { b in
        let base = b.baseAddress!
        var dec = rANSDecoder(base: base, count: buf.count)
        for i in 0..<min(count, sm.count) where sm[i] == .inter {
            let sym = model.symbol(forCumFreq: dec.getCumulativeFreq())
            refDirs[i] = (sym == 1)
            let iv = model.interval(sym)
            dec.advanceSymbol(cumFreq: iv.cum, freq: iv.freq)
            model.update(sym)
        }
    }
    return refDirs
}

// MARK: - treeMap
//
// Frame-local models for the same reason as refDir: the decoder only decodes
// the treeMap when it has a skipMap. Context is (left, up) over each plane's
// own block grid, 3 states each (0 / 1 / not coded) = 9 contexts per plane,
// and the three planes are independent.

@inline(__always)
private func treeMapContext(isTreez: [Bool], isSkip: [Bool], i: Int, cols: Int) -> Int {
    var l = 2
    if 0 < cols, 0 < (i % cols), isSkip[i - 1] != true {
        l = 0
        if isTreez[i - 1] {
            l = 1
        }
    }
    var u = 2
    if 0 < cols, cols <= i, isSkip[i - cols] != true {
        u = 0
        if isTreez[i - cols] {
            u = 1
        }
    }
    return (l * 3) + u
}

@inline(__always)
private func encodeTreePlaneContext(isTreez: [Bool], isSkip: [Bool], cols: Int) -> [UInt8] {
    var idxs = [Int]()
    idxs.reserveCapacity(isSkip.count)
    for i in 0..<isSkip.count where isSkip[i] != true {
        idxs.append(i)
    }
    if idxs.isEmpty {
        return []
    }

    var models = [AdaptiveBinModel](repeating: AdaptiveBinModel(oneIn: 2), count: 9)
    var cums = [UInt32](repeating: 0, count: idxs.count)
    var freqs = [UInt32](repeating: 0, count: idxs.count)
    for k in 0..<idxs.count {
        let i = idxs[k]
        let context = treeMapContext(isTreez: isTreez, isSkip: isSkip, i: i, cols: cols)
        var sym = 0
        if isTreez[i] {
            sym = 1
        }
        let iv = models[context].interval(sym)
        cums[k] = iv.cum
        freqs[k] = iv.freq
        models[context].update(sym)
    }
    var enc = rANSEncoder()
    for k in stride(from: idxs.count - 1, through: 0, by: -1) {
        enc.encodeSymbol(cumFreq: cums[k], freq: freqs[k])
    }
    enc.flush()
    return enc.getBitstream()
}

@inline(__always)
private func decodeTreePlaneContext(base: UnsafePointer<UInt8>, count: Int, isSkip: [Bool], cols: Int) -> [Bool] {
    var isTreez = [Bool](repeating: false, count: isSkip.count)
    guard isSkip.contains(false), 4 <= count else {
        return isTreez
    }
    var models = [AdaptiveBinModel](repeating: AdaptiveBinModel(oneIn: 2), count: 9)
    var dec = rANSDecoder(base: base, count: count)
    for i in 0..<isSkip.count where isSkip[i] != true {
        let context = treeMapContext(isTreez: isTreez, isSkip: isSkip, i: i, cols: cols)
        let sym = models[context].symbol(forCumFreq: dec.getCumulativeFreq())
        isTreez[i] = (sym == 1)
        let iv = models[context].interval(sym)
        dec.advanceSymbol(cumFreq: iv.cum, freq: iv.freq)
        models[context].update(sym)
    }
    return isTreez
}

/// Three planes, each length-prefixed with a VLQ so the decoder can find the
/// plane boundaries (the old format derived them from the skip counts, which
/// only worked because every plane was a fixed-width bitmap).
@inline(__always)
func encodeTreeMapContextProfile2(
    isTreezY: [Bool], ySkip: [Bool], colsY: Int,
    isTreezCb: [Bool], cbSkip: [Bool],
    isTreezCr: [Bool], crSkip: [Bool], colsC: Int
) -> [UInt8] {
    let bufY = encodeTreePlaneContext(isTreez: isTreezY, isSkip: ySkip, cols: colsY)
    let bufCb = encodeTreePlaneContext(isTreez: isTreezCb, isSkip: cbSkip, cols: colsC)
    let bufCr = encodeTreePlaneContext(isTreez: isTreezCr, isSkip: crSkip, cols: colsC)
    var out = [UInt8]()
    writeVLQSize(&out, bufY.count)
    writeVLQSize(&out, bufCb.count)
    writeVLQSize(&out, bufCr.count)
    out.append(contentsOf: bufY)
    out.append(contentsOf: bufCb)
    out.append(contentsOf: bufCr)
    return out
}

@inline(__always)
func decodeTreeMapContextProfile2(
    buf: [UInt8],
    ySkip: [Bool], colsY: Int,
    cbSkip: [Bool], crSkip: [Bool], colsC: Int
) throws -> (isTreezY: [Bool], isTreezCb: [Bool], isTreezCr: [Bool]) {
    return try buf.withUnsafeBufferPointer { b -> (isTreezY: [Bool], isTreezCb: [Bool], isTreezCr: [Bool]) in
        let base = b.baseAddress!
        var offset = 0
        let nY = try readVLQSize(base, at: &offset, count: buf.count)
        let nCb = try readVLQSize(base, at: &offset, count: buf.count)
        let nCr = try readVLQSize(base, at: &offset, count: buf.count)
        guard ((offset + nY) + nCb) + nCr <= buf.count else {
            throw DecodeError.insufficientData
        }
        let tzY = decodeTreePlaneContext(base: base.advanced(by: offset), count: nY, isSkip: ySkip, cols: colsY)
        offset += nY
        let tzCb = decodeTreePlaneContext(base: base.advanced(by: offset), count: nCb, isSkip: cbSkip, cols: colsC)
        offset += nCb
        let tzCr = decodeTreePlaneContext(base: base.advanced(by: offset), count: nCr, isSkip: crSkip, cols: colsC)
        return (tzY, tzCb, tzCr)
    }
}
