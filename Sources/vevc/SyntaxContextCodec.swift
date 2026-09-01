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

/// 16-symbol adaptive model, frequencies summing to `rANSScale` (16384).
/// Backward-adaptive mass-transfer: sum is strictly preserved, no division.
struct AdaptiveModel16 {
    var freqs: [UInt32]
    var cumFreqs: [UInt32]

    init() {
        let initialFreq = adaptiveModelFrequencyTotal / 16
        self.freqs = [UInt32](repeating: initialFreq, count: 16)
        var cf = [UInt32](repeating: 0, count: 17)
        var sum: UInt32 = 0
        for i in 0..<16 {
            cf[i] = sum
            sum += initialFreq
        }
        cf[16] = sum
        self.cumFreqs = cf
    }

    @inline(__always)
    func interval(_ sym: Int) -> (cum: UInt32, freq: UInt32) {
        return (cumFreqs[sym], freqs[sym])
    }

    @inline(__always)
    func symbol(forCumFreq cf: UInt32) -> Int {
        for i in 0..<15 {
            if cf < cumFreqs[i + 1] {
                return i
            }
        }
        return 15
    }

    @inline(__always)
    mutating func update(_ sym: Int) {
        var gained: UInt32 = 0
        for t in 0..<16 where t != sym {
            let v = freqs[t]
            var d: UInt32 = 0
            if adaptiveModelMinFrequency < v {
                d = (v - adaptiveModelMinFrequency) >> adaptiveModelAdaptationShift
            }
            if 0 < d {
                freqs[t] = v - d
                gained += d
            }
        }
        if 0 < gained {
            freqs[sym] += gained
            var sum: UInt32 = 0
            for i in 0..<16 {
                cumFreqs[i] = sum
                sum += freqs[i]
            }
            cumFreqs[16] = sum
        }
    }
}

// MARK: - MV Tokenization & Classification

@inline(__always)
func mvOffsetBits(classIndex: Int) -> Int {
    switch classIndex {
    case 0...3: return 0
    case 4...5: return 1
    case 6...7: return 2
    case 8...9: return 3
    case 10: return 5
    case 11: return 6
    case 12: return 7
    case 13: return 8
    case 14: return 9
    default: return 11
    }
}

@inline(__always)
func mvOffsetBaseIndex(classIndex: Int) -> Int {
    switch classIndex {
    case 0...3: return 0
    case 4: return 0
    case 5: return 1
    case 6: return 2
    case 7: return 4
    case 8: return 6
    case 9: return 9
    case 10: return 12
    case 11: return 17
    case 12: return 23
    case 13: return 30
    case 14: return 38
    default: return 47
    }
}

@inline(__always)
func mvClassify(_ value: Int16) -> (classIndex: Int, sign: Int, offset: UInt32, bits: Int) {
    if value == 0 {
        return (0, 0, 0, 0)
    }
    var sign = 0
    if value < 0 {
        sign = 1
    }
    let m = UInt32(value.magnitude)
    switch m {
    case 1: return (1, sign, 0, 0)
    case 2: return (2, sign, 0, 0)
    case 3: return (3, sign, 0, 0)
    case 4...5: return (4, sign, m - 4, 1)
    case 6...7: return (5, sign, m - 6, 1)
    case 8...11: return (6, sign, m - 8, 2)
    case 12...15: return (7, sign, m - 12, 2)
    case 16...23: return (8, sign, m - 16, 3)
    case 24...31: return (9, sign, m - 24, 3)
    case 32...63: return (10, sign, m - 32, 5)
    case 64...127: return (11, sign, m - 64, 6)
    case 128...255: return (12, sign, m - 128, 7)
    case 256...511: return (13, sign, m - 256, 8)
    case 512...1023: return (14, sign, m - 512, 9)
    default:
        let clamped = min(UInt32(2048), m)
        return (15, sign, clamped - 1024, 11)
    }
}

@inline(__always)
func mvDeclassify(classIndex: Int, sign: Int, offset: UInt32) -> Int16 {
    if classIndex == 0 {
        return 0
    }
    var m: UInt32 = 0
    switch classIndex {
    case 1: m = 1
    case 2: m = 2
    case 3: m = 3
    case 4: m = 4 + offset
    case 5: m = 6 + offset
    case 6: m = 8 + offset
    case 7: m = 12 + offset
    case 8: m = 16 + offset
    case 9: m = 24 + offset
    case 10: m = 32 + offset
    case 11: m = 64 + offset
    case 12: m = 128 + offset
    case 13: m = 256 + offset
    case 14: m = 512 + offset
    default: m = 1024 + offset
    }
    if sign == 1 {
        return Int16(-1 * Int(m))
    }
    return Int16(m)
}

/// Per-GOP syntax coding state. Reset at every coded I frame (periodic,
/// scene-change and floor-fired alike), so a decoder started at any I frame
/// — which is what GOP-parallel decoding does — reproduces the same states.
public final class SyntaxContextModels: @unchecked Sendable {
    /// skipMap: spatial (left, up, upleft) x temporal (previous frame's mode
    /// at the same position, or "absent"). 27 * 4 = 108 contexts. Carried
    /// across frames within a GOP: the decoder always decodes the skipMap of
    /// a P frame, so both sides update in lockstep.
    var skipModels = [AdaptiveTriModel](repeating: AdaptiveTriModel(), count: 108)
    /// Previous frame's skipMap, for the temporal context. nil at a GOP start.
    var prevSkipMap: [BlockMode]? = nil

    /// Motion vector adaptive models (#36 Phase 2 Stage 1 Order-0).
    var dxClassModel = AdaptiveModel16()
    var dyClassModel = AdaptiveModel16()

    var dxSignModels = [AdaptiveBinModel](repeating: AdaptiveBinModel(oneIn: 2), count: 16)
    var dySignModels = [AdaptiveBinModel](repeating: AdaptiveBinModel(oneIn: 2), count: 16)

    var dxOffsetModels = [AdaptiveBinModel](repeating: AdaptiveBinModel(oneIn: 2), count: 58)
    var dyOffsetModels = [AdaptiveBinModel](repeating: AdaptiveBinModel(oneIn: 2), count: 58)

    public init() {}

    public func reset() {
        for i in 0..<skipModels.count {
            skipModels[i] = AdaptiveTriModel()
        }
        prevSkipMap = nil
        dxClassModel = AdaptiveModel16()
        dyClassModel = AdaptiveModel16()
        for i in 0..<16 {
            dxSignModels[i] = AdaptiveBinModel(oneIn: 2)
            dySignModels[i] = AdaptiveBinModel(oneIn: 2)
        }
        for i in 0..<58 {
            dxOffsetModels[i] = AdaptiveBinModel(oneIn: 2)
            dyOffsetModels[i] = AdaptiveBinModel(oneIn: 2)
        }
    }

    func clone() -> SyntaxContextModels {
        let m = SyntaxContextModels()
        m.skipModels = self.skipModels
        m.prevSkipMap = self.prevSkipMap
        m.dxClassModel = self.dxClassModel
        m.dyClassModel = self.dyClassModel
        m.dxSignModels = self.dxSignModels
        m.dySignModels = self.dySignModels
        m.dxOffsetModels = self.dxOffsetModels
        m.dyOffsetModels = self.dyOffsetModels
        return m
    }

    func copyFrom(_ other: SyntaxContextModels) {
        self.skipModels = other.skipModels
        self.prevSkipMap = other.prevSkipMap
        self.dxClassModel = other.dxClassModel
        self.dyClassModel = other.dyClassModel
        self.dxSignModels = other.dxSignModels
        self.dySignModels = other.dySignModels
        self.dxOffsetModels = other.dxOffsetModels
        self.dyOffsetModels = other.dyOffsetModels
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

// MARK: - skipMap

/// Context-coded skipMap. Encodes forward to snapshot each symbol's interval
/// under the model state the decoder will hold at that symbol, then emits the
/// rANS stream in reverse (rANS decodes in the opposite order it encodes).
@inline(__always)
func encodeSkipMapContext(map: [BlockMode], cols: Int, state: SyntaxContextModels) -> [UInt8] {
    guard 0 < map.count, 0 < cols else {
        return []
    }

    let prev = state.prevSkipMap
    var cumulativeFreqs = [UInt32](repeating: 0, count: map.count)
    var frequencies = [UInt32](repeating: 0, count: map.count)
    for i in 0..<map.count {
        let context = SyntaxContextModels.skipContext(map: map, i: i, cols: cols, prev: prev)
        let sym = Int(map[i].rawValue)
        let interval = state.skipModels[context].interval(sym)
        cumulativeFreqs[i] = interval.cum
        frequencies[i] = interval.freq
        state.skipModels[context].update(sym)
    }

    var encoder = rANSEncoder()
    for i in stride(from: map.count - 1, through: 0, by: -1) {
        encoder.encodeSymbol(cumFreq: cumulativeFreqs[i], freq: frequencies[i])
    }
    encoder.flush()
    state.prevSkipMap = map
    return encoder.getBitstream()
}

@inline(__always)
public func decodeSkipMapContext(data: [UInt8], count: Int, cols: Int, state: SyntaxContextModels) throws -> [BlockMode] {
    guard 0 < count, 0 < cols else {
        return []
    }
    guard 4 <= data.count else {
        throw DecodeError.insufficientData
    }

    var map = [BlockMode](repeating: .inter, count: count)
    let prev = state.prevSkipMap
    try data.withUnsafeBufferPointer { buf in
        let base = buf.baseAddress!
        var decoder = rANSDecoder(base: base, count: data.count)
        for i in 0..<count {
            let context = SyntaxContextModels.skipContext(map: map, i: i, cols: cols, prev: prev)
            let sym = state.skipModels[context].symbol(forCumFreq: decoder.getCumulativeFreq())
            guard let mode = BlockMode(rawValue: UInt8(sym)) else {
                throw DecodeError.invalidBlockData
            }
            map[i] = mode
            let interval = state.skipModels[context].interval(sym)
            decoder.advanceSymbol(cumFreq: interval.cum, freq: interval.freq)
            state.skipModels[context].update(sym)
        }
    }
    state.prevSkipMap = map
    return map
}

// MARK: - Motion Vectors (Profile 2 Adaptive rANS)

private struct MVSymbolSnapshot {
    var cum: UInt32
    var freq: UInt32
}

@inline(__always)
private func encodeMVSnapshot(
    dxList: [Int16],
    dyList: [Int16],
    state: inout SyntaxContextModels
) -> [UInt8] {
    guard 0 < dxList.count else { return [] }
    var snapshots = [MVSymbolSnapshot]()
    snapshots.reserveCapacity(dxList.count * 8)

    for i in 0..<dxList.count {
        let dxVal = dxList[i]
        let dyVal = dyList[i]

        // dx
        let dxClassified = mvClassify(dxVal)
        let intervalDxClass = state.dxClassModel.interval(dxClassified.classIndex)
        snapshots.append(MVSymbolSnapshot(cum: intervalDxClass.cum, freq: intervalDxClass.freq))
        state.dxClassModel.update(dxClassified.classIndex)

        if 0 < dxClassified.classIndex {
            let intervalDxSign = state.dxSignModels[dxClassified.classIndex].interval(dxClassified.sign)
            snapshots.append(MVSymbolSnapshot(cum: intervalDxSign.cum, freq: intervalDxSign.freq))
            state.dxSignModels[dxClassified.classIndex].update(dxClassified.sign)

            if 0 < dxClassified.bits {
                let offsetModelBaseIndex = mvOffsetBaseIndex(classIndex: dxClassified.classIndex)
                for b in 0..<dxClassified.bits {
                    let shift = (dxClassified.bits - 1) - b
                    let bitVal = Int((dxClassified.offset >> shift) & 1)
                    let intervalBit = state.dxOffsetModels[offsetModelBaseIndex + b].interval(bitVal)
                    snapshots.append(MVSymbolSnapshot(cum: intervalBit.cum, freq: intervalBit.freq))
                    state.dxOffsetModels[offsetModelBaseIndex + b].update(bitVal)
                }
            }
        }

        // dy
        let dyClassified = mvClassify(dyVal)
        let intervalDyClass = state.dyClassModel.interval(dyClassified.classIndex)
        snapshots.append(MVSymbolSnapshot(cum: intervalDyClass.cum, freq: intervalDyClass.freq))
        state.dyClassModel.update(dyClassified.classIndex)

        if 0 < dyClassified.classIndex {
            let intervalDySign = state.dySignModels[dyClassified.classIndex].interval(dyClassified.sign)
            snapshots.append(MVSymbolSnapshot(cum: intervalDySign.cum, freq: intervalDySign.freq))
            state.dySignModels[dyClassified.classIndex].update(dyClassified.sign)

            if 0 < dyClassified.bits {
                let offsetModelBaseIndex = mvOffsetBaseIndex(classIndex: dyClassified.classIndex)
                for b in 0..<dyClassified.bits {
                    let shift = (dyClassified.bits - 1) - b
                    let bitVal = Int((dyClassified.offset >> shift) & 1)
                    let intervalBit = state.dyOffsetModels[offsetModelBaseIndex + b].interval(bitVal)
                    snapshots.append(MVSymbolSnapshot(cum: intervalBit.cum, freq: intervalBit.freq))
                    state.dyOffsetModels[offsetModelBaseIndex + b].update(bitVal)
                }
            }
        }
    }

    var encoder = rANSEncoder()
    for k in stride(from: snapshots.count - 1, through: 0, by: -1) {
        encoder.encodeSymbol(cumFreq: snapshots[k].cum, freq: snapshots[k].freq)
    }
    encoder.flush()
    return encoder.getBitstream()
}

@inline(__always)
func encodeMVsContextProfile2(
    mvs: MotionVectors,
    skipMap: [BlockMode]?,
    cols: Int,
    prevMVs: MotionVectors?,
    state: SyntaxContextModels,
    updateHistory: Bool
) -> [UInt8] {
    var rawDxs = [Int16]()
    var rawDys = [Int16]()
    var predDxs = [Int16]()
    var predDys = [Int16]()
    var valid = [Bool](repeating: false, count: mvs.count)

    for i in 0..<mvs.count {
        if let sm = skipMap, sm[i] != .inter {
            continue
        }
        valid[i] = true
        let origDx = mvs.dx[i]
        let origDy = mvs.dy[i]
        rawDxs.append(origDx)
        rawDys.append(origDy)

        let col = i % cols
        var aDx: Int16 = 0
        var aDy: Int16 = 0
        if 0 < col {
            aDx = mvs.dx[i - 1]
            aDy = mvs.dy[i - 1]
        }
        var bDx: Int16 = 0
        var bDy: Int16 = 0
        if cols <= i {
            bDx = mvs.dx[i - cols]
            bDy = mvs.dy[i - cols]
        }
        var cDx: Int16 = 0
        var cDy: Int16 = 0
        if cols <= i, col < (cols - 1) {
            cDx = mvs.dx[i - cols + 1]
            cDy = mvs.dy[i - cols + 1]
        }

        let predDx = max(min(Int(aDx), Int(bDx)), min(max(Int(aDx), Int(bDx)), Int(cDx)))
        let predDy = max(min(Int(aDy), Int(bDy)), min(max(Int(aDy), Int(bDy)), Int(cDy)))

        predDxs.append(Int16(Int(origDx) - predDx))
        predDys.append(Int16(Int(origDy) - predDy))
    }

    guard 0 < rawDxs.count else {
        return []
    }

    struct Candidate {
        let mode: UInt8
        let data: [UInt8]
        let finalState: SyntaxContextModels
    }
    var candidates = [Candidate]()

    // Mode 0: Raw
    var state0 = state.clone()
    let data0 = encodeMVSnapshot(dxList: rawDxs, dyList: rawDys, state: &state0)
    candidates.append(Candidate(mode: 0x00, data: data0, finalState: state0))

    // Mode 1: Spatial Pred
    var state1 = state.clone()
    let data1 = encodeMVSnapshot(dxList: predDxs, dyList: predDys, state: &state1)
    candidates.append(Candidate(mode: 0x01, data: data1, finalState: state1))

    // Mode 2: Temporal Pred
    if let pm = prevMVs, pm.count == mvs.count {
        var tempDxs = [Int16]()
        var tempDys = [Int16]()
        tempDxs.reserveCapacity(rawDxs.count)
        tempDys.reserveCapacity(rawDys.count)
        for i in 0..<mvs.count {
            if valid[i] != true { continue }
            tempDxs.append(Int16(Int(mvs.dx[i]) - Int(pm.dx[i])))
            tempDys.append(Int16(Int(mvs.dy[i]) - Int(pm.dy[i])))
        }
        var state2 = state.clone()
        let data2 = encodeMVSnapshot(dxList: tempDxs, dyList: tempDys, state: &state2)
        candidates.append(Candidate(mode: 0x02, data: data2, finalState: state2))
    }

    var best = candidates[0]
    for c in candidates.dropFirst() where c.data.count < best.data.count {
        best = c
    }

    if updateHistory {
        state.copyFrom(best.finalState)
    }

    var out = [UInt8]()
    out.append(best.mode)
    out.append(contentsOf: best.data)
    return out
}

@inline(__always)
func decodeMVsContextProfile2(
    data: [UInt8],
    count: Int,
    skipMap: [BlockMode]?,
    cols: Int,
    prevMVs: MotionVectors?,
    state: SyntaxContextModels,
    updateHistory: Bool
) throws -> MotionVectors {
    var valid = [Bool](repeating: false, count: count)
    var codedCount = 0
    for i in 0..<count {
        if let sm = skipMap, sm[i] != .inter {
            continue
        }
        valid[i] = true
        codedCount += 1
    }

    guard 0 < codedCount else {
        return MotionVectors(dx: [Int16](repeating: 0, count: count), dy: [Int16](repeating: 0, count: count))
    }

    guard 0 < data.count else {
        throw DecodeError.insufficientData
    }
    let mode = data[0]
    guard mode <= 2 else {
        throw DecodeError.invalidBlockData
    }
    if mode == 2, prevMVs == nil {
        throw DecodeError.invalidBlockData
    }

    guard 5 <= data.count else {
        throw DecodeError.insufficientData
    }

    var mvsDx = [Int16](repeating: 0, count: count)
    var mvsDy = [Int16](repeating: 0, count: count)

    data.withUnsafeBufferPointer { buf in
        let base = buf.baseAddress!
        var decoder = rANSDecoder(base: base.advanced(by: 1), count: data.count - 1)

        for i in 0..<count {
            if valid[i] != true { continue }

            // dx
            let cumFreqX = decoder.getCumulativeFreq()
            let classIndexX = state.dxClassModel.symbol(forCumFreq: cumFreqX)
            let intervalClassX = state.dxClassModel.interval(classIndexX)
            decoder.advanceSymbol(cumFreq: intervalClassX.cum, freq: intervalClassX.freq)
            if updateHistory {
                state.dxClassModel.update(classIndexX)
            }

            var signX = 0
            var offsetX: UInt32 = 0
            if 0 < classIndexX {
                let cumFreqSignX = decoder.getCumulativeFreq()
                signX = state.dxSignModels[classIndexX].symbol(forCumFreq: cumFreqSignX)
                let intervalSignX = state.dxSignModels[classIndexX].interval(signX)
                decoder.advanceSymbol(cumFreq: intervalSignX.cum, freq: intervalSignX.freq)
                if updateHistory {
                    state.dxSignModels[classIndexX].update(signX)
                }

                let bitsX = mvOffsetBits(classIndex: classIndexX)
                if 0 < bitsX {
                    let offsetModelBaseIndexX = mvOffsetBaseIndex(classIndex: classIndexX)
                    for b in 0..<bitsX {
                        let cumFreqBit = decoder.getCumulativeFreq()
                        let bitVal = state.dxOffsetModels[offsetModelBaseIndexX + b].symbol(forCumFreq: cumFreqBit)
                        let intervalBit = state.dxOffsetModels[offsetModelBaseIndexX + b].interval(bitVal)
                        decoder.advanceSymbol(cumFreq: intervalBit.cum, freq: intervalBit.freq)
                        if updateHistory {
                            state.dxOffsetModels[offsetModelBaseIndexX + b].update(bitVal)
                        }
                        offsetX = (offsetX << 1) | UInt32(bitVal)
                    }
                }
            }
            let resDx = mvDeclassify(classIndex: classIndexX, sign: signX, offset: offsetX)

            // dy
            let cumFreqY = decoder.getCumulativeFreq()
            let classIndexY = state.dyClassModel.symbol(forCumFreq: cumFreqY)
            let intervalClassY = state.dyClassModel.interval(classIndexY)
            decoder.advanceSymbol(cumFreq: intervalClassY.cum, freq: intervalClassY.freq)
            if updateHistory {
                state.dyClassModel.update(classIndexY)
            }

            var signY = 0
            var offsetY: UInt32 = 0
            if 0 < classIndexY {
                let cumFreqSignY = decoder.getCumulativeFreq()
                signY = state.dySignModels[classIndexY].symbol(forCumFreq: cumFreqSignY)
                let intervalSignY = state.dySignModels[classIndexY].interval(signY)
                decoder.advanceSymbol(cumFreq: intervalSignY.cum, freq: intervalSignY.freq)
                if updateHistory {
                    state.dySignModels[classIndexY].update(signY)
                }

                let bitsY = mvOffsetBits(classIndex: classIndexY)
                if 0 < bitsY {
                    let offsetModelBaseIndexY = mvOffsetBaseIndex(classIndex: classIndexY)
                    for b in 0..<bitsY {
                        let cumFreqBit = decoder.getCumulativeFreq()
                        let bitVal = state.dyOffsetModels[offsetModelBaseIndexY + b].symbol(forCumFreq: cumFreqBit)
                        let intervalBit = state.dyOffsetModels[offsetModelBaseIndexY + b].interval(bitVal)
                        decoder.advanceSymbol(cumFreq: intervalBit.cum, freq: intervalBit.freq)
                        if updateHistory {
                            state.dyOffsetModels[offsetModelBaseIndexY + b].update(bitVal)
                        }
                        offsetY = (offsetY << 1) | UInt32(bitVal)
                    }
                }
            }
            let resDy = mvDeclassify(classIndex: classIndexY, sign: signY, offset: offsetY)

            // Prediction
            switch mode {
            case 0:
                mvsDx[i] = resDx
                mvsDy[i] = resDy
            case 1:
                let col = i % cols
                var aDx: Int16 = 0
                var aDy: Int16 = 0
                if 0 < col {
                    aDx = mvsDx[i - 1]
                    aDy = mvsDy[i - 1]
                }
                var bDx: Int16 = 0
                var bDy: Int16 = 0
                if cols <= i {
                    bDx = mvsDx[i - cols]
                    bDy = mvsDy[i - cols]
                }
                var cDx: Int16 = 0
                var cDy: Int16 = 0
                if cols <= i, col < (cols - 1) {
                    cDx = mvsDx[i - cols + 1]
                    cDy = mvsDy[i - cols + 1]
                }
                let predDx = max(min(Int(aDx), Int(bDx)), min(max(Int(aDx), Int(bDx)), Int(cDx)))
                let predDy = max(min(Int(aDy), Int(bDy)), min(max(Int(aDy), Int(bDy)), Int(cDy)))
                mvsDx[i] = Int16(predDx + Int(resDx))
                mvsDy[i] = Int16(predDy + Int(resDy))
            default:
                var pDx: Int16 = 0
                var pDy: Int16 = 0
                if let pm = prevMVs, i < pm.count {
                    pDx = pm.dx[i]
                    pDy = pm.dy[i]
                }
                mvsDx[i] = Int16(Int(pDx) + Int(resDx))
                mvsDy[i] = Int16(Int(pDy) + Int(resDy))
            }
        }
    }

    return MotionVectors(dx: mvsDx, dy: mvsDy)
}

// MARK: - refDir
//
// Frame-local models: the decoder only decodes refDir when it needs it
// (`nextPd != nil` in parseProfile2Frame), so a model carried across frames
// would diverge from the encoder's. Adaptation therefore restarts every
// frame, seeded at p(1) = 1/16 because refDir is 0.25-2% ones.

@inline(__always)
func encodeRefDirsContextProfile2(refDirs: [Bool], skipMap: [BlockMode]) -> [UInt8] {
    var symbols = [Int]()
    symbols.reserveCapacity(skipMap.count)
    for i in 0..<skipMap.count where skipMap[i] == .inter {
        var sym = 0
        if refDirs[i] {
            sym = 1
        }
        symbols.append(sym)
    }
    if symbols.isEmpty {
        return []
    }

    var model = AdaptiveBinModel(oneIn: 16)
    var cumulativeFreqs = [UInt32](repeating: 0, count: symbols.count)
    var frequencies = [UInt32](repeating: 0, count: symbols.count)
    for i in 0..<symbols.count {
        let interval = model.interval(symbols[i])
        cumulativeFreqs[i] = interval.cum
        frequencies[i] = interval.freq
        model.update(symbols[i])
    }
    var encoder = rANSEncoder()
    for i in stride(from: symbols.count - 1, through: 0, by: -1) {
        encoder.encodeSymbol(cumFreq: cumulativeFreqs[i], freq: frequencies[i])
    }
    encoder.flush()
    return encoder.getBitstream()
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
        var decoder = rANSDecoder(base: base, count: buf.count)
        for i in 0..<min(count, sm.count) where sm[i] == .inter {
            let sym = model.symbol(forCumFreq: decoder.getCumulativeFreq())
            refDirs[i] = (sym == 1)
            let interval = model.interval(sym)
            decoder.advanceSymbol(cumFreq: interval.cum, freq: interval.freq)
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
    var indices = [Int]()
    indices.reserveCapacity(isSkip.count)
    for i in 0..<isSkip.count where isSkip[i] != true {
        indices.append(i)
    }
    if indices.isEmpty {
        return []
    }

    var models = [AdaptiveBinModel](repeating: AdaptiveBinModel(oneIn: 2), count: 9)
    var cumulativeFreqs = [UInt32](repeating: 0, count: indices.count)
    var frequencies = [UInt32](repeating: 0, count: indices.count)
    for k in 0..<indices.count {
        let i = indices[k]
        let context = treeMapContext(isTreez: isTreez, isSkip: isSkip, i: i, cols: cols)
        var sym = 0
        if isTreez[i] {
            sym = 1
        }
        let interval = models[context].interval(sym)
        cumulativeFreqs[k] = interval.cum
        frequencies[k] = interval.freq
        models[context].update(sym)
    }
    var encoder = rANSEncoder()
    for k in stride(from: indices.count - 1, through: 0, by: -1) {
        encoder.encodeSymbol(cumFreq: cumulativeFreqs[k], freq: frequencies[k])
    }
    encoder.flush()
    return encoder.getBitstream()
}

@inline(__always)
private func decodeTreePlaneContext(base: UnsafePointer<UInt8>, count: Int, isSkip: [Bool], cols: Int) -> [Bool] {
    var isTreez = [Bool](repeating: false, count: isSkip.count)
    guard isSkip.contains(false), 4 <= count else {
        return isTreez
    }
    var models = [AdaptiveBinModel](repeating: AdaptiveBinModel(oneIn: 2), count: 9)
    var decoder = rANSDecoder(base: base, count: count)
    for i in 0..<isSkip.count where isSkip[i] != true {
        let context = treeMapContext(isTreez: isTreez, isSkip: isSkip, i: i, cols: cols)
        let sym = models[context].symbol(forCumFreq: decoder.getCumulativeFreq())
        isTreez[i] = (sym == 1)
        let interval = models[context].interval(sym)
        decoder.advanceSymbol(cumFreq: interval.cum, freq: interval.freq)
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
        let sizeY = try readVLQSize(base, at: &offset, count: buf.count)
        let sizeCb = try readVLQSize(base, at: &offset, count: buf.count)
        let sizeCr = try readVLQSize(base, at: &offset, count: buf.count)
        guard ((offset + sizeY) + sizeCb) + sizeCr <= buf.count else {
            throw DecodeError.insufficientData
        }
        let isTreezY = decodeTreePlaneContext(base: base.advanced(by: offset), count: sizeY, isSkip: ySkip, cols: colsY)
        offset += sizeY
        let isTreezCb = decodeTreePlaneContext(base: base.advanced(by: offset), count: sizeCb, isSkip: cbSkip, cols: colsC)
        offset += sizeCb
        let isTreezCr = decodeTreePlaneContext(base: base.advanced(by: offset), count: sizeCr, isSkip: crSkip, cols: colsC)
        return (isTreezY, isTreezCb, isTreezCr)
    }
}
