// MARK: - Learned skip-safety decider (profile 2 P-frames)
//
// A 20-16-2 integer MLP that answers one question per 32x32 block: would
// copying this block straight from a reference be as good as coding it? Blocks
// it calls skip-safe are turned from `inter` into `skip_prev` / `skip_ltr`
// before anything downstream reads the map, so the coded residual, the motion
// vectors, the skip map and the reconstruction all describe the same decision.
//
// The conversion is one-way: an existing skip is never turned back into an
// inter block. The offline oracle this model was trained against produced no
// case where removing a skip was the right move, and keeping the change
// monotone means the decider can only ever spend fewer bits than the rule-based
// map it starts from.
//
// Inference is pure integer and branch-deterministic (Int32 accumulators, ReLU,
// arithmetic-shift requantization), so a decision is reproducible bit-for-bit
// across runs and machines. The decoder is untouched: the model only chooses
// among modes the existing bitstream syntax already carries.
//
// Weights are compiled in (`SkipModelTables`). `-skip-model 0` disables the
// whole path, and the encoder then takes exactly the same route it took before
// the model existed.

/// Feature columns the model reads, in order. Every one of them is available in
/// the encoder's first pass — nothing here comes from a trial reconstruction.
/// The column order is fixed by the trained weights and must not be reordered.
///
///    0 sadZeroPrevRecon  zero-MV SAD (Y+Cb+Cr) against the prev reconstruction
///    1 sadZeroLtrRecon   zero-MV SAD against the LTR reconstruction
///    2 meSadPrev         motion search SAD toward prev
///    3 meSadLtr          motion search SAD toward LTR
///    4 mvPrevMag         |dx|+|dy| of the prev-direction vector
///    5 mvLtrMag          |dx|+|dy| of the LTR-direction vector
///    6 variance          block variance from computeBlockActivityMap
///    7 activityClass     0 normal / 1 flat / 2 textured / 3 incoherent
///    8 adjustedStep      frame qstep
///    9 gopPosition
///   10 ltrAge
///   11 staticCounter     consecutive static frames for this block
///   12 curDecision       rule-based skip map: 0 inter / 1 skip_prev / 2 skip_ltr
///   13 curRefDir         rule-based reference direction of an inter block
///   14 nbrLeftDecision   rule-based decision of the left neighbour
///   15 nbrUpDecision     rule-based decision of the upper neighbour
///   16 prevFrameLabel    this decider's own answer for this block one frame
///                        ago: 1 skip / 0 code / -1 none
///   17 prevFrameSad0     column 0 of this block one frame ago (-1 if none)
///   18 maxGrad           max centre-difference gradient in the block
///   19 localRange        max 8x8 sub-block (max - min)
enum SkipModelFeature {
    /// Columns the model consumes. Also the row stride of the feature buffer.
    static let count = 20
}

/// Trained coefficients of the decider. The stored layout mirrors the trainer's
/// serialization so the compiled-in table is the training artefact verbatim.
struct SkipModelWeights: Sendable {
    let f: Int          // feature count
    let h: Int          // hidden width
    let mu: [Int32]     // per-feature offset
    let scale: [Int32]  // per-feature multiplier, applied as (x-mu)*scale >> normShift
    let normShift: Int32
    let w1: [Int32]
    let b1: [Int32]
    let w2: [Int32]
    let b2: [Int32]
    let shift1: Int32

    /// Flat Int32 array: magic, kind, f, h, t, mu[f], scale[f], normShift,
    /// w1[h*f], b1[h], w2[2*h], b2[2], shift1, shift2.
    static func parse(_ a: [Int32]) -> SkipModelWeights? {
        if a.count < 8 { return nil }
        if a[0] != 0x504C4D31 { return nil }
        if a[1] != 1 { return nil }        // int8 MLP is the only kind adopted
        let f = Int(a[2])
        let h = Int(a[3])
        if f < 1 { return nil }
        if h < 1 { return nil }
        var p = 5
        func take(_ n: Int) -> [Int32] {
            if a.count < p + n { return [] }
            let s = Array(a[p..<(p + n)])
            p += n
            return s
        }
        let mu = take(f)
        let scale = take(f)
        let normShift = take(1).first ?? 8
        let w1 = take(h * f)
        let b1 = take(h)
        let w2 = take(2 * h)
        let b2 = take(2)
        let shift1 = take(1).first ?? 8
        switch true {
        case mu.count != f, scale.count != f:
            return nil
        case w1.count != h * f, b1.count != h:
            return nil
        case w2.count != 2 * h, b2.count != 2:
            return nil
        default:
            break
        }
        return SkipModelWeights(f: f, h: h, mu: mu, scale: scale, normShift: normShift, w1: w1, b1: b1, w2: w2, b2: b2, shift1: shift1)
    }
}

/// Integer feature normalization. Int64 intermediate: SAD features reach ~10^6
/// and the scale can be large, so the product does not fit in Int32 before the
/// shift.
@inline(__always)
func skipModelNormalize(_ x: UnsafePointer<Int32>, _ w: SkipModelWeights, into out: UnsafeMutablePointer<Int32>) {
    let n = w.f
    let sh = Int64(w.normShift)
    for j in 0..<n {
        var v = ((Int64(x[j]) - Int64(w.mu[j])) &* Int64(w.scale[j])) >> sh
        if v < -127 { v = -127 }
        if 127 < v { v = 127 }
        out[j] = Int32(v)
    }
}

/// One forward pass. int8-range weights, Int32 accumulation, ReLU, arithmetic
/// shift requantization. Returns 1 for "skip is safe here", 0 for "code it".
@inline(__always)
func skipModelRunMLP(_ q: UnsafePointer<Int32>, _ w: SkipModelWeights, hbuf: UnsafeMutablePointer<Int32>) -> Int {
    let f = w.f
    let h = w.h
    w.w1.withUnsafeBufferPointer { w1 in
        w.b1.withUnsafeBufferPointer { b1 in
            for i in 0..<h {
                var acc = b1[i]
                let row = w1.baseAddress! + i * f
                for j in 0..<f {
                    acc &+= row[j] &* q[j]
                }
                acc >>= w.shift1
                hbuf[i] = 0 < acc ? acc : 0
            }
        }
    }
    var o0: Int32 = w.b2[0]
    var o1: Int32 = w.b2[1]
    w.w2.withUnsafeBufferPointer { w2 in
        let r0 = w2.baseAddress!
        let r1 = w2.baseAddress! + h
        for i in 0..<h {
            o0 &+= r0[i] &* hbuf[i]
            o1 &+= r1[i] &* hbuf[i]
        }
    }
    return o1 > o0 ? 1 : 0
}

/// Zero-motion SAD of one 32x32 luma block plus its two 16x16 chroma blocks
/// against a reference plane: the "would a straight copy do?" measure, uncapped.
@inline(__always)
func skipModelZeroSAD32(cur: PlanePointers, ref: PlanePointers, bx: Int, by: Int, width: Int, height: Int) -> Int {
    var sad = 0
    let maxY = min(by + 32, height)
    let maxX = min(bx + 32, width)
    for y in by..<maxY {
        let rc = cur.y + y * width
        let rr = ref.y + y * width
        for x in bx..<maxX {
            sad += abs(Int(rc[x]) - Int(rr[x]))
        }
    }
    let cw = (width + 1) / 2
    let chh = (height + 1) / 2
    let cbx = bx / 2
    let cby = by / 2
    let cMaxY = min(cby + 16, chh)
    let cMaxX = min(cbx + 16, cw)
    for y in cby..<cMaxY {
        let rcb = cur.cb + y * cw
        let rrb = ref.cb + y * cw
        let rcr = cur.cr + y * cw
        let rrr = ref.cr + y * cw
        for x in cbx..<cMaxX {
            sad += abs(Int(rcb[x]) - Int(rrb[x]))
            sad += abs(Int(rcr[x]) - Int(rrr[x]))
        }
    }
    return sad
}

/// Everything the feature rows need from the frame that is not a plane pointer.
struct SkipModelFrameInputs {
    let mvsPrev: MotionVectors
    let mvsLtr: MotionVectors
    let meSadPrev: [Int]
    let meSadLtr: [Int]
    let variance: [Int32]
    let activityClass: [BlockActivityClass]
    let staticCounters: [Int]
    let adjustedStep: Int
    let gopPosition: Int
    let ltrAge: Int
}

/// Per-encoder decider state. Owns every buffer inference touches, sized once
/// per resolution, so a steady-state frame allocates nothing here.
final class SkipModelDecider: @unchecked Sendable {
    let weights: SkipModelWeights

    private var blockCount = -1
    private var feat: [Int32] = []       // blockCount * SkipModelFeature.count
    private var q: [Int32]
    private var hbuf: [Int32]
    /// Column 16/17 carriers: this decider's own answer and column 0 for each
    /// block one frame ago. Written every P-frame, read by the next one.
    private var prevLabel: [Int32] = []
    private var prevSad0: [Int32] = []
    private var curLabel: [Int32] = []
    private var curSad0: [Int32] = []

    /// nil when the compiled-in table fails to parse, which leaves the encoder
    /// on its pre-model path rather than guessing.
    static func make() -> SkipModelDecider? {
        guard let w = SkipModelWeights.parse(SkipModelTables.blob) else { return nil }
        if w.f != SkipModelFeature.count { return nil }
        return SkipModelDecider(weights: w)
    }

    private init(weights: SkipModelWeights) {
        self.weights = weights
        self.q = [Int32](repeating: 0, count: weights.f)
        self.hbuf = [Int32](repeating: 0, count: weights.h)
    }

    private func ensure(_ n: Int) {
        if blockCount == n { return }
        blockCount = n
        feat = [Int32](repeating: 0, count: n * SkipModelFeature.count)
        prevLabel = [Int32](repeating: -1, count: n)
        prevSad0 = [Int32](repeating: -1, count: n)
        curLabel = [Int32](repeating: -1, count: n)
        curSad0 = [Int32](repeating: -1, count: n)
    }

    /// Runs the decider over one P-frame and converts the blocks it calls
    /// skip-safe. `skipMap`, `mvs` and `refDirs` are the rule-based decision on
    /// entry and the final decision on exit.
    func apply(
        skipMap: inout [BlockMode],
        mvs: inout MotionVectors,
        refDirs: inout [Bool],
        inputs: SkipModelFrameInputs,
        source: PlaneData420,
        prevRef: PlaneData420,
        ltrRef: PlaneData420
    ) {
        let width = source.width
        let height = source.height
        let n = skipMap.count
        if n < 1 { return }
        ensure(n)
        let bw = (width + 31) / 32

        withUnsafePlanePointers(source, prevRef, ltrRef) { src, pRef, lRef in
            for i in 0..<n {
                let bx = (i % bw) * 32
                let by = (i / bw) * 32
                extract(into: i, inputs: inputs, bw: bw, bx: bx, by: by, width: width, height: height, skipMap: skipMap, refDirs: refDirs, src: src, pRef: pRef, lRef: lRef)
            }
        }

        feat.withUnsafeBufferPointer { rows in
            q.withUnsafeMutableBufferPointer { qb in
                hbuf.withUnsafeMutableBufferPointer { hb in
                    for i in 0..<n {
                        let row = rows.baseAddress! + i * SkipModelFeature.count
                        skipModelNormalize(row, weights, into: qb.baseAddress!)
                        let decision = skipModelRunMLP(qb.baseAddress!, weights, hbuf: hb.baseAddress!)
                        curSad0[i] = row[0]
                        curLabel[i] = Int32(decision)
                        switch true {
                        case skipMap[i] != .inter, decision != 1:
                            continue
                        default:
                            break
                        }
                        // Reference choice: whichever straight copy is closer
                        // to the source.
                        if row[0] <= row[1] {
                            skipMap[i] = .skip_prev
                            refDirs[i] = false
                        } else {
                            skipMap[i] = .skip_ltr
                            refDirs[i] = true
                        }
                        mvs.dx[i] = 0
                        mvs.dy[i] = 0
                    }
                }
            }
        }

        swap(&prevLabel, &curLabel)
        swap(&prevSad0, &curSad0)
    }

    /// Fills one feature row. See `SkipModelFeature.names` for the columns.
    @inline(__always)
    private func extract(
        into i: Int,
        inputs: SkipModelFrameInputs,
        bw: Int,
        bx: Int,
        by: Int,
        width: Int,
        height: Int,
        skipMap: [BlockMode],
        refDirs: [Bool],
        src: PlanePointers,
        pRef: PlanePointers,
        lRef: PlanePointers
    ) {
        let blockW = min(32, width - bx)
        let blockH = min(32, height - by)
        let (maxG, _, localR) = computeBlockGradientAndContrast(
            source: src.y, stride: width, width: width, height: height,
            bx: bx, by: by, bw: blockW, bh: blockH
        )
        let col = i % bw
        let leftDec: Int32 = 0 < col ? Int32(skipMap[i - 1].rawValue) : -1
        let upDec: Int32 = bw <= i ? Int32(skipMap[i - bw].rawValue) : -1
        let base = i * SkipModelFeature.count
        feat[base + 0] = Int32(clamping: skipModelZeroSAD32(cur: src, ref: pRef, bx: bx, by: by, width: width, height: height))
        feat[base + 1] = Int32(clamping: skipModelZeroSAD32(cur: src, ref: lRef, bx: bx, by: by, width: width, height: height))
        feat[base + 2] = Int32(clamping: i < inputs.meSadPrev.count ? inputs.meSadPrev[i] : 0)
        feat[base + 3] = Int32(clamping: i < inputs.meSadLtr.count ? inputs.meSadLtr[i] : 0)
        feat[base + 4] = Int32(abs(Int(inputs.mvsPrev.dx[i])) + abs(Int(inputs.mvsPrev.dy[i])))
        feat[base + 5] = Int32(abs(Int(inputs.mvsLtr.dx[i])) + abs(Int(inputs.mvsLtr.dy[i])))
        feat[base + 6] = i < inputs.variance.count ? inputs.variance[i] : 0
        feat[base + 7] = i < inputs.activityClass.count ? Int32(inputs.activityClass[i].rawValue) : 0
        feat[base + 8] = Int32(clamping: inputs.adjustedStep)
        feat[base + 9] = Int32(clamping: inputs.gopPosition)
        feat[base + 10] = Int32(clamping: inputs.ltrAge)
        feat[base + 11] = Int32(clamping: i < inputs.staticCounters.count ? inputs.staticCounters[i] : 0)
        feat[base + 12] = Int32(skipMap[i].rawValue)
        feat[base + 13] = refDirs[i] ? 1 : 0
        feat[base + 14] = leftDec
        feat[base + 15] = upDec
        feat[base + 16] = prevLabel[i]
        feat[base + 17] = prevSad0[i]
        feat[base + 18] = maxG
        feat[base + 19] = localR
    }
}
