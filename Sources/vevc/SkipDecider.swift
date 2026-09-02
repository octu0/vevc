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
// Weights are compiled in (`SkipDeciderWeightsData`). `-skip-model 0` disables the
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
enum SkipDeciderFeature {
    /// Columns the model consumes. Also the row stride of the feature buffer.
    static let count = 20
}

/// Integer feature normalization. Int64 intermediate: SAD features reach ~10^6
/// and the scale can be large, so the product does not fit in Int32 before the
/// shift.
@inline(__always)
func skipDeciderNormalize(
    _ x: UnsafePointer<Int32>,
    mu: UnsafePointer<Int32>,
    scale: UnsafePointer<Int32>,
    normShift: Int32,
    n: Int,
    into out: UnsafeMutablePointer<Int32>
) {
    let sh = Int64(normShift)
    for j in 0..<n {
        var v = ((Int64(x[j]) - Int64(mu[j])) &* Int64(scale[j])) >> sh
        if v < -127 { v = -127 }
        if 127 < v { v = 127 }
        out[j] = Int32(v)
    }
}

@inline(__always)
func skipDeciderNormalizeWithWeights(_ x: UnsafePointer<Int32>, _ w: SkipDeciderWeights, into out: UnsafeMutablePointer<Int32>) {
    withUnsafePointers(w.mu, w.scale) { mu, scale in
        skipDeciderNormalize(x, mu: mu, scale: scale, normShift: w.normShift, n: w.f, into: out)
    }
}

/// One forward pass. int8-range weights, Int32 accumulation, ReLU, arithmetic
/// shift requantization. Returns 1 for "skip is safe here", 0 for "code it".
@inline(__always)
func skipDeciderClassify(
    _ q: UnsafePointer<Int32>,
    w1: UnsafePointer<Int32>,
    b1: UnsafePointer<Int32>,
    w2: UnsafePointer<Int32>,
    b2: [Int32],
    f: Int,
    h: Int,
    shift1: Int32,
    hbuf: UnsafeMutablePointer<Int32>
) -> Int {
    let q0 = UnsafeRawPointer(q).loadUnaligned(as: SIMD16<Int32>.self)
    let q1 = UnsafeRawPointer(q.advanced(by: 16)).loadUnaligned(as: SIMD4<Int32>.self)
    var hVec = SIMD16<Int32>()
    for i in 0..<h {
        let rowPtr = w1 + (i * f)
        let wRow0 = UnsafeRawPointer(rowPtr).loadUnaligned(as: SIMD16<Int32>.self)
        let wRow1 = UnsafeRawPointer(rowPtr.advanced(by: 16)).loadUnaligned(as: SIMD4<Int32>.self)
        let dot = (wRow0 &* q0).wrappedSum() &+ (wRow1 &* q1).wrappedSum()
        let acc = (b1[i] &+ dot) >> shift1
        let relu = max(0, acc)
        hbuf[i] = relu
        hVec[i] = relu
    }
    let w2_0 = UnsafeRawPointer(w2).loadUnaligned(as: SIMD16<Int32>.self)
    let w2_1 = UnsafeRawPointer(w2.advanced(by: h)).loadUnaligned(as: SIMD16<Int32>.self)
    let o0 = b2[0] &+ (w2_0 &* hVec).wrappedSum()
    let o1 = b2[1] &+ (w2_1 &* hVec).wrappedSum()
    if o0 < o1 {
        return 1
    }
    return 0
}

@inline(__always)
func skipDeciderClassifyWithWeights(_ q: UnsafePointer<Int32>, _ w: SkipDeciderWeights, hbuf: UnsafeMutablePointer<Int32>) -> Int {
    return withUnsafePointers(w.w1, w.b1, w.w2) { w1, b1, w2 in
        skipDeciderClassify(q, w1: w1, b1: b1, w2: w2, b2: w.b2, f: w.f, h: w.h, shift1: w.shift1, hbuf: hbuf)
    }
}

/// Zero-motion SAD of one 32x32 luma block plus its two 16x16 chroma blocks
/// against a reference plane: the "would a straight copy do?" measure, uncapped.
@inline(__always)
func skipDeciderZeroSAD32(cur: PlanePointers, ref: PlanePointers, bx: Int, by: Int, width: Int, height: Int) -> Int {
    let maxY = min(by + 32, height)
    let maxX = min(bx + 32, width)
    let cw = (width + 1) / 2
    let chh = (height + 1) / 2
    let cbx = bx / 2
    let cby = by / 2
    let cMaxY = min(cby + 16, chh)
    let cMaxX = min(cbx + 16, cw)
    
    // Fast path for inner blocks (width/height multiple of 32, or inside image boundaries)
    if bx + 32 <= width && by + 32 <= height && cbx + 16 <= cw && cby + 16 <= chh {
        var sad: Int = 0
        for y in 0..<32 {
            let rc = cur.y.advanced(by: (by + y) * width + bx)
            let rr = ref.y.advanced(by: (by + y) * width + bx)
            let c0 = UnsafeRawPointer(rc).loadUnaligned(as: SIMD16<Int16>.self)
            let r0 = UnsafeRawPointer(rr).loadUnaligned(as: SIMD16<Int16>.self)
            let d0 = pointwiseMax(c0, r0) &- pointwiseMin(c0, r0)
            let c1 = UnsafeRawPointer(rc.advanced(by: 16)).loadUnaligned(as: SIMD16<Int16>.self)
            let r1 = UnsafeRawPointer(rr.advanced(by: 16)).loadUnaligned(as: SIMD16<Int16>.self)
            let d1 = pointwiseMax(c1, r1) &- pointwiseMin(c1, r1)
            sad &+= Int(SIMD16<Int32>(truncatingIfNeeded: d0).wrappedSum())
                &+ Int(SIMD16<Int32>(truncatingIfNeeded: d1).wrappedSum())
        }
        for y in 0..<16 {
            let rcb = cur.cb.advanced(by: (cby + y) * cw + cbx)
            let rrb = ref.cb.advanced(by: (cby + y) * cw + cbx)
            let cb0 = UnsafeRawPointer(rcb).loadUnaligned(as: SIMD16<Int16>.self)
            let rb0 = UnsafeRawPointer(rrb).loadUnaligned(as: SIMD16<Int16>.self)
            let db0 = pointwiseMax(cb0, rb0) &- pointwiseMin(cb0, rb0)
            
            let rcr = cur.cr.advanced(by: (cby + y) * cw + cbx)
            let rrr = ref.cr.advanced(by: (cby + y) * cw + cbx)
            let cr0 = UnsafeRawPointer(rcr).loadUnaligned(as: SIMD16<Int16>.self)
            let rr0 = UnsafeRawPointer(rrr).loadUnaligned(as: SIMD16<Int16>.self)
            let dr0 = pointwiseMax(cr0, rr0) &- pointwiseMin(cr0, rr0)
            
            sad &+= Int(SIMD16<Int32>(truncatingIfNeeded: db0).wrappedSum())
                &+ Int(SIMD16<Int32>(truncatingIfNeeded: dr0).wrappedSum())
        }
        return sad
    }
    
    // Boundary blocks fallback
    var sad = 0
    for y in by..<maxY {
        let rc = cur.y + y * width
        let rr = ref.y + y * width
        for x in bx..<maxX {
            sad += abs(Int(rc[x]) - Int(rr[x]))
        }
    }
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
struct SkipDeciderFrameInputs {
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
final class SkipDecider: @unchecked Sendable {
    let weights: SkipDeciderWeights

    private var blockCount = -1
    // blockCount * SkipDeciderFeature.count
    private var feat: [Int32] = []
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
    @inline(__always)
    static func make() -> SkipDecider? {
        guard let w = SkipDeciderWeights.parse(SkipDeciderWeightsData.blob) else { return nil }
        if w.f != SkipDeciderFeature.count { return nil }
        // The SIMD forward pass in skipDeciderClassify hardcodes a 16-lane
        // hidden layer: a wider one traps on the lane store and a narrower
        // one reads past the end of w2. Reject any retrained weights whose
        // hidden width no longer matches instead of running them.
        if w.h != 16 { return nil }
        return SkipDecider(weights: w)
    }

    private init(weights: SkipDeciderWeights) {
        self.weights = weights
        self.q = [Int32](repeating: 0, count: weights.f)
        self.hbuf = [Int32](repeating: 0, count: weights.h)
    }

    @inline(__always)
    private func ensure(_ n: Int) {
        if blockCount == n { return }
        blockCount = n
        feat = [Int32](repeating: 0, count: n * SkipDeciderFeature.count)
        prevLabel = [Int32](repeating: -1, count: n)
        prevSad0 = [Int32](repeating: -1, count: n)
        curLabel = [Int32](repeating: -1, count: n)
        curSad0 = [Int32](repeating: -1, count: n)
    }

    /// Converts the blocks the decider calls skip-safe over one P-frame.
    /// `skipMap`, `mvs` and `refDirs` are the rule-based decision on entry and
    /// the final decision on exit.
    @inline(__always)
    func apply(
        skipMap: inout [BlockMode],
        mvs: inout MotionVectors,
        refDirs: inout [Bool],
        inputs: SkipDeciderFrameInputs,
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

        withUnsafePointers(feat, mut: &q, mut: &hbuf) { rows, qb, hb in
            withUnsafePointers(weights.w1, weights.b1, weights.w2) { w1, b1, w2 in
                withUnsafePointers(weights.mu, weights.scale) { mu, scale in
                    let b2 = weights.b2
                    let f = weights.f
                    let h = weights.h
                    let shift1 = weights.shift1
                    let normShift = weights.normShift
                    for i in 0..<n {
                        let row = rows + (i * SkipDeciderFeature.count)
                        skipDeciderNormalize(row, mu: mu, scale: scale, normShift: normShift, n: f, into: qb)
                        let decision = skipDeciderClassify(qb, w1: w1, b1: b1, w2: w2, b2: b2, f: f, h: h, shift1: shift1, hbuf: hb)
                        curSad0[i] = row[0]
                        curLabel[i] = Int32(decision)
                        if skipMap[i] != .inter || decision != 1 {
                            continue
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

    /// Fills one feature row. See `SkipDeciderFeature.names` for the columns.
    @inline(__always)
    private func extract(
        into i: Int,
        inputs: SkipDeciderFrameInputs,
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
        let leftDec: Int32 = switch true {
        case 0 < col: Int32(skipMap[i - 1].rawValue)
        default: -1
        }
        let upDec: Int32 = switch true {
        case bw <= i: Int32(skipMap[i - bw].rawValue)
        default: -1
        }
        let base = i * SkipDeciderFeature.count
        feat[base + 0] = Int32(clamping: skipDeciderZeroSAD32(cur: src, ref: pRef, bx: bx, by: by, width: width, height: height))
        feat[base + 1] = Int32(clamping: skipDeciderZeroSAD32(cur: src, ref: lRef, bx: bx, by: by, width: width, height: height))
        let sadPrev: Int = switch true {
        case i < inputs.meSadPrev.count: inputs.meSadPrev[i]
        default: 0
        }
        feat[base + 2] = Int32(clamping: sadPrev)
        let sadLtr: Int = switch true {
        case i < inputs.meSadLtr.count: inputs.meSadLtr[i]
        default: 0
        }
        feat[base + 3] = Int32(clamping: sadLtr)
        feat[base + 4] = Int32(abs(Int(inputs.mvsPrev.dx[i])) + abs(Int(inputs.mvsPrev.dy[i])))
        feat[base + 5] = Int32(abs(Int(inputs.mvsLtr.dx[i])) + abs(Int(inputs.mvsLtr.dy[i])))
        let variance: Int32 = switch true {
        case i < inputs.variance.count: inputs.variance[i]
        default: 0
        }
        feat[base + 6] = variance
        let actClass: Int32 = switch true {
        case i < inputs.activityClass.count: Int32(inputs.activityClass[i].rawValue)
        default: 0
        }
        feat[base + 7] = actClass
        feat[base + 8] = Int32(clamping: inputs.adjustedStep)
        feat[base + 9] = Int32(clamping: inputs.gopPosition)
        feat[base + 10] = Int32(clamping: inputs.ltrAge)
        let staticCount: Int = switch true {
        case i < inputs.staticCounters.count: inputs.staticCounters[i]
        default: 0
        }
        feat[base + 11] = Int32(clamping: staticCount)
        feat[base + 12] = Int32(skipMap[i].rawValue)
        let refDirVal: Int32 = switch true {
        case refDirs[i]: 1
        default: 0
        }
        feat[base + 13] = refDirVal
        feat[base + 14] = leftDec
        feat[base + 15] = upDec
        feat[base + 16] = prevLabel[i]
        feat[base + 17] = prevSad0[i]
        feat[base + 18] = maxG
        feat[base + 19] = localR
    }
}
