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
func skipDeciderNormalize(_ x: UnsafePointer<Int32>, _ w: SkipDeciderWeights, into out: UnsafeMutablePointer<Int32>) {
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
func skipDeciderClassify(_ q: UnsafePointer<Int32>, _ w: SkipDeciderWeights, hbuf: UnsafeMutablePointer<Int32>) -> Int {
    let f = w.f
    let h = w.h
    withUnsafePointers(w.w1, w.b1) { w1, b1 in
        for i in 0..<h {
            var acc = b1[i]
            let row = w1 + (i * f)
            for j in 0..<f {
                acc &+= row[j] &* q[j]
            }
            acc >>= w.shift1
            var relu: Int32 = 0
            if 0 < acc {
                relu = acc
            }
            hbuf[i] = relu
        }
    }
    var o0: Int32 = w.b2[0]
    var o1: Int32 = w.b2[1]
    withUnsafePointers(w.w2) { w2 in
        let r0 = w2
        let r1 = w2 + h
        for i in 0..<h {
            o0 &+= r0[i] &* hbuf[i]
            o1 &+= r1[i] &* hbuf[i]
        }
    }
    if o0 < o1 {
        return 1
    }
    return 0
}

/// Zero-motion SAD of one 32x32 luma block plus its two 16x16 chroma blocks
/// against a reference plane: the "would a straight copy do?" measure, uncapped.
@inline(__always)
func skipDeciderZeroSAD32(cur: PlanePointers, ref: PlanePointers, bx: Int, by: Int, width: Int, height: Int) -> Int {
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
            for i in 0..<n {
                let row = rows + (i * SkipDeciderFeature.count)
                skipDeciderNormalize(row, weights, into: qb)
                let decision = skipDeciderClassify(qb, weights, hbuf: hb)
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
        let leftDec: Int32 = 0 < col ? Int32(skipMap[i - 1].rawValue) : -1
        let upDec: Int32 = bw <= i ? Int32(skipMap[i - bw].rawValue) : -1
        let base = i * SkipDeciderFeature.count
        feat[base + 0] = Int32(clamping: skipDeciderZeroSAD32(cur: src, ref: pRef, bx: bx, by: by, width: width, height: height))
        feat[base + 1] = Int32(clamping: skipDeciderZeroSAD32(cur: src, ref: lRef, bx: bx, by: by, width: width, height: height))
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

// MARK: - Periodic skip refresh

/// Periodic refresh of long-running skip blocks (#28).
///
/// A block coded as skip carries its reference pixels forward unchanged, so the
/// requantization noise already present in that reference is never corrected
/// while the block keeps skipping. Forcing such a block back to inter after `r`
/// consecutive skip frames codes a residual against the source again and clears
/// the accumulated error.
///
/// The counters are per 32x32 block — the same grid as the skip map — and the
/// encoder resets them at every I frame, because an I frame recodes every block.
/// Nothing here changes the bitstream syntax: a forced block is an ordinary
/// inter block, so the decoder needs no knowledge of the mechanism.
final class SkipRefreshState: @unchecked Sendable {
    /// Consecutive skip frames per block, reset to 0 whenever the block is
    /// coded as inter (including the frame it is forced back to inter).
    private(set) var counters: [Int] = []
    /// Diagnostics only: blocks forced back to inter, summed over the stream.
    private(set) var forcedBlocks: Int = 0
    /// Diagnostics only: block slots examined on P frames, summed over the
    /// stream. The denominator for the forced-inter ratio.
    private(set) var examinedBlocks: Int = 0

    /// Diagnostic for the #28 follow-up. 0 keeps the shipping behaviour: every
    /// counter restarts at 0 on an I frame, so blocks that then skip together
    /// reach the period on the same frame and refresh in one burst. A positive
    /// value seeds counter i with `i % phaseSpreadPeriod` instead, spreading the
    /// same average refresh rate over the period rather than concentrating it.
    var phaseSpreadPeriod: Int = 0

    init() {}

    @inline(__always)
    func resetCounters() {
        if 0 < phaseSpreadPeriod {
            for i in counters.indices {
                counters[i] = i % phaseSpreadPeriod
            }
        } else {
            for i in counters.indices {
                counters[i] = 0
            }
        }
    }

    @inline(__always)
    private func ensure(_ n: Int) {
        if counters.count != n {
            counters = [Int](repeating: 0, count: n)
            // Seeds the phase on the first allocation too, so the first GOP is
            // not the one case that still refreshes as a single burst.
            resetCounters()
        }
    }

    /// Applies the refresh rule to a final skip map. Returns the number of
    /// blocks forced back to inter in this frame.
    ///
    /// `meMVs` / `meRefDirs` are the motion search results for the frame, which
    /// exist for every block regardless of the skip decision; a forced block
    /// takes its own search result back, so it codes as a normal inter block
    /// with a real motion vector rather than a zero one.
    @inline(__always)
    func apply(
        skipMap: inout [BlockMode],
        mvs: inout MotionVectors,
        refDirs: inout [Bool],
        meMVs: MotionVectors,
        meRefDirs: [Bool],
        period r: Int
    ) -> Int {
        guard 0 < r else { return 0 }
        let n = skipMap.count
        guard 0 < n else { return 0 }
        ensure(n)

        var forced = 0
        for i in 0..<n {
            switch skipMap[i] {
            case .inter:
                counters[i] = 0
            default:
                counters[i] += 1
                if r <= counters[i] {
                    skipMap[i] = .inter
                    if i < meMVs.dx.count {
                        mvs.dx[i] = meMVs.dx[i]
                        mvs.dy[i] = meMVs.dy[i]
                    }
                    if i < meRefDirs.count {
                        refDirs[i] = meRefDirs[i]
                    }
                    counters[i] = 0
                    forced += 1
                }
            }
        }
        forcedBlocks += forced
        examinedBlocks += n
        return forced
    }
}
