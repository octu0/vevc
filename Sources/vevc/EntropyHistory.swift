// MARK: - Backward-Adaptive Entropy Tables (history mode)
//
// Per coefficient stream (layer × plane), both the encoder and the decoder
// maintain decayed token counts of previously coded P-frames:
//
//     acc = acc/2 + currentFrameCounts   (after every rANS-coded stream)
//
// When the accumulated distribution models the current frame better than the
// static / dynamic / merged options (dynamic header cost included), the
// encoder selects history mode (flags bit 0x20): no frequency tables are
// transmitted and the decoder rebuilds identical models from the token counts
// of what it already decoded.
//
// Rules keeping both sides in lockstep:
//   - State resets at every I-frame (random access boundary).
//   - Streams coded in raw-bypass mode (<= 32 pairs) and empty streams
//     neither use nor update history.
//   - Per-frame counts are capped at 65535 per token before accumulation
//     (same cap the encoder applies before model selection).
//   - Enabled for profile 0x02 only.

/// Reference type by design (same rationale as L0RefState): each of the nine
/// per-stream states is a long-lived box mutated in place across frames and
/// from the parallel entropy-decode tasks (each task owns one distinct
/// state), and it caches 12 renormalized 16KB LUT model buffers refilled in
/// place per frame — value semantics would copy those buffers at every
/// mutation boundary. ARC traffic is a few retain/release per frame on
/// long-lived objects, not per token.
final class EntropyHistoryState: @unchecked Sendable {
    private(set) var runCounts: [[Int]]
    private(set) var valCounts: [[Int]]
    private(set) var primed = false

    // Cached model sets, renormalized in place after each update — the LUT
    // buffers are allocated once and refilled (memset) per frame, avoiding
    // per-frame allocation of 12 × 16KB lookup tables on the decoder.
    private var lutRun: [rANSModel] = []
    private var lutVal: [rANSModel] = []
    private var lutDirty = true
    private var plainRun: [rANSModel] = []
    private var plainVal: [rANSModel] = []
    private var plainDirty = true

    init() {
        runCounts = [[Int]](repeating: [Int](repeating: 0, count: 64), count: entropyContextCount)
        valCounts = [[Int]](repeating: [Int](repeating: 0, count: 64), count: entropyContextCount)
    }

    func reset() {
        for c in 0..<entropyContextCount {
            for t in 0..<64 {
                runCounts[c][t] = 0
                valCounts[c][t] = 0
            }
        }
        primed = false
        lutDirty = true
        plainDirty = true
    }

    /// acc = acc/2 + cur. Must be fed the identical per-context token counts
    /// on both sides: the encoder passes what it encoded, the decoder what it
    /// decoded.
    func update(runTokenCounts: [[Int]], valTokenCounts: [[Int]]) {
        for c in 0..<entropyContextCount {
            for t in 0..<64 {
                runCounts[c][t] = runCounts[c][t] / 2 + min(65535, runTokenCounts[c][t])
                valCounts[c][t] = valCounts[c][t] / 2 + min(65535, valTokenCounts[c][t])
            }
        }
        primed = true
        lutDirty = true
        plainDirty = true
    }

    /// Models built from the accumulated history. Contexts never seen so far
    /// normalize to the uniform distribution (the cost comparison in the
    /// encoder protects against picking history when that would hurt).
    /// The decoder requests LUTs (needed by findToken); the encoder does not.
    func models(buildLUT: Bool) -> (run: [rANSModel], val: [rANSModel]) {
        if buildLUT {
            if lutDirty {
                if lutRun.isEmpty {
                    lutRun = (0..<entropyContextCount).map { _ in rANSModel() }
                    lutVal = (0..<entropyContextCount).map { _ in rANSModel() }
                }
                for c in 0..<entropyContextCount {
                    lutRun[c].normalize(tokenCounts: runCounts[c])
                    lutVal[c].normalize(tokenCounts: valCounts[c])
                }
                lutDirty = false
            }
            return (lutRun, lutVal)
        }
        if plainDirty {
            if plainRun.isEmpty {
                plainRun = (0..<entropyContextCount).map { _ in rANSModel(buildLUT: false) }
                plainVal = (0..<entropyContextCount).map { _ in rANSModel(buildLUT: false) }
            }
            for c in 0..<entropyContextCount {
                plainRun[c].normalize(tokenCounts: runCounts[c])
                plainVal[c].normalize(tokenCounts: valCounts[c])
            }
            plainDirty = false
        }
        return (plainRun, plainVal)
    }
}

/// The 3×3 stream states of one frame sequence: [layer 0,1,2][plane Y,Cb,Cr].
/// @unchecked Sendable: the per-plane encode/decode tasks each touch a
/// distinct EntropyHistoryState, so access is disjoint by construction.
final class FrameEntropyHistories: @unchecked Sendable {
    let streams: [[EntropyHistoryState]]

    init() {
        streams = (0..<3).map { _ in (0..<3).map { _ in EntropyHistoryState() } }
    }

    func reset() {
        for layer in streams {
            for s in layer {
                s.reset()
            }
        }
    }

    @inline(__always)
    func stream(layer: Int, plane: Int) -> EntropyHistoryState {
        streams[layer][plane]
    }
}
