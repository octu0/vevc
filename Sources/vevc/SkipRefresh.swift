import Foundation

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
        switch 0 < phaseSpreadPeriod {
        case true:
            for i in counters.indices {
                counters[i] = i % phaseSpreadPeriod
            }
        case false:
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
                switch r <= counters[i] {
                case true:
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
                case false:
                    break
                }
            }
        }
        forcedBlocks += forced
        examinedBlocks += n
        return forced
    }
}
