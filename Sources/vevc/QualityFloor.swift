import Foundation

/// Quality-floor driven early I frames (#31).
///
/// With `keyint` acting as an upper bound rather than a fixed period, the GOP
/// can run long enough for requantization drift to open a visible gap between
/// the reconstruction and the source. The encoder already holds both, so the
/// gap is measured rather than predicted: the luma MSE of every coded frame is
/// compared against the luma MSE of the I frame that opened the GOP, and the
/// next frame is coded as I once the ratio passes the floor.
///
/// Nothing here changes the bitstream syntax. A floor-fired frame is an
/// ordinary I frame, so the decoder needs no knowledge of the mechanism.

/// Integer luma MSE between a reconstruction and its source, truncating.
/// `Σ(recon − src)² / N` over the width x height luma samples, so the value is
/// deterministic and free of floating point.
@inline(__always)
func lumaMSEInteger(reconstructed: PlaneData420, source: PlaneData420) -> Int {
    let width = min(reconstructed.width, source.width)
    let height = min(reconstructed.height, source.height)
    let n = width * height
    guard 0 < n else { return 0 }

    var acc = 0
    withUnsafePointers(reconstructed.y, source.y) { rBase, sBase in
        for row in 0..<height {
            let rRow = rBase + row * reconstructed.width
            let sRow = sBase + row * source.width
            for col in 0..<width {
                let d = Int(rRow[col]) - Int(sRow[col])
                acc += d * d
            }
        }
    }
    return acc / n
}

/// Per-encoder state for the floor. Allocated only when the floor is enabled
/// and applicable (profile 0x02), so a disabled encoder never runs the MSE
/// pass at all.
final class QualityFloorState: @unchecked Sendable {
    /// Floor ratio in Q8: the next frame is coded as I once
    /// `frameMSE * 256 > alphaQ8 * iMSE`.
    let alphaQ8: Int
    /// Luma MSE of the I frame that opened the current GOP.
    private(set) var iMSE: Int = 0
    /// Set when a coded P frame crossed the floor; consumed by the next frame.
    private(set) var pendingIFrame: Bool = false

    // Diagnostics only; none of this reaches the bitstream.
    /// `k` is the periodic-grid position and `dist` the true distance from the
    /// last coded I frame. The guard uses `dist`; the two differ after a cut.
    private(set) var firings: [(frame: Int, k: Int, dist: Int, frameMSE: Int, iMSE: Int)] = []
    private(set) var codedFrames: Int = 0

    init(alphaQ8: Int) {
        self.alphaQ8 = alphaQ8
    }

    /// Records the I frame that opens a GOP. Called for every I frame,
    /// including scene-change and floor-fired ones.
    @inline(__always)
    func noteIFrame(mse: Int) {
        iMSE = mse
        codedFrames += 1
    }

    /// Evaluates a coded P frame against the floor. `k` is the periodic-grid
    /// position of the frame just coded and `dist` its true distance from the
    /// last coded I frame. Returns true when this frame armed an early I.
    @inline(__always)
    func notePFrame(mse: Int, frameIndex: Int, k: Int, dist: Int) -> Bool {
        codedFrames += 1
        // A GOP needs a few frames before drift is meaningful, and firing on
        // the frames right after an I would collapse the GOP length. Measured
        // from the last coded I, not from the periodic grid, so a cut-driven I
        // also restarts the window.
        guard 8 < dist else { return false }
        guard mse * 256 > alphaQ8 * iMSE else { return false }
        pendingIFrame = true
        firings.append((frame: frameIndex, k: k, dist: dist, frameMSE: mse, iMSE: iMSE))
        return true
    }

    /// Consumes the pending request, if any.
    @inline(__always)
    func takePendingIFrame() -> Bool {
        let pending = pendingIFrame
        pendingIFrame = false
        return pending
    }
}
