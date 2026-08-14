// MARK: - Multi-reference skip oracle (measurement hook)
//
// Answers, without any bitstream change: if the encoder could skip-copy from
// any of the last N reconstructed frames (not just prev/LTR), how many blocks
// currently coded as inter would become skips, and how far back do the
// matching references sit?
//
// Enabled by VEVC_MULTIREF_ORACLE=<poolSize> (e.g. 15). Per P-frame it tests
// every non-skip 32x32 block against each pooled reconstruction with the
// exact production criterion: zero-MV SAD over the four 16x16 sub-blocks
// (luma + chroma, computeZeroSAD16x16 / computeZeroSADSubBlock) against the
// same skipThreshold-per-pixel budget, requiring a single reference to
// satisfy all four sub-blocks. The pool is cleared at every I-frame so no
// candidate crosses a random-access boundary.
//
// Output: one cumulative "MRORACLE" line per frame on stderr; the last line
// holds the totals. dist1 counts nearest-match age 1 (frames the current
// skip_prev rules missed), dist2_4 / dist5_15 older re-appearances.
import Foundation

final class MultiRefOracle: @unchecked Sendable {
    static let shared: MultiRefOracle? = {
        guard let v = ProcessInfo.processInfo.environment["VEVC_MULTIREF_ORACLE"], let n = Int(v), 0 < n else { return nil }
        return MultiRefOracle(poolSize: n)
    }()

    private let poolSize: Int
    private let lock = NSLock()
    private var pool: [PlaneData420] = []

    private var frames = 0
    private var totalBlocks = 0
    private var currentSkips = 0
    private var upgrades = 0
    private var dist1 = 0
    private var dist2_4 = 0
    private var dist5_15 = 0

    init(poolSize: Int) {
        self.poolSize = poolSize
    }

    /// I-frame: random-access boundary — no reference crosses it.
    func reset() {
        lock.lock()
        pool.removeAll()
        lock.unlock()
    }

    /// Called with the final reconstruction of every frame (I and P alike),
    /// newest first in the pool.
    func push(recon: PlaneData420) {
        var y = [Int16](repeating: 0, count: recon.y.count)
        var cb = [Int16](repeating: 0, count: recon.cb.count)
        var cr = [Int16](repeating: 0, count: recon.cr.count)
        y.withUnsafeMutableBufferPointer { d in recon.y.withUnsafeBufferPointer { d.baseAddress!.update(from: $0.baseAddress!, count: $0.count) } }
        cb.withUnsafeMutableBufferPointer { d in recon.cb.withUnsafeBufferPointer { d.baseAddress!.update(from: $0.baseAddress!, count: $0.count) } }
        cr.withUnsafeMutableBufferPointer { d in recon.cr.withUnsafeBufferPointer { d.baseAddress!.update(from: $0.baseAddress!, count: $0.count) } }
        let copy = PlaneData420(width: recon.width, height: recon.height, y: y, cb: cb, cr: cr)
        lock.lock()
        pool.insert(copy, at: 0)
        if poolSize < pool.count { pool.removeLast() }
        lock.unlock()
    }

    /// Called on P-frames after the production skipMap is decided.
    func evaluate(pd: PlaneData420, skipMap: [BlockMode], skipThreshold: Int) {
        lock.lock()
        let refs = pool
        lock.unlock()
        guard refs.isEmpty != true else { return }

        let dx = pd.width
        let dy = pd.height
        let bw = (dx + 31) / 32
        var fUpgrades = 0
        var fDist1 = 0
        var fDist2_4 = 0
        var fDist5_15 = 0
        var fSkips = 0

        for i in 0..<skipMap.count {
            if skipMap[i] != .inter {
                fSkips += 1
                continue
            }
            let bx = (i % bw) * 32
            let by = (i / bw) * 32
            var nearest = -1
            for (age0, ref) in refs.enumerated() {
                if blockMatches(cur: pd, ref: ref, bx: bx, by: by, dx: dx, dy: dy, skipThreshold: skipThreshold) {
                    nearest = age0 + 1
                    break
                }
            }
            if 0 < nearest {
                fUpgrades += 1
                switch nearest {
                case 1: fDist1 += 1
                case 2...4: fDist2_4 += 1
                default: fDist5_15 += 1
                }
            }
        }

        lock.lock()
        frames += 1
        totalBlocks += skipMap.count
        currentSkips += fSkips
        upgrades += fUpgrades
        dist1 += fDist1
        dist2_4 += fDist2_4
        dist5_15 += fDist5_15
        let line = "MRORACLE frames=\(frames) blocks=\(totalBlocks) skips=\(currentSkips) upgrades=\(upgrades) dist1=\(dist1) dist2_4=\(dist2_4) dist5_15=\(dist5_15)\n"
        lock.unlock()
        fputs(line, stderr)
    }

    /// Production skip criterion: all four 16x16 sub-blocks of the 32x32
    /// block within skipThreshold-per-pixel SAD (luma + chroma) against a
    /// single reference's reconstruction.
    private func blockMatches(cur: PlaneData420, ref: PlaneData420, bx: Int, by: Int, dx: Int, dy: Int, skipThreshold: Int) -> Bool {
        withUnsafePlanePointers(cur, ref) { c, r in
            for sy in 0..<2 {
                for sx in 0..<2 {
                    let subX = bx + sx * 16
                    let subY = by + sy * 16
                    let mw = min(16, dx - subX)
                    let mh = min(16, dy - subY)
                    if mw <= 0 || mh <= 0 { continue }
                    let mwc = (mw + 1) / 2
                    let mhc = (mh + 1) / 2
                    let area = mw * mh + mwc * mhc * 2
                    let blockThreshold = skipThreshold * area
                    let sad: Int
                    if mw == 16 && mh == 16 && mwc == 8 && mhc == 8 {
                        sad = computeZeroSAD16x16(cY: c.y, rY: r.y, cCb: c.cb, rCb: r.cb, cCr: c.cr, rCr: r.cr, bx: subX, by: subY, width: dx, limit: blockThreshold)
                    } else {
                        sad = computeZeroSADSubBlock(cY: c.y, rY: r.y, cCb: c.cb, rCb: r.cb, cCr: c.cr, rCr: r.cr, bx: subX, by: subY, width: dx, height: dy, subWidth: mw, subHeight: mh, subWc: mwc, subHc: mhc, limit: blockThreshold)
                    }
                    if blockThreshold < sad {
                        return false
                    }
                }
            }
            return true
        }
    }
}
