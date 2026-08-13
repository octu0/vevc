// MARK: - σ-Conditioned Entropy Measurement (offline)
//
// Reads a VEVC_DUMP_COEFFS file (see SigmaDump.swift) and evaluates how much
// rate the rANS coefficient coding would save if the pair context were derived
// from decoder-available temporal side information ("σ maps") instead of the
// current 6-context scheme:
//   A: local coefficient energy of the DWT pyramid of the previous
//      reconstructed frame (recomputed from pixels on both sides, no signaling)
//   B: local energy of the previous P-frame's coded coefficients
//      (already present on both sides after decoding the previous frame)
// Energy windows of 3×3 / 7×7 / 11×11 and 4/6/8-bucket context counts are
// compared.
//
// The pair stream (LSCP + zero-run/value pairs, scan orders, parent context)
// is regenerated exactly as the encoder emits it; bits are computed with the
// same tokenization, normalization and table header sizing as the real coder.
// Three signaling schemes are costed per variant:
//   adaptive    per-frame dynamic tables + headers (merged if cheaper);
//               the baseline replicates unifiedSelectModel's 3-way choice
//   prev-tables backward adaptation: tables built from the decayed token
//               counts of previously decoded frames (no header at all;
//               first frame of a stream bootstraps with adaptive)
//   model-only  conditional-entropy bound (ideal pre-trained static tables)

import Foundation

public enum SigmaMeasureError: Error {
    case badFormat(String)
}

// MARK: Dump reading

private struct MPlane {
    let w: Int
    let h: Int
    let data: [Int16]
}

private struct MEntry {
    let subs: [MPlane]
}

private struct MFrame {
    let gop: Int
    let width: Int
    let height: Int
    let qsteps: [[Int]]    // 6 tables (Y2,C2,Y1,C1,Y0,C0) × [low, mid, high]
    let layerBytes: [Int]  // L0, L1, L2
    let coded: [MEntry]    // 9: L2 Y,Cb,Cr / L1 Y,Cb,Cr / L0 Y,Cb,Cr
    let parents: [MEntry]  // 6: recon L1 quadrants ×3, recon L0 quadrants ×3
    let pyr: [MEntry]      // 9, same order/shape as coded
}

private final class DumpReader {
    private let data: Data
    private var off: Int

    init(path: String) throws {
        self.data = try Data(contentsOf: URL(fileURLWithPath: path), options: .mappedIfSafe)
        self.off = 0
        guard try magic() == "VSD1" else { throw SigmaMeasureError.badFormat("bad magic") }
    }

    var atEnd: Bool { data.count <= off }

    private func magic() throws -> String {
        guard off + 4 <= data.count else { throw SigmaMeasureError.badFormat("truncated magic") }
        let s = String(decoding: data[data.startIndex + off ..< data.startIndex + off + 4], as: UTF8.self)
        off += 4
        return s
    }

    private func i32() throws -> Int {
        guard off + 4 <= data.count else { throw SigmaMeasureError.badFormat("truncated i32") }
        var u: UInt32 = 0
        withUnsafeMutableBytes(of: &u) { dst in
            data.copyBytes(to: dst, from: data.startIndex + off ..< data.startIndex + off + 4)
        }
        off += 4
        return Int(Int32(bitPattern: UInt32(littleEndian: u)))
    }

    private func int16Array(_ count: Int) throws -> [Int16] {
        let byteCount = count * 2
        guard off + byteCount <= data.count else { throw SigmaMeasureError.badFormat("truncated plane") }
        var arr = [Int16](repeating: 0, count: count)
        arr.withUnsafeMutableBytes { dst in
            _ = data.copyBytes(to: dst, from: data.startIndex + off ..< data.startIndex + off + byteCount)
        }
        off += byteCount
        return arr
    }

    private func entry() throws -> MEntry {
        let nSub = try i32()
        guard 1 <= nSub && nSub <= 4 else { throw SigmaMeasureError.badFormat("bad subband count \(nSub)") }
        var subs: [MPlane] = []
        subs.reserveCapacity(nSub)
        for _ in 0..<nSub {
            let w = try i32()
            let h = try i32()
            subs.append(MPlane(w: w, h: h, data: try int16Array(w * h)))
        }
        return MEntry(subs: subs)
    }

    func nextFrame() throws -> MFrame? {
        if atEnd { return nil }
        guard try magic() == "FRAM" else { throw SigmaMeasureError.badFormat("bad frame marker") }
        let gop = try i32()
        let width = try i32()
        let height = try i32()
        var qsteps: [[Int]] = []
        for _ in 0..<6 {
            qsteps.append([try i32(), try i32(), try i32()])
        }
        let layerBytes = [try i32(), try i32(), try i32()]
        var coded: [MEntry] = []
        for _ in 0..<9 { coded.append(try entry()) }
        var parents: [MEntry] = []
        for _ in 0..<6 { parents.append(try entry()) }
        var pyr: [MEntry] = []
        for _ in 0..<9 { pyr.append(try entry()) }
        return MFrame(gop: gop, width: width, height: height, qsteps: qsteps, layerBytes: layerBytes, coded: coded, parents: parents, pyr: pyr)
    }
}

// MARK: Context variants

/// Per-pair features available causally on the decoder side.
/// Energies are qstep- and tap-normalized local sums; -1 = source unavailable.
private struct Features {
    let isLscp: Bool
    let baseCtx: Int
    let prevNZ: Bool
    let parentZ: Bool
    let eA3: Int
    let eB3: Int
    let eB7: Int
    let eB11: Int
}

@inline(__always)
private func lg2(_ e: Int) -> Int { 63 - e.leadingZeroBitCount }

@inline(__always)
private func b4(_ e: Int) -> Int { e <= 0 ? 0 : min(3, 1 + lg2(e) / 3) }

@inline(__always)
private func b6(_ e: Int) -> Int { e <= 0 ? 0 : min(5, 1 + lg2(e) / 2) }

@inline(__always)
private func b8(_ e: Int) -> Int { e <= 0 ? 0 : min(7, 1 + (lg2(e) * 2) / 3) }

private struct VariantDef {
    let name: String
    let contextCount: Int
    let ctx: (Features) -> Int
}

/// Variant list. Every variant reserves its last context for LSCP pairs and,
/// for σ_B variants, one "cold" context for frames without a previous P-frame.
private func makeVariants() -> [VariantDef] {
    return [
        VariantDef(name: "base6 (current)", contextCount: 6) { f in
            f.baseCtx
        },
        VariantDef(name: "sigA6 w3", contextCount: 7) { f in
            f.isLscp ? 6 : b6(f.eA3)
        },
        VariantDef(name: "sigB6 w3", contextCount: 8) { f in
            if f.isLscp { return 7 }
            return f.eB3 < 0 ? 6 : b6(f.eB3)
        },
        VariantDef(name: "sigB4 w7", contextCount: 6) { f in
            if f.isLscp { return 5 }
            return f.eB7 < 0 ? 4 : b4(f.eB7)
        },
        VariantDef(name: "sigB6 w7", contextCount: 8) { f in
            if f.isLscp { return 7 }
            return f.eB7 < 0 ? 6 : b6(f.eB7)
        },
        VariantDef(name: "sigB6 w11", contextCount: 8) { f in
            if f.isLscp { return 7 }
            return f.eB11 < 0 ? 6 : b6(f.eB11)
        },
        VariantDef(name: "sigB8 w7", contextCount: 10) { f in
            if f.isLscp { return 9 }
            return f.eB7 < 0 ? 8 : b8(f.eB7)
        },
        VariantDef(name: "sigB6 w7 x prevNZ", contextCount: 14) { f in
            if f.isLscp { return 13 }
            return f.eB7 < 0 ? 12 : b6(f.eB7) * 2 + (f.prevNZ ? 1 : 0)
        },
        VariantDef(name: "sigA4 x sigB4 w3", contextCount: 21) { f in
            if f.isLscp { return 20 }
            return f.eB3 < 0 ? 16 + b4(f.eA3) : b4(f.eA3) * 4 + b4(f.eB3)
        },
    ]
}

// MARK: Token counting

private final class StreamCounter {
    let variants: [VariantDef]
    var run: [[[Int]]]
    var val: [[[Int]]]
    var sharedBits = 0  // block flags + hasNonZero flags + token bypass bits

    init(variants: [VariantDef]) {
        self.variants = variants
        run = variants.map { [[Int]](repeating: [Int](repeating: 0, count: 64), count: $0.contextCount) }
        val = variants.map { [[Int]](repeating: [Int](repeating: 0, count: 64), count: $0.contextCount) }
    }

    @inline(__always)
    func addPair(runLen: Int, value: Int16, features: Features) {
        let rt = valueTokenizeUnsigned(UInt32(runLen))
        let vt = valueTokenize(value)
        sharedBits += rt.bypassLen + vt.bypassLen
        for (vi, v) in variants.enumerated() {
            let ctx = v.ctx(features)
            run[vi][ctx][Int(rt.token)] += 1
            val[vi][ctx][Int(vt.token)] += 1
        }
    }
}

// MARK: σ energy planes

/// Clamped box sum with the given radius (window = 2*radius+1), via prefix sums.
private func boxSum(_ mag: [Int], _ w: Int, _ h: Int, radius: Int) -> [Int] {
    var tmp = [Int](repeating: 0, count: w * h)
    var pref = [Int](repeating: 0, count: max(w, h) + 1)
    for y in 0..<h {
        let ro = y * w
        for x in 0..<w { pref[x + 1] = pref[x] + mag[ro + x] }
        for x in 0..<w {
            let lo = max(0, x - radius)
            let hi = min(w - 1, x + radius)
            tmp[ro + x] = pref[hi + 1] - pref[lo]
        }
    }
    var out = [Int](repeating: 0, count: w * h)
    for x in 0..<w {
        for y in 0..<h { pref[y + 1] = pref[y] + tmp[y * w + x] }
        for y in 0..<h {
            let lo = max(0, y - radius)
            let hi = min(h - 1, y + radius)
            out[y * w + x] = pref[hi + 1] - pref[lo]
        }
    }
    return out
}

/// Normalized energy plane: (windowed sum, rescaled to 3×3-equivalent taps) × num / den.
private func energyPlane(mag: [Int], w: Int, h: Int, radius: Int, num: Int, den: Int) -> [Int] {
    let taps = (2 * radius + 1) * (2 * radius + 1)
    let d = max(1, den)
    var out = boxSum(mag, w, h, radius: radius)
    for i in out.indices {
        out[i] = ((out[i] * 9) / taps) * num / d
    }
    return out
}

/// Quantizer step (Q4) applying to a given entry/subband of the dump.
private func stepFor(entry e: Int, sub: Int, qsteps: [[Int]]) -> Int {
    let layerRow = e / 3  // 0 → L2, 1 → L1, 2 → L0
    let chroma = (e % 3) != 0
    let t = qsteps[layerRow * 2 + (chroma ? 1 : 0)]
    if layerRow == 2 {
        // L0 subband order: LL, HL, LH, HH
        switch sub {
        case 0: return t[0]
        case 1, 2: return t[1]
        default: return t[2]
        }
    }
    // L2/L1 subband order: HL, LH, HH
    return sub == 2 ? t[2] : t[1]
}

/// |quantized value| for coded planes: AC subbands store zig-zag mapped
/// non-negative values; L0 LL stores signed DPCM-quantized values.
private func codedMagnitudes(_ p: MPlane, isZigzag: Bool) -> [Int] {
    var out = [Int](repeating: 0, count: p.data.count)
    if isZigzag {
        for i in p.data.indices {
            let u = Int(UInt16(bitPattern: p.data[i]))
            out[i] = (u + 1) >> 1
        }
    } else {
        for i in p.data.indices {
            out[i] = Int(p.data[i].magnitude)
        }
    }
    return out
}

private func absMagnitudes(_ p: MPlane) -> [Int] {
    var out = [Int](repeating: 0, count: p.data.count)
    for i in p.data.indices {
        out[i] = Int(p.data[i].magnitude)
    }
    return out
}

// MARK: μ-prediction (shift/integer lifting) simulation

@inline(__always)
private func unzigzag(_ v: Int16) -> Int {
    let u = Int(UInt16(bitPattern: v))
    return (u >> 1) ^ -(u & 1)
}

@inline(__always)
private func zigzag(_ s: Int) -> Int16 {
    let v = Int16(clamping: s)
    return Int16(bitPattern: UInt16(bitPattern: (v << 1) ^ (v >> 15)))
}

/// Residual planes for μ-prediction: predict each quantized coefficient from
/// the co-located coefficient of the previous P-frame (dequantized and
/// re-quantized to the current step, integer rounding — exactly reproducible
/// on the decoder). r = c − pred, stored back in the plane's native mapping
/// so the unmodified pair walker measures the NET bit cost including the
/// zero-run / LSCP / block-flag structure of the residual.
private func muResidualEntry(cur: MEntry, prev: MEntry, entryIndex e: Int, qstepsCur: [[Int]], qstepsPrev: [[Int]]) -> MEntry {
    let layerRow = e / 3
    var subs: [MPlane] = []
    subs.reserveCapacity(cur.subs.count)
    for s in 0..<cur.subs.count {
        let isSignedLL = (layerRow == 2 && s == 0)
        let stepCur = stepFor(entry: e, sub: s, qsteps: qstepsCur)
        let stepPrev = stepFor(entry: e, sub: s, qsteps: qstepsPrev)
        let cp = cur.subs[s]
        let pp = prev.subs[s]
        var data = [Int16](repeating: 0, count: cp.data.count)
        for i in cp.data.indices {
            let curS = isSignedLL ? Int(cp.data[i]) : unzigzag(cp.data[i])
            let prevS = isSignedLL ? Int(pp.data[i]) : unzigzag(pp.data[i])
            let num = prevS * stepPrev
            let half = stepCur / 2
            let pred = (num + (0 <= num ? half : -half)) / stepCur
            let r = curS - pred
            data[i] = isSignedLL ? Int16(clamping: r) : zigzag(r)
        }
        subs.append(MPlane(w: cp.w, h: cp.h, data: data))
    }
    return MEntry(subs: subs)
}

/// Per-block oracle diagnostic: fraction of coding blocks where the μ
/// residual has strictly fewer nonzero coefficients than the original
/// (upper bound on what per-block predict-on/off signaling could exploit).
private func muBlockWinRate(orig: MEntry, mu: MEntry, tile: Int) -> (wins: Int, total: Int) {
    let cols = orig.subs[0].w / tile
    let rows = orig.subs[0].h / tile
    var wins = 0
    for r in 0..<rows {
        for c in 0..<cols {
            var nzO = 0
            var nzM = 0
            for s in orig.subs.indices {
                let po = orig.subs[s]
                let pm = mu.subs[s]
                for y in 0..<tile {
                    let ro = (r * tile + y) * po.w + c * tile
                    for x in 0..<tile {
                        if po.data[ro + x] != 0 { nzO += 1 }
                        if pm.data[ro + x] != 0 { nzM += 1 }
                    }
                }
            }
            if nzM < nzO { wins += 1 }
        }
    }
    return (wins, rows * cols)
}

// MARK: Pair stream regeneration (mirrors blockEncode*)

@inline(__always)
private func tileAllZero(_ p: MPlane, _ ox: Int, _ oy: Int, _ n: Int) -> Bool {
    for y in 0..<n {
        let ro = (oy + y) * p.w + ox
        for x in 0..<n {
            if p.data[ro + x] != 0 { return false }
        }
    }
    return true
}

private struct SigmaPlanes {
    let a3: [[Int]]    // per subband, 3×3 energy from prev-recon pyramid
    let b3: [[Int]]?   // nil = cold (no previous P-frame in this GOP)
    let b7: [[Int]]?
    let b11: [[Int]]?
}

/// Walk one coefficient tile exactly like blockEncode{16,8,4}{V,H}[WithParent]:
/// 1 hasNonZero bypass bit, LSCP as two ctx-5 pairs, then (zero-run, value)
/// pairs in scan order with the context derived at the run-start position.
private func walkTile(
    plane: MPlane, ox: Int, oy: Int, n: Int, scanV: Bool,
    parent: MPlane?, pox: Int, poy: Int,
    sub: Int, sigma: SigmaPlanes,
    counter: StreamCounter
) {
    counter.sharedBits += 1  // hasNonZero flag

    var lscpIdx = -1
    for idx in 0..<(n * n) {
        let x = scanV ? idx / n : idx % n
        let y = scanV ? idx % n : idx / n
        if plane.data[(oy + y) * plane.w + ox + x] != 0 { lscpIdx = idx }
    }
    if lscpIdx < 0 { return }

    let lx = scanV ? lscpIdx / n : lscpIdx % n
    let ly = scanV ? lscpIdx % n : lscpIdx / n
    let lscpFeatures = Features(isLscp: true, baseCtx: 5, prevNZ: false, parentZ: false, eA3: 0, eB3: 0, eB7: 0, eB11: 0)
    counter.addPair(runLen: lx, value: 0, features: lscpFeatures)
    counter.addPair(runLen: ly, value: 0, features: lscpFeatures)

    var run = 0
    var prevVal: Int16 = 0
    var startIdx = 0
    for idx in 0...lscpIdx {
        let x = scanV ? idx / n : idx % n
        let y = scanV ? idx % n : idx / n
        let v = plane.data[(oy + y) * plane.w + ox + x]
        if run == 0 { startIdx = idx }
        if v == 0 {
            run += 1
        } else {
            let sx = scanV ? startIdx / n : startIdx % n
            let sy = scanV ? startIdx % n : startIdx / n
            var parentZ = false
            if let par = parent {
                parentZ = par.data[(poy + (sy >> 1)) * par.w + pox + (sx >> 1)] == 0
            }
            let baseCtx = (parent != nil && parentZ ? 2 : 0) + (prevVal != 0 ? 1 : 0)
            let si = (oy + sy) * plane.w + (ox + sx)
            counter.addPair(runLen: run, value: v, features: Features(
                isLscp: false, baseCtx: baseCtx,
                prevNZ: prevVal != 0, parentZ: parentZ,
                eA3: sigma.a3[sub][si],
                eB3: sigma.b3.map { $0[sub][si] } ?? -1,
                eB7: sigma.b7.map { $0[sub][si] } ?? -1,
                eB11: sigma.b11.map { $0[sub][si] } ?? -1))
            prevVal = v
            run = 0
        }
    }
}

/// L2 (tile 16) / L1 (tile 8) plane stream, mirroring encodePlaneSubbands32/16.
private func walkStreamUpper(entry: MEntry, parent: MEntry, tile: Int, sigma: SigmaPlanes, counter: StreamCounter) {
    let half = tile / 2
    let hl = entry.subs[0]
    let lh = entry.subs[1]
    let hh = entry.subs[2]
    let cols = hl.w / tile
    let rows = hl.h / tile
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
                            plane: entry.subs[s], ox: ox + qx, oy: oy + qy, n: half, scanV: false,
                            parent: parent.subs[s], pox: c * half + qx / 2, poy: r * half + qy / 2,
                            sub: s, sigma: sigma, counter: counter)
                    }
                }
            } else {
                counter.sharedBits += 2
                walkTile(plane: hl, ox: ox, oy: oy, n: tile, scanV: true, parent: parent.subs[0], pox: c * half, poy: r * half, sub: 0, sigma: sigma, counter: counter)
                walkTile(plane: lh, ox: ox, oy: oy, n: tile, scanV: false, parent: parent.subs[1], pox: c * half, poy: r * half, sub: 1, sigma: sigma, counter: counter)
                walkTile(plane: hh, ox: ox, oy: oy, n: tile, scanV: false, parent: parent.subs[2], pox: c * half, poy: r * half, sub: 2, sigma: sigma, counter: counter)
            }
        }
    }
}

/// L0 (tile 4) P-frame plane stream, mirroring encodePlaneBaseSubbands8PFrame.
private func walkStreamBase(entry: MEntry, sigma: SigmaPlanes, counter: StreamCounter) {
    let ll = entry.subs[0]
    let hl = entry.subs[1]
    let lh = entry.subs[2]
    let hh = entry.subs[3]
    let cols = ll.w / 4
    let rows = ll.h / 4
    for r in 0..<rows {
        for c in 0..<cols {
            let ox = c * 4
            let oy = r * 4
            counter.sharedBits += 2  // zero flag + reserved flag
            if tileAllZero(ll, ox, oy, 4) && tileAllZero(hl, ox, oy, 4) && tileAllZero(lh, ox, oy, 4) && tileAllZero(hh, ox, oy, 4) {
                continue
            }
            walkTile(plane: ll, ox: ox, oy: oy, n: 4, scanV: false, parent: nil, pox: 0, poy: 0, sub: 0, sigma: sigma, counter: counter)
            walkTile(plane: hl, ox: ox, oy: oy, n: 4, scanV: true, parent: nil, pox: 0, poy: 0, sub: 1, sigma: sigma, counter: counter)
            walkTile(plane: lh, ox: ox, oy: oy, n: 4, scanV: false, parent: nil, pox: 0, poy: 0, sub: 2, sigma: sigma, counter: counter)
            walkTile(plane: hh, ox: ox, oy: oy, n: 4, scanV: false, parent: nil, pox: 0, poy: 0, sub: 3, sigma: sigma, counter: counter)
        }
    }
}

// MARK: Bit cost simulation

private func mergedCostQ8(run: [[Int]], val: [[Int]]) -> Int {
    var mr = [Int](repeating: 0, count: 64)
    var mv = [Int](repeating: 0, count: 64)
    for c in run.indices {
        for t in 0..<64 {
            mr[t] += run[c][t]
            mv[t] += val[c][t]
        }
    }
    var rm = rANSModel(buildLUT: false)
    var vm = rANSModel(buildLUT: false)
    rm.normalize(tokenCounts: mr)
    vm.normalize(tokenCounts: mv)
    var q8 = estimateBitCostQ8(tokenCounts: mr, model: rm) + estimateBitCostQ8(tokenCounts: mv, model: vm)
    q8 += (headerCostBits(model: rm) + headerCostBits(model: vm)) << 8
    return q8
}

/// Baseline adaptive cost, replicating unifiedSelectModel's 3-way choice.
private func costBase6Q8(run: [[Int]], val: [[Int]]) -> (adaptive: Int, modelOnly: Int) {
    let s = StaticRANSModels.shared
    let statRun = [s.runModel0, s.runModel1, s.runModel2, s.runModel3, s.dpcmRunModel, s.lscpRunModel]
    let statVal = [s.valModel0, s.valModel1, s.valModel2, s.valModel3, s.dpcmValModel, s.dpcmValModel]
    var staticQ8 = 0
    var dynQ8 = 0
    var dynHdrBits = 0
    for c in 0..<6 {
        staticQ8 += estimateBitCostQ8(tokenCounts: run[c], model: statRun[c])
        staticQ8 += estimateBitCostQ8(tokenCounts: val[c], model: statVal[c])
        var rm = rANSModel(buildLUT: false)
        var vm = rANSModel(buildLUT: false)
        rm.normalize(tokenCounts: run[c])
        vm.normalize(tokenCounts: val[c])
        dynQ8 += estimateBitCostQ8(tokenCounts: run[c], model: rm)
        dynQ8 += estimateBitCostQ8(tokenCounts: val[c], model: vm)
        dynHdrBits += headerCostBits(model: rm) + headerCostBits(model: vm)
    }
    let adaptive = min(staticQ8, min(dynQ8 + (dynHdrBits << 8), mergedCostQ8(run: run, val: val)))
    return (adaptive, dynQ8)
}

/// σ-variant adaptive cost: dynamic per-context tables (empty contexts cost one
/// presence bit) or the merged single-table option, whichever is cheaper.
private func costVariantQ8(run: [[Int]], val: [[Int]]) -> (adaptive: Int, modelOnly: Int) {
    var dynQ8 = 0
    var hdrBits = run.count  // presence flags
    for c in run.indices {
        let total = run[c].reduce(0, +) + val[c].reduce(0, +)
        if total == 0 { continue }
        var rm = rANSModel(buildLUT: false)
        var vm = rANSModel(buildLUT: false)
        rm.normalize(tokenCounts: run[c])
        vm.normalize(tokenCounts: val[c])
        dynQ8 += estimateBitCostQ8(tokenCounts: run[c], model: rm)
        dynQ8 += estimateBitCostQ8(tokenCounts: val[c], model: vm)
        hdrBits += headerCostBits(model: rm) + headerCostBits(model: vm)
    }
    let adaptive = min(dynQ8 + (hdrBits << 8), mergedCostQ8(run: run, val: val))
    return (adaptive, dynQ8)
}

// MARK: Backward-adapted (prev-tables) cost

/// Decayed token counts per (stream, variant), mirroring what a decoder could
/// maintain from its own decoded symbols: acc = acc/2 + current after each frame.
private final class BackwardState {
    var run: [[[[Int]]]]  // [stream][variant][ctx][64]
    var val: [[[[Int]]]]
    var primed: [[Bool]]

    init(streams: Int, variants: [VariantDef]) {
        run = (0..<streams).map { _ in variants.map { [[Int]](repeating: [Int](repeating: 0, count: 64), count: $0.contextCount) } }
        val = run
        primed = [[Bool]](repeating: [Bool](repeating: false, count: variants.count), count: streams)
    }
}

/// Bits to code `cur` counts under models built from `acc` counts (no header).
/// Contexts unseen so far fall back to the merged accumulated model.
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
        if curRun[c].reduce(0, +) + curVal[c].reduce(0, +) == 0 { continue }
        if 0 < accRun[c].reduce(0, +) + accVal[c].reduce(0, +) {
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

// MARK: Aggregation & entry point

private struct LayerAgg {
    var actualBytes = 0
    var sharedBits = 0
    var adaptiveQ8: [Int]
    var backwardQ8: [Int]
    var modelOnlyQ8: [Int]

    init(variantCount: Int) {
        adaptiveQ8 = [Int](repeating: 0, count: variantCount)
        backwardQ8 = [Int](repeating: 0, count: variantCount)
        modelOnlyQ8 = [Int](repeating: 0, count: variantCount)
    }
}

public func runSigmaMeasurement(dumpPath: String) throws -> String {
    let reader = try DumpReader(path: dumpPath)
    let variants = makeVariants()
    var agg = [LayerAgg(variantCount: variants.count), LayerAgg(variantCount: variants.count), LayerAgg(variantCount: variants.count)]
    let backward = BackwardState(streams: 9, variants: variants)
    // μ-prediction lane: same variants/costing over residual planes
    var aggMu = [LayerAgg(variantCount: variants.count), LayerAgg(variantCount: variants.count), LayerAgg(variantCount: variants.count)]
    let backwardMu = BackwardState(streams: 9, variants: variants)
    var muWins = [0, 0, 0]
    var muBlocks = [0, 0, 0]
    var frameCount = 0
    var coldCount = 0
    var width = 0
    var height = 0

    var prev: MFrame? = nil
    while let frame = try reader.nextFrame() {
        width = frame.width
        height = frame.height
        // The previous dumped P-frame remains a valid σ_B source across copy
        // frames (they leave the reconstruction unchanged); only a GOP reset
        // (gopPosition not increasing) invalidates it.
        let prevOK = prev.map { $0.gop < frame.gop } ?? false
        if prevOK != true { coldCount += 1 }

        for e in 0..<9 {
            let layerRow = e / 3  // 0=L2, 1=L1, 2=L0
            let entry = frame.coded[e]
            let subCount = entry.subs.count

            var a3: [[Int]] = []
            var b3: [[Int]]? = prevOK ? [] : nil
            var b7: [[Int]]? = prevOK ? [] : nil
            var b11: [[Int]]? = prevOK ? [] : nil
            for s in 0..<subCount {
                let step = stepFor(entry: e, sub: s, qsteps: frame.qsteps)
                let pyrPlane = frame.pyr[e].subs[s]
                a3.append(energyPlane(mag: absMagnitudes(pyrPlane), w: pyrPlane.w, h: pyrPlane.h, radius: 1, num: 16, den: step))
                if prevOK, let p = prev {
                    let isZigzag = !(layerRow == 2 && s == 0)
                    let prevStep = stepFor(entry: e, sub: s, qsteps: p.qsteps)
                    let prevPlane = p.coded[e].subs[s]
                    let mag = codedMagnitudes(prevPlane, isZigzag: isZigzag)
                    b3!.append(energyPlane(mag: mag, w: prevPlane.w, h: prevPlane.h, radius: 1, num: prevStep, den: step))
                    b7!.append(energyPlane(mag: mag, w: prevPlane.w, h: prevPlane.h, radius: 3, num: prevStep, den: step))
                    b11!.append(energyPlane(mag: mag, w: prevPlane.w, h: prevPlane.h, radius: 5, num: prevStep, den: step))
                }
            }
            let sigma = SigmaPlanes(a3: a3, b3: b3, b7: b7, b11: b11)

            let counter = StreamCounter(variants: variants)
            switch layerRow {
            case 0:
                walkStreamUpper(entry: entry, parent: frame.parents[e], tile: 16, sigma: sigma, counter: counter)
            case 1:
                walkStreamUpper(entry: entry, parent: frame.parents[e], tile: 8, sigma: sigma, counter: counter)
            default:
                walkStreamBase(entry: entry, sigma: sigma, counter: counter)
            }

            for vi in variants.indices {
                let cost: (adaptive: Int, modelOnly: Int)
                if vi == 0 {
                    cost = costBase6Q8(run: counter.run[vi], val: counter.val[vi])
                } else {
                    cost = costVariantQ8(run: counter.run[vi], val: counter.val[vi])
                }
                agg[layerRow].adaptiveQ8[vi] += cost.adaptive
                agg[layerRow].modelOnlyQ8[vi] += cost.modelOnly

                // prev-tables: bootstrap each stream with the adaptive scheme,
                // then switch to header-free backward-adapted tables.
                let bw: Int
                if backward.primed[e][vi] {
                    bw = costBackwardQ8(curRun: counter.run[vi], curVal: counter.val[vi], accRun: backward.run[e][vi], accVal: backward.val[e][vi])
                } else {
                    bw = cost.adaptive
                }
                agg[layerRow].backwardQ8[vi] += bw
                for c in 0..<variants[vi].contextCount {
                    for t in 0..<64 {
                        backward.run[e][vi][c][t] = backward.run[e][vi][c][t] / 2 + counter.run[vi][c][t]
                        backward.val[e][vi][c][t] = backward.val[e][vi][c][t] / 2 + counter.val[vi][c][t]
                    }
                }
                backward.primed[e][vi] = true
            }
            agg[layerRow].sharedBits += counter.sharedBits

            // μ-prediction lane: identical costing over the residual planes
            // (cold frames fall back to the original planes, pred = 0).
            let muEntry: MEntry
            if prevOK, let p = prev {
                muEntry = muResidualEntry(cur: entry, prev: p.coded[e], entryIndex: e, qstepsCur: frame.qsteps, qstepsPrev: p.qsteps)
            } else {
                muEntry = entry
            }
            let counterMu = StreamCounter(variants: variants)
            switch layerRow {
            case 0:
                walkStreamUpper(entry: muEntry, parent: frame.parents[e], tile: 16, sigma: sigma, counter: counterMu)
                let w = muBlockWinRate(orig: entry, mu: muEntry, tile: 16)
                muWins[layerRow] += w.wins
                muBlocks[layerRow] += w.total
            case 1:
                walkStreamUpper(entry: muEntry, parent: frame.parents[e], tile: 8, sigma: sigma, counter: counterMu)
                let w = muBlockWinRate(orig: entry, mu: muEntry, tile: 8)
                muWins[layerRow] += w.wins
                muBlocks[layerRow] += w.total
            default:
                walkStreamBase(entry: muEntry, sigma: sigma, counter: counterMu)
                let w = muBlockWinRate(orig: entry, mu: muEntry, tile: 4)
                muWins[layerRow] += w.wins
                muBlocks[layerRow] += w.total
            }
            for vi in variants.indices {
                let cost: (adaptive: Int, modelOnly: Int)
                if vi == 0 {
                    cost = costBase6Q8(run: counterMu.run[vi], val: counterMu.val[vi])
                } else {
                    cost = costVariantQ8(run: counterMu.run[vi], val: counterMu.val[vi])
                }
                aggMu[layerRow].adaptiveQ8[vi] += cost.adaptive
                aggMu[layerRow].modelOnlyQ8[vi] += cost.modelOnly
                let bw: Int
                if backwardMu.primed[e][vi] {
                    bw = costBackwardQ8(curRun: counterMu.run[vi], curVal: counterMu.val[vi], accRun: backwardMu.run[e][vi], accVal: backwardMu.val[e][vi])
                } else {
                    bw = cost.adaptive
                }
                aggMu[layerRow].backwardQ8[vi] += bw
                for c in 0..<variants[vi].contextCount {
                    for t in 0..<64 {
                        backwardMu.run[e][vi][c][t] = backwardMu.run[e][vi][c][t] / 2 + counterMu.run[vi][c][t]
                        backwardMu.val[e][vi][c][t] = backwardMu.val[e][vi][c][t] / 2 + counterMu.val[vi][c][t]
                    }
                }
                backwardMu.primed[e][vi] = true
            }
            aggMu[layerRow].sharedBits += counterMu.sharedBits
        }

        // layerBytes dump order is (L0, L1, L2); agg rows are (L2, L1, L0)
        agg[0].actualBytes += frame.layerBytes[2]
        agg[1].actualBytes += frame.layerBytes[1]
        agg[2].actualBytes += frame.layerBytes[0]

        prev = frame
        frameCount += 1
    }

    guard 0 < frameCount else { throw SigmaMeasureError.badFormat("no frames in dump") }

    func kb(_ bits: Double) -> String { String(format: "%9.1f", bits / 8.0 / 1024.0) }
    func pct(_ base: Double, _ v: Double) -> String {
        base <= 0 ? "      - " : String(format: "%+6.2f%%", (base - v) / base * 100.0)
    }
    func pad(_ s: String, _ n: Int) -> String {
        s.count < n ? s + String(repeating: " ", count: n - s.count) : s
    }
    func row(_ name: String, _ cols: [String]) -> String {
        "  " + pad(name, 22) + cols.map { pad($0, 11) }.joined(separator: " ") + "\n"
    }

    var out = ""
    out += "=== sigma-conditioned entropy measurement ===\n"
    out += "frames: \(frameCount) P-frames (\(coldCount) cold starts), resolution \(width)x\(height)\n"
    out += "columns: adaptive    = per-frame dynamic tables + headers (merged if cheaper)\n"
    out += "         prev-tables = header-free backward adaptation from decoded history\n"
    out += "         model-only  = conditional-entropy bound (ideal static tables)\n\n"

    var totalActual = 0
    var totalAdaptive = [Double](repeating: 0, count: variants.count)
    var totalBackward = [Double](repeating: 0, count: variants.count)
    var totalModelOnly = [Double](repeating: 0, count: variants.count)

    let layerNames = ["Layer 2 (full res)", "Layer 1 (half res)", "Layer 0 (base)"]
    for li in 0..<3 {
        let a = agg[li]
        let shared = Double(a.sharedBits)
        totalActual += a.actualBytes
        out += "[\(layerNames[li])]  actual=\(String(format: "%.1f", Double(a.actualBytes) / 1024.0)) KB  shared(flags+bypass)=\(String(format: "%.1f", shared / 8.0 / 1024.0)) KB\n"
        let baseAd = Double(a.adaptiveQ8[0]) / 256.0 + shared
        let baseBw = Double(a.backwardQ8[0]) / 256.0 + shared
        let baseMo = Double(a.modelOnlyQ8[0]) / 256.0 + shared
        out += row("variant", ["adaptive KB", "delta", "prevTbl KB", "delta", "modelOnly", "delta"])
        for vi in variants.indices {
            let ad = Double(a.adaptiveQ8[vi]) / 256.0 + shared
            let bw = Double(a.backwardQ8[vi]) / 256.0 + shared
            let mo = Double(a.modelOnlyQ8[vi]) / 256.0 + shared
            totalAdaptive[vi] += ad
            totalBackward[vi] += bw
            totalModelOnly[vi] += mo
            out += row(variants[vi].name, [kb(ad), pct(baseAd, ad), kb(bw), pct(baseBw, bw), kb(mo), pct(baseMo, mo)])
        }
        out += "\n"
    }

    out += "[TOTAL]  actual=\(String(format: "%.1f", Double(totalActual) / 1024.0)) KB (coeff layers, excl. frame headers/MV/skipMap)\n"
    out += row("variant", ["adaptive KB", "delta", "prevTbl KB", "delta", "modelOnly", "delta"])
    for vi in variants.indices {
        out += row(variants[vi].name, [
            kb(totalAdaptive[vi]), pct(totalAdaptive[0], totalAdaptive[vi]),
            kb(totalBackward[vi]), pct(totalBackward[0], totalBackward[vi]),
            kb(totalModelOnly[vi]), pct(totalModelOnly[0], totalModelOnly[vi]),
        ])
    }
    out += "\nsanity: simulated base6 adaptive total vs actual coeff bytes = "
    out += String(format: "%.1f KB vs %.1f KB (%.1f%%)\n", totalAdaptive[0] / 8.0 / 1024.0, Double(totalActual) / 1024.0, (totalAdaptive[0] / 8.0) / Double(max(1, totalActual)) * 100.0)

    // μ-prediction report: original vs residual planes, same costing.
    // Deltas are relative to the SAME variant/column on the original planes,
    // i.e. the pure gain of the integer prediction lifting step.
    out += "\n=== mu-prediction (r = c - pred(prev coeff)) vs original planes ===\n"
    for vi in [0, 2] {
        out += "[variant: \(variants[vi].name)]\n"
        out += row("layer", ["adaptive KB", "delta", "prevTbl KB", "delta", "modelOnly", "delta"])
        var tO = [0.0, 0.0, 0.0]
        var tM = [0.0, 0.0, 0.0]
        for li in 0..<3 {
            let o = agg[li]
            let m = aggMu[li]
            let oCols = [Double(o.adaptiveQ8[vi]) / 256.0 + Double(o.sharedBits),
                         Double(o.backwardQ8[vi]) / 256.0 + Double(o.sharedBits),
                         Double(o.modelOnlyQ8[vi]) / 256.0 + Double(o.sharedBits)]
            let mCols = [Double(m.adaptiveQ8[vi]) / 256.0 + Double(m.sharedBits),
                         Double(m.backwardQ8[vi]) / 256.0 + Double(m.sharedBits),
                         Double(m.modelOnlyQ8[vi]) / 256.0 + Double(m.sharedBits)]
            for k in 0..<3 {
                tO[k] += oCols[k]
                tM[k] += mCols[k]
            }
            out += row(layerNames[li], [kb(mCols[0]), pct(oCols[0], mCols[0]), kb(mCols[1]), pct(oCols[1], mCols[1]), kb(mCols[2]), pct(oCols[2], mCols[2])])
        }
        out += row("TOTAL", [kb(tM[0]), pct(tO[0], tM[0]), kb(tM[1]), pct(tO[1], tM[1]), kb(tM[2]), pct(tO[2], tM[2])])
        out += "\n"
    }
    out += "mu per-block win rate (residual strictly sparser than original; oracle upper bound for per-block signaling):\n"
    for li in 0..<3 {
        let rate = muBlocks[li] == 0 ? 0.0 : Double(muWins[li]) / Double(muBlocks[li]) * 100.0
        out += String(format: "  %@ %6.2f%% (%d / %d blocks)\n", pad(layerNames[li], 20), rate, muWins[li], muBlocks[li])
    }
    return out
}
