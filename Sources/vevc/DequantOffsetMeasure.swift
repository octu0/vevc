// MARK: - Conditional Dequantization Offset Measurement (offline)
//
// The dead-zone quantizer reconstructs bin k at its nominal position k·Δ,
// but the coefficient distribution inside each bin is far from uniform
// (Laplacian-like), so the MSE-optimal reconstruction point is the bin
// centroid. This tool fits a Laplacian scale per (layer, plane, subband)
// from the quantized magnitude histograms of a VEVC_DUMP_COEFFS dump,
// derives the centroid offsets against the current k·Δ reconstruction, and
// predicts the residual-domain MSE/PSNR gain of centroid reconstruction.
//
// Bin geometry mirrors Quant.swift exactly: q = floor(|c|/Δ + β) with
// β = bias/65536 (round-to-nearest β=0.5 for qLow; per-layer dead-zone
// biases for qMid/qHigh). Since dequantize* is shared by the encoder's
// reconstruction loop and the decoder, applying an offset there keeps both
// sides in sync with no signaling.

import Foundation

private struct SubbandFit {
    var counts: [Int] = [Int](repeating: 0, count: 64)  // magnitude histogram (k = |q|)
    var total = 0
}

/// β (bias in bin units) replicated from QuantizationTable.init.
private func betaFor(entry e: Int, sub: Int) -> Double {
    let layerRow = e / 3          // 0 → L2, 1 → L1, 2 → L0
    let chroma = (e % 3) != 0
    let layerIndex = [2, 1, 0][layerRow]
    if layerRow == 2 && sub == 0 {
        return 0.5  // qLow: roundToNearest
    }
    // subband → mid or high (L2/L1: HL,LH,HH ; L0: LL,HL,LH,HH)
    let isHigh = (layerRow == 2) ? (sub == 3) : (sub == 2)
    let dz: Double
    if chroma {
        switch layerIndex {
        case 0: dz = isHigh ? -16000 : -8000
        case 1: dz = isHigh ? -32000 : -16000
        default: dz = isHigh ? -64000 : -32000
        }
    } else {
        switch layerIndex {
        case 0: dz = isHigh ? 8192 : 16384
        case 1: dz = isHigh ? 0 : 8192
        default: dz = isHigh ? -8000 : 0
        }
    }
    return dz / 65536.0
}

/// Centroid and second moment of an exponential density e^{-x/b} on [a1, a2].
private func exponentialBinStats(a1: Double, a2: Double, b: Double) -> (mass: Double, mean: Double, m2: Double) {
    // numeric integration (Simpson, 64 intervals) — robust for all bin shapes
    let n = 64
    let h = (a2 - a1) / Double(n)
    var s0 = 0.0
    var s1 = 0.0
    var s2 = 0.0
    for i in 0...n {
        let x = a1 + Double(i) * h
        let w = (i == 0 || i == n) ? 1.0 : (i % 2 == 1 ? 4.0 : 2.0)
        let f = exp(-x / b)
        s0 += w * f
        s1 += w * f * x
        s2 += w * f * x * x
    }
    let mass = s0 * h / 3.0
    return (mass, s1 / s0, s2 / s0)
}

public func runDequantOffsetEstimation(dumpPath: String) throws -> String {
    let reader = try DumpReader(path: dumpPath)
    // fits[entry][sub]
    var fits = [[SubbandFit]](repeating: [], count: 9)
    var qsteps: [[Int]] = []
    var frameCount = 0

    while let frame = try reader.nextFrame() {
        if qsteps.isEmpty { qsteps = frame.qsteps }
        for e in 0..<9 {
            let entry = frame.coded[e]
            if fits[e].isEmpty {
                fits[e] = [SubbandFit](repeating: SubbandFit(), count: entry.subs.count)
            }
            let layerRow = e / 3
            for s in 0..<entry.subs.count {
                let isSignedLL = (layerRow == 2 && s == 0)
                let p = entry.subs[s]
                var f = fits[e][s]
                for v in p.data {
                    let mag = isSignedLL ? abs(Int(v)) : abs(unzigzag(v))
                    if mag < 64 {
                        f.counts[mag] += 1
                    }
                    f.total += 1
                }
                fits[e][s] = f
            }
        }
        frameCount += 1
    }
    guard 0 < frameCount else { throw SigmaMeasureError.badFormat("no frames") }

    let entryNames = ["L2 Y", "L2 Cb", "L2 Cr", "L1 Y", "L1 Cb", "L1 Cr", "L0 Y", "L0 Cb", "L0 Cr"]
    var out = "=== dequantization centroid offsets (Laplacian fit from \(frameCount) P-frames) ===\n"
    out += "delta_k in units of the real step (Δ = step/16); current reconstruction is k·Δ\n"
    out += String(format: "%-8@ %-4@ %8@ %8@ %8@ %8@ %10@ %12@\n", "entry", "sub", "b/Δ", "δ1/Δ", "δ2/Δ", "δ3/Δ", "nz-frac", "ΔMSE(resid)")

    var totalGain = [0.0, 0.0, 0.0]   // per layerRow, summed δ²-weighted counts
    var totalPix = [0.0, 0.0, 0.0]
    var totalMSE = [0.0, 0.0, 0.0]    // current quantization MSE (model), same domain

    for e in 0..<9 {
        let layerRow = e / 3
        let subNames = fits[e].count == 4 ? ["LL", "HL", "LH", "HH"] : ["HL", "LH", "HH"]
        for s in 0..<fits[e].count {
            let f = fits[e][s]
            let nz = (1..<64).reduce(0) { $0 + f.counts[$1] }
            guard 100 < nz else { continue }
            // geometric ratio fit on k>=2 tail (k=1 is distorted by
            // zero-threshold clearing at encode time)
            var num = 0.0
            var den = 0.0
            for k in 2..<8 {
                num += Double(f.counts[k + 1])
                den += Double(f.counts[k])
            }
            let r = den <= 0 ? 0.3 : max(0.02, min(0.95, num / den))
            let bOverD = -1.0 / log(r)   // b in Δ units
            let beta = betaFor(entry: e, sub: s)

            var d = [0.0, 0.0, 0.0]
            var gain = 0.0
            var mse = 0.0
            for k in 1..<16 {
                let a1 = (Double(k) - beta)
                let a2 = (Double(k) + 1.0 - beta)
                let st = exponentialBinStats(a1: max(0, a1), a2: a2, b: bOverD)
                let delta = st.mean - Double(k)
                if k <= 3 { d[k - 1] = delta }
                let cnt = Double(f.counts[min(k, 63)])
                gain += cnt * delta * delta
                // current MSE within bin: E[(c - k)^2] = m2 - 2k·mean + k²  (Δ units)
                mse += cnt * (st.m2 - 2.0 * Double(k) * st.mean + Double(k) * Double(k))
            }
            // zero-bin MSE (reconstruction at 0): E[c²] on [0, (1-β)Δ), symmetric
            let z = exponentialBinStats(a1: 0, a2: max(0.05, 1.0 - beta), b: bOverD)
            mse += Double(f.counts[0]) * z.m2 * (z.mass > 0 ? 1.0 : 0.0)

            let step = Double(stepFor(entry: e, sub: s, qsteps: qsteps)) / 16.0
            totalGain[layerRow] += gain * step * step
            totalMSE[layerRow] += mse * step * step
            totalPix[layerRow] += Double(f.total)

            out += String(format: "%-8@ %-4@ %8.3f %8.3f %8.3f %8.3f %9.2f%% %11.4f\n",
                          entryNames[e], subNames[s], bOverD, d[0], d[1], d[2],
                          Double(nz) / Double(f.total) * 100.0,
                          gain * step * step / Double(f.total))
        }
    }

    out += "\nper-layer predicted gain (residual domain, LeGall synthesis-gain caveat applies):\n"
    let layerNames = ["Layer 2", "Layer 1", "Layer 0"]
    for l in 0..<3 {
        guard 0 < totalPix[l], 0 < totalMSE[l] else { continue }
        let mseBefore = totalMSE[l] / totalPix[l]
        let mseAfter = (totalMSE[l] - totalGain[l]) / totalPix[l]
        let dPSNR = 10.0 * log10(mseBefore / max(1e-9, mseAfter))
        out += String(format: "  %@: quantization MSE %.4f → %.4f  (ΔPSNR_resid +%.3f dB)\n", layerNames[l], mseBefore, mseAfter, dPSNR)
    }
    return out
}
