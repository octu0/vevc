/// Trained coefficients of the decider. The stored layout mirrors the trainer's
/// serialization so the compiled-in table is the training artefact verbatim.
struct SkipDeciderWeights: Sendable {
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

    // BitNet b1.58 precomputed masks and scalers
    let w1PosMask0: [SIMD16<Int32>]
    let w1NegMask0: [SIMD16<Int32>]
    let w1PosMask1: [SIMD4<Int32>]
    let w1NegMask1: [SIMD4<Int32>]
    let w1Alpha: [Int32]

    let w2PosMask: [SIMD16<Int32>]
    let w2NegMask: [SIMD16<Int32>]
    let w2Alpha: [Int32]

    // Precomputed exact SIMD weights for zero-overhead inference
    let w1SIMD0: [SIMD16<Int32>]
    let w1SIMD1: [SIMD4<Int32>]
    let w2Diff: SIMD16<Int32>
    let b2Diff: Int32

    /// Flat Int32 array: magic, kind, f, h, t, mu[f], scale[f], normShift,
    /// w1[h*f], b1[h], w2[2*h], b2[2], shift1, shift2.
    static func parse(_ a: [Int32]) -> SkipDeciderWeights? {
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

        var p0Masks = [SIMD16<Int32>]()
        var n0Masks = [SIMD16<Int32>]()
        var p1Masks = [SIMD4<Int32>]()
        var n1Masks = [SIMD4<Int32>]()
        var a1List = [Int32]()
        p0Masks.reserveCapacity(h)
        n0Masks.reserveCapacity(h)
        p1Masks.reserveCapacity(h)
        n1Masks.reserveCapacity(h)
        a1List.reserveCapacity(h)

        var row = 0
        while row < h {
            var sumAbs: Int32 = 0
            var col = 0
            while col < f {
                let val = w1[row * f + col]
                if val < 0 {
                    sumAbs &+= (0 &- val)
                } else {
                    sumAbs &+= val
                }
                col += 1
            }
            let alpha = max(1, (sumAbs + Int32(f / 2)) / Int32(f))
            let tau = max(1, alpha / 2)
            a1List.append(alpha)

            var p0 = SIMD16<Int32>()
            var n0 = SIMD16<Int32>()
            var c0 = 0
            while c0 < 16 {
                let wVal = w1[row * f + c0]
                if tau < wVal {
                    p0[c0] = -1
                } else {
                    if wVal < -tau {
                        n0[c0] = -1
                    }
                }
                c0 += 1
            }
            p0Masks.append(p0)
            n0Masks.append(n0)

            var p1 = SIMD4<Int32>()
            var n1 = SIMD4<Int32>()
            var c1 = 0
            while c1 < 4 {
                let wVal = w1[row * f + 16 + c1]
                if tau < wVal {
                    p1[c1] = -1
                } else {
                    if wVal < -tau {
                        n1[c1] = -1
                    }
                }
                c1 += 1
            }
            p1Masks.append(p1)
            n1Masks.append(n1)

            row += 1
        }

        var p2Masks = [SIMD16<Int32>]()
        var n2Masks = [SIMD16<Int32>]()
        var a2List = [Int32]()
        p2Masks.reserveCapacity(2)
        n2Masks.reserveCapacity(2)
        a2List.reserveCapacity(2)

        var outIdx = 0
        while outIdx < 2 {
            var sumAbs: Int32 = 0
            var col = 0
            while col < h {
                let val = w2[outIdx * h + col]
                if val < 0 {
                    sumAbs &+= (0 &- val)
                } else {
                    sumAbs &+= val
                }
                col += 1
            }
            let alpha = max(1, (sumAbs + Int32(h / 2)) / Int32(h))
            let tau = max(1, alpha / 2)
            a2List.append(alpha)

            var p2 = SIMD16<Int32>()
            var n2 = SIMD16<Int32>()
            var c = 0
            while c < 16 {
                let wVal = w2[outIdx * h + c]
                if tau < wVal {
                    p2[c] = -1
                } else {
                    if wVal < -tau {
                        n2[c] = -1
                    }
                }
                c += 1
            }
            p2Masks.append(p2)
            n2Masks.append(n2)

            outIdx += 1
        }

        var simdW1_0 = [SIMD16<Int32>]()
        var simdW1_1 = [SIMD4<Int32>]()
        simdW1_0.reserveCapacity(h)
        simdW1_1.reserveCapacity(h)
        var rowIdx = 0
        while rowIdx < h {
            var v0 = SIMD16<Int32>()
            var c0 = 0
            while c0 < 16 {
                v0[c0] = w1[rowIdx * f + c0]
                c0 += 1
            }
            simdW1_0.append(v0)

            var v1 = SIMD4<Int32>()
            var c1 = 0
            while c1 < 4 {
                v1[c1] = w1[rowIdx * f + 16 + c1]
                c1 += 1
            }
            simdW1_1.append(v1)
            rowIdx += 1
        }

        var w2DiffVec = SIMD16<Int32>()
        var hIdx = 0
        while hIdx < h {
            w2DiffVec[hIdx] = w2[h + hIdx] - w2[hIdx]
            hIdx += 1
        }
        let b2DiffVal = b2[1] - b2[0]

        return SkipDeciderWeights(
            f: f, h: h, mu: mu, scale: scale, normShift: normShift,
            w1: w1, b1: b1, w2: w2, b2: b2, shift1: shift1,
            w1PosMask0: p0Masks, w1NegMask0: n0Masks,
            w1PosMask1: p1Masks, w1NegMask1: n1Masks,
            w1Alpha: a1List,
            w2PosMask: p2Masks, w2NegMask: n2Masks,
            w2Alpha: a2List,
            w1SIMD0: simdW1_0,
            w1SIMD1: simdW1_1,
            w2Diff: w2DiffVec,
            b2Diff: b2DiffVal
        )
    }
}

// Generated table: flat little-endian Int32 serialization of the trained
// decider, in the exact layout `SkipDeciderWeights.parse` reads. Do not
// hand-edit.
enum SkipDeciderWeightsData {
    static let blob: [Int32] = [
    1347177777,1,20,16,1,13566,21230,577,894,27,
    38,919,1,16105,15,7,3,0,0,0,
    0,1,13414,102,90,11,6,235,100,3679,
    3200,113,106993,70,16384,38947,37288,131072,1527040,186722,
    168992,263399,11,1488,1938,12,20,5,-72,-1,
    2,-1,-2,2,11,-12,-1,5,-2,-2,
    1,0,-2,-1,-1,6,47,-103,21,-49,
    -2,3,-2,-1,1,-1,6,-2,0,0,
    0,-1,8,9,1,5,10,10,-9,-4,
    -10,7,8,-7,1,0,-12,-30,-1,-9,
    -3,1,-17,-6,-6,8,44,22,-44,-6,
    -1,-1,-3,0,-3,1,-1,-54,-1,-29,
    1,-1,3,-19,-5,7,5,-30,0,-17,
    -40,7,-7,2,0,4,2,0,2,-12,
    3,-3,-3,1,3,-1,11,-112,24,-48,
    1,-1,9,5,2,6,0,-5,2,-15,
    -1,3,3,-8,0,-17,11,-6,64,4,
    -10,4,-6,0,1,0,1,-37,1,-8,
    -1,1,0,3,-2,3,-2,-6,-30,-2,
    9,-8,-5,1,1,9,-5,11,-2,0,
    0,-1,-5,-10,0,-8,49,-2,-58,-3,
    1,3,-5,-1,13,0,0,-3,0,-4,
    -3,-2,-12,5,-7,1,0,-2,-12,-11,
    -6,-6,-1,2,4,-13,-49,20,-2,-6,
    -1,1,11,15,-2,-1,6,10,3,-8,
    -12,6,-8,21,-3,-3,-9,-15,1,-5,
    1,1,-3,4,-6,-3,-11,20,-11,-6,
    -43,7,2,0,6,-3,3,-6,0,-5,
    -4,-5,-6,-8,-4,12,13,-8,-9,10,
    -4,-1,0,0,5,-13,19,-39,1,2,
    3,3,-7,0,-3,0,20,-3,-27,-13,
    -127,0,0,0,1,1,2,-4,1,-17,
    0,2,-4,11,-3,4,9,-10,4,1,
    1,0,1,1,-3,-19,17,-7,-1,15,
    -6,-3,1,-8,-1,3,0,-34,4,-16,
    -13,-22,2,3,1,-2,3,2,0,2,
    -6,-2,20,10,2,-3,-875,-1335,-1848,-1033,
    -595,-2785,-2174,-2254,-463,-2331,-1477,-1722,-1235,-2579,
    -1374,-884,68,-86,37,55,-50,-117,55,64,
    57,70,48,-33,37,-71,71,-32,-40,104,
    -57,-32,21,108,-72,-100,-42,-127,-54,59,
    -51,76,-107,34,-380,380,5,0,
    ]
}
