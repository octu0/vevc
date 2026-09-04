// MARK: - Learned Motion Estimation Decider (Profile 2 P-frames)
//
// An 8-8-4 integer SIMD neural decider that classifies motion search mode and
// reference affinity per 8x8 block into three search regimes and one LTR affinity:
//
// 1. .earlyCandidate (0): Candidate or zero vector is dominant; skip ±2 near-refinement
//    and skip LDSP diamond search, eliminating 24-40 SAD evaluations per block.
// 2. .refineNear (1): Candidate is in the basin of attraction; run ±2 integer search
//    (24 offsets) with hoisted boundary checks, skip LDSP.
// 3. .wideDiamond (2): Poor match or dynamic scene motion; run ±2 refinement and LDSP
//    large diamond pattern search.
// 4. ltrScore (output 3): Directional affinity score tracking LTR background persistence.
//
// Inference is 100% integer SIMD (Int32 accumulators, ReLU, arithmetic shifts),
// zero dynamic heap allocations, and branch-deterministic.

public enum MESearchMode: Int32, Sendable {
    case earlyCandidate = 0
    case refineNear = 1
    case wideDiamond = 2
}

public struct MEDecision: Sendable {
    public let mode: MESearchMode
    public let ltrScore: Int32

    @inline(__always)
    public init(mode: MESearchMode, ltrScore: Int32) {
        self.mode = mode
        self.ltrScore = ltrScore
    }
}

public enum MEFeature {
    public static let count = 8

    public static let zeroSAD = 0
    public static let candSAD = 1
    public static let pmvSAD = 2
    public static let costDelta = 3
    public static let candMvMag = 4
    public static let contrast = 5
    public static let membrane = 6
    public static let gopPosition = 7
}

public struct MEDeciderWeights: Sendable {
    public let f: Int
    public let h: Int
    public let o: Int
    public let mu: [Int32]
    public let scale: [Int32]
    public let normShift: Int32
    public let w1: [Int32]
    public let b1: [Int32]
    public let w2: [Int32]
    public let b2: [Int32]
    public let shift1: Int32

    // SIMD precomputed weights
    public let muSIMD: SIMD8<Int32>
    public let scaleSIMD: SIMD8<Int32>
    public let normShiftSIMD: SIMD8<Int32>
    public let b1SIMD: SIMD8<Int32>
    public let b2SIMD: SIMD4<Int32>
    public let w1_0: SIMD8<Int32>
    public let w1_1: SIMD8<Int32>
    public let w1_2: SIMD8<Int32>
    public let w1_3: SIMD8<Int32>
    public let w1_4: SIMD8<Int32>
    public let w1_5: SIMD8<Int32>
    public let w1_6: SIMD8<Int32>
    public let w1_7: SIMD8<Int32>
    public let w2_0: SIMD8<Int32>
    public let w2_1: SIMD8<Int32>
    public let w2_2: SIMD8<Int32>
    public let w2_3: SIMD8<Int32>
    public let w1SIMD: [SIMD8<Int32>]
    public let w2SIMD: [SIMD8<Int32>]

    // BitNet b1.58 ternary masks and scaling factors
    public let w1PosMask: [SIMD8<Int32>]
    public let w1NegMask: [SIMD8<Int32>]
    public let w1Alpha: [Int32]
    public let w2PosMask: [SIMD8<Int32>]
    public let w2NegMask: [SIMD8<Int32>]
    public let w2Alpha: [Int32]

    public static func parse(_ a: [Int32]) -> MEDeciderWeights? {
        if a.count < 8 { return nil }
        if a[0] != 0x4D454431 { return nil } // 'MED1'
        if a[1] != 1 { return nil }
        let f = Int(a[2])
        let h = Int(a[3])
        let o = Int(a[4])
        if f != 8 || h != 8 || o != 4 { return nil }

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
        let w2 = take(o * h)
        let b2 = take(o)
        let shift1 = take(1).first ?? 6

        switch true {
        case mu.count != f, scale.count != f:
            return nil
        case w1.count != h * f, b1.count != h:
            return nil
        case w2.count != o * h, b2.count != o:
            return nil
        default:
            break
        }

        var muVec = SIMD8<Int32>()
        var scaleVec = SIMD8<Int32>()
        for i in 0..<8 {
            muVec[i] = mu[i]
            scaleVec[i] = scale[i]
        }

        var simdW1 = [SIMD8<Int32>]()
        simdW1.reserveCapacity(h)
        var p1Masks = [SIMD8<Int32>]()
        var n1Masks = [SIMD8<Int32>]()
        var a1List = [Int32]()
        p1Masks.reserveCapacity(h)
        n1Masks.reserveCapacity(h)
        a1List.reserveCapacity(h)

        var r = 0
        while r < h {
            var v = SIMD8<Int32>()
            var sumAbs: Int32 = 0
            var c = 0
            while c < f {
                let val = w1[r * f + c]
                v[c] = val
                if val < 0 {
                    sumAbs &+= (0 &- val)
                } else {
                    sumAbs &+= val
                }
                c += 1
            }
            simdW1.append(v)

            let alpha = max(1, (sumAbs + Int32(f / 2)) / Int32(f))
            let tau = max(1, alpha / 2)
            a1List.append(alpha)

            var pMask = SIMD8<Int32>()
            var nMask = SIMD8<Int32>()
            var colIdx = 0
            while colIdx < f {
                let wVal = w1[r * f + colIdx]
                if tau < wVal {
                    pMask[colIdx] = -1
                } else {
                    if wVal < -tau {
                        nMask[colIdx] = -1
                    }
                }
                colIdx += 1
            }
            p1Masks.append(pMask)
            n1Masks.append(nMask)
            r += 1
        }

        var simdW2 = [SIMD8<Int32>]()
        simdW2.reserveCapacity(o)
        var p2Masks = [SIMD8<Int32>]()
        var n2Masks = [SIMD8<Int32>]()
        var a2List = [Int32]()
        p2Masks.reserveCapacity(o)
        n2Masks.reserveCapacity(o)
        a2List.reserveCapacity(o)

        var outIdx = 0
        while outIdx < o {
            var v = SIMD8<Int32>()
            var sumAbs: Int32 = 0
            var c = 0
            while c < h {
                let val = w2[outIdx * h + c]
                v[c] = val
                if val < 0 {
                    sumAbs &+= (0 &- val)
                } else {
                    sumAbs &+= val
                }
                c += 1
            }
            simdW2.append(v)

            let alpha = max(1, (sumAbs + Int32(h / 2)) / Int32(h))
            let tau = max(1, alpha / 2)
            a2List.append(alpha)

            var pMask = SIMD8<Int32>()
            var nMask = SIMD8<Int32>()
            var hIdx = 0
            while hIdx < h {
                let wVal = w2[outIdx * h + hIdx]
                if tau < wVal {
                    pMask[hIdx] = -1
                } else {
                    if wVal < -tau {
                        nMask[hIdx] = -1
                    }
                }
                hIdx += 1
            }
            p2Masks.append(pMask)
            n2Masks.append(nMask)
            outIdx += 1
        }

        var b1Vec = SIMD8<Int32>()
        for i in 0..<8 {
            b1Vec[i] = b1[i]
        }
        var b2Vec = SIMD4<Int32>()
        for i in 0..<4 {
            b2Vec[i] = b2[i]
        }

        return MEDeciderWeights(
            f: f, h: h, o: o,
            mu: mu, scale: scale, normShift: normShift,
            w1: w1, b1: b1, w2: w2, b2: b2, shift1: shift1,
            muSIMD: muVec, scaleSIMD: scaleVec,
            normShiftSIMD: SIMD8<Int32>(repeating: normShift),
            b1SIMD: b1Vec, b2SIMD: b2Vec,
            w1_0: simdW1[0], w1_1: simdW1[1], w1_2: simdW1[2], w1_3: simdW1[3],
            w1_4: simdW1[4], w1_5: simdW1[5], w1_6: simdW1[6], w1_7: simdW1[7],
            w2_0: simdW2[0], w2_1: simdW2[1], w2_2: simdW2[2], w2_3: simdW2[3],
            w1SIMD: simdW1, w2SIMD: simdW2,
            w1PosMask: p1Masks, w1NegMask: n1Masks, w1Alpha: a1List,
            w2PosMask: p2Masks, w2NegMask: n2Masks, w2Alpha: a2List
        )
    }

    public static let shared: MEDeciderWeights = {
        guard let w = MEDeciderWeights.parse(MEDeciderWeightsData.blob) else {
            fatalError("Failed to parse MEDeciderWeightsData")
        }
        return w
    }()
}

public enum MEDeciderWeightsData {
    // Serialization layout:
    // [0..4]: magic(0x4D454431), kind(1), f(8), h(8), o(4)
    // [5..12]: mu[8]
    // [13..20]: scale[8]
    // [21]: normShift(8)
    // [22..85]: w1[64]
    // [86..93]: b1[8]
    // [94..125]: w2[32]
    // [126..129]: b2[4]
    // [130]: shift1(6)
    public static let blob: [Int32] = [
        0x4D454431, 1, 8, 8, 4,
        // mu[8]
        48, 48, 48, 0, 0, 32, 0, 8,
        // scale[8]
        64, 64, 64, 16, 64, 32, 128, 64,
        // normShift
        8,
        // w1[64] (8x8)
        -32, -64,   0, -16, -16,   0,  16,   0,
        -32, -16,   0, -16, -32,   0,  48,   8,
        -32,   0,   0, -32, -16,   0,  16,   0,
          0,  64,   0,  16,  16,   0, -16,   0,
         16,  64,  16,   0,  16,   0, -32,   0,
          0,  32,   0,  16,  32,   0, -48,   0,
          0, -16,   0,   0, -16,  -8,  64,  16,
          0,  16,   0,   0,   0,  32,   0,   0,
        // b1[8]
        0, 32, 48, 0, -5376, -1200, 32, 0,
        // w2[32] (4x8)
         64,  32,  32, -32,  -32, -32,   0,   0,
        -32, -16, -16,  32, -128, -16,   0,   8,
        -64, -32, -32, -16,  160,  48,   0,   0,
          8,  24,   0,   0,  -16,  -8,  48,   0,
        // b2[4]
        0, 32, 0, -64,
        // shift1
        6
    ]
}

@inline(__always)
public func meDeciderNormalize(
    _ x: SIMD8<Int32>,
    mu: SIMD8<Int32>,
    scale: SIMD8<Int32>,
    normShift: SIMD8<Int32>
) -> SIMD8<Int32> {
    let diff = x &- mu
    let scaled = (diff &* scale) &>> normShift
    return scaled.clamped(
        lowerBound: SIMD8<Int32>(repeating: -127),
        upperBound: SIMD8<Int32>(repeating: 127)
    )
}

/// Fast exact integer SIMD classification.
@inline(__always)
public func meDeciderClassify(
    q: SIMD8<Int32>,
    weights: MEDeciderWeights
) -> MEDecision {
    let normQ = meDeciderNormalize(
        q,
        mu: weights.muSIMD,
        scale: weights.scaleSIMD,
        normShift: weights.normShiftSIMD
    )

    let sh = Int(weights.shift1)
    let h0 = max(0, (weights.b1SIMD[0] &+ (weights.w1_0 &* normQ).wrappedSum()) >> sh)
    let h1 = max(0, (weights.b1SIMD[1] &+ (weights.w1_1 &* normQ).wrappedSum()) >> sh)
    let h2 = max(0, (weights.b1SIMD[2] &+ (weights.w1_2 &* normQ).wrappedSum()) >> sh)
    let h3 = max(0, (weights.b1SIMD[3] &+ (weights.w1_3 &* normQ).wrappedSum()) >> sh)
    let h4 = max(0, (weights.b1SIMD[4] &+ (weights.w1_4 &* normQ).wrappedSum()) >> sh)
    let h5 = max(0, (weights.b1SIMD[5] &+ (weights.w1_5 &* normQ).wrappedSum()) >> sh)
    let h6 = max(0, (weights.b1SIMD[6] &+ (weights.w1_6 &* normQ).wrappedSum()) >> sh)
    let h7 = max(0, (weights.b1SIMD[7] &+ (weights.w1_7 &* normQ).wrappedSum()) >> sh)
    let hVec = SIMD8<Int32>(h0, h1, h2, h3, h4, h5, h6, h7)

    let s0 = weights.b2SIMD[0] &+ (weights.w2_0 &* hVec).wrappedSum()
    let s1 = weights.b2SIMD[1] &+ (weights.w2_1 &* hVec).wrappedSum()
    let s2 = weights.b2SIMD[2] &+ (weights.w2_2 &* hVec).wrappedSum()
    let s3 = weights.b2SIMD[3] &+ (weights.w2_3 &* hVec).wrappedSum()

    let mode: MESearchMode
    switch true {
    case s1 <= s0 && s2 <= s0:
        mode = .earlyCandidate
    case s2 <= s1:
        mode = .refineNear
    default:
        mode = .wideDiamond
    }

    return MEDecision(mode: mode, ltrScore: s3)
}

/// BitNet b1.58 ternary-weight forward pass (multiplication-free inner product).
@inline(__always)
public func meDeciderClassifyBitNet(
    q: SIMD8<Int32>,
    weights: MEDeciderWeights
) -> MEDecision {
    let normQ = meDeciderNormalize(
        q,
        mu: weights.muSIMD,
        scale: weights.scaleSIMD,
        normShift: weights.normShiftSIMD
    )

    var hVec = SIMD8<Int32>()
    var i = 0
    while i < 8 {
        let p = weights.w1PosMask[i]
        let n = weights.w1NegMask[i]
        let posSum = (p & normQ).wrappedSum()
        let negSum = (n & normQ).wrappedSum()
        let dot = (posSum &- negSum) &* weights.w1Alpha[i]
        let acc = (weights.b1[i] &+ dot) >> weights.shift1
        hVec[i] = max(0, acc)
        i += 1
    }

    var s = SIMD4<Int32>()
    var k = 0
    while k < 4 {
        let p = weights.w2PosMask[k]
        let n = weights.w2NegMask[k]
        let posSum = (p & hVec).wrappedSum()
        let negSum = (n & hVec).wrappedSum()
        let dot = (posSum &- negSum) &* weights.w2Alpha[k]
        s[k] = weights.b2[k] &+ dot
        k += 1
    }

    let mode: MESearchMode
    switch true {
    case s[1] <= s[0] && s[2] <= s[0]:
        mode = .earlyCandidate
    case s[2] <= s[1]:
        mode = .refineNear
    default:
        mode = .wideDiamond
    }

    return MEDecision(mode: mode, ltrScore: s[3])
}
