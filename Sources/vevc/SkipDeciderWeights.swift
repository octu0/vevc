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
        return SkipDeciderWeights(f: f, h: h, mu: mu, scale: scale, normShift: normShift, w1: w1, b1: b1, w2: w2, b2: b2, shift1: shift1)
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
