import Testing
@testable import vevc

@Suite("MotionEstimation Tests")
struct METests {

    @Test("getMVDPenalty calculates zero penalty when MVD is zero")
    func testGetMVDPenaltyZeroMVD() {
        let penalty: Int = MotionEstimation.getMVDPenalty(dx: 5, dy: -3, pmvDx: 5, pmvDy: -3)
        #expect(penalty == 0)
    }

    @Test("getMVDPenalty calculates correctly with positive MVD")
    func testGetMVDPenaltyPositiveMVD() {
        let penalty: Int = MotionEstimation.getMVDPenalty(dx: 10, dy: 8, pmvDx: 5, pmvDy: 4)
        // mvdX = abs(10 - 5) = 5
        // mvdY = abs(8 - 4) = 4
        // penalty = 5 + 4 = 9
        #expect(penalty == 9)
    }

    @Test("getMVDPenalty calculates correctly with negative MVD")
    func testGetMVDPenaltyNegativeMVD() {
        let penalty: Int = MotionEstimation.getMVDPenalty(dx: -5, dy: -10, pmvDx: 2, pmvDy: -2)
        // mvdX = abs(-5 - 2) = 7
        // mvdY = abs(-10 - (-2)) = abs(-8) = 8
        // penalty = 7 + 8 = 15
        #expect(penalty == 15)
    }

    @Test("getMVDPenalty calculates correctly with asymmetric differences")
    func testGetMVDPenaltyAsymmetricMVD() {
        let penalty: Int = MotionEstimation.getMVDPenalty(dx: 0, dy: 15, pmvDx: 5, pmvDy: 15)
        // mvdX = abs(0 - 5) = 5
        // mvdY = abs(15 - 15) = 0
        // penalty = 5 + 0 = 5
        #expect(penalty == 5)
    }

    @Test("getMVDPenalty calculates correctly with large differences")
    func testGetMVDPenaltyLargeMVD() {
        let penalty: Int = MotionEstimation.getMVDPenalty(dx: 100, dy: -50, pmvDx: 0, pmvDy: 0)
        // mvdX = 100
        // mvdY = 50
        // penalty = 150
        #expect(penalty == 150)
    }

    @Test("MEMembraneState accumulates and resets potentials correctly")
    func testMEMembraneStateUpdateAndReset() {
        let mem = MEMembraneState(count: 4)
        #expect(mem.potentials.count == 4)
        for p in mem.potentials {
            #expect(p == 0)
        }

        let mvs = MotionVectors(dx: [0, 0, 4, 0], dy: [0, 0, 0, 0])
        let refDirs = [true, false, false, false]
        let skipMap: [BlockMode] = [.inter, .inter, .inter, .skip_ltr]

        mem.update(mvs: mvs, refDirs: refDirs, skipMap: skipMap)
        #expect(mem.potentials[0] == 3)
        #expect(mem.potentials[1] == 1)
        #expect(mem.potentials[2] == -2)
        #expect(mem.potentials[3] == 3)

        for _ in 0..<5 {
            mem.update(mvs: mvs, refDirs: refDirs, skipMap: skipMap)
        }
        #expect(mem.potentials[0] == 8)
        #expect(mem.potentials[1] == 6)
        #expect(mem.potentials[2] == -8)
        #expect(mem.potentials[3] == 8)

        mem.reset()
        for p in mem.potentials {
            #expect(p == 0)
        }
    }

    @Test("MEDeciderWeights parses blob correctly")
    func testMEDeciderWeightsParse() {
        let weights = MEDeciderWeights.parse(MEDeciderWeightsData.blob)
        #expect(weights != nil)
        if let w = weights {
            #expect(w.f == 8)
            #expect(w.h == 8)
            #expect(w.o == 4)
            #expect(w.w1SIMD.count == 8)
            #expect(w.w2SIMD.count == 4)
            #expect(w.w1PosMask.count == 8)
            #expect(w.w1NegMask.count == 8)
            #expect(w.w2PosMask.count == 4)
            #expect(w.w2NegMask.count == 4)
        }
    }

    @Test("MEDecider classifies small residual / static as earlyCandidate")
    func testMEDeciderClassifyEarlyCandidate() {
        let weights = MEDeciderWeights.shared
        // Low SAD, zero MV, static membrane
        let q = SIMD8<Int32>(20, 20, 20, 0, 0, 16, 4, 2)
        let dec = meDeciderClassify(q: q, weights: weights)
        #expect(dec.mode == .earlyCandidate)
    }

    @Test("MEDecider classifies moderate motion as refineNear")
    func testMEDeciderClassifyRefineNear() {
        let weights = MEDeciderWeights.shared
        // Moderate candidate SAD (120), small MV (4)
        let q = SIMD8<Int32>(180, 120, 150, 40, 4, 32, 0, 4)
        let dec = meDeciderClassify(q: q, weights: weights)
        #expect(dec.mode == .refineNear)
    }

    @Test("MEDecider classifies large mismatch as wideDiamond")
    func testMEDeciderClassifyWideDiamond() {
        let weights = MEDeciderWeights.shared
        // Large SAD (480), dynamic motion, negative membrane
        let q = SIMD8<Int32>(520, 480, 500, 30, 12, 48, -4, 8)
        let dec = meDeciderClassify(q: q, weights: weights)
        #expect(dec.mode == .wideDiamond)
    }

    @Test("MEDecider BitNet inference is functional")
    func testMEDeciderClassifyBitNet() {
        let weights = MEDeciderWeights.shared
        let q = SIMD8<Int32>(20, 20, 20, 0, 0, 16, 4, 2)
        let decBitNet = meDeciderClassifyBitNet(q: q, weights: weights)
        #expect(decBitNet.mode == .earlyCandidate)
    }

    @Test("MEDecider elevates LTR score for persistent static background")
    func testMEDeciderLtrScore() {
        let weights = MEDeciderWeights.shared
        let qLtr = SIMD8<Int32>(30, 30, 30, 0, 0, 16, 6, 16)
        let dec = meDeciderClassify(q: qLtr, weights: weights)
        #expect(0 < dec.ltrScore)
    }

    @Test("meRefine2Offsets covers all 24 points without duplicates or omissions")
    func testRefineNear2OffsetsCompleteAndUnique() {
        #expect(meRefine2OffsetsX.count == 24)
        #expect(meRefine2OffsetsY.count == 24)

        var seen = Set<String>()
        var i = 0
        while i < 24 {
            let x = meRefine2OffsetsX[i]
            let y = meRefine2OffsetsY[i]
            #expect(-2 <= x && x <= 2)
            #expect(-2 <= y && y <= 2)
            #expect(x != 0 || y != 0)
            let key = "\(x),\(y)"
            #expect(seen.contains(key) != true)
            seen.insert(key)
            i += 1
        }
        #expect(seen.count == 24)
    }

    @Test("MEDecider direct SIMD weights and normalization consistency")
    func testMEDeciderDirectSIMDConsistency() {
        let w = MEDeciderWeights.shared
        var r = 0
        while r < 8 {
            #expect(w.b1SIMD[r] == w.b1[r])
            r += 1
        }
        var k = 0
        while k < 4 {
            #expect(w.b2SIMD[k] == w.b2[k])
            k += 1
        }
        #expect(w.w1_0 == w.w1SIMD[0])
        #expect(w.w1_7 == w.w1SIMD[7])
        #expect(w.w2_0 == w.w2SIMD[0])
        #expect(w.w2_3 == w.w2SIMD[3])

        // Clamping edge check
        let hugeVec = SIMD8<Int32>(repeating: 100000)
        let normHuge = meDeciderNormalize(hugeVec, mu: w.muSIMD, scale: w.scaleSIMD, normShift: w.normShiftSIMD)
        var j = 0
        while j < 8 {
            #expect(normHuge[j] <= 127)
            #expect(-127 <= normHuge[j])
            j += 1
        }
    }
}

