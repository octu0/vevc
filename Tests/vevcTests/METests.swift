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
}
