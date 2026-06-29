import Testing
@testable import vevc

@Suite("MotionEstimation Tests")
struct METests {

    @Test("getMVDPenalty calculates zero penalty when MVD is zero")
    func testGetMVDPenaltyZeroMVD() {
        let penalty: Int = MotionEstimation.getMVDPenalty(dx: 5, dy: -3, pmvDx: 5, pmvDy: -3, lambda: 10)
        #expect(penalty == 0)
    }

    @Test("getMVDPenalty calculates correctly with positive MVD")
    func testGetMVDPenaltyPositiveMVD() {
        let penalty: Int = MotionEstimation.getMVDPenalty(dx: 10, dy: 8, pmvDx: 5, pmvDy: 4, lambda: 2)
        // mvdX = abs(10 - 5) = 5
        // mvdY = abs(8 - 4) = 4
        // mvdCost = 5 + 4 = 9
        // penalty = 9 * 2 = 18
        #expect(penalty == 18)
    }

    @Test("getMVDPenalty calculates correctly with negative MVD")
    func testGetMVDPenaltyNegativeMVD() {
        let penalty: Int = MotionEstimation.getMVDPenalty(dx: -5, dy: -10, pmvDx: 2, pmvDy: -2, lambda: 3)
        // mvdX = abs(-5 - 2) = 7
        // mvdY = abs(-10 - (-2)) = abs(-8) = 8
        // mvdCost = 7 + 8 = 15
        // penalty = 15 * 3 = 45
        #expect(penalty == 45)
    }

    @Test("getMVDPenalty calculates correctly with asymmetric differences")
    func testGetMVDPenaltyAsymmetricMVD() {
        let penalty: Int = MotionEstimation.getMVDPenalty(dx: 0, dy: 15, pmvDx: 5, pmvDy: 15, lambda: 4)
        // mvdX = abs(0 - 5) = 5
        // mvdY = abs(15 - 15) = 0
        // mvdCost = 5 + 0 = 5
        // penalty = 5 * 4 = 20
        #expect(penalty == 20)
    }

    @Test("getMVDPenalty returns zero when lambda is zero")
    func testGetMVDPenaltyZeroLambda() {
        let penalty: Int = MotionEstimation.getMVDPenalty(dx: 100, dy: -50, pmvDx: 0, pmvDy: 0, lambda: 0)
        #expect(penalty == 0)
    }
}
