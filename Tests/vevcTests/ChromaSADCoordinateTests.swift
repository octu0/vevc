import XCTest
@testable import vevc

/// Pins the computeChromaSAD coordinate contract: bx/by are quarter-resolution
/// block coordinates (the chroma block region is 16×16 at 2bx, 2by on the
/// half-resolution chroma planes), refDx/refDy are the full-resolution-pixel
/// motion vector, and the chroma displacement is refDx >> 1 (arithmetic shift,
/// so negative vectors round toward −∞ exactly like the shift MC applies).
/// Sampling is a 4×4 grid strided by 4 across the whole 16×16 region.
final class ChromaSADCoordinateTests: XCTestCase {

    private func makePlanes(fillCurrAt: (Int, Int), fillRefAt: (Int, Int)) -> (PlaneData420, PlaneData420) {
        let width = 64
        let height = 64
        let cbw = 32
        let cbh = 32

        var currCb = [Int16](repeating: 0, count: cbw * cbh)
        var currCr = [Int16](repeating: 0, count: cbw * cbh)
        var refCb = [Int16](repeating: 0, count: cbw * cbh)
        var refCr = [Int16](repeating: 0, count: cbw * cbh)

        for y in 0..<16 {
            for x in 0..<16 {
                let cOffset = (fillCurrAt.1 + y) * cbw + (fillCurrAt.0 + x)
                currCb[cOffset] = 100
                currCr[cOffset] = 100
                let rOffset = (fillRefAt.1 + y) * cbw + (fillRefAt.0 + x)
                refCb[rOffset] = 100
                refCr[rOffset] = 100
            }
        }

        let zeroY = [Int16](repeating: 0, count: width * height)
        let curr = PlaneData420(width: width, height: height, y: zeroY, cb: currCb, cr: currCr)
        let ref = PlaneData420(width: width, height: height, y: zeroY, cb: refCb, cr: refCr)
        return (curr, ref)
    }

    func testChromaSADRoundingNegative() throws {
        // Quarter-res block (1,1) → chroma region at (2,2).
        // Full-res MV (-3,-3) → chroma displacement (-3 >> 1) = -2 → the
        // matching reference region sits at (0,0).
        let (curr, ref) = makePlanes(fillCurrAt: (2, 2), fillRefAt: (0, 0))
        let sad = MotionEstimation.computeChromaSAD(curr: curr, ref: ref, bx: 1, by: 1, refDx: -3, refDy: -3)
        XCTAssertEqual(sad, 0, "Chroma SAD rounding should use >> 1 to match Luma negative vector accurately")
    }

    func testChromaSADRoundingPositive() throws {
        // Quarter-res block (1,1) → chroma region at (2,2).
        // Full-res MV (3,3) → chroma displacement (3 >> 1) = 1 → the matching
        // reference region sits at (3,3).
        let (curr, ref) = makePlanes(fillCurrAt: (2, 2), fillRefAt: (3, 3))
        let sad = MotionEstimation.computeChromaSAD(curr: curr, ref: ref, bx: 1, by: 1, refDx: 3, refDy: 3)
        XCTAssertEqual(sad, 0, "Chroma SAD rounding should use >> 1 to match Luma positive vector accurately")
    }

    func testChromaSADDetectsMismatchAwayFromCorner() throws {
        // Coverage regression guard: a mismatch confined to the bottom-right
        // of the 16×16 chroma region must be visible to the strided sampling
        // (the old top-left-4×4-only sampling missed it entirely).
        let (curr, ref) = makePlanes(fillCurrAt: (2, 2), fillRefAt: (2, 2))
        var refCb = ref.cb
        for y in 10..<18 {
            for x in 10..<18 {
                refCb[y * 32 + x] = 0
            }
        }
        let broken = PlaneData420(width: ref.width, height: ref.height, y: ref.y, cb: refCb, cr: ref.cr)
        let sad = MotionEstimation.computeChromaSAD(curr: curr, ref: broken, bx: 1, by: 1, refDx: 0, refDy: 0)
        XCTAssertGreaterThan(sad, 0, "strided sampling must see mismatches outside the top-left corner")
    }
}
