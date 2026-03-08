import XCTest
@testable import vevc

final class GMCTests: XCTestCase {
    func createPatternPlaneData(width: Int, height: Int) -> PlaneData420 {
        var y = [Int16](repeating: 0, count: width * height)
        for j in 0..<height {
            for i in 0..<width {
                // Large-scale pattern to survive downscaling and sparse sampling
                let v = ((i / 16) + (j / 16)) % 2 == 0 ? Int16(64) : Int16(-64)
                y[j * width + i] = v
            }
        }
        let cw = (width + 1) / 2
        let ch = (height + 1) / 2
        let cb = [Int16](repeating: 0, count: cw * ch)
        let cr = [Int16](repeating: 0, count: cw * ch)
        return PlaneData420(width: width, height: height, y: y, cb: cb, cr: cr)
    }

    func shiftPlaneNonCircular(_ plane: PlaneData420, dx: Int, dy: Int) -> PlaneData420 {
        var newY = [Int16](repeating: 0, count: plane.width * plane.height)
        for y in 0..<plane.height {
            let srcY = y - dy
            if srcY < 0 || srcY >= plane.height { continue }
            for x in 0..<plane.width {
                let srcX = x - dx
                if srcX < 0 || srcX >= plane.width { continue }
                newY[y * plane.width + x] = plane.y[srcY * plane.width + srcX]
            }
        }
        let cw = (plane.width + 1) / 2
        let ch = (plane.height + 1) / 2
        let cb = [Int16](repeating: 0, count: cw * ch)
        let cr = [Int16](repeating: 0, count: cw * ch)
        return PlaneData420(width: plane.width, height: plane.height, y: newY, cb: cb, cr: cr)
    }

    func testEstimateGMV_ZeroMotion() {
        let width = 512
        let height = 512
        let pd = createPatternPlaneData(width: width, height: height)

        let (dx, dy) = estimateGMV(curr: pd, prev: pd)

        XCTAssertEqual(dx, 0)
        XCTAssertEqual(dy, 0)
    }

    func testEstimateGMV_Translation_MultipleOf8() {
        let width = 512
        let height = 512
        let pdPrev = createPatternPlaneData(width: width, height: height)

        // Coarse search range: +- 32 pixels in full res (+- 4 in 1/8 scale)
        let testCases = [
            (dx: 8, dy: 0),
            (dx: 0, dy: 8)
        ]

        for tc in testCases {
            let pdCurr = shiftPlaneNonCircular(pdPrev, dx: tc.dx, dy: tc.dy)
            let (dx, dy) = estimateGMV(curr: pdCurr, prev: pdPrev)

            XCTAssertEqual(dx, tc.dx, "Failed for expected dx: \(tc.dx)")
            XCTAssertEqual(dy, tc.dy, "Failed for expected dy: \(tc.dy)")
        }
    }

    func testCalculateSAD() {
        let p1: [Int16] = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        let p2: [Int16] = [11, 19, 32, 38, 55, 55, 77, 72, 90]

        let sad = p1.withUnsafeBufferPointer { ptr1 in
            p2.withUnsafeBufferPointer { ptr2 in
                calculateSAD(p1: ptr1.baseAddress!, p2: ptr2.baseAddress!, count: 9)
            }
        }

        XCTAssertEqual(sad, 31)
    }
}
