import XCTest
@testable import vevc

final class ChromaSADCoordinateTests: XCTestCase {
    
    func testChromaSADRoundingNegative() throws {
        let width = 16
        let height = 16
        let cbw = 8
        let cbh = 8
        
        var currCb = [Int16](repeating: 0, count: cbw * cbh)
        var currCr = [Int16](repeating: 0, count: cbw * cbh)
        var refCb = [Int16](repeating: 0, count: cbw * cbh)
        var refCr = [Int16](repeating: 0, count: cbw * cbh)
        
        // Layer1 (half-res Luma) pixel coordinates. Matches Chroma coordinates.
        let bx = 2
        let by = 2
        // Chroma block coordinate: cx = 2, cy = 2
        
        // Curr block values
        for y in 0..<4 {
            for x in 0..<4 {
                let offset = (2 + y) * cbw + (2 + x)
                currCb[offset] = 100
                currCr[offset] = 100
            }
        }
        
        // Ref block values: 
        // Luma Motion Vector (-3, -3)
        // Chroma MV should be (-3 >> 1, -3 >> 1) = (-2, -2)
        // Therefore, it should match Chroma block at cx - 2 = 0, cy - 2 = 0.
        for y in 0..<4 {
            for x in 0..<4 {
                let offset = y * cbw + x
                refCb[offset] = 100
                refCr[offset] = 100
            }
        }
        
        let currPlane = PlaneData420(width: width, height: height, y: [Int16](repeating: 0, count: width * height), cb: currCb, cr: currCr)
        let refPlane = PlaneData420(width: width, height: height, y: [Int16](repeating: 0, count: width * height), cb: refCb, cr: refCr)
        
        // If bug exists (-3 / 2 == -1), it will look at (cx - 1, cy - 1) = (1, 1) and will result in SAD > 0
        let sad = MotionEstimation.computeChromaSAD(curr: currPlane, ref: refPlane, bx: bx, by: by, refDx: -3, refDy: -3)
        
        // Should perfectly match, so SAD == 0
        XCTAssertEqual(sad, 0, "Chroma SAD rounding should use >> 1 to match Luma negative vector accurately")
    }
    
    func testChromaSADRoundingPositive() throws {
        let width = 16
        let height = 16
        let cbw = 8
        let cbh = 8
        
        var currCb = [Int16](repeating: 0, count: cbw * cbh)
        var currCr = [Int16](repeating: 0, count: cbw * cbh)
        var refCb = [Int16](repeating: 0, count: cbw * cbh)
        var refCr = [Int16](repeating: 0, count: cbw * cbh)
        
        let bx = 2
        let by = 2
        // Chroma block coordinate: cx = 2, cy = 2
        
        for y in 0..<4 {
            for x in 0..<4 {
                let offset = (2 + y) * cbw + (2 + x)
                currCb[offset] = 100
                currCr[offset] = 100
            }
        }
        
        // Luma MV (3, 3) => Chroma MV (3 >> 1) = 1
        // Expected ref coords: cx + 1 = 3, cy + 1 = 3
        for y in 0..<4 {
            for x in 0..<4 {
                let offset = (3 + y) * cbw + (3 + x)
                refCb[offset] = 100
                refCr[offset] = 100
            }
        }
        
        let currPlane = PlaneData420(width: width, height: height, y: [Int16](repeating: 0, count: width * height), cb: currCb, cr: currCr)
        let refPlane = PlaneData420(width: width, height: height, y: [Int16](repeating: 0, count: width * height), cb: refCb, cr: refCr)
        
        let sad = MotionEstimation.computeChromaSAD(curr: currPlane, ref: refPlane, bx: bx, by: by, refDx: 3, refDy: 3)
        
        XCTAssertEqual(sad, 0, "Chroma SAD rounding should properly shift positive vectors")
    }
}
