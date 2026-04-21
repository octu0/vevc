import XCTest
@testable import vevc

final class EncodePlaneTests: XCTestCase {
    
    func testToPlaneData420Downsampling() {
        // Create a 4x4 image
        let width = 4
        let height = 4
        var img = YCbCrImage(width: width, height: height, ratio: .ratio444)
        
        // Fill Cb and Cr with identifiable values
        for y in 0..<height {
            for x in 0..<width {
                let offset = y * width + x
                // yPlane will just be zero
                img.yPlane[offset] = 128
                // Cb pattern, e.g., 0, 1, 2, 3...
                img.cbPlane[offset] = UInt8(128 + offset)
                // Cr pattern
                img.crPlane[offset] = UInt8(128 + offset * 2)
            }
        }
        
        let pool = BlockViewPool()
        let (plane, _) = toPlaneData420(image: img, pool: pool)
        
        XCTAssertEqual(plane.width, 4)
        XCTAssertEqual(plane.height, 4)
        
        let expectedCWidth = 2
        let expectedCHeight = 2
        XCTAssertEqual(plane.cb.count, expectedCWidth * expectedCHeight)
        XCTAssertEqual(plane.cr.count, expectedCWidth * expectedCHeight)
        
        // Verify downsampling picks the top-left pixel of each 2x2 block
        // Block (0,0) -> img offset 0
        XCTAssertEqual(plane.cb[0], Int16(img.cbPlane[0]) - 128)
        XCTAssertEqual(plane.cr[0], Int16(img.crPlane[0]) - 128)
        
        // Block (1,0) -> img offset 2 (x=2, y=0)
        XCTAssertEqual(plane.cb[1], Int16(img.cbPlane[2]) - 128)
        XCTAssertEqual(plane.cr[1], Int16(img.crPlane[2]) - 128)
        
        // Block (0,1) -> img offset 8 (x=0, y=2)
        XCTAssertEqual(plane.cb[2], Int16(img.cbPlane[8]) - 128)
        XCTAssertEqual(plane.cr[2], Int16(img.crPlane[8]) - 128)
        
        // Block (1,1) -> img offset 10 (x=2, y=2)
        XCTAssertEqual(plane.cb[3], Int16(img.cbPlane[10]) - 128)
        XCTAssertEqual(plane.cr[3], Int16(img.crPlane[10]) - 128)
    }

    func testToPlaneData420Ratio420() {
        // Create a 4x4 image with .ratio420
        let width = 4
        let height = 4
        var img = YCbCrImage(width: width, height: height, ratio: .ratio420)

        let cWidth = (width / 2)
        let cHeight = (height / 2)

        for i in 0..<img.yPlane.count { img.yPlane[i] = UInt8(128 + i) }
        for i in 0..<img.cbPlane.count { img.cbPlane[i] = UInt8(128 + i) }
        for i in 0..<img.crPlane.count { img.crPlane[i] = UInt8(128 + i) }

        let pool = BlockViewPool()
        let (plane, _) = toPlaneData420(image: img, pool: pool)

        XCTAssertEqual(plane.y.count, width * height)
        XCTAssertEqual(plane.cb.count, cWidth * cHeight)
        XCTAssertEqual(plane.cr.count, cWidth * cHeight)

        for i in 0..<plane.y.count {
            XCTAssertEqual(plane.y[i], Int16(img.yPlane[i]) - 128)
        }
        for i in 0..<plane.cb.count {
            XCTAssertEqual(plane.cb[i], Int16(img.cbPlane[i]) - 128)
        }
        for i in 0..<plane.cr.count {
            XCTAssertEqual(plane.cr[i], Int16(img.crPlane[i]) - 128)
        }
    }

    func testToPlaneData420OddSize() {
        // Create a 3x3 image with .ratio444
        let width = 3
        let height = 3
        var img = YCbCrImage(width: width, height: height, ratio: .ratio444)

        for i in 0..<img.yPlane.count { img.yPlane[i] = UInt8(128 + i) }
        for i in 0..<img.cbPlane.count { img.cbPlane[i] = UInt8(128 + i) }
        for i in 0..<img.crPlane.count { img.crPlane[i] = UInt8(128 + i) }

        let pool = BlockViewPool()
        let (plane, _) = toPlaneData420(image: img, pool: pool)

        // Chroma dimensions for 3x3 are 2x2
        XCTAssertEqual(plane.width, 3)
        XCTAssertEqual(plane.height, 3)
        XCTAssertEqual(plane.cb.count, 4)

        // Check mapping (cy*2, cx*2)
        // cy=0, cx=0 -> (0,0) srcOffset=0
        XCTAssertEqual(plane.cb[0], Int16(img.cbPlane[0]) - 128)
        // cy=0, cx=1 -> (0,2) srcOffset=2
        XCTAssertEqual(plane.cb[1], Int16(img.cbPlane[2]) - 128)
        // cy=1, cx=0 -> (2,0) srcOffset=6
        XCTAssertEqual(plane.cb[2], Int16(img.cbPlane[6]) - 128)
        // cy=1, cx=1 -> (2,2) srcOffset=8
        XCTAssertEqual(plane.cb[3], Int16(img.cbPlane[8]) - 128)
    }
}
