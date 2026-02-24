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
        
        // Convert
        let planes = toPlaneData420(images: [img])
        XCTAssertEqual(planes.count, 1)
        let plane = planes[0]
        
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
}
