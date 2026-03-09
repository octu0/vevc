import XCTest
@testable import vevc

final class ImageUtilTests: XCTestCase {
    
    func testRGBAtoYCbCr444() {
        // Create 4x4 RGBA data with specific colors
        // Red, Green, Blue, White, Black, Grey
        let width = 4
        let height = 4
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        
        // (0,0) Red
        rgba[0] = 255; rgba[1] = 0; rgba[2] = 0; rgba[3] = 255
        // (1,0) Green
        rgba[4] = 0; rgba[5] = 255; rgba[6] = 0; rgba[7] = 255
        // (2,0) Blue
        rgba[8] = 0; rgba[9] = 0; rgba[10] = 255; rgba[11] = 255
        // (3,0) White
        rgba[12] = 255; rgba[13] = 255; rgba[14] = 255; rgba[15] = 255
        
        let ycbcr = rgbaToYCbCr(data: rgba, width: width, height: height)
        
        XCTAssertEqual(ycbcr.ratio, .ratio444)
        
        // Red pixel (255, 0, 0) -> Y:76, Cb:85, Cr:255 (approx)
        XCTAssertEqual(Double(ycbcr.yPlane[0]), 76, accuracy: 2)
        XCTAssertEqual(Double(ycbcr.cbPlane[0]), 85, accuracy: 2)
        XCTAssertEqual(Double(ycbcr.crPlane[0]), 255, accuracy: 2)
    }
    
    func testYCbCr444toRGBA() {
        let width = 4
        let height = 4
        var ycbcr = YCbCrImage(width: width, height: height, ratio: .ratio444)
        
        // Fill with neutral grey (Y=128, Cb=128, Cr=128)
        for i in 0..<(width * height) {
            ycbcr.yPlane[i] = 128
            ycbcr.cbPlane[i] = 128
            ycbcr.crPlane[i] = 128
        }
        
        let rgba = ycbcrToRGBA(img: ycbcr)
        
        XCTAssertEqual(rgba.count, width * height * 4)
        
        for i in 0..<(width * height) {
            let offset = i * 4
            // (128, 128, 128) should result in approx (128, 128, 128)
            XCTAssertEqual(Double(rgba[offset + 0]), 128, accuracy: 5)
            XCTAssertEqual(Double(rgba[offset + 1]), 128, accuracy: 5)
            XCTAssertEqual(Double(rgba[offset + 2]), 128, accuracy: 5)
            XCTAssertEqual(rgba[offset + 3], 255) // Alpha should be 255
        }
    }
    
    func testYCbCr420toRGBA() {
        let width = 4
        let height = 4
        var ycbcr = YCbCrImage(width: width, height: height, ratio: .ratio420)
        
        // Fill luminance with gradient
        for i in 0..<(width * height) {
            ycbcr.yPlane[i] = UInt8(i * 10)
        }
        
        // Fill chroma with something distinct
        // Chroma size for 4x4 4:2:0 is 2x2
        for i in 0..<4 {
            ycbcr.cbPlane[i] = 100
            ycbcr.crPlane[i] = 200
        }
        
        let rgba = ycbcrToRGBA(img: ycbcr)
        
        XCTAssertEqual(rgba.count, width * height * 4)
        
        // Check if alpha is still 255
        for i in 0..<(width * height) {
            XCTAssertEqual(rgba[i * 4 + 3], 255)
        }
    }
    
    func testRoundTrip() {
        let width = 8
        let height = 8
        var originalRGBA = [UInt8](repeating: 0, count: width * height * 4)
        
        for i in 0..<(width * height) {
            let offset = i * 4
            originalRGBA[offset + 0] = UInt8((i * 7) % 256) // R
            originalRGBA[offset + 1] = UInt8((i * 13) % 256) // G
            originalRGBA[offset + 2] = UInt8((i * 17) % 256) // B
            originalRGBA[offset + 3] = 255 // A
        }
        
        let ycbcr = rgbaToYCbCr(data: originalRGBA, width: width, height: height)
        let convertedRGBA = ycbcrToRGBA(img: ycbcr)
        
        for i in 0..<(width * height) {
            let offset = i * 4
            XCTAssertEqual(Double(convertedRGBA[offset + 0]), Double(originalRGBA[offset + 0]), accuracy: 10)
            XCTAssertEqual(Double(convertedRGBA[offset + 1]), Double(originalRGBA[offset + 1]), accuracy: 10)
            XCTAssertEqual(Double(convertedRGBA[offset + 2]), Double(originalRGBA[offset + 2]), accuracy: 10)
            XCTAssertEqual(convertedRGBA[offset + 3], 255)
        }
    }
}
