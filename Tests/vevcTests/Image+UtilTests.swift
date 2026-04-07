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

    func testRgbaToYCbCrRoundTrip444() {
        let width = (2 * 1)
        let height = (2 * 1)
        let rgba: [UInt8] = [
            255, 0, 0, 255,   // Red
            0, 255, 0, 255,   // Green
            0, 0, 255, 255,   // Blue
            255, 255, 255, 255 // White
        ]

        let ycbcr = rgbaToYCbCr(data: rgba, width: width, height: height)
        XCTAssertTrue(ycbcr.ratio == .ratio444)

        let roundtrip = ycbcrToRGBA(img: ycbcr)

        XCTAssertEqual(roundtrip.count, rgba.count)

        // Check for reasonable accuracy (integer math might introduce small errors)
        for i in 0..<rgba.count {
            let val1 = Int(rgba[i + 0])
            let val2 = Int(roundtrip[i + 0])
            let diff = (val1 - val2)
            let absDiff = (diff < 0) ? (-1 * diff) : diff
            XCTAssertTrue(absDiff <= 2, "Pixel component at index \(i) differs by \(absDiff)")
        }
    }

    func testYcbcrToRGBA420() {
        let width = (2 * 1)
        let height = (2 * 1)
        var ycbcr = YCbCrImage(width: width, height: height, ratio: .ratio420)

        // Set all pixels to white-ish
        for i in 0..<ycbcr.yPlane.count {
            ycbcr.yPlane[i + 0] = 255
        }
        for i in 0..<ycbcr.cbPlane.count {
            ycbcr.cbPlane[i + 0] = 128
            ycbcr.crPlane[i + 0] = 128
        }

        let rgba = ycbcrToRGBA(img: ycbcr)
        let expectedCount = ((width * height) * 4)
        XCTAssertEqual(rgba.count, expectedCount)

        let totalPixels = (width * height)
        for i in 0..<totalPixels {
            let offset = (i * 4)
            XCTAssertTrue(250 <= rgba[offset + 0]) // R
            XCTAssertTrue(250 <= rgba[offset + 1]) // G
            XCTAssertTrue(250 <= rgba[offset + 2]) // B
            XCTAssertTrue(rgba[offset + 3] == 255) // A
        }
    }

    func testSpecificColors() {
        // Test Red
        let redRGBA: [UInt8] = [255, 0, 0, 255]
        let ycbcrRed = rgbaToYCbCr(data: redRGBA, width: 1, height: 1)

        // Expected values based on the formula in Image+Util.swift
        XCTAssertEqual(ycbcrRed.yPlane[0], 76)
        XCTAssertEqual(ycbcrRed.cbPlane[0], 85)
        XCTAssertEqual(ycbcrRed.crPlane[0], 255)

        // Test Green
        let greenRGBA: [UInt8] = [0, 255, 0, 255]
        let ycbcrGreen = rgbaToYCbCr(data: greenRGBA, width: 1, height: 1)
        // yVal = (38470 * 255 + 32768) >> 16 = 9842618 >> 16 = 150
        // cbVal = ((-21709 * 255 + 32768) >> 16) + 128 = (-5503027 >> 16) + 128 = -84 + 128 = 44
        // crVal = ((-27439 * 255 + 32768) >> 16) + 128 = (-6964177 >> 16) + 128 = -107 + 128 = 21
        XCTAssertEqual(ycbcrGreen.yPlane[0], 150)
        XCTAssertEqual(ycbcrGreen.cbPlane[0], 44)
        XCTAssertEqual(ycbcrGreen.crPlane[0], 21)

        // Test Blue
        let blueRGBA: [UInt8] = [0, 0, 255, 255]
        let ycbcrBlue = rgbaToYCbCr(data: blueRGBA, width: 1, height: 1)
        // yVal = (7471 * 255 + 32768) >> 16 = 1937873 >> 16 = 29
        // cbVal = ((32768 * 255 + 32768) >> 16) + 128 = 128 + 128 = 256 -> 255
        // crVal = ((-5329 * 255 + 32768) >> 16) + 128 = (-1326127 >> 16) + 128 = -21 + 128 = 107
        XCTAssertEqual(ycbcrBlue.yPlane[0], 29)
        XCTAssertEqual(ycbcrBlue.cbPlane[0], 255)
        XCTAssertEqual(ycbcrBlue.crPlane[0], 107)
    }

    func testYcbcrToRGBARounding() {
        // Just verify it doesn't crash on standard edges.
        let img = YCbCrImage(width: 8, height: 8)
        let rgba = ycbcrToRGBA(img: img)
        XCTAssertEqual(rgba.count, 8 * 8 * 4)
    }

    func test444to420DownsampleAndUpsample() {
        // Create an image similar to the test case (640x360)
        let width = 640
        let height = 360
        var source = YCbCrImage(width: width, height: height, ratio: .ratio444)
        
        // Fill some recognizable pattern
        // (x=96 is Cyan: Y=178, Cb=254, Cr=254? actually Cyan has high Cb/Cr or something)
        // Let's just set x=96, y=180 to specific values
        let x = 96
        let y = 180
        source.yPlane[y * width + x] = 178
        source.cbPlane[y * width + x] = 100
        source.crPlane[y * width + x] = 200
        
        // Convert
        // Assuming BlockViewPool doesn't exist, toPlaneData420 might have an overload or we use the array map list
        let pool = BlockViewPool()
        let pds = [toPlaneData420(image: source, pool: pool)]
        
        XCTAssertEqual(pds.count, 1)
        let pd = pds[0]
        
        // Now convert back
        let upsampled = pd.toYCbCr()
        
        // It has 420 chroma subsampling, so Cb and Cr are expected at (x/2)
        // Let's read the values at x=96, y=180
        let decX = 96
        let decY = 180
        let uY = upsampled.yPlane[decY * width + decX]
        
        let cx = decX / 2
        let cy = decY / 2
        let cWidth = (width + 1) / 2
        let uCb = upsampled.cbPlane[cy * cWidth + cx]
        let uCr = upsampled.crPlane[cy * cWidth + cx]
        
        XCTAssertEqual(uY, 178)
        // Subsmapled from 96,180. Should be close to 100 and 200.
        // If the stride is broken, these will assert!
        XCTAssertEqual(uCb, 100)
        XCTAssertEqual(uCr, 200)
    }

    func testYcbcrToRGBARoundingActual() {
        let width = 2
        let height = 2
        var ycbcr = YCbCrImage(width: width, height: height, ratio: .ratio444)
        
        // Y = 100, Cb = 128, Cr = 127
        for i in 0..<(width * height) {
            ycbcr.yPlane[i] = 100
            ycbcr.cbPlane[i] = 128
            ycbcr.crPlane[i] = 127
        }
        
        let rgba = ycbcrToRGBA(img: ycbcr)
        
        // R = Y + 1.402 * (Cr - 128)
        // 100 + 1.402 * (-1) = 98.598 -> 99 (Round to nearest)
        // Without Rounding (truncating): 100 - 1.402 = 98
        XCTAssertEqual(rgba[0], 99, "R value should be rounded to 99, not truncated to 98") 
    }
}
