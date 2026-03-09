import XCTest
@testable import vevc

final class ImageUtilTests: XCTestCase {
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
}
