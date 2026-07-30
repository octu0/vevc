import XCTest
@testable import vevc

final class FrameRateConverterTests: XCTestCase {

    func test60To30Conversion() {
        var converter = FrameRateConverter(inFps: 60, outFps: 30)
        var sequence = [Int]()
        for _ in 0..<20 {
            sequence.append(converter.repeatCount())
        }
        let expected = [
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1
        ]
        XCTAssertEqual(sequence, expected)
    }

    func test24To30Conversion() {
        var converter = FrameRateConverter(inFps: 24, outFps: 30)
        var sequence = [Int]()
        for _ in 0..<20 {
            sequence.append(converter.repeatCount())
        }
        let expected = [
            1, 1, 1, 2, 1, 1, 1, 2, 1, 1,
            1, 2, 1, 1, 1, 2, 1, 1, 1, 2
        ]
        XCTAssertEqual(sequence, expected)
    }

    func testIdentityConversion() {
        var converter = FrameRateConverter(inFps: 30, outFps: 30)
        var sequence = [Int]()
        for _ in 0..<20 {
            sequence.append(converter.repeatCount())
        }
        let expected = [Int](repeating: 1, count: 20)
        XCTAssertEqual(sequence, expected)
    }

    func testFrameCountIdentity() {
        // miko1.y4m (60fps) -> 30fps
        var converter = FrameRateConverter(inFps: 60, outFps: 30)
        var count = 0
        for _ in 0..<1801 {
            count += converter.repeatCount()
        }
        XCTAssertEqual(count, 900)

        // miko1.y4m (24fps) -> 30fps
        var converter2 = FrameRateConverter(inFps: 24, outFps: 30)
        var count2 = 0
        for _ in 0..<1801 {
            count2 += converter2.repeatCount()
        }
        // 1801 * 30 / 24 = 2251.25 -> 2251
        XCTAssertEqual(count2, 2251)
    }
}
