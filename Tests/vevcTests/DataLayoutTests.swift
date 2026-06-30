import XCTest
@testable import vevc

final class DataLayoutTests: XCTestCase {

    func testDeserializeVEVCFileHeaderInvalidMagic() {
        // Serialize a valid VEVC file header
        let expectedWidth = 1920
        let expectedHeight = 1080
        let expectedFramerate = 30
        var chunk = VEVCFileHeader(width: expectedWidth, height: expectedHeight, framerate: expectedFramerate).serialize()

        // Mutate the first byte of the magic number to an invalid value
        chunk[0] = 0x00

        var offset = 0
        XCTAssertThrowsError(try VEVCFileHeader.deserialize(from: chunk, offset: &offset)) { error in
            guard let decodeError = error as? DecodeError else {
                XCTFail("Expected DecodeError, but got \(type(of: error))")
                return
            }
            if case .insufficientDataContext(let message) = decodeError {
                XCTAssertEqual(message, "VEVC Magic NotFound")
            } else {
                XCTFail("Expected insufficientDataContext error, but got \(decodeError)")
            }
        }
    }

    func testDeriveMVCountEdgeCases() {
        let expectedZero: Int = 0
        let expectedOne: Int = 1
        let expectedTwo: Int = 2
        let expectedFour: Int = 4
        let expected1080p: Int = 2040
        let expectedOdd: Int = 2074

        // Zero dimensions
        XCTAssertEqual(deriveMVCount(width: 0, height: 0), expectedZero)

        // Single pixel
        XCTAssertEqual(deriveMVCount(width: 1, height: 1), expectedOne)

        // Exact block boundaries
        // 32x32 -> l1: 16x16 -> l0: 8x8 -> cols: 1, rows: 1 -> 1
        XCTAssertEqual(deriveMVCount(width: 32, height: 32), expectedOne)

        // Just over block boundary
        // 33x33 -> l1: 17x17 -> l0: 9x9 -> cols: 2, rows: 2 -> 4
        XCTAssertEqual(deriveMVCount(width: 33, height: 33), expectedFour)

        // Common resolutions
        // 1920x1080 -> l1: 960x540 -> l0: 480x270 -> cols: 60, rows: 34 -> 2040
        XCTAssertEqual(deriveMVCount(width: 1920, height: 1080), expected1080p)

        // Odd resolutions
        // 1921x1081 -> l1: 961x541 -> l0: 481x271 -> cols: 61, rows: 34 -> 2074
        XCTAssertEqual(deriveMVCount(width: 1921, height: 1081), expectedOdd)

        // Asymmetric dimensions
        XCTAssertEqual(deriveMVCount(width: 32, height: 33), expectedTwo)
        XCTAssertEqual(deriveMVCount(width: 33, height: 32), expectedTwo)

        // Negative inputs
        // -10 + 1 = -9 -> / 2 = -4 -> + 1 = -3 -> / 2 = -1 -> + 7 = 6 -> / 8 = 0
        XCTAssertEqual(deriveMVCount(width: -10, height: -10), expectedZero)
    }
}
