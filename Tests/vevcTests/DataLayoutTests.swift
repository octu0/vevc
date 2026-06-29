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
}
