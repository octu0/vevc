import XCTest
@testable import vevc

final class DataLayoutTests: XCTestCase {

    func testDeserializeInvalidMagicThrowsError() {
        let chunk: [UInt8] = [0x00, 0x00, 0x00, 0x00, 0x00]
        var offset: Int = 0

        do {
            let _: VEVCFileHeader = try VEVCFileHeader.deserialize(from: chunk, offset: &offset)
            XCTFail("Should have thrown an error")
        } catch let error as DecodeError {
            if case .insufficientDataContext(let msg) = error {
                XCTAssertEqual(msg, "VEVC Magic NotFound")
            } else {
                XCTFail("Expected insufficientDataContext with VEVC Magic NotFound, got \(error)")
            }
        } catch {
            XCTFail("Expected DecodeError, got \(error)")
        }
    }
}
