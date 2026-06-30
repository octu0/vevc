import XCTest
@testable import vevc

final class SplitterTests: XCTestCase {

    func testSplitVEVCStreamInvalidMagic() {
        let input: [UInt8] = [0x00, 0x00, 0x00, 0x00, 0x00]
        let maxLayer: Int = 2

        XCTAssertThrowsError(try splitVEVCStream(input: input, maxLayer: maxLayer)) { (error: Error) in
            guard let splitterError: SplitterError = error as? SplitterError else {
                XCTFail("Expected SplitterError, but got \(error)")
                return
            }
            guard case .invalidMagic = splitterError else {
                XCTFail("Expected .invalidMagic, but got \(splitterError)")
                return
            }
        }
    }
}
