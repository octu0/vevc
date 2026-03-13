import XCTest
@testable import vevc

final class CABACTests: XCTestCase {
    func testCABACRoundTrip() throws {
        let inputBits: [UInt8] = [0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1]

        var encoder = CABACEncoder()
        var ctx = ContextModel()

        for bit in inputBits {
            encoder.encodeBin(binVal: bit, ctx: &ctx)
        }
        encoder.flush()

        let encodedData = encoder.getData()
        XCTAssertFalse(encodedData.isEmpty)

        var decoder = try CABACDecoder(data: encodedData)
        var decCtx = ContextModel()

        var decodedBits: [UInt8] = []
        for _ in 0..<inputBits.count {
            let bit = try decoder.decodeBin(ctx: &decCtx)
            decodedBits.append(bit)
        }

        XCTAssertEqual(inputBits, decodedBits)
    }
}
