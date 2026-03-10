import XCTest
@testable import vevc

final class CABACTests: XCTestCase {
    func testContextModelUpdate() {
        var context = ContextModel()
        XCTAssertEqual(context.mps, 0)

        // Update with MPS
        context.update(bit: 0)
        XCTAssertEqual(context.mps, 0)
        XCTAssertLessThan(context.prob, 16384)

        // Update with LPS multiple times to switch MPS
        for _ in 0..<100 {
            context.update(bit: 1)
        }
        XCTAssertEqual(context.mps, 1)
    }

    func testArithmeticCoding() {
        var encoder = CABACEncoder()
        var context = ContextModel()

        let bits = [0, 0, 1, 0, 1, 1, 0, 0, 0, 1]
        for b in bits {
            encoder.encode(bit: b, context: &context)
        }
        encoder.flush()

        var decoder = CABACDecoder(data: encoder.data)
        var decodeContext = ContextModel()
        for i in 0..<bits.count {
            let b = decoder.decode(context: &decodeContext)
            XCTAssertEqual(b, bits[i], "Bit at index \(i) mismatch")
        }
    }

    func testBypassCoding() {
        var encoder = CABACEncoder()
        let bits = [1, 0, 1, 1, 0, 0, 1]
        for b in bits {
            encoder.encodeBypass(bit: b)
        }
        encoder.flush()

        var decoder = CABACDecoder(data: encoder.data)
        for i in 0..<bits.count {
            let b = decoder.decodeBypass()
            XCTAssertEqual(b, bits[i], "Bypass bit at index \(i) mismatch")
        }
    }

    func testCABACSimple() {
        var encoder = CABACEncoder()
        var ctxG1 = [ContextModel](repeating: ContextModel(), count: 3)

        let values: [Int] = [0, 1, 2, 3, 10, 100, 1000, 0, 1]
        for v in values {
            encoder.encodeVal(v, ctxG1: &ctxG1)
        }
        encoder.flush()

        let encodedData = encoder.data
        var decoder = CABACDecoder(data: encodedData)
        var dCtxG1 = [ContextModel](repeating: ContextModel(), count: 3)

        for i in 0..<values.count {
            let decoded = decoder.decodeVal(ctxG1: &dCtxG1)
            XCTAssertEqual(decoded, values[i], "Value at index \(i) mismatch")
        }
    }

    func testCABACLong() {
        var encoder = CABACEncoder()
        var ctxG1 = [ContextModel](repeating: ContextModel(), count: 5)

        var values: [Int] = []
        for i in 0..<1000 {
            let v = i % 100
            values.append(v)
            encoder.encodeVal(v, ctxG1: &ctxG1)
        }
        encoder.flush()

        var decoder = CABACDecoder(data: encoder.data)
        var dCtxG1 = [ContextModel](repeating: ContextModel(), count: 5)
        for i in 0..<values.count {
            let decoded = decoder.decodeVal(ctxG1: &dCtxG1)
            XCTAssertEqual(decoded, values[i], "Value at index \(i) mismatch")
        }
    }
}
