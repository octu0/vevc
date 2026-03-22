import XCTest
@testable import vevc

final class CrashDebug: XCTestCase {
    func testRunInter() {
        let t = InterleavedrANSTests()
        print("TEST STARTING")
        t.testInterleavedrANSEncodeDecodeAndPerformance()
        print("TEST FINISHED")
    }
}
