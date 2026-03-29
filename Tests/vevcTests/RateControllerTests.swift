import XCTest
@testable import vevc

final class RateControllerTests: XCTestCase {
    
    func testRateControllerLogic() {
        let framerate = 30
        let keyint = 15
        let bitrateParam = 1000 * 1000 // 1Mbps
        
        var controller = RateController(maxbitrate: bitrateParam, framerate: framerate, keyint: keyint)
        
        let iFrameTarget = controller.beginGOP()
        
        // 1Mbps -> 33,333 bits per frame -> 500,000 bits per 15-frame GOP
        XCTAssertEqual(controller.gopTargetBits, 500_000)
        XCTAssertEqual(controller.gopRemainingBits, 500_000)
        XCTAssertEqual(controller.gopRemainingFrames, 15)
        
        // I-Frame gets 25% of GOP budget initially
        XCTAssertEqual(iFrameTarget, 125_000)
        
        // Simulate encoding an I-Frame
        controller.consumeIFrame(bits: 100_000, qStep: 32)
        XCTAssertEqual(controller.gopRemainingBits, 400_000)
        XCTAssertEqual(controller.gopRemainingFrames, 14)
        XCTAssertEqual(controller.lastPFrameBits, 0)
        
        // P-Frame 1
        let p1SAD = 1000
        let qStep1 = controller.calculatePFrameQStep(currentSAD: p1SAD, baseStep: 32)
        XCTAssertTrue(qStep1 >= 1 && qStep1 <= 128)
        
        controller.consumePFrame(bits: 20_000, qStep: qStep1, sad: p1SAD)
        XCTAssertEqual(controller.gopRemainingBits, 380_000)
        XCTAssertEqual(controller.gopRemainingFrames, 13)
        XCTAssertEqual(controller.lastPFrameBits, 20_000)
        XCTAssertEqual(controller.lastPFrameQStep, qStep1)
        
        // P-Frame 2 (high motion)
        let p2SAD = 2000
        let qStep2 = controller.calculatePFrameQStep(currentSAD: p2SAD, baseStep: 32)
        controller.consumePFrame(bits: 50_000, qStep: qStep2, sad: p2SAD)
        XCTAssertEqual(controller.gopRemainingBits, 330_000)
        XCTAssertEqual(controller.gopRemainingFrames, 12)
    }
}
