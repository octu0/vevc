import XCTest
@testable import vevc

final class RateControllerTests: XCTestCase {

    func testRateControllerLogic() {
        let framerate = 30
        let keyint = 15
        let bitrateParam = 1000 * 1000  // 1Mbps

        var controller = RateController(maxbitrate: bitrateParam, framerate: framerate, keyint: keyint)

        let iFrameTarget = controller.beginGOP()

        // 1Mbps -> 33,333 bits per frame -> 500,000 bits per 15-frame GOP
        XCTAssertEqual(controller.gopTargetBits, 500_000)
        XCTAssertEqual(controller.gopRemainingBits, 500_000)
        XCTAssertEqual(controller.gopRemainingFrames, 15)

        // I-Frame gets ~26.3% of GOP budget initially (5 / (keyint + 4)) or absoluteFloor which is 200,000 (maxbitrate * 6 / framerate)
        XCTAssertEqual(iFrameTarget, 200_000)

        // Simulate encoding an I-Frame
        controller.consumeIFrame(bits: 100_000, qStep: 32)
        XCTAssertEqual(controller.gopRemainingBits, 400_000)
        XCTAssertEqual(controller.gopRemainingFrames, 14)
        XCTAssertEqual(controller.lastPFrameBits, 0)

        // P-Frame 1
        let p1SAD = 1000
        let qStep1 = controller.calculatePFrameQStep(currentSAD: p1SAD, baseStep: 32)
        XCTAssertTrue(qStep1 >= 1 && qStep1 <= 128)

        controller.consumePFrame(bits: 20_000, qStep: qStep1, sad: p1SAD, distortion: 500)
        XCTAssertEqual(controller.gopRemainingBits, 380_000)
        XCTAssertEqual(controller.gopRemainingFrames, 13)
        XCTAssertEqual(controller.lastPFrameBits, 20_000)
        XCTAssertEqual(controller.lastPFrameQStep, qStep1)

        // P-Frame 2 (high motion)
        let p2SAD = 2000
        let qStep2 = controller.calculatePFrameQStep(currentSAD: p2SAD, baseStep: 32)
        controller.consumePFrame(bits: 50_000, qStep: qStep2, sad: p2SAD, distortion: 800)
        XCTAssertEqual(controller.gopRemainingBits, 330_000)
        XCTAssertEqual(controller.gopRemainingFrames, 12)
    }

    func testSaturationState() {
        let framerate = 30
        let keyint = 15
        let bitrateParam = 1000 * 1000 // 1Mbps
        var controller = RateController(maxbitrate: bitrateParam, framerate: framerate, keyint: keyint, targetDistortion: 2)
        
        let _ = controller.beginGOP()
        controller.consumeIFrame(bits: 100_000, qStep: 32)
        
        // At start, budgetSurplusEMAQ8 is 256. Quality is not saturated.
        XCTAssertFalse(controller.isQualitySaturated)
        
        // P-Frame 1: Small size, low distortion.
        // Theoretical budget per P-frame is ~80% of 1Mbps/30 = ~26.6Kbits.
        // Let's use 5000 bits. Ratio = 26666 * 256 / 5000 = ~1365.
        // EMA will go up quickly.
        let q1 = controller.calculatePFrameQStep(currentSAD: 1000, baseStep: 32)
        controller.consumePFrame(bits: 5000, qStep: q1, sad: 1000, distortion: 1)
        
        // Calculate step again to update EMA
        _ = controller.calculatePFrameQStep(currentSAD: 1000, baseStep: 32)
        controller.consumePFrame(bits: 5000, qStep: q1, sad: 1000, distortion: 1)
        
        _ = controller.calculatePFrameQStep(currentSAD: 1000, baseStep: 32)
        controller.consumePFrame(bits: 5000, qStep: q1, sad: 1000, distortion: 1)
        
        // Now budgetSurplusEMAQ8 should be > 320, and avgDistortion < targetDistortion (1 < 2)
        XCTAssertTrue(controller.isQualitySaturated)
        
        // Now simulate high bitrate (budget tight). size = 50_000
        _ = controller.calculatePFrameQStep(currentSAD: 1000, baseStep: 32)
        controller.consumePFrame(bits: 50_000, qStep: q1, sad: 1000, distortion: 1)
        
        _ = controller.calculatePFrameQStep(currentSAD: 1000, baseStep: 32)
        controller.consumePFrame(bits: 50_000, qStep: q1, sad: 1000, distortion: 1)
        
        _ = controller.calculatePFrameQStep(currentSAD: 1000, baseStep: 32)
        controller.consumePFrame(bits: 50_000, qStep: q1, sad: 1000, distortion: 1)
        
        // budgetSurplusEMAQ8 drops < 256
        XCTAssertFalse(controller.isQualitySaturated)
        
        // Now test hysteresis on distortion side.
        // Make budget high again
        _ = controller.calculatePFrameQStep(currentSAD: 1000, baseStep: 32)
        controller.consumePFrame(bits: 5000, qStep: q1, sad: 1000, distortion: 1)
        _ = controller.calculatePFrameQStep(currentSAD: 1000, baseStep: 32)
        controller.consumePFrame(bits: 5000, qStep: q1, sad: 1000, distortion: 1)
        _ = controller.calculatePFrameQStep(currentSAD: 1000, baseStep: 32)
        controller.consumePFrame(bits: 5000, qStep: q1, sad: 1000, distortion: 1)
        
        XCTAssertTrue(controller.isQualitySaturated)
        
        // Now make distortion high (e.g. 3). avgDistortion goes up
        _ = controller.calculatePFrameQStep(currentSAD: 1000, baseStep: 32)
        controller.consumePFrame(bits: 5000, qStep: q1, sad: 1000, distortion: 3)
        _ = controller.calculatePFrameQStep(currentSAD: 1000, baseStep: 32)
        controller.consumePFrame(bits: 5000, qStep: q1, sad: 1000, distortion: 3)
        _ = controller.calculatePFrameQStep(currentSAD: 1000, baseStep: 32)
        controller.consumePFrame(bits: 5000, qStep: q1, sad: 1000, distortion: 3)
        
        // avgDistortion should now be > 2.5 (which is 2 * 1.25)
        XCTAssertFalse(controller.isQualitySaturated)
    }
}
