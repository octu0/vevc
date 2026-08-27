import XCTest
@testable import vevc

final class RateControllerTests: XCTestCase {

    func testRateControllerLogic() {
        let framerate = 30
        let keyint = 15
        let bitrateParam = 1000 * 1000  // 1Mbps

        var controller = RateController(maxbitrate: bitrateParam, framerate: framerate, keyint: keyint)

        let iFrameTarget = controller.beginGOP()

        // 1Mbps * 1.3 -> 43,333 bits per frame -> 650,000 bits per 15-frame GOP
        XCTAssertEqual(controller.gopTargetBits, 650_000)
        XCTAssertEqual(controller.gopRemainingBits, 650_000)
        XCTAssertEqual(controller.gopRemainingFrames, 15)

        // I-Frame gets ~26.3% of GOP budget initially (5 / (keyint + 4)) or absoluteFloor which is 260,000 (plannedBitrate * 6 / framerate)
        XCTAssertEqual(iFrameTarget, 260_000)

        // Simulate encoding an I-Frame
        controller.consumeIFrame(bits: 100_000, qStep: 32)
        XCTAssertEqual(controller.gopRemainingBits, 550_000)
        XCTAssertEqual(controller.gopRemainingFrames, 14)
        XCTAssertEqual(controller.lastPFrameBits, 0)

        // P-Frame 1
        let p1SAD = 1000
        let qStep1 = controller.calculatePFrameQStep(currentSAD: p1SAD, baseStep: 32)
        XCTAssertTrue(1 <= qStep1 && qStep1 <= 128)

        controller.consumePFrame(bits: 20_000, qStep: qStep1, sad: p1SAD, distortion: 500, interRatioQ8: 0, detailThinned: false)
        XCTAssertEqual(controller.gopRemainingBits, 530_000)
        XCTAssertEqual(controller.gopRemainingFrames, 13)
        XCTAssertEqual(controller.lastPFrameBits, 20_000)
        XCTAssertEqual(controller.lastPFrameQStep, qStep1)

        // P-Frame 2 (high motion)
        let p2SAD = 2000
        let qStep2 = controller.calculatePFrameQStep(currentSAD: p2SAD, baseStep: 32)
        controller.consumePFrame(bits: 50_000, qStep: qStep2, sad: p2SAD, distortion: 800, interRatioQ8: 0, detailThinned: false)
        XCTAssertEqual(controller.gopRemainingBits, 480_000)
        XCTAssertEqual(controller.gopRemainingFrames, 12)
    }

    func testSaturationState() {
        let framerate = 30
        let keyint = 15
        let bitrateParam = 1000 * 1000 // 1Mbps
        var controller = RateController(maxbitrate: bitrateParam, framerate: framerate, keyint: keyint, targetDistortion: 500)
        
        let _ = controller.beginGOP()
        controller.consumeIFrame(bits: 100_000, qStep: 32)
        
        // At start, quality is not saturated.
        XCTAssertFalse(controller.isQualitySaturated)
        
        // P-Frames: Small size, low distortion (e.g. 200 < 500)
        let q1 = controller.calculatePFrameQStep(currentSAD: 1000, baseStep: 32)
        controller.consumePFrame(bits: 5000, qStep: q1, sad: 1000, distortion: 200, interRatioQ8: 0, detailThinned: false)
        
        _ = controller.calculatePFrameQStep(currentSAD: 1000, baseStep: 32)
        controller.consumePFrame(bits: 5000, qStep: q1, sad: 1000, distortion: 200, interRatioQ8: 0, detailThinned: false)
        
        _ = controller.calculatePFrameQStep(currentSAD: 1000, baseStep: 32)
        controller.consumePFrame(bits: 5000, qStep: q1, sad: 1000, distortion: 200, interRatioQ8: 0, detailThinned: false)
        
        // avgDistortion < targetDistortion (200 < 500) -> Quality saturated
        XCTAssertTrue(controller.isQualitySaturated)
        
        // Now increase distortion beyond hysteresis threshold: (500 * 5) / 4 = 625
        for _ in 0..<10 {
            _ = controller.calculatePFrameQStep(currentSAD: 1000, baseStep: 32)
            controller.consumePFrame(bits: 5000, qStep: q1, sad: 1000, distortion: 800, interRatioQ8: 0, detailThinned: false)
        }
        
        // avgDistortion should now exceed 625 -> Saturation released
        XCTAssertFalse(controller.isQualitySaturated)
    }

    func testDriftAcceleratingWithDetailThinned() {
        var controller = RateController(maxbitrate: 1_000_000, framerate: 30, keyint: 30, targetDistortion: 600)
        let _ = controller.beginGOP()
        controller.consumeIFrame(bits: 100_000, qStep: 32)

        // 通常フレーム: avgDistortion を確立 (30 * 256)
        let baseDistortion = 30 * 256
        let q1 = controller.calculatePFrameQStep(currentSAD: 1000, baseStep: 32)
        controller.consumePFrame(bits: 10_000, qStep: q1, sad: 1000, distortion: baseDistortion, interRatioQ8: 0, detailThinned: false)
        XCTAssertFalse(controller.isDriftAccelerating(framesSinceKeyframe: 8))
        XCTAssertEqual(controller.driftStreak, 0)

        // (1) detailThinned=true のフレームでは driftStreak が増えないこと
        let highDistortion = 100 * 256
        let q2 = controller.calculatePFrameQStep(currentSAD: 1000, baseStep: 32)
        controller.consumePFrame(bits: 5_000, qStep: q2, sad: 1000, distortion: highDistortion, interRatioQ8: 0, detailThinned: true)
        XCTAssertEqual(controller.driftStreak, 0)
        XCTAssertFalse(controller.isDriftAccelerating(framesSinceKeyframe: 8))

        // (2) detailThinned=false で条件成立が 1 回だけなら isDriftAccelerating は false
        controller.consumePFrame(bits: 5_000, qStep: q2, sad: 1000, distortion: highDistortion, interRatioQ8: 0, detailThinned: false)
        XCTAssertEqual(controller.driftStreak, 1)
        XCTAssertFalse(controller.isDriftAccelerating(framesSinceKeyframe: 8))

        // (2) detailThinned=false で条件成立が 2 連続すると isDriftAccelerating(framesSinceKeyframe: 8) が true
        controller.consumePFrame(bits: 5_000, qStep: q2, sad: 1000, distortion: highDistortion, interRatioQ8: 0, detailThinned: false)
        XCTAssertEqual(controller.driftStreak, 2)
        XCTAssertTrue(controller.isDriftAccelerating(framesSinceKeyframe: 8))

        // (3) framesSinceKeyframe が min(8, keyint/2) 未満なら streak が揃っていても false
        XCTAssertFalse(controller.isDriftAccelerating(framesSinceKeyframe: 7))
        XCTAssertFalse(controller.isDriftAccelerating(framesSinceKeyframe: 0))
    }
}
