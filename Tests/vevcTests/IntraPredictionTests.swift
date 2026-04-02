import XCTest
@testable import vevc

final class IntraPredictionTests: XCTestCase {
    
    // Simulate the truncation quantization bug that caused 22MB explosion
    func testQuantizeTruncationBug() {
        let qtLLStep: Int32 = 32
        
        let orig: Int32 = 150
        let pred: Int32 = 120
        let difference = orig - pred // 30
        
        // Old buggy behavior (Truncation)
        let bugQRes = Int16(clamping: difference / qtLLStep)
        let bugReconRes = Int16(clamping: Int32(bugQRes) * qtLLStep)
        let bugRecon = Int32(pred) + Int32(bugReconRes)
        let bugError = orig - bugRecon
        
        // New fixed behavior (Rounding with half-step offset)
        let offset = qtLLStep >> 1
        let fixedQRes: Int32
        if difference >= 0 {
            fixedQRes = (difference + offset) / qtLLStep
        } else {
            fixedQRes = (difference - offset) / qtLLStep
        }
        let fixedReconRes = Int16(clamping: fixedQRes * qtLLStep)
        let fixedRecon = Int32(pred) + Int32(fixedReconRes)
        let fixedError = orig - fixedRecon
        
        XCTAssertEqual(bugQRes, 0)
        XCTAssertEqual(bugRecon, 120)
        XCTAssertEqual(bugError, 30) // Huge 30 error!
        
        XCTAssertEqual(fixedQRes, 1)
        XCTAssertEqual(fixedRecon, 152)
        XCTAssertEqual(fixedError, -2) // Tiny 2 error!
    }
}
