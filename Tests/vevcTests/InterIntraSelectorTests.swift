import XCTest
@testable import vevc

final class InterIntraSelectorTests: XCTestCase {
    
    // RDO selection logic extracted for testing
    func selectModeForPFrame(original: [Int16], interPredicted: [Int16], intraPredictions: [(mode: IntraPredictorMode, pred: [Int16])]) -> (isInter: Bool, bestIntraMode: IntraPredictorMode?) {
        // ... will implement in tests based on SAD ...
        var interSAD = 0
        for i in 0..<1024 {
            interSAD += Int(abs(Int32(original[i]) - Int32(interPredicted[i])))
        }
        
        // Fast path: if inter is very good, skip intra
        if interSAD < 500 {
            return (true, nil)
        }
        
        var bestIntraSAD = Int.max
        var bestMode: IntraPredictorMode = .dc
        
        for p in intraPredictions {
            var sad = 0
            for i in 0..<1024 {
                sad += Int(abs(Int32(original[i]) - Int32(p.pred[i])))
            }
            // Add penalty
            if p.mode != .dc { sad += 64 }
            
            if sad < bestIntraSAD {
                bestIntraSAD = sad
                bestMode = p.mode
            }
        }
        
        if interSAD <= bestIntraSAD + 128 {
            return (true, nil)
        } else {
            return (false, bestMode)
        }
    }
    
    func testFastInterSelection() {
        let original = [Int16](repeating: 100, count: 1024)
        let inter = [Int16](repeating: 100, count: 1024) // perfect match
        let intra = [(IntraPredictorMode.dc, [Int16](repeating: 50, count: 1024))]
        
        let result = selectModeForPFrame(original: original, interPredicted: inter, intraPredictions: intra)
        XCTAssertTrue(result.isInter, "Perfect inter match must use Inter mode")
    }
    
    func testIntraSelectionWhenInterIsBad() {
        let original = [Int16](repeating: 100, count: 1024)
        var inter = [Int16](repeating: 100, count: 1024)
        // Inter is very bad
        for i in 0..<1024 { inter[i] = 10 }
        
        // Intra is perfect
        let intra = [(IntraPredictorMode.dc, [Int16](repeating: 100, count: 1024))]
        
        let result = selectModeForPFrame(original: original, interPredicted: inter, intraPredictions: intra)
        XCTAssertFalse(result.isInter, "Bad inter match with perfect intra must use Intra mode")
        XCTAssertEqual(result.bestIntraMode, .dc)
    }
}
