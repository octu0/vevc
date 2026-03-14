import XCTest
@testable import vevc

final class MotionTests: XCTestCase {
    func testBenchmarkApplyMBME() async throws {
        let w = 1920
        let h = 1080
        
        var yData = [Int16](repeating: 0, count: w * h)
        for i in 0..<yData.count {
            yData[i] = Int16(i % 256)
        }
        let cbData = [Int16](repeating: 0, count: (w/2) * (h/2))
        let crData = [Int16](repeating: 0, count: (w/2) * (h/2))
        
        let prev = PlaneData420(width: w, height: h, y: yData, cb: cbData, cr: crData)
        
        let mbSize = 32
        let mbCols = (w + mbSize - 1) / mbSize
        let mbRows = (h + mbSize - 1) / mbSize
        var mvsMutable = MotionVectors(count: mbCols * mbRows)
        for i in 0..<mvsMutable.vectors.count {
            mvsMutable.vectors[i] = SIMD2(Int16.random(in: -16...16), Int16.random(in: -16...16))
        }
        let mvs = mvsMutable
        
        measure {
            let exp = expectation(description: "apply")
            Task {
                let _ = await applyMBME(prev: prev, mvs: mvs)
                exp.fulfill()
            }
            wait(for: [exp], timeout: 10.0)
        }
    }

    func testBenchmarkCalculateSADEdge() {
        let w = 1920
        let h = 1080
        
        var currData = [Int16](repeating: 0, count: w * h)
        var prevData = [Int16](repeating: 0, count: w * h)
        for i in 0..<currData.count {
            currData[i] = Int16(i % 256)
            prevData[i] = Int16((i + 13) % 256)
        }
        
        var totalSAD = 0
        
        measure {
            for i in 0..<10000 {
                let startX = (i * 17) % (w - 32)
                let startY = (i * 19) % (h - 32)
                
                currData.withUnsafeBufferPointer { currPtr in
                    prevData.withUnsafeBufferPointer { prevPtr in
                        let sad = calculateSADEdge(
                            pCurr: currPtr.baseAddress!,
                            pPrev: prevPtr.baseAddress!,
                            w: w, h: h,
                            startX: startX, startY: startY,
                            actW: 32, actH: 32,
                            dx: -16 + (i % 32),
                            dy: -16 + ((i / 32) % 32)
                        )
                        totalSAD &+= sad
                    }
                }
            }
        }
        print("totalSAD = \(totalSAD)")
    }
}
