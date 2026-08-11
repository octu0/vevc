import XCTest
@testable import vevc

final class ImplicitConditioningTests: XCTestCase {
    
    func testResidualModulationRoundtrip() {
        let blockSize = 16
        let pool = BlockViewPool()
        let blockX = pool.get(width: blockSize, height: blockSize)
        let blockMu = pool.get(width: blockSize, height: blockSize)
        defer {
            pool.put(blockX)
            pool.put(blockMu)
        }
        
        var originalX = [Int16](repeating: 0, count: blockSize * blockSize)
        var originalMu = [Int16](repeating: 0, count: blockSize * blockSize)
        
        for y in 0..<blockSize {
            let rX = blockX.rowPointer(y: y)
            let rMu = blockMu.rowPointer(y: y)
            for x in 0..<blockSize {
                let valX = Int16.random(in: -500...500)
                let valMu = Int16.random(in: -300...300)
                rX[x] = valX
                rMu[x] = valMu
                originalX[y * blockSize + x] = valX
                originalMu[y * blockSize + x] = valMu
            }
        }
        
        var blocks = [blockX]
        let mcBlocks = [blockMu]
        
        // Z = X - μ
        ImplicitConditioning.applyResidualModulation(blocks: &blocks, mcBlocks: mcBlocks, blockSize: blockSize)
        
        // Check Z values
        for y in 0..<blockSize {
            let rZ = blocks[0].rowPointer(y: y)
            for x in 0..<blockSize {
                let expectedZ = originalX[y * blockSize + x] - originalMu[y * blockSize + x]
                XCTAssertEqual(rZ[x], expectedZ)
            }
        }
        
        // X = Z + μ
        ImplicitConditioning.applyResidualDemodulation(blocks: &blocks, mcBlocks: mcBlocks, blockSize: blockSize)
        
        // Check restored X
        for y in 0..<blockSize {
            let rRestored = blocks[0].rowPointer(y: y)
            for x in 0..<blockSize {
                let origX = originalX[y * blockSize + x]
                XCTAssertEqual(rRestored[x], origX)
            }
        }
    }
    
    func testDownsample4x4Average() {
        let srcWidth = 16
        let srcHeight = 16
        let dstWidth = 4
        let dstHeight = 4
        
        var src = [Int16](repeating: 100, count: srcWidth * srcHeight)
        src.withUnsafeBufferPointer { sBuf in
            let dst = ImplicitConditioning.downsamplePlane4x4Average(
                src: sBuf,
                srcWidth: srcWidth,
                srcHeight: srcHeight,
                dstWidth: dstWidth,
                dstHeight: dstHeight
            )
            XCTAssertEqual(dst.count, dstWidth * dstHeight)
            for val in dst {
                XCTAssertEqual(val, 100)
            }
        }
    }
}

