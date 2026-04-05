import Testing
@testable import vevc

@Suite("SpatialAdaptive Tests")
struct SpatialAdaptiveTests {
    
    // MARK: - spatialWeight tests
    
    @Test("spatialWeight: center block returns minimum weight (≈1024)")
    func centerBlockWeight() {
        // Center of an 11x11 grid (odd size = exact center)
        let w = spatialWeight(blockCol: 5, blockRow: 5, colCount: 11, rowCount: 11)
        #expect(abs(w - 1024) < 10, "Center block should have weight ≈ 1024, got \(w)")
    }
    
    @Test("spatialWeight: corner blocks return maximum weight")
    func cornerBlockWeight() {
        let colCount = 10
        let rowCount = 10
        let center = spatialWeight(blockCol: 5, blockRow: 5, colCount: colCount, rowCount: rowCount)
        let corner = spatialWeight(blockCol: 0, blockRow: 0, colCount: colCount, rowCount: rowCount)
        #expect(center < corner, "Corner weight \(corner) should be > center weight \(center)")
        // 1.3 * 1024 = 1331
        #expect(1331 < corner, "Corner weight \(corner) should be > 1331 (1.3 in 1024-scale)")
    }
    
    @Test("spatialWeight: monotonically increases from center to edge")
    func monotonicIncrease() {
        let colCount = 20
        let rowCount = 20
        let centerCol = colCount / 2
        let centerRow = rowCount / 2
        
        var prevWeight = spatialWeight(blockCol: centerCol, blockRow: centerRow, colCount: colCount, rowCount: rowCount)
        // Walk from center to corner
        for step in 1...min(centerCol, centerRow) {
            let w = spatialWeight(blockCol: centerCol - step, blockRow: centerRow - step, colCount: colCount, rowCount: rowCount)
            #expect(w >= prevWeight, "Weight should increase from center: step=\(step) w=\(w) prev=\(prevWeight)")
            prevWeight = w
        }
    }
    
    @Test("spatialWeight: symmetric across center")
    func symmetric() {
        let colCount = 10
        let rowCount = 10
        let w1 = spatialWeight(blockCol: 2, blockRow: 3, colCount: colCount, rowCount: rowCount)
        let w2 = spatialWeight(blockCol: colCount - 1 - 2, blockRow: rowCount - 1 - 3, colCount: colCount, rowCount: rowCount)
        #expect(abs(w1 - w2) < 2, "Symmetric blocks should have equal weight: \(w1) vs \(w2)")
    }
    
    @Test("spatialWeight: 1x1 grid returns 1024")
    func singleBlock() {
        let w = spatialWeight(blockCol: 0, blockRow: 0, colCount: 1, rowCount: 1)
        #expect(abs(w - 1024) < 10, "Single block should have weight 1024, got \(w)")
    }
    
    // MARK: - Spatial adaptive SAD threshold tests
    
    @Test("spatialSADThreshold: center blocks use base threshold")
    func centerSADThreshold() {
        let baseThreshold = 150
        let colCount = 10
        let rowCount = 10
        let threshold = spatialSADThreshold(baseSAD: baseThreshold, blockCol: 5, blockRow: 5, colCount: colCount, rowCount: rowCount)
        // Center should be close to base
        #expect(threshold >= baseThreshold, "Center threshold \(threshold) should be >= base \(baseThreshold)")
        #expect(threshold <= baseThreshold + 20, "Center threshold \(threshold) should be close to base \(baseThreshold)")
    }
    
    @Test("spatialSADThreshold: edge blocks have higher threshold than center")
    func edgeHigherThanCenter() {
        let baseThreshold = 150
        let colCount = 10
        let rowCount = 10
        let center = spatialSADThreshold(baseSAD: baseThreshold, blockCol: 5, blockRow: 5, colCount: colCount, rowCount: rowCount)
        let edge = spatialSADThreshold(baseSAD: baseThreshold, blockCol: 0, blockRow: 0, colCount: colCount, rowCount: rowCount)
        #expect(center < edge, "Edge threshold \(edge) should be > center \(center)")
    }
}

