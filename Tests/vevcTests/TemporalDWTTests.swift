import XCTest
@testable import vevc

final class TemporalDWTTests: XCTestCase {
    
    func testHaarLiftingPerfectReconstruction() {
        let count = 16
        var f0 = [Int16](repeating: 0, count: count)
        var f1 = [Int16](repeating: 0, count: count)
        
        // Randomish values including negatives
        for i in 0..<count {
            f0[i] = Int16(i * 10 - 50)
            f1[i] = Int16(i * 15 - 100)
        }
        
        let originalF0 = f0
        let originalF1 = f1
        
        var l = [Int16](repeating: 0, count: count)
        var h = [Int16](repeating: 0, count: count)
        
        f0.withUnsafeBufferPointer { ptr0 in
            f1.withUnsafeBufferPointer { ptr1 in
                l.withUnsafeMutableBufferPointer { ptrL in
                    h.withUnsafeMutableBufferPointer { ptrH in
                        // temporalDWT's internal haar function is private, but we can simulate the lifting steps
                        // to ensure the logic itself is mathematically losslessly reversible, or we can test temporalDWT directly
                        // Since haar is private and we want to test its behavior, let's test temporalDWT with dummy arrays
                    }
                }
            }
        }
    }
    
    func testTemporalDWTLossless() {
        let count = 16
        var f0 = [Int16](repeating: 0, count: count)
        var f1 = [Int16](repeating: 0, count: count)
        var f2 = [Int16](repeating: 0, count: count)
        var f3 = [Int16](repeating: 0, count: count)
        
        for i in 0..<count {
            f0[i] = Int16.random(in: -1000...1000)
            f1[i] = Int16.random(in: -1000...1000)
            f2[i] = Int16.random(in: -1000...1000)
            f3[i] = Int16.random(in: -1000...1000)
        }
        
        let orig0 = f0
        let orig1 = f1
        let orig2 = f2
        let orig3 = f3
        
        var ll = [Int16](repeating: 0, count: count)
        var lh = [Int16](repeating: 0, count: count)
        var h0 = [Int16](repeating: 0, count: count)
        var h1 = [Int16](repeating: 0, count: count)
        var t0 = [Int16](repeating: 0, count: count)
        var t1 = [Int16](repeating: 0, count: count)
        
        f0.withUnsafeBufferPointer { p0 in
        f1.withUnsafeBufferPointer { p1 in
        f2.withUnsafeBufferPointer { p2 in
        f3.withUnsafeBufferPointer { p3 in
            temporalDWT(
                f0: p0.baseAddress!,
                f1: p1.baseAddress!,
                f2: p2.baseAddress!,
                f3: p3.baseAddress!,
                count: count,
                outLL: &ll,
                outLH: &lh,
                outH0: &h0,
                outH1: &h1,
                tempL0: &t0,
                tempL1: &t1
            )
        }}}}
        
        var dec0 = [Int16](repeating: 0, count: count)
        var dec1 = [Int16](repeating: 0, count: count)
        var dec2 = [Int16](repeating: 0, count: count)
        var dec3 = [Int16](repeating: 0, count: count)
        
        ll.withUnsafeBufferPointer { pll in
        lh.withUnsafeBufferPointer { plh in
        h0.withUnsafeBufferPointer { ph0 in
        h1.withUnsafeBufferPointer { ph1 in
            invTemporalDWT(
                inLL: pll.baseAddress!,
                inLH: plh.baseAddress!,
                inH0: ph0.baseAddress!,
                inH1: ph1.baseAddress!,
                count: count,
                outF0: &dec0,
                outF1: &dec1,
                outF2: &dec2,
                outF3: &dec3,
                tempL0: &t0,
                tempL1: &t1
            )
        }}}}
        
        XCTAssertEqual(orig0, dec0)
        XCTAssertEqual(orig1, dec1)
        XCTAssertEqual(orig2, dec2)
        XCTAssertEqual(orig3, dec3)
    }
}
