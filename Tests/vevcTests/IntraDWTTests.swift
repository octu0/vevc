import XCTest
@testable import vevc

final class IntraDWTTests: XCTestCase {
    
    func checkRoundtrip1D(filter: DWTFilterType, length: Int, count: Int, isXPass: Bool) {
        let total = length * count
        var input = [Int16](repeating: 0, count: total)
        for i in 0..<total {
            input[i] = Int16.random(in: -1000...1000)
        }
        
        // Include extremes that won't overflow Int16 during lifting
        if total >= 4 {
            input[0] = 4000
            input[1] = -4000
            input[2] = 0
            input[3] = 1
        }
        
        var buffer = input
        buffer.withUnsafeMutableBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return }
            let stride = isXPass ? length : count
            
            if filter == .cdf97 {
                cdf97LiftParametric(base: base, length: length, count: count, stride: stride, isXPass: isXPass, isInverse: false)
                cdf97LiftParametric(base: base, length: length, count: count, stride: stride, isXPass: isXPass, isInverse: true)
            } else {
                leGall53LiftParametric(base: base, length: length, count: count, stride: stride, isXPass: isXPass, isInverse: false)
                leGall53LiftParametric(base: base, length: length, count: count, stride: stride, isXPass: isXPass, isInverse: true)
            }
        }
        
        for i in 0..<total {
            if filter == .cdf97 {
                let diff = abs(Int(input[i]) - Int(buffer[i]))
                XCTAssertTrue(diff <= 2, "CDF97 Mismatch at \(i): expected \(input[i]), got \(buffer[i])")
            } else {
                XCTAssertEqual(input[i], buffer[i], "Mismatch at \(i) for filter=\(filter) length=\(length) count=\(count) isXPass=\(isXPass)")
            }
        }
    }
    
    func checkRoundtrip2D(filter: DWTFilterType, size: Int, levels: Int) {
        let total = size * size
        var input = [Int16](repeating: 0, count: total)
        for y in 0..<size {
            for x in 0..<size {
                let v = Int16.random(in: -1000...1000)
                input[y * size + x] = v
            }
        }
        
        // Add extreme values (checkerboard etc)
        for i in 0..<min(total, 16) {
            input[i] = (i % 2 == 0) ? 255 : -255
        }
        
        var buffer = input
        buffer.withUnsafeMutableBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return }
            intraDwt2D(base: base, size: size, stride: size, levels: levels, filter: filter)
            inverseIntraDwt2D(base: base, size: size, stride: size, levels: levels, filter: filter)
        }
        
        for i in 0..<total {
            XCTAssertEqual(input[i], buffer[i], "2D Mismatch at \(i) for filter=\(filter) size=\(size) levels=\(levels)")
        }
    }
    
    func testParametric1D_CDF97() {
        for length in [16, 32, 64, 128, 256, 512] {
            checkRoundtrip1D(filter: .cdf97, length: length, count: 16, isXPass: true)
            checkRoundtrip1D(filter: .cdf97, length: length, count: 16, isXPass: false)
        }
    }
    
    func testParametric1D_LeGall53() {
        for length in [4, 8, 16, 32] {
            // Count can be 4 or 8
            checkRoundtrip1D(filter: .leGall53, length: length, count: 4, isXPass: true)
            checkRoundtrip1D(filter: .leGall53, length: length, count: 4, isXPass: false)
            checkRoundtrip1D(filter: .leGall53, length: length, count: 8, isXPass: true)
            checkRoundtrip1D(filter: .leGall53, length: length, count: 8, isXPass: false)
        }
    }
    
    func testParametric2D_CDF97_512() {
        checkRoundtrip2D(filter: .cdf97, size: 512, levels: 6)
    }
    
    func testParametric2D_CDF97_128() {
        checkRoundtrip2D(filter: .cdf97, size: 128, levels: 4)
    }
    
    func testParametric2D_LeGall53_32() {
        checkRoundtrip2D(filter: .leGall53, size: 32, levels: 2)
    }
    
    func testParametric2D_LeGall53_8() {
        checkRoundtrip2D(filter: .leGall53, size: 8, levels: 2)
    }
    
    // Gain check for CDF97 (as requested in §6-4)
    func testCDF97Gain() {
        let size = 512
        let total = size * size
        var input = [Int16](repeating: 100, count: total)
        
        buffer: do {
            var buffer = input
            buffer.withUnsafeMutableBufferPointer { ptr in
                guard let base = ptr.baseAddress else { return }
                intraDwt2D(base: base, size: size, stride: size, levels: 6, filter: .cdf97)
                
                // For level=6, the LL is 8x8. It is stored at the beginning of the buffer.
                // Guard bit: * 2
                // Gain per level: K^2 ≈ 1.5133
                // For 6 levels: gain ≈ 2 * 100 * (1.5133)^6 ≈ 200 * 11.45 ≈ 2290
                // Wait, K^12 ≈ 12.1.  2 * 100 * 12.1 ≈ 2420
                let llValue = ptr[0]
                // print("LL value: \(llValue)")
                // Check if it's in the expected range (e.g. 2300...2500)
                XCTAssertTrue(llValue > 2300 && llValue < 2500, "Unexpected LL gain: \(llValue)")
            }
        }
    }
}
