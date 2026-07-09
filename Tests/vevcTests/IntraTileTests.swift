import XCTest
@testable import vevc

final class IntraTileTests: XCTestCase {

    // 1. タイルマップ
    func testIntraTileMap() {
        let cases = [
            (1920, 1080),
            (1280, 720),
            (1080, 1920),
            (960, 540)
        ]
        
        for (w, h) in cases {
            let paddedW = (w + 7) & ~7
            let paddedH = (h + 7) & ~7
            let rects = computeIntraTileMap(width: w, height: h)
            
            var totalArea = 0
            var occupied = Set<Int>()
            
            for r in rects {
                totalArea += r.size * r.size
                
                // (b) 重複なしの検証
                for y in r.y..<(r.y + r.size) {
                    for x in r.x..<(r.x + r.size) {
                        let idx = y * paddedW + x
                        XCTAssertFalse(occupied.contains(idx), "Overlap at \(x), \(y)")
                        occupied.insert(idx)
                    }
                }
                
                // (d) サイズに応じたフィルタ・レベル
                if r.size == 512 || r.size == 128 {
                    XCTAssertEqual(r.filter, .cdf97)
                    let expectedLevel = (r.size == 512) ? 6 : 4
                    XCTAssertEqual(r.levels, expectedLevel)
                } else {
                    XCTAssertEqual(r.filter, .leGall53)
                    XCTAssertEqual(r.levels, 2)
                }
            }
            // (a) タイル面積の合計がパディング後プレーンと一致
            XCTAssertEqual(totalArea, paddedW * paddedH)
        }
        
        // (c) 決定的
        let r1 = computeIntraTileMap(width: 1920, height: 1080)
        let r2 = computeIntraTileMap(width: 1920, height: 1080)
        XCTAssertEqual(r1.count, r2.count)
        if r1.count == r2.count {
            for i in 0..<r1.count {
                XCTAssertEqual(r1[i].x, r2[i].x)
                XCTAssertEqual(r1[i].y, r2[i].y)
                XCTAssertEqual(r1[i].size, r2[i].size)
            }
        }
        
        // 余白の並び順アサート (左・上は昇順、右・下は降順)
        let rects = computeIntraTileMap(width: 1080, height: 1920)
        var xSequence: [Int] = []
        for r in rects where r.y == 0 {
            xSequence.append(r.size)
        }
        if xSequence.count >= 2 {
            XCTAssertLessThanOrEqual(xSequence[0], xSequence[1], "Left margin should be ascending")
            XCTAssertGreaterThanOrEqual(xSequence[xSequence.count - 2], xSequence[xSequence.count - 1], "Right margin should be descending")
        }
    }

    // 2. リフティング恒等
    func testLiftingIdentity() {
        let sizes = [512, 128, 32, 16, 8]
        
        for size in sizes {
            let levels = (size >= 128) ? ((size == 512) ? 6 : 4) : 2
            let filter: DWTFilterType = (size >= 128) ? .cdf97 : .leGall53
            
            // Random
            var input = [Int16](repeating: 0, count: size * size)
            for i in 0..<input.count {
                input[i] = Int16.random(in: -255...255)
            }
            verifyIdentity(input: input, size: size, levels: levels, filter: filter, name: "Random")
            
            // All 255
            var input255 = [Int16](repeating: 255, count: size * size)
            verifyIdentity(input: input255, size: size, levels: levels, filter: filter, name: "All255")
            
            // Checkerboard
            var inputChecker = [Int16](repeating: 0, count: size * size)
            for i in 0..<inputChecker.count {
                inputChecker[i] = (i % 2 == 0) ? 255 : -255
            }
            verifyIdentity(input: inputChecker, size: size, levels: levels, filter: filter, name: "Checkerboard")
            
            // Impulse
            var inputImpulse = [Int16](repeating: 0, count: size * size)
            inputImpulse[size * size / 2 + size / 2] = 255
            verifyIdentity(input: inputImpulse, size: size, levels: levels, filter: filter, name: "Impulse")
        }
    }
    
    private func verifyIdentity(input: [Int16], size: Int, levels: Int, filter: DWTFilterType, name: String) {
        var work = input
        
        // Transform
        work.withUnsafeMutableBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return }
            intraDwt2D(base: base, size: size, stride: size, levels: levels, filter: filter)
        }
        
        // Inverse
        work.withUnsafeMutableBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return }
            inverseIntraDwt2D(base: base, size: size, stride: size, levels: levels, filter: filter)
        }
        
        // Verify
        for i in 0..<input.count {
            let expected = input[i]
            let actual = work[i]
            let diff = abs(Int(expected) - Int(actual))
            if filter == .cdf97 {
                XCTAssertLessThanOrEqual(diff, 1, "Identity failed for \(name) size \(size) at \(i): expected \(expected), got \(actual)")
            } else {
                XCTAssertEqual(expected, actual, "Identity failed for \(name) size \(size) at \(i): expected \(expected), got \(actual)")
            }
        }
    }

    // 3. ゲイン規約
    func testGainConvention() {
        let size = 512
        let c: Int16 = 100
        var input = [Int16](repeating: c, count: size * size)
        
        input.withUnsafeMutableBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return }
            intraDwt2D(base: base, size: size, stride: size, levels: 6, filter: .cdf97)
        }
        
        let llVal = input[0]
        
        let K: Double = 1.229928
        let K12 = pow(K, 12.0)
        let expectedGain = 2.0 * K12 * Double(c)
        let expected = Int16(expectedGain)
        
        let diff = abs(Int(llVal) - Int(expected))
        let maxError = Int(Double(expected) * 0.02)
        
        XCTAssertLessThanOrEqual(diff, maxError, "Gain convention failed: expected \(expected), got \(llVal)")
    }

    // 4. 部分デコード輝度
    func testPartialDecodeBrightness() async throws {
        let width = 512
        let height = 512
        let yPlane = [Int16](repeating: 128, count: width * height)
        let cbPlane = [Int16](repeating: 0, count: (width/2) * (height/2))
        let crPlane = [Int16](repeating: 0, count: (width/2) * (height/2))
        
        let pool = BlockViewPool()
        let pd = PlaneData420(width: width, height: height, y: yPlane, cb: cbPlane, cr: crPlane)
        
        let qtY = QuantizationTable(baseStep: 16, isChroma: false, layerIndex: 0, profile: 0x02)
        let qtC = QuantizationTable(baseStep: 16, isChroma: true, layerIndex: 0, profile: 0x02)
        
        let payload = try await encodeIntraTiles(pd: pd, pool: pool, qtY: qtY, qtC: qtC)
        let fHeader = VEVCFileHeader(width: width, height: height, framerate: 0, profile: 0x02)
        
        let img2 = try await decodeIntraTiles(from: payload, pool: pool, header: fHeader, maxLayer: 2)
        let avgY2 = img2.y.reduce(0) { $0 + Int($1) } / img2.y.count
        XCTAssertTrue(abs(avgY2 - 128) <= 1, "maxLayer=2 brightness expected ~128, got \(avgY2)")
        
        let img1 = try await decodeIntraTiles(from: payload, pool: pool, header: fHeader, maxLayer: 1)
        let avgY1 = img1.y.reduce(0) { $0 + Int($1) } / img1.y.count
        XCTAssertTrue(abs(avgY1 - 128) <= 1, "maxLayer=1 brightness expected ~128, got \(avgY1)")
        
        let img0 = try await decodeIntraTiles(from: payload, pool: pool, header: fHeader, maxLayer: 0)
        let avgY0 = img0.y.reduce(0) { $0 + Int($1) } / img0.y.count
        XCTAssertTrue(abs(avgY0 - 128) <= 1, "maxLayer=0 brightness expected ~128, got \(avgY0)")
    }

    // 5. E2E
    func testEndToEndEncodeDecode() async throws {
        let width = 1920
        let height = 1080
        var yPlane = [Int16](repeating: 0, count: width * height)
        var cbPlane = [Int16](repeating: 0, count: (width/2) * (height/2))
        var crPlane = [Int16](repeating: 0, count: (width/2) * (height/2))
        
        for y in 0..<height {
            for x in 0..<width {
                yPlane[y * width + x] = Int16((x + y) % 255)
            }
        }
        for y in 0..<(height/2) {
            for x in 0..<(width/2) {
                cbPlane[y * (width/2) + x] = Int16((x) % 255)
                crPlane[y * (width/2) + x] = Int16((y) % 255)
            }
        }
        
        let pool = BlockViewPool()
        let pd = PlaneData420(width: width, height: height, y: yPlane, cb: cbPlane, cr: crPlane)
        
        let qtY = QuantizationTable(baseStep: 16, isChroma: false, layerIndex: 0, profile: 0x02)
        let qtC = QuantizationTable(baseStep: 16, isChroma: true, layerIndex: 0, profile: 0x02)
        
        let payload = try await encodeIntraTiles(pd: pd, pool: pool, qtY: qtY, qtC: qtC)
        let fHeader = VEVCFileHeader(width: width, height: height, framerate: 0, profile: 0x02)
        
        let decoded = try await decodeIntraTiles(from: payload, pool: pool, header: fHeader, maxLayer: 2)
        
        XCTAssertEqual(decoded.width, width)
        XCTAssertEqual(decoded.height, height)
        XCTAssertEqual(decoded.y.count, width * height)
        XCTAssertEqual(decoded.cb.count, (width/2) * (height/2))
        XCTAssertEqual(decoded.cr.count, (width/2) * (height/2))
    }
}
