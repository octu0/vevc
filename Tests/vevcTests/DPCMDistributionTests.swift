import XCTest
@testable import vevc

/// Oneモード(DPCM)の実データトークン分布を測定し、最適な静的rANSテーブルを設計するためのテスト
final class DPCMDistributionTests: XCTestCase {
    
    /// 128x128画像を使ったDPCMトークン分布測定
    /// 多様なパターンを含む画像を複数生成してトークン分布を集約
    func testCollectDPCMTokenDistribution() async throws {
        let pool = BlockViewPool()
        var globalRunCounts = [Int](repeating: 0, count: 64)
        var globalValCounts = [Int](repeating: 0, count: 64)
        var totalPairs = 0
        
        // generate diverse test images to approximate real-world distribution
        let patterns: [(String, (Int, Int, Int, Int) -> Int16)] = [
            ("gradient_h", { x, y, w, h in Int16(x * 255 / max(1, w - 1)) }),
            ("gradient_v", { x, y, w, h in Int16(y * 255 / max(1, h - 1)) }),
            ("gradient_diag", { x, y, w, h in Int16((x + y) * 127 / max(1, w + h - 2)) }),
            ("checkerboard", { x, y, w, h in Int16((x / 8 + y / 8) % 2 == 0 ? 200 : 50) }),
            ("noise_low", { x, y, w, h in Int16.random(in: 0...30) }),
            ("noise_mid", { x, y, w, h in Int16.random(in: 0...128) }),
            ("noise_high", { x, y, w, h in Int16.random(in: 0...255) }),
            ("flat_dark", { _, _, _, _ in Int16(20) }),
            ("flat_mid", { _, _, _, _ in Int16(128) }),
            ("flat_bright", { _, _, _, _ in Int16(230) }),
            ("stripe_h8", { x, y, w, h in Int16((y / 8) % 2 == 0 ? 180 : 60) }),
            ("stripe_v8", { x, y, w, h in Int16((x / 8) % 2 == 0 ? 180 : 60) }),
            ("circle", { x, y, w, h in
                let cx = w / 2, cy = h / 2
                let dist = (x - cx) * (x - cx) + (y - cy) * (y - cy)
                let rad = min(w, h) / 3
                return Int16(dist < rad * rad ? 200 : 50)
            }),
            ("smooth_wave", { x, y, w, h in
                let v = sin(Double(x) * 6.28 / Double(w)) * 50.0 + sin(Double(y) * 6.28 / Double(h)) * 50.0 + 128.0
                return Int16(max(0, min(255, Int(v))))
            }),
            ("edge_block", { x, y, w, h in
                Int16(x < w / 2 ? (y < h / 2 ? 200 : 100) : (y < h / 2 ? 50 : 170))
            }),
        ]
        
        let width = 256
        let height = 256
        
        for (name, gen) in patterns {
            var yPlane = [Int16](repeating: 0, count: width * height)
            let cbPlane = [Int16](repeating: 128, count: (width / 2) * (height / 2))
            let crPlane = [Int16](repeating: 128, count: (width / 2) * (height / 2))
            
            for y in 0..<height {
                for x in 0..<width {
                    yPlane[y * width + x] = gen(x, y, width, height)
                }
            }
            
            let pd = PlaneData420(width: width, height: height, y: yPlane, cb: cbPlane, cr: crPlane)
            let qtY = QuantizationTable(baseStep: 24, isChroma: false, layerIndex: 0)
            
            // extract blocks and quantize
            var (blocks, _) = await extractSingleTransformBlocks32(r: pd.rY, width: width, height: height, pool: pool)
            for i in blocks.indices {
                evaluateQuantizeBase32(block: &blocks[i], qt: qtY)
            }
            
            // encode with DynamicEntropyModel to get actual pairs
            var encoder = EntropyEncoder<DynamicEntropyModel>()
            var lastVal: Int16 = 0
            
            for i in blocks.indices {
                let isZero = isEffectivelyZeroBase32(data: blocks[i].base, threshold: 3)
                if isZero {
                    lastVal = 0
                    continue
                }
                let view = blocks[i]
                let subs = getSubbands32(view: view)
                blockEncodeDPCM16(encoder: &encoder, block: subs.ll, lastVal: &lastVal)
                blockEncode16V(encoder: &encoder, block: subs.hl, parentBlock: nil)
                blockEncode16H(encoder: &encoder, block: subs.lh, parentBlock: nil)
                blockEncode16H(encoder: &encoder, block: subs.hh, parentBlock: nil)
                        }
            encoder.flush()
            
            // collect token distribution from pairs
            for pair in encoder.pairs {
                let runResult = valueTokenizeUnsigned(pair.run)
                let valResult = valueTokenize(pair.val)
                globalRunCounts[Int(runResult.token)] += 1
                globalValCounts[Int(valResult.token)] += 1
                totalPairs += 1
            }
            
            print("[\(name)] pairs=\(encoder.pairs.count)")
        }
        
        print("\n=== DPCM Token Distribution (total \(totalPairs) pairs) ===")
        
        // Output run token distribution
        print("\nRun token frequencies (for staticDPCMRunModel):")
        let runTotal = globalRunCounts.reduce(0, +)
        var runNormalized = [UInt32](repeating: 1, count: 64)
        for i in 0..<64 {
            if 0 < globalRunCounts[i] {
                let normalized = max(1, UInt32(Double(globalRunCounts[i]) / Double(runTotal) * Double(RANS_SCALE)))
                runNormalized[i] = normalized
            }
        }
        print("let staticDPCMRunModel = buildStaticModel(rawFreqs: [")
        for row in 0..<8 {
            let start = row * 8
            let vals = (start..<start+8).map { String(format: "%5d", runNormalized[$0]) }
            print("    \(vals.joined(separator: ",")),")
        }
        print("])")
        
        // Output val token distribution
        print("\nValue token frequencies (for staticDPCMValModel):")
        let valTotal = globalValCounts.reduce(0, +)
        var valNormalized = [UInt32](repeating: 1, count: 64)
        for i in 0..<64 {
            if 0 < globalValCounts[i] {
                let normalized = max(1, UInt32(Double(globalValCounts[i]) / Double(valTotal) * Double(RANS_SCALE)))
                valNormalized[i] = normalized
            }
        }
        print("let staticDPCMValModel = buildStaticModel(rawFreqs: [")
        for row in 0..<8 {
            let start = row * 8
            let vals = (start..<start+8).map { String(format: "%5d", valNormalized[$0]) }
            print("    \(vals.joined(separator: ",")),")
        }
        print("])")
        
        // Also print raw counts for debugging
        print("\nRaw run counts:")
        for i in 0..<64 where 0 < globalRunCounts[i] {
            print("  token[\(i)] = \(globalRunCounts[i]) (\(String(format: "%.2f", Double(globalRunCounts[i]) / Double(runTotal) * 100))%)")
        }
        print("\nRaw val counts:")
        for i in 0..<64 where 0 < globalValCounts[i] {
            print("  token[\(i)] = \(globalValCounts[i]) (\(String(format: "%.2f", Double(globalValCounts[i]) / Double(valTotal) * 100))%)")
        }
        
        XCTAssertTrue(0 < totalPairs, "Should have collected token distribution")
    }
}
