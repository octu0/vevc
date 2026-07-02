import XCTest
@testable import vevc

final class ChromaDriftInvestigationTests: XCTestCase {

    // 滑らかなランダムブロック（値域±30）を生成する関数
    private func generateSmoothRandomBlock(size: Int) -> [Int16] {
        var block = [Int16](repeating: 0, count: size * size)
        let base = Double.random(in: -20...20)
        let freqX = Double.random(in: 0.05...0.2)
        let freqY = Double.random(in: 0.05...0.2)
        let phaseX = Double.random(in: 0...2 * .pi)
        let phaseY = Double.random(in: 0...2 * .pi)
        let amp = Double.random(in: 5...10)
        for y in 0..<size {
            for x in 0..<size {
                let val = base + amp * sin(Double(x) * freqX + phaseX) * cos(Double(y) * freqY + phaseY)
                block[y * size + x] = Int16(max(-30, min(30, val.rounded())))
            }
        }
        return block
    }

    // DWT+ゼロ化バイアス測定
    func testDWTAndZeroingBias() {
        let sizes = [8, 16, 32]
        let numBlocks = 5000

        print("\n=== DWT + Zeroing Bias Measurement ===")
        for size in sizes {
            var totalDiff: Double = 0.0

            for _ in 0..<numBlocks {
                let input = generateSmoothRandomBlock(size: size)
                let block = BlockView.allocate(width: size, height: size)
                for y in 0..<size {
                    for x in 0..<size {
                        block.base[y * size + x] = input[y * size + x]
                    }
                }

                // 2D DWT
                switch size {
                case 8:
                    dwt2DBlock8(block)
                case 16:
                    dwt2DBlock16(block)
                case 32:
                    dwt2DBlock32(block)
                default:
                    break
                }

                // HL/LH/HH を全ゼロ化
                let half = size / 2
                let subbands = Subbands(
                    ll: BlockView(base: block.base, width: half, height: half, stride: size),
                    hl: BlockView(base: block.base.advanced(by: half), width: half, height: half, stride: size),
                    lh: BlockView(base: block.base.advanced(by: half * size), width: half, height: half, stride: size),
                    hh: BlockView(base: block.base.advanced(by: half * size + half), width: half, height: half, stride: size),
                    size: half
                )

                for y in 0..<half {
                    for x in 0..<half {
                        subbands.hl.base[y * size + x] = 0
                        subbands.lh.base[y * size + x] = 0
                        subbands.hh.base[y * size + x] = 0
                    }
                }

                // 逆 2D DWT
                switch size {
                case 8:
                    inverseDWT2DBlock8(block)
                case 16:
                    inverseDWT2DBlock16(block)
                case 32:
                    inverseDWT2DBlock32(block)
                default:
                    break
                }

                // 出力と入力の平均差を測定
                var inputSum: Double = 0.0
                var outputSum: Double = 0.0
                for i in 0..<(size * size) {
                    inputSum += Double(input[i])
                    outputSum += Double(block.base[i])
                }
                let inputAvg = inputSum / Double(size * size)
                let outputAvg = outputSum / Double(size * size)
                totalDiff += (outputAvg - inputAvg)

                block.deallocate()
            }

            let avgBias = totalDiff / Double(numBlocks)
            print("Size \(size)x\(size): average bias = \(avgBias)")
            XCTAssertLessThan(abs(avgBias), 0.02, "DWT + Zeroing Bias for size \(size) exceeds 0.02")
        }
    }

    // MC FIR バイアス測定
    func testMCFIRBias() {
        let size = 32
        let chromaBlockSize = 16
        let numIterations = 500

        print("\n=== MC FIR Bias Measurement ===")
        
        for roundOffset in [0, 1] {
            var maxMCBias = 0.0
            
            for fractY in 0...7 {
                for fractX in 0...7 {
                    var totalDiff = 0.0
                    
                    // 定数平面を用いて、丸めによる系統的バイアスを測定する
                    let pad = 8
                    let planeSize = size + pad * 2
                    
                    for c in -30...30 {
                        let prevPlane = [Int16](repeating: Int16(c), count: planeSize * planeSize)
                        var plane = [Int16](repeating: 0, count: planeSize * planeSize)
                        
                        // 動ベクトルを設定。fractX, fractY を指定
                        let mvs = MotionVectors(dx: [Int16(fractX)], dy: [Int16(fractY)])
                        
                        // 動き補償を適用
                        applyScaledChromaMCForTest(
                            plane: &plane,
                            prevPlane: prevPlane,
                            mvs: mvs,
                            width: planeSize,
                            height: planeSize,
                            chromaBlockSize: chromaBlockSize,
                            roundOffset: roundOffset
                        )
                        
                        // 中央の 16x16 ブロックの平均差を測定
                        var planeSum = 0.0
                        var count = 0
                        let startY = pad
                        let endY = pad + chromaBlockSize
                        let startX = pad
                        let endX = pad + chromaBlockSize
                        
                        for y in startY..<endY {
                            for x in startX..<endX {
                                planeSum += Double(plane[y * planeSize + x])
                                count += 1
                            }
                        }
                        
                        let planeAvg = planeSum / Double(count)
                        totalDiff += (planeAvg - Double(c))
                    }
                    let avgBias = totalDiff / 61.0
                    if abs(avgBias) > maxMCBias {
                        maxMCBias = abs(avgBias)
                    }
                }
            }
            print("RoundOffset \(roundOffset): max MC average bias = \(maxMCBias)")
            XCTAssertLessThan(maxMCBias, 0.02, "MC FIR bias exceeds 0.02")
        }
    }
    
    private func applyScaledChromaMCForTest(
        plane: inout [Int16],
        prevPlane: [Int16],
        mvs: MotionVectors,
        width: Int,
        height: Int,
        chromaBlockSize: Int,
        roundOffset: Int
    ) {
        applyScaledMotionCompensationChroma(
            plane: &plane,
            prevPlane: prevPlane,
            mvs: mvs,
            width: width,
            height: height,
            chromaBlockSize: chromaBlockSize,
            mvShift: 0, // 1/8画素単位
            roundOffset: roundOffset
        )
    }
}
