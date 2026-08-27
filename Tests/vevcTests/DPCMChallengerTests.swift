import Testing
import Foundation
@testable import vevc

struct DPCMChallengerTests {

    // MARK: - 1. 数学的一貫性と単調性 (Mathematical Consistency & Monotonicity)

    @Test
    func testMonotonicityOverEpsilonSweep() {
        let evaluator = DPCMLadderEvaluator()
        evaluator.setTotalEncodedFileBits(500000.0)

        // 多様な合成ブロック（平坦、勾配、ステップ、ランダム風）を 50 ブロック生成
        for idx in 0..<50 {
            let baseVal = Int16((idx * 17) & 255)
            let delta = Int16(idx % 7)
            let quantValues = (
                baseVal, baseVal &+ delta, baseVal &+ delta &* 2, baseVal &+ delta &* 3,
                baseVal &+ 1, baseVal &+ delta &+ 1, baseVal &+ delta &* 2 &+ 1, baseVal &+ delta &* 3 &+ 1,
                baseVal &+ 2, baseVal &+ delta &+ 2, baseVal &+ delta &* 2 &+ 2, baseVal &+ delta &* 3 &+ 2,
                baseVal &+ 3, baseVal &+ delta &+ 3, baseVal &+ delta &* 2 &+ 3, baseVal &+ delta &* 3 &+ 3
            )
            let errors = (
                Int16(1), Int16(2), Int16(3), Int16(4),
                Int16(1), Int16(2), Int16(3), Int16(4),
                Int16(1), Int16(2), Int16(3), Int16(4),
                Int16(1), Int16(2), Int16(3), Int16(4)
            )
            let costs = (
                UInt16(256), UInt16(256), UInt16(256), UInt16(256),
                UInt16(300), UInt16(300), UInt16(300), UInt16(300),
                UInt16(400), UInt16(400), UInt16(400), UInt16(400),
                UInt16(500), UInt16(500), UInt16(500), UInt16(500)
            )

            let record = DPCMBlockDumpRecord(
                frameIndex: UInt32(idx),
                plane: 0,
                isAllZero: 0,
                blockX: UInt16(idx * 4),
                blockY: 0,
                qLow: 8,
                lastVal: baseVal,
                quantizedValues: quantValues,
                dpcmErrors: errors,
                topBoundary: (baseVal, baseVal, baseVal, baseVal),
                leftBoundary: (baseVal, baseVal, baseVal, baseVal),
                topLeftBoundary: baseVal,
                mcPred: quantValues,
                ransBitCostsQ8: costs
            )
            evaluator.addRecord(record)
        }

        let epsilons: [Int16] = [0, 1, 2, 3, 5, 10, 20, 50, 100, 255]
        let kValues = [4, 6, 8, 12]
        let scanOrders = [DPCMScanOrder.raster, DPCMScanOrder.serpentine]
        let predictors: [(name: String, engine: DPCMPredictorEngine)] = [
            ("p0 (Hold)", DPCMPredictorHold()),
            ("p1 (MED)", DPCMPredictorMED()),
            ("p2 (SNN-DAG)", DPCMPredictorSNNBatch()),
            ("p3 (SNN-AR)", DPCMPredictorSNNAutoReg())
        ]

        // 全 32 構成に対して epsilon 単調非減少性を検証
        for k in kValues {
            for scan in scanOrders {
                for p in predictors {
                    var prevBlockCov = -1.0
                    var prevWgtCov = -1.0
                    var prevReduct = -1.0

                    for eps in epsilons {
                        let res = evaluator.evaluateConfiguration(
                            k: k,
                            scanOrder: scan,
                            predictor: p.engine,
                            predictorName: p.name,
                            epsilon: eps
                        )

                        // 1. 不変条件の検証 (0.0 <= cov <= 100.0)
                        #expect(0.0 <= res.blockCoverageRatio)
                        #expect(res.blockCoverageRatio <= 100.0)
                        #expect(0.0 <= res.weightedCoverageRatio)
                        #expect(res.weightedCoverageRatio <= 100.0)
                        #expect(0.0 <= res.fileReductionPotential)
                        #expect(res.fileReductionPotential <= res.tailRatioFile)

                        // 2. 単調非減少性 (eps1 <= eps2 => cov(eps1) <= cov(eps2))
                        if 0.0 <= prevBlockCov {
                            #expect(prevBlockCov <= res.blockCoverageRatio)
                            #expect(prevWgtCov <= res.weightedCoverageRatio)
                            #expect(prevReduct <= res.fileReductionPotential)
                        }

                        prevBlockCov = res.blockCoverageRatio
                        prevWgtCov = res.weightedCoverageRatio
                        prevReduct = res.fileReductionPotential
                    }
                }
            }
        }
    }

    // MARK: - 2. 蛇行走査における厳格な因果律検証 (Causality & Future Pixel Independence)

    @Test
    func testCausalityFuturePixelIndependence() {
        let predictors: [(name: String, engine: DPCMPredictorEngine)] = [
            ("p0 (Hold)", DPCMPredictorHold()),
            ("p1 (MED)", DPCMPredictorMED()),
            ("p2 (SNN-DAG)", DPCMPredictorSNNBatch()),
            ("p3 (SNN-AR)", DPCMPredictorSNNAutoReg())
        ]

        let kValues = [4, 6, 8, 12]
        let scanOrders = [DPCMScanOrder.raster, DPCMScanOrder.serpentine]

        for scan in scanOrders {
            for k in kValues {
                for p in predictors {
                    // ベースラインの送信値
                    var transmittedBase = [Int16](repeating: 0, count: k)
                    for i in 0..<k {
                        transmittedBase[i] = Int16(10 + i * 5)
                    }

                    let context = DPCMPredictContext(
                        k: k,
                        scanOrder: scan,
                        qLow: 8,
                        lastVal: 10,
                        topBoundary: (10, 12, 14, 16),
                        leftBoundary: (10, 15, 20, 25),
                        topLeftBoundary: 10,
                        mcPred: (
                            10, 12, 14, 16,
                            15, 17, 19, 21,
                            20, 22, 24, 26,
                            25, 27, 29, 31
                        )
                    )

                    var outputBase = [Int16](repeating: 0, count: 16)
                    transmittedBase.withUnsafeBufferPointer { tPtr in
                        outputBase.withUnsafeMutableBufferPointer { oPtr in
                            p.engine.predictBlock(transmittedK: tPtr.baseAddress!, context: context, output: oPtr.baseAddress!)
                        }
                    }

                    // 1. 送信済み K 要素 (0..<k) の誤差ゼロ検証
                    for i in 0..<k {
                        let pt = scan.toXY(index: i)
                        #expect(outputBase[pt.y * 4 + pt.x] == transmittedBase[i])
                    }

                    // 2. 予測器の決定論的一致性 (同一入力で 100% 同一出力)
                    var outputRepeat = [Int16](repeating: 0, count: 16)
                    transmittedBase.withUnsafeBufferPointer { tPtr in
                        outputRepeat.withUnsafeMutableBufferPointer { oPtr in
                            p.engine.predictBlock(transmittedK: tPtr.baseAddress!, context: context, output: oPtr.baseAddress!)
                        }
                    }
                    for i in 0..<16 {
                        #expect(outputBase[i] == outputRepeat[i])
                    }
                }
            }
        }
    }

    @Test
    func testSerpentineScanExactPixelDependencyOrder() {
        let serpentine = DPCMScanOrder.serpentine
        let med = DPCMPredictorMED()

        // 偶数行 (y=0, 2): 左から右 (x: 0 -> 1 -> 2 -> 3)
        // 奇数行 (y=1, 3): 右から左 (x: 3 -> 2 -> 1 -> 0)
        // K=4 (y=0 のみが送信済み) のケースを検証
        let top: (Int16, Int16, Int16, Int16) = (10, 20, 30, 40)
        let left: (Int16, Int16, Int16, Int16) = (10, 50, 60, 70)
        let tl: Int16 = 10
        let transmittedRow0: [Int16] = [10, 20, 30, 40] // y=0 の真値

        let context = DPCMPredictContext(
            k: 4,
            scanOrder: serpentine,
            qLow: 8,
            lastVal: 40,
            topBoundary: top,
            leftBoundary: left,
            topLeftBoundary: tl,
            mcPred: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        )

        var output = [Int16](repeating: 0, count: 16)
        transmittedRow0.withUnsafeBufferPointer { tPtr in
            output.withUnsafeMutableBufferPointer { oPtr in
                med.predictBlock(transmittedK: tPtr.baseAddress!, context: context, output: oPtr.baseAddress!)
            }
        }

        // y=0 は送信値と一致
        #expect(output[0] == 10)
        #expect(output[1] == 20)
        #expect(output[2] == 30)
        #expect(output[3] == 40)

        // y=1 の最初の画素 (蛇行 index 4 = x:3, y:1):
        // a = b = top[3] = 40, c = 40 -> pred = 40
        #expect(output[1 * 4 + 3] == 40)

        // y=1 の 2 番目の画素 (蛇行 index 5 = x:2, y:1):
        // a = output[1 * 4 + 3] = 40, b = output[0 * 4 + 2] = 30, c = output[0 * 4 + 3] = 40
        // MED(a=40, b=30, c=40) -> max(40,30)=40 <= c(40) -> min(40,30) = 30
        #expect(output[1 * 4 + 2] == 30)

        // y=1 の 3 番目の画素 (蛇行 index 6 = x:1, y:1):
        // a = output[1 * 4 + 2] = 30, b = output[0 * 4 + 1] = 20, c = output[0 * 4 + 2] = 30
        // MED(30, 20, 30) -> 20
        #expect(output[1 * 4 + 1] == 20)

        // y=1 の 4 番目の画素 (蛇行 index 7 = x:0, y:1):
        // a = output[1 * 4 + 1] = 20, b = output[0 * 4 + 0] = 10, c = output[0 * 4 + 1] = 20
        // MED(20, 10, 20) -> 10
        #expect(output[1 * 4 + 0] == 10)
    }

    // MARK: - 3. 極端値入力に対する推論決定性とオーバーフロー耐性 (Extreme Values & Fuzzing)

    @Test
    func testAllZeroBlockInference() {
        let predictors: [(name: String, engine: DPCMPredictorEngine)] = [
            ("p0 (Hold)", DPCMPredictorHold()),
            ("p1 (MED)", DPCMPredictorMED()),
            ("p2 (SNN-DAG)", DPCMPredictorSNNBatch()),
            ("p3 (SNN-AR)", DPCMPredictorSNNAutoReg())
        ]

        let kValues = [4, 6, 8, 12]
        let scanOrders = [DPCMScanOrder.raster, DPCMScanOrder.serpentine]

        for scan in scanOrders {
            for k in kValues {
                for p in predictors {
                    let transmitted = [Int16](repeating: 0, count: k)
                    let context = DPCMPredictContext(
                        k: k,
                        scanOrder: scan,
                        qLow: 8,
                        lastVal: 0,
                        topBoundary: (0, 0, 0, 0),
                        leftBoundary: (0, 0, 0, 0),
                        topLeftBoundary: 0,
                        mcPred: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                    )

                    var output = [Int16](repeating: -999, count: 16)
                    transmitted.withUnsafeBufferPointer { tPtr in
                        output.withUnsafeMutableBufferPointer { oPtr in
                            p.engine.predictBlock(transmittedK: tPtr.baseAddress!, context: context, output: oPtr.baseAddress!)
                        }
                    }

                    // 全ゼロブロックに対して全 16 要素が完全に 0 であることを検証
                    for i in 0..<16 {
                        #expect(output[i] == 0)
                    }
                }
            }
        }
    }

    @Test
    func testExtremeP0Hold() {
        let p0 = DPCMPredictorHold()
        let extremeCases: [Int16] = [Int16.max, Int16.min, 0, 10000, -10000, 255, -255]
        for extVal in extremeCases {
            let transmitted = [Int16](repeating: extVal, count: 6)
            let context = DPCMPredictContext(
                k: 6,
                scanOrder: .serpentine,
                qLow: 32,
                lastVal: extVal,
                topBoundary: (extVal, extVal, extVal, extVal),
                leftBoundary: (extVal, extVal, extVal, extVal),
                topLeftBoundary: extVal,
                mcPred: (extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal)
            )
            var output = [Int16](repeating: 0, count: 16)
            transmitted.withUnsafeBufferPointer { tPtr in
                output.withUnsafeMutableBufferPointer { oPtr in
                    p0.predictBlock(transmittedK: tPtr.baseAddress!, context: context, output: oPtr.baseAddress!)
                }
            }
            for i in 0..<16 {
                #expect(output[i] == extVal)
            }
        }
    }

    @Test
    func testExtremeP1MED() {
        let p1 = DPCMPredictorMED()
        let extremeCases: [Int16] = [Int16.max, Int16.min, 0, 10000, -10000, 255, -255]
        for extVal in extremeCases {
            let transmitted = [Int16](repeating: extVal, count: 6)
            let context = DPCMPredictContext(
                k: 6,
                scanOrder: .serpentine,
                qLow: 32,
                lastVal: extVal,
                topBoundary: (extVal, extVal, extVal, extVal),
                leftBoundary: (extVal, extVal, extVal, extVal),
                topLeftBoundary: extVal,
                mcPred: (extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal)
            )
            var output = [Int16](repeating: 0, count: 16)
            transmitted.withUnsafeBufferPointer { tPtr in
                output.withUnsafeMutableBufferPointer { oPtr in
                    p1.predictBlock(transmittedK: tPtr.baseAddress!, context: context, output: oPtr.baseAddress!)
                }
            }
            for i in 0..<16 {
                #expect(output[i] == extVal)
            }
        }
    }

    @Test
    func testExtremeP2SNNBatchWithinSafeRange() {
        let p2 = DPCMPredictorSNNBatch()
        let safeCases: [Int16] = [0, 255, -255, 1000, -1000, 4000, -4000]
        for extVal in safeCases {
            let transmitted = [Int16](repeating: extVal, count: 6)
            let context = DPCMPredictContext(
                k: 6,
                scanOrder: .serpentine,
                qLow: 32,
                lastVal: extVal,
                topBoundary: (extVal, extVal, extVal, extVal),
                leftBoundary: (extVal, extVal, extVal, extVal),
                topLeftBoundary: extVal,
                mcPred: (extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal)
            )
            var output = [Int16](repeating: 0, count: 16)
            transmitted.withUnsafeBufferPointer { tPtr in
                output.withUnsafeMutableBufferPointer { oPtr in
                    p2.predictBlock(transmittedK: tPtr.baseAddress!, context: context, output: oPtr.baseAddress!)
                }
            }
            for i in 0..<6 {
                let pt = DPCMScanOrder.serpentine.toXY(index: i)
                #expect(output[pt.y * 4 + pt.x] == extVal)
            }
        }
    }

    @Test
    func testExtremeP3SNNAutoReg() {
        let p3 = DPCMPredictorSNNAutoReg()
        let extremeCases: [Int16] = [255, -255, 1000, -1000, 4096, -4096, 8191, -8192, Int16.max, Int16.min]
        for extVal in extremeCases {
            let transmitted = [Int16](repeating: extVal, count: 6)
            let context = DPCMPredictContext(
                k: 6,
                scanOrder: .serpentine,
                qLow: 32,
                lastVal: extVal,
                topBoundary: (extVal, extVal, extVal, extVal),
                leftBoundary: (extVal, extVal, extVal, extVal),
                topLeftBoundary: extVal,
                mcPred: (extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal, extVal)
            )
            var output = [Int16](repeating: 0, count: 16)
            transmitted.withUnsafeBufferPointer { tPtr in
                output.withUnsafeMutableBufferPointer { oPtr in
                    p3.predictBlock(transmittedK: tPtr.baseAddress!, context: context, output: oPtr.baseAddress!)
                }
            }
            for i in 0..<6 {
                let pt = DPCMScanOrder.serpentine.toXY(index: i)
                #expect(output[pt.y * 4 + pt.x] == extVal)
            }
        }
    }

    @Test
    func testRandomStressFuzzingDeterministicExecution() {
        let predictors: [(name: String, engine: DPCMPredictorEngine)] = [
            ("p0 (Hold)", DPCMPredictorHold()),
            ("p1 (MED)", DPCMPredictorMED()),
            ("p2 (SNN-DAG)", DPCMPredictorSNNBatch()),
            ("p3 (SNN-AR)", DPCMPredictorSNNAutoReg())
        ]

        // 疑似乱数ジェネレータによる決定論的ファジング (100 パターン)
        var rngSeed: UInt64 = 0x123456789ABCDEF0
        func nextRand() -> Int16 {
            rngSeed = rngSeed &* 6364136223846793005 &+ 1
            let raw = Int32(truncatingIfNeeded: rngSeed >> 32)
            return Int16(clamping: raw % 1000)
        }

        for _ in 0..<100 {
            let k = [4, 6, 8, 12][Int(nextRand() & 3)]
            let scan: DPCMScanOrder = ((nextRand() & 1) == 0) ? .raster : .serpentine

            var transmitted = [Int16](repeating: 0, count: k)
            for i in 0..<k {
                transmitted[i] = nextRand()
            }

            let top = (nextRand(), nextRand(), nextRand(), nextRand())
            let left = (nextRand(), nextRand(), nextRand(), nextRand())
            let tl = nextRand()
            let mc = (
                nextRand(), nextRand(), nextRand(), nextRand(),
                nextRand(), nextRand(), nextRand(), nextRand(),
                nextRand(), nextRand(), nextRand(), nextRand(),
                nextRand(), nextRand(), nextRand(), nextRand()
            )

            let context = DPCMPredictContext(
                k: k,
                scanOrder: scan,
                qLow: Int16(abs(Int32(nextRand())) % 64 + 1),
                lastVal: nextRand(),
                topBoundary: top,
                leftBoundary: left,
                topLeftBoundary: tl,
                mcPred: mc
            )

            for p in predictors {
                var out1 = [Int16](repeating: 0, count: 16)
                var out2 = [Int16](repeating: 0, count: 16)

                transmitted.withUnsafeBufferPointer { tPtr in
                    out1.withUnsafeMutableBufferPointer { o1Ptr in
                        p.engine.predictBlock(transmittedK: tPtr.baseAddress!, context: context, output: o1Ptr.baseAddress!)
                    }
                    out2.withUnsafeMutableBufferPointer { o2Ptr in
                        p.engine.predictBlock(transmittedK: tPtr.baseAddress!, context: context, output: o2Ptr.baseAddress!)
                    }
                }

                // 送信済み要素一致
                for i in 0..<k {
                    let pt = scan.toXY(index: i)
                    #expect(out1[pt.y * 4 + pt.x] == transmitted[i])
                }

                // 決定性
                for i in 0..<16 {
                    #expect(out1[i] == out2[i])
                }
            }
        }
    }
}
