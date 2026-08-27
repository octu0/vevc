import Testing
import Foundation
@testable import vevc

struct DPCMLadderTests {

    @Test
    func testScanOrderCoordinateMapping() {
        // 1. ラスタ走査の双方向マッピング検証
        let raster = DPCMScanOrder.raster
        for i in 0..<16 {
            let pt = raster.toXY(index: i)
            let roundtrip = raster.toIndex(x: pt.x, y: pt.y)
            #expect(roundtrip == i)
            #expect(pt.y == i / 4)
            #expect(pt.x == i % 4)
        }

        // 2. 蛇行走査の双方向マッピング検証
        let serpentine = DPCMScanOrder.serpentine
        for i in 0..<16 {
            let pt = serpentine.toXY(index: i)
            let roundtrip = serpentine.toIndex(x: pt.x, y: pt.y)
            #expect(roundtrip == i)
            let y = i / 4
            if (y & 1) == 0 {
                #expect(pt.x == i % 4)
            } else {
                #expect(pt.x == 3 - (i % 4))
            }
        }

        // 3. 蛇行走査の奇数行 (行 1: y=1) の進行方向確認 (右から左)
        let pt4 = serpentine.toXY(index: 4) // y=1, rem=0 -> x=3
        let pt5 = serpentine.toXY(index: 5) // y=1, rem=1 -> x=2
        let pt6 = serpentine.toXY(index: 6) // y=1, rem=2 -> x=1
        let pt7 = serpentine.toXY(index: 7) // y=1, rem=3 -> x=0
        #expect(pt4.x == 3 && pt4.y == 1)
        #expect(pt5.x == 2 && pt5.y == 1)
        #expect(pt6.x == 1 && pt6.y == 1)
        #expect(pt7.x == 0 && pt7.y == 1)
    }

    @Test
    func testDumpRecordBinaryLayout() throws {
        // 1. 固定長 164 バイトのアライメントおよびサイズ検証
        #expect(MemoryLayout<DPCMBlockDumpRecord>.size == 164)
        #expect(MemoryLayout<DPCMBlockDumpRecord>.stride == 164)

        // 2. 一時ファイルへの書き込みとローダーのラウンドトリップ検証
        let tempDir = NSTemporaryDirectory()
        let testPath = (tempDir as NSString).appendingPathComponent("test_dpcm_\(UUID().uuidString).bin")
        defer {
            try? FileManager.default.removeItem(atPath: testPath)
        }

        // ヘッダ 64B + レコード 164B の作成
        var fileData = Data()
        var header = [UInt8](repeating: 0, count: 64)
        header[0] = 0x56; header[1] = 0x44; header[2] = 0x50; header[3] = 0x44
        header[4] = 1
        header[8] = 164
        fileData.append(contentsOf: header)

        var record = DPCMBlockDumpRecord(
            frameIndex: 42,
            plane: 0,
            isAllZero: 0,
            blockX: 10,
            blockY: 20,
            qLow: 8,
            lastVal: 100,
            quantizedValues: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
            dpcmErrors: (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160),
            topBoundary: (100, 101, 102, 103),
            leftBoundary: (104, 105, 106, 107),
            topLeftBoundary: 99,
            mcPred: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
            ransBitCostsQ8: (256, 256, 256, 256, 512, 512, 512, 512, 768, 768, 768, 768, 1024, 1024, 1024, 1024)
        )

        withUnsafeBytes(of: &record) { rawPtr in
            fileData.append(contentsOf: rawPtr)
        }

        try fileData.write(to: URL(fileURLWithPath: testPath))

        let evaluator = DPCMLadderEvaluator()
        try evaluator.loadDumpFile(at: testPath, totalFileBits: 10000.0)

        #expect(evaluator.records.count == 1)
        let loaded = evaluator.records[0]
        #expect(loaded.frameIndex == 42)
        #expect(loaded.blockX == 10)
        #expect(loaded.blockY == 20)
        #expect(loaded.qLow == 8)
        #expect(loaded.lastVal == 100)
        #expect(loaded.quantizedValues.0 == 1)
        #expect(loaded.quantizedValues.15 == 16)
        #expect(loaded.dpcmErrors.0 == 10)
        #expect(loaded.dpcmErrors.15 == 160)
        #expect(loaded.topBoundary.0 == 100)
        #expect(loaded.leftBoundary.0 == 104)
        #expect(loaded.topLeftBoundary == 99)
        #expect(loaded.ransBitCostsQ8.0 == 256)
        #expect(loaded.ransBitCostsQ8.15 == 1024)
    }

    @Test
    func testPredictorHoldZeroErrorOnTransmitted() {
        let hold = DPCMPredictorHold()
        let kValues = [4, 6, 8, 12]
        let trueValues: [Int16] = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]

        for k in kValues {
            var transmitted = [Int16](repeating: 0, count: k)
            for i in 0..<k {
                transmitted[i] = trueValues[i]
            }

            let context = DPCMPredictContext(
                k: k,
                scanOrder: .raster,
                qLow: 8,
                lastVal: 0,
                topBoundary: (0, 0, 0, 0),
                leftBoundary: (0, 0, 0, 0),
                topLeftBoundary: 0,
                mcPred: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            )

            var output = [Int16](repeating: 0, count: 16)
            transmitted.withUnsafeBufferPointer { tPtr in
                output.withUnsafeMutableBufferPointer { oPtr in
                    hold.predictBlock(transmittedK: tPtr.baseAddress!, context: context, output: oPtr.baseAddress!)
                }
            }

            // 送信済み K 要素の誤差ゼロ検証
            for i in 0..<k {
                #expect(output[i] == trueValues[i])
            }
            // 未送信テール要素が transmitted[k - 1] でホールドされていることを検証
            for i in k..<16 {
                #expect(output[i] == transmitted[k - 1])
            }
        }
    }

    @Test
    func testPredictorMEDFlatBlockZeroError() {
        let med = DPCMPredictorMED()
        let flatVal: Int16 = 50
        let trueValues = [Int16](repeating: flatVal, count: 16)
        let transmitted = [Int16](repeating: flatVal, count: 4)

        let context = DPCMPredictContext(
            k: 4,
            scanOrder: .raster,
            qLow: 8,
            lastVal: flatVal,
            topBoundary: (flatVal, flatVal, flatVal, flatVal),
            leftBoundary: (flatVal, flatVal, flatVal, flatVal),
            topLeftBoundary: flatVal,
            mcPred: (flatVal, flatVal, flatVal, flatVal,
                    flatVal, flatVal, flatVal, flatVal,
                    flatVal, flatVal, flatVal, flatVal,
                    flatVal, flatVal, flatVal, flatVal)
        )

        var output = [Int16](repeating: 0, count: 16)
        transmitted.withUnsafeBufferPointer { tPtr in
            output.withUnsafeMutableBufferPointer { oPtr in
                med.predictBlock(transmittedK: tPtr.baseAddress!, context: context, output: oPtr.baseAddress!)
            }
        }

        // 平坦ブロックでは全 16 要素が完全一致することを検証
        for i in 0..<16 {
            #expect(output[i] == trueValues[i])
        }
    }

    @Test
    func testPredictorSNNBatchDeterministicSIMD16() {
        let snn = DPCMPredictorSNNBatch()
        let trueValues: [Int16] = [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
        let transmitted = Array(trueValues[0..<6])

        let context = DPCMPredictContext(
            k: 6,
            scanOrder: .serpentine,
            qLow: 8,
            lastVal: 18,
            topBoundary: (15, 16, 17, 18),
            leftBoundary: (18, 20, 22, 24),
            topLeftBoundary: 15,
            mcPred: (20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50)
        )

        var output1 = [Int16](repeating: 0, count: 16)
        var output2 = [Int16](repeating: 0, count: 16)

        transmitted.withUnsafeBufferPointer { tPtr in
            output1.withUnsafeMutableBufferPointer { o1Ptr in
                snn.predictBlock(transmittedK: tPtr.baseAddress!, context: context, output: o1Ptr.baseAddress!)
            }
            output2.withUnsafeMutableBufferPointer { o2Ptr in
                snn.predictBlock(transmittedK: tPtr.baseAddress!, context: context, output: o2Ptr.baseAddress!)
            }
        }

        // 1. 送信済み K 要素 (蛇行走査 0..<6) の誤差ゼロ検証
        for i in 0..<6 {
            let pt = DPCMScanOrder.serpentine.toXY(index: i)
            #expect(output1[pt.y * 4 + pt.x] == transmitted[i])
        }

        // 2. 決定論的一致 (2 回の推論結果が全 16 要素完全一致) 検証
        for i in 0..<16 {
            #expect(output1[i] == output2[i])
        }
    }

    @Test
    func testPredictorSNNAutoRegDeterministic() {
        let snnAR = DPCMPredictorSNNAutoReg()
        let trueValues: [Int16] = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
        let transmitted = Array(trueValues[0..<4])

        let context = DPCMPredictContext(
            k: 4,
            scanOrder: .raster,
            qLow: 8,
            lastVal: 10,
            topBoundary: (8, 9, 10, 11),
            leftBoundary: (10, 12, 14, 16),
            topLeftBoundary: 8,
            mcPred: (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85)
        )

        var output = [Int16](repeating: 0, count: 16)
        transmitted.withUnsafeBufferPointer { tPtr in
            output.withUnsafeMutableBufferPointer { oPtr in
                snnAR.predictBlock(transmittedK: tPtr.baseAddress!, context: context, output: oPtr.baseAddress!)
            }
        }

        // 送信済み 4 要素の誤差ゼロ検証
        for i in 0..<4 {
            #expect(output[i] == transmitted[i])
        }
    }

    @Test
    func testLadderEvaluatorSyntheticEvaluation() {
        let evaluator = DPCMLadderEvaluator()
        evaluator.setTotalEncodedFileBits(100000.0)

        // 10 個の平坦ブロック (誤差 0 で被覆されるブロック)
        for idx in 0..<10 {
            let record = DPCMBlockDumpRecord(
                frameIndex: UInt32(idx),
                plane: 0,
                isAllZero: 0,
                blockX: UInt16(idx * 4),
                blockY: 0,
                qLow: 8,
                lastVal: 30,
                quantizedValues: (30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30),
                dpcmErrors: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                topBoundary: (30, 30, 30, 30),
                leftBoundary: (30, 30, 30, 30),
                topLeftBoundary: 30,
                mcPred: (30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30),
                ransBitCostsQ8: (256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
            )
            evaluator.addRecord(record)
        }

        let results = evaluator.runFullLadderEvaluation(epsilon: 1)
        // 32 構成空間 (4 K x 2 Scan x 4 Predictors) の検証
        #expect(results.count == 32)

        // 平坦ブロックに対しては全予測器で被覆率 100% となることを検証
        for res in results {
            #expect(res.totalBlocks == 10)
            #expect(res.coveredBlocks == 10)
            #expect(res.blockCoverageRatio == 100.0)
            #expect(res.weightedCoverageRatio == 100.0)
        }
    }

    @Test
    func testRealDumpEvaluationIfAvailable() throws {
        guard let dumpDir = ProcessInfo.processInfo.environment["VEVC_DPCM_EVAL_DIR"] else {
            return
        }
        let dumpPath = (dumpDir as NSString).appendingPathComponent("dpcm_blocks.bin")
        guard FileManager.default.fileExists(atPath: dumpPath) else {
            return
        }

        var totalFileBits = 0.0
        if let bitsStr = ProcessInfo.processInfo.environment["VEVC_DPCM_TOTAL_FILE_BITS"], let bits = Double(bitsStr) {
            totalFileBits = bits
        }

        let evaluator = DPCMLadderEvaluator()
        try evaluator.loadDumpFile(at: dumpPath, totalFileBits: totalFileBits)

        fputs("\n================================================================================\n", stderr)
        fputs("[DPCM Ladder Evaluation] 32 Configurations Matrix (epsilon = 1)\n", stderr)
        fputs("================================================================================\n", stderr)
        fputs(String(format: "Total Loaded Blocks: %ld\n", evaluator.records.count), stderr)
        fputs("--------------------------------------------------------------------------------\n", stderr)
        fputs(String(format: "| %-5@ | %-10@ | %-12@ | %-12@ | %-12@ | %-12@ | %-15@ |\n", "K", "Scan", "Predictor", "Blk Cov (%)", "Wgt Cov (%)", "Reduct (%)", "Gate 1 Status"), stderr)
        fputs("--------------------------------------------------------------------------------\n", stderr)

        let results = evaluator.runFullLadderEvaluation(epsilon: 1)
        var maxReduction = 0.0
        var bestConfig = ""

        for res in results {
            var scanStr = "Raster"
            if res.scanOrder == .serpentine {
                scanStr = "Serpentine"
            }
            var gateStr = "FAIL (< 2.0%)"
            if res.gate1Passed {
                gateStr = "PASS (>= 2.0%)"
            }
            let line = String(
                format: "| K=%-3d | %-10@ | %-12@ | %10.2f%% | %10.2f%% | %10.2f%% | %-15@ |\n",
                res.k, scanStr, res.predictorName, res.blockCoverageRatio, res.weightedCoverageRatio, res.fileReductionPotential, gateStr
            )
            fputs(line, stderr)

            if maxReduction < res.fileReductionPotential {
                maxReduction = res.fileReductionPotential
                bestConfig = String(format: "K=%d, Scan=%@, Predictor=%@", res.k, scanStr, res.predictorName)
            }
        }

        fputs("--------------------------------------------------------------------------------\n", stderr)
        fputs(String(format: "Best Configuration: %@\n", bestConfig), stderr)
        fputs(String(format: "Max File Reduction Potential: %.2f%%\n", maxReduction), stderr)
        var overallStatus = "FAIL (< 2.0%)"
        if 2.0 <= maxReduction {
            overallStatus = "PASS (>= 2.0%)"
        }
        fputs(String(format: "Gate 1 Assessment: %@\n", overallStatus), stderr)
        fputs("================================================================================\n\n", stderr)

        #expect(2.0 <= maxReduction)
    }
}
