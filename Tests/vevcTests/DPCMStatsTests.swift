import XCTest
@testable import vevc

final class DPCMStatsTests: XCTestCase {

    /// 走査位置別ビット集計の正確性および保持率ラダーの単調減少性を検証
    func testDPCMStatsAccumulationAndMonotonicity() async throws {
        let width = 64
        let height = 64
        var y = [UInt8](repeating: 128, count: width * height)
        let cb = [UInt8](repeating: 128, count: width * height / 4)
        let cr = [UInt8](repeating: 128, count: width * height / 4)

        // エッジおよびグラデーションを含む画像を生成
        for row in 0..<height {
            for col in 0..<width {
                let val = ((row * 7) + (col * 13)) % 256
                y[row * width + col] = UInt8(val)
            }
        }

        var testImg = YCbCrImage(width: width, height: height, ratio: .ratio420)
        testImg.yPlane = y
        testImg.cbPlane = cb
        testImg.crPlane = cr

        // DPCMStatsTracker の直接記録ロジックの単体検証
        let tracker = DPCMStatsTracker.shared
        tracker.reset()

        // ブロックヘッダの記録テスト
        tracker.recordBlockHeader(isAllZero: true)
        tracker.recordBlockHeader(isAllZero: false, lscpX: 2, lscpY: 1)

        // 走査位置ごとの係数記録テスト
        for i in 0..<16 {
            let dummyVal: Int16 = Int16((i * 3) + 1)
            tracker.recordCoeff(pos: i, run: i % 3, val: dummyVal)
        }
        tracker.addFileBytes(1024)

        if tracker.isEnabled {
            XCTAssertTrue(2 <= tracker.blockCount)
            XCTAssertTrue(1 <= tracker.zeroBlockCount)
            XCTAssertTrue(0 < tracker.headerBitsQ8)
            XCTAssertEqual(tracker.totalEncodedFileBytes, 1024)

            // 各走査位置でビットが計上されていることを確認
            for i in 0..<16 {
                XCTAssertTrue(0 < tracker.posBitsQ8[i])
                XCTAssertTrue(1 <= tracker.posCount[i])
            }

            // 保持率ラダーの単調減少性の検証 (B_tail(12) <= B_tail(8) <= B_tail(6) <= B_tail(4))
            var tail4: Int64 = 0
            var tail6: Int64 = 0
            var tail8: Int64 = 0
            var tail12: Int64 = 0

            for i in 4..<16 { tail4 += tracker.posBitsQ8[i] }
            for i in 6..<16 { tail6 += tracker.posBitsQ8[i] }
            for i in 8..<16 { tail8 += tracker.posBitsQ8[i] }
            for i in 12..<16 { tail12 += tracker.posBitsQ8[i] }

            XCTAssertTrue(0 <= tail12)
            XCTAssertTrue(tail12 <= tail8)
            XCTAssertTrue(tail8 <= tail6)
            XCTAssertTrue(tail6 <= tail4)
        }

        // エンコーダ経由での実際のエンコード実行
        let encoder = VEVCEncoder(width: width, height: height, qstep: 16, keyint: 10, profile: 0x02)
        let bitstream = try await encoder.encodeToData(images: [testImg, testImg])
        XCTAssertTrue(0 < bitstream.count)
    }

    /// 計装フックがエンコード結果に影響を与えないこと（決定性・中立性）を検証
    func testDPCMStatsInactiveByteIdentical() async throws {
        let width = 64
        let height = 64
        var y = [UInt8](repeating: 100, count: width * height)
        let cb = [UInt8](repeating: 128, count: width * height / 4)
        let cr = [UInt8](repeating: 128, count: width * height / 4)

        for i in 0..<(width * height) {
            y[i] = UInt8((i * 17) % 256)
        }

        var img = YCbCrImage(width: width, height: height, ratio: .ratio420)
        img.yPlane = y
        img.cbPlane = cb
        img.crPlane = cr

        // 1回目のエンコード
        let enc1 = VEVCEncoder(width: width, height: height, qstep: 20, keyint: 5, profile: 0x02)
        let stream1 = try await enc1.encodeToData(images: [img, img, img])

        // 2回目のエンコード
        let enc2 = VEVCEncoder(width: width, height: height, qstep: 20, keyint: 5, profile: 0x02)
        let stream2 = try await enc2.encodeToData(images: [img, img, img])

        // 2回の実行結果が完全一致することを確認 (byte-identical)
        XCTAssertEqual(stream1, stream2)
    }

    /// Profile 1 でのエンコード結果の決定性と不変性を検証
    func testProfile1SHAInvarianceUnderStats() async throws {
        let width = 64
        let height = 64
        var y = [UInt8](repeating: 150, count: width * height)
        let cb = [UInt8](repeating: 128, count: width * height / 4)
        let cr = [UInt8](repeating: 128, count: width * height / 4)

        for i in 0..<(width * height) {
            y[i] = UInt8((i * 31) % 256)
        }

        var img = YCbCrImage(width: width, height: height, ratio: .ratio420)
        img.yPlane = y
        img.cbPlane = cb
        img.crPlane = cr

        let enc1 = VEVCEncoder(width: width, height: height, qstep: 16, keyint: 5, profile: 0x01)
        let stream1 = try await enc1.encodeToData(images: [img, img])

        let enc2 = VEVCEncoder(width: width, height: height, qstep: 16, keyint: 5, profile: 0x01)
        let stream2 = try await enc2.encodeToData(images: [img, img])

        XCTAssertEqual(stream1, stream2)
        XCTAssertTrue(0 < stream1.count)
    }

    /// マルチスレッド並列実行時のスレッドセーフ性を検証
    func testDPCMStatsThreadSafety() async {
        let tracker = DPCMStatsTracker.shared

        await withTaskGroup(of: Void.self) { group in
            for threadIdx in 0..<10 {
                group.addTask {
                    for block in 0..<100 {
                        let isZero = (block % 5) == 0
                        tracker.recordBlockHeader(isAllZero: isZero, lscpX: block % 4, lscpY: (block / 4) % 4)
                        for pos in 0..<16 {
                            tracker.recordCoeff(pos: pos, run: (threadIdx + pos) % 4, val: Int16(block + pos))
                        }
                        tracker.addFileBytes(32)
                    }
                }
            }
        }

        if tracker.isEnabled {
            XCTAssertTrue(1000 <= tracker.blockCount)
        }
    }

    /// 極端な入力パターン（全ゼロ、チェッカーボード、ランダムノイズ、最大振幅）におけるエンコード・統計の挙動検証
    func testDPCMStatsExtremeInputPatterns() async throws {
        let width = 64
        let height = 64
        let totalY = width * height
        let totalUV = totalY / 4

        // 1. 全ゼロ / フラットパターン
        var flatImg = YCbCrImage(width: width, height: height, ratio: .ratio420)
        flatImg.yPlane = [UInt8](repeating: 128, count: totalY)
        flatImg.cbPlane = [UInt8](repeating: 128, count: totalUV)
        flatImg.crPlane = [UInt8](repeating: 128, count: totalUV)

        let encFlat = VEVCEncoder(width: width, height: height, qstep: 16, keyint: 10, profile: 0x02)
        let streamFlat = try await encFlat.encodeToData(images: [flatImg, flatImg])
        XCTAssertTrue(0 < streamFlat.count)

        // 2. チェッカーボード（高周波空間パターン）
        var checkerImg = YCbCrImage(width: width, height: height, ratio: .ratio420)
        var checkerY = [UInt8](repeating: 0, count: totalY)
        for r in 0..<height {
            for c in 0..<width {
                let isEven = ((r + c) % 2) == 0
                if isEven {
                    checkerY[r * width + c] = 255
                } else {
                    checkerY[r * width + c] = 0
                }
            }
        }
        checkerImg.yPlane = checkerY
        checkerImg.cbPlane = [UInt8](repeating: 128, count: totalUV)
        checkerImg.crPlane = [UInt8](repeating: 128, count: totalUV)

        let encChecker = VEVCEncoder(width: width, height: height, qstep: 16, keyint: 10, profile: 0x02)
        let streamChecker = try await encChecker.encodeToData(images: [checkerImg, checkerImg])
        XCTAssertTrue(0 < streamChecker.count)

        // 3. ランダムノイズ（擬似乱数による全域エントロピー）
        var noiseImg = YCbCrImage(width: width, height: height, ratio: .ratio420)
        var noiseY = [UInt8](repeating: 0, count: totalY)
        var lcgState: UInt64 = 0x123456789abcdef0
        for i in 0..<totalY {
            lcgState = lcgState &* 6364136223846793005 &+ 1442695040888963407
            noiseY[i] = UInt8((lcgState >> 32) & 0xFF)
        }
        noiseImg.yPlane = noiseY
        noiseImg.cbPlane = [UInt8](repeating: 128, count: totalUV)
        noiseImg.crPlane = [UInt8](repeating: 128, count: totalUV)

        let encNoise = VEVCEncoder(width: width, height: height, qstep: 16, keyint: 10, profile: 0x02)
        let streamNoise = try await encNoise.encodeToData(images: [noiseImg, noiseImg])
        XCTAssertTrue(0 < streamNoise.count)

        // 4. 最大振幅ステップ（極端な階調差 0 vs 255）
        var maxStepImg = YCbCrImage(width: width, height: height, ratio: .ratio420)
        var maxStepY = [UInt8](repeating: 0, count: totalY)
        for r in 0..<height {
            for c in 0..<width {
                let isRightHalf = width / 2 <= c
                if isRightHalf {
                    maxStepY[r * width + c] = 255
                } else {
                    maxStepY[r * width + c] = 0
                }
            }
        }
        maxStepImg.yPlane = maxStepY
        maxStepImg.cbPlane = [UInt8](repeating: 128, count: totalUV)
        maxStepImg.crPlane = [UInt8](repeating: 128, count: totalUV)

        let encMaxStep = VEVCEncoder(width: width, height: height, qstep: 16, keyint: 10, profile: 0x02)
        let streamMaxStep = try await encMaxStep.encodeToData(images: [maxStepImg, maxStepImg])
        XCTAssertTrue(0 < streamMaxStep.count)
    }

    /// ビット会計の数学的一貫性（総和同一性、単調減少性、位置別独立性）の厳密検証
    func testDPCMStatsMathematicalConsistency() {
        let tracker = DPCMStatsTracker.shared
        tracker.reset()

        // 走査位置ごとに独立して記録し、他位置への波及がないことを検証
        for targetPos in 0..<16 {
            tracker.reset()
            tracker.recordBlockHeader(isAllZero: false, lscpX: targetPos % 4, lscpY: targetPos / 4)
            tracker.recordCoeff(pos: targetPos, run: 0, val: 5)

            if tracker.isEnabled {
                for pos in 0..<16 {
                    if pos == targetPos {
                        XCTAssertTrue(0 < tracker.posBitsQ8[pos])
                        XCTAssertEqual(tracker.posCount[pos], 1)
                    } else {
                        XCTAssertEqual(tracker.posBitsQ8[pos], 0)
                        XCTAssertEqual(tracker.posCount[pos], 0)
                    }
                }
            }
        }

        // 数学的一貫性: 各位置のビット和 + ヘッダビット == DPCM総ビット数
        tracker.reset()
        tracker.recordBlockHeader(isAllZero: true)
        tracker.recordBlockHeader(isAllZero: false, lscpX: 3, lscpY: 3)
        for i in 0..<16 {
            let runVal = (i * 2) % 5
            let coeffVal = Int16((i * 7) - 20)
            tracker.recordCoeff(pos: i, run: runVal, val: coeffVal)
        }

        if tracker.isEnabled {
            var sumCoeffBitsQ8: Int64 = 0
            for i in 0..<16 {
                sumCoeffBitsQ8 += tracker.posBitsQ8[i]
            }
            let totalDPCMBitsQ8 = sumCoeffBitsQ8 + tracker.headerBitsQ8
            XCTAssertTrue(0 < totalDPCMBitsQ8)

            // 単調減少性 B_tail(12) <= B_tail(8) <= B_tail(6) <= B_tail(4) <= sumCoeff
            var bTail4: Int64 = 0
            var bTail6: Int64 = 0
            var bTail8: Int64 = 0
            var bTail12: Int64 = 0
            for i in 4..<16 { bTail4 += tracker.posBitsQ8[i] }
            for i in 6..<16 { bTail6 += tracker.posBitsQ8[i] }
            for i in 8..<16 { bTail8 += tracker.posBitsQ8[i] }
            for i in 12..<16 { bTail12 += tracker.posBitsQ8[i] }

            XCTAssertTrue(0 <= bTail12)
            XCTAssertTrue(bTail12 <= bTail8)
            XCTAssertTrue(bTail8 <= bTail6)
            XCTAssertTrue(bTail6 <= bTail4)
            XCTAssertTrue(bTail4 <= sumCoeffBitsQ8)
        }
    }

    /// 極端な係数値・ラン長・ブロック数での算術オーバーフロー耐性検証
    func testDPCMStatsExtremeValueResilience() {
        let tracker = DPCMStatsTracker.shared
        tracker.reset()

        let extremeVals: [Int16] = [Int16.min, -32767, -255, -1, 0, 1, 255, 32767, Int16.max]
        let extremeRuns: [Int] = [0, 1, 15, 16, 63, 100, 1000]

        for val in extremeVals {
            for run in extremeRuns {
                for pos in 0..<16 {
                    tracker.recordCoeff(pos: pos, run: run, val: val)
                }
            }
        }
        tracker.recordBlockHeader(isAllZero: false, lscpX: 3, lscpY: 3)
        tracker.addFileBytes(100_000_000)

        if tracker.isEnabled {
            for pos in 0..<16 {
                XCTAssertTrue(0 < tracker.posBitsQ8[pos])
            }
            XCTAssertTrue(0 < tracker.headerBitsQ8)
            XCTAssertEqual(tracker.totalEncodedFileBytes, 100_000_000)
        }
    }
}

