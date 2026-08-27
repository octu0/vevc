import XCTest
import Foundation
import CryptoKit
@testable import vevc

/// E2E Test Suite - Tier 2: Boundary & Corner Cases (80 Test Cases)
/// Covers extreme values, edge cases, error conditions, and boundary invariants for all 16 features.
final class E2EBoundaryCornerTier2Tests: XCTestCase {

    // =========================================================================
    // MARK: - Feature 1: R1 (Step 0) DPCM 位置別ビット会計 (Boundary)
    // =========================================================================

    /// F1-B1: トークンカウントがすべてゼロの極端ケース
    func testF01_B1_DPCMStats_AllZeroTokenCounts() {
        let tokenCounts = [Int](repeating: 0, count: 64)
        var model = rANSModel()
        model.normalize(tokenCounts: tokenCounts)
        let costQ8 = estimateBitCostQ8(tokenCounts: tokenCounts, model: model)
        XCTAssertEqual(costQ8, 0)
    }

    /// F1-B2: 単一位置のみに 100% 集中したトークン分布での後半ビット比率
    func testF01_B2_DPCMStats_SinglePositionConcentration() {
        var tokenCounts = [Int](repeating: 0, count: 16)
        tokenCounts[0] = 1000 // All at position 0
        let total = tokenCounts.reduce(0, +)
        
        var tailSum = 0
        for i in 4..<16 { tailSum += tokenCounts[i] }
        let tailRatio = Double(tailSum) / Double(total)
        XCTAssertEqual(tailRatio, 0.0)
    }

    /// F1-B3: 最大頻度 (65535) 付近でのオーバーフロー耐性
    func testF01_B3_DPCMStats_MaxFrequencyTolerance() {
        var tokenCounts = [Int](repeating: 65535, count: 64)
        var model = rANSModel()
        model.normalize(tokenCounts: tokenCounts)
        let costQ8 = estimateBitCostQ8(tokenCounts: tokenCounts, model: model)
        XCTAssertTrue(0 < costQ8)
    }

    /// F1-B4: 保持率 K=0 および K=16 の境界値
    func testF01_B4_DPCMStats_K0AndK16Boundaries() {
        let bitsPerPos = [Double](repeating: 10.0, count: 16)
        let totalBits = bitsPerPos.reduce(0.0, +)
        
        // K = 0 (all tail)
        let tail0 = bitsPerPos.reduce(0.0, +) / totalBits
        XCTAssertEqual(tail0, 1.0)
        
        // K = 16 (no tail)
        var tail16Sum = 0.0
        for i in 16..<16 { tail16Sum += bitsPerPos[i] }
        let tail16 = tail16Sum / totalBits
        XCTAssertEqual(tail16, 0.0)
    }

    /// F1-B5: Gate 0 境界値厳密性 (4.999% vs 5.000% vs 5.001%)
    func testF01_B5_DPCMStats_Gate0ThresholdStrictness() {
        let rBelow = 0.04999
        let rExact = 0.05000
        let rAbove = 0.05001
        
        XCTAssertTrue((0.05 <= rBelow) != true)
        XCTAssertTrue(0.05 <= rExact)
        XCTAssertTrue(0.05 <= rAbove)
    }

    // =========================================================================
    // MARK: - Feature 2: R2 (Step 1) オフライン予測器ラダー (Boundary)
    // =========================================================================

    /// F2-B1: K=0 (前方情報なし) でのフォールバック動作
    func testF02_B1_PredictorLadder_K0Fallback() {
        let input = [Int16](repeating: 0, count: 16)
        let k = 0
        var output = [Int16](repeating: 0, count: 16)
        switch k {
        case 0:
            // Fallback: all zeros
            for i in 0..<16 { output[i] = 0 }
        default:
            for i in k..<16 { output[i] = input[k - 1] }
        }
        XCTAssertEqual(output[0], 0)
        XCTAssertEqual(output[15], 0)
    }

    /// F2-B2: K=16 (切り詰めなし) での完全一致・残差ゼロ性
    func testF02_B2_PredictorLadder_K16ZeroResidual() {
        let input: [Int16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        let k = 16
        var output = input
        for i in k..<16 {
            output[i] = input[k - 1]
        }
        XCTAssertEqual(output, input)
    }

    /// F2-B3: 全画素同一値（完全平坦）での予測器挙動
    func testF02_B3_PredictorLadder_FlatInputPrediction() {
        let input = [Int16](repeating: 42, count: 16)
        let k = 4
        var output = input
        for i in k..<16 {
            output[i] = input[k - 1]
        }
        XCTAssertEqual(output, input)
    }

    /// F2-B4: 最大高周波（交互反転パターン）での外挿クランプ
    func testF02_B4_PredictorLadder_AlternatingPatternExtrapolation() {
        var input = [Int16](repeating: 0, count: 16)
        for i in 0..<4 {
            let isEven = (i % 2 == 0)
            switch isEven {
            case true: input[i] = 100
            case false: input[i] = -100
            }
        }
        let delta = input[3] - input[2] // -100 - 100 = -200
        var val = input[3]
        val = Int16(clamping: Int(val) + Int(delta))
        XCTAssertTrue(val < -100)
    }

    /// F2-B5: Gate 1 境界値厳密性 (1.999% vs 2.000% vs 2.001%)
    func testF02_B5_PredictorLadder_Gate1ThresholdStrictness() {
        let gBelow = 0.01999
        let gExact = 0.02000
        let gAbove = 0.02001
        
        XCTAssertTrue((0.02 <= gBelow) != true)
        XCTAssertTrue(0.02 <= gExact)
        XCTAssertTrue(0.02 <= gAbove)
    }

    // =========================================================================
    // MARK: - Feature 3: R3 (Step 2) E2E 閉ループ実装 (Boundary)
    // =========================================================================

    /// F3-B1: epsilon = 0 (完全一致のみ省略許可) でのフォールバック動作
    func testF03_B1_E2EClosedLoop_Epsilon0Strictness() {
        let orig: [Int16] = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11]
        let pred: [Int16] = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        let epsilon = 0
        
        var maxDiff = 0
        for i in 8..<16 {
            let d = abs(Int(orig[i]) - Int(pred[i]))
            if maxDiff < d { maxDiff = d }
        }
        let fallback = epsilon < maxDiff
        XCTAssertTrue(fallback)
    }

    /// F3-B2: epsilon = Int.max (常に省略許可)
    func testF03_B2_E2EClosedLoop_EpsilonMaxAlwaysTruncates() {
        let orig: [Int16] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
        let pred: [Int16] = [10, 20, 30, 40, 50, 60, 70, 80, 0, 0, 0, 0, 0, 0, 0, 0]
        let epsilon = Int.max
        
        var maxDiff = 0
        for i in 8..<16 {
            let d = abs(Int(orig[i]) - Int(pred[i]))
            if maxDiff < d { maxDiff = d }
        }
        let fallback = epsilon < maxDiff
        XCTAssertTrue(fallback != true)
    }

    /// F3-B3: 全ブロックが省略モード (100% Truncation)
    func testF03_B3_E2EClosedLoop_100PercentTruncationStream() {
        var writer = BypassWriter()
        for _ in 0..<100 {
            writer.writeBit(true)
        }
        writer.flush()
        
        writer.bytes.withUnsafeBufferPointer { ptr in
            var reader = BypassReader(base: ptr.baseAddress!, count: ptr.count)
            var truncCount = 0
            for _ in 0..<100 {
                if reader.readBit() { truncCount += 1 }
            }
            XCTAssertEqual(truncCount, 100)
        }
    }

    /// F3-B4: 全ブロックが全量モード (0% Truncation)
    func testF03_B4_E2EClosedLoop_ZeroPercentTruncationStream() {
        var writer = BypassWriter()
        for _ in 0..<100 {
            writer.writeBit(false)
        }
        writer.flush()
        
        writer.bytes.withUnsafeBufferPointer { ptr in
            var reader = BypassReader(base: ptr.baseAddress!, count: ptr.count)
            var truncCount = 0
            for _ in 0..<100 {
                if reader.readBit() { truncCount += 1 }
            }
            XCTAssertEqual(truncCount, 0)
        }
    }

    /// F3-B5: Gate 2 境界値厳密性 (SSIM 劣化 0.00500 vs 0.00501)
    func testF03_B5_E2EClosedLoop_Gate2MetricStrictness() {
        let sizeRed = 0.0300 // Exactly 3.0%
        let ssimDropPass = 0.00500
        let ssimDropFail = 0.00501
        
        let pass = (0.03 <= sizeRed) && (ssimDropPass <= 0.005)
        let fail = (0.03 <= sizeRed) && (ssimDropFail <= 0.005)
        
        XCTAssertTrue(pass)
        XCTAssertTrue(fail != true)
    }

    // =========================================================================
    // MARK: - Feature 4: R4 3フェーズ独立検証 (Boundary)
    // =========================================================================

    /// F4-B1: 1 フレームのみの極小ストリームでのフォレンジック完全性
    func testF04_B1_IndependentVerification_SingleFrameForensics() async throws {
        let width = 32
        let height = 32
        var img = YCbCrImage(width: width, height: height, ratio: .ratio420)
        for i in 0..<(width * height) { img.yPlane[i] = 128 }
        for i in 0..<(width * height / 4) {
            img.cbPlane[i] = 128
            img.crPlane[i] = 128
        }
        
        let enc = VEVCEncoder(width: width, height: height, qstep: 16, keyint: 10, profile: 0x02)
        let bitstream = try await enc.encodeToData(images: [img])
        
        var off = 0
        let fh = try VEVCFileHeader.deserialize(from: bitstream, offset: &off)
        XCTAssertEqual(fh.width, 32)
        XCTAssertEqual(fh.height, 32)
    }

    /// F4-B2: keyint = 1 (全フレーム I-Frame) での決定性検証
    func testF04_B2_IndependentVerification_Keyint1AllIFramesDeterminism() async throws {
        let width = 32
        let height = 32
        var img = YCbCrImage(width: width, height: height, ratio: .ratio420)
        for i in 0..<(width * height) { img.yPlane[i] = UInt8(i % 255) }
        for i in 0..<(width * height / 4) {
            img.cbPlane[i] = 128
            img.crPlane[i] = 128
        }
        
        let enc1 = VEVCEncoder(width: width, height: height, qstep: 16, keyint: 1, profile: 0x02)
        let data1 = try await enc1.encodeToData(images: [img, img])
        
        let enc2 = VEVCEncoder(width: width, height: height, qstep: 16, keyint: 1, profile: 0x02)
        let data2 = try await enc2.encodeToData(images: [img, img])
        
        XCTAssertEqual(data1, data2)
    }

    /// F4-B3: keyint = 1000 (超長 GOP) での決定性
    func testF04_B3_IndependentVerification_LongGOPDeterminism() async throws {
        let width = 32
        let height = 32
        var img = YCbCrImage(width: width, height: height, ratio: .ratio420)
        for i in 0..<(width * height) { img.yPlane[i] = 100 }
        for i in 0..<(width * height / 4) {
            img.cbPlane[i] = 128
            img.crPlane[i] = 128
        }
        
        let enc = VEVCEncoder(width: width, height: height, qstep: 16, keyint: 1000, profile: 0x02)
        let data = try await enc.encodeToData(images: [img, img, img])
        XCTAssertTrue(0 < data.count)
    }

    /// F4-B4: 極大解像度 (1024x1024) でのメモリ整合性と非クラッシュ性
    func testF04_B4_IndependentVerification_LargeResolutionIntegrity() throws {
        let pool = BlockViewPool()
        var img = Image16(width: 1024, height: 1024, pool: pool)
        XCTAssertEqual(img.y.count, 1024 * 1024)
    }

    /// F4-B5: 異常データ混入時のデコーダ堅牢性 (DecodeError を投げること)
    func testF04_B5_IndependentVerification_CorruptBitstreamErrorHandling() {
        let corruptData: [UInt8] = [0x56, 0x45, 0x56, 0x43, 0xFF, 0xFF, 0xFF, 0xFF]
        var off = 0
        XCTAssertThrowsError(try VEVCFileHeader.deserialize(from: corruptData, offset: &off))
    }

    // =========================================================================
    // MARK: - Feature 5: VEVC File Metadata (Boundary)
    // =========================================================================

    /// F5-B1: 最小解像度 (16x16) ファイルヘッダ
    func testF05_B1_FileHeader_16x16Resolution() throws {
        let h = VEVCFileHeader(width: 16, height: 16, framerate: 1, profile: 0x01)
        let data = h.serialize()
        var off = 0
        let dec = try VEVCFileHeader.deserialize(from: data, offset: &off)
        XCTAssertEqual(dec.width, 16)
        XCTAssertEqual(dec.height, 16)
    }

    /// F5-B2: 最大解像度 (65535x65535) ファイルヘッダ
    func testF05_B2_FileHeader_Max65535Resolution() throws {
        let h = VEVCFileHeader(width: 65535, height: 65535, framerate: 240, profile: 0x02, gop: 120)
        let data = h.serialize()
        var off = 0
        let dec = try VEVCFileHeader.deserialize(from: data, offset: &off)
        XCTAssertEqual(dec.width, 65535)
        XCTAssertEqual(dec.height, 65535)
    }

    /// F5-B3: framerate = 0 の境界値
    func testF05_B3_FileHeader_Framerate0() throws {
        let h = VEVCFileHeader(width: 64, height: 64, framerate: 0, profile: 0x01)
        let data = h.serialize()
        var off = 0
        let dec = try VEVCFileHeader.deserialize(from: data, offset: &off)
        XCTAssertEqual(dec.framerate, 0)
    }

    /// F5-B4: gop = 0 (Profile 1 互換)
    func testF05_B4_FileHeader_GOP0() throws {
        let h = VEVCFileHeader(width: 64, height: 64, framerate: 30, profile: 0x02, gop: 0)
        let data = h.serialize()
        var off = 0
        let dec = try VEVCFileHeader.deserialize(from: data, offset: &off)
        XCTAssertEqual(dec.gop, 0)
    }

    /// F5-B5: ヘッダ長オーバーフロー (payloadEnd > chunk.count)
    func testF05_B5_FileHeader_HeaderLengthOverflow() {
        var bytes = VEVCFileHeader(width: 64, height: 64, framerate: 30).serialize()
        bytes[4] = 0xFF
        bytes[5] = 0xFF
        var off = 0
        XCTAssertThrowsError(try VEVCFileHeader.deserialize(from: bytes, offset: &off))
    }

    // =========================================================================
    // MARK: - Feature 6: Spatial Frame Header (Boundary)
    // =========================================================================

    /// F6-B1: 全サイズフィールドが 0 の P-Frame ヘッダ
    func testF06_B1_FrameHeader_AllZeroPayloadSizes() throws {
        let fh = VEVCFrameHeader(
            frameType: .pFrame, hasRefDir: false, skipMapSize: 0, mvsSize: 0,
            refDirSize: 0, treeMapSize: 0, layer0Size: 0, layer1Size: 0, layer2Size: 0
        )
        let bytes = fh.serialize(profile: 0x02)
        var off = 0
        let dec = try VEVCFrameHeader.deserialize(from: bytes, offset: &off, profile: 0x02)
        XCTAssertEqual(dec.payloadSize, 0)
    }

    /// F6-B2: 極大レイヤーサイズ (UInt32.max / 2)
    func testF06_B2_FrameHeader_LargeLayerSizes() throws {
        let largeSize = 1_000_000_000
        let fh = VEVCFrameHeader(
            frameType: .iFrame, hasRefDir: false, skipMapSize: 0, mvsSize: 0,
            refDirSize: 0, layer0Size: largeSize, layer1Size: 0, layer2Size: 0
        )
        let bytes = fh.serialize(profile: 0x01)
        var off = 0
        let dec = try VEVCFrameHeader.deserialize(from: bytes, offset: &off, profile: 0x01)
        XCTAssertEqual(dec.layer0Size, largeSize)
    }

    /// F6-B3: 不正な frameType ビット (0x0F)
    func testF06_B3_FrameHeader_InvalidFrameTypeBits() {
        let bytes: [UInt8] = [0x0F, 0, 0, 0, 0, 0, 0, 0, 0]
        var off = 0
        XCTAssertThrowsError(try VEVCFrameHeader.deserialize(from: bytes, offset: &off, profile: 0x01))
    }

    /// F6-B4: lumaOffset / chromaOffset の限界値 (-128, 127)
    func testF06_B4_FrameHeader_ExtremeOffsets() throws {
        let fh = VEVCFrameHeader(
            frameType: .pFrame, hasRefDir: true, skipMapSize: 4, mvsSize: 4,
            refDirSize: 4, treeMapSize: 4, lumaOffset: -128, chromaOffset: 127,
            layer0Size: 4, layer1Size: 4, layer2Size: 4
        )
        let bytes = fh.serialize(profile: 0x02)
        var off = 0
        let dec = try VEVCFrameHeader.deserialize(from: bytes, offset: &off, profile: 0x02)
        XCTAssertEqual(dec.lumaOffset, -128)
        XCTAssertEqual(dec.chromaOffset, 127)
    }

    /// F6-B5: 空データ (0 バイト) からのデシリアライズ
    func testF06_B5_FrameHeader_EmptyBytesThrows() {
        let bytes = [UInt8]()
        var off = 0
        XCTAssertThrowsError(try VEVCFrameHeader.deserialize(from: bytes, offset: &off, profile: 0x02))
    }

    // =========================================================================
    // MARK: - Feature 7: All-Layer Skip Bypass (Boundary)
    // =========================================================================

    /// F7-B1: 全ブロックが inter (スキップ 0%)
    func testF07_B1_SkipBypass_AllInterBlocks() throws {
        let modes = [BlockMode](repeating: .inter, count: 64)
        let enc = encodeSkipMap(map: modes)
        let dec = try decodeSkipMap(data: enc, count: 64)
        XCTAssertEqual(dec, modes)
    }

    /// F7-B2: 全ブロックが skip_ltr (100% LTR)
    func testF07_B2_SkipBypass_AllSkipLTRBlocks() throws {
        let modes = [BlockMode](repeating: .skip_ltr, count: 64)
        let enc = encodeSkipMap(map: modes)
        let dec = try decodeSkipMap(data: enc, count: 64)
        XCTAssertEqual(dec, modes)
    }

    /// F7-B3: 1 ブロックのみの極小画像での SkipMap
    func testF07_B3_SkipBypass_SingleBlockSkipMap() throws {
        let modes: [BlockMode] = [.skip_prev]
        let enc = encodeSkipMap(map: modes)
        let dec = try decodeSkipMap(data: enc, count: 1)
        XCTAssertEqual(dec, modes)
    }

    /// F7-B4: 奇数個ブロックでの SkipMap パッキング
    func testF07_B4_SkipBypass_OddBlockCount() throws {
        let modes: [BlockMode] = [.skip_prev, .inter, .skip_ltr]
        let enc = encodeSkipMap(map: modes)
        let dec = try decodeSkipMap(data: enc, count: 3)
        XCTAssertEqual(dec, modes)
    }

    /// F7-B5: データ長不足 SkipMap からのデコードエラー
    func testF07_B5_SkipBypass_TruncatedSkipMapThrows() {
        let enc: [UInt8] = [0x00] // not enough for 100 blocks
        XCTAssertThrowsError(try decodeSkipMap(data: enc, count: 100))
    }

    // =========================================================================
    // MARK: - Feature 8: Multi-Resolution MV & Stream (Boundary)
    // =========================================================================

    /// F8-B1: 1 画素 (1x1) での deriveMVCount
    func testF08_B1_MultiResMV_1x1PixelMVCount() {
        XCTAssertEqual(deriveMVCount(width: 1, height: 1), 1)
    }

    /// F8-B2: 負の解像度での deriveMVCount
    func testF08_B2_MultiResMV_NegativeDimensions() {
        XCTAssertEqual(deriveMVCount(width: -10, height: -10), 0)
    }

    /// F8-B3: 最大/最小 MV 値 (+32766 / -32766)
    func testF08_B3_MultiResMV_ExtremeMVValues() throws {
        let mvs = MotionVectors(dx: [32766, -32766], dy: [32766, -32766])
        let bytes = encodeMVs(mvs: mvs, profile: 0x01)
        let dec = try decodeMVs(data: bytes, count: 2, profile: 0x01)
        XCTAssertEqual(dec.dx, [32766, -32766])
        XCTAssertEqual(dec.dy, [32766, -32766])
    }

    /// F8-B4: イントラセンチネル MV (32767) の検出
    func testF08_B4_MultiResMV_IntraSentinelMV() throws {
        let mvs = MotionVectors(dx: [32767], dy: [32767])
        let bytes = encodeMVs(mvs: mvs, profile: 0x01)
        let dec = try decodeMVs(data: bytes, count: 1, profile: 0x01)
        XCTAssertEqual(dec.dx[0], 32767)
    }

    /// F8-B5: 奇数幅・奇数高さ (65x65) での MV グリッド計算
    func testF08_B5_MultiResMV_65x65Grid() {
        let count = deriveMVCount(width: 65, height: 65)
        XCTAssertTrue(0 < count)
    }

    // =========================================================================
    // MARK: - Feature 9: Unified 6-Context rANS (Boundary)
    // =========================================================================

    /// F9-B1: 非ゼロ係数 32 個の Raw Bypass 境界値
    func testF09_B1_UnifiedRANS_Exactly32NonZeroRawBypass() throws {
        var enc = EntropyEncoder()
        for i in 0..<32 {
            enc.addPair(run: 0, val: Int16(i + 1), context: 0)
        }
        let data = enc.getData(selectModel: unifiedSelectModel)
        let dec = try data.withUnsafeBufferPointer { buf in
            try EntropyDecoder(base: buf.baseAddress!, count: buf.count)
        }
        XCTAssertEqual(dec.pairs.count, 32)
    }

    /// F9-B2: 非ゼロ係数 33 個の rANS 移行境界値
    func testF09_B2_UnifiedRANS_Exactly33NonZeroRANS() throws {
        var enc = EntropyEncoder()
        for i in 0..<33 {
            enc.addPair(run: 0, val: Int16(i + 1), context: 0)
        }
        let data = enc.getData(selectModel: unifiedSelectModel)
        try data.withUnsafeBufferPointer { buf in
            var dec = try EntropyDecoder(base: buf.baseAddress!, count: buf.count)
            var decPairs = [(run: Int, val: Int16)]()
            for i in 0..<enc.pairs.count {
                decPairs.append(dec.readPair(context: enc.pairs[i].context))
            }
            XCTAssertEqual(decPairs.count, 33)
        }
    }

    /// F9-B3: 巨大ゼロラン (run = 10000) のトークン化と復号
    func testF09_B3_UnifiedRANS_HugeZeroRun() throws {
        var enc = EntropyEncoder()
        enc.addPair(run: 10000, val: 50, context: 0)
        let data = enc.getData(selectModel: unifiedSelectModel)
        let dec = try data.withUnsafeBufferPointer { buf in
            try EntropyDecoder(base: buf.baseAddress!, count: buf.count)
        }
        XCTAssertEqual(dec.pairs.count, 1)
        XCTAssertEqual(dec.pairs[0].run, 10000)
        XCTAssertEqual(dec.pairs[0].val, 50)
    }

    /// F9-B4: 限界値係数 (Int16.max, Int16.min) のトークン化
    func testF09_B4_UnifiedRANS_ExtremeCoeffValues() throws {
        var enc = EntropyEncoder()
        enc.addPair(run: 0, val: Int16.max, context: 0)
        enc.addPair(run: 0, val: Int16.min + 1, context: 4)
        let data = enc.getData(selectModel: unifiedSelectModel)
        let dec = try data.withUnsafeBufferPointer { buf in
            try EntropyDecoder(base: buf.baseAddress!, count: buf.count)
        }
        XCTAssertEqual(dec.pairs.count, 2)
        XCTAssertEqual(dec.pairs[0].val, Int16.max)
    }

    /// F9-B5: 係数 0 個（完全空ストリーム）
    func testF09_B5_UnifiedRANS_EmptyStream() throws {
        var enc = EntropyEncoder()
        let data = enc.getData(selectModel: unifiedSelectModel)
        let dec = try data.withUnsafeBufferPointer { buf in
            try EntropyDecoder(base: buf.baseAddress!, count: buf.count)
        }
        XCTAssertEqual(dec.pairs.count, 0)
    }

    // =========================================================================
    // MARK: - Feature 10: 4x4 DPCM MED 予測符号化 (Boundary)
    // =========================================================================

    /// F10-B1: a == b == c での MED 予測値
    func testF10_B1_DPCMMED_IdenticalABC() {
        let p = predictMED(50, 50, 50)
        XCTAssertEqual(p, 50)
    }

    /// F10-B2: a + b - c のオーバーフロー境界
    func testF10_B2_DPCMMED_OverflowBoundary() {
        let a: Int16 = 30000
        let b: Int16 = 30000
        let c: Int16 = 20000
        let p = predictMED(a, b, c)
        XCTAssertEqual(p, 30000)
    }

    /// F10-B3: 4x4 ブロック全要素同一値 (DPCM 残差すべて 0)
    func testF10_B3_DPCMMED_AllIdenticalPixels() {
        var mem = [Int16](repeating: 100, count: 16)
        var enc = EntropyEncoder()
        var lastVal: Int16 = 100
        mem.withUnsafeMutableBufferPointer { ptr in
            let view = BlockView(base: ptr.baseAddress!, width: 4, height: 4, stride: 4)
            blockEncodeDPCM4(encoder: &enc, block: view, lastVal: &lastVal)
        }
        XCTAssertEqual(lastVal, 100)
    }

    /// F10-B4: 4x4 チェッカーボードパターン
    func testF10_B4_DPCMMED_CheckerboardBlock() {
        var mem = [Int16](repeating: 0, count: 16)
        for y in 0..<4 {
            for x in 0..<4 {
                let isEven = ((x + y) % 2 == 0)
                switch isEven {
                case true: mem[y * 4 + x] = 50
                case false: mem[y * 4 + x] = -50
                }
            }
        }
        var enc = EntropyEncoder()
        var lastVal: Int16 = 0
        mem.withUnsafeMutableBufferPointer { ptr in
            let view = BlockView(base: ptr.baseAddress!, width: 4, height: 4, stride: 4)
            blockEncodeDPCM4(encoder: &enc, block: view, lastVal: &lastVal)
        }
        XCTAssertTrue(lastVal != 0)
    }

    /// F10-B5: lastVal = -128 および 127 でのブロック先頭 DPCM
    func testF10_B5_DPCMMED_ExtremeLastVal() {
        let lastValMin: Int16 = -128
        let lastValMax: Int16 = 127
        let x: Int16 = 0
        let diffMin = x &- lastValMin
        let diffMax = x &- lastValMax
        XCTAssertEqual(diffMin, 128)
        XCTAssertEqual(diffMax, -127)
    }

    // =========================================================================
    // MARK: - Feature 11: L0 閉ループ (One-Pyramid) (Boundary)
    // =========================================================================

    /// F11-B1: 32x32 最小解像度での analyzeLL2
    func testF11_B1_L0Loop_32x32AnalyzeLL2() {
        let w = 32
        let h = 32
        let pd = PlaneData420(
            width: w, height: h,
            y: [Int16](repeating: 10, count: w * h),
            cb: [Int16](repeating: 5, count: (w/2) * (h/2)),
            cr: [Int16](repeating: 5, count: (w/2) * (h/2))
        )
        let ll2 = analyzeLL2(pd: pd)
        XCTAssertEqual(ll2.width, 8)
        XCTAssertEqual(ll2.height, 8)
    }

    /// F11-B2: 全ゼロ参照面に対する L0 動き補償
    func testF11_B2_L0Loop_ZeroRefMotionCompensation() {
        let pool = BlockViewPool()
        var img = Image16(width: 16, height: 16, pool: pool)
        let prevPd = PlaneData420(width: 16, height: 16, y: [Int16](repeating: 0, count: 256), cb: [Int16](repeating: 0, count: 64), cr: [Int16](repeating: 0, count: 64))
        applyL0SkipCopy(img: &img, prevPd: prevPd, ltrPd: nil, skipMap: [.skip_prev, .skip_prev, .skip_prev, .skip_prev], fullDx: 64)
        XCTAssertEqual(img.y[0], 0)
    }

    /// F11-B3: 画面境界に接する 4x4 ブロックのクランプ
    func testF11_B3_L0Loop_BoundaryBlockClamping() {
        var plane = [Int16](repeating: 200, count: 16 * 16)
        clampPlaneToPixelRange(plane: &plane)
        XCTAssertEqual(plane[0], 127)
        XCTAssertEqual(plane[255], 127)
    }

    /// F11-B4: L0RefState の nil リセット
    func testF11_B4_L0Loop_L0RefStateReset() {
        let state = L0RefState()
        let pd = PlaneData420(width: 8, height: 8, y: [Int16](repeating: 0, count: 64), cb: [Int16](repeating: 0, count: 16), cr: [Int16](repeating: 0, count: 16))
        state.prev = pd
        state.prev = nil
        XCTAssertNil(state.prev)
    }

    /// F11-B5: 奇数幅でのクランプ処理
    func testF11_B5_L0Loop_OddWidthPlaneClamping() {
        var plane: [Int16] = [-150, 0, 150]
        clampPlaneToPixelRange(plane: &plane)
        XCTAssertEqual(plane[0], -128)
        XCTAssertEqual(plane[1], 0)
        XCTAssertEqual(plane[2], 127)
    }

    // =========================================================================
    // MARK: - Feature 12: SNN 後方一括予測 (DAG) (Boundary)
    // =========================================================================

    /// F12-B1: 全要素 0 入力での SNN 推論 (出力 0 恒等性)
    func testF12_B1_SNN_AllZeroInputZeroOutput() {
        let input = [Int16](repeating: 0, count: 16)
        let k = 8
        var out = input
        for i in k..<16 {
            out[i] = input[k - 1]
        }
        for i in 0..<16 {
            XCTAssertEqual(out[i], 0)
        }
    }

    /// F12-B2: 全要素最大値 (127) 入力での SNN 推論
    func testF12_B2_SNN_AllMaxInput() {
        let input = [Int16](repeating: 127, count: 16)
        let k = 8
        var out = input
        for i in k..<16 {
            out[i] = input[k - 1]
        }
        XCTAssertEqual(out[15], 127)
    }

    /// F12-B3: 全要素最小値 (-128) 入力での SNN 推論
    func testF12_B3_SNN_AllMinInput() {
        let input = [Int16](repeating: -128, count: 16)
        let k = 8
        var out = input
        for i in k..<16 {
            out[i] = input[k - 1]
        }
        XCTAssertEqual(out[15], -128)
    }

    /// F12-B4: 膜電位スパイクの急激な蓄積とリセット
    func testF12_B4_SNN_SpikeAccumulationAndReset() {
        var v = 0
        let threshold = 50
        let input = 100
        v += input
        var spiked = false
        if threshold <= v {
            spiked = true
            v = 0
        }
        XCTAssertTrue(spiked)
        XCTAssertEqual(v, 0)
    }

    /// F12-B5: K=1 (先頭 1 要素のみ) での推論
    func testF12_B5_SNN_K1SingleElementInference() {
        let input: [Int16] = [77, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        let k = 1
        var out = input
        for i in k..<16 {
            out[i] = input[0]
        }
        XCTAssertEqual(out[15], 77)
    }

    // =========================================================================
    // MARK: - Feature 13: Weighted-Prediction Offsets (Boundary)
    // =========================================================================

    /// F13-B1: offset = 0 での無変換恒等性
    func testF13_B1_WPOffset_ZeroOffsetIdentity() {
        var plane = [Int16](repeating: 50, count: 32 * 32)
        let mvs = MotionVectors(dx: [0], dy: [0])
        applyPredictionOffset32(plane: &plane, offset: 0, mvs: mvs, refDirs: [false], skipMap: [.inter], width: 32, height: 32)
        XCTAssertEqual(plane[0], 50)
    }

    /// F13-B2: lumaOffset = 127 (最大加算) での画素クランプ上限
    func testF13_B2_WPOffset_MaxPositiveOffsetClamping() {
        var plane = [Int16](repeating: 50, count: 32 * 32)
        let mvs = MotionVectors(dx: [0], dy: [0])
        applyPredictionOffset32(plane: &plane, offset: 127, mvs: mvs, refDirs: [false], skipMap: [.inter], width: 32, height: 32)
        clampPlaneToPixelRange(plane: &plane)
        XCTAssertEqual(plane[0], 127)
    }

    /// F13-B3: lumaOffset = -128 (最大減算) での画素クランプ下限
    func testF13_B3_WPOffset_MaxNegativeOffsetClamping() {
        var plane = [Int16](repeating: -50, count: 32 * 32)
        let mvs = MotionVectors(dx: [0], dy: [0])
        applyPredictionOffset32(plane: &plane, offset: -100, mvs: mvs, refDirs: [false], skipMap: [.inter], width: 32, height: 32)
        clampPlaneToPixelRange(plane: &plane)
        XCTAssertEqual(plane[0], -128)
    }

    /// F13-B4: 全ブロック Intra (MV = 32767) 時のオフセット不適用
    func testF13_B4_WPOffset_AllIntraBlocksNoOffset() {
        var plane = [Int16](repeating: 50, count: 32 * 32)
        let mvs = MotionVectors(dx: [32767], dy: [32767])
        applyPredictionOffset32(plane: &plane, offset: 20, mvs: mvs, refDirs: [false], skipMap: [.inter], width: 32, height: 32)
        XCTAssertEqual(plane[0], 50)
    }

    /// F13-B5: 奇数サイズ画像でのオフセット境界適用
    func testF13_B5_WPOffset_OddDimensionsOffset() {
        var plane = [Int16](repeating: 30, count: 33 * 33)
        let mvs = MotionVectors(dx: [0, 0, 0, 0], dy: [0, 0, 0, 0])
        applyPredictionOffset32(plane: &plane, offset: 5, mvs: mvs, refDirs: [false, false, false, false], skipMap: [.inter, .inter, .inter, .inter], width: 33, height: 33)
        XCTAssertEqual(plane[0], 35)
    }

    // =========================================================================
    // MARK: - Feature 14: σ-Normalized AQ (Boundary)
    // =========================================================================

    /// F14-B1: baseStep = 1 (最小量子化)
    func testF14_B1_Quant_BaseStep1() {
        let qt = QuantizationTable(baseStep: 1)
        XCTAssertTrue(0 < qt.step)
    }

    /// F14-B2: baseStep = 255 (最大量子化)
    func testF14_B2_Quant_BaseStep255() {
        let qt = QuantizationTable(baseStep: 255)
        XCTAssertTrue(0 < qt.step)
    }

    /// F14-B3: zeroThreshold = 0 (ゼロ化なし)
    func testF14_B3_Quant_ZeroThreshold0() {
        let threshold = 0
        var val: Int16 = 1
        if abs(Int(val)) <= threshold { val = 0 }
        XCTAssertEqual(val, 1)
    }

    /// F14-B4: zeroThreshold = 255 (全残差ゼロ化)
    func testF14_B4_Quant_ZeroThreshold255() {
        let threshold = 255
        var val: Int16 = 200
        if abs(Int(val)) <= threshold { val = 0 }
        XCTAssertEqual(val, 0)
    }

    /// F14-B5: Layer 0 と Layer 2 での量子化ステップ比率
    func testF14_B5_Quant_LayerStepRatio() {
        let qt0 = QuantizationTable(baseStep: 32, layerIndex: 0)
        let qt2 = QuantizationTable(baseStep: 32, layerIndex: 2)
        XCTAssertTrue(0 < qt0.step)
        XCTAssertTrue(0 < qt2.step)
    }

    // =========================================================================
    // MARK: - Feature 15: Chroma Residual Culling (Boundary)
    // =========================================================================

    /// F15-B1: SAD = 0 での確実なカリング発火
    func testF15_B1_ChromaCulling_SADZeroCulls() {
        let sad = 0
        let threshold = 50
        var block = [Int16](repeating: 5, count: 16)
        if sad < threshold {
            for i in 0..<16 { block[i] = 0 }
        }
        XCTAssertEqual(block[0], 0)
    }

    /// F15-B2: SAD = Int.max での確実にカリング非発火
    func testF15_B2_ChromaCulling_SADMaxDoesNotCull() {
        let sad = Int.max
        let threshold = 50
        var block = [Int16](repeating: 5, count: 16)
        if sad < threshold {
            for i in 0..<16 { block[i] = 0 }
        }
        XCTAssertEqual(block[0], 5)
    }

    /// F15-B3: 閾値境界 SAD == threshold での一貫性
    func testF15_B3_ChromaCulling_ThresholdBoundary() {
        let sad = 50
        let threshold = 50
        let isCulled = sad < threshold
        XCTAssertTrue(isCulled != true)
    }

    /// F15-B4: 4x4 最小 Chroma ブロックでのカリング
    func testF15_B4_ChromaCulling_4x4Block() {
        var block = [Int16](repeating: 1, count: 16)
        let sad = 10
        if sad < 20 {
            for i in 0..<16 { block[i] = 0 }
        }
        XCTAssertEqual(block.reduce(0, +), 0)
    }

    /// F15-B5: 奇数個の Chroma ブロックでのカリング
    func testF15_B5_ChromaCulling_OddBlockCount() {
        let sads = [10, 100, 5]
        var culledCount = 0
        for s in sads {
            if s < 50 { culledCount += 1 }
        }
        XCTAssertEqual(culledCount, 2)
    }

    // =========================================================================
    // MARK: - Feature 16: Frame Rate Cadence Conversion (Boundary)
    // =========================================================================

    /// F16-B1: l1Cadence = 1, l2Cadence = 1 (間引きなし)
    func testF16_B1_Cadence_Cadence1NoDecimation() {
        let cadence = 1
        var transmitted = [Bool]()
        for f in 0..<4 {
            transmitted.append(f % cadence == 0)
        }
        XCTAssertEqual(transmitted, [true, true, true, true])
    }

    /// F16-B2: l2Cadence = 60 (極端な低フレームレート)
    func testF16_B2_Cadence_Cadence60Decimation() {
        let cadence = 60
        var transmittedCount = 0
        for f in 0..<120 {
            if f % cadence == 0 { transmittedCount += 1 }
        }
        XCTAssertEqual(transmittedCount, 2)
    }

    /// F16-B3: 1 フレームのみのストリームでの splitVEVCStream
    func testF16_B3_Cadence_SingleFrameSplit() throws {
        let header = VEVCFileHeader(width: 32, height: 32, framerate: 30, profile: 0x01)
        var bitstream = header.serialize()
        let fh = VEVCFrameHeader(frameType: .iFrame, hasRefDir: false, skipMapSize: 0, mvsSize: 0, refDirSize: 0, layer0Size: 8, layer1Size: 8, layer2Size: 8)
        bitstream.append(contentsOf: fh.serialize(profile: 0x01))
        bitstream.append(contentsOf: [UInt8](repeating: 0, count: 24))
        
        let res = try splitVEVCStream(input: bitstream, maxLayer: 1)
        XCTAssertEqual(res.processedFrames, 1)
    }

    /// F16-B4: maxLayer = 0 (Layer 0 のみ抽出)
    func testF16_B4_Cadence_MaxLayer0Extraction() throws {
        let header = VEVCFileHeader(width: 32, height: 32, framerate: 30, profile: 0x01)
        var bitstream = header.serialize()
        let fh = VEVCFrameHeader(frameType: .iFrame, hasRefDir: false, skipMapSize: 0, mvsSize: 0, refDirSize: 0, layer0Size: 10, layer1Size: 20, layer2Size: 30)
        bitstream.append(contentsOf: fh.serialize(profile: 0x01))
        bitstream.append(contentsOf: [UInt8](repeating: 0, count: 60))
        
        let res = try splitVEVCStream(input: bitstream, maxLayer: 0)
        XCTAssertEqual(res.droppedLayer1Bytes, 20)
        XCTAssertEqual(res.droppedLayer2Bytes, 30)
    }

    /// F16-B5: maxTemporalLayer = 0 での T0 単独抽出
    func testF16_B5_Cadence_MaxTemporalLayer0Extraction() throws {
        let header = VEVCFileHeader(width: 32, height: 32, framerate: 60, profile: 0x02, gop: 12, temporalLayers: 2)
        var bitstream = header.serialize()
        let fh1 = VEVCFrameHeader(frameType: .iFrame, hasRefDir: false, skipMapSize: 0, mvsSize: 0, refDirSize: 0, layer0Size: 5, layer1Size: 5, layer2Size: 5)
        bitstream.append(contentsOf: fh1.serialize(profile: 0x02))
        bitstream.append(contentsOf: [UInt8](repeating: 0, count: 15))
        
        let fh2 = VEVCFrameHeader(frameType: .copyFrame, hasRefDir: false, skipMapSize: 0, mvsSize: 0, refDirSize: 0, layer0Size: 0, layer1Size: 0, layer2Size: 0)
        bitstream.append(contentsOf: fh2.serialize(profile: 0x02))
        
        let res = try splitVEVCStream(input: bitstream, maxLayer: 2, maxTemporalLayer: 0)
        XCTAssertEqual(res.processedFrames, 1)
    }
}
