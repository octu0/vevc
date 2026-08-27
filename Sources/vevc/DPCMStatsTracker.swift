import Foundation

/// DPCM 走査位置別ビット配分および切り詰め上限分析用トラッカー
public final class DPCMStatsTracker: @unchecked Sendable {
    public static let shared = DPCMStatsTracker()

    /// 統計計測が有効か否か (環境変数 VEVC_DPCM_STATS)
    public let isEnabled: Bool
    private let lock = NSLock()

    /// 走査位置 i in 0..<16 ごとの積算ビットコスト (Q8 固定小数点: 256 = 1.0 bit)
    public private(set) var posBitsQ8 = [Int64](repeating: 0, count: 16)
    /// 走査位置 i in 0..<16 ごとの非ゼロ係数出現回数
    public private(set) var posCount = [Int64](repeating: 0, count: 16)
    /// 走査位置 i in 0..<16 ごとのゼロラン総数
    public private(set) var posRunTotal = [Int64](repeating: 0, count: 16)
    /// DPCM ヘッダビット積算 (Bypass 1bit + LSCP 座標 Context 5)
    public private(set) var headerBitsQ8: Int64 = 0
    /// 処理した DPCM ブロック総数
    public private(set) var blockCount: Int64 = 0
    /// 全ゼロ DPCM ブロック数
    public private(set) var zeroBlockCount: Int64 = 0
    /// エンコードされた全ファイルの総バイト数
    public private(set) var totalEncodedFileBytes: Int64 = 0

    private init() {
        if let val = getenv("VEVC_DPCM_STATS") {
            let str = String(cString: val)
            switch str {
            case "1", "true", "TRUE":
                self.isEnabled = true
            default:
                self.isEnabled = false
            }
        } else {
            self.isEnabled = false
        }
    }

    /// 統計情報をリセット
    public func reset() {
        guard isEnabled else { return }
        lock.lock()
        defer { lock.unlock() }
        posBitsQ8 = [Int64](repeating: 0, count: 16)
        posCount = [Int64](repeating: 0, count: 16)
        posRunTotal = [Int64](repeating: 0, count: 16)
        headerBitsQ8 = 0
        blockCount = 0
        zeroBlockCount = 0
        totalEncodedFileBytes = 0
    }

    /// DPCM ブロックヘッダ (全ゼロフラグおよび LSCP 座標) のビットコストを記録
    @inline(__always)
    public func recordBlockHeader(isAllZero: Bool, lscpX: Int = 0, lscpY: Int = 0) {
        guard isEnabled else { return }
        lock.lock()
        defer { lock.unlock() }
        blockCount += 1
        if isAllZero {
            zeroBlockCount += 1
            // Bypass bit 0: 1 bit (256 in Q8)
            headerBitsQ8 += 256
        } else {
            // Bypass bit 1: 1 bit (256 in Q8)
            var bitsQ8: Int64 = 256

            // lscpX (Context 5: lscpRunModel / dpcmValModel)
            let lx = valueTokenizeUnsigned(UInt32(lscpX))
            let lxToken = Int(lx.token)
            let lxFreq = Int(StaticRANSModels.shared.lscpRunModel.tokenFreqs[lxToken])
            let lxCostQ8 = log2Q8(Int(rANSScale)) - log2Q8(lxFreq) + (lx.bypassLen << 8)

            // lscpY (Context 5: lscpRunModel / dpcmValModel)
            let ly = valueTokenizeUnsigned(UInt32(lscpY))
            let lyToken = Int(ly.token)
            let lyFreq = Int(StaticRANSModels.shared.lscpRunModel.tokenFreqs[lyToken])
            let lyCostQ8 = log2Q8(Int(rANSScale)) - log2Q8(lyFreq) + (ly.bypassLen << 8)

            // lscp pairVal (=0, Context 5 の valModel は dpcmValModel)
            let v0 = valueTokenize(0)
            let v0Freq = Int(StaticRANSModels.shared.dpcmValModel.tokenFreqs[Int(v0.token)])
            let v0CostQ8 = log2Q8(Int(rANSScale)) - log2Q8(v0Freq) + (v0.bypassLen << 8)

            bitsQ8 += Int64(lxCostQ8 + lyCostQ8 + (v0CostQ8 * 2))
            headerBitsQ8 += bitsQ8
        }
    }

    /// 走査位置 pos における (run, val) 係数ペアのビットコストを記録
    @inline(__always)
    public func recordCoeff(pos: Int, run: Int, val: Int16) {
        guard isEnabled else { return }
        // 係数の正確な bitCostQ8 計算 (Context 4: dpcmRunModel / dpcmValModel)
        let runRes = valueTokenizeUnsigned(UInt32(run))
        let runToken = Int(runRes.token)
        let runFreq = Int(StaticRANSModels.shared.dpcmRunModel.tokenFreqs[runToken])
        let runBitsQ8 = log2Q8(Int(rANSScale)) - log2Q8(runFreq) + (runRes.bypassLen << 8)

        let valRes = valueTokenize(val)
        let valToken = Int(valRes.token)
        let valFreq = Int(StaticRANSModels.shared.dpcmValModel.tokenFreqs[valToken])
        let valBitsQ8 = log2Q8(Int(rANSScale)) - log2Q8(valFreq) + (valRes.bypassLen << 8)

        let totalPairBitsQ8 = Int64(runBitsQ8 + valBitsQ8)

        lock.lock()
        defer { lock.unlock() }
        posBitsQ8[pos] += totalPairBitsQ8
        posCount[pos] += 1
        posRunTotal[pos] += Int64(run)
    }

    /// エンコード済みファイルバイト数を加算
    public func addFileBytes(_ bytes: Int) {
        guard isEnabled else { return }
        lock.lock()
        defer { lock.unlock() }
        totalEncodedFileBytes += Int64(bytes)
    }

    /// 走査位置別統計および Gate 0 判定サマリーを stderr に出力
    public func printSummary() {
        guard isEnabled else { return }
        lock.lock()
        defer { lock.unlock() }

        var totalCoeffBitsQ8: Int64 = 0
        for i in 0..<16 {
            totalCoeffBitsQ8 += posBitsQ8[i]
        }
        let totalDPCMBitsQ8 = totalCoeffBitsQ8 + headerBitsQ8
        let totalDPCMBits = Double(totalDPCMBitsQ8) / 256.0
        let totalCoeffBits = Double(totalCoeffBitsQ8) / 256.0
        let headerBits = Double(headerBitsQ8) / 256.0
        let totalFileBits = Double(totalEncodedFileBytes * 8)

        var safeFileBits = totalFileBits
        if totalFileBits <= 0.0 {
            safeFileBits = max(1.0, totalDPCMBits)
        }
        var safeDPCMBits = totalDPCMBits
        if totalDPCMBits <= 0.0 {
            safeDPCMBits = 1.0
        }

        var zeroBlockRatio = 0.0
        if blockCount != 0 {
            zeroBlockRatio = (Double(zeroBlockCount) / Double(blockCount)) * 100.0
        }

        fputs("\n================================================================================\n", stderr)
        fputs("[DPCM Stats] Position Bit Accounting & Truncation Upper Bound Analysis\n", stderr)
        fputs("================================================================================\n", stderr)
        let headerMsg = String(
            format: "Total Encoded File Size : %lld bytes (%.0f bits)\nTotal DPCM Coded Bits   : %.1f bits (%.2f%% of file)\n  - DPCM Header / LSCP  : %.1f bits (%.2f%% of file)\n  - DPCM Residual Coeffs: %.1f bits (%.2f%% of file)\n  - Total Blocks        : %lld (Zero Blocks: %lld, %.1f%%)\n",
            totalEncodedFileBytes,
            totalFileBits,
            totalDPCMBits,
            (totalDPCMBits / safeFileBits) * 100.0,
            headerBits,
            (headerBits / safeFileBits) * 100.0,
            totalCoeffBits,
            (totalCoeffBits / safeFileBits) * 100.0,
            blockCount,
            zeroBlockCount,
            zeroBlockRatio
        )
        fputs(headerMsg, stderr)
        fputs("--------------------------------------------------------------------------------\n", stderr)
        fputs("Position Bits Distribution (i in [0, 15]):\n", stderr)

        for i in 0..<16 {
            let y = i / 4
            let x = i % 4
            let b = Double(posBitsQ8[i]) / 256.0
            let ratioDPCM = (b / safeDPCMBits) * 100.0
            let ratioFile = (b / safeFileBits) * 100.0
            let cnt = posCount[i]
            let line = String(
                format: "  pos[%2d] (y=%d, x=%d): %10.1f bits (%6.2f%% DPCM, %5.2f%% File, %6lld non-zeros)\n",
                i, y, x, b, ratioDPCM, ratioFile, cnt
            )
            fputs(line, stderr)
        }

        fputs("--------------------------------------------------------------------------------\n", stderr)
        fputs("Truncated Tail Bits Summary (Keep K / Total 16):\n", stderr)

        let kConfigs: [(name: String, k: Int)] = [
            ("1/4", 4),
            ("3/8", 6),
            ("1/2", 8),
            ("3/4", 12)
        ]

        var maxTailRatioFile = 0.0

        for cfg in kConfigs {
            var tailBitsQ8: Int64 = 0
            for i in cfg.k..<16 {
                tailBitsQ8 += posBitsQ8[i]
            }
            let tailBits = Double(tailBitsQ8) / 256.0
            let ratioDPCM = (tailBits / safeDPCMBits) * 100.0
            let ratioFile = (tailBits / safeFileBits) * 100.0
            if maxTailRatioFile < ratioFile {
                maxTailRatioFile = ratioFile
            }
            var gateStatus = "FAIL (< 5%)"
            if 5.0 <= ratioFile {
                gateStatus = "PASS (>= 5%)"
            }
            let line = String(
                format: "  K/N = %@ (K=%2d, tail [%2d..15]): %10.1f bits (%6.2f%% DPCM, %5.2f%% File) [Gate 0: %@]\n",
                cfg.name, cfg.k, cfg.k, tailBits, ratioDPCM, ratioFile, gateStatus
            )
            fputs(line, stderr)
        }

        // スクリプト解析用の構造化プレフィックス出力
        fputs("--------------------------------------------------------------------------------\n", stderr)
        let structuredSummary = String(
            format: "[VEVC_DPCM_STATS] Total File Size: %lld bytes (%.0f bits)\n[VEVC_DPCM_STATS] Total DPCM Blocks: %lld\n[VEVC_DPCM_STATS] Total DPCM Bits: %.1f bits (%.2f%% of file)\n[VEVC_DPCM_STATS] Position Bits (i=0..15):\n",
            totalEncodedFileBytes,
            totalFileBits,
            blockCount,
            totalDPCMBits,
            (totalDPCMBits / safeFileBits) * 100.0
        )
        fputs(structuredSummary, stderr)

        for i in 0..<16 {
            let b = Double(posBitsQ8[i]) / 256.0
            let ratioDPCM = (b / safeDPCMBits) * 100.0
            let ratioFile = (b / safeFileBits) * 100.0
            let posLine = String(
                format: "[VEVC_DPCM_STATS]   pos %2d: %10.1f bits (%5.2f%% dpcm, %5.2f%% file)\n",
                i, b, ratioDPCM, ratioFile
            )
            fputs(posLine, stderr)
        }

        fputs("[VEVC_DPCM_STATS] Retention Ladder (Tail Bits & Ratio of File):\n", stderr)
        for cfg in kConfigs {
            var tailBitsQ8: Int64 = 0
            for i in cfg.k..<16 {
                tailBitsQ8 += posBitsQ8[i]
            }
            let tailBits = Double(tailBitsQ8) / 256.0
            let ratioFile = (tailBits / safeFileBits) * 100.0
            var gateStatus = "FAIL (< 5%)"
            if 5.0 <= ratioFile {
                gateStatus = "PASS (>= 5%)"
            }
            let ladderLine = String(
                format: "[VEVC_DPCM_STATS]   K=%2d (%@ keep, tail [%2d..15]): %10.1f bits (%5.2f%% of file) [Gate 0: %@]\n",
                cfg.k, cfg.name, cfg.k, tailBits, ratioFile, gateStatus
            )
            fputs(ladderLine, stderr)
        }

        var overallAssessment = "FAIL (< 5.0%)"
        if 5.0 <= maxTailRatioFile {
            overallAssessment = "PASS (max tail ratio >= 5.0%)"
        }
        let assessmentLine = String(
            format: "[VEVC_DPCM_STATS] Gate 0 Assessment: %@ (max tail ratio = %.2f%%)\n",
            overallAssessment, maxTailRatioFile
        )
        fputs(assessmentLine, stderr)
        fputs("================================================================================\n\n", stderr)
    }
}
