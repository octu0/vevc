import Foundation

/// DPCM 走査順序の定義
public enum DPCMScanOrder: UInt8, Sendable, CaseIterable {
    case raster = 0
    case serpentine = 1

    /// 走査インデックス (0..15) から 2D 座標 (x, y) への変換
    @inline(__always)
    public func toXY(index: Int) -> (x: Int, y: Int) {
        let y = index >> 2
        let rem = index & 3
        switch self {
        case .raster:
            return (rem, y)
        case .serpentine:
            if (y & 1) == 0 {
                return (rem, y)
            } else {
                return (3 - rem, y)
            }
        }
    }

    /// 2D 座標 (x, y) から走査インデックス (0..15) への変換
    @inline(__always)
    public func toIndex(x: Int, y: Int) -> Int {
        let base = y << 2
        switch self {
        case .raster:
            return base + x
        case .serpentine:
            if (y & 1) == 0 {
                return base + x
            } else {
                return base + (3 - x)
            }
        }
    }
}

/// DPCM ブロック予測コンテキスト
public struct DPCMPredictContext: Sendable {
    public let k: Int                         // 保持要素数 (4, 6, 8, 12)
    public let scanOrder: DPCMScanOrder       // 走査順
    public let qLow: Int16                    // 量子化ステップ (qt.qLow.step)
    public let lastVal: Int16                 // 直前ブロック末尾再構成値
    public let topBoundary: (Int16, Int16, Int16, Int16)   // 上境界画素 (4要素)
    public let leftBoundary: (Int16, Int16, Int16, Int16)  // 左境界画素 (4要素)
    public let topLeftBoundary: Int16                      // 左上境界画素
    public let mcPred: (Int16, Int16, Int16, Int16,
                        Int16, Int16, Int16, Int16,
                        Int16, Int16, Int16, Int16,
                        Int16, Int16, Int16, Int16)        // MC 予測画素 (16要素)

    public init(
        k: Int,
        scanOrder: DPCMScanOrder,
        qLow: Int16,
        lastVal: Int16,
        topBoundary: (Int16, Int16, Int16, Int16),
        leftBoundary: (Int16, Int16, Int16, Int16),
        topLeftBoundary: Int16,
        mcPred: (Int16, Int16, Int16, Int16,
                 Int16, Int16, Int16, Int16,
                 Int16, Int16, Int16, Int16,
                 Int16, Int16, Int16, Int16)
    ) {
        self.k = k
        self.scanOrder = scanOrder
        self.qLow = qLow
        self.lastVal = lastVal
        self.topBoundary = topBoundary
        self.leftBoundary = leftBoundary
        self.topLeftBoundary = topLeftBoundary
        self.mcPred = mcPred
    }
}

/// DPCM 予測器エンジンのプロトコル
public protocol DPCMPredictorEngine: Sendable {
    /// 前方 K 個の伝送値およびコンテキストから 16 要素のブロックを復元
    /// - Parameters:
    ///   - transmittedK: 送信された K 個の値
    ///   - context: 予測コンテキスト
    ///   - output: 出力 16 要素バッファ (row-major: y * 4 + x)
    func predictBlock(
        transmittedK: UnsafePointer<Int16>,
        context: DPCMPredictContext,
        output: UnsafeMutablePointer<Int16>
    )
}

/// MED 予測ヘルパー関数 (純整数)
@inline(__always)
public func predictMEDValue(_ a: Int16, _ b: Int16, _ c: Int16) -> Int16 {
    let maxAB = max(a, b)
    let minAB = min(a, b)
    if maxAB <= c {
        return minAB
    }
    if c <= minAB {
        return maxAB
    }
    return a &+ b &- c
}

// MARK: - p0: 最終値ホールド予測器 (Zero / Hold Baseline)
public struct DPCMPredictorHold: DPCMPredictorEngine {
    public init() {}

    public func predictBlock(
        transmittedK: UnsafePointer<Int16>,
        context: DPCMPredictContext,
        output: UnsafeMutablePointer<Int16>
    ) {
        let k = context.k
        let lastVal = transmittedK[k - 1]
        
        // 送信済み K 要素を配置
        for i in 0..<k {
            let pt = context.scanOrder.toXY(index: i)
            output[pt.y * 4 + pt.x] = transmittedK[i]
        }
        
        // 残りのテール要素を最後の値でホールド
        for i in k..<16 {
            let pt = context.scanOrder.toXY(index: i)
            output[pt.y * 4 + pt.x] = lastVal
        }
    }
}

// MARK: - p1: MED 空間外挿予測器 (Deterministic Spatial Extrapolation)
public struct DPCMPredictorMED: DPCMPredictorEngine {
    public init() {}

    public func predictBlock(
        transmittedK: UnsafePointer<Int16>,
        context: DPCMPredictContext,
        output: UnsafeMutablePointer<Int16>
    ) {
        let k = context.k
        
        // 送信済み K 要素を初期配置
        var isKnown = [Bool](repeating: false, count: 16)
        for i in 0..<k {
            let pt = context.scanOrder.toXY(index: i)
            output[pt.y * 4 + pt.x] = transmittedK[i]
            isKnown[pt.y * 4 + pt.x] = true
        }

        let top = [context.topBoundary.0, context.topBoundary.1, context.topBoundary.2, context.topBoundary.3]
        let left = [context.leftBoundary.0, context.leftBoundary.1, context.leftBoundary.2, context.leftBoundary.3]
        let tl = context.topLeftBoundary

        // 走査順に沿って未確定画素を因果的 MED 外挿
        for i in 0..<16 {
            let pt = context.scanOrder.toXY(index: i)
            let y = pt.y
            let x = pt.x
            let idx = y * 4 + x
            
            if isKnown[idx] {
                continue
            }
            
            var a: Int16 = 0
            var b: Int16 = 0
            var c: Int16 = 0
            
            switch context.scanOrder {
            case .raster:
                // ラスタ走査 (常に左から右)
                if x == 0 {
                    a = left[y]
                } else {
                    a = output[y * 4 + (x - 1)]
                }
                if y == 0 {
                    b = top[x]
                } else {
                    b = output[(y - 1) * 4 + x]
                }
                switch (x, y) {
                case (0, 0):
                    c = tl
                case let (curX, 0):
                    c = top[curX - 1]
                case let (0, curY):
                    c = left[curY - 1]
                default:
                    c = output[(y - 1) * 4 + (x - 1)]
                }
            case .serpentine:
                if (y & 1) == 0 {
                    // 偶数行: 左から右
                    if x == 0 {
                        a = left[y]
                    } else {
                        a = output[y * 4 + (x - 1)]
                    }
                    if y == 0 {
                        b = top[x]
                    } else {
                        b = output[(y - 1) * 4 + x]
                    }
                    switch (x, y) {
                    case (0, 0):
                        c = tl
                    case let (curX, 0):
                        c = top[curX - 1]
                    case let (0, curY):
                        c = left[curY - 1]
                    default:
                        c = output[(y - 1) * 4 + (x - 1)]
                    }
                } else {
                    // 奇数行: 右から左
                    if y == 0 {
                        b = top[x]
                    } else {
                        b = output[(y - 1) * 4 + x]
                    }
                    if x == 3 {
                        a = b
                        c = b
                    } else {
                        a = output[y * 4 + (x + 1)]
                        if y == 0 {
                            c = top[x + 1]
                        } else {
                            c = output[(y - 1) * 4 + (x + 1)]
                        }
                    }
                }
            }
            
            let pred = predictMEDValue(a, b, c)
            output[idx] = pred
            isKnown[idx] = true
        }
    }
}

// MARK: - p2: SNN 一括推論予測器 (Pure Integer LIF Feedforward DAG)
public struct DPCMPredictorSNNBatch: DPCMPredictorEngine {
    public init() {}

    public func predictBlock(
        transmittedK: UnsafePointer<Int16>,
        context: DPCMPredictContext,
        output: UnsafeMutablePointer<Int16>
    ) {
        let k = context.k
        
        // 1. 初期推定として p1 (MED) 予測値を生成
        let medPredictor = DPCMPredictorMED()
        medPredictor.predictBlock(transmittedK: transmittedK, context: context, output: output)
        
        // 2. 特徴量ベクトルの構築 (4チャンネル x 16要素)
        var ch0 = SIMD16<Int16>() // 初期推定 (Q11: val << 3)
        var ch1 = SIMD16<Int16>() // 空間境界コンテキスト
        var ch2 = SIMD16<Int16>() // MC 予測
        var ch3 = SIMD16<Int16>() // 位置・スケール
        
        let top = [context.topBoundary.0, context.topBoundary.1, context.topBoundary.2, context.topBoundary.3]
        let left = [context.leftBoundary.0, context.leftBoundary.1, context.leftBoundary.2, context.leftBoundary.3]
        
        let mcArray = [
            context.mcPred.0, context.mcPred.1, context.mcPred.2, context.mcPred.3,
            context.mcPred.4, context.mcPred.5, context.mcPred.6, context.mcPred.7,
            context.mcPred.8, context.mcPred.9, context.mcPred.10, context.mcPred.11,
            context.mcPred.12, context.mcPred.13, context.mcPred.14, context.mcPred.15
        ]
        
        for i in 0..<16 {
            let pt = context.scanOrder.toXY(index: i)
            let y = pt.y
            let x = pt.x
            let idx = y * 4 + x
            
            // Ch 0: 初期推定値
            ch0[i] = output[idx]
            
            // Ch 1: 境界からの距離加重コンテキスト
            let distL = Int32(x + 1)
            let distT = Int32(y + 1)
            let boundaryVal = Int16((Int32(left[y]) * distT + Int32(top[x]) * distL) / (distL + distT))
            ch1[i] = boundaryVal
            
            // Ch 2: MC 予測
            ch2[i] = mcArray[idx]
            
            // Ch 3: 位置および量子化ステップの正規化表現
            let posVal = Int16((y * 4 + x) * 64)
            ch3[i] = posVal &+ (context.qLow &* 8)
        }
        
        // 3. 純整数 LIF SNN 推論 (T=2, Q7/Q11 固定小数点)
        var delta = [Int16](repeating: 0, count: 16)
        let vTh: Int16 = 2048
        
        for i in 0..<16 {
            let diffMC = ch2[i] &- ch0[i]
            let diffBound = ch1[i] &- ch0[i]
            let curr = (diffMC >> 1) &+ (diffBound >> 2)
            
            // LIF タイムステップ t = 0
            var u = curr
            var corr: Int16 = 0
            if vTh <= u {
                corr = corr &+ 1
                u = u &- vTh
            }
            
            // LIF タイムステップ t = 1 (リーク減衰 u = u - (u >> 2))
            let uLeak = u &- (u >> 2)
            u = uLeak &+ curr
            if vTh <= u {
                corr = corr &+ 1
                u = u &- vTh
            }
            
            // 出力層 (補正値の累積)
            let fineAdj = (u &+ 512) >> 10
            delta[i] = corr &+ fineAdj
        }
        
        // 4. 補正値の適用と送信済み K 要素の強制一致 (誤差ゼロ保証)
        for i in 0..<16 {
            let pt = context.scanOrder.toXY(index: i)
            let y = pt.y
            let x = pt.x
            let idx = y * 4 + x
            
            if i < k {
                output[idx] = transmittedK[i]
            } else {
                output[idx] = output[idx] &+ delta[i]
            }
        }
    }
}

// MARK: - p3: SNN 自己回帰推論予測器 (Autoregressive Sequential/Chunked LIF)
public struct DPCMPredictorSNNAutoReg: DPCMPredictorEngine {
    public init() {}

    public func predictBlock(
        transmittedK: UnsafePointer<Int16>,
        context: DPCMPredictContext,
        output: UnsafeMutablePointer<Int16>
    ) {
        let k = context.k
        
        // 送信済み K 要素を配置
        var currentValues = [Int16](repeating: 0, count: 16)
        for i in 0..<k {
            let pt = context.scanOrder.toXY(index: i)
            currentValues[pt.y * 4 + pt.x] = transmittedK[i]
        }
        
        // テール要素を行・走査順に 1 要素ずつ因果的に自己回帰推論
        let batchSNN = DPCMPredictorSNNBatch()
        
        for step in k..<16 {
            // 現在の確定値 (0..<step) を入力として SNN 推論を実行
            var tempOut = [Int16](repeating: 0, count: 16)
            var currentTransmitted = [Int16](repeating: 0, count: step)
            for i in 0..<step {
                let pt = context.scanOrder.toXY(index: i)
                currentTransmitted[i] = currentValues[pt.y * 4 + pt.x]
            }
            
            let subContext = DPCMPredictContext(
                k: step,
                scanOrder: context.scanOrder,
                qLow: context.qLow,
                lastVal: context.lastVal,
                topBoundary: context.topBoundary,
                leftBoundary: context.leftBoundary,
                topLeftBoundary: context.topLeftBoundary,
                mcPred: context.mcPred
            )
            
            currentTransmitted.withUnsafeBufferPointer { tbPtr in
                tempOut.withUnsafeMutableBufferPointer { outPtr in
                    batchSNN.predictBlock(
                        transmittedK: tbPtr.baseAddress!,
                        context: subContext,
                        output: outPtr.baseAddress!
                    )
                }
            }
            
            // step 番目の予測値を確定値に組み込む
            let nextPt = context.scanOrder.toXY(index: step)
            currentValues[nextPt.y * 4 + nextPt.x] = tempOut[nextPt.y * 4 + nextPt.x]
        }
        
        // 最終出力をコピー
        for i in 0..<16 {
            output[i] = currentValues[i]
        }
    }
}

// MARK: - オフラインラダー評価および Gate 1 判定エンジン
public struct DPCMLadderResult: Sendable {
    public let k: Int
    public let scanOrder: DPCMScanOrder
    public let predictorName: String
    public let epsilon: Int16
    public let totalBlocks: Int64
    public let coveredBlocks: Int64
    public let blockCoverageRatio: Double
    public let weightedCoverageRatio: Double
    public let tailRatioFile: Double
    public let fileReductionPotential: Double
    public let gate1Passed: Bool
}

public final class DPCMLadderEvaluator: @unchecked Sendable {
    public private(set) var records: [DPCMBlockDumpRecord] = []
    public private(set) var totalEncodedFileBits: Double = 0.0
    
    public init() {}

    /// ダンプバイナリファイルからレコードをロード
    public func loadDumpFile(at path: String, totalFileBits: Double = 0.0) throws {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        
        guard 64 <= data.count else {
            throw NSError(domain: "DPCMLadderEvaluator", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid header size"])
        }
        
        // マジックヘッダ確認: "VDPD"
        guard data[0] == 0x56, data[1] == 0x44, data[2] == 0x50, data[3] == 0x44 else {
            throw NSError(domain: "DPCMLadderEvaluator", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid magic header"])
        }
        
        let recordSize = 164
        let recordBytes = data.subdata(in: 64..<data.count)
        let count = recordBytes.count / recordSize
        
        records.removeAll()
        records.reserveCapacity(count)
        
        recordBytes.withUnsafeBytes { rawPtr in
            var offset = 0
            for _ in 0..<count {
                let recPtr = rawPtr.baseAddress!.advanced(by: offset).assumingMemoryBound(to: DPCMBlockDumpRecord.self)
                records.append(recPtr.pointee)
                offset += recordSize
            }
        }
        
        self.totalEncodedFileBits = totalFileBits
    }
    
    public func addRecord(_ record: DPCMBlockDumpRecord) {
        records.append(record)
    }
    
    public func setTotalEncodedFileBits(_ bits: Double) {
        self.totalEncodedFileBits = bits
    }

    /// 単一の構成 (K, scanOrder, predictor, epsilon) に対する評価を実行
    public func evaluateConfiguration(
        k: Int,
        scanOrder: DPCMScanOrder,
        predictor: DPCMPredictorEngine,
        predictorName: String,
        epsilon: Int16
    ) -> DPCMLadderResult {
        var totalBlocksCount: Int64 = 0
        var coveredBlocksCount: Int64 = 0
        var totalTailBitsQ8: Int64 = 0
        var coveredTailBitsQ8: Int64 = 0
        
        var outBuf = [Int16](repeating: 0, count: 16)
        
        for record in records {
            // 全ゼロブロックはスキップ (DPCM コストなし)
            if record.isAllZero == 1 {
                continue
            }
            
            totalBlocksCount += 1
            
            // 真値量子化配列
            let quant = [
                record.quantizedValues.0, record.quantizedValues.1, record.quantizedValues.2, record.quantizedValues.3,
                record.quantizedValues.4, record.quantizedValues.5, record.quantizedValues.6, record.quantizedValues.7,
                record.quantizedValues.8, record.quantizedValues.9, record.quantizedValues.10, record.quantizedValues.11,
                record.quantizedValues.12, record.quantizedValues.13, record.quantizedValues.14, record.quantizedValues.15
            ]
            
            // ビットコスト配列
            let costsQ8 = [
                record.ransBitCostsQ8.0, record.ransBitCostsQ8.1, record.ransBitCostsQ8.2, record.ransBitCostsQ8.3,
                record.ransBitCostsQ8.4, record.ransBitCostsQ8.5, record.ransBitCostsQ8.6, record.ransBitCostsQ8.7,
                record.ransBitCostsQ8.8, record.ransBitCostsQ8.9, record.ransBitCostsQ8.10, record.ransBitCostsQ8.11,
                record.ransBitCostsQ8.12, record.ransBitCostsQ8.13, record.ransBitCostsQ8.14, record.ransBitCostsQ8.15
            ]
            
            // 走査順に並べた真値および送信済み K 要素
            var transmittedK = [Int16](repeating: 0, count: k)
            var blockTailBitsQ8: Int64 = 0
            
            for i in 0..<16 {
                let pt = scanOrder.toXY(index: i)
                let val = quant[pt.y * 4 + pt.x]
                if i < k {
                    transmittedK[i] = val
                } else {
                    blockTailBitsQ8 += Int64(costsQ8[pt.y * 4 + pt.x])
                }
            }
            
            totalTailBitsQ8 += blockTailBitsQ8
            
            // 予測器実行
            let context = DPCMPredictContext(
                k: k,
                scanOrder: scanOrder,
                qLow: record.qLow,
                lastVal: record.lastVal,
                topBoundary: record.topBoundary,
                leftBoundary: record.leftBoundary,
                topLeftBoundary: record.topLeftBoundary,
                mcPred: record.mcPred
            )
            
            transmittedK.withUnsafeBufferPointer { kPtr in
                outBuf.withUnsafeMutableBufferPointer { outPtr in
                    predictor.predictBlock(
                        transmittedK: kPtr.baseAddress!,
                        context: context,
                        output: outPtr.baseAddress!
                    )
                }
            }
            
            // テール位置の最大絶対誤差を計算
            var maxErr: Int16 = 0
            for i in k..<16 {
                let pt = scanOrder.toXY(index: i)
                let trueVal = quant[pt.y * 4 + pt.x]
                let predVal = outBuf[pt.y * 4 + pt.x]
                let diff = abs(Int32(trueVal) - Int32(predVal))
                if maxErr < Int16(clamping: diff) {
                    maxErr = Int16(clamping: diff)
                }
            }
            
            if maxErr <= epsilon {
                coveredBlocksCount += 1
                coveredTailBitsQ8 += blockTailBitsQ8
            }
        }
        
        var blockCoverage = 0.0
        if 0 < totalBlocksCount {
            blockCoverage = (Double(coveredBlocksCount) / Double(totalBlocksCount)) * 100.0
        }
        var weightedCoverage = 0.0
        if 0 < totalTailBitsQ8 {
            weightedCoverage = (Double(coveredTailBitsQ8) / Double(totalTailBitsQ8)) * 100.0
        }
        
        var safeFileBits = totalEncodedFileBits
        if totalEncodedFileBits <= 0.0 {
            safeFileBits = Double(max(1, totalTailBitsQ8)) / 256.0
        }
        
        let totalTailBits = Double(totalTailBitsQ8) / 256.0
        let tailRatioFile = (totalTailBits / safeFileBits) * 100.0
        let fileReductionPotential = (weightedCoverage / 100.0) * tailRatioFile
        let gate1Passed = 2.0 <= fileReductionPotential
        
        return DPCMResultItem(
            k: k,
            scanOrder: scanOrder,
            predictorName: predictorName,
            epsilon: epsilon,
            totalBlocks: totalBlocksCount,
            coveredBlocks: coveredBlocksCount,
            blockCoverageRatio: blockCoverage,
            weightedCoverageRatio: weightedCoverage,
            tailRatioFile: tailRatioFile,
            fileReductionPotential: fileReductionPotential,
            gate1Passed: gate1Passed
        )
    }
    
    /// 32 構成空間 (4 K x 2 Scan x 4 Predictors) の全評価を実行
    public func runFullLadderEvaluation(epsilon: Int16 = 1) -> [DPCMLadderResult] {
        let kValues = [4, 6, 8, 12]
        let scanOrders = [DPCMScanOrder.raster, DPCMScanOrder.serpentine]
        let predictors: [(name: String, engine: DPCMPredictorEngine)] = [
            ("p0 (Hold)", DPCMPredictorHold()),
            ("p1 (MED)", DPCMPredictorMED()),
            ("p2 (SNN-DAG)", DPCMPredictorSNNBatch()),
            ("p3 (SNN-AR)", DPCMPredictorSNNAutoReg())
        ]
        
        var results = [DPCMLadderResult]()
        
        for k in kValues {
            for scan in scanOrders {
                for p in predictors {
                    let res = evaluateConfiguration(
                        k: k,
                        scanOrder: scan,
                        predictor: p.engine,
                        predictorName: p.name,
                        epsilon: epsilon
                    )
                    results.append(res)
                }
            }
        }
        
        return results
    }
}

public typealias DPCMResultItem = DPCMLadderResult
