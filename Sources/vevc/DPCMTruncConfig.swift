import Foundation

/// DPCM 後方切り詰め (VEVC_DPCM_TRUNC) 設定管理クラス (シングルトン)
public final class DPCMTruncConfig: @unchecked Sendable {
    public static let shared = DPCMTruncConfig()

    public let isEnabled: Bool
    public let k: Int
    public let scanOrder: DPCMScanOrder
    public let predictor: DPCMPredictorEngine
    public let predictorName: String
    public let epsilon: Int16

    private init() {
        if let val = getenv("VEVC_DPCM_TRUNC") {
            let str = String(cString: val).trimmingCharacters(in: .whitespacesAndNewlines)
            if str.isEmpty {
                self.isEnabled = false
                self.k = 6
                self.scanOrder = .serpentine
                self.predictor = SNNPureIntegerPredictor()
                self.predictorName = "p2 (SNN-DAG)"
                self.epsilon = 1
            } else {
                let parts = str.split(separator: ":").map { String($0) }
                
                // 1. 有効フラグおよび K 値の解析 (既定: 6)
                var enabled = true
                var parsedK = 6
                if 0 < parts.count {
                    switch parts[0].lowercased() {
                    case "0", "false", "off", "no":
                        enabled = false
                    case "1", "true", "on", "yes":
                        enabled = true
                        parsedK = 6
                    default:
                        if let kv = Int(parts[0]) {
                            if 0 < kv {
                                parsedK = kv
                            }
                        }
                    }
                }
                self.isEnabled = enabled
                self.k = parsedK
                
                // 2. 走査順序の解析 (既定: serpentine)
                var parsedScan = DPCMScanOrder.serpentine
                if 1 < parts.count {
                    switch parts[1].lowercased() {
                    case "raster":
                        parsedScan = .raster
                    default:
                        parsedScan = .serpentine
                    }
                }
                self.scanOrder = parsedScan
                
                // 3. 予測器エンジンの解析 (既定: SNNPureIntegerPredictor / p2)
                var parsedPred: DPCMPredictorEngine = SNNPureIntegerPredictor()
                var pName = "p2 (SNN-DAG)"
                if 2 < parts.count {
                    switch parts[2].lowercased() {
                    case "hold", "p0":
                        parsedPred = DPCMPredictorHold()
                        pName = "p0 (Hold)"
                    case "med", "p1":
                        parsedPred = DPCMPredictorMED()
                        pName = "p1 (MED)"
                    case "ar", "p3":
                        parsedPred = DPCMPredictorSNNAutoReg()
                        pName = "p3 (SNN-AR)"
                    case "batch", "snn", "p2":
                        parsedPred = SNNPureIntegerPredictor()
                        pName = "p2 (SNN-DAG)"
                    default:
                        parsedPred = SNNPureIntegerPredictor()
                        pName = "p2 (SNN-DAG)"
                    }
                }
                self.predictor = parsedPred
                self.predictorName = pName
                
                // 4. 許容誤差 epsilon の解析 (既定: 1)
                var parsedEps: Int16 = 1
                if 3 < parts.count {
                    if let ev = Int16(parts[3]) {
                        parsedEps = ev
                    }
                }
                self.epsilon = parsedEps
            }
        } else {
            self.isEnabled = false
            self.k = 6
            self.scanOrder = .serpentine
            self.predictor = SNNPureIntegerPredictor()
            self.predictorName = "p2 (SNN-DAG)"
            self.epsilon = 1
        }
    }
}
