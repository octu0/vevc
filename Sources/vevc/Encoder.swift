import Foundation

public class Encoder {
    public let width: Int
    public let height: Int
    public let maxbitrate: Int
    public let framerate: Int
    public let zeroThreshold: Int
    public let gopSize: Int
    public let sceneChangeThreshold: Int
    public let isOne: Bool
    
    private var prevReconstructed: PlaneData420? = nil
    private var gopCount = 0
    private var qt: QuantizationTable? = nil
    private let mbSize = 64
    private var frameIndex = 0
    
    // Rate Control
    private var rateController: RateController
    
    public init(width: Int, height: Int, maxbitrate: Int, framerate: Int = 30, zeroThreshold: Int = 3, gopSize: Int = 15, sceneChangeThreshold: Int = 8, isOne: Bool = false) {
        self.width = width
        self.height = height
        self.maxbitrate = maxbitrate
        self.framerate = framerate
        self.zeroThreshold = zeroThreshold
        self.gopSize = gopSize
        self.sceneChangeThreshold = sceneChangeThreshold
        self.isOne = isOne
        
        self.rateController = RateController(maxbitrate: maxbitrate, framerate: framerate, gopSize: gopSize)
    }
    
    #if (arch(arm64) || arch(x86_64) || arch(wasm32))
    public func encode(image: YCbCrImage) async throws -> [UInt8] {
        var out: [UInt8] = []
        let curr = toPlaneData420(images: [image])[0]
        
        var forceIFrame = false
        var predictedPlane: PlaneData420? = nil
        var mvs = MotionVectors(count: 0)
        var meanSAD: Int = 0
        var maxBlockSAD: Int = 0
        
        if gopSize <= gopCount || prevReconstructed == nil {
            forceIFrame = true
        } else {
            guard let prev = prevReconstructed else { throw NSError(domain: "vevc.Encoder", code: 2, userInfo: nil) }
            
            mvs = estimateMBME(curr: curr, prev: prev)
            let predicted = await applyMBME(prev: prev, mvs: mvs)
            predictedPlane = predicted
            let res = await subPlanes(curr: curr, predicted: predicted)
            
            let stats = calculateSADAndMaxBlockSAD(res: res, mbSize: mbSize)
            meanSAD = stats.meanSAD
            maxBlockSAD = stats.maxBlockSAD
            
            if sceneChangeThreshold < meanSAD {
                forceIFrame = true
                debugLog("[Frame \(frameIndex)] Adaptive GOP: Forced I-Frame due to high meanSAD (\(meanSAD) > \(sceneChangeThreshold))")
            }
        }
        
        if forceIFrame {
            let targetBits = rateController.beginGOP()
            let baseQt = estimateQuantization(img: image, targetBits: targetBits)
            if isOne {
                self.qt = QuantizationTable(baseStep: max(1, Int(Double(baseQt.step) * 1.5)))
            } else {
                self.qt = baseQt
            }
        }
        guard let qt = self.qt else { throw NSError(domain: "vevc.Encoder", code: 1, userInfo: nil) }
        
        if isOne {
            if forceIFrame {
                let qtY = QuantizationTable(baseStep: max(1, Int(qt.step)))
                let qtC = QuantizationTable(baseStep: max(1, Int(qt.step) * 2), isChroma: true)
                
                let (layer0, reconstructed) = try await encodePlaneBase32(pd: curr, predictedPd: nil, layer: 0, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
                
                out.append(contentsOf: [0x56, 0x45, 0x4F, 0x49]) // VEOI
                
                var chunkOut: [UInt8] = []
                chunkOut.append(contentsOf: layer0)
                
                appendUInt32BE(&out, UInt32(chunkOut.count))
                out.append(contentsOf: chunkOut)
                debugLog("[Frame \(frameIndex)] I-Frame (One): \(chunkOut.count) bytes (\(String(format: "%.2f", Double(chunkOut.count) / 1024.0)) KB)")
                
                rateController.consumeIFrame(bits: chunkOut.count * 8, qStep: Int(qtY.step))
                
                prevReconstructed = reconstructed
                gopCount = 1
            } else {
                // Adaptive Quantization (isOne mode)
                let currentSAD = Double(meanSAD)
                let newStepInt = rateController.calculatePFrameQStep(currentSAD: currentSAD, baseStep: Int(qt.step))
                let newStep = Int16(newStepInt)
                
                var qtY: QuantizationTable
                var qtC: QuantizationTable
                
                let fineQuantizationThreshold = mbSize * mbSize * 1
                if meanSAD > 1 || maxBlockSAD > fineQuantizationThreshold {
                    qtY = QuantizationTable(baseStep: max(1, Int(newStep)), isChroma: false)
                    qtC = QuantizationTable(baseStep: max(1, Int(newStep) * 2), isChroma: true)
                } else {
                    qtY = QuantizationTable(baseStep: max(1, Int(newStep) / 2), isChroma: false)
                    qtC = QuantizationTable(baseStep: max(1, Int(newStep)), isChroma: true)
                }
                
                let (layer0, reconstructedResidual) = try await encodePlaneBase32(pd: curr, predictedPd: predictedPlane, layer: 0, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
                
                out.append(contentsOf: [0x56, 0x45, 0x4F, 0x50]) // VEOP

                var mvBw = EntropyEncoder()
                let mbCols = (curr.width + mbSize - 1) / mbSize
                for mvIdx in 0..<mvs.vectors.count {
                    let mbX = mvIdx % mbCols
                    let mbY = mvIdx / mbCols
                    let pmv = calculatePMV(mvs: mvs, mbX: mbX, mbY: mbY, mbCols: mbCols)
                    let vec = mvs.vectors[mvIdx]
                    let mvdX = Int(vec.x) - pmv.dx
                    let mvdY = Int(vec.y) - pmv.dy

                    if mvdX == 0 && mvdY == 0 {
                        mvBw.encodeBypass(binVal: 0)
                    } else {
                        mvBw.encodeBypass(binVal: 1)
                        let sx: UInt8 = mvdX <= -1 ? 1 : 0
                        mvBw.encodeBypass(binVal: sx)
                        let mx = UInt32(abs(mvdX))
                        encodeExpGolomb(val: mx, encoder: &mvBw)

                        let sy: UInt8 = mvdY <= -1 ? 1 : 0
                        mvBw.encodeBypass(binVal: sy)
                        let my = UInt32(abs(mvdY))
                        encodeExpGolomb(val: my, encoder: &mvBw)
                    }
                }
                mvBw.flush()
                let mvOut = mvBw.getData()
                appendUInt32BE(&out, UInt32(mvs.vectors.count))
                appendUInt32BE(&out, UInt32(mvOut.count))
                out.append(contentsOf: mvOut)

                var chunkOut: [UInt8] = []
                chunkOut.append(contentsOf: layer0)
                
                appendUInt32BE(&out, UInt32(chunkOut.count))
                out.append(contentsOf: chunkOut)
                let totalBytes = chunkOut.count + mvOut.count
                debugLog("[Frame \(frameIndex)] P-Frame (One): \(totalBytes) bytes (MV: \(mvOut.count) bytes, Data: \(chunkOut.count) bytes) MVs=\(mvs.vectors.count) meanSAD=\(meanSAD) [PMV & LSCP applied]")
                
                rateController.consumePFrame(bits: totalBytes * 8, qStep: Int(qtY.step), sad: Double(meanSAD))
                
                if let predicted = predictedPlane {
                    let reconstructed = await addPlanes(residual: reconstructedResidual, predicted: predicted)
                    prevReconstructed = reconstructed
                } else {
                    prevReconstructed = reconstructedResidual
                }
                gopCount += 1
            }
        } else {
            if forceIFrame {
                let qtY = QuantizationTable(baseStep: max(1, Int(qt.step)))
                let qtC = QuantizationTable(baseStep: max(1, Int(qt.step) * 2), isChroma: true)
                let (bytes, reconstructed) = try await encodeSpatialLayers(pd: curr, predictedPd: nil, maxbitrate: maxbitrate, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
                
                out.append(contentsOf: [0x56, 0x45, 0x56, 0x49])
                appendUInt32BE(&out, UInt32(bytes.count))
                out.append(contentsOf: bytes)
                debugLog("[Frame \(frameIndex)] I-Frame: \(bytes.count) bytes (\(String(format: "%.2f", Double(bytes.count) / 1024.0)) KB)")
                
                rateController.consumeIFrame(bits: bytes.count * 8, qStep: Int(qtY.step))
                
                prevReconstructed = reconstructed
                gopCount = 1
            } else {
                let currentSAD = Double(meanSAD)
                let newStepInt = rateController.calculatePFrameQStep(currentSAD: currentSAD, baseStep: Int(qt.step))
                let newStep = Int16(newStepInt)
                
                var qtY: QuantizationTable
                var qtC: QuantizationTable
                
                let fineQuantizationThreshold = mbSize * mbSize * 1
                if meanSAD > 1 || maxBlockSAD > fineQuantizationThreshold {
                    qtY = QuantizationTable(baseStep: max(1, Int(newStep)), isChroma: false)
                    qtC = QuantizationTable(baseStep: max(1, Int(newStep) * 2), isChroma: true)
                } else {
                    qtY = QuantizationTable(baseStep: max(1, Int(newStep) / 2), isChroma: false)
                    qtC = QuantizationTable(baseStep: max(1, Int(newStep)), isChroma: true)
                }
                let (bytes, reconstructedResidual) = try await encodeSpatialLayers(pd: curr, predictedPd: predictedPlane, maxbitrate: maxbitrate, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
                
                out.append(contentsOf: [0x56, 0x45, 0x56, 0x50])

                var mvBw = EntropyEncoder()
                let mbCols = (curr.width + mbSize - 1) / mbSize
                for mvIdx in 0..<mvs.vectors.count {
                    let mbX = mvIdx % mbCols
                    let mbY = mvIdx / mbCols
                    let pmv = calculatePMV(mvs: mvs, mbX: mbX, mbY: mbY, mbCols: mbCols)
                    let vec = mvs.vectors[mvIdx]
                    let mvdX = Int(vec.x) - pmv.dx
                    let mvdY = Int(vec.y) - pmv.dy

                    if mvdX == 0 && mvdY == 0 {
                        mvBw.encodeBypass(binVal: 0)
                    } else {
                        mvBw.encodeBypass(binVal: 1)
                        let sx: UInt8 = mvdX <= -1 ? 1 : 0
                        mvBw.encodeBypass(binVal: sx)
                        let mx = UInt32(abs(mvdX))
                        encodeExpGolomb(val: mx, encoder: &mvBw)

                        let sy: UInt8 = mvdY <= -1 ? 1 : 0
                        mvBw.encodeBypass(binVal: sy)
                        let my = UInt32(abs(mvdY))
                        encodeExpGolomb(val: my, encoder: &mvBw)
                    }
                }
                mvBw.flush()
                let mvOut = mvBw.getData()
                appendUInt32BE(&out, UInt32(mvs.vectors.count))
                appendUInt32BE(&out, UInt32(mvOut.count))
                out.append(contentsOf: mvOut)

                appendUInt32BE(&out, UInt32(bytes.count))
                out.append(contentsOf: bytes)
                let totalBytes = bytes.count + mvOut.count
                debugLog("[Frame \(frameIndex)] P-Frame: \(totalBytes) bytes (MV: \(mvOut.count) bytes, Data: \(bytes.count) bytes) MVs=\(mvs.vectors.count) meanSAD=\(meanSAD) maxBlockSAD=\(maxBlockSAD) [PMV & LSCP applied]")
                
                rateController.consumePFrame(bits: totalBytes * 8, qStep: Int(qtY.step), sad: Double(meanSAD))
                
                if let predicted = predictedPlane {
                    let reconstructed = await addPlanes(residual: reconstructedResidual, predicted: predicted)
                    prevReconstructed = reconstructed
                } else {
                    prevReconstructed = reconstructedResidual
                }
                gopCount += 1
            }
        }
        
        frameIndex += 1
        return out
    }
    #else
    public func encode(image: YCbCrImage) async throws -> [UInt8] {
        throw EncodeError.unsupportedArchitecture
    }
    #endif
    
    @inline(__always)
    private func appendUInt32BE(_ out: inout [UInt8], _ val: UInt32) {
        out.append(UInt8((val >> 24) & 0xFF))
        out.append(UInt8((val >> 16) & 0xFF))
        out.append(UInt8((val >> 8) & 0xFF))
        out.append(UInt8(val & 0xFF))
    }

}
