import Foundation

public class Encoder {
    public let width: Int
    public let height: Int
    public let maxbitrate: Int
    public let framerate: Int
    public let zeroThreshold: Int
    public let keyint: Int
    public let sceneChangeThreshold: Int
    public let isOne: Bool
    
    private var prevReconstructed: PlaneData420? = nil
    private var framesSinceKeyframe = 0
    private var qt: QuantizationTable? = nil
    private let mbSize = 64
    private var frameIndex = 0
    
    // Rate Control
    private var rateController: RateController
    
    public init(width: Int, height: Int, maxbitrate: Int, framerate: Int = 30, zeroThreshold: Int = 3, keyint: Int = 60, sceneChangeThreshold: Int = 8, isOne: Bool = false) {
        self.width = width
        self.height = height
        self.maxbitrate = maxbitrate
        self.framerate = framerate
        self.zeroThreshold = zeroThreshold
        self.keyint = keyint
        self.sceneChangeThreshold = sceneChangeThreshold
        self.isOne = isOne
        
        self.rateController = RateController(maxbitrate: maxbitrate, framerate: framerate, keyint: keyint)
    }
    
    #if (arch(arm64) || arch(x86_64) || arch(wasm32))
    public func encode(image: YCbCrImage) async throws -> [UInt8] {
        var out: [UInt8] = []
        let curr = toPlaneData420(images: [image])[0]
        
        var forceIFrame = false
        var predictedPlane: PlaneData420? = nil
        var motionTree = MotionTree(ctuNodes: [], width: 0, height: 0)
        var meanSAD: Int = 0
        var maxBlockSAD: Int = 0
        
        if keyint <= framesSinceKeyframe || prevReconstructed == nil {
            forceIFrame = true
        } else {
            guard let prev = prevReconstructed else { throw NSError(domain: "vevc.Encoder", code: 2, userInfo: nil) }
            
            let detector = SceneChangeDetector(threshold: sceneChangeThreshold * 4) // Raw SAD requires higher threshold than Residual SAD
            if detector.isSceneChanged(prev: prev, curr: curr) {
                forceIFrame = true
                meanSAD = 999999
                debugLog("[Frame \(frameIndex)] Adaptive GOP: Forced I-Frame due to Scene Change (Fast Detector)")
            } else {
                motionTree = estimateMotionQuadtree(curr: curr, prev: prev)
                let predicted = await applyMotionQuadtree(prev: prev, tree: motionTree)
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
                let qtY = QuantizationTable(baseStep: max(1, Int(qt.step)), isChroma: false, layerIndex: 0, isOne: true)
                let qtC = QuantizationTable(baseStep: max(1, Int(qt.step) * 2), isChroma: true, layerIndex: 0, isOne: true)
                
                let (layer0, reconstructed) = try await encodePlaneBase32Causal(pd: curr, predictedPd: nil, layer: 0, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
                
                out.append(contentsOf: [0x56, 0x45, 0x4F, 0x49]) // VEOI
                
                var chunkOut: [UInt8] = []
                chunkOut.append(contentsOf: layer0)
                
                appendUInt32BE(&out, UInt32(chunkOut.count))
                out.append(contentsOf: chunkOut)
                debugLog("[Frame \(frameIndex)] I-Frame (One): \(chunkOut.count) bytes (\(String(format: "%.2f", Double(chunkOut.count) / 1024.0)) KB)")
                
                rateController.consumeIFrame(bits: chunkOut.count * 8, qStep: Int(qtY.step))
                
                prevReconstructed = reconstructed
                framesSinceKeyframe = 1
            } else {
                // Adaptive Quantization (isOne mode)
                let currentSAD = Double(meanSAD)
                let newStepInt = rateController.calculatePFrameQStep(currentSAD: currentSAD, baseStep: Int(qt.step))
                let newStep = Int16(clamping: newStepInt)
                
                // Adaptive Quantization Factor based on frame motion activity (0 to 10)
                let activity = min(10, meanSAD)
                
                // Luma (Y): Scale baseStep from 0.5x (static) to 1.0x (active)
                let stepY = max(1, (Int(newStep) * (10 + activity)) / 20)
                let qtY = QuantizationTable(baseStep: stepY, isChroma: false, layerIndex: 0, isOne: true)
                
                // Chroma (Cb/Cr): Scale baseStep from 1.0x (static) to 2.0x (active)
                let stepC = max(1, (Int(newStep) * (10 + activity)) / 10)
                let qtC = QuantizationTable(baseStep: stepC, isChroma: true, layerIndex: 0, isOne: true)
                
                let (layer0, reconstructedPlane) = try await encodePlaneBase32Causal(pd: curr, predictedPd: predictedPlane, layer: 0, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
                
                out.append(contentsOf: [0x56, 0x45, 0x4F, 0x50]) // VEOP

                var mvBw = EntropyEncoder()
                var grid = MVGrid(width: curr.width, height: curr.height, minSize: 8)
                let mbCols = (curr.width + mbSize - 1) / mbSize
                let mbRows = (curr.height + mbSize - 1) / mbSize
                
                for mbY in 0..<mbRows {
                    let startY = mbY * mbSize
                    for mbX in 0..<mbCols {
                        let startX = mbX * mbSize
                        let nodeIdx = mbY * mbCols + mbX
                        if nodeIdx < motionTree.ctuNodes.count {
                            let node = motionTree.ctuNodes[nodeIdx]
                            encodeMotionQuadtreeNode(node: node, w: curr.width, h: curr.height, startX: startX, startY: startY, size: mbSize, grid: &grid, bw: &mvBw)
                        }
                    }
                }
                mvBw.flush()
                let mvOut = mvBw.getData()
                appendUInt32BE(&out, UInt32(motionTree.ctuNodes.count)) // Keep compatible format header or just count of CTU nodes
                appendUInt32BE(&out, UInt32(mvOut.count))
                out.append(contentsOf: mvOut)

                var chunkOut: [UInt8] = []
                chunkOut.append(contentsOf: layer0)
                
                appendUInt32BE(&out, UInt32(chunkOut.count))
                out.append(contentsOf: chunkOut)
                let totalBytes = chunkOut.count + mvOut.count
                debugLog("[Frame \(frameIndex)] P-Frame (One): \(totalBytes) bytes (MV: \(mvOut.count) bytes, Data: \(chunkOut.count) bytes) CTUs=\(motionTree.ctuNodes.count) meanSAD=\(meanSAD) [Quadtree & LSCP applied]")
                
                rateController.consumePFrame(bits: totalBytes * 8, qStep: Int(qtY.step), sad: Double(meanSAD))
                
                // encodePlaneBase32Causal already adds the predicted plane internally and returns fully reconstructed pixels!
                prevReconstructed = reconstructedPlane
                framesSinceKeyframe += 1
            }
        } else {
            if forceIFrame {
                let qtY = QuantizationTable(baseStep: max(1, Int(qt.step)), isChroma: false, layerIndex: 0, isOne: false)
                let qtC = QuantizationTable(baseStep: max(1, Int(qt.step) * 2), isChroma: true, layerIndex: 0, isOne: false)
                let (bytes, reconstructed) = try await encodeSpatialLayers(pd: curr, predictedPd: nil, maxbitrate: maxbitrate, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
                
                out.append(contentsOf: [0x56, 0x45, 0x56, 0x49])
                appendUInt32BE(&out, UInt32(bytes.count))
                out.append(contentsOf: bytes)
                debugLog("[Frame \(frameIndex)] I-Frame: \(bytes.count) bytes (\(String(format: "%.2f", Double(bytes.count) / 1024.0)) KB)")
                
                rateController.consumeIFrame(bits: bytes.count * 8, qStep: Int(qtY.step))
                
                prevReconstructed = reconstructed
                framesSinceKeyframe = 1
            } else {
                let currentSAD = Double(meanSAD)
                let newStepInt = rateController.calculatePFrameQStep(currentSAD: currentSAD, baseStep: Int(qt.step))
                let newStep = Int16(clamping: newStepInt)
                
                // Adaptive Quantization Factor based on frame motion activity (0 to 10)
                let activity = min(10, meanSAD)
                
                // Luma (Y): Scale baseStep from 0.5x (static) to 1.0x (active)
                let stepY = max(1, (Int(newStep) * (10 + activity)) / 20)
                let qtY = QuantizationTable(baseStep: stepY, isChroma: false, layerIndex: 0, isOne: false)
                
                // Chroma (Cb/Cr): Scale baseStep from 1.0x (static) to 2.0x (active)
                let stepC = max(1, (Int(newStep) * (10 + activity)) / 10)
                let qtC = QuantizationTable(baseStep: stepC, isChroma: true, layerIndex: 0, isOne: false)
                let (bytes, reconstructedResidual) = try await encodeSpatialLayers(pd: curr, predictedPd: predictedPlane, maxbitrate: maxbitrate, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
                
                out.append(contentsOf: [0x56, 0x45, 0x56, 0x50])

                var mvBw = EntropyEncoder()
                var grid = MVGrid(width: curr.width, height: curr.height, minSize: 8)
                let mbCols = (curr.width + mbSize - 1) / mbSize
                let mbRows = (curr.height + mbSize - 1) / mbSize
                
                for mbY in 0..<mbRows {
                    let startY = mbY * mbSize
                    for mbX in 0..<mbCols {
                        let startX = mbX * mbSize
                        let nodeIdx = mbY * mbCols + mbX
                        if nodeIdx < motionTree.ctuNodes.count {
                            let node = motionTree.ctuNodes[nodeIdx]
                            encodeMotionQuadtreeNode(node: node, w: curr.width, h: curr.height, startX: startX, startY: startY, size: mbSize, grid: &grid, bw: &mvBw)
                        }
                    }
                }
                mvBw.flush()
                let mvOut = mvBw.getData()
                appendUInt32BE(&out, UInt32(motionTree.ctuNodes.count))
                appendUInt32BE(&out, UInt32(mvOut.count))
                out.append(contentsOf: mvOut)

                appendUInt32BE(&out, UInt32(bytes.count))
                out.append(contentsOf: bytes)
                let totalBytes = bytes.count + mvOut.count
                
                rateController.consumePFrame(bits: totalBytes * 8, qStep: Int(qtY.step), sad: Double(meanSAD))
                
                if let predicted = predictedPlane {
                    let reconstructed = await addPlanes(residual: reconstructedResidual, predicted: predicted)
                    prevReconstructed = reconstructed
                } else {
                    prevReconstructed = reconstructedResidual
                }
                framesSinceKeyframe += 1
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
