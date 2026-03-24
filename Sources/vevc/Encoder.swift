import Foundation

@inline(__always)
private func estimateRiceBitsDPCM4(block: BlockView, lastVal: inout Int16) -> Int {
    let count = 4 * 4
    let ptr0 = block.rowPointer(y: 0)
    let ptr1 = block.rowPointer(y: 1)
    let ptr2 = block.rowPointer(y: 2)
    let ptr3 = block.rowPointer(y: 3)
    
    @inline(__always)
    func errorMED(_ x: Int16, _ a: Int16, _ b: Int16, _ c: Int16) -> Int {
        let ia = Int(a), ib = Int(b), ic = Int(c)
        let predicted: Int
        if ia <= ic && ib <= ic {
            predicted = min(ia, ib)
        } else if ic <= ia && ic <= ib {
            predicted = max(ia, ib)
        } else {
            predicted = ia + ib - ic
        }
        return abs(Int(x) - predicted)
    }

    var sumDiffAbs = abs(Int(ptr0[0]) - Int(lastVal))
    sumDiffAbs += abs(Int(ptr0[1]) - Int(ptr0[0]))
    sumDiffAbs += abs(Int(ptr0[2]) - Int(ptr0[1]))
    sumDiffAbs += abs(Int(ptr0[3]) - Int(ptr0[2]))

    sumDiffAbs += abs(Int(ptr1[0]) - Int(ptr0[0]))
    sumDiffAbs += errorMED(ptr1[1], ptr1[0], ptr0[1], ptr0[0])
    sumDiffAbs += errorMED(ptr1[2], ptr1[1], ptr0[2], ptr0[1])
    sumDiffAbs += errorMED(ptr1[3], ptr1[2], ptr0[3], ptr0[2])
    
    sumDiffAbs += abs(Int(ptr2[0]) - Int(ptr1[0]))
    sumDiffAbs += errorMED(ptr2[1], ptr2[0], ptr1[1], ptr1[0])
    sumDiffAbs += errorMED(ptr2[2], ptr2[1], ptr1[2], ptr1[1])
    sumDiffAbs += errorMED(ptr2[3], ptr2[2], ptr1[3], ptr1[2])

    sumDiffAbs += abs(Int(ptr3[0]) - Int(ptr2[0]))
    sumDiffAbs += errorMED(ptr3[1], ptr3[0], ptr2[1], ptr2[0])
    sumDiffAbs += errorMED(ptr3[2], ptr3[1], ptr2[2], ptr2[1])
    sumDiffAbs += errorMED(ptr3[3], ptr3[2], ptr2[3], ptr2[2])

    lastVal = ptr3[3]
    
    let meanInt = sumDiffAbs / count
    let k: Int
    if meanInt < 1 {
        k = 0
    } else {
        k = (Int.bitWidth - 1) - meanInt.leadingZeroBitCount
    }
    
    let divisorShift = max(0, k - 1)
    let bodyBits = sumDiffAbs >> divisorShift
    let headerBits = count * (1 + k)
    
    return bodyBits + headerBits
}

@inline(__always)
private func estimateRiceBitsDPCM16(block: BlockView, lastVal: inout Int16) -> Int {
    let count = 16 * 16
    
    @inline(__always)
    func errorMED(_ x: Int16, _ a: Int16, _ b: Int16, _ c: Int16) -> Int {
        let ia = Int(a), ib = Int(b), ic = Int(c)
        let predicted: Int
        if ia <= ic && ib <= ic {
            predicted = min(ia, ib)
        } else if ic <= ia && ic <= ib {
            predicted = max(ia, ib)
        } else {
            predicted = ia + ib - ic
        }
        return abs(Int(x) - predicted)
    }

    var sumDiffAbs = 0
    var last = lastVal
    for y in 0..<16 {
        let ptrY = block.rowPointer(y: y)
        if y == 0 {
            for x in 0..<16 {
                if x == 0 {
                    sumDiffAbs += abs(Int(ptrY[0]) - Int(last))
                } else {
                    sumDiffAbs += abs(Int(ptrY[x]) - Int(ptrY[x - 1]))
                }
            }
        } else {
            let ptrPrevY = block.rowPointer(y: y - 1)
            for x in 0..<16 {
                if x == 0 {
                    sumDiffAbs += abs(Int(ptrY[0]) - Int(ptrPrevY[0]))
                } else {
                    sumDiffAbs += errorMED(ptrY[x], ptrY[x - 1], ptrPrevY[x], ptrPrevY[x - 1])
                }
            }
        }
        last = ptrY[15]
    }
    lastVal = last
    
    let meanInt = sumDiffAbs / count
    let k: Int
    if meanInt < 1 {
        k = 0
    } else {
        k = (Int.bitWidth - 1) - meanInt.leadingZeroBitCount
    }
    
    let divisorShift = max(0, k - 1)
    let bodyBits = sumDiffAbs >> divisorShift
    let headerBits = count * (1 + k)
    
    return bodyBits + headerBits
}

@inline(__always)
private func measureBlockBits8(block: inout Block2D, qt: QuantizationTable) -> Int {
    var sub = block.withView { view in
        return dwt2d_8_sb(&view)
    }
    
    quantizeSIMD(&sub.ll, q: qt.qLow)
    quantizeSIMD(&sub.hl, q: qt.qMid)
    quantizeSIMD(&sub.lh, q: qt.qMid)
    quantizeSIMD(&sub.hh, q: qt.qHigh)
    
    let isZero = block.data.withUnsafeMutableBufferPointer { ptr in
        return isEffectivelyZeroBase4(data: ptr, threshold: 0)
    }
    if isZero {
        return 1
    }
    
    var bits = 1
    var lastVal: Int16 = 0
    bits += estimateRiceBitsDPCM4(block: sub.ll, lastVal: &lastVal)
    bits += estimateRiceBits4(block: sub.hl)
    bits += estimateRiceBits4(block: sub.lh)
    bits += estimateRiceBits4(block: sub.hh)
    
    return bits
}

@inline(__always)
private func estimateRiceBits32(block: BlockView) -> Int {
    var sumAbs = 0
    let count = (32 * 32)
    
    for y in 0..<32 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<32 {
            sumAbs += abs(Int(ptr[x]))
        }
    }
    let meanInt = sumAbs / count
    let k: Int
    if meanInt < 1 {
        k = 0
    } else {
        k = (Int.bitWidth - 1) - meanInt.leadingZeroBitCount
    }
    
    let divisorShift = max(0, k - 1)
    let bodyBits = sumAbs >> divisorShift
    let headerBits = count * (1 + k)
    
    return bodyBits + headerBits
}

@inline(__always)
private func estimateRiceBits16(block: BlockView) -> Int {
    var sumAbs = 0
    let count = (16 * 16)
    
    for y in 0..<16 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<16 {
            sumAbs += abs(Int(ptr[x]))
        }
    }
    let meanInt = sumAbs / count
    let k: Int
    if meanInt < 1 {
        k = 0
    } else {
        k = (Int.bitWidth - 1) - meanInt.leadingZeroBitCount
    }
    
    let divisorShift = max(0, k - 1)
    let bodyBits = sumAbs >> divisorShift
    let headerBits = count * (1 + k)
    
    return bodyBits + headerBits
}

@inline(__always)
private func estimateRiceBits8(block: BlockView) -> Int {
    var sumAbs = 0
    let count = (8 * 8)
    
    for y in 0..<8 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<8 {
            sumAbs += abs(Int(ptr[x]))
        }
    }
    let meanInt = sumAbs / count
    let k: Int
    if meanInt < 1 {
        k = 0
    } else {
        k = (Int.bitWidth - 1) - meanInt.leadingZeroBitCount
    }
    
    let divisorShift = max(0, k - 1)
    let bodyBits = sumAbs >> divisorShift
    let headerBits = count * (1 + k)
    
    return bodyBits + headerBits
}

@inline(__always)
private func estimateRiceBits4(block: BlockView) -> Int {
    var sumAbs = 0
    let count = (4 * 4)
    
    for y in 0..<4 {
        let ptr = block.rowPointer(y: y)
        for x in 0..<4 {
            sumAbs += abs(Int(ptr[x]))
        }
    }
    let meanInt = sumAbs / count
    let k: Int
    if meanInt < 1 {
        k = 0
    } else {
        k = (Int.bitWidth - 1) - meanInt.leadingZeroBitCount
    }
    
    let divisorShift = max(0, k - 1)
    let bodyBits = sumAbs >> divisorShift
    let headerBits = count * (1 + k)
    
    return bodyBits + headerBits
}

@inline(__always)
private func measureBlockBits32(block: inout Block2D, qt: QuantizationTable) -> Int {
    var sub = block.withView { view in
        return dwt2d_32_sb(&view)
    }
    
    quantizeSIMD(&sub.ll, q: qt.qLow)
    quantizeSIMD(&sub.hl, q: qt.qMid)
    quantizeSIMD(&sub.lh, q: qt.qMid)
    quantizeSIMD(&sub.hh, q: qt.qHigh)
    
    let isZero = block.data.withUnsafeMutableBufferPointer { ptr in
        return isEffectivelyZeroBase32(data: ptr, threshold: 0)
    }
    if isZero {
        return 1
    }
    
    var bits = 1
    var lastVal: Int16 = 0
    bits += estimateRiceBitsDPCM16(block: sub.ll, lastVal: &lastVal)
    bits += estimateRiceBits16(block: sub.hl)
    bits += estimateRiceBits16(block: sub.lh)
    bits += estimateRiceBits16(block: sub.hh)
    
    return bits
}

@inline(__always)
func estimateQuantization(img: YCbCrImage, targetBits: Int) -> QuantizationTable {
    let probeStep = 64
    let qt = QuantizationTable(baseStep: probeStep)
    
    let w = (img.width / 8)
    let h = (img.height / 8)
    
    let points: [(Int, Int)] = [
        (0, 0),
        ((img.width - w), 0),
        (0, (img.height - h)),
        ((img.width - w), (img.height - h)),
        (((img.width - w) / 2), 0),
        ((img.width - w), ((img.height - h) / 2)),
        (((img.width - w) / 2), (img.height - h)),
        (0, ((img.height - h) / 2)),
    ]
    
    var totalSampleBits = 0
    let reader = ImageReader(img: img)
    @inline(__always)
    func fetchBlockY(reader: ImageReader, x: Int, y: Int, w: Int, h: Int) -> Block2D {
        var block = Block2D(width: w, height: h)
        block.withView { view in
            for i in 0..<h {
                view.setRow(offsetY: i, row: reader.rowY(x: x, y: y + i, size: w))
            }
        }
        return block
    }

    @inline(__always)
    func fetchBlockCb(reader: ImageReader, x: Int, y: Int, w: Int, h: Int) -> Block2D {
        var block = Block2D(width: w, height: h)
        block.withView { view in
            for i in 0..<h {
                view.setRow(offsetY: i, row: reader.rowCb(x: x, y: y + i, size: w))
            }
        }
        return block
    }

    @inline(__always)
    func fetchBlockCr(reader: ImageReader, x: Int, y: Int, w: Int, h: Int) -> Block2D {
        var block = Block2D(width: w, height: h)
        block.withView { view in
            for i in 0..<h {
                view.setRow(offsetY: i, row: reader.rowCr(x: x, y: y + i, size: w))
            }
        }
        return block
    }
    
    for (sx, sy) in points {
        var blockY = fetchBlockY(reader: reader, x: sx, y: sy, w: w, h: h)
        totalSampleBits += measureBlockBits8(block: &blockY, qt: qt)
        
        var blockCb = fetchBlockCb(reader: reader, x: sx, y: sy, w: w, h: h)
        totalSampleBits += measureBlockBits8(block: &blockCb, qt: qt)
        
        var blockCr = fetchBlockCr(reader: reader, x: sx, y: sy, w: w, h: h)
        totalSampleBits += measureBlockBits8(block: &blockCr, qt: qt)
    }
    
    let samplePixels = points.count * (w * h) * 3
    let totalPixels = img.width * img.height * 3
    
    let estimatedTotalBits = Double(totalSampleBits) * (Double(totalPixels) / Double(samplePixels))
        
    let ratio = estimatedTotalBits / Double(targetBits)
    let predictedStep = Double(probeStep) * ratio * 3.5
    let q = min(10000, Int(max(1, predictedStep)))
    
    return QuantizationTable(baseStep: q)
}

public class Encoder {
    public let width: Int
    public let height: Int
    public let maxbitrate: Int
    public let framerate: Int
    public let zeroThreshold: Int
    public let keyint: Int
    public let sceneChangeThreshold: Int
    
    private var prevReconstructed: PlaneData420? = nil
    private var framesSinceKeyframe = 0
    private var qt: QuantizationTable? = nil
    private let mbSize = 64
    private var frameIndex = 0
    
    public let isOne: Bool
    
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

        if keyint <= framesSinceKeyframe || prevReconstructed == nil {
            forceIFrame = true
        } else {
            guard let prev = prevReconstructed else { throw NSError(domain: "vevc.Encoder", code: 2, userInfo: nil) }
            
            let layer0Curr = downscale8x(pd: curr)
            let layer0Prev = downscale8x(pd: prev)
            let rawStats = calculateDownscaledSADStats(layer0Curr: layer0Curr.data, layer0Prev: layer0Prev.data, w: layer0Curr.w, h: layer0Curr.h)
            
            // Layer0 is heavily filtered, so its SAD correctly represents structural changes without noise
            if rawStats.meanSAD > sceneChangeThreshold * 16 {
                forceIFrame = true
                meanSAD = 999999
                debugLog("[Frame \(frameIndex)] Adaptive GOP: Forced I-Frame due to Layer0 Scene Change (\(rawStats.meanSAD))")
            } else {
                motionTree = estimateMotionQuadtree(curr: curr, prev: prev, layer0Curr: layer0Curr, layer0Prev: layer0Prev)
                let predicted = await applyMotionQuadtree(prev: prev, tree: motionTree)
                predictedPlane = predicted
                // Compute actual residual to encode
                let res = await subPlanes(curr: curr, predicted: predicted)
                let resStats = calculateSADAndMaxBlockSAD(res: res, mbSize: mbSize)
                
                // Use structural Layer0 raw SAD as the primary global activity metric for stable Rate Control (AQ)
                meanSAD = rawStats.meanSAD
                
                if sceneChangeThreshold < resStats.meanSAD {
                    forceIFrame = true
                    meanSAD = 999999
                    debugLog("[Frame \(frameIndex)] Adaptive GOP: Forced I-Frame due to high residual meanSAD (\(resStats.meanSAD) > \(sceneChangeThreshold))")
                }
            }
        }
        
        if forceIFrame {
            let targetBits = rateController.beginGOP()
            let baseQt = estimateQuantization(img: image, targetBits: targetBits)
            self.qt = baseQt
        }
        guard let qt = self.qt else { throw NSError(domain: "vevc.Encoder", code: 1, userInfo: nil) }
        
        if forceIFrame {
            let bytes: [UInt8]
            let reconstructed: PlaneData420
            let appliedQtY: QuantizationTable
            
            if isOne {
                let qtY = QuantizationTable(baseStep: max(1, Int(qt.step)), isChroma: false, layerIndex: 0, isOne: true)
                let qtC = QuantizationTable(baseStep: max(1, Int(qt.step) * 2), isChroma: true, layerIndex: 0, isOne: true)
                (bytes, reconstructed) = try await encodePlaneBase32(pd: curr, predictedPd: nil, layer: 0, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
                appliedQtY = qtY
            } else {
                let qtY = QuantizationTable(baseStep: max(1, Int(qt.step)), isChroma: false, layerIndex: 0, isOne: false)
                let qtC = QuantizationTable(baseStep: max(1, Int(qt.step) * 2), isChroma: true, layerIndex: 0, isOne: false)
                (bytes, reconstructed) = try await encodeSpatialLayers(pd: curr, predictedPd: nil, maxbitrate: maxbitrate, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
                appliedQtY = qtY
            }
                if isOne {
                    out.append(contentsOf: [0x56, 0x45, 0x4F, 0x49]) // VEOI
                } else {
                    out.append(contentsOf: [0x56, 0x45, 0x56, 0x49]) // VEVI
                }
                appendUInt32BE(&out, UInt32(bytes.count))
                out.append(contentsOf: bytes)
                debugLog("[Frame \(frameIndex)] I-Frame: \(bytes.count) bytes (\(String(format: "%.2f", Double(bytes.count) / 1024.0)) KB)")
                
                rateController.consumeIFrame(bits: bytes.count * 8, qStep: Int(appliedQtY.step))
                
                prevReconstructed = reconstructed
                framesSinceKeyframe = 1
            } else {
                let currentSAD = Double(meanSAD)
                let newStepInt = rateController.calculatePFrameQStep(currentSAD: currentSAD, baseStep: Int(qt.step))
                let newStep = Int16(clamping: newStepInt)
                
                // Adaptive Quantization Factor based on frame motion activity (0 to 10)
                let activity = min(10, meanSAD)
                
                let bytes: [UInt8]
                let reconstructedResidual: PlaneData420
                let appliedQtY: QuantizationTable
                
                if isOne {
                    let stepY = max(1, (Int(newStep) * (10 + activity)) / 15)
                    let stepC = max(1, (Int(newStep) * (10 + activity)) / 10)
                    let qtY = QuantizationTable(baseStep: stepY, isChroma: false, layerIndex: 0, isOne: true)
                    let qtC = QuantizationTable(baseStep: stepC, isChroma: true, layerIndex: 0, isOne: true)
                    (bytes, reconstructedResidual) = try await encodePlaneBase32(pd: curr, predictedPd: predictedPlane, layer: 0, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
                    appliedQtY = qtY
                } else {
                    // Luma (Y): Scale baseStep from 0.5x (static) to 1.0x (active)
                    let stepY = max(1, (Int(newStep) * (10 + activity)) / 20)
                    let qtY = QuantizationTable(baseStep: stepY, isChroma: false, layerIndex: 0, isOne: false)
                    
                    // Chroma (Cb/Cr): Scale baseStep from 1.0x (static) to 2.0x (active)
                    let stepC = max(1, (Int(newStep) * (10 + activity)) / 10)
                    let qtC = QuantizationTable(baseStep: stepC, isChroma: true, layerIndex: 0, isOne: false)
                    (bytes, reconstructedResidual) = try await encodeSpatialLayers(pd: curr, predictedPd: predictedPlane, maxbitrate: maxbitrate, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
                    appliedQtY = qtY
                }
                    if isOne {
                    out.append(contentsOf: [0x56, 0x45, 0x4F, 0x50]) // VEOP
                } else {
                    out.append(contentsOf: [0x56, 0x45, 0x56, 0x50]) // VEVP
                }

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
                
                rateController.consumePFrame(bits: totalBytes * 8, qStep: Int(appliedQtY.step), sad: Double(meanSAD))
                
                if let predicted = predictedPlane {
                    let reconstructed = await addPlanes(residual: reconstructedResidual, predicted: predicted)
                    prevReconstructed = reconstructed
                } else {
                    prevReconstructed = reconstructedResidual
                }
                framesSinceKeyframe += 1
            }
        

        frameIndex += 1
        return out
    }
    #else
    public func encode(image: YCbCrImage) async throws -> [UInt8] {
        throw EncodeError.unsupportedArchitecture
    }
    #endif

}
