import Foundation

@inline(__always)
func encodeCausalPlaneComponent32(r: Int16Reader, predictedR: Int16Reader?, width: Int, height: Int, qt: QuantizationTable, zeroThreshold: Int, isChroma: Bool) -> ([UInt8], [Int16]) {
    let rowCount = (height + 31) / 32
    let colCount = (width + 31) / 32
    var lastVal: Int16 = 0
    var bwFlags = BypassWriter()
    var encoder = EntropyEncoder()
    
    var reconData = [Int16](repeating: 0, count: width * height)
    var topBuffer = [Int16](repeating: 0, count: 32)
    var leftBuffer = [Int16](repeating: 0, count: 32)
    var predictedBlock = [Int16](repeating: 0, count: 32 * 32)
    var modeWriter = BypassWriter()
    
    for row in 0..<rowCount {
        let h = row * 32
        for col in 0..<colCount {
            let w = col * 32
            
            // 1. Original Extract
            var block = Block2D(width: 32, height: 32)
            block.withView { view in
                r.readBlock(x: w, y: h, width: 32, height: 32, into: &view)
            }
            
            // 2. Prediction
            // Setup buffers for Intra Prediction regardless of frame type
            var hasTop = false
            var hasLeft = false
            if row > 0 {
                hasTop = true
                for x in 0..<32 {
                    let rx = min(w + x, width - 1)
                    topBuffer[x] = reconData[(h - 1) * width + rx]
                }
            }
            if col > 0 {
                hasLeft = true
                for y in 0..<32 {
                    let ry = min(h + y, height - 1)
                    leftBuffer[y] = reconData[ry * width + w - 1]
                }
            }
            
            // Define closure to evaluate Intra modes
            let evaluateIntra = { () -> (IntraPredictor.Mode, Int, [Int16]) in
                var bestMode = IntraPredictor.Mode.dc
                var bestSATD = Int.max
                var bestPred = [Int16](repeating: 0, count: 32 * 32)
                
                let modesToTest: [IntraPredictor.Mode] = [.dc, .vertical, .horizontal, .planar]
                for mode in modesToTest {
                    var pred = [Int16](repeating: 0, count: 32 * 32)
                    IntraPredictor.predict(mode: mode, block: &pred, width: 32, height: 32, top: hasTop ? topBuffer : nil, left: hasLeft ? leftBuffer : nil)
                    
                    var testBlock = Block2D(width: 32, height: 32)
                    for i in 0..<(32*32) {
                        testBlock.data[i] = block.data[i] &- pred[i]
                    }
                    testBlock.withView { view in
                        dwt2d_32(&view)
                        var ll1 = BlockView(base: view.base, width: 16, height: 16, stride: view.stride)
                        dwt2d_16(&ll1)
                        var ll2 = BlockView(base: view.base, width: 8, height: 8, stride: view.stride)
                        dwt2d_8(&ll2)
                    }
                    var satd = 0
                    testBlock.withView { view in
                        var ll3Sum = 0, hl3Sum = 0, lh3Sum = 0, hh3Sum = 0
                        var hl2Sum = 0, lh2Sum = 0, hh2Sum = 0
                        var hl1Sum = 0, lh1Sum = 0, hh1Sum = 0
                        
                        let base = view.base
                        for y in 0..<32 {
                            let ptr = base.advanced(by: y * 32)
                            for x in 0..<32 {
                                let v = Int(abs(Int32(ptr[x])))
                                if y < 16 && x < 16 {
                                    if y < 8 && x < 8 {
                                        if y < 4 && x < 4 { ll3Sum += v }
                                        else if x >= 4 && y < 4 { hl3Sum += v }
                                        else if x < 4 && y >= 4 { lh3Sum += v }
                                        else { hh3Sum += v }
                                    } else {
                                        if x >= 8 && y < 8 { hl2Sum += v }
                                        else if x < 8 && y >= 8 { lh2Sum += v }
                                        else { hh2Sum += v }
                                    }
                                } else {
                                    if x >= 16 && y < 16 { hl1Sum += v }
                                    else if x < 16 && y >= 16 { lh1Sum += v }
                                    else { hh1Sum += v }
                                }
                            }
                        }
                        
                        let level3 = ll3Sum + (hl3Sum * 1) + (lh3Sum * 1) + (hh3Sum * 2)
                        let level2 = (hl2Sum * 4) + (lh2Sum * 4) + (hh2Sum * 6)
                        let level1 = (hl1Sum * 8) + (lh1Sum * 8) + (hh1Sum * 16)
                        satd = level3 + level2 + level1
                        if mode != .dc { satd += 64 }
                    }
                    if satd < bestSATD {
                        bestSATD = satd
                        bestMode = mode
                        bestPred = pred
                    }
                }
                return (bestMode, bestSATD, bestPred)
            }
            
            if let pR = predictedR {
                // P-Frame: Evaluate Inter vs Intra
                var pBlock = Block2D(width: 32, height: 32)
                pBlock.withView { view in
                    pR.readBlock(x: w, y: h, width: 32, height: 32, into: &view)
                }
                var interSAD = 0
                for i in 0..<(32*32) {
                    interSAD += Int(abs(Int32(block.data[i]) - Int32(pBlock.data[i])))
                }
                
                // Adaptive Threshold for Early Exit (Save CPU)
                if interSAD < 500 {
                    // Inter is highly likely to be optimal
                    predictedBlock = pBlock.data
                    modeWriter.writeBit(false) // isInter = true
                } else {
                    let intraResult = evaluateIntra()
                    // Apply penalty to Intra because it takes more bits (1 bit vs 3 bits)
                    if interSAD <= intraResult.1 + 128 {
                        predictedBlock = pBlock.data
                        modeWriter.writeBit(false) // isInter = true
                    } else {
                        predictedBlock = intraResult.2
                        modeWriter.writeBit(true) // isIntra = true
                        let m = intraResult.0.rawValue
                        modeWriter.writeBit((m & 0b10) != 0)
                        modeWriter.writeBit((m & 0b01) != 0)
                    }
                }
            } else {
                // I-Frame: Only Intra
                let intraResult = evaluateIntra()
                predictedBlock = intraResult.2
                let m = intraResult.0.rawValue
                modeWriter.writeBit((m & 0b10) != 0)
                modeWriter.writeBit((m & 0b01) != 0)
            }
            
            // 3. Subtract Prediction (Spatial Domain)
            for i in 0..<(32*32) {
                block.data[i] = block.data[i] &- predictedBlock[i]
            }
            
            // 4. Transform
            block.withView { view in
                dwt2d_32(&view)
                var ll1 = BlockView(base: view.base, width: 16, height: 16, stride: view.stride)
                dwt2d_16(&ll1)
                var ll2 = BlockView(base: view.base, width: 8, height: 8, stride: view.stride)
                dwt2d_8(&ll2)
            }
            
            // 5. Evaluate Effectively Zero (MUST happen BEFORE quantization!)
            let isZero = block.data.withUnsafeMutableBufferPointer { ptr in
                isEffectivelyZeroBase32(data: ptr, threshold: zeroThreshold)
            }
            
            // 6. Quantize (Use cascaded hierarchical quantization)
            quantizeCascaded32(block: &block, qt: qt, isChroma: isChroma)
            
            // 7. Entropy Encode & apply LSCP Zero-clearing
            
            if isZero {
                bwFlags.writeBit(true)
                block.withView { $0.clearAll() }
                lastVal = 0
            } else {
                bwFlags.writeBit(false)
                block.withView { view in
                    let hl1 = BlockView(base: view.base.advanced(by: 16), width: 16, height: 16, stride: view.stride)
                    let lh1 = BlockView(base: view.base.advanced(by: 16 * view.stride), width: 16, height: 16, stride: view.stride)
                    let hh1 = BlockView(base: view.base.advanced(by: 16 * view.stride + 16), width: 16, height: 16, stride: view.stride)
                    
                    let hl2 = BlockView(base: view.base.advanced(by: 8), width: 8, height: 8, stride: view.stride)
                    let lh2 = BlockView(base: view.base.advanced(by: 8 * view.stride), width: 8, height: 8, stride: view.stride)
                    let hh2 = BlockView(base: view.base.advanced(by: 8 * view.stride + 8), width: 8, height: 8, stride: view.stride)
                    
                    let ll3 = BlockView(base: view.base, width: 4, height: 4, stride: view.stride)
                    let hl3 = BlockView(base: view.base.advanced(by: 4), width: 4, height: 4, stride: view.stride)
                    let lh3 = BlockView(base: view.base.advanced(by: 4 * view.stride), width: 4, height: 4, stride: view.stride)
                    let hh3 = BlockView(base: view.base.advanced(by: 4 * view.stride + 4), width: 4, height: 4, stride: view.stride)
                    
                    blockEncodeDPCM4(encoder: &encoder, block: ll3, lastVal: &lastVal)
                    blockEncode4(encoder: &encoder, block: hl3)
                    blockEncode4(encoder: &encoder, block: lh3)
                    blockEncode4(encoder: &encoder, block: hh3)
                    
                    blockEncode8(encoder: &encoder, block: hl2)
                    blockEncode8(encoder: &encoder, block: lh2)
                    blockEncode8(encoder: &encoder, block: hh2)
                    
                    blockEncode16(encoder: &encoder, block: hl1)
                    blockEncode16(encoder: &encoder, block: lh1)
                    blockEncode16(encoder: &encoder, block: hh1)
                }
            }
            
            // 7. Inverse Quantize
            dequantizeCascaded32(block: &block, qt: qt, isChroma: isChroma)
            
            block.withView { view in
                var ll2 = BlockView(base: view.base, width: 8, height: 8, stride: view.stride)
                invDwt2d_8(&ll2)
                var ll1 = BlockView(base: view.base, width: 16, height: 16, stride: view.stride)
                invDwt2d_16(&ll1)
                invDwt2d_32(&view)
            }
            
            // 8. Add Prediction to get Reconstructed pixels
            for i in 0..<(32*32) {
                block.data[i] = block.data[i] &+ predictedBlock[i]
            }
            
            // 9. Write to reconData
            let validEndY = min(height, h + 32)
            let validEndX = min(width, w + 32)
            let loopH = validEndY - h
            let loopW = validEndX - w
            
            reconData.withUnsafeMutableBufferPointer { dstBuf in
                guard let dstBase = dstBuf.baseAddress else { return }
                block.withView { v in
                    for dy in 0..<loopH {
                        let srcPtr = v.rowPointer(y: dy)
                        let destPtr = dstBase.advanced(by: (h + dy) * width + w)
                        destPtr.update(from: srcPtr, count: loopW)
                    }
                }
            }
        }
    }
    bwFlags.flush()
    encoder.flush()
    var buf = bwFlags.bytes
    buf.append(contentsOf: encoder.getData())
    
    var out: [UInt8] = []
    modeWriter.flush()
    let modeBytes = modeWriter.bytes
    appendUInt32BE(&out, UInt32(modeBytes.count))
    out.append(contentsOf: modeBytes)
    out.append(contentsOf: buf)
    
    return (out, reconData)
}

@inline(__always)
func encodePlaneBase32Causal(pd: PlaneData420, predictedPd: PlaneData420?, layer: UInt8, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int) async throws -> ([UInt8], PlaneData420) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    
    async let taskBufY = { encodeCausalPlaneComponent32(r: pd.rY, predictedR: predictedPd?.rY, width: dx, height: dy, qt: qtY, zeroThreshold: zeroThreshold, isChroma: false) }()
    async let taskBufCb = { encodeCausalPlaneComponent32(r: pd.rCb, predictedR: predictedPd?.rCb, width: cbDx, height: cbDy, qt: qtC, zeroThreshold: zeroThreshold, isChroma: true) }()
    async let taskBufCr = { encodeCausalPlaneComponent32(r: pd.rCr, predictedR: predictedPd?.rCr, width: cbDx, height: cbDy, qt: qtC, zeroThreshold: zeroThreshold, isChroma: true) }()
    
    let (bufY, reconY) = await taskBufY
    var mutReconY = reconY
    DeblockingFilter.apply(plane: &mutReconY, width: dx, height: dy, blockSize: 32, qStep: Int(qtY.step))
    
    let (bufCb, reconCb) = await taskBufCb
    var mutReconCb = reconCb
    DeblockingFilter.apply(plane: &mutReconCb, width: cbDx, height: cbDy, blockSize: 32, qStep: Int(qtC.step))
    
    let (bufCr, reconCr) = await taskBufCr
    var mutReconCr = reconCr
    DeblockingFilter.apply(plane: &mutReconCr, width: cbDx, height: cbDy, blockSize: 32, qStep: Int(qtC.step))
    
    let reconstructed = PlaneData420(width: dx, height: dy, y: mutReconY, cb: mutReconCb, cr: mutReconCr)
    
    var out: [UInt8] = []
    out.append(contentsOf: [0x56, 0x45, 0x56, 0x43, layer])
    appendUInt16BE(&out, UInt16(dx))
    appendUInt16BE(&out, UInt16(dy))
    appendUInt16BE(&out, UInt16(qtY.step))
    appendUInt16BE(&out, UInt16(qtC.step))
    
    appendUInt32BE(&out, UInt32(bufY.count))
    out.append(contentsOf: bufY)
    appendUInt32BE(&out, UInt32(bufCb.count))
    out.append(contentsOf: bufCb)
    appendUInt32BE(&out, UInt32(bufCr.count))
    out.append(contentsOf: bufCr)
    
    return (out, reconstructed)
}
