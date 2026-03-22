import Foundation

func reconstructCausalPlaneComponent32(blocks: [Block2D], width: Int, height: Int, qt: QuantizationTable, predictedR: Int16Reader?, modes: [IntraPredictor.Mode]?) -> [Int16] {
    let colCount = (width + 31) / 32
    var reconData = [Int16](repeating: 0, count: width * height)
    var topBuffer = [Int16](repeating: 0, count: 32)
    var leftBuffer = [Int16](repeating: 0, count: 32)
    var predictedBlock = [Int16](repeating: 0, count: 32 * 32)
    
    for idx in blocks.indices {
        var block = blocks[idx]
        let row = idx / colCount
        let col = idx % colCount
        let h = row * 32
        let w = col * 32
        
        // 1. Prediction
        if let pR = predictedR {
            // Inter Prediction
            var pBlock = Block2D(width: 32, height: 32)
            pBlock.withView { view in
                pR.readBlock(x: w, y: h, width: 32, height: 32, into: &view)
            }
            for i in 0..<(32*32) { predictedBlock[i] = pBlock.data[i] }
        } else {
            // Intra Prediction
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
            let mode = modes?[idx] ?? .dc
            IntraPredictor.predict(mode: mode, block: &predictedBlock, width: 32, height: 32, top: hasTop ? topBuffer : nil, left: hasLeft ? leftBuffer : nil)
        }
        
        // 2. Inverse Quantization
        dequantizeCascaded32(block: &block, qt: qt, isChroma: false)  // isChroma check not needed for qt but we passed it usually. Wait, in encode we used `isChroma`. Is there a way to pass it? Let's just pass `isChroma: qt.isChroma`.
        
        // 3. Inverse DWT
        block.withView { view in
            var ll2 = BlockView(base: view.base, width: 8, height: 8, stride: view.stride)
            invDwt2d_8(&ll2)
            var ll1 = BlockView(base: view.base, width: 16, height: 16, stride: view.stride)
            invDwt2d_16(&ll1)
            invDwt2d_32(&view)
        }
        
        // 4. Add Prediction
        for i in 0..<(32*32) {
            block.data[i] = block.data[i] &+ predictedBlock[i]
        }
        
        // 5. Write to reconData
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
    return reconData
}

@inline(__always)
func parseDecodePlaneBase32Causal(data: [UInt8], dx: Int, dy: Int, qtY: QuantizationTable, qtC: QuantizationTable, predictedPd: PlaneData420?) async throws -> PlaneData420 {
    var inx = 0
    let ySize = Int(try readUInt32BEFromBytes(data, offset: &inx))
    let yData = Array(data[inx..<inx+ySize])
    inx += ySize
    
    let cbSize = Int(try readUInt32BEFromBytes(data, offset: &inx))
    let cbData = Array(data[inx..<inx+cbSize])
    inx += cbSize
    
    let crSize = Int(try readUInt32BEFromBytes(data, offset: &inx))
    let crData = Array(data[inx..<inx+crSize])
    inx += crSize
    
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    
    @inline(__always)
    func parseComponent(data: [UInt8], dx: Int, dy: Int, pR: Int16Reader?, qt: QuantizationTable) throws -> [Int16] {
        var inx = 0
        var modes: [IntraPredictor.Mode]? = nil
        if pR == nil {
            let modeBytesCount = Int(try readUInt32BEFromBytes(data, offset: &inx))
            let modeBytes = Array(data[inx..<inx+modeBytesCount])
            inx += modeBytesCount
            
            var modeReader = BypassReader(data: modeBytes)
            let blockCount = ((dy + 31) / 32) * ((dx + 31) / 32)
            var mList = [IntraPredictor.Mode]()
            mList.reserveCapacity(blockCount)
            for _ in 0..<blockCount {
                let b1 = modeReader.readBit() ? 1 : 0
                let b0 = modeReader.readBit() ? 1 : 0
                let mVal = UInt8((b1 << 1) | b0)
                mList.append(IntraPredictor.Mode(rawValue: mVal) ?? .dc)
            }
            modes = mList
        }
        
        let subbandsData = Array(data[inx...])
        let blockCount = ((dy + 31) / 32) * ((dx + 31) / 32)
        var blocks: [Block2D] = []
        blocks.reserveCapacity(blockCount)
        for _ in 0..<blockCount {
            blocks.append(Block2D(width: 32, height: 32))
        }
        try decodeCascadedPlaneSubbands32(data: subbandsData, blocks: &blocks)
        return reconstructCausalPlaneComponent32(blocks: blocks, width: dx, height: dy, qt: qt, predictedR: pR, modes: modes)
    }

    async let taskY = {
        return try parseComponent(data: yData, dx: dx, dy: dy, pR: predictedPd?.rY, qt: qtY)
    }()
    async let taskCb = {
        return try parseComponent(data: cbData, dx: cbDx, dy: cbDy, pR: predictedPd?.rCb, qt: qtC)
    }()
    async let taskCr = {
        return try parseComponent(data: crData, dx: cbDx, dy: cbDy, pR: predictedPd?.rCr, qt: qtC)
    }()
    
    let reconY = try await taskY
    var mutReconY = reconY
    DeblockingFilter.apply(plane: &mutReconY, width: dx, height: dy, blockSize: 32, qStep: Int(qtY.step))
    
    let reconCb = try await taskCb
    var mutReconCb = reconCb
    DeblockingFilter.apply(plane: &mutReconCb, width: cbDx, height: cbDy, blockSize: 32, qStep: Int(qtC.step))
    
    let reconCr = try await taskCr
    var mutReconCr = reconCr
    DeblockingFilter.apply(plane: &mutReconCr, width: cbDx, height: cbDy, blockSize: 32, qStep: Int(qtC.step))
    
    return PlaneData420(width: dx, height: dy, y: mutReconY, cb: mutReconCb, cr: mutReconCr)
}

@inline(__always)
func decodeBase32Causal(r: [UInt8], layer: UInt8, predictedPd: PlaneData420?) async throws -> PlaneData420 {
    var offset = 0
    guard (offset + 5) <= r.count else { throw DecodeError.insufficientData }
    let header = Array(r[offset..<(offset + 5)])
    offset += 5
    
    guard header[0] == 0x56 && header[1] == 0x45 && header[2] == 0x56 && header[3] == 0x43 else {
         throw DecodeError.invalidHeader
    }
    guard header[4] == layer else {
        throw DecodeError.invalidLayerNumber
    }
    
    let dx = Int(try readUInt16BEFromBytes(r, offset: &offset))
    let dy = Int(try readUInt16BEFromBytes(r, offset: &offset))
    let qtY = QuantizationTable(baseStep: Int(try readUInt16BEFromBytes(r, offset: &offset)), isChroma: false, layerIndex: Int(layer), isOne: true)
    let qtC = QuantizationTable(baseStep: Int(try readUInt16BEFromBytes(r, offset: &offset)), isChroma: true, layerIndex: Int(layer), isOne: true)
    
    let dataSlice = Array(r[offset...])
    return try await parseDecodePlaneBase32Causal(data: dataSlice, dx: dx, dy: dy, qtY: qtY, qtC: qtC, predictedPd: predictedPd)
}
