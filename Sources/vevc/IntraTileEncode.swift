import Foundation

// MARK: - Intra Tile Encode

/// Encodes an image using the IntraTile (Profile 0x02) pipeline.
/// - Parameters:
///   - pd: PlaneData420 containing Y, Cb, Cr planes
///   - pool: BlockViewPool for memory management
///   - qtY: Base quantization table for Y
///   - qtC: Base quantization table for Cb/Cr
/// - Returns: Serialized byte array for the whole frame payload
func encodeIntraTiles(
    pd: PlaneData420,
    pool: BlockViewPool,
    qtY: QuantizationTable,
    qtC: QuantizationTable
) async throws -> [UInt8] {
    let mapY = computeIntraTileMap(width: pd.width, height: pd.height)
    let mapC = computeIntraTileMap(width: (pd.width + 1) / 2, height: (pd.height + 1) / 2)
    
    // We will accumulate entropy encoded data for Y, Cb, Cr
    var tileYBytes: [UInt8] = []
    var tileCbBytes: [UInt8] = []
    var tileCrBytes: [UInt8] = []
    
    // Encode Y plane
    for t in mapY {
        let tileBytes = try encodeSingleIntraTile(r: pd.rY, tileInfo: t, isChroma: false, qt: qtY, pool: pool)
        tileYBytes.append(contentsOf: tileBytes)
    }
    
    // Encode Cb plane
    for t in mapC {
        let tileBytes = try encodeSingleIntraTile(r: pd.rCb, tileInfo: t, isChroma: true, qt: qtC, pool: pool)
        tileCbBytes.append(contentsOf: tileBytes)
    }
    
    // Encode Cr plane
    for t in mapC {
        let tileBytes = try encodeSingleIntraTile(r: pd.rCr, tileInfo: t, isChroma: true, qt: qtC, pool: pool)
        tileCrBytes.append(contentsOf: tileBytes)
    }
    
    // Serialize as VEVCLayerData (using Layer0 payload as the whole frame payload)
    let payload = VEVCLayerData.serialize(
        qtYStep: UInt16(qtY.step),
        qtCStep: UInt16(qtC.step),
        bufY: tileYBytes,
        bufCb: tileCbBytes,
        bufCr: tileCrBytes
    )
    
    return payload
}

private func encodeSingleIntraTile(
    r: Int16Reader,
    tileInfo: IntraTileRect,
    isChroma: Bool,
    qt: QuantizationTable,
    pool: BlockViewPool
) throws -> [UInt8] {
    let block = pool.get(width: tileInfo.size, height: tileInfo.size)
    defer { pool.put(block) }
    
    // Copy data from reader to padded block
    r.readBlock(x: tileInfo.x, y: tileInfo.y, width: tileInfo.size, height: tileInfo.size, into: block)
    
    // 2D DWT
    intraDwt2D(base: block.base, size: tileInfo.size, stride: block.stride, levels: tileInfo.levels, filter: tileInfo.filter)
    
    // Entropy Encode Subbands
    var encoder = EntropyEncoder()
    var lastVal: Int16 = 0
    
    for l in (1...tileInfo.levels).reversed() {
        let (ll, hl, lh, hh) = getIntraSubbands(view: block, level: l)
        
        if l == tileInfo.levels {
            // Encode LL
            let qLL = QuantizationTable.quantizerForIntraTile(baseStep: Int(qt.step), filter: tileInfo.filter, level: l, subband: .LL, isChroma: isChroma)
            quantizeSIMD(ll, q: qLL)
            encodeSubbandGrid(view: ll, encoder: &encoder, isLL: true, lastVal: &lastVal)
        }
        
        let qHL = QuantizationTable.quantizerForIntraTile(baseStep: Int(qt.step), filter: tileInfo.filter, level: l, subband: .HL, isChroma: isChroma)
        let qLH = QuantizationTable.quantizerForIntraTile(baseStep: Int(qt.step), filter: tileInfo.filter, level: l, subband: .LH, isChroma: isChroma)
        let qHH = QuantizationTable.quantizerForIntraTile(baseStep: Int(qt.step), filter: tileInfo.filter, level: l, subband: .HH, isChroma: isChroma)
        
        quantizeSIMD(hl, q: qHL)
        encodeSubbandGrid(view: hl, encoder: &encoder, isLL: false, lastVal: &lastVal)
        
        quantizeSIMD(lh, q: qLH)
        encodeSubbandGrid(view: lh, encoder: &encoder, isLL: false, lastVal: &lastVal)
        
        quantizeSIMD(hh, q: qHH)
        encodeSubbandGrid(view: hh, encoder: &encoder, isLL: false, lastVal: &lastVal)
    }
    
    encoder.flush()
    return encoder.getData(selectModel: unifiedSelectModel)
}

/// Helper to get LL, HL, LH, HH subbands for a specific DWT level
private func getIntraSubbands(view: BlockView, level: Int) -> (ll: BlockView, hl: BlockView, lh: BlockView, hh: BlockView) {
    let scale = 1 << level
    let w = view.width / scale
    let h = view.height / scale
    
    let ll = BlockView(base: view.base, width: w, height: h, stride: view.stride)
    let hl = BlockView(base: view.base.advanced(by: w), width: w, height: h, stride: view.stride)
    let lh = BlockView(base: view.base.advanced(by: h * view.stride), width: w, height: h, stride: view.stride)
    let hh = BlockView(base: view.base.advanced(by: h * view.stride + w), width: w, height: h, stride: view.stride)
    
    return (ll, hl, lh, hh)
}

/// Encodes a subband by dividing it into 16x16 grid blocks (or smaller if necessary)
private func encodeSubbandGrid(view: BlockView, encoder: inout EntropyEncoder, isLL: Bool, lastVal: inout Int16) {
    let bw = view.width
    let bh = view.height
    let colCount = (bw + 15) / 16
    let rowCount = (bh + 15) / 16
    
    for y in 0..<rowCount {
        for x in 0..<colCount {
            let cx = x * 16
            let cy = y * 16
            let cw = min(16, bw - cx)
            let ch = min(16, bh - cy)
            
            let subView = BlockView(base: view.base.advanced(by: cy * view.stride + cx), width: cw, height: ch, stride: view.stride)
            
            // Check if block is entirely zero
            let isZero = isEffectivelyZero(data: subView.base, width: cw, height: ch, stride: subView.stride)
            encoder.encodeBypass(binVal: isZero ? 1 : 0)
            
            if !isZero {
                if cw == 16 && ch == 16 {
                    if isLL {
                        blockEncodeDPCM16(encoder: &encoder, block: subView, lastVal: &lastVal)
                    } else {
                        blockEncode16V(encoder: &encoder, block: subView)
                    }
                } else if cw == 8 && ch == 8 {
                    if isLL {
                        blockEncodeDPCM8(encoder: &encoder, block: subView, lastVal: &lastVal)
                    } else {
                        blockEncode8V(encoder: &encoder, block: subView)
                    }
                } else if cw == 4 && ch == 4 {
                    if isLL {
                        blockEncodeDPCM4(encoder: &encoder, block: subView, lastVal: &lastVal)
                    } else {
                        blockEncode4V(encoder: &encoder, block: subView)
                    }
                } else {
                    // Fallback for non-standard sizes: generic block encoding
                    blockEncodeGeneric(encoder: &encoder, block: subView)
                }
            }
        }
    }
}

// Fallback generic encoding if sizes are not 16x16, 8x8, or 4x4
private func blockEncodeGeneric(encoder: inout EntropyEncoder, block: BlockView) {
    // A simple raster scan encoding for non-standard blocks
    let width = block.width
    let height = block.height
    var run = 0
    var prevVal: Int16 = 0
    
    for y in 0..<height {
        let ptr = block.rowPointer(y: y)
        for x in 0..<width {
            let val = ptr[x]
            if val == 0 {
                run += 1
            } else {
                encodeCoeffRun(val: val, encoder: &encoder, run: run, context: getContext(prevVal: prevVal, isParentZero: false))
                prevVal = val
                run = 0
            }
        }
    }
}

private func isEffectivelyZero(data: UnsafeMutablePointer<Int16>, width: Int, height: Int, stride: Int) -> Bool {
    for y in 0..<height {
        let ptr = data.advanced(by: y * stride)
        for x in 0..<width {
            if ptr[x] != 0 {
                return false
            }
        }
    }
    return true
}
