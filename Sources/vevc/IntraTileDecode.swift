import Foundation

// MARK: - Intra Tile Decode

/// Decodes an IntraTile payload into an image.
/// - Parameters:
///   - r: Serialized payload data
///   - pool: BlockViewPool
///   - header: VEVCFileHeader
/// - Returns: Reconstructed Image16
func decodeIntraTiles(
    from r: [UInt8],
    pool: BlockViewPool,
    header: VEVCFileHeader,
    maxLayer: Int
) async throws -> Image16 {
    let mapY = computeIntraTileMap(width: header.width, height: header.height)
    let mapC = computeIntraTileMap(width: (header.width + 1) / 2, height: (header.height + 1) / 2)
    
    // We expect the payload to be serialized as a single VEVCLayerData structure.
    let (qtY, qtC, bufY, bufCb, bufCr) = try VEVCLayerData.deserialize(from: r, layer: 0, layerLabel: "IntraTile", profile: header.profile)
    
    // Decode planes sequentially
    let (yData, releaseY) = try decodeSinglePlane(mapTiles: mapY, buf: bufY, pool: pool, qt: qtY, isChroma: false, maxLayer: maxLayer)
    let (cbData, releaseCb) = try decodeSinglePlane(mapTiles: mapC, buf: bufCb, pool: pool, qt: qtC, isChroma: true, maxLayer: maxLayer)
    let (crData, releaseCr) = try decodeSinglePlane(mapTiles: mapC, buf: bufCr, pool: pool, qt: qtC, isChroma: true, maxLayer: maxLayer)
    
    // YCbCrImage takes ownership of these buffers if needed, or copies them.
    // In VEVC, Image16 directly owns the buffers.
    let img = Image16(width: header.width, height: header.height, y: yData, cb: cbData, cr: crData)
    
    releaseY()
    releaseCb()
    releaseCr()
    
    return img
}

private func decodeSinglePlane(
    mapTiles: [IntraTileRect],
    buf: [UInt8],
    pool: BlockViewPool,
    qt: QuantizationTable,
    isChroma: Bool,
    maxLayer: Int
) throws -> ([Int16], @Sendable () -> Void) {
    let dropLevels = max(0, 2 - maxLayer)
    
    // Total plane data array
    var totalWidth = 0
    var totalHeight = 0
    for t in mapTiles {
        totalWidth = max(totalWidth, t.x + t.size)
        totalHeight = max(totalHeight, t.y + t.size)
    }
    
    let scaledTotalWidth = max(1, totalWidth >> dropLevels)
    let scaledTotalHeight = max(1, totalHeight >> dropLevels)
    var outData = [Int16](repeating: 0, count: scaledTotalWidth * scaledTotalHeight)
    
    try buf.withUnsafeBufferPointer { ptr in
        guard let bufBase = ptr.baseAddress else { return }
        var decoder = try EntropyDecoder(base: bufBase, count: buf.count)
        
        for tileInfo in mapTiles {
            let block = pool.get(width: tileInfo.size, height: tileInfo.size)
            defer { pool.put(block) }
            
            clearBlockRegion(base: block.base, width: block.width, height: block.height, stride: block.stride)
            
            var lastVal: Int16 = 0
            
            // Decode subbands from levels down to 1
            for l in (1...tileInfo.levels).reversed() {
                let (ll, hl, lh, hh) = getIntraSubbands(view: block, level: l)
                
                if l == tileInfo.levels {
                    try decodeSubbandGrid(view: ll, decoder: &decoder, isLL: true, lastVal: &lastVal)
                    let qLL = QuantizationTable.quantizerForIntraTile(baseStep: Int(qt.step), filter: tileInfo.filter, level: l, subband: .LL, isChroma: isChroma)
                    dequantizeSIMD(ll, q: qLL)
                }
                
                try decodeSubbandGrid(view: hl, decoder: &decoder, isLL: false, lastVal: &lastVal)
                let qHL = QuantizationTable.quantizerForIntraTile(baseStep: Int(qt.step), filter: tileInfo.filter, level: l, subband: .HL, isChroma: isChroma)
                dequantizeSIMD(hl, q: qHL)
                
                try decodeSubbandGrid(view: lh, decoder: &decoder, isLL: false, lastVal: &lastVal)
                let qLH = QuantizationTable.quantizerForIntraTile(baseStep: Int(qt.step), filter: tileInfo.filter, level: l, subband: .LH, isChroma: isChroma)
                dequantizeSIMD(lh, q: qLH)
                
                try decodeSubbandGrid(view: hh, decoder: &decoder, isLL: false, lastVal: &lastVal)
                let qHH = QuantizationTable.quantizerForIntraTile(baseStep: Int(qt.step), filter: tileInfo.filter, level: l, subband: .HH, isChroma: isChroma)
                dequantizeSIMD(hh, q: qHH)
            }
            
            // Inverse 2D DWT
            let targetLevels = max(1, tileInfo.levels - dropLevels)
            inverseIntraDwt2D(base: block.base, size: tileInfo.size, stride: block.stride, levels: targetLevels, filter: tileInfo.filter)
            
            // Copy to output array with scaling
            let outputSize = tileInfo.size >> dropLevels
            let outX = tileInfo.x >> dropLevels
            let outY = tileInfo.y >> dropLevels
            
            outData.withUnsafeMutableBufferPointer { outPtr in
                guard let base = outPtr.baseAddress else { return }
                for y in 0..<outputSize {
                    let src = block.rowPointer(y: y)
                    let dst = base.advanced(by: (outY + y) * scaledTotalWidth + outX)
                    
                    if tileInfo.filter == .cdf97 {
                        if dropLevels == 1 {
                            for x in 0..<outputSize {
                                let v = Int32(src[x])
                                dst[x] = Int16(clamping: (v * 5413) >> 14)
                            }
                        } else if dropLevels >= 2 {
                            for x in 0..<outputSize {
                                let v = Int32(src[x])
                                dst[x] = Int16(clamping: (v * 3577) >> 14)
                            }
                        } else {
                            for x in 0..<outputSize {
                                let v = Int32(src[x])
                                dst[x] = Int16(clamping: (v + 1) >> 1)
                            }
                        }
                    } else {
                        // LeGall53
                        for x in 0..<outputSize {
                            let v = Int32(src[x])
                            dst[x] = Int16(clamping: (v + 1) >> 1)
                        }
                    }
                }
            }
        }
    }
    
    return (outData, {})
}

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

private func decodeSubbandGrid(view: BlockView, decoder: inout EntropyDecoder, isLL: Bool, lastVal: inout Int16) throws {
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
            
            let isZero = try decoder.decodeBypass() == 1
            if isZero {
                clearBlockRegion(base: subView.base, width: cw, height: ch, stride: subView.stride)
            } else {
                if cw == 16 && ch == 16 {
                    if isLL {
                        try blockDecodeDPCM16(decoder: &decoder, block: subView, lastVal: &lastVal)
                    } else {
                        try blockDecode16V(decoder: &decoder, block: subView)
                    }
                } else if cw == 8 && ch == 8 {
                    if isLL {
                        try blockDecodeDPCM8(decoder: &decoder, block: subView, lastVal: &lastVal)
                    } else {
                        try blockDecode8V(decoder: &decoder, block: subView)
                    }
                } else if cw == 4 && ch == 4 {
                    if isLL {
                        try blockDecodeDPCM4(decoder: &decoder, block: subView, lastVal: &lastVal)
                    } else {
                        try blockDecode4V(decoder: &decoder, block: subView)
                    }
                } else {
                    try blockDecodeGeneric(decoder: &decoder, block: subView)
                }
            }
        }
    }
}

private func blockDecodeGeneric(decoder: inout EntropyDecoder, block: BlockView) throws {
    let width = block.width
    let height = block.height
    
    let lastIdxUInt = try decoder.readPair(context: 5).run
    let lastIdx = Int(lastIdxUInt)
    if lastIdx < 0 || lastIdx >= width * height { return }
    
    var prevVal: Int16 = 0
    var remainRun = 0
    
    var i = 0
    while i <= lastIdx {
        let y = i / width
        let x = i % width
        let ptr = block.rowPointer(y: y)
        if remainRun > 0 {
            ptr[x] = 0
            remainRun -= 1
        } else {
            let (run, val) = try decodeCoeffRun(decoder: &decoder, context: getContext(prevVal: prevVal, isParentZero: false))
            remainRun = Int(run)
            ptr[x] = val
            prevVal = val
        }
        i += 1
    }
    
    while i < width * height {
        let y = i / width
        let x = i % width
        let ptr = block.rowPointer(y: y)
        ptr[x] = 0
        i += 1
    }
}
