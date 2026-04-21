// MARK: - Encode Spatial

@inline(__always)
func encodeSpatialLayers(pd: PlaneData420, pool: BlockViewPool, maxbitrate: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, roundOffset: Int) async throws -> ([UInt8], PlaneData420, @Sendable () -> Void) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    
    let qtY2 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 2)
    let qtC2 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 2)
    let qtY1 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 1)
    let qtC1 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 1)
    let qtY0 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 0)
    let qtC0 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 0)
    
    let resPd = PlaneData420(width: dx, height: dy, y: pd.y, cb: pd.cb, cr: pd.cr)
    let isPFrame = false
    
    var (sub2, l2yBlocks, l2cbBlocks, l2crBlocks) = try await preparePlaneLayer32(pd: resPd, pool: pool, sads: nil, layer: 2, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold)
    var (sub1, l1yBlocks, l1cbBlocks, l1crBlocks) = try await preparePlaneLayer16(pd: sub2, pool: pool, sads: nil, layer: 1, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold)
    let (layer0, baseRecon, base8YBlocks, base8CbBlocks, base8CrBlocks) = try await encodePlaneBase8(pd: sub1, pool: pool, sads: nil, layer: 0, qtY: qtY0, qtC: qtC0, zeroThreshold: zeroThreshold)
    
    let baseImg = Image16(width: baseRecon.width, height: baseRecon.height, y: baseRecon.y, cb: baseRecon.cb, cr: baseRecon.cr)
    
    let l1dx = sub2.width
    let l1dy = sub2.height
    let l1cbDx = ((l1dx + 1) / 2)
    let l1cbDy = ((l1dy + 1) / 2)
    let layer1 = entropyEncodeLayer16(dx: sub2.width, dy: sub2.height, layer: 1, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold, isPFrame: isPFrame, yBlocks: &l1yBlocks, cbBlocks: &l1cbBlocks, crBlocks: &l1crBlocks, parentYBlocks: base8YBlocks, parentCbBlocks: base8CbBlocks, parentCrBlocks: base8CrBlocks)
    
    let (mutReconL1Y, r1Y) = reconstructPlaneLayer16Y(blocks: l1yBlocks, prevImg: baseImg, width: l1dx, height: l1dy, qt: qtY1, pool: pool)
    let (mutReconL1Cb, r1Cb) = reconstructPlaneLayer16Cb(blocks: l1cbBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    let (mutReconL1Cr, r1Cr) = reconstructPlaneLayer16Cr(blocks: l1crBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    
    let l1Img = Image16(width: l1dx, height: l1dy, y: mutReconL1Y, cb: mutReconL1Cb, cr: mutReconL1Cr)
    defer { r1Y(); r1Cb(); r1Cr() }
    
    let layer2 = entropyEncodeLayer32(dx: pd.width, dy: pd.height, layer: 2, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold, isPFrame: isPFrame, yBlocks: &l2yBlocks, cbBlocks: &l2cbBlocks, crBlocks: &l2crBlocks, parentYBlocks: l1yBlocks, parentCbBlocks: l1cbBlocks, parentCrBlocks: l1crBlocks)
    
    let (reconL2Y, r2Y) = reconstructPlaneLayer32Y(blocks: l2yBlocks, prevImg: l1Img, width: dx, height: dy, qt: qtY2, pool: pool)
    var mutReconL2Y = reconL2Y
        let (reconL2Cb, r2Cb) = reconstructPlaneLayer32Cb(blocks: l2cbBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
    var mutReconL2Cb = reconL2Cb
        let (reconL2Cr, r2Cr) = reconstructPlaneLayer32Cr(blocks: l2crBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
    var mutReconL2Cr = reconL2Cr
        
    applyDeblockingFilter(plane: &mutReconL2Y, width: dx, height: dy, blockSize: 32, qStep: Int(qtY2.step))
    applyDeblockingFilter(plane: &mutReconL2Cb, width: cbDx, height: cbDy, blockSize: 32, qStep: Int(qtC2.step))
    applyDeblockingFilter(plane: &mutReconL2Cr, width: cbDx, height: cbDy, blockSize: 32, qStep: Int(qtC2.step))
    
    let reconstructed = PlaneData420(width: dx, height: dy, y: mutReconL2Y, cb: mutReconL2Cb, cr: mutReconL2Cr)
    
    debugLog({
        return "  [Summary] Layer0=\(layer0.count) Layer1=\(layer1.count) Layer2=\(layer2.count) total=\(layer0.count + layer1.count + layer2.count) bytes"
    }())
    
    var out: [UInt8] = []
    appendUInt32BE(&out, 0)
    appendUInt32BE(&out, 0)
    appendUInt32BE(&out, UInt32(layer0.count))
    out.append(contentsOf: layer0)
    appendUInt32BE(&out, UInt32(layer1.count))
    out.append(contentsOf: layer1)
    appendUInt32BE(&out, UInt32(layer2.count))
    out.append(contentsOf: layer2)
    
    let rY2 = r2Y; let rCb2 = r2Cb; let rCr2 = r2Cr
    return (out, reconstructed, { rY2(); rCb2(); rCr2() })
}

@inline(__always)
func encodeSpatialLayers(pd: PlaneData420, pool: BlockViewPool, predictedPd: PlaneData420, maxbitrate: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, roundOffset: Int) async throws -> ([UInt8], PlaneData420, @Sendable () -> Void) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    
    let qtY2 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 2)
    let qtC2 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 2)
    let qtY1 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 1)
    let qtC1 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 1)
    let qtY0 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 0)
    let qtC0 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 0)
    
    let (mvs, sads) = await computeMotionVectors(curr: pd, prev: predictedPd, pool: pool, roundOffset: roundOffset)
    
    var mutPdY = pool.getInt16(count: pd.y.count)
    var mutPdCb = pool.getInt16(count: pd.cb.count)
    var mutPdCr = pool.getInt16(count: pd.cr.count)
    mutPdY.withUnsafeMutableBufferPointer { dst in pd.y.withUnsafeBufferPointer({ dst.baseAddress!.update(from: $0.baseAddress!, count: $0.count) }) }
    mutPdCb.withUnsafeMutableBufferPointer { dst in pd.cb.withUnsafeBufferPointer({ dst.baseAddress!.update(from: $0.baseAddress!, count: $0.count) }) }
    mutPdCr.withUnsafeMutableBufferPointer { dst in pd.cr.withUnsafeBufferPointer({ dst.baseAddress!.update(from: $0.baseAddress!, count: $0.count) }) }
    
    subtractMotionCompensationPixelsLuma32(plane: &mutPdY, prevPlane: predictedPd.y, mvs: mvs, width: dx, height: dy, roundOffset: roundOffset)
    subtractMotionCompensationPixelsChroma16(plane: &mutPdCb, prevPlane: predictedPd.cb, mvs: mvs, width: cbDx, height: cbDy, roundOffset: roundOffset)
    subtractMotionCompensationPixelsChroma16(plane: &mutPdCr, prevPlane: predictedPd.cr, mvs: mvs, width: cbDx, height: cbDy, roundOffset: roundOffset)
    
    let resPd = PlaneData420(width: dx, height: dy, y: mutPdY, cb: mutPdCb, cr: mutPdCr)
    let isPFrame = true
    
    var (sub2, l2yBlocks, l2cbBlocks, l2crBlocks) = try await preparePlaneLayer32(pd: resPd, pool: pool, sads: sads, layer: 2, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold)
    var (sub1, l1yBlocks, l1cbBlocks, l1crBlocks) = try await preparePlaneLayer16(pd: sub2, pool: pool, sads: sads, layer: 1, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold)
    let (layer0, baseRecon, base8YBlocks, base8CbBlocks, base8CrBlocks) = try await encodePlaneBase8(pd: sub1, pool: pool, sads: sads, layer: 0, qtY: qtY0, qtC: qtC0, zeroThreshold: zeroThreshold)
    
    let baseImg = Image16(width: baseRecon.width, height: baseRecon.height, y: baseRecon.y, cb: baseRecon.cb, cr: baseRecon.cr)
    
    let l1dx = sub2.width
    let l1dy = sub2.height
    let l1cbDx = ((l1dx + 1) / 2)
    let l1cbDy = ((l1dy + 1) / 2)
    let layer1 = entropyEncodeLayer16(dx: sub2.width, dy: sub2.height, layer: 1, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold, isPFrame: isPFrame, yBlocks: &l1yBlocks, cbBlocks: &l1cbBlocks, crBlocks: &l1crBlocks, parentYBlocks: base8YBlocks, parentCbBlocks: base8CbBlocks, parentCrBlocks: base8CrBlocks)
    
    let (mutReconL1Y, r1Y) = reconstructPlaneLayer16Y(blocks: l1yBlocks, prevImg: baseImg, width: l1dx, height: l1dy, qt: qtY1, pool: pool)
    let (mutReconL1Cb, r1Cb) = reconstructPlaneLayer16Cb(blocks: l1cbBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    let (mutReconL1Cr, r1Cr) = reconstructPlaneLayer16Cr(blocks: l1crBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    
    let l1Img = Image16(width: l1dx, height: l1dy, y: mutReconL1Y, cb: mutReconL1Cb, cr: mutReconL1Cr)
    defer { r1Y(); r1Cb(); r1Cr() }
    
    let layer2 = entropyEncodeLayer32(dx: pd.width, dy: pd.height, layer: 2, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold, isPFrame: isPFrame, yBlocks: &l2yBlocks, cbBlocks: &l2cbBlocks, crBlocks: &l2crBlocks, parentYBlocks: l1yBlocks, parentCbBlocks: l1cbBlocks, parentCrBlocks: l1crBlocks)
    
    let (reconL2Y, r2Y) = reconstructPlaneLayer32Y(blocks: l2yBlocks, prevImg: l1Img, width: dx, height: dy, qt: qtY2, pool: pool)
    var mutReconL2Y = reconL2Y
        let (reconL2Cb, r2Cb) = reconstructPlaneLayer32Cb(blocks: l2cbBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
    var mutReconL2Cb = reconL2Cb
        let (reconL2Cr, r2Cr) = reconstructPlaneLayer32Cr(blocks: l2crBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
    var mutReconL2Cr = reconL2Cr
        
    applyMotionCompensationPixelsLuma32(plane: &mutReconL2Y, prevPlane: predictedPd.y, mvs: mvs, width: dx, height: dy, roundOffset: roundOffset)
    applyMotionCompensationPixelsChroma16(plane: &mutReconL2Cb, prevPlane: predictedPd.cb, mvs: mvs, width: cbDx, height: cbDy, roundOffset: roundOffset)
    applyMotionCompensationPixelsChroma16(plane: &mutReconL2Cr, prevPlane: predictedPd.cr, mvs: mvs, width: cbDx, height: cbDy, roundOffset: roundOffset)
    
    applyDeblockingFilter(plane: &mutReconL2Y, width: dx, height: dy, blockSize: 32, qStep: Int(qtY2.step))
    applyDeblockingFilter(plane: &mutReconL2Cb, width: cbDx, height: cbDy, blockSize: 32, qStep: Int(qtC2.step))
    applyDeblockingFilter(plane: &mutReconL2Cr, width: cbDx, height: cbDy, blockSize: 32, qStep: Int(qtC2.step))
    
    let reconstructed = PlaneData420(width: dx, height: dy, y: mutReconL2Y, cb: mutReconL2Cb, cr: mutReconL2Cr)
    
    debugLog({
        return "  [Summary] Layer0=\(layer0.count) Layer1=\(layer1.count) Layer2=\(layer2.count) total=\(layer0.count + layer1.count + layer2.count) bytes"
    }())
    
    var out: [UInt8] = []
    
    let mvCount = mvs.count
    appendUInt32BE(&out, UInt32(mvCount))
    let mvData = encodeMVs(mvs: mvs)
    appendUInt32BE(&out, UInt32(mvData.count))
    out.append(contentsOf: mvData)

    appendUInt32BE(&out, UInt32(layer0.count))
    out.append(contentsOf: layer0)
    appendUInt32BE(&out, UInt32(layer1.count))
    out.append(contentsOf: layer1)
    appendUInt32BE(&out, UInt32(layer2.count))
    out.append(contentsOf: layer2)
    
    let rY2 = r2Y; let rCb2 = r2Cb; let rCr2 = r2Cr
    return (out, reconstructed, { rY2(); rCb2(); rCr2() })
}

@inline(__always)
func encodeSpatialLayers(pd: PlaneData420, pool: BlockViewPool, predictedPd: PlaneData420, nextPd: PlaneData420, maxbitrate: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, roundOffset: Int, gopPosition: Int = 0) async throws -> ([UInt8], PlaneData420, @Sendable () -> Void) {
    let pPd = predictedPd
    let nPd = nextPd
    
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)
    
    let qtY2 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 2)
    let qtC2 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 2)
    let qtY1 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 1)
    let qtC1 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 1)
    let qtY0 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 0)
    let qtC0 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 0)
    
    // bidirectional MV calculation: search MVs for both forward and backward and select the one with the smaller SAD for each block
    let (mvs, sads, refDirs) = await computeBidirectionalMotionVectors(curr: pd, prev: pPd, next: nPd, pool: pool, roundOffset: roundOffset, gopPosition: gopPosition)
    
    // pixel level residual calculation based on reference direction
    var mutPdY = pool.getInt16(count: pd.y.count)
    var mutPdCb = pool.getInt16(count: pd.cb.count)
    var mutPdCr = pool.getInt16(count: pd.cr.count)
    mutPdY.withUnsafeMutableBufferPointer { dst in pd.y.withUnsafeBufferPointer({ dst.baseAddress!.update(from: $0.baseAddress!, count: $0.count) }) }
    mutPdCb.withUnsafeMutableBufferPointer { dst in pd.cb.withUnsafeBufferPointer({ dst.baseAddress!.update(from: $0.baseAddress!, count: $0.count) }) }
    mutPdCr.withUnsafeMutableBufferPointer { dst in pd.cr.withUnsafeBufferPointer({ dst.baseAddress!.update(from: $0.baseAddress!, count: $0.count) }) }
    
    // Y represents full Luma, scaleDen = 1 means 1 mv unit = 1/4 Luma pixel (as provided by QuarterRefinement)
    // Cb/Cr are half size, so 1 mv unit in Luma = 1/8 pixel in Chroma -> scaleDen = 2 converts to 1/4 Chroma pixel
    subtractBidirectionalMotionCompensationPixelsLuma32(plane: &mutPdY, prevPlane: pPd.y, nextPlane: nPd.y, mvs: mvs, refDirs: refDirs, width: dx, height: dy, roundOffset: roundOffset)
    subtractBidirectionalMotionCompensationPixelsChroma16(plane: &mutPdCb, prevPlane: pPd.cb, nextPlane: nPd.cb, mvs: mvs, refDirs: refDirs, width: cbDx, height: cbDy, roundOffset: roundOffset)
    subtractBidirectionalMotionCompensationPixelsChroma16(plane: &mutPdCr, prevPlane: pPd.cr, nextPlane: nPd.cr, mvs: mvs, refDirs: refDirs, width: cbDx, height: cbDy, roundOffset: roundOffset)
    let resPd = PlaneData420(width: dx, height: dy, y: mutPdY, cb: mutPdCb, cr: mutPdCr)

    let isPFrame = true
    
    var (sub2, l2yBlocks, l2cbBlocks, l2crBlocks) = try await preparePlaneLayer32(pd: resPd, pool: pool, sads: sads, layer: 2, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold)
    var (sub1, l1yBlocks, l1cbBlocks, l1crBlocks) = try await preparePlaneLayer16(pd: sub2, pool: pool, sads: sads, layer: 1, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold)
    let (layer0, baseRecon, base8YBlocks, base8CbBlocks, base8CrBlocks) = try await encodePlaneBase8(pd: sub1, pool: pool, sads: sads, layer: 0, qtY: qtY0, qtC: qtC0, zeroThreshold: zeroThreshold)
    
    let baseImg = Image16(width: baseRecon.width, height: baseRecon.height, y: baseRecon.y, cb: baseRecon.cb, cr: baseRecon.cr)
    
    let l1dx = sub2.width
    let l1dy = sub2.height
    let l1cbDx = ((l1dx + 1) / 2)
    let l1cbDy = ((l1dy + 1) / 2)
    let layer1 = entropyEncodeLayer16(dx: sub2.width, dy: sub2.height, layer: 1, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold, isPFrame: isPFrame, yBlocks: &l1yBlocks, cbBlocks: &l1cbBlocks, crBlocks: &l1crBlocks, parentYBlocks: base8YBlocks, parentCbBlocks: base8CbBlocks, parentCrBlocks: base8CrBlocks)
    
    let (mutReconL1Y, r1Y) = reconstructPlaneLayer16Y(blocks: l1yBlocks, prevImg: baseImg, width: l1dx, height: l1dy, qt: qtY1, pool: pool)
    let (mutReconL1Cb, r1Cb) = reconstructPlaneLayer16Cb(blocks: l1cbBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    let (mutReconL1Cr, r1Cr) = reconstructPlaneLayer16Cr(blocks: l1crBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    
    let l1Img = Image16(width: l1dx, height: l1dy, y: mutReconL1Y, cb: mutReconL1Cb, cr: mutReconL1Cr)
    defer { r1Y(); r1Cb(); r1Cr() }
    
    let layer2 = entropyEncodeLayer32(dx: pd.width, dy: pd.height, layer: 2, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold, isPFrame: isPFrame, yBlocks: &l2yBlocks, cbBlocks: &l2cbBlocks, crBlocks: &l2crBlocks, parentYBlocks: l1yBlocks, parentCbBlocks: l1cbBlocks, parentCrBlocks: l1crBlocks)
    
    let (reconL2Y, r2Y) = reconstructPlaneLayer32Y(blocks: l2yBlocks, prevImg: l1Img, width: dx, height: dy, qt: qtY2, pool: pool)
    var mutReconL2Y = reconL2Y
    let (reconL2Cb, r2Cb) = reconstructPlaneLayer32Cb(blocks: l2cbBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
    var mutReconL2Cb = reconL2Cb
    let (reconL2Cr, r2Cr) = reconstructPlaneLayer32Cr(blocks: l2crBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
    var mutReconL2Cr = reconL2Cr
    
    // bidirectional motion compensation addition (reconstruction)
    applyBidirectionalMotionCompensationPixelsLuma32(plane: &mutReconL2Y, prevPlane: pPd.y, nextPlane: nPd.y, mvs: mvs, refDirs: refDirs, width: dx, height: dy, roundOffset: roundOffset)
    applyBidirectionalMotionCompensationPixelsChroma16(plane: &mutReconL2Cb, prevPlane: pPd.cb, nextPlane: nPd.cb, mvs: mvs, refDirs: refDirs, width: cbDx, height: cbDy, roundOffset: roundOffset)
    applyBidirectionalMotionCompensationPixelsChroma16(plane: &mutReconL2Cr, prevPlane: pPd.cr, nextPlane: nPd.cr, mvs: mvs, refDirs: refDirs, width: cbDx, height: cbDy, roundOffset: roundOffset)
    
    applyDeblockingFilter(plane: &mutReconL2Y, width: dx, height: dy, blockSize: 32, qStep: Int(qtY2.step))
    applyDeblockingFilter(plane: &mutReconL2Cb, width: cbDx, height: cbDy, blockSize: 32, qStep: Int(qtC2.step))
    applyDeblockingFilter(plane: &mutReconL2Cr, width: cbDx, height: cbDy, blockSize: 32, qStep: Int(qtC2.step))
    
    let reconstructed = PlaneData420(width: dx, height: dy, y: mutReconL2Y, cb: mutReconL2Cb, cr: mutReconL2Cr)
    
    debugLog({
        return "  [Summary/BiDir] Layer0=\(layer0.count) Layer1=\(layer1.count) Layer2=\(layer2.count) total=\(layer0.count + layer1.count + layer2.count) bytes"
    }())
    
    var out: [UInt8] = []
    
    let mvCount = mvs.count
    appendUInt32BE(&out, UInt32(mvCount))
    
    // serialize MV data + reference direction flags
    let mvData = encodeMVs(mvs: mvs)
    appendUInt32BE(&out, UInt32(mvData.count))
    out.append(contentsOf: mvData)
    
    // reference direction flags: 1 bit per block, packed in bytes
    let refDirByteCount = (refDirs.count + 7) / 8
    appendUInt32BE(&out, UInt32(refDirByteCount))
    var refDirBuf = [UInt8](repeating: 0, count: refDirByteCount)
    for i in refDirs.indices {
        if refDirs[i] {
            refDirBuf[i / 8] |= UInt8(1 << (i % 8))
        }
    }
    out.append(contentsOf: refDirBuf)

    appendUInt32BE(&out, UInt32(layer0.count))
    out.append(contentsOf: layer0)
    
    appendUInt32BE(&out, UInt32(layer1.count))
    out.append(contentsOf: layer1)
    
    appendUInt32BE(&out, UInt32(layer2.count))
    out.append(contentsOf: layer2)
    
    let rY2 = r2Y; let rCb2 = r2Cb; let rCr2 = r2Cr
    return (out, reconstructed, { rY2(); rCb2(); rCr2() })
}
