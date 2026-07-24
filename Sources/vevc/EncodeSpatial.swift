// MARK: - Encode Spatial
import Foundation

@inline(__always)
func encodeSpatialLayers(pd: PlaneData420, pool: BlockViewPool, maxbitrate: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, roundOffset: Int, profile: UInt8 = 0x01) async throws -> ([UInt8], PlaneData420, MotionVectors, [Int], @Sendable () -> Void) {
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
    
    var (sub2, l2yBlocks, l2cbBlocks, l2crBlocks, releaseL2) = try await preparePlaneLayer32(pd: resPd, pool: pool, sads: nil, layer: 2, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold)
    defer { releaseL2() }
    var (sub1, l1yBlocks, l1cbBlocks, l1crBlocks, releaseL1) = try await preparePlaneLayer16(pd: sub2, pool: pool, sads: nil, occlusionScores: nil, layer: 1, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold)
    defer { releaseL1() }
    let (layer0, baseRecon, base8YBlocks, base8CbBlocks, base8CrBlocks, releaseBase) = try await encodePlaneBase8(pd: sub1, pool: pool, sads: nil, occlusionScores: nil, layer: 0, qtY: qtY0, qtC: qtC0, zeroThreshold: zeroThreshold)
    defer { releaseBase() }
    
    let baseImg = Image16(width: baseRecon.width, height: baseRecon.height, y: baseRecon.y, cb: baseRecon.cb, cr: baseRecon.cr)
    
    let l1dx = sub2.width
    let l1dy = sub2.height
    let l1cbDx = ((l1dx + 1) / 2)
    let l1cbDy = ((l1dy + 1) / 2)
    let layer1 = entropyEncodeLayer16(dx: sub2.width, dy: sub2.height, layer: 1, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold, isPFrame: isPFrame, yBlocks: &l1yBlocks, cbBlocks: &l1cbBlocks, crBlocks: &l1crBlocks, parentYBlocks: base8YBlocks, parentCbBlocks: base8CbBlocks, parentCrBlocks: base8CrBlocks)
    
    let (mutReconL1Y, r1Y) = reconstructPlaneLayer16Y(blocks: l1yBlocks, prevImg: baseImg, width: l1dx, height: l1dy, qt: qtY1, pool: pool)
    let (mutReconL1Cb, r1Cb) = reconstructPlaneLayer16Cb(blocks: l1cbBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    let (mutReconL1Cr, r1Cr) = reconstructPlaneLayer16Cr(blocks: l1crBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    defer { r1Y(); r1Cb(); r1Cr() }

    let l1Img = Image16(width: l1dx, height: l1dy, y: mutReconL1Y, cb: mutReconL1Cb, cr: mutReconL1Cr)
    let layer2 = entropyEncodeLayer32(dx: pd.width, dy: pd.height, layer: 2, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold, isPFrame: isPFrame, yBlocks: &l2yBlocks, cbBlocks: &l2cbBlocks, crBlocks: &l2crBlocks, parentYBlocks: l1yBlocks, parentCbBlocks: l1cbBlocks, parentCrBlocks: l1crBlocks)
    
    let (reconL2Y, r2Y) = reconstructPlaneLayer32Y(blocks: l2yBlocks, prevImg: l1Img, width: dx, height: dy, qt: qtY2, pool: pool)
    var mutReconL2Y = reconL2Y
    let (reconL2Cb, r2Cb) = reconstructPlaneLayer32Cb(blocks: l2cbBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
    var mutReconL2Cb = reconL2Cb
    let (reconL2Cr, r2Cr) = reconstructPlaneLayer32Cr(blocks: l2crBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
    var mutReconL2Cr = reconL2Cr
        
    applyDeblockingFilter32(plane: &mutReconL2Y, width: dx, height: dy, qStep: (Int(qtY2.step) + 8) >> 4)
    applyDeblockingFilterChroma16(plane: &mutReconL2Cb, width: cbDx, height: cbDy, qStep: (Int(qtC2.step) + 8) >> 4, mvs: MotionVectors.empty)
    applyDeblockingFilterChroma16(plane: &mutReconL2Cr, width: cbDx, height: cbDy, qStep: (Int(qtC2.step) + 8) >> 4, mvs: MotionVectors.empty)
    
    let reconstructed = PlaneData420(width: dx, height: dy, y: mutReconL2Y, cb: mutReconL2Cb, cr: mutReconL2Cr)
    
    debugLog({
        return "  [Summary] Layer0=\(layer0.count) Layer1=\(layer1.count) Layer2=\(layer2.count) total=\(layer0.count + layer1.count + layer2.count) bytes"
    }())
    
    var out: [UInt8] = []
    let frameHeader = VEVCFrameHeader(frameType: .iFrame, mvsSize: 0, refDirSize: 0, layer0Size: layer0.count, layer1Size: layer1.count, layer2Size: layer2.count)
    out.append(contentsOf: frameHeader.serialize())
    out.append(contentsOf: layer0)
    out.append(contentsOf: layer1)
    out.append(contentsOf: layer2)
    
    return (out, reconstructed, MotionVectors.empty, [], { r2Y(); r2Cb(); r2Cr() })
}

@inline(__always)
func encodeSpatialLayers(pd: PlaneData420, pool: BlockViewPool, predictedPd: PlaneData420, prevMVs: MotionVectors?, maxbitrate: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, roundOffset: Int, profile: UInt8 = 0x01) async throws -> ([UInt8], PlaneData420, MotionVectors, [Int], @Sendable () -> Void) {
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
    
    var skipMap = [BlockMode]()
    if profile == 0x02 {
        let bw = (dx + 31) / 32
        let bh = (dy + 31) / 32
        let blockCount = bw * bh
        skipMap = [BlockMode](repeating: .inter, count: blockCount)
        let skipThresholdPerPixel = ProcessInfo.processInfo.environment["VEVC_SKIP_THRESH"].flatMap { Int($0) } ?? 2
        
        pd.y.withUnsafeBufferPointer { currYPtr in
        pd.cb.withUnsafeBufferPointer { currCbPtr in
        pd.cr.withUnsafeBufferPointer { currCrPtr in
        predictedPd.y.withUnsafeBufferPointer { prevYPtr in
        predictedPd.cb.withUnsafeBufferPointer { prevCbPtr in
        predictedPd.cr.withUnsafeBufferPointer { prevCrPtr in
            for i in 0..<blockCount {
                let bx = (i % bw) * 32
                let by = (i / bw) * 32
                
                var allSubBlocksMatchPrev = true
                
                for sy in 0..<2 {
                    for sx in 0..<2 {
                        let subX = bx + sx * 16
                        let subY = by + sy * 16
                        let mw = min(16, dx - subX)
                        let mh = min(16, dy - subY)
                        if mw <= 0 || mh <= 0 { continue }
                        let mwc = min(8, cbDx - subX / 2)
                        let mhc = min(8, cbDy - subY / 2)
                        let area = mw * mh + mwc * mhc * 2
                        let blockThreshold = skipThresholdPerPixel * area
                        
                        let sadPrev = computeZeroSADSubBlock(cY: currYPtr.baseAddress!, rY: prevYPtr.baseAddress!, cCb: currCbPtr.baseAddress!, rCb: prevCbPtr.baseAddress!, cCr: currCrPtr.baseAddress!, rCr: prevCrPtr.baseAddress!, bx: subX, by: subY, width: dx, height: dy, subWidth: mw, subHeight: mh, subWc: mwc, subHc: mhc)
                        
                        if sadPrev > blockThreshold { allSubBlocksMatchPrev = false }
                    }
                }
                
                if allSubBlocksMatchPrev {
                    skipMap[i] = .skip_prev
                }
            }
        }}}}}}
    }
    
    let (mvs_original, sads, occlusionScores) = await computeMotionVectors(curr: pd, prev: predictedPd, prevMVs: prevMVs, pool: pool, roundOffset: roundOffset, skipMap: profile == 0x02 ? skipMap : nil)
    var mvs = mvs_original
    if profile == 0x02 {
        for i in 0..<skipMap.count {
            if skipMap[i] != .inter {
                mvs.dx[i] = 0
                mvs.dy[i] = 0
            }
        }
    }
    
    var mutPdY = pool.getInt16(count: pd.y.count)
    var mutPdCb = pool.getInt16(count: pd.cb.count)
    var mutPdCr = pool.getInt16(count: pd.cr.count)
    mutPdY.withUnsafeMutableBufferPointer { dst in pd.y.withUnsafeBufferPointer({ dst.baseAddress!.update(from: $0.baseAddress!, count: $0.count) }) }
    mutPdCb.withUnsafeMutableBufferPointer { dst in pd.cb.withUnsafeBufferPointer({ dst.baseAddress!.update(from: $0.baseAddress!, count: $0.count) }) }
    mutPdCr.withUnsafeMutableBufferPointer { dst in pd.cr.withUnsafeBufferPointer({ dst.baseAddress!.update(from: $0.baseAddress!, count: $0.count) }) }

    // MV is layer0 precision -> mvScale=4 for applying to layer2 (full resolution)
    subtractScaledMotionCompensationLuma(plane: &mutPdY, prevPlane: predictedPd.y, mvs: mvs, width: dx, height: dy, lumaBlockSize: 32, mvShift: 0, roundOffset: roundOffset)
    subtractScaledMotionCompensationChroma(plane: &mutPdCb, prevPlane: predictedPd.cb, mvs: mvs, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
    subtractScaledMotionCompensationChroma(plane: &mutPdCr, prevPlane: predictedPd.cr, mvs: mvs, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
    
    if profile == 0x02 {
        let bw = (dx + 31) / 32
        for i in 0..<skipMap.count {
            if skipMap[i] != .inter {
                let bx = (i % bw) * 32
                let by = (i / bw) * 32
                clearResidualBlock(planeY: &mutPdY, planeCb: &mutPdCb, planeCr: &mutPdCr, bx: bx, by: by, width: dx, height: dy)
            }
        }
    }
    
    let resPd = PlaneData420(width: dx, height: dy, y: mutPdY, cb: mutPdCb, cr: mutPdCr)
    let isPFrame = true
    
    var (sub2, l2yBlocks, l2cbBlocks, l2crBlocks, releaseL2) = try await preparePlaneLayer32(pd: resPd, pool: pool, sads: sads, layer: 2, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold)
    defer { releaseL2() }
    
    var (sub1, l1yBlocks, l1cbBlocks, l1crBlocks, releaseL1) = try await preparePlaneLayer16(pd: sub2, pool: pool, sads: sads, occlusionScores: occlusionScores, layer: 1, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold)
    defer { releaseL1() }
    
    let (layer0, baseRecon, base8YBlocks, base8CbBlocks, base8CrBlocks, releaseBase) = try await encodePlaneBase8(pd: sub1, pool: pool, sads: sads, occlusionScores: occlusionScores, layer: 0, qtY: qtY0, qtC: qtC0, zeroThreshold: zeroThreshold)
    defer { releaseBase() }
    
    let baseImg = Image16(width: baseRecon.width, height: baseRecon.height, y: baseRecon.y, cb: baseRecon.cb, cr: baseRecon.cr)
    
    let l1dx = sub2.width
    let l1dy = sub2.height
    let l1cbDx = ((l1dx + 1) / 2)
    let l1cbDy = ((l1dy + 1) / 2)
    let layer1 = entropyEncodeLayer16(dx: sub2.width, dy: sub2.height, layer: 1, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold, isPFrame: isPFrame, yBlocks: &l1yBlocks, cbBlocks: &l1cbBlocks, crBlocks: &l1crBlocks, parentYBlocks: base8YBlocks, parentCbBlocks: base8CbBlocks, parentCrBlocks: base8CrBlocks)
    
    let (mutReconL1Y, r1Y) = reconstructPlaneLayer16Y(blocks: l1yBlocks, prevImg: baseImg, width: l1dx, height: l1dy, qt: qtY1, pool: pool)
    let (mutReconL1Cb, r1Cb) = reconstructPlaneLayer16Cb(blocks: l1cbBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    let (mutReconL1Cr, r1Cr) = reconstructPlaneLayer16Cr(blocks: l1crBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    defer { r1Y(); r1Cb(); r1Cr() }

    let l1Img = Image16(width: l1dx, height: l1dy, y: mutReconL1Y, cb: mutReconL1Cb, cr: mutReconL1Cr)
    let layer2 = entropyEncodeLayer32(dx: pd.width, dy: pd.height, layer: 2, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold, isPFrame: isPFrame, yBlocks: &l2yBlocks, cbBlocks: &l2cbBlocks, crBlocks: &l2crBlocks, parentYBlocks: l1yBlocks, parentCbBlocks: l1cbBlocks, parentCrBlocks: l1crBlocks, sads: sads)
    
    let (reconL2Y, r2Y) = reconstructPlaneLayer32Y(blocks: l2yBlocks, prevImg: l1Img, width: dx, height: dy, qt: qtY2, pool: pool)
    var mutReconL2Y = reconL2Y
    let (reconL2Cb, r2Cb) = reconstructPlaneLayer32Cb(blocks: l2cbBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
    var mutReconL2Cb = reconL2Cb
    let (reconL2Cr, r2Cr) = reconstructPlaneLayer32Cr(blocks: l2crBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
    var mutReconL2Cr = reconL2Cr
        
    // Reconstruction adds back the reference prediction (mvScale=4 for layer2)
    applyScaledMotionCompensationLuma(plane: &mutReconL2Y, prevPlane: predictedPd.y, mvs: mvs, width: dx, height: dy, lumaBlockSize: 32, mvShift: 0, roundOffset: roundOffset)
    applyScaledMotionCompensationChroma(plane: &mutReconL2Cb, prevPlane: predictedPd.cb, mvs: mvs, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
    applyScaledMotionCompensationChroma(plane: &mutReconL2Cr, prevPlane: predictedPd.cr, mvs: mvs, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
    
    applyDeblockingFilter32(plane: &mutReconL2Y, width: dx, height: dy, qStep: (Int(qtY2.step) + 8) >> 4, mvs: mvs)
    applyDeblockingFilterChroma16(plane: &mutReconL2Cb, width: cbDx, height: cbDy, qStep: (Int(qtC2.step) + 8) >> 4, mvs: mvs)
    applyDeblockingFilterChroma16(plane: &mutReconL2Cr, width: cbDx, height: cbDy, qStep: (Int(qtC2.step) + 8) >> 4, mvs: mvs)
    
    let reconstructed = PlaneData420(width: dx, height: dy, y: mutReconL2Y, cb: mutReconL2Cb, cr: mutReconL2Cr)
    
    debugLog({
        return "  [Summary] Layer0=\(layer0.count) Layer1=\(layer1.count) Layer2=\(layer2.count) total=\(layer0.count + layer1.count + layer2.count) bytes"
    }())
    
    var out: [UInt8] = []
    var skipMapData: [UInt8] = []
    if profile == 0x02 {
        skipMapData = encodeSkipMap(map: skipMap)
    }
    let mvData = encodeMVs(mvs: mvs, skipMap: skipMap, profile: profile)
    
    let frameHeader = VEVCFrameHeader(frameType: .pFrame, skipMapSize: skipMapData.count, mvsSize: mvData.count, refDirSize: 0, layer0Size: layer0.count, layer1Size: layer1.count, layer2Size: layer2.count)
    out.append(contentsOf: frameHeader.serialize(profile: profile))
    out.append(contentsOf: skipMapData)
    out.append(contentsOf: mvData)
    out.append(contentsOf: layer0)
    out.append(contentsOf: layer1)
    out.append(contentsOf: layer2)
    
    return (out, reconstructed, mvs, sads, { r2Y(); r2Cb(); r2Cr() })
}

@inline(__always)
func encodeSpatialLayers(pd: PlaneData420, pool: BlockViewPool, predictedPd: PlaneData420, nextPd: PlaneData420, prevInput: PlaneData420, ltrInput: PlaneData420, prevMVs: MotionVectors?, maxbitrate: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, roundOffset: Int, gopPosition: Int = 0, profile: UInt8 = 0x01, staticCounters: inout [Int]) async throws -> ([UInt8], PlaneData420, MotionVectors, [Int], @Sendable () -> Void) {
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
    
    var skipMap = [BlockMode]()
    if profile == 0x02 {
        let bw = (dx + 31) / 32
        let bh = (dy + 31) / 32
        let blockCount = bw * bh
        skipMap = [BlockMode](repeating: .inter, count: blockCount)
        let skipThresholdPerPixel = ProcessInfo.processInfo.environment["VEVC_SKIP_THRESH"].flatMap { Int($0) } ?? 2
        
        pd.y.withUnsafeBufferPointer { currYPtr in
        pd.cb.withUnsafeBufferPointer { currCbPtr in
        pd.cr.withUnsafeBufferPointer { currCrPtr in
        prevInput.y.withUnsafeBufferPointer { prevInYPtr in
        prevInput.cb.withUnsafeBufferPointer { prevInCbPtr in
        prevInput.cr.withUnsafeBufferPointer { prevInCrPtr in
            for i in 0..<blockCount {
                let bx = (i % bw) * 32
                let by = (i / bw) * 32
                
                var allSubBlocksMatchPrev = true
                
                for sy in 0..<2 {
                    for sx in 0..<2 {
                        let subX = bx + sx * 16
                        let subY = by + sy * 16
                        let mw = min(16, dx - subX)
                        let mh = min(16, dy - subY)
                        if mw <= 0 || mh <= 0 { continue }
                        let mwc = min(8, cbDx - subX / 2)
                        let mhc = min(8, cbDy - subY / 2)
                        let area = mw * mh + mwc * mhc * 2
                        let blockThreshold = skipThresholdPerPixel * area
                        
                        let sadPrevIn = computeZeroSADSubBlock(cY: currYPtr.baseAddress!, rY: prevInYPtr.baseAddress!, cCb: currCbPtr.baseAddress!, rCb: prevInCbPtr.baseAddress!, cCr: currCrPtr.baseAddress!, rCr: prevInCrPtr.baseAddress!, bx: subX, by: subY, width: dx, height: dy, subWidth: mw, subHeight: mh, subWc: mwc, subHc: mhc)
                        
                        if sadPrevIn > blockThreshold { allSubBlocksMatchPrev = false }
                    }
                }
                
                if allSubBlocksMatchPrev {
                    staticCounters[i] += 1
                } else {
                    staticCounters[i] = 0
                }
                
                if staticCounters[i] > 0 {
                    if staticCounters[i] == gopPosition {
                        skipMap[i] = .skip_ltr
                    } else {
                        skipMap[i] = .skip_prev
                    }
                }
            }
        }}}}}}
    }
    
    // bidirectional MV calculation: search MVs for both forward and backward and select the one with the smaller SAD for each block
    let (mvs_original, sads, refDirs_original, occlusionScores) = await computeBidirectionalMotionVectors(curr: pd, prev: pPd, next: nPd, prevMVs: prevMVs, pool: pool, roundOffset: roundOffset, gopPosition: gopPosition, skipMap: profile == 0x02 ? skipMap : nil)
    var mvs = mvs_original
    var refDirs = refDirs_original
    if profile == 0x02 {
        for i in 0..<skipMap.count {
            if skipMap[i] == .skip_ltr {
                mvs.dx[i] = 0
                mvs.dy[i] = 0
                refDirs[i] = true
            } else if skipMap[i] == .skip_prev {
                mvs.dx[i] = 0
                mvs.dy[i] = 0
                refDirs[i] = false
            }
        }
    }

    // pixel level residual calculation based on reference direction
    var mutPdY = pool.getInt16(count: pd.y.count)
    var mutPdCb = pool.getInt16(count: pd.cb.count)
    var mutPdCr = pool.getInt16(count: pd.cr.count)
    mutPdY.withUnsafeMutableBufferPointer { dst in pd.y.withUnsafeBufferPointer({ dst.baseAddress!.update(from: $0.baseAddress!, count: $0.count) }) }
    mutPdCb.withUnsafeMutableBufferPointer { dst in pd.cb.withUnsafeBufferPointer({ dst.baseAddress!.update(from: $0.baseAddress!, count: $0.count) }) }
    mutPdCr.withUnsafeMutableBufferPointer { dst in pd.cr.withUnsafeBufferPointer({ dst.baseAddress!.update(from: $0.baseAddress!, count: $0.count) }) }
    
    // MV is layer0 precision -> mvScale=4 for applying to layer2 (full resolution)
    subtractScaledBidirectionalMotionCompensationLuma(plane: &mutPdY, prevPlane: pPd.y, nextPlane: nPd.y, mvs: mvs, refDirs: refDirs, width: dx, height: dy, lumaBlockSize: 32, mvShift: 0, roundOffset: roundOffset)
    subtractScaledBidirectionalMotionCompensationChroma(plane: &mutPdCb, prevPlane: pPd.cb, nextPlane: nPd.cb, mvs: mvs, refDirs: refDirs, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
    subtractScaledBidirectionalMotionCompensationChroma(plane: &mutPdCr, prevPlane: pPd.cr, nextPlane: nPd.cr, mvs: mvs, refDirs: refDirs, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
    
    if profile == 0x02 {
        let bw = (dx + 31) / 32
        for i in 0..<skipMap.count {
            if skipMap[i] != .inter {
                let bx = (i % bw) * 32
                let by = (i / bw) * 32
                clearResidualBlock(planeY: &mutPdY, planeCb: &mutPdCb, planeCr: &mutPdCr, bx: bx, by: by, width: dx, height: dy)
            }
        }
    }
    
    let resPd = PlaneData420(width: dx, height: dy, y: mutPdY, cb: mutPdCb, cr: mutPdCr)

    let isPFrame = true
    
    var (sub2, l2yBlocks, l2cbBlocks, l2crBlocks, releaseL2) = try await preparePlaneLayer32(pd: resPd, pool: pool, sads: sads, layer: 2, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold)
    defer { releaseL2() }
    var (sub1, l1yBlocks, l1cbBlocks, l1crBlocks, releaseL1) = try await preparePlaneLayer16(pd: sub2, pool: pool, sads: sads, occlusionScores: occlusionScores, layer: 1, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold)
    defer { releaseL1() }
    let (layer0, baseRecon, base8YBlocks, base8CbBlocks, base8CrBlocks, releaseBase) = try await encodePlaneBase8(pd: sub1, pool: pool, sads: sads, occlusionScores: occlusionScores, layer: 0, qtY: qtY0, qtC: qtC0, zeroThreshold: zeroThreshold)
    defer { releaseBase() }
    
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
    
    let layer2 = entropyEncodeLayer32(dx: pd.width, dy: pd.height, layer: 2, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold, isPFrame: isPFrame, yBlocks: &l2yBlocks, cbBlocks: &l2cbBlocks, crBlocks: &l2crBlocks, parentYBlocks: l1yBlocks, parentCbBlocks: l1cbBlocks, parentCrBlocks: l1crBlocks, sads: sads)
    
    let (reconL2Y, r2Y) = reconstructPlaneLayer32Y(blocks: l2yBlocks, prevImg: l1Img, width: dx, height: dy, qt: qtY2, pool: pool)
    var mutReconL2Y = reconL2Y
    let (reconL2Cb, r2Cb) = reconstructPlaneLayer32Cb(blocks: l2cbBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
    var mutReconL2Cb = reconL2Cb
    let (reconL2Cr, r2Cr) = reconstructPlaneLayer32Cr(blocks: l2crBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
    var mutReconL2Cr = reconL2Cr
    
    // Reconstruction adds back reference prediction (mvScale=4 for layer2)
    applyScaledBidirectionalMotionCompensationLuma(plane: &mutReconL2Y, prevPlane: pPd.y, nextPlane: nPd.y, mvs: mvs, refDirs: refDirs, width: dx, height: dy, lumaBlockSize: 32, mvShift: 0, roundOffset: roundOffset)
    applyScaledBidirectionalMotionCompensationChroma(plane: &mutReconL2Cb, prevPlane: pPd.cb, nextPlane: nPd.cb, mvs: mvs, refDirs: refDirs, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
    applyScaledBidirectionalMotionCompensationChroma(plane: &mutReconL2Cr, prevPlane: pPd.cr, nextPlane: nPd.cr, mvs: mvs, refDirs: refDirs, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
    
    applyDeblockingFilter32(plane: &mutReconL2Y, width: dx, height: dy, qStep: (Int(qtY2.step) + 8) >> 4, mvs: mvs)
    applyDeblockingFilterChroma16(plane: &mutReconL2Cb, width: cbDx, height: cbDy, qStep: (Int(qtC2.step) + 8) >> 4, mvs: mvs)
    applyDeblockingFilterChroma16(plane: &mutReconL2Cr, width: cbDx, height: cbDy, qStep: (Int(qtC2.step) + 8) >> 4, mvs: mvs)
    
    if profile == 0x02 {
        let bw = (dx + 31) / 32
        for i in 0..<skipMap.count {
            if skipMap[i] == .skip_ltr {
                let bx = (i % bw) * 32
                let by = (i / bw) * 32
                copyBlock(from: nPd.y, to: &mutReconL2Y, bx: bx, by: by, width: dx, height: dy, blockSize: 32)
                copyBlock(from: nPd.cb, to: &mutReconL2Cb, bx: bx/2, by: by/2, width: cbDx, height: cbDy, blockSize: 16)
                copyBlock(from: nPd.cr, to: &mutReconL2Cr, bx: bx/2, by: by/2, width: cbDx, height: cbDy, blockSize: 16)
            }
        }
    }
    
    let reconstructed = PlaneData420(width: dx, height: dy, y: mutReconL2Y, cb: mutReconL2Cb, cr: mutReconL2Cr)
    
    debugLog({
        return "  [Summary/BiDir] Layer0=\(layer0.count) Layer1=\(layer1.count) Layer2=\(layer2.count) total=\(layer0.count + layer1.count + layer2.count) bytes"
    }())
    
    var out: [UInt8] = []
    var skipMapData: [UInt8] = []
    if profile == 0x02 {
        skipMapData = encodeSkipMap(map: skipMap)
    }
    let mvData = encodeMVs(mvs: mvs, skipMap: skipMap, profile: profile)
    
    let refDirByteCount = (refDirs.count + 7) / 8
    var refDirBuf = [UInt8](repeating: 0, count: refDirByteCount)
    for i in refDirs.indices {
        // refDirs encode: skip_ltr -> true (1), skip_prev -> false (0)
        // we've already set refDirs in skipMap logic.
        if refDirs[i] {
            refDirBuf[i / 8] |= UInt8(1 << (i % 8))
        }
    }
    
    let frameHeader = VEVCFrameHeader(frameType: .pFrame, hasRefDir: true, skipMapSize: skipMapData.count, mvsSize: mvData.count, refDirSize: refDirBuf.count, layer0Size: layer0.count, layer1Size: layer1.count, layer2Size: layer2.count)
    out.append(contentsOf: frameHeader.serialize(profile: profile))
    out.append(contentsOf: skipMapData)
    out.append(contentsOf: mvData)
    out.append(contentsOf: refDirBuf)
    out.append(contentsOf: layer0)
    out.append(contentsOf: layer1)
    out.append(contentsOf: layer2)
    
    return (out, reconstructed, mvs, sads, { r2Y(); r2Cb(); r2Cr() })
}

@inline(__always)
func computeZeroSADSubBlock(
    cY: UnsafePointer<Int16>, rY: UnsafePointer<Int16>,
    cCb: UnsafePointer<Int16>, rCb: UnsafePointer<Int16>,
    cCr: UnsafePointer<Int16>, rCr: UnsafePointer<Int16>,
    bx: Int, by: Int, width: Int, height: Int,
    subWidth: Int, subHeight: Int, subWc: Int, subHc: Int
) -> Int {
    var sad: Int = 0
    let strideY = width
    for y in 0..<subHeight {
        let yy = by + y
        let offset = yy * strideY + bx
        for x in 0..<subWidth {
            sad &+= Int((Int32(cY[offset + x]) - Int32(rY[offset + x])).magnitude)
        }
    }
    
    let bxC = bx / 2
    let byC = by / 2
    let strideC = (width + 1) / 2
    for y in 0..<subHc {
        let yy = byC + y
        let offset = yy * strideC + bxC
        for x in 0..<subWc {
            sad &+= Int((Int32(cCb[offset + x]) - Int32(rCb[offset + x])).magnitude)
            sad &+= Int((Int32(cCr[offset + x]) - Int32(rCr[offset + x])).magnitude)
        }
    }
    return sad
}

@inline(__always)
func clearResidualBlock(
    planeY: inout [Int16], planeCb: inout [Int16], planeCr: inout [Int16],
    bx: Int, by: Int, width: Int, height: Int
) {
    let strideY = width
    planeY.withUnsafeMutableBufferPointer { ptr in
        for y in 0..<32 {
            let yy = by + y
            if yy >= height { break }
            let offset = yy * strideY + bx
            let maxW = min(32, width - bx)
            for x in 0..<maxW {
                ptr[offset + x] = 0
            }
        }
    }
    
    let bxC = bx / 2
    let byC = by / 2
    let strideC = (width + 1) / 2
    let heightC = (height + 1) / 2
    planeCb.withUnsafeMutableBufferPointer { ptr in
        for y in 0..<16 {
            let yy = byC + y
            if yy >= heightC { break }
            let offset = yy * strideC + bxC
            let maxW = min(16, strideC - bxC)
            for x in 0..<maxW {
                ptr[offset + x] = 0
            }
        }
    }
    planeCr.withUnsafeMutableBufferPointer { ptr in
        for y in 0..<16 {
            let yy = byC + y
            if yy >= heightC { break }
            let offset = yy * strideC + bxC
            let maxW = min(16, strideC - bxC)
            for x in 0..<maxW {
                ptr[offset + x] = 0
            }
        }
    }
}

@inline(__always)
private func copyBlock(from src: [Int16], to dst: inout [Int16], bx: Int, by: Int, width: Int, height: Int, blockSize: Int) {
    let maxY = min(by + blockSize, height)
    let maxX = min(bx + blockSize, width)
    let copyCount = maxX - bx
    if copyCount <= 0 { return }
    
    src.withUnsafeBufferPointer { sPtr in
        dst.withUnsafeMutableBufferPointer { dPtr in
            guard let sBase = sPtr.baseAddress, let dBase = dPtr.baseAddress else { return }
            for y in by..<maxY {
                let offset = y * width + bx
                dBase.advanced(by: offset).update(from: sBase.advanced(by: offset), count: copyCount)
            }
        }
    }
}
