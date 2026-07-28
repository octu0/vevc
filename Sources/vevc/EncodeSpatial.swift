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
        
        withUnsafePointers(pd.y, pd.cb, pd.cr, predictedPd.y, predictedPd.cb, predictedPd.cr) { (currYPtr: UnsafePointer<Int16>, currCbPtr: UnsafePointer<Int16>, currCrPtr: UnsafePointer<Int16>, prevYPtr: UnsafePointer<Int16>, prevCbPtr: UnsafePointer<Int16>, prevCrPtr: UnsafePointer<Int16>) -> Void in
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
                        
                        let sadPrev = if mw == 16 && mh == 16 && mwc == 8 && mhc == 8 {
                            computeZeroSAD16x16(cY: currYPtr, rY: prevYPtr, cCb: currCbPtr, rCb: prevCbPtr, cCr: currCrPtr, rCr: prevCrPtr, bx: subX, by: subY, width: dx)
                        } else {
                            computeZeroSADSubBlock(cY: currYPtr, rY: prevYPtr, cCb: currCbPtr, rCb: prevCbPtr, cCr: currCrPtr, rCr: prevCrPtr, bx: subX, by: subY, width: dx, height: dy, subWidth: mw, subHeight: mh, subWc: mwc, subHc: mhc)
                        }
                        
                        if sadPrev > blockThreshold { allSubBlocksMatchPrev = false }
                    }
                }
                
                if allSubBlocksMatchPrev {
                    skipMap[i] = .skip_prev
                }
            }
        }
    }
    
    let (mvs_original, sads, occlusionScores) = await computeMotionVectors(curr: pd, prev: predictedPd, prevMVs: prevMVs ?? MotionVectors.empty, pool: pool, roundOffset: roundOffset, skipMap: profile == 0x02 ? skipMap : [])
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
    
    if profile == 0x02 {
        mutPdY.withUnsafeMutableBufferPointer { dst in _ = memset(dst.baseAddress!, 0, dst.count * 2) }
        mutPdCb.withUnsafeMutableBufferPointer { dst in _ = memset(dst.baseAddress!, 0, dst.count * 2) }
        mutPdCr.withUnsafeMutableBufferPointer { dst in _ = memset(dst.baseAddress!, 0, dst.count * 2) }
        
        let bw = (dx + 31) / 32
        let cbBw = (cbDx + 15) / 16
        let yDstCount = mutPdY.count
        let cbDstCount = mutPdCb.count
        withUnsafePointers(pd.y, pd.cb, pd.cr, mut: &mutPdY, mut: &mutPdCb, mut: &mutPdCr) { (ySrc: UnsafePointer<Int16>, cbSrc: UnsafePointer<Int16>, crSrc: UnsafePointer<Int16>, yDst: UnsafeMutablePointer<Int16>, cbDst: UnsafeMutablePointer<Int16>, crDst: UnsafeMutablePointer<Int16>) -> Void in
            for i in 0..<skipMap.count {
                if skipMap[i] == .inter {
                    let bx = (i % bw) * 32
                    let by = (i / bw) * 32
                    for r in 0..<32 {
                        let offset = (by + r) * dx + bx
                        if offset < yDstCount && offset + 32 <= yDstCount {
                            let count = min(32, dx - bx)
                            if count == 32 {
                                let dPtr = UnsafeMutableRawPointer(yDst.advanced(by: offset))
                                let sPtr = UnsafeRawPointer(ySrc.advanced(by: offset))
                                dPtr.storeBytes(of: sPtr.loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
                                dPtr.advanced(by: 32).storeBytes(of: sPtr.advanced(by: 32).loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
                            } else {
                                yDst.advanced(by: offset).update(from: ySrc.advanced(by: offset), count: count)
                            }
                        }
                    }
                    
                    let cBx = (i % cbBw) * 16
                    let cBy = (i / cbBw) * 16
                    for r in 0..<16 {
                        let offset = (cBy + r) * cbDx + cBx
                        if offset < cbDstCount && offset + 16 <= cbDstCount {
                            let count = min(16, cbDx - cBx)
                            if count == 16 {
                                let cbDPtr = UnsafeMutableRawPointer(cbDst.advanced(by: offset))
                                let cbSPtr = UnsafeRawPointer(cbSrc.advanced(by: offset))
                                cbDPtr.storeBytes(of: cbSPtr.loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
                                
                                let crDPtr = UnsafeMutableRawPointer(crDst.advanced(by: offset))
                                let crSPtr = UnsafeRawPointer(crSrc.advanced(by: offset))
                                crDPtr.storeBytes(of: crSPtr.loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
                            } else {
                                cbDst.advanced(by: offset).update(from: cbSrc.advanced(by: offset), count: count)
                                crDst.advanced(by: offset).update(from: crSrc.advanced(by: offset), count: count)
                            }
                        }
                    }
                }
            }
        }
    } else {
        mutPdY.withUnsafeMutableBufferPointer { dst in pd.y.withUnsafeBufferPointer({ dst.baseAddress!.update(from: $0.baseAddress!, count: $0.count) }) }
        mutPdCb.withUnsafeMutableBufferPointer { dst in pd.cb.withUnsafeBufferPointer({ dst.baseAddress!.update(from: $0.baseAddress!, count: $0.count) }) }
        mutPdCr.withUnsafeMutableBufferPointer { dst in pd.cr.withUnsafeBufferPointer({ dst.baseAddress!.update(from: $0.baseAddress!, count: $0.count) }) }
    }

    // MV is layer0 precision -> mvScale=4 for applying to layer2 (full resolution)
    let sMap: [BlockMode]? = (profile == 0x02) ? skipMap : nil
    subtractScaledMotionCompensationLuma(plane: &mutPdY, prevPlane: predictedPd.y, mvs: mvs, skipMap: sMap, width: dx, height: dy, lumaBlockSize: 32, mvShift: 0, roundOffset: roundOffset)
    subtractScaledMotionCompensationChroma(plane: &mutPdCb, prevPlane: predictedPd.cb, mvs: mvs, skipMap: sMap, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
    subtractScaledMotionCompensationChroma(plane: &mutPdCr, prevPlane: predictedPd.cr, mvs: mvs, skipMap: sMap, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
    
    let resPd = PlaneData420(width: dx, height: dy, y: mutPdY, cb: mutPdCb, cr: mutPdCr)
    let isPFrame = true
    
    var (sub2, l2yBlocks, l2cbBlocks, l2crBlocks, releaseL2) = try await preparePlaneLayer32(pd: resPd, pool: pool, sads: sads, layer: 2, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold, skipMap: sMap)
    defer { releaseL2() }
    
    var (sub1, l1yBlocks, l1cbBlocks, l1crBlocks, releaseL1) = try await preparePlaneLayer16(pd: sub2, pool: pool, sads: sads, occlusionScores: occlusionScores, layer: 1, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold, skipMap: sMap)
    defer { releaseL1() }
    
    let (layer0, baseRecon, base8YBlocks, base8CbBlocks, base8CrBlocks, releaseBase) = try await encodePlaneBase8(pd: sub1, pool: pool, sads: sads, occlusionScores: occlusionScores, layer: 0, qtY: qtY0, qtC: qtC0, zeroThreshold: zeroThreshold, skipMap: sMap)
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
    await applyScaledMotionCompensationLuma(plane: &mutReconL2Y, prevPlane: predictedPd.y, mvs: mvs, width: dx, height: dy, lumaBlockSize: 32, mvShift: 0, roundOffset: roundOffset)
    await applyScaledMotionCompensationChroma(plane: &mutReconL2Cb, prevPlane: predictedPd.cb, mvs: mvs, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
    await applyScaledMotionCompensationChroma(plane: &mutReconL2Cr, prevPlane: predictedPd.cr, mvs: mvs, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
    
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
        
        let matchResults = await withTaskGroup(of: [(Int, Bool)].self) { group in
            let batchSize = 128
            for batchStart in stride(from: 0, to: blockCount, by: batchSize) {
                let batchEnd = min(batchStart + batchSize, blockCount)
                group.addTask {
                    var results = [(Int, Bool)]()
                    results.reserveCapacity(batchEnd - batchStart)
                    // ポインタ取得はタスク内・クロージャスコープ内で行う（外へ持ち出すと未定義動作）
                    pd.y.withUnsafeBufferPointer { cYBuf in
                    pd.cb.withUnsafeBufferPointer { cCbBuf in
                    pd.cr.withUnsafeBufferPointer { cCrBuf in
                    prevInput.y.withUnsafeBufferPointer { pYBuf in
                    prevInput.cb.withUnsafeBufferPointer { pCbBuf in
                    prevInput.cr.withUnsafeBufferPointer { pCrBuf in
                    let cYPtr = cYBuf.baseAddress!
                    let cCbPtr = cCbBuf.baseAddress!
                    let cCrPtr = cCrBuf.baseAddress!
                    let pYPtr = pYBuf.baseAddress!
                    let pCbPtr = pCbBuf.baseAddress!
                    let pCrPtr = pCrBuf.baseAddress!
                    
                    for i in batchStart..<batchEnd {
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
                                
                                let sadPrevIn: Int
                                if mw == 16 && mh == 16 && mwc == 8 && mhc == 8 {
                                    sadPrevIn = computeZeroSAD16x16(cY: cYPtr, rY: pYPtr, cCb: cCbPtr, rCb: pCbPtr, cCr: cCrPtr, rCr: pCrPtr, bx: subX, by: subY, width: dx)
                                } else {
                                    sadPrevIn = computeZeroSADSubBlock(cY: cYPtr, rY: pYPtr, cCb: cCbPtr, rCb: pCbPtr, cCr: cCrPtr, rCr: pCrPtr, bx: subX, by: subY, width: dx, height: dy, subWidth: mw, subHeight: mh, subWc: mwc, subHc: mhc)
                                }
                                
                                if sadPrevIn > blockThreshold { allSubBlocksMatchPrev = false }
                            }
                        }
                        results.append((i, allSubBlocksMatchPrev))
                    }
                    }}}}}}
                    return results
                }
            }
            
            var allResults = [(Int, Bool)]()
            for await batch in group {
                allResults.append(contentsOf: batch)
            }
            return allResults
        }
        
        for (i, allSubBlocksMatchPrev) in matchResults {
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
    }
    
    // bidirectional MV calculation: search MVs for both forward and backward and select the one with the smaller SAD for each block
    let (mvs_original, sads, refDirs_original, occlusionScores) = await computeBidirectionalMotionVectors(curr: pd, prev: pPd, next: nPd, prevMVs: prevMVs ?? MotionVectors.empty, pool: pool, roundOffset: roundOffset, gopPosition: gopPosition, skipMap: profile == 0x02 ? skipMap : [])
    var mvs = mvs_original
    var refDirs = refDirs_original
    if profile == 0x02 {
        for i in 0..<skipMap.count {
            switch skipMap[i] {
            case .skip_ltr:
                mvs.dx[i] = 0
                mvs.dy[i] = 0
                refDirs[i] = true
            case .skip_prev:
                mvs.dx[i] = 0
                mvs.dy[i] = 0
                refDirs[i] = false
            default: break
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
    let sMap: [BlockMode]? = (profile == 0x02) ? skipMap : nil
    let mvsConst = mvs
    let refDirsConst = refDirs
    async let tY = { [mvsConst, refDirsConst, sMap] () -> [Int16] in
        var y = mutPdY
        subtractScaledBidirectionalMotionCompensationLuma(plane: &y, prevPlane: pPd.y, nextPlane: nPd.y, mvs: mvsConst, refDirs: refDirsConst, skipMap: sMap, width: dx, height: dy, lumaBlockSize: 32, mvShift: 0, roundOffset: roundOffset)
        return y
    }()
    async let tCb = { [mvsConst, refDirsConst, sMap] () -> [Int16] in
        var cb = mutPdCb
        subtractScaledBidirectionalMotionCompensationChroma(plane: &cb, prevPlane: pPd.cb, nextPlane: nPd.cb, mvs: mvsConst, refDirs: refDirsConst, skipMap: sMap, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
        return cb
    }()
    async let tCr = { [mvsConst, refDirsConst, sMap] () -> [Int16] in
        var cr = mutPdCr
        subtractScaledBidirectionalMotionCompensationChroma(plane: &cr, prevPlane: pPd.cr, nextPlane: nPd.cr, mvs: mvsConst, refDirs: refDirsConst, skipMap: sMap, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
        return cr
    }()
    
    let resY = await tY
    let resCb = await tCb
    let resCr = await tCr
    
    let resPd = PlaneData420(width: dx, height: dy, y: resY, cb: resCb, cr: resCr)

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
    
    async let tL1Y = { () -> ([Int16], @Sendable () -> Void) in
        return reconstructPlaneLayer16Y(blocks: l1yBlocks, prevImg: baseImg, width: l1dx, height: l1dy, qt: qtY1, pool: pool)
    }()
    async let tL1Cb = { () -> ([Int16], @Sendable () -> Void) in
        return reconstructPlaneLayer16Cb(blocks: l1cbBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    }()
    async let tL1Cr = { () -> ([Int16], @Sendable () -> Void) in
        return reconstructPlaneLayer16Cr(blocks: l1crBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    }()
    
    let (mutReconL1Y, r1Y) = await tL1Y
    let (mutReconL1Cb, r1Cb) = await tL1Cb
    let (mutReconL1Cr, r1Cr) = await tL1Cr
    
    let l1Img = Image16(width: l1dx, height: l1dy, y: mutReconL1Y, cb: mutReconL1Cb, cr: mutReconL1Cr)
    defer { r1Y(); r1Cb(); r1Cr() }
    
    let layer2 = entropyEncodeLayer32(dx: pd.width, dy: pd.height, layer: 2, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold, isPFrame: isPFrame, yBlocks: &l2yBlocks, cbBlocks: &l2cbBlocks, crBlocks: &l2crBlocks, parentYBlocks: l1yBlocks, parentCbBlocks: l1cbBlocks, parentCrBlocks: l1crBlocks, sads: sads)
    
    let mvsConst2 = mvs
    let refDirsConst2 = refDirs
    let skipMapConst = skipMap
    let l1ImgConst = l1Img
    
    async let aY = { [mvsConst2, refDirsConst2, skipMapConst, l1ImgConst] () -> ([Int16], @Sendable () -> Void) in
        let (reconL2Y, r2Y) = reconstructPlaneLayer32Y(blocks: l2yBlocks, prevImg: l1ImgConst, width: dx, height: dy, qt: qtY2, pool: pool)
        var y = reconL2Y
        await applyScaledBidirectionalMotionCompensationLuma(plane: &y, prevPlane: pPd.y, nextPlane: nPd.y, mvs: mvsConst2, refDirs: refDirsConst2, skipMap: skipMapConst, width: dx, height: dy, lumaBlockSize: 32, mvShift: 0, roundOffset: roundOffset)
        applyDeblockingFilter32(plane: &y, width: dx, height: dy, qStep: (Int(qtY2.step) + 8) >> 4, mvs: mvsConst2)
        
        if profile == 0x02 {
            let bw = (dx + 31) / 32
            for i in 0..<skipMapConst.count {
                if skipMapConst[i] == .skip_ltr {
                    let bx = (i % bw) * 32
                    let by = (i / bw) * 32
                    copyBlock(from: nPd.y, to: &y, bx: bx, by: by, width: dx, height: dy, blockSize: 32)
                }
            }
        }
        return (y, r2Y)
    }()
    
    async let aCb = { [mvsConst2, refDirsConst2, skipMapConst, l1ImgConst] () -> ([Int16], @Sendable () -> Void) in
        let (reconL2Cb, r2Cb) = reconstructPlaneLayer32Cb(blocks: l2cbBlocks, prevImg: l1ImgConst, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
        var cb = reconL2Cb
        await applyScaledBidirectionalMotionCompensationChroma(plane: &cb, prevPlane: pPd.cb, nextPlane: nPd.cb, mvs: mvsConst2, refDirs: refDirsConst2, skipMap: skipMapConst, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
        applyDeblockingFilterChroma16(plane: &cb, width: cbDx, height: cbDy, qStep: (Int(qtC2.step) + 8) >> 4, mvs: mvsConst2)
        if profile == 0x02 {
            let bw = (dx + 31) / 32
            for i in 0..<skipMapConst.count {
                if skipMapConst[i] == .skip_ltr {
                    let bx = (i % bw) * 32
                    let by = (i / bw) * 32
                    copyBlock(from: nPd.cb, to: &cb, bx: bx/2, by: by/2, width: cbDx, height: cbDy, blockSize: 16)
                }
            }
        }
        return (cb, r2Cb)
    }()
    
    async let aCr = { [mvsConst2, refDirsConst2, skipMapConst, l1ImgConst] () -> ([Int16], @Sendable () -> Void) in
        let (reconL2Cr, r2Cr) = reconstructPlaneLayer32Cr(blocks: l2crBlocks, prevImg: l1ImgConst, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
        var cr = reconL2Cr
        await applyScaledBidirectionalMotionCompensationChroma(plane: &cr, prevPlane: pPd.cr, nextPlane: nPd.cr, mvs: mvsConst2, refDirs: refDirsConst2, skipMap: skipMapConst, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
        applyDeblockingFilterChroma16(plane: &cr, width: cbDx, height: cbDy, qStep: (Int(qtC2.step) + 8) >> 4, mvs: mvsConst2)
        if profile == 0x02 {
            let bw = (dx + 31) / 32
            for i in 0..<skipMapConst.count {
                if skipMapConst[i] == .skip_ltr {
                    let bx = (i % bw) * 32
                    let by = (i / bw) * 32
                    copyBlock(from: nPd.cr, to: &cr, bx: bx/2, by: by/2, width: cbDx, height: cbDy, blockSize: 16)
                }
            }
        }
        return (cr, r2Cr)
    }()
    
    let (mutReconL2Y, r2Y) = await aY
    let (mutReconL2Cb, r2Cb) = await aCb
    let (mutReconL2Cr, r2Cr) = await aCr
    
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
func computeZeroSAD16x16(
    cY: UnsafePointer<Int16>, rY: UnsafePointer<Int16>,
    cCb: UnsafePointer<Int16>, rCb: UnsafePointer<Int16>,
    cCr: UnsafePointer<Int16>, rCr: UnsafePointer<Int16>,
    bx: Int, by: Int, width: Int
) -> Int {
    var acc16 = SIMD16<Int32>(repeating: 0)
    let strideY = width
    let strideC = (width + 1) / 2
    let bxC = bx / 2
    let byC = by / 2

    for y in 0..<16 {
        let offset = (by + y) * strideY + bx
        let cv = UnsafeRawPointer(cY.advanced(by: offset)).loadUnaligned(as: SIMD16<Int16>.self)
        let rv = UnsafeRawPointer(rY.advanced(by: offset)).loadUnaligned(as: SIMD16<Int16>.self)
        let d1 = cv &- rv
        let d2 = rv &- cv
        let diff = d1.replacing(with: d2, where: cv .< rv)
        acc16 &+= SIMD16<Int32>(clamping: diff)
    }
    var sad = Int(acc16.wrappedSum())
    
    var acc8 = SIMD8<Int32>(repeating: 0)
    for y in 0..<8 {
        let offset = (byC + y) * strideC + bxC
        
        let cvCb = UnsafeRawPointer(cCb.advanced(by: offset)).loadUnaligned(as: SIMD8<Int16>.self)
        let rvCb = UnsafeRawPointer(rCb.advanced(by: offset)).loadUnaligned(as: SIMD8<Int16>.self)
        let d1Cb = cvCb &- rvCb
        let d2Cb = rvCb &- cvCb
        let diffCb = d1Cb.replacing(with: d2Cb, where: cvCb .< rvCb)
        acc8 &+= SIMD8<Int32>(clamping: diffCb)
        
        let cvCr = UnsafeRawPointer(cCr.advanced(by: offset)).loadUnaligned(as: SIMD8<Int16>.self)
        let rvCr = UnsafeRawPointer(rCr.advanced(by: offset)).loadUnaligned(as: SIMD8<Int16>.self)
        let d1Cr = cvCr &- rvCr
        let d2Cr = rvCr &- cvCr
        let diffCr = d1Cr.replacing(with: d2Cr, where: cvCr .< rvCr)
        acc8 &+= SIMD8<Int32>(clamping: diffCr)
    }
    sad += Int(acc8.wrappedSum())
    return sad
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
    planeY: UnsafeMutablePointer<Int16>, planeCb: UnsafeMutablePointer<Int16>, planeCr: UnsafeMutablePointer<Int16>,
    bx: Int, by: Int, width: Int, height: Int
) {
    let strideY = width
    if bx + 32 <= width && by + 32 <= height {
        for y in 0..<32 {
            let offset = (by + y) * strideY + bx
            for x in 0..<32 { planeY[offset + x] = 0 }
        }
        let bxC = bx / 2
        let byC = by / 2
        let strideC = (width + 1) / 2
        for y in 0..<16 {
            let offset = (byC + y) * strideC + bxC
            for x in 0..<16 { 
                planeCb[offset + x] = 0
                planeCr[offset + x] = 0
            }
        }
    } else {
        for y in 0..<32 {
            let yy = by + y
            if yy >= height { break }
            let offset = yy * strideY + bx
            let maxW = min(32, width - bx)
            for x in 0..<maxW {
                planeY[offset + x] = 0
            }
        }
        
        let bxC = bx / 2
        let byC = by / 2
        let strideC = (width + 1) / 2
        let heightC = (height + 1) / 2
        for y in 0..<16 {
            let yy = byC + y
            if yy >= heightC { break }
            let offset = yy * strideC + bxC
            let maxW = min(16, strideC - bxC)
            for x in 0..<maxW {
                planeCb[offset + x] = 0
                planeCr[offset + x] = 0
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
    
    withUnsafePointers(src, mut: &dst) { (sPtr: UnsafePointer<Int16>, dPtr: UnsafeMutablePointer<Int16>) in
            let sBase = sPtr
            let dBase = dPtr
            if bx + blockSize <= width && by + blockSize <= height {
                for y in 0..<blockSize {
                    let offset = (by + y) * width + bx
                    dBase.advanced(by: offset).update(from: sBase.advanced(by: offset), count: blockSize)
                }
            } else {
                for y in by..<maxY {
                    let offset = y * width + bx
                    dBase.advanced(by: offset).update(from: sBase.advanced(by: offset), count: copyCount)
                }
            }
        }
    }

