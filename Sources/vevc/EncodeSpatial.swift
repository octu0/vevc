// MARK: - Encode Spatial
import Foundation

@inline(__always)
func encodeSpatialLayers(pd: PlaneData420, pool: BlockViewPool, maxbitrate: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, roundOffset: Int, profile: UInt8 = 0x01, skipThreshold: Int = 2, reconThresholdScale: Int = 1, dwtGainScale: Int = 1, l0State: L0RefState? = nil) async throws -> ([UInt8], PlaneData420, MotionVectors, [Int], @Sendable () -> Void) {
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

    // L0 closed loop (One-Pyramid §4): the I-frame L0 reference is the
    // dequantized Base8 output — identical on both sides by construction.
    if profile == 0x02, let l0s = l0State {
        let ref = freshCopy(baseImg)
        l0s.prev = ref
        l0s.ltr = ref
    }

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
        
    // Must be the decoder's exact I-frame deblock sequence (its mvs==nil
    // branch): plain luma + applyDeblockingFilter16 chroma. The encoder
    // previously used applyDeblockingFilterChroma16 with empty mvs, which
    // differs by ±1 at some block edges; with the L0 closed loop that drift
    // reaches the entropy contexts (LL2 slot couples P into the parent
    // blocks) and desyncs backward-adaptive tables.
    applyDeblockingFilter32(plane: &mutReconL2Y, width: dx, height: dy, qStep: (Int(qtY2.step) + 8) >> 4)
    applyDeblockingFilter16(plane: &mutReconL2Cb, width: cbDx, height: cbDy, qStep: (Int(qtC2.step) + 8) >> 4)
    applyDeblockingFilter16(plane: &mutReconL2Cr, width: cbDx, height: cbDy, qStep: (Int(qtC2.step) + 8) >> 4)

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
func encodeSpatialLayers(pd: PlaneData420, pool: BlockViewPool, predictedPd: PlaneData420, nextPd: PlaneData420, prevInput: PlaneData420, ltrInput: PlaneData420, prevMVs: MotionVectors?, maxbitrate: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, roundOffset: Int, gopPosition: Int = 0, profile: UInt8 = 0x01, skipThreshold: Int = 2, reconThresholdScale: Int = 1, staticCounters: inout [Int], cachedNextSub2: [Int16]? = nil, cachedNextSub1: [Int16]? = nil, entropyHistories: FrameEntropyHistories? = nil, l0State: L0RefState? = nil) async throws -> ([UInt8], PlaneData420, MotionVectors, [Int], @Sendable () -> Void, [Int16], [Int16]) {
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
        let skipThresholdPerPixel = skipThreshold
        let prevStaticCounters = staticCounters
        
        let matchResults = await withTaskGroup(of: [(Int, Bool, Bool, Bool)].self) { group in
            let batchSize = 128
            for batchStart in stride(from: 0, to: blockCount, by: batchSize) {
                let batchEnd = min(batchStart + batchSize, blockCount)
                group.addTask {
                    var results = [(Int, Bool, Bool, Bool)]()
                    results.reserveCapacity(batchEnd - batchStart)
                    
                    pd.y.withUnsafeBufferPointer { cYBuf in
                    pd.cb.withUnsafeBufferPointer { cCbBuf in
                    pd.cr.withUnsafeBufferPointer { cCrBuf in
                    prevInput.y.withUnsafeBufferPointer { pYBuf in
                    prevInput.cb.withUnsafeBufferPointer { pCbBuf in
                    prevInput.cr.withUnsafeBufferPointer { pCrBuf in
                    ltrInput.y.withUnsafeBufferPointer { lYBuf in
                    ltrInput.cb.withUnsafeBufferPointer { lCbBuf in
                    ltrInput.cr.withUnsafeBufferPointer { lCrBuf in
                    predictedPd.y.withUnsafeBufferPointer { pReconYBuf in
                    predictedPd.cb.withUnsafeBufferPointer { pReconCbBuf in
                    predictedPd.cr.withUnsafeBufferPointer { pReconCrBuf in
                    let cYPtr = cYBuf.baseAddress!
                    let cCbPtr = cCbBuf.baseAddress!
                    let cCrPtr = cCrBuf.baseAddress!
                    let pYPtr = pYBuf.baseAddress!
                    let pCbPtr = pCbBuf.baseAddress!
                    let pCrPtr = pCrBuf.baseAddress!
                    let lYPtr = lYBuf.baseAddress!
                    let lCbPtr = lCbBuf.baseAddress!
                    let lCrPtr = lCrBuf.baseAddress!
                    let pReconYPtr = pReconYBuf.baseAddress!
                    let pReconCbPtr = pReconCbBuf.baseAddress!
                    let pReconCrPtr = pReconCrBuf.baseAddress!
                    
                    for i in batchStart..<batchEnd {
                        let bx = (i % bw) * 32
                        let by = (i / bw) * 32
                        let prevCount = prevStaticCounters[i]
                        
                        var allSubBlocksMatchPrev = true
                        var allSubBlocksMatchLtr = false
                        var allSubBlocksMatchPrevRecon = false
                        
                        for sy in 0..<2 {
                            for sx in 0..<2 {
                                let subX = bx + sx * 16
                                let subY = by + sy * 16
                                let mw = min(16, dx - subX)
                                let mh = min(16, dy - subY)
                                if mw <= 0 || mh <= 0 { continue }
                                let mwc = ((mw + 1) / 2)
                                let mhc = ((mh + 1) / 2)
                                
                                let area = mw * mh + mwc * mhc * 2
                                let blockThreshold = skipThresholdPerPixel * area
                                
                                let sadPrevIn: Int
                                if mw == 16 && mh == 16 && mwc == 8 && mhc == 8 {
                                    sadPrevIn = computeZeroSAD16x16(cY: cYPtr, rY: pYPtr, cCb: cCbPtr, rCb: pCbPtr, cCr: cCrPtr, rCr: pCrPtr, bx: subX, by: subY, width: dx, limit: blockThreshold)
                                } else {
                                    sadPrevIn = computeZeroSADSubBlock(cY: cYPtr, rY: pYPtr, cCb: cCbPtr, rCb: pCbPtr, cCr: cCrPtr, rCr: pCrPtr, bx: subX, by: subY, width: dx, height: dy, subWidth: mw, subHeight: mh, subWc: mwc, subHc: mhc, limit: blockThreshold)
                                }
                                
                                if blockThreshold < sadPrevIn {
                                    allSubBlocksMatchPrev = false
                                    break
                                }
                            }
                            if allSubBlocksMatchPrev != true { break }
                        }
                        
                        let nextStaticCount = allSubBlocksMatchPrev ? (prevCount + 1) : 0
                        
                        if nextStaticCount == gopPosition {
                            allSubBlocksMatchLtr = true
                            for sy in 0..<2 {
                                for sx in 0..<2 {
                                    let subX = bx + sx * 16
                                    let subY = by + sy * 16
                                    let mw = min(16, dx - subX)
                                    let mh = min(16, dy - subY)
                                    if mw <= 0 || mh <= 0 { continue }
                                    let mwc = ((mw + 1) / 2)
                                    let mhc = ((mh + 1) / 2)
                                    
                                    let area = mw * mh + mwc * mhc * 2
                                    let blockThreshold = skipThresholdPerPixel * area
                                    
                                    let sadLtrIn: Int
                                    if mw == 16 && mh == 16 && mwc == 8 && mhc == 8 {
                                        sadLtrIn = computeZeroSAD16x16(cY: cYPtr, rY: lYPtr, cCb: cCbPtr, rCb: lCbPtr, cCr: cCrPtr, rCr: lCrPtr, bx: subX, by: subY, width: dx, limit: blockThreshold)
                                    } else {
                                        sadLtrIn = computeZeroSADSubBlock(cY: cYPtr, rY: lYPtr, cCb: cCbPtr, rCb: lCbPtr, cCr: cCrPtr, rCr: lCrPtr, bx: subX, by: subY, width: dx, height: dy, subWidth: mw, subHeight: mh, subWc: mwc, subHc: mhc, limit: blockThreshold)
                                    }
                                    
                                    if blockThreshold < sadLtrIn {
                                        allSubBlocksMatchLtr = false
                                        break
                                    }
                                }
                                if allSubBlocksMatchLtr != true { break }
                            }
                        }
                        
                        let isLtrMatch = allSubBlocksMatchLtr && (nextStaticCount == gopPosition)
                        if isLtrMatch != true && allSubBlocksMatchPrev && (3 < nextStaticCount) {
                            allSubBlocksMatchPrevRecon = true
                            for sy in 0..<2 {
                                for sx in 0..<2 {
                                    let subX = bx + sx * 16
                                    let subY = by + sy * 16
                                    let mw = min(16, dx - subX)
                                    let mh = min(16, dy - subY)
                                    if mw <= 0 || mh <= 0 { continue }
                                    let mwc = ((mw + 1) / 2)
                                    let mhc = ((mh + 1) / 2)
                                    
                                    let area = mw * mh + mwc * mhc * 2
                                    let blockThreshold = skipThresholdPerPixel * area
                                    let reconBlockThreshold = blockThreshold * reconThresholdScale
                                    
                                    let sadPrevRecon: Int
                                    if mw == 16 && mh == 16 && mwc == 8 && mhc == 8 {
                                        sadPrevRecon = computeZeroSAD16x16(cY: cYPtr, rY: pReconYPtr, cCb: cCbPtr, rCb: pReconCbPtr, cCr: cCrPtr, rCr: pReconCrPtr, bx: subX, by: subY, width: dx, limit: reconBlockThreshold)
                                    } else {
                                        sadPrevRecon = computeZeroSADSubBlock(cY: cYPtr, rY: pReconYPtr, cCb: cCbPtr, rCb: pReconCbPtr, cCr: cCrPtr, rCr: pReconCrPtr, bx: subX, by: subY, width: dx, height: dy, subWidth: mw, subHeight: mh, subWc: mwc, subHc: mhc, limit: reconBlockThreshold)
                                    }
                                    
                                    if reconBlockThreshold < sadPrevRecon {
                                        allSubBlocksMatchPrevRecon = false
                                        break
                                    }
                                }
                                if allSubBlocksMatchPrevRecon != true { break }
                            }
                        }
                        
                        results.append((i, allSubBlocksMatchPrev, allSubBlocksMatchLtr, allSubBlocksMatchPrevRecon))
                    }
                    }}}}}}}}}}}}
                    return results
                }
            }
            
            var allResults = [(Int, Bool, Bool, Bool)]()
            for await batch in group {
                allResults.append(contentsOf: batch)
            }
            return allResults
        }
        
        for (i, allSubBlocksMatchPrev, allSubBlocksMatchLtr, allSubBlocksMatchPrevRecon) in matchResults {
            if allSubBlocksMatchPrev {
                staticCounters[i] += 1
            } else {
                staticCounters[i] = 0
            }
            
            if allSubBlocksMatchLtr && staticCounters[i] == gopPosition {
                skipMap[i] = .skip_ltr
            } else if allSubBlocksMatchPrev && allSubBlocksMatchPrevRecon && (3 < staticCounters[i]) {
                skipMap[i] = .skip_prev
            } else {
                skipMap[i] = .inter
            }
        }
    }
    
    var searchSkipMap: [BlockMode] = []
    if profile == 0x02 {
        searchSkipMap = skipMap
    }
    let (mvs_original, sads, refDirs_original, occlusionScores, nextSub2Res, nextSub1Res) = await computeBidirectionalMotionVectors(curr: pd, prev: pPd, next: nPd, prevMVs: prevMVs ?? MotionVectors.empty, pool: pool, roundOffset: roundOffset, gopPosition: gopPosition, skipMap: searchSkipMap, cachedNextSub2: cachedNextSub2, cachedNextSub1: cachedNextSub1)
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

    var mutPdY = pool.getInt16(count: pd.y.count)
    var mutPdCb = pool.getInt16(count: pd.cb.count)
    var mutPdCr = pool.getInt16(count: pd.cr.count)
    
    mutPdY.withUnsafeMutableBufferPointer { dst in pd.y.withUnsafeBufferPointer({ dst.baseAddress!.update(from: $0.baseAddress!, count: $0.count) }) }
    mutPdCb.withUnsafeMutableBufferPointer { dst in pd.cb.withUnsafeBufferPointer({ dst.baseAddress!.update(from: $0.baseAddress!, count: $0.count) }) }
    mutPdCr.withUnsafeMutableBufferPointer { dst in pd.cr.withUnsafeBufferPointer({ dst.baseAddress!.update(from: $0.baseAddress!, count: $0.count) }) }
    
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

    // L0 closed loop (One-Pyramid §4): Base8 codes r0 = LL2(source) −
    // MC_L0(L0_ref) instead of LL2(residual), so the quarter-resolution
    // reconstruction closes over the bitstream alone (bit-exact with the
    // decoder's layer0 chain). Requires an L0 reference from a preceding
    // I-frame; without l0State the legacy LL2(residual) semantics apply.
    let isL0Loop = (profile == 0x02) && (l0State?.prev != nil)
    var base8Input = sub1
    if isL0Loop, let l0s = l0State, let l0Prev = l0s.prev {
        let tSrc = analyzeLL2(pd: pd)
        var r0 = Image16(width: tSrc.width, height: tSrc.height, y: tSrc.y, cb: tSrc.cb, cr: tSrc.cr)
        var pred0 = Image16(
            width: tSrc.width, height: tSrc.height,
            y: [Int16](repeating: 0, count: tSrc.y.count),
            cb: [Int16](repeating: 0, count: tSrc.cb.count),
            cr: [Int16](repeating: 0, count: tSrc.cr.count))
        await applyL0MotionCompensation(img: &pred0, prevPd: l0Prev, ltrPd: l0s.ltr, mvs: mvs, refDirs: refDirs, skipMap: sMap, roundOffset: roundOffset)
        subtractPlanes(&r0, PlaneData420(img16: pred0))
        if let map = sMap {
            clearL0SkipResidual(img: &r0, skipMap: map, fullDx: dx)
        }
        base8Input = PlaneData420(img16: r0)
    }

    let (layer0, baseRecon, base8YBlocks, base8CbBlocks, base8CrBlocks, releaseBase) = try await encodePlaneBase8(pd: base8Input, pool: pool, sads: sads, occlusionScores: occlusionScores, layer: 0, qtY: qtY0, qtC: qtC0, zeroThreshold: zeroThreshold, histories: entropyHistories?.streams[0])
    defer { releaseBase() }

    var baseImg = Image16(width: baseRecon.width, height: baseRecon.height, y: baseRecon.y, cb: baseRecon.cb, cr: baseRecon.cr)

    if isL0Loop, let l0s = l0State, let l0Prev = l0s.prev {
        // L0 reconstruction — the decoder's exact layer0 pipeline:
        // deq(r0) + MC_L0, clamp, deblock, skip copy.
        let baseCopy = freshCopy(baseImg)
        var l0Cur = Image16(width: baseRecon.width, height: baseRecon.height, y: baseCopy.y, cb: baseCopy.cb, cr: baseCopy.cr)
        await applyL0MotionCompensation(img: &l0Cur, prevPd: l0Prev, ltrPd: l0s.ltr, mvs: mvs, refDirs: refDirs, skipMap: sMap, roundOffset: roundOffset)
        finishL0Reconstruction(img: &l0Cur, qtYStepQ4: Int(qtY0.step), qtCStepQ4: Int(qtC0.step))
        if let map = sMap {
            applyL0SkipCopy(img: &l0Cur, prevPd: l0Prev, ltrPd: l0s.ltr, skipMap: map, fullDx: dx)
        }
        let newRef = PlaneData420(img16: l0Cur)

        // The full loop's LL2 coefficient slot is L0_recon − LL2(P), with P
        // built by the identical MC call sequence the decoder uses.
        let fullP = await buildFullResolutionPrediction(dx: dx, dy: dy, prevPd: pPd, ltrPd: nPd, mvs: mvs, refDirs: refDirs, skipMap: sMap, roundOffset: roundOffset)
        let tP = analyzeLL2(pd: fullP)
        var slot = Image16(width: newRef.width, height: newRef.height, y: newRef.y, cb: newRef.cb, cr: newRef.cr)
        subtractPlanes(&slot, tP)
        baseImg = slot

        l0s.prev = newRef
    }

    let l1dx = sub2.width
    let l1dy = sub2.height
    let l1cbDx = ((l1dx + 1) / 2)
    let l1cbDy = ((l1dy + 1) / 2)
    let layer1 = entropyEncodeLayer16(dx: sub2.width, dy: sub2.height, layer: 1, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold, isPFrame: isPFrame, yBlocks: &l1yBlocks, cbBlocks: &l1cbBlocks, crBlocks: &l1crBlocks, parentYBlocks: base8YBlocks, parentCbBlocks: base8CbBlocks, parentCrBlocks: base8CrBlocks, histories: entropyHistories?.streams[1])

    if let dumper = CoeffDumper.shared {
        dumper.stash("L1Y", blocks: l1yBlocks, planeW: sub2.width, planeH: sub2.height, blockSize: 16, includeLL: false)
        dumper.stash("L1Cb", blocks: l1cbBlocks, planeW: l1cbDx, planeH: l1cbDy, blockSize: 16, includeLL: false)
        dumper.stash("L1Cr", blocks: l1crBlocks, planeW: l1cbDx, planeH: l1cbDy, blockSize: 16, includeLL: false)
    }

    let (mutReconL1Y, r1Y) = reconstructPlaneLayer16Y(blocks: l1yBlocks, prevImg: baseImg, width: l1dx, height: l1dy, qt: qtY1, pool: pool)
    let (mutReconL1Cb, r1Cb) = reconstructPlaneLayer16Cb(blocks: l1cbBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    let (mutReconL1Cr, r1Cr) = reconstructPlaneLayer16Cr(blocks: l1crBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    defer { r1Y(); r1Cb(); r1Cr() }

    let l1Img = Image16(width: l1dx, height: l1dy, y: mutReconL1Y, cb: mutReconL1Cb, cr: mutReconL1Cr)
    let layer2 = entropyEncodeLayer32(dx: pd.width, dy: pd.height, layer: 2, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold, isPFrame: isPFrame, yBlocks: &l2yBlocks, cbBlocks: &l2cbBlocks, crBlocks: &l2crBlocks, parentYBlocks: l1yBlocks, parentCbBlocks: l1cbBlocks, parentCrBlocks: l1crBlocks, histories: entropyHistories?.streams[2])

    if let dumper = CoeffDumper.shared {
        dumper.stash("L2Y", blocks: l2yBlocks, planeW: dx, planeH: dy, blockSize: 32, includeLL: false)
        dumper.stash("L2Cb", blocks: l2cbBlocks, planeW: cbDx, planeH: cbDy, blockSize: 32, includeLL: false)
        dumper.stash("L2Cr", blocks: l2crBlocks, planeW: cbDx, planeH: cbDy, blockSize: 32, includeLL: false)
        dumper.finalizePFrame(
            gopPosition: gopPosition, width: dx, height: dy, predictedPd: pPd,
            l1yBlocks: l1yBlocks, l1cbBlocks: l1cbBlocks, l1crBlocks: l1crBlocks,
            b8yBlocks: base8YBlocks, b8cbBlocks: base8CbBlocks, b8crBlocks: base8CrBlocks,
            sub2W: sub2.width, sub2H: sub2.height, sub1W: sub1.width, sub1H: sub1.height,
            qtY2: qtY2, qtC2: qtC2, qtY1: qtY1, qtC1: qtC1, qtY0: qtY0, qtC0: qtC0,
            layer0Bytes: layer0.count, layer1Bytes: layer1.count, layer2Bytes: layer2.count)
    }

    // skipMap must be passed here: the decoder's layer2 reconstruction skips
    // skip blocks entirely (they stay zero until the final skip copy), and
    // the deblocking filter runs BEFORE that copy, reading pixels from skip
    // regions at mixed inter|skip block edges. Reconstructing those regions
    // on the encoder only (as before) diverges the reconstructions by ±1 at
    // such edges — harmless historically, but the L0 closed loop couples the
    // reconstruction into the entropy contexts where any divergence desyncs
    // the backward-adaptive tables.
    let (reconL2Y, r2Y) = reconstructPlaneLayer32Y(blocks: l2yBlocks, prevImg: l1Img, width: dx, height: dy, qt: qtY2, pool: pool, skipMap: sMap)
    var mutReconL2Y = reconL2Y
    let (reconL2Cb, r2Cb) = reconstructPlaneLayer32Cb(blocks: l2cbBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool, skipMap: sMap)
    var mutReconL2Cb = reconL2Cb
    let (reconL2Cr, r2Cr) = reconstructPlaneLayer32Cr(blocks: l2crBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool, skipMap: sMap)
    var mutReconL2Cr = reconL2Cr

    await applyScaledBidirectionalMotionCompensationLuma(plane: &mutReconL2Y, prevPlane: pPd.y, nextPlane: nPd.y, mvs: mvs, refDirs: refDirs, skipMap: sMap, width: dx, height: dy, lumaBlockSize: 32, mvShift: 0, roundOffset: roundOffset)
    await applyScaledBidirectionalMotionCompensationChroma(plane: &mutReconL2Cb, prevPlane: pPd.cb, nextPlane: nPd.cb, mvs: mvs, refDirs: refDirs, skipMap: sMap, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
    await applyScaledBidirectionalMotionCompensationChroma(plane: &mutReconL2Cr, prevPlane: pPd.cr, nextPlane: nPd.cr, mvs: mvs, refDirs: refDirs, skipMap: sMap, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
    
    var skipMapData: [UInt8] = []
    if profile == 0x02 {
        skipMapData = encodeSkipMap(map: skipMap)
    }
    let mvData = encodeMVs(mvs: mvs, skipMap: skipMap, profile: profile)
    
    let refDirByteCount = (refDirs.count + 7) / 8
    var refDirBuf = [UInt8](repeating: 0, count: refDirByteCount)
    for i in refDirs.indices {
        if refDirs[i] {
            refDirBuf[i / 8] |= UInt8(1 << (i % 8))
        }
    }
    
    // Must match the decoder's deblock invocation exactly (mvs + skipMap
    // variants): the encoder previously filtered its reconstruction with the
    // plain/partial variants while the decoder used the intra-boundary
    // enhanced + skip-gated ones, so the two reconstructions diverged and the
    // P-chain accumulated the difference into chroma-heavy smears in
    // intra-dense motion regions (grew with GOP position, immune to bitrate).
    applyDeblockingFilter32(plane: &mutReconL2Y, width: dx, height: dy, qStep: (Int(qtY2.step) + 8) >> 4, mvs: mvs, skipMap: sMap)
    applyDeblockingFilterChroma16(plane: &mutReconL2Cb, width: cbDx, height: cbDy, qStep: (Int(qtC2.step) + 8) >> 4, mvs: mvs, skipMap: sMap)
    applyDeblockingFilterChroma16(plane: &mutReconL2Cr, width: cbDx, height: cbDy, qStep: (Int(qtC2.step) + 8) >> 4, mvs: mvs, skipMap: sMap)

    if profile == 0x02 {
        let bw = (dx + 31) / 32
        let targetCbDx = (dx + 1) / 2
        let targetCbDy = (dy + 1) / 2
        let targetBSize = 32
        let tCbSize = 16
        
        withUnsafePointers(
            nPd.y, nPd.cb, nPd.cr,
            pPd.y, pPd.cb, pPd.cr,
            mut: &mutReconL2Y, mut: &mutReconL2Cb, mut: &mutReconL2Cr
        ) { ltrYPtr, ltrCbPtr, ltrCrPtr, prevYPtr, prevCbPtr, prevCrPtr, currYPtr, currCbPtr, currCrPtr in
            for i in 0..<skipMap.count {
                let mode = skipMap[i]
                if mode != .inter {
                    let bx = (i % bw) * 32
                    let by = (i / bw) * 32
                    
                    if bx + targetBSize <= dx && by + targetBSize <= dy {
                        switch mode {
                        case .skip_ltr:
                            copyBlockPointer(from: ltrYPtr, to: currYPtr, bx: bx, by: by, stride: dx, blockSize: targetBSize)
                            copyBlockPointer(from: ltrCbPtr, to: currCbPtr, bx: bx/2, by: by/2, stride: targetCbDx, blockSize: tCbSize)
                            copyBlockPointer(from: ltrCrPtr, to: currCrPtr, bx: bx/2, by: by/2, stride: targetCbDx, blockSize: tCbSize)
                        case .skip_prev:
                            copyBlockPointer(from: prevYPtr, to: currYPtr, bx: bx, by: by, stride: dx, blockSize: targetBSize)
                            copyBlockPointer(from: prevCbPtr, to: currCbPtr, bx: bx/2, by: by/2, stride: targetCbDx, blockSize: tCbSize)
                            copyBlockPointer(from: prevCrPtr, to: currCrPtr, bx: bx/2, by: by/2, stride: targetCbDx, blockSize: tCbSize)
                        default: break
                        }
                    } else {
                        switch mode {
                        case .skip_ltr:
                            copyBlockSafe(from: ltrYPtr, to: currYPtr, bx: bx, by: by, width: dx, height: dy, blockSize: targetBSize)
                            copyBlockSafe(from: ltrCbPtr, to: currCbPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: tCbSize)
                            copyBlockSafe(from: ltrCrPtr, to: currCrPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: tCbSize)
                        case .skip_prev:
                            copyBlockSafe(from: prevYPtr, to: currYPtr, bx: bx, by: by, width: dx, height: dy, blockSize: targetBSize)
                            copyBlockSafe(from: prevCbPtr, to: currCbPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: tCbSize)
                            copyBlockSafe(from: prevCrPtr, to: currCrPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: tCbSize)
                        default: break
                        }
                    }
                }
            }
        }
    }
    
    let releaseY: @Sendable () -> Void = { r2Y() }
    let releaseCb: @Sendable () -> Void = { r2Cb() }
    let releaseCr: @Sendable () -> Void = { r2Cr() }
    
    let reconstructed = PlaneData420(width: dx, height: dy, y: mutReconL2Y, cb: mutReconL2Cb, cr: mutReconL2Cr)
    
    debugLog({
        return "  [Summary] Layer0=\(layer0.count) Layer1=\(layer1.count) Layer2=\(layer2.count) total=\(layer0.count + layer1.count + layer2.count) bytes"
    }())
    
    var out: [UInt8] = []
    let frameHeader = VEVCFrameHeader(frameType: .pFrame, hasRefDir: true, skipMapSize: skipMapData.count, mvsSize: mvData.count, refDirSize: refDirBuf.count, layer0Size: layer0.count, layer1Size: layer1.count, layer2Size: layer2.count)
    out.append(contentsOf: frameHeader.serialize(profile: profile))
    out.append(contentsOf: skipMapData)
    out.append(contentsOf: mvData)
    out.append(contentsOf: refDirBuf)
    out.append(contentsOf: layer0)
    out.append(contentsOf: layer1)
    out.append(contentsOf: layer2)
    
    return (out, reconstructed, mvs, sads, { releaseY(); releaseCb(); releaseCr() }, nextSub2Res, nextSub1Res)
}

@inline(__always)
func computeZeroSAD16x16(
    cY: UnsafePointer<Int16>, rY: UnsafePointer<Int16>,
    cCb: UnsafePointer<Int16>, rCb: UnsafePointer<Int16>,
    cCr: UnsafePointer<Int16>, rCr: UnsafePointer<Int16>,
    bx: Int, by: Int, width: Int, limit: Int = Int.max
) -> Int {
    var sad: Int = 0
    let strideY = width
    let strideC = (width + 1) / 2
    let bxC = bx / 2
    let byC = by / 2

    for y in 0..<16 {
        let offset = (by + y) * strideY + bx
        for x in 0..<16 {
            sad &+= Int((Int32(cY[offset + x]) - Int32(rY[offset + x])).magnitude)
        }
        if limit < sad { return sad }
    }
    
    for y in 0..<8 {
        let offset = (byC + y) * strideC + bxC
        for x in 0..<8 {
            sad &+= Int((Int32(cCb[offset + x]) - Int32(rCb[offset + x])).magnitude)
            sad &+= Int((Int32(cCr[offset + x]) - Int32(rCr[offset + x])).magnitude)
        }
        if limit < sad { return sad }
    }
    return sad
}

@inline(__always)
func computeZeroSADSubBlock(
    cY: UnsafePointer<Int16>, rY: UnsafePointer<Int16>,
    cCb: UnsafePointer<Int16>, rCb: UnsafePointer<Int16>,
    cCr: UnsafePointer<Int16>, rCr: UnsafePointer<Int16>,
    bx: Int, by: Int, width: Int, height: Int,
    subWidth: Int, subHeight: Int, subWc: Int, subHc: Int,
    limit: Int = Int.max
) -> Int {
    var sad: Int = 0
    let strideY = width
    for y in 0..<subHeight {
        let yy = by + y
        let offset = yy * strideY + bx
        for x in 0..<subWidth {
            sad &+= Int((Int32(cY[offset + x]) - Int32(rY[offset + x])).magnitude)
        }
        if limit < sad { return sad }
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
        if limit < sad { return sad }
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

