// MARK: - Decode Spatial

@inline(__always)
func decodeSpatialLayers(r: [UInt8], pool: BlockViewPool, maxLayer: Int, dx: Int, dy: Int, predictedPd: PlaneData420? = nil, nextPd: PlaneData420? = nil, roundOffset: Int, profile: UInt8 = 0x01) async throws -> Image16 {
    var offset = 0

    // Compute per-layer dimensions matching encoder DWT subband sizes:
    // Layer2 (32x32): original size
    // Layer1 (16x16): DWT LL subband of Layer2 = (dx+1)/2 × (dy+1)/2
    // Layer0 (Base8): DWT LL subband of Layer1 = ((dx+1)/2+1)/2 × ((dy+1)/2+1)/2
    let l2dx = dx
    let l2dy = dy
    let l1dx = (dx + 1) / 2
    let l1dy = (dy + 1) / 2
    let l0dx = (l1dx + 1) / 2
    let l0dy = (l1dy + 1) / 2

    let frameHeader = try VEVCFrameHeader.deserialize(from: r, offset: &offset, profile: profile)
    if frameHeader.isCopyFrame {
        throw DecodeError.insufficientDataContext("decodeSpatialLayers passed copy frame")
    }
    
    var mvs: MotionVectors? = nil
    var refDirs: [Bool]? = nil
    
    let mvsCount = deriveMVCount(width: dx, height: dy)
    
    var skipMap: [BlockMode]? = nil
    if frameHeader.isIFrame != true && frameHeader.skipMapSize > 0 {
        guard (offset + frameHeader.skipMapSize) <= r.count else { throw DecodeError.insufficientData }
        let smData = Array(r[offset..<(offset + frameHeader.skipMapSize)])
        skipMap = try decodeSkipMap(data: smData, count: mvsCount)
        offset += frameHeader.skipMapSize
    }
    
    if frameHeader.isIFrame != true && 0 < frameHeader.mvsSize {
        guard (offset + frameHeader.mvsSize) <= r.count else { throw DecodeError.insufficientData }
        mvs = try decodeMVs(data: Array(r[offset..<(offset + frameHeader.mvsSize)]), count: mvsCount, skipMap: skipMap, profile: profile)
        offset += frameHeader.mvsSize
    }
    
    // Direction flag only exists for bidirectional prediction frames.
    // Indicates whether each block uses forward (prev) or backward (next) reference.
    if frameHeader.hasRefDir {
        let refDirByteCount = frameHeader.refDirSize
        guard (offset + refDirByteCount) <= r.count else { throw DecodeError.insufficientData }
        let refDirBuf = Array(r[offset..<(offset + refDirByteCount)])
        offset += refDirByteCount
        
        if nextPd != nil {
            var dirs = [Bool]()
            dirs.reserveCapacity(mvsCount)
            for i in 0..<mvsCount {
                let byteIdx = i / 8
                let bitIdx = i % 8
                let isBackward = (byteIdx < refDirBuf.count) && ((refDirBuf[byteIdx] & UInt8(1 << bitIdx)) != 0)
                dirs.append(isBackward)
            }
            refDirs = dirs
        }
    }
    
    guard (offset + frameHeader.layer0Size) <= r.count else { throw DecodeError.insufficientData }
    let layer0Data = Array(r[offset..<(offset + frameHeader.layer0Size)])
    offset += frameHeader.layer0Size
    
    // Base layer (layer 0) is always Base8
    let (baseImg, base8YBlocks, base8CbBlocks, base8CrBlocks, qtYStep, qtCStep) = try await decodeBase8(r: layer0Data, pool: pool, layer: 0, dx: l0dx, dy: l0dy, isIFrame: frameHeader.isIFrame)
    var current = baseImg
    var parentYBlocks: [BlockView]? = base8YBlocks
    var parentCbBlocks: [BlockView]? = base8CbBlocks
    var parentCrBlocks: [BlockView]? = base8CrBlocks
    
    // Determine the effective layer based on maxLayer AND available data.
    // When the splitter strips upper layers, their size becomes 0.
    // MC must be applied at the highest available layer's resolution.
    let hasLayer1 = (1 <= maxLayer && 0 < frameHeader.layer1Size)
    let hasLayer2 = (2 <= maxLayer && 0 < frameHeader.layer2Size)
    
    // Decode Layer1 if data is present
    if hasLayer1 {
        guard (offset + frameHeader.layer1Size) <= r.count else { throw DecodeError.insufficientData }
        let layer1Data = Array(r[offset..<(offset + frameHeader.layer1Size)])
        offset += frameHeader.layer1Size
        
        let (l16Img, l16YBlocks, l16CbBlocks, l16CrBlocks) = try await decodeLayer16(r: layer1Data, pool: pool, layer: 1, dx: l1dx, dy: l1dy, prev: current, parentYBlocks: parentYBlocks, parentCbBlocks: parentCbBlocks, parentCrBlocks: parentCrBlocks)
        current = l16Img
        parentYBlocks = l16YBlocks
        parentCbBlocks = l16CbBlocks
        parentCrBlocks = l16CrBlocks
    } else {
        offset += frameHeader.layer1Size
    }
    
    // Decode Layer2 if data is present (MC is applied inside decodeLayer32)
    if hasLayer2 {
        guard (offset + frameHeader.layer2Size) <= r.count else { throw DecodeError.insufficientData }
        let layer2Data = Array(r[offset..<(offset + frameHeader.layer2Size)])
        offset += frameHeader.layer2Size
        
        current = try await decodeLayer32(r: layer2Data, pool: pool, layer: 2, dx: l2dx, dy: l2dy, prev: current, parentYBlocks: parentYBlocks, parentCbBlocks: parentCbBlocks, parentCrBlocks: parentCrBlocks, predictedPd: predictedPd, nextPd: nextPd, mvs: mvs, refDirs: refDirs, roundOffset: roundOffset)
    } else {
        offset += frameHeader.layer2Size
        
        // Layer2 data is absent: apply MC at the highest available layer's resolution.
        // This handles the case where the splitter stripped upper layers.
        if hasLayer1 {
            // Apply MC at layer1 resolution (16x16 blocks, mvShift=1)
            if let tMVs = mvs, let tPrev = predictedPd {
                let cbDx1 = (l1dx + 1) / 2
                let cbDy1 = (l1dy + 1) / 2
                if let tNext = nextPd, let dirs = refDirs {
                    applyScaledBidirectionalMotionCompensationLuma(plane: &current.y, prevPlane: tPrev.y, nextPlane: tNext.y, mvs: tMVs, refDirs: dirs, width: l1dx, height: l1dy, lumaBlockSize: 16, mvShift: 1, roundOffset: roundOffset)
                    applyScaledBidirectionalMotionCompensationChroma(plane: &current.cb, prevPlane: tPrev.cb, nextPlane: tNext.cb, mvs: tMVs, refDirs: dirs, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
                    applyScaledBidirectionalMotionCompensationChroma(plane: &current.cr, prevPlane: tPrev.cr, nextPlane: tNext.cr, mvs: tMVs, refDirs: dirs, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
                } else {
                    applyScaledMotionCompensationLuma(plane: &current.y, prevPlane: tPrev.y, mvs: tMVs, width: l1dx, height: l1dy, lumaBlockSize: 16, mvShift: 1, roundOffset: roundOffset)
                    applyScaledMotionCompensationChroma(plane: &current.cb, prevPlane: tPrev.cb, mvs: tMVs, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
                    applyScaledMotionCompensationChroma(plane: &current.cr, prevPlane: tPrev.cr, mvs: tMVs, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
                }
                
                applyDeblockingFilterN(plane: &current.y, width: l1dx, height: l1dy, qStep: qtYStep, blockSize: 16)
                let cStep = min(qtCStep * 2, 255)
                applyDeblockingFilterN(plane: &current.cb, width: cbDx1, height: cbDy1, qStep: cStep, blockSize: 8)
                applyDeblockingFilterN(plane: &current.cr, width: cbDx1, height: cbDy1, qStep: cStep, blockSize: 8)
            }
        } else {
            // Apply MC at layer0 resolution (8x8 blocks, mvShift=2)
            if let tMVs = mvs, let tPrev = predictedPd {
                let cbDx0 = (l0dx + 1) / 2
                let cbDy0 = (l0dy + 1) / 2
                if let tNext = nextPd, let dirs = refDirs {
                    applyScaledBidirectionalMotionCompensationLuma(plane: &current.y, prevPlane: tPrev.y, nextPlane: tNext.y, mvs: tMVs, refDirs: dirs, width: l0dx, height: l0dy, lumaBlockSize: 8, mvShift: 2, roundOffset: roundOffset)
                    applyScaledBidirectionalMotionCompensationChroma(plane: &current.cb, prevPlane: tPrev.cb, nextPlane: tNext.cb, mvs: tMVs, refDirs: dirs, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 2, roundOffset: roundOffset)
                    applyScaledBidirectionalMotionCompensationChroma(plane: &current.cr, prevPlane: tPrev.cr, nextPlane: tNext.cr, mvs: tMVs, refDirs: dirs, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 2, roundOffset: roundOffset)
                } else {
                    applyScaledMotionCompensationLuma(plane: &current.y, prevPlane: tPrev.y, mvs: tMVs, width: l0dx, height: l0dy, lumaBlockSize: 8, mvShift: 2, roundOffset: roundOffset)
                    applyScaledMotionCompensationChroma(plane: &current.cb, prevPlane: tPrev.cb, mvs: tMVs, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 2, roundOffset: roundOffset)
                    applyScaledMotionCompensationChroma(plane: &current.cr, prevPlane: tPrev.cr, mvs: tMVs, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 2, roundOffset: roundOffset)
                }
                
                applyDeblockingFilterN(plane: &current.y, width: l0dx, height: l0dy, qStep: qtYStep, blockSize: 8)
                let cStep = min(qtCStep * 2, 255)
                applyDeblockingFilterN(plane: &current.cb, width: cbDx0, height: cbDy0, qStep: cStep, blockSize: 4)
                applyDeblockingFilterN(plane: &current.cr, width: cbDx0, height: cbDy0, qStep: cStep, blockSize: 4)
            }
        }
    }
    
    let targetDx = hasLayer2 ? l2dx : (hasLayer1 ? l1dx : l0dx)
    let targetDy = hasLayer2 ? l2dy : (hasLayer1 ? l1dy : l0dy)
    let targetBSize = hasLayer2 ? 32 : (hasLayer1 ? 16 : 8)

    if profile == 0x02, let ltrPd = nextPd, let map = skipMap {
        let bw = (dx + 31) / 32
        let targetCbDx = (targetDx + 1) / 2
        let targetCbDy = (targetDy + 1) / 2
        let scale = hasLayer2 ? 1 : (hasLayer1 ? 2 : 4)
        
        for i in 0..<map.count {
            if map[i] == .skip_ltr {
                let bxL2 = (i % bw) * 32
                let byL2 = (i / bw) * 32
                
                let bx = bxL2 / scale
                let by = byL2 / scale
                
                copyBlock(from: ltrPd.y, to: &current.y, bx: bx, by: by, width: targetDx, height: targetDy, blockSize: targetBSize)
                copyBlock(from: ltrPd.cb, to: &current.cb, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: targetBSize/2)
                copyBlock(from: ltrPd.cr, to: &current.cr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: targetBSize/2)
            }
        }
    }
    
    return current
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
