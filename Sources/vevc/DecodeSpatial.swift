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
        
        if let y = parentYBlocks { pool.putBlockViewArray64(y) }
        if let cb = parentCbBlocks { pool.putBlockViewArray64(cb) }
        if let cr = parentCrBlocks { pool.putBlockViewArray64(cr) }
        
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
        
        current = try await decodeLayer32(r: layer2Data, pool: pool, layer: 2, dx: l2dx, dy: l2dy, prev: current, parentYBlocks: parentYBlocks, parentCbBlocks: parentCbBlocks, parentCrBlocks: parentCrBlocks, predictedPd: predictedPd, nextPd: nextPd, mvs: mvs, refDirs: refDirs, roundOffset: roundOffset, skipMap: skipMap)
        
        if let y = parentYBlocks {
            if hasLayer1 { pool.putBlockViewArray256(y) }
            else { pool.putBlockViewArray64(y) }
        }
        if let cb = parentCbBlocks {
            if hasLayer1 { pool.putBlockViewArray256(cb) }
            else { pool.putBlockViewArray64(cb) }
        }
        if let cr = parentCrBlocks {
            if hasLayer1 { pool.putBlockViewArray256(cr) }
            else { pool.putBlockViewArray64(cr) }
        }
        parentYBlocks = nil
        parentCbBlocks = nil
        parentCrBlocks = nil
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
                    await applyScaledBidirectionalMotionCompensationLuma(plane: &current.y, prevPlane: tPrev.y, nextPlane: tNext.y, mvs: tMVs, refDirs: dirs, skipMap: skipMap, width: l1dx, height: l1dy, lumaBlockSize: 16, mvShift: 1, roundOffset: roundOffset)
                    await applyScaledBidirectionalMotionCompensationChroma(plane: &current.cb, prevPlane: tPrev.cb, nextPlane: tNext.cb, mvs: tMVs, refDirs: dirs, skipMap: skipMap, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
                    await applyScaledBidirectionalMotionCompensationChroma(plane: &current.cr, prevPlane: tPrev.cr, nextPlane: tNext.cr, mvs: tMVs, refDirs: dirs, skipMap: skipMap, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
                } else {
                    await applyScaledMotionCompensationLuma(plane: &current.y, prevPlane: tPrev.y, mvs: tMVs, skipMap: skipMap, width: l1dx, height: l1dy, lumaBlockSize: 16, mvShift: 1, roundOffset: roundOffset)
                    await applyScaledMotionCompensationChroma(plane: &current.cb, prevPlane: tPrev.cb, mvs: tMVs, skipMap: skipMap, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
                    await applyScaledMotionCompensationChroma(plane: &current.cr, prevPlane: tPrev.cr, mvs: tMVs, skipMap: skipMap, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
                }
                
                clampPlane(plane: &current.y)
                clampPlane(plane: &current.cb)
                clampPlane(plane: &current.cr)
                
                applyDeblockingFilterN(plane: &current.y, width: l1dx, height: l1dy, qStep: qtYStep, blockSize: 16)
                let cStep = min(qtCStep * 2, 255)
                applyDeblockingFilterN(plane: &current.cb, width: cbDx1, height: cbDy1, qStep: cStep, blockSize: 8)
                applyDeblockingFilterN(plane: &current.cr, width: cbDx1, height: cbDy1, qStep: cStep, blockSize: 8)
            }
        } else {
            // Apply MC at layer0 resolution (8x8 blocks, mvShift=2 for luma, mvShift=0 for chroma)
            if let tMVs = mvs, let tPrev = predictedPd {
                let cbDx0 = (l0dx + 1) / 2
                let cbDy0 = (l0dy + 1) / 2
                if let tNext = nextPd, let dirs = refDirs {
                    await applyScaledBidirectionalMotionCompensationLuma(plane: &current.y, prevPlane: tPrev.y, nextPlane: tNext.y, mvs: tMVs, refDirs: dirs, skipMap: skipMap, width: l0dx, height: l0dy, lumaBlockSize: 8, mvShift: 2, roundOffset: roundOffset)
                    await applyScaledBidirectionalMotionCompensationChroma(plane: &current.cb, prevPlane: tPrev.cb, nextPlane: tNext.cb, mvs: tMVs, refDirs: dirs, skipMap: skipMap, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 0, roundOffset: roundOffset)
                    await applyScaledBidirectionalMotionCompensationChroma(plane: &current.cr, prevPlane: tPrev.cr, nextPlane: tNext.cr, mvs: tMVs, refDirs: dirs, skipMap: skipMap, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 0, roundOffset: roundOffset)
                } else {
                    await applyScaledMotionCompensationLuma(plane: &current.y, prevPlane: tPrev.y, mvs: tMVs, skipMap: skipMap, width: l0dx, height: l0dy, lumaBlockSize: 8, mvShift: 2, roundOffset: roundOffset)
                    await applyScaledMotionCompensationChroma(plane: &current.cb, prevPlane: tPrev.cb, mvs: tMVs, skipMap: skipMap, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 0, roundOffset: roundOffset)
                    await applyScaledMotionCompensationChroma(plane: &current.cr, prevPlane: tPrev.cr, mvs: tMVs, skipMap: skipMap, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 0, roundOffset: roundOffset)
                }
                
                clampPlane(plane: &current.y)
                clampPlane(plane: &current.cb)
                clampPlane(plane: &current.cr)
                
                applyDeblockingFilterN(plane: &current.y, width: l0dx, height: l0dy, qStep: qtYStep, blockSize: 8)
                let cStep = min(qtCStep * 2, 255)
                applyDeblockingFilterN(plane: &current.cb, width: cbDx0, height: cbDy0, qStep: cStep, blockSize: 4)
                applyDeblockingFilterN(plane: &current.cr, width: cbDx0, height: cbDy0, qStep: cStep, blockSize: 4)
            }
        }
    }
    
    if let y = parentYBlocks {
        if hasLayer1 { pool.putBlockViewArray256(y) }
        else { pool.putBlockViewArray64(y) }
    }
    if let cb = parentCbBlocks {
        if hasLayer1 { pool.putBlockViewArray256(cb) }
        else { pool.putBlockViewArray64(cb) }
    }
    if let cr = parentCrBlocks {
        if hasLayer1 { pool.putBlockViewArray256(cr) }
        else { pool.putBlockViewArray64(cr) }
    }
    
    let targetDx = hasLayer2 ? l2dx : (hasLayer1 ? l1dx : l0dx)
    let targetDy = hasLayer2 ? l2dy : (hasLayer1 ? l1dy : l0dy)
    let targetBSize = hasLayer2 ? 32 : (hasLayer1 ? 16 : 8)

    if profile == 0x02, let map = skipMap {
        let bw = (dx + 31) / 32
        let targetCbDx = (targetDx + 1) / 2
        let targetCbDy = (targetDy + 1) / 2
        let scale = hasLayer2 ? 1 : (hasLayer1 ? 2 : 4)
        
        let tCbSize = targetBSize / 2
        let pPd = predictedPd ?? PlaneData420(width: dx, height: dy, y: [], cb: [], cr: [])
        let lPd = nextPd ?? pPd
        
        withUnsafePointers(
            lPd.y, lPd.cb, lPd.cr,
            pPd.y, pPd.cb, pPd.cr,
            mut: &current.y, mut: &current.cb, mut: &current.cr
        ) { ltrYPtr, ltrCbPtr, ltrCrPtr, prevYPtr, prevCbPtr, prevCrPtr, currYPtr, currCbPtr, currCrPtr in
            for i in 0..<map.count {
                let mode = map[i]
                if mode != .inter {
                    let bxL2 = (i % bw) * 32
                    let byL2 = (i / bw) * 32
                    
                    let bx = bxL2 / scale
                    let by = byL2 / scale
                    
                    if bx + targetBSize <= targetDx && by + targetBSize <= targetDy {
                        switch mode {
                        case .skip_ltr where nextPd != nil:
                            copyBlockPointer(from: ltrYPtr, to: currYPtr, bx: bx, by: by, stride: targetDx, blockSize: targetBSize)
                            copyBlockPointer(from: ltrCbPtr, to: currCbPtr, bx: bx/2, by: by/2, stride: targetCbDx, blockSize: tCbSize)
                            copyBlockPointer(from: ltrCrPtr, to: currCrPtr, bx: bx/2, by: by/2, stride: targetCbDx, blockSize: tCbSize)
                        case .skip_prev:
                            copyBlockPointer(from: prevYPtr, to: currYPtr, bx: bx, by: by, stride: targetDx, blockSize: targetBSize)
                            copyBlockPointer(from: prevCbPtr, to: currCbPtr, bx: bx/2, by: by/2, stride: targetCbDx, blockSize: tCbSize)
                            copyBlockPointer(from: prevCrPtr, to: currCrPtr, bx: bx/2, by: by/2, stride: targetCbDx, blockSize: tCbSize)
                        default: break
                        }
                    } else {
                        switch mode {
                        case .skip_ltr where nextPd != nil:
                            copyBlockSafe(from: ltrYPtr, to: currYPtr, bx: bx, by: by, width: targetDx, height: targetDy, blockSize: targetBSize)
                            copyBlockSafe(from: ltrCbPtr, to: currCbPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: tCbSize)
                            copyBlockSafe(from: ltrCrPtr, to: currCrPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: tCbSize)
                        case .skip_prev:
                            copyBlockSafe(from: prevYPtr, to: currYPtr, bx: bx, by: by, width: targetDx, height: targetDy, blockSize: targetBSize)
                            copyBlockSafe(from: prevCbPtr, to: currCbPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: tCbSize)
                            copyBlockSafe(from: prevCrPtr, to: currCrPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: tCbSize)
                        default: break
                        }
                    }
                }
            }
        }
    }
    
    return current
}

@inline(__always)
fileprivate func clampPlane(plane: inout [Int16]) {
    plane.withUnsafeMutableBufferPointer { ptr in
        guard let base = ptr.baseAddress else { return }
        var x = 0
        let c = ptr.count
        let vMin = SIMD16<Int16>(repeating: -128)
        let vMax = SIMD16<Int16>(repeating: 127)
        while x < c - 15 {
            let p = base.advanced(by: x)
            let v = UnsafeRawPointer(p).loadUnaligned(as: SIMD16<Int16>.self)
            let clampedMin = v.replacing(with: vMin, where: v .< vMin)
            let clamped = clampedMin.replacing(with: vMax, where: clampedMin .> vMax)
            UnsafeMutableRawPointer(p).storeBytes(of: clamped, as: SIMD16<Int16>.self)
            x &+= 16
        }
        while x < c {
            let v = ptr[x]
            if v < -128 { ptr[x] = -128 }
            else if v > 127 { ptr[x] = 127 }
            x &+= 1
        }
    }
}


@inline(__always)
func copyBlockPointer(from src: UnsafePointer<Int16>, to dst: UnsafeMutablePointer<Int16>, bx: Int, by: Int, stride: Int, blockSize: Int) {
    switch blockSize {
    case 32:
        for y in 0..<32 {
            let offset = (by + y) * stride + bx
            let dstPtr = UnsafeMutableRawPointer(dst.advanced(by: offset))
            let srcPtr = UnsafeRawPointer(src.advanced(by: offset))
            dstPtr.storeBytes(of: srcPtr.loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
            dstPtr.advanced(by: 32).storeBytes(of: srcPtr.advanced(by: 32).loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
        }
    case 16:
        for y in 0..<16 {
            let offset = (by + y) * stride + bx
            let dstPtr = UnsafeMutableRawPointer(dst.advanced(by: offset))
            let srcPtr = UnsafeRawPointer(src.advanced(by: offset))
            dstPtr.storeBytes(of: srcPtr.loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
        }
    case 8:
        for y in 0..<8 {
            let offset = (by + y) * stride + bx
            let dstPtr = UnsafeMutableRawPointer(dst.advanced(by: offset))
            let srcPtr = UnsafeRawPointer(src.advanced(by: offset))
            dstPtr.storeBytes(of: srcPtr.loadUnaligned(as: SIMD8<Int16>.self), as: SIMD8<Int16>.self)
        }
    default:
        for y in 0..<blockSize {
            let offset = (by + y) * stride + bx
            dst.advanced(by: offset).update(from: src.advanced(by: offset), count: blockSize)
        }
    }
}

@inline(__always)
func copyBlockSafe(from src: UnsafePointer<Int16>, to dst: UnsafeMutablePointer<Int16>, bx: Int, by: Int, width: Int, height: Int, blockSize: Int) {
    let maxY = min(by + blockSize, height)
    let maxX = min(bx + blockSize, width)
    let copyCount = maxX - bx
    if copyCount <= 0 { return }
    
    switch copyCount {
    case 32:
        for y in by..<maxY {
            let offset = y * width + bx
            let dstPtr = UnsafeMutableRawPointer(dst.advanced(by: offset))
            let srcPtr = UnsafeRawPointer(src.advanced(by: offset))
            dstPtr.storeBytes(of: srcPtr.loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
            dstPtr.advanced(by: 32).storeBytes(of: srcPtr.advanced(by: 32).loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
        }
    case 16:
        for y in by..<maxY {
            let offset = y * width + bx
            let dstPtr = UnsafeMutableRawPointer(dst.advanced(by: offset))
            let srcPtr = UnsafeRawPointer(src.advanced(by: offset))
            dstPtr.storeBytes(of: srcPtr.loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
        }
    case 8:
        for y in by..<maxY {
            let offset = y * width + bx
            let dstPtr = UnsafeMutableRawPointer(dst.advanced(by: offset))
            let srcPtr = UnsafeRawPointer(src.advanced(by: offset))
            dstPtr.storeBytes(of: srcPtr.loadUnaligned(as: SIMD8<Int16>.self), as: SIMD8<Int16>.self)
        }
    default:
        for y in by..<maxY {
            let offset = y * width + bx
            dst.advanced(by: offset).update(from: src.advanced(by: offset), count: copyCount)
        }
    }
}
