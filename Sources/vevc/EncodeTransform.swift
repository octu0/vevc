// MARK: - Transform Functions

@inline(__always)
func isEffectivelyZero32(data base: UnsafeMutablePointer<Int16>, threshold: Int) -> Bool {
    let th = Int16(threshold)
    let thPos = SIMD16<Int16>(repeating: th)
    let thNeg = SIMD16<Int16>(repeating: -th)

    let lowerHalfBase = base + 16 * 32
    for i in stride(from: 0, to: 512, by: 16) {
        let vec: SIMD16<Int16> = UnsafeRawPointer(lowerHalfBase + i).loadUnaligned(as: SIMD16<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
    }
    for y in 0..<16 {
        let ptr = base + y * 32 + 16
        let vec: SIMD16<Int16> = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD16<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
    }

    let zeroVec = SIMD16<Int16>(repeating: 0)
    for i in stride(from: 0, to: 512, by: 16) {
        let ptr = UnsafeMutableRawPointer(lowerHalfBase + i).assumingMemoryBound(to: SIMD16<Int16>.self)
        ptr.pointee = zeroVec
    }
    for y in 0..<16 {
        let ptr = UnsafeMutableRawPointer(base + y * 32 + 16).assumingMemoryBound(to: SIMD16<Int16>.self)
        ptr.pointee = zeroVec
    }
    return true
}

@inline(__always)
func isEffectivelyZero16(data base: UnsafeMutablePointer<Int16>, threshold: Int) -> Bool {
    let th = Int16(threshold)
    let thPos = SIMD8<Int16>(repeating: th)
    let thNeg = SIMD8<Int16>(repeating: -th)

    let lowerHalfBase = base + 8 * 16
    for i in stride(from: 0, to: 128, by: 8) {
        let vec: SIMD8<Int16> = UnsafeRawPointer(lowerHalfBase + i).loadUnaligned(as: SIMD8<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
    }
    for y in 0..<8 {
        let ptr = base + y * 16 + 8
        let vec: SIMD8<Int16> = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
    }

    let zeroVec = SIMD8<Int16>(repeating: 0)
    for i in stride(from: 0, to: 128, by: 8) {
        let ptr = UnsafeMutableRawPointer(lowerHalfBase + i).assumingMemoryBound(to: SIMD8<Int16>.self)
        ptr.pointee = zeroVec
    }
    for y in 0..<8 {
        let ptr = UnsafeMutableRawPointer(base + y * 16 + 8).assumingMemoryBound(to: SIMD8<Int16>.self)
        ptr.pointee = zeroVec
    }
    return true
}

@inline(__always)
func isEffectivelyZero8(data base: UnsafeMutablePointer<Int16>, threshold: Int) -> Bool {
    let th = Int16(threshold)
    let thPos = SIMD4<Int16>(repeating: th)
    let thNeg = SIMD4<Int16>(repeating: -th)

    let lowerHalfBase = base + 4 * 8
    for i in stride(from: 0, to: 32, by: 4) {
        let vec: SIMD4<Int16> = UnsafeRawPointer(lowerHalfBase + i).loadUnaligned(as: SIMD4<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
    }
    for y in 0..<4 {
        let ptr = base + y * 8 + 4
        let vec: SIMD4<Int16> = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD4<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
    }

    let zeroVec = SIMD4<Int16>(repeating: 0)
    for i in stride(from: 0, to: 32, by: 4) {
        let ptr = UnsafeMutableRawPointer(lowerHalfBase + i).assumingMemoryBound(to: SIMD4<Int16>.self)
        ptr.pointee = zeroVec
    }
    for y in 0..<4 {
        let ptr = UnsafeMutableRawPointer(base + y * 8 + 4).assumingMemoryBound(to: SIMD4<Int16>.self)
        ptr.pointee = zeroVec
    }
    return true
}

@inline(__always)
func checkQuadrants16x16(base: UnsafeMutablePointer<Int16>, stride: Int, q0: inout Bool, q1: inout Bool, q2: inout Bool, q3: inout Bool) {
    let zero8 = SIMD8<Int16>(repeating: 0)
    for y in 0..<8 {
        let ptr = base.advanced(by: y * stride)
        if q0 != true {
            let v = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
            if any(v .!= zero8) { q0 = true }
        }
        if q1 != true {
            let v = UnsafeRawPointer(ptr.advanced(by: 8)).loadUnaligned(as: SIMD8<Int16>.self)
            if any(v .!= zero8) { q1 = true }
        }
    }
    for y in 8..<16 {
        let ptr = base.advanced(by: y * stride)
        if q2 != true {
            let v = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
            if any(v .!= zero8) { q2 = true }
        }
        if q3 != true {
            let v = UnsafeRawPointer(ptr.advanced(by: 8)).loadUnaligned(as: SIMD8<Int16>.self)
            if any(v .!= zero8) { q3 = true }
        }
    }
}

@inline(__always)
func shouldSplit32WithLL(data base: UnsafeMutablePointer<Int16>) -> Bool {
    var q0 = false, q1 = false, q2 = false, q3 = false
    
    checkQuadrants16x16(base: base, stride: 32, q0: &q0, q1: &q1, q2: &q2, q3: &q3)
    if (q0 && q1 && q2 && q3) != true {
        checkQuadrants16x16(base: base + 16, stride: 32, q0: &q0, q1: &q1, q2: &q2, q3: &q3)
    }
    if (q0 && q1 && q2 && q3) != true {
        checkQuadrants16x16(base: base + 16 * 32, stride: 32, q0: &q0, q1: &q1, q2: &q2, q3: &q3)
    }
    if (q0 && q1 && q2 && q3) != true {
        checkQuadrants16x16(base: base + 16 * 32 + 16, stride: 32, q0: &q0, q1: &q1, q2: &q2, q3: &q3)
    }
    
    // If not all 4 quadrants are busy, splitting avoids encoding zeros.
    return (q0 && q1 && q2 && q3) != true
}

@inline(__always)
func shouldSplit32WithoutLL(data base: UnsafeMutablePointer<Int16>) -> Bool {
    // LL quadrant is skipped because it is encoded separately (DPCM path)
    var q0 = false, q1 = false, q2 = false, q3 = false
    
    if (q0 && q1 && q2 && q3) != true {
        checkQuadrants16x16(base: base + 16, stride: 32, q0: &q0, q1: &q1, q2: &q2, q3: &q3)
    }
    if (q0 && q1 && q2 && q3) != true {
        checkQuadrants16x16(base: base + 16 * 32, stride: 32, q0: &q0, q1: &q1, q2: &q2, q3: &q3)
    }
    if (q0 && q1 && q2 && q3) != true {
        checkQuadrants16x16(base: base + 16 * 32 + 16, stride: 32, q0: &q0, q1: &q1, q2: &q2, q3: &q3)
    }
    
    return (q0 && q1 && q2 && q3) != true
}

@inline(__always)
func checkQuadrants8x8(base: UnsafeMutablePointer<Int16>, stride: Int, q0: inout Bool, q1: inout Bool, q2: inout Bool, q3: inout Bool) {
    let zero4 = SIMD4<Int16>(repeating: 0)
    for y in 0..<4 {
        let ptr = base.advanced(by: y * stride)
        if q0 != true {
            let v = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD4<Int16>.self)
            if any(v .!= zero4) { q0 = true }
        }
        if q1 != true {
            let v = UnsafeRawPointer(ptr.advanced(by: 4)).loadUnaligned(as: SIMD4<Int16>.self)
            if any(v .!= zero4) { q1 = true }
        }
    }
    for y in 4..<8 {
        let ptr = base.advanced(by: y * stride)
        if q2 != true {
            let v = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD4<Int16>.self)
            if any(v .!= zero4) { q2 = true }
        }
        if q3 != true {
            let v = UnsafeRawPointer(ptr.advanced(by: 4)).loadUnaligned(as: SIMD4<Int16>.self)
            if any(v .!= zero4) { q3 = true }
        }
    }
}

@inline(__always)
func shouldSplit16(data base: UnsafeMutablePointer<Int16>) -> Bool {
    // LL quadrant is skipped because it is encoded separately (DPCM path)
    var q0 = false, q1 = false, q2 = false, q3 = false
    
    if (q0 && q1 && q2 && q3) != true {
        checkQuadrants8x8(base: base + 8, stride: 16, q0: &q0, q1: &q1, q2: &q2, q3: &q3)
    }
    if (q0 && q1 && q2 && q3) != true {
        checkQuadrants8x8(base: base + 8 * 16, stride: 16, q0: &q0, q1: &q1, q2: &q2, q3: &q3)
    }
    if (q0 && q1 && q2 && q3) != true {
        checkQuadrants8x8(base: base + 8 * 16 + 8, stride: 16, q0: &q0, q1: &q1, q2: &q2, q3: &q3)
    }
    
    return (q0 && q1 && q2 && q3) != true
}

@inline(__always)
func isEffectivelyZeroBase4(data base: UnsafeMutablePointer<Int16>, threshold: Int) -> Bool {
    // I-frame path: LL must be exactly zero (DPCM requires exact values)
    for y in 0..<2 {
        let ptr = base + y * 4
        let vec: SIMD2<Int16> = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD2<Int16>.self)
        if vec[0] != 0 || vec[1] != 0 { return false }
    }
    
    let th = Int16(threshold)
    let thPos = SIMD2<Int16>(repeating: th)
    let thNeg = SIMD2<Int16>(repeating: -th)
    
    // Check Subbands
    let lowerHalfBase = base + 2 * 4
    for i in stride(from: 0, to: 8, by: 2) {
        let vec: SIMD2<Int16> = UnsafeRawPointer(lowerHalfBase + i).loadUnaligned(as: SIMD2<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
    }
    for y in 0..<2 {
        let ptr = base + y * 4 + 2
        let vec: SIMD2<Int16> = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD2<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
    }

    let zeroVec = SIMD2<Int16>(repeating: 0)
    for i in stride(from: 0, to: 8, by: 2) {
        let ptr = UnsafeMutableRawPointer(lowerHalfBase + i).assumingMemoryBound(to: SIMD2<Int16>.self)
        ptr.pointee = zeroVec
    }
    for y in 0..<2 {
        let ptr = UnsafeMutableRawPointer(base + y * 4 + 2).assumingMemoryBound(to: SIMD2<Int16>.self)
        ptr.pointee = zeroVec
    }
    return true
}

@inline(__always)
func isEffectivelyZeroBase4PFrame(data base: UnsafeMutablePointer<Int16>, threshold: Int) -> Bool {
    // P-frame path: LL is threshold-checked (residual values after motion compensation)
    let th = Int16(threshold)
    let thPos = SIMD2<Int16>(repeating: th)
    let thNeg = SIMD2<Int16>(repeating: -th)
    
    // Check LL with threshold
    for y in 0..<2 {
        let ptr = base + y * 4
        let vec: SIMD2<Int16> = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD2<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
    }
    
    // Check Subbands
    let lowerHalfBase = base + 2 * 4
    for i in stride(from: 0, to: 8, by: 2) {
        let vec: SIMD2<Int16> = UnsafeRawPointer(lowerHalfBase + i).loadUnaligned(as: SIMD2<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
    }
    for y in 0..<2 {
        let ptr = base + (y * 4) + 2
        let vec: SIMD2<Int16> = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD2<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
    }

    let zeroVec = SIMD2<Int16>(repeating: 0)
    for i in stride(from: 0, to: 8, by: 2) {
        let ptr = UnsafeMutableRawPointer(lowerHalfBase + i).assumingMemoryBound(to: SIMD2<Int16>.self)
        ptr.pointee = zeroVec
    }
    for y in 0..<2 {
        let ptr = UnsafeMutableRawPointer(base + y * 4 + 2).assumingMemoryBound(to: SIMD2<Int16>.self)
        ptr.pointee = zeroVec
    }
    return true
}

@inline(__always)
func isEffectivelyZeroBase32(data base: UnsafeMutablePointer<Int16>, threshold: Int) -> Bool {
    // Check LL
    let zeroVec16 = SIMD16<Int16>(repeating: 0)
    for y in 0..<16 {
        let ptr = base + y * 32
        let vec: SIMD16<Int16> = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD16<Int16>.self)
        let mask = vec .!= zeroVec16
        if any(mask) { return false }
    }
    
    // Check Subbands
    let th = Int16(threshold)
    let thPos = SIMD16<Int16>(repeating: th)
    let thNeg = SIMD16<Int16>(repeating: -th)

    let lowerHalfBase = base + 16 * 32
    for i in stride(from: 0, to: 512, by: 16) {
        let vec: SIMD16<Int16> = UnsafeRawPointer(lowerHalfBase + i).loadUnaligned(as: SIMD16<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
    }
    for y in 0..<16 {
        let ptr = base + y * 32 + 16
        let vec: SIMD16<Int16> = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD16<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) { return false }
    }

    for i in stride(from: 0, to: 512, by: 16) {
        let ptr = UnsafeMutableRawPointer(lowerHalfBase + i).assumingMemoryBound(to: SIMD16<Int16>.self)
        ptr.pointee = zeroVec16
    }
    for y in 0..<16 {
        let ptr = UnsafeMutableRawPointer(base + y * 32 + 16).assumingMemoryBound(to: SIMD16<Int16>.self)
        ptr.pointee = zeroVec16
    }
    return true
}

enum EncodeTask32 {
    case encode16
    case split8(Bool, Bool, Bool, Bool)
}

@inline(__always)
func encodePlaneSubbands32(blocks: inout [BlockView], zeroThreshold: Int, parentBlocks: [BlockView]?, sads: [Int]? = nil, colCount: Int = 0, rowCount: Int = 0) -> [UInt8] {
    var bwFlags = BypassWriter()
    var tasks: [(Int, EncodeTask32)] = []
    tasks.reserveCapacity(blocks.count)
    
    // Spatial adaptive threshold: when colCount/rowCount are provided,
    // apply higher zero-thresholds to peripheral blocks where human
    // visual attention is lower, increasing zero-block rate at edges.
    let useSpatialWeight = 1 < colCount && 1 < rowCount
    
    var zeroCount = 0
    for i in blocks.indices {
        let blockThreshold: Int
        if useSpatialWeight {
            let col = i % colCount
            let row = i / colCount
            let weight = spatialWeight(blockCol: col, blockRow: row, colCount: colCount, rowCount: rowCount)
            blockThreshold = (max(1, zeroThreshold) * weight) / 1024
        } else {
            blockThreshold = zeroThreshold
        }
        let isZero = isEffectivelyZero32(data: blocks[i].base, threshold: blockThreshold)
        if isZero {
            bwFlags.writeBit(true)
            let view = blocks[i]
            let half = 32 / 2
            let base = view.base
            let hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 32)
            let lhView = BlockView(base: base.advanced(by: half * 32), width: half, height: half, stride: 32)
            let hhView = BlockView(base: base.advanced(by: half * 32 + half), width: half, height: half, stride: 32)
            hlView.clearAll()
            lhView.clearAll()
            hhView.clearAll()
            zeroCount += 1
        } else {
            bwFlags.writeBit(false)
            
            let forceSplit = shouldSplit32WithoutLL(data: blocks[i].base)
            if forceSplit {
                bwFlags.writeBit(true)
                
                bwFlags.writeBit(false) // TL isZero = false
                bwFlags.writeBit(false) // TL MB_Type = false (No further split)
                
                bwFlags.writeBit(false) // TR isZero = false
                bwFlags.writeBit(false) // TR MB_Type = false
                
                bwFlags.writeBit(false) // BL isZero = false
                bwFlags.writeBit(false) // BL MB_Type = false
                
                bwFlags.writeBit(false) // BR isZero = false
                bwFlags.writeBit(false) // BR MB_Type = false
                
                tasks.append((i, .split8(true, true, true, true)))
            } else {
                bwFlags.writeBit(false) // MB_Type = false
                tasks.append((i, .encode16))
            }
        }
    }
    bwFlags.flush()
    debugLog({
        let zeroPermyriad = (zeroCount * 10000) / max(1, blocks.count)
        let rateStr = "\(zeroPermyriad / 100).\(zeroPermyriad / 10 % 10)"
        return "    [Subbands] blocks=\(blocks.count) zeroBlocks=\(zeroCount) zeroRate=\(rateStr)%"
    }())
    
    var encoder = EntropyEncoder<DynamicEntropyModel>()
    
    if let pb = parentBlocks {
        for (i, task) in tasks {
            if i < pb.count {
                let pBlock = pb[i]
                let pView = pBlock
                let pSubs = getSubbands16(view: pView)
                let view = blocks[i]
                let subs = getSubbands32(view: view)
                encodeSubbands32WithParent(task: task, encoder: &encoder, subs: subs, parentHL: pSubs.hl, parentLH: pSubs.lh, parentHH: pSubs.hh)
            } else {
                let view = blocks[i]
                let subs = getSubbands32(view: view)
                encodeSubbands32WithoutParent(task: task, encoder: &encoder, subs: subs)
            }
        }
    } else {
        for (i, task) in tasks {
            let view = blocks[i]
            let subs = getSubbands32(view: view)
            encodeSubbands32WithoutParent(task: task, encoder: &encoder, subs: subs)
        }
    }
    
    encoder.flush()
    var out = bwFlags.bytes
    out.append(contentsOf: encoder.getData())
    return out
}

enum EncodeTask16 {
    case encode8
    case split4(Bool, Bool, Bool, Bool)
}

@inline(__always)
func encodePlaneSubbands16(blocks: inout [BlockView], zeroThreshold: Int, parentBlocks: [BlockView]?, sads: [Int]? = nil, colCount: Int = 0, rowCount: Int = 0) -> [UInt8] {
    var bwFlags = BypassWriter()
    var tasks: [(Int, EncodeTask16)] = []
    tasks.reserveCapacity(blocks.count)
    
    let useSpatialWeight = 1 < colCount && 1 < rowCount
    
    var zeroCount = 0
    for i in blocks.indices {
        let safeCol = if 0 < colCount { colCount } else { 1 }
        let col = i % safeCol
        let row = i / safeCol
        let colCount32 = (colCount + 1) / 2
        let sadIdx = ((row / 2) * colCount32) + (col / 2)
        
        let isHighError = if let sads = sads, sadIdx < sads.count, 1500 <= sads[sadIdx] { true } else { false }
        let blockThreshold: Int
        switch true {
        case isHighError:
            // Adaptive AC Preservation (緩和版): if the prediction error is significant,
            // half the zero thresholds to preserve edge details and suppress ghosts,
            // while still discarding the ±1 mosquito noise.
            blockThreshold = max(1, zeroThreshold / 2)
        case useSpatialWeight:
            let weight = spatialWeight(blockCol: col, blockRow: row, colCount: colCount, rowCount: rowCount)
            blockThreshold = (max(1, zeroThreshold) * weight) / 1024
        default:
            blockThreshold = zeroThreshold
        }
        if isEffectivelyZero16(data: blocks[i].base, threshold: blockThreshold) {
            bwFlags.writeBit(true)
            let view = blocks[i]
            let half = 16 / 2
            let base = view.base
            let hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 16)
            let lhView = BlockView(base: base.advanced(by: half * 16), width: half, height: half, stride: 16)
            let hhView = BlockView(base: base.advanced(by: half * 16 + half), width: half, height: half, stride: 16)
            hlView.clearAll()
            lhView.clearAll()
            hhView.clearAll()
            zeroCount += 1
        } else {
            bwFlags.writeBit(false)
            let forceSplit = shouldSplit16(data: blocks[i].base)
            if forceSplit {
                bwFlags.writeBit(true)
                
                bwFlags.writeBit(false)
                bwFlags.writeBit(false)
                
                bwFlags.writeBit(false)
                bwFlags.writeBit(false)
                
                bwFlags.writeBit(false)
                bwFlags.writeBit(false)
                
                bwFlags.writeBit(false)
                bwFlags.writeBit(false)
                
                tasks.append((i, .split4(true, true, true, true)))
            } else {
                bwFlags.writeBit(false)
                tasks.append((i, .encode8))
            }
        }
    }
    bwFlags.flush()
    debugLog({
        let zeroPermyriad = (zeroCount * 10000) / max(1, blocks.count)
        let rateStr = "\(zeroPermyriad / 100).\(zeroPermyriad / 10 % 10)"
        return "    [Subbands] blocks=\(blocks.count) zeroBlocks=\(zeroCount) zeroRate=\(rateStr)%"
    }())
    
    var encoder = EntropyEncoder<DynamicEntropyModel>()
    
    if let pb = parentBlocks {
        for (i, task) in tasks {
            if i < pb.count {
                let pBlock = pb[i]
                let pView = pBlock
                let pSubs = getSubbands8(view: pView)
                let view = blocks[i]
                let subs = getSubbands16(view: view)
                encodeSubbands16WithParent(task: task, encoder: &encoder, subs: subs, parentHL: pSubs.hl, parentLH: pSubs.lh, parentHH: pSubs.hh)
            } else {
                let view = blocks[i]
                let subs = getSubbands16(view: view)
                encodeSubbands16WithoutParent(task: task, encoder: &encoder, subs: subs)
            }
        }
    } else {
        for (i, task) in tasks {
            let view = blocks[i]
            let subs = getSubbands16(view: view)
            encodeSubbands16WithoutParent(task: task, encoder: &encoder, subs: subs)
        }
    }
    
    encoder.flush()
    var out = bwFlags.bytes
    out.append(contentsOf: encoder.getData())
    return out
}

@inline(__always)
func encodePlaneBaseSubbands8(blocks: inout [BlockView], zeroThreshold: Int) -> [UInt8] {
    var bwFlags = BypassWriter()
    var nonZeroIndices: [Int] = []
    
    for i in blocks.indices {
        let isZero = isEffectivelyZeroBase4(data: blocks[i].base, threshold: zeroThreshold)
        if isZero {
            bwFlags.writeBit(true)
            bwFlags.writeBit(false)
            blocks[i].clearAll()
        } else {
            bwFlags.writeBit(false)
            bwFlags.writeBit(false)
            nonZeroIndices.append(i)
        }
    }
    bwFlags.flush()
    let zeroCount = blocks.count - nonZeroIndices.count
    debugLog({
        let zeroPermyriad = (zeroCount * 10000) / max(1, blocks.count)
        let rateStr = "\(zeroPermyriad / 100).\(zeroPermyriad / 10 % 10)"
        return "    [BaseSubbands] blocks=\(blocks.count) zeroBlocks=\(zeroCount) zeroRate=\(rateStr)%"
    }())
    
    var encoder = EntropyEncoder<DynamicEntropyModel>()
    var lastVal: Int16 = 0
    
    var nzCur = 0
    let nzCount = nonZeroIndices.count
    for i in blocks.indices {
        if nzCur < nzCount && nonZeroIndices[nzCur] == i {
            nzCur += 1

            let view = blocks[i]
            let subs = getSubbands8(view: view)
            blockEncodeDPCM4(encoder: &encoder, block: subs.ll, lastVal: &lastVal)
            blockEncode4V(encoder: &encoder, block: subs.hl, parentBlock: nil)
            blockEncode4H(encoder: &encoder, block: subs.lh, parentBlock: nil)
            blockEncode4H(encoder: &encoder, block: subs.hh, parentBlock: nil)
        } else {
            lastVal = 0
        }
    }
    
    encoder.flush()
    var out = bwFlags.bytes
    out.append(contentsOf: encoder.getData())
    return out
}

@inline(__always)
func encodePlaneBaseSubbands8PFrame(blocks: inout [BlockView], zeroThreshold: Int) -> [UInt8] {
    var bwFlags = BypassWriter()
    var nonZeroIndices: [Int] = []
    
    for i in blocks.indices {
        let isZero = isEffectivelyZeroBase4PFrame(data: blocks[i].base, threshold: zeroThreshold)
        if isZero {
            bwFlags.writeBit(true)
            bwFlags.writeBit(false)
            blocks[i].clearAll()
        } else {
            bwFlags.writeBit(false)
            bwFlags.writeBit(false)
            nonZeroIndices.append(i)
        }
    }
    bwFlags.flush()
    let zeroCount = blocks.count - nonZeroIndices.count
    debugLog({
        let zeroPermyriad = (zeroCount * 10000) / max(1, blocks.count)
        let rateStr = "\(zeroPermyriad / 100).\(zeroPermyriad / 10 % 10)"
        return "    [BaseSubbands] blocks=\(blocks.count) zeroBlocks=\(zeroCount) zeroRate=\(rateStr)%"
    }())
    
    var encoder = EntropyEncoder<DynamicEntropyModel>()
    
    var nzCur = 0
    let nzCount = nonZeroIndices.count
    for i in blocks.indices {
        if nzCur < nzCount && nonZeroIndices[nzCur] == i {
            nzCur += 1

            let view = blocks[i]
            let subs = getSubbands8(view: view)
            blockEncode4H(encoder: &encoder, block: subs.ll, parentBlock: nil)
            blockEncode4V(encoder: &encoder, block: subs.hl, parentBlock: nil)
            blockEncode4H(encoder: &encoder, block: subs.lh, parentBlock: nil)
            blockEncode4H(encoder: &encoder, block: subs.hh, parentBlock: nil)
        }
    }
    
    encoder.flush()
    var out = bwFlags.bytes
    out.append(contentsOf: encoder.getData())
    return out
}

enum EncodeTaskBase32 {
    case skip
    case encode16
    case split8(Bool, Bool, Bool, Bool)
}

@inline(__always)
func encodePlaneBaseSubbands32(blocks: inout [BlockView], zeroThreshold: Int) -> [UInt8] {
    var bwFlags = BypassWriter()
    var tasks: [(Int, EncodeTaskBase32)] = []
    tasks.reserveCapacity(blocks.count)
    
    var zeroCount = 0
    for i in blocks.indices {
        let isZero = isEffectivelyZeroBase32(data: blocks[i].base, threshold: zeroThreshold)
        if isZero {
            bwFlags.writeBit(true)
            blocks[i].clearAll()
            tasks.append((i, .skip))
            zeroCount += 1
        } else {
            bwFlags.writeBit(false)
            let forceSplit = shouldSplit32WithLL(data: blocks[i].base)
            if forceSplit {
                bwFlags.writeBit(true)
                
                bwFlags.writeBit(false) // TL
                bwFlags.writeBit(false)
                
                bwFlags.writeBit(false) // TR
                bwFlags.writeBit(false)
                
                bwFlags.writeBit(false) // BL
                bwFlags.writeBit(false)
                
                bwFlags.writeBit(false) // BR
                bwFlags.writeBit(false)
                
                tasks.append((i, .split8(true, true, true, true)))
            } else {
                bwFlags.writeBit(false)
                tasks.append((i, .encode16))
            }
        }
    }
    bwFlags.flush()
    debugLog({
        let zeroPermyriad32 = (zeroCount * 10000) / max(1, blocks.count)
        return "    [BaseSubbands32] blocks=\(blocks.count) zeroBlocks=\(zeroCount) zeroRate=\(zeroPermyriad32 / 100).\(zeroPermyriad32 / 10 % 10)%"
    }())
    
    var encoder = EntropyEncoder<StaticDPCMEntropyModel>()
    var lastVal: Int16 = 0
    
    for (i, task) in tasks {
        switch task {
        case .skip:
            lastVal = 0
            
        case .encode16:
            let view = blocks[i]
            let subs = getSubbands32(view: view)
            blockEncodeDPCM16(encoder: &encoder, block: subs.ll, lastVal: &lastVal)
            blockEncode16V(encoder: &encoder, block: subs.hl, parentBlock: nil)
            blockEncode16H(encoder: &encoder, block: subs.lh, parentBlock: nil)
            blockEncode16H(encoder: &encoder, block: subs.hh, parentBlock: nil)
        
        case .split8(let tl, let tr, let bl, let br):
            let view = blocks[i]
            let subs = getSubbands32(view: view)
            if tl {
                let ll = BlockView(base: subs.ll.base, width: 8, height: 8, stride: 32)
                let hl = BlockView(base: subs.hl.base, width: 8, height: 8, stride: 32)
                let lh = BlockView(base: subs.lh.base, width: 8, height: 8, stride: 32)
                let hh = BlockView(base: subs.hh.base, width: 8, height: 8, stride: 32)
                blockEncodeDPCM8(encoder: &encoder, block: ll, lastVal: &lastVal)
                blockEncode8H(encoder: &encoder, block: hl, parentBlock: nil)
                blockEncode8H(encoder: &encoder, block: lh, parentBlock: nil)
                blockEncode8H(encoder: &encoder, block: hh, parentBlock: nil)
            }
            if tr {
                let ll = BlockView(base: subs.ll.base.advanced(by: 8), width: 8, height: 8, stride: 32)
                let hl = BlockView(base: subs.hl.base.advanced(by: 8), width: 8, height: 8, stride: 32)
                let lh = BlockView(base: subs.lh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
                let hh = BlockView(base: subs.hh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
                blockEncodeDPCM8(encoder: &encoder, block: ll, lastVal: &lastVal)
                blockEncode8H(encoder: &encoder, block: hl, parentBlock: nil)
                blockEncode8H(encoder: &encoder, block: lh, parentBlock: nil)
                blockEncode8H(encoder: &encoder, block: hh, parentBlock: nil)
            }
            if bl {
                let ll = BlockView(base: subs.ll.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                let hl = BlockView(base: subs.hl.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                let lh = BlockView(base: subs.lh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                let hh = BlockView(base: subs.hh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                blockEncodeDPCM8(encoder: &encoder, block: ll, lastVal: &lastVal)
                blockEncode8H(encoder: &encoder, block: hl, parentBlock: nil)
                blockEncode8H(encoder: &encoder, block: lh, parentBlock: nil)
                blockEncode8H(encoder: &encoder, block: hh, parentBlock: nil)
            }
            if br {
                let ll = BlockView(base: subs.ll.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                let hl = BlockView(base: subs.hl.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                let lh = BlockView(base: subs.lh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                let hh = BlockView(base: subs.hh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                blockEncodeDPCM8(encoder: &encoder, block: ll, lastVal: &lastVal)
                blockEncode8H(encoder: &encoder, block: hl, parentBlock: nil)
                blockEncode8H(encoder: &encoder, block: lh, parentBlock: nil)
                blockEncode8H(encoder: &encoder, block: hh, parentBlock: nil)
            }
        }
    }
    
    encoder.flush()
    var out = bwFlags.bytes
    out.append(contentsOf: encoder.getData())
    return out
}

// MARK: - Dedicated Subband Process Functions

@inline(__always)
func encodeSubbands32WithParent<M: EntropyModelProvider>(
    task: EncodeTask32,
    encoder: inout EntropyEncoder<M>,
    subs: Subbands,
    parentHL: BlockView,
    parentLH: BlockView,
    parentHH: BlockView
) {
    switch task {
    case .encode16:
        blockEncode16V(encoder: &encoder, block: subs.hl, parentBlock: parentHL)
        blockEncode16H(encoder: &encoder, block: subs.lh, parentBlock: parentLH)
        blockEncode16H(encoder: &encoder, block: subs.hh, parentBlock: parentHH)
    case .split8(let tl, let tr, let bl, let br):
        if tl {
            let pbHL = BlockView(base: parentHL.base, width: 4, height: 4, stride: parentHL.stride)
            let pbLH = BlockView(base: parentLH.base, width: 4, height: 4, stride: parentLH.stride)
            let pbHH = BlockView(base: parentHH.base, width: 4, height: 4, stride: parentHH.stride)
            let hl = BlockView(base: subs.hl.base, width: 8, height: 8, stride: 32)
            let lh = BlockView(base: subs.lh.base, width: 8, height: 8, stride: 32)
            let hh = BlockView(base: subs.hh.base, width: 8, height: 8, stride: 32)
            blockEncode8H(encoder: &encoder, block: hl, parentBlock: pbHL)
            blockEncode8H(encoder: &encoder, block: lh, parentBlock: pbLH)
            blockEncode8H(encoder: &encoder, block: hh, parentBlock: pbHH)
        } else {
            BlockView(base: subs.hl.base, width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.lh.base, width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.hh.base, width: 8, height: 8, stride: 32).clearAll()
        }
        if tr {
            let pbHL = BlockView(base: parentHL.base.advanced(by: 4), width: 4, height: 4, stride: parentHL.stride)
            let pbLH = BlockView(base: parentLH.base.advanced(by: 4), width: 4, height: 4, stride: parentLH.stride)
            let pbHH = BlockView(base: parentHH.base.advanced(by: 4), width: 4, height: 4, stride: parentHH.stride)
            let hl = BlockView(base: subs.hl.base.advanced(by: 8), width: 8, height: 8, stride: 32)
            let lh = BlockView(base: subs.lh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
            let hh = BlockView(base: subs.hh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
            blockEncode8H(encoder: &encoder, block: hl, parentBlock: pbHL)
            blockEncode8H(encoder: &encoder, block: lh, parentBlock: pbLH)
            blockEncode8H(encoder: &encoder, block: hh, parentBlock: pbHH)
        } else {
            BlockView(base: subs.hl.base.advanced(by: 8), width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.lh.base.advanced(by: 8), width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.hh.base.advanced(by: 8), width: 8, height: 8, stride: 32).clearAll()
        }
        if bl {
            let pbHL = BlockView(base: parentHL.base.advanced(by: 4 * parentHL.stride), width: 4, height: 4, stride: parentHL.stride)
            let pbLH = BlockView(base: parentLH.base.advanced(by: 4 * parentLH.stride), width: 4, height: 4, stride: parentLH.stride)
            let pbHH = BlockView(base: parentHH.base.advanced(by: 4 * parentHH.stride), width: 4, height: 4, stride: parentHH.stride)
            let hl = BlockView(base: subs.hl.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
            let lh = BlockView(base: subs.lh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
            let hh = BlockView(base: subs.hh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
            blockEncode8H(encoder: &encoder, block: hl, parentBlock: pbHL)
            blockEncode8H(encoder: &encoder, block: lh, parentBlock: pbLH)
            blockEncode8H(encoder: &encoder, block: hh, parentBlock: pbHH)
        } else {
            BlockView(base: subs.hl.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.lh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.hh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32).clearAll()
        }
        if br {
            let pbHL = BlockView(base: parentHL.base.advanced(by: 4 * parentHL.stride + 4), width: 4, height: 4, stride: parentHL.stride)
            let pbLH = BlockView(base: parentLH.base.advanced(by: 4 * parentLH.stride + 4), width: 4, height: 4, stride: parentLH.stride)
            let pbHH = BlockView(base: parentHH.base.advanced(by: 4 * parentHH.stride + 4), width: 4, height: 4, stride: parentHH.stride)
            let hl = BlockView(base: subs.hl.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
            let lh = BlockView(base: subs.lh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
            let hh = BlockView(base: subs.hh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
            blockEncode8H(encoder: &encoder, block: hl, parentBlock: pbHL)
            blockEncode8H(encoder: &encoder, block: lh, parentBlock: pbLH)
            blockEncode8H(encoder: &encoder, block: hh, parentBlock: pbHH)
        } else {
            BlockView(base: subs.hl.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.lh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.hh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32).clearAll()
        }
    }
}

@inline(__always)
func encodeSubbands32WithoutParent<M: EntropyModelProvider>(
    task: EncodeTask32,
    encoder: inout EntropyEncoder<M>,
    subs: Subbands
) {
    switch task {
    case .encode16:
        blockEncode16V(encoder: &encoder, block: subs.hl, parentBlock: nil)
        blockEncode16H(encoder: &encoder, block: subs.lh, parentBlock: nil)
        blockEncode16H(encoder: &encoder, block: subs.hh, parentBlock: nil)
    case .split8(let tl, let tr, let bl, let br):
        if tl {
            let hl = BlockView(base: subs.hl.base, width: 8, height: 8, stride: 32)
            let lh = BlockView(base: subs.lh.base, width: 8, height: 8, stride: 32)
            let hh = BlockView(base: subs.hh.base, width: 8, height: 8, stride: 32)
            blockEncode8H(encoder: &encoder, block: hl, parentBlock: nil)
            blockEncode8H(encoder: &encoder, block: lh, parentBlock: nil)
            blockEncode8H(encoder: &encoder, block: hh, parentBlock: nil)
        } else {
            BlockView(base: subs.hl.base, width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.lh.base, width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.hh.base, width: 8, height: 8, stride: 32).clearAll()
        }
        if tr {
            let hl = BlockView(base: subs.hl.base.advanced(by: 8), width: 8, height: 8, stride: 32)
            let lh = BlockView(base: subs.lh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
            let hh = BlockView(base: subs.hh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
            blockEncode8H(encoder: &encoder, block: hl, parentBlock: nil)
            blockEncode8H(encoder: &encoder, block: lh, parentBlock: nil)
            blockEncode8H(encoder: &encoder, block: hh, parentBlock: nil)
        } else {
            BlockView(base: subs.hl.base.advanced(by: 8), width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.lh.base.advanced(by: 8), width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.hh.base.advanced(by: 8), width: 8, height: 8, stride: 32).clearAll()
        }
        if bl {
            let hl = BlockView(base: subs.hl.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
            let lh = BlockView(base: subs.lh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
            let hh = BlockView(base: subs.hh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
            blockEncode8H(encoder: &encoder, block: hl, parentBlock: nil)
            blockEncode8H(encoder: &encoder, block: lh, parentBlock: nil)
            blockEncode8H(encoder: &encoder, block: hh, parentBlock: nil)
        } else {
            BlockView(base: subs.hl.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.lh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.hh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32).clearAll()
        }
        if br {
            let hl = BlockView(base: subs.hl.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
            let lh = BlockView(base: subs.lh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
            let hh = BlockView(base: subs.hh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
            blockEncode8H(encoder: &encoder, block: hl, parentBlock: nil)
            blockEncode8H(encoder: &encoder, block: lh, parentBlock: nil)
            blockEncode8H(encoder: &encoder, block: hh, parentBlock: nil)
        } else {
            BlockView(base: subs.hl.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.lh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32).clearAll()
            BlockView(base: subs.hh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32).clearAll()
        }
    }
}

@inline(__always)
func encodeSubbands16WithParent<M: EntropyModelProvider>(
    task: EncodeTask16,
    encoder: inout EntropyEncoder<M>,
    subs: Subbands,
    parentHL: BlockView,
    parentLH: BlockView,
    parentHH: BlockView
) {
    switch task {
    case .encode8:
        blockEncode8V(encoder: &encoder, block: subs.hl, parentBlock: parentHL)
        blockEncode8H(encoder: &encoder, block: subs.lh, parentBlock: parentLH)
        blockEncode8H(encoder: &encoder, block: subs.hh, parentBlock: parentHH)
    case .split4(let tl, let tr, let bl, let br):
        if tl {
            let pbHL = BlockView(base: parentHL.base, width: 2, height: 2, stride: parentHL.stride)
            let pbLH = BlockView(base: parentLH.base, width: 2, height: 2, stride: parentLH.stride)
            let pbHH = BlockView(base: parentHH.base, width: 2, height: 2, stride: parentHH.stride)
            let hl = BlockView(base: subs.hl.base, width: 4, height: 4, stride: 16)
            let lh = BlockView(base: subs.lh.base, width: 4, height: 4, stride: 16)
            let hh = BlockView(base: subs.hh.base, width: 4, height: 4, stride: 16)
            blockEncode4H(encoder: &encoder, block: hl, parentBlock: pbHL)
            blockEncode4H(encoder: &encoder, block: lh, parentBlock: pbLH)
            blockEncode4H(encoder: &encoder, block: hh, parentBlock: pbHH)
        }
        if tr {
            let pbHL = BlockView(base: parentHL.base.advanced(by: 2), width: 2, height: 2, stride: parentHL.stride)
            let pbLH = BlockView(base: parentLH.base.advanced(by: 2), width: 2, height: 2, stride: parentLH.stride)
            let pbHH = BlockView(base: parentHH.base.advanced(by: 2), width: 2, height: 2, stride: parentHH.stride)
            let hl = BlockView(base: subs.hl.base.advanced(by: 4), width: 4, height: 4, stride: 16)
            let lh = BlockView(base: subs.lh.base.advanced(by: 4), width: 4, height: 4, stride: 16)
            let hh = BlockView(base: subs.hh.base.advanced(by: 4), width: 4, height: 4, stride: 16)
            blockEncode4H(encoder: &encoder, block: hl, parentBlock: pbHL)
            blockEncode4H(encoder: &encoder, block: lh, parentBlock: pbLH)
            blockEncode4H(encoder: &encoder, block: hh, parentBlock: pbHH)
        }
        if bl {
            let pbHL = BlockView(base: parentHL.base.advanced(by: 2 * parentHL.stride), width: 2, height: 2, stride: parentHL.stride)
            let pbLH = BlockView(base: parentLH.base.advanced(by: 2 * parentLH.stride), width: 2, height: 2, stride: parentLH.stride)
            let pbHH = BlockView(base: parentHH.base.advanced(by: 2 * parentHH.stride), width: 2, height: 2, stride: parentHH.stride)
            let hl = BlockView(base: subs.hl.base.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
            let lh = BlockView(base: subs.lh.base.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
            let hh = BlockView(base: subs.hh.base.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
            blockEncode4H(encoder: &encoder, block: hl, parentBlock: pbHL)
            blockEncode4H(encoder: &encoder, block: lh, parentBlock: pbLH)
            blockEncode4H(encoder: &encoder, block: hh, parentBlock: pbHH)
        }
        if br {
            let pbHL = BlockView(base: parentHL.base.advanced(by: 2 * parentHL.stride + 2), width: 2, height: 2, stride: parentHL.stride)
            let pbLH = BlockView(base: parentLH.base.advanced(by: 2 * parentLH.stride + 2), width: 2, height: 2, stride: parentLH.stride)
            let pbHH = BlockView(base: parentHH.base.advanced(by: 2 * parentHH.stride + 2), width: 2, height: 2, stride: parentHH.stride)
            let hl = BlockView(base: subs.hl.base.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
            let lh = BlockView(base: subs.lh.base.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
            let hh = BlockView(base: subs.hh.base.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
            blockEncode4H(encoder: &encoder, block: hl, parentBlock: pbHL)
            blockEncode4H(encoder: &encoder, block: lh, parentBlock: pbLH)
            blockEncode4H(encoder: &encoder, block: hh, parentBlock: pbHH)
        }
    }
}

@inline(__always)
func encodeSubbands16WithoutParent<M: EntropyModelProvider>(
    task: EncodeTask16,
    encoder: inout EntropyEncoder<M>,
    subs: Subbands
) {
    switch task {
    case .encode8:
        blockEncode8V(encoder: &encoder, block: subs.hl, parentBlock: nil)
        blockEncode8H(encoder: &encoder, block: subs.lh, parentBlock: nil)
        blockEncode8H(encoder: &encoder, block: subs.hh, parentBlock: nil)
    case .split4(let tl, let tr, let bl, let br):
        if tl {
            let hl = BlockView(base: subs.hl.base, width: 4, height: 4, stride: 16)
            let lh = BlockView(base: subs.lh.base, width: 4, height: 4, stride: 16)
            let hh = BlockView(base: subs.hh.base, width: 4, height: 4, stride: 16)
            blockEncode4H(encoder: &encoder, block: hl, parentBlock: nil)
            blockEncode4H(encoder: &encoder, block: lh, parentBlock: nil)
            blockEncode4H(encoder: &encoder, block: hh, parentBlock: nil)
        }
        if tr {
            let hl = BlockView(base: subs.hl.base.advanced(by: 4), width: 4, height: 4, stride: 16)
            let lh = BlockView(base: subs.lh.base.advanced(by: 4), width: 4, height: 4, stride: 16)
            let hh = BlockView(base: subs.hh.base.advanced(by: 4), width: 4, height: 4, stride: 16)
            blockEncode4H(encoder: &encoder, block: hl, parentBlock: nil)
            blockEncode4H(encoder: &encoder, block: lh, parentBlock: nil)
            blockEncode4H(encoder: &encoder, block: hh, parentBlock: nil)
        }
        if bl {
            let hl = BlockView(base: subs.hl.base.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
            let lh = BlockView(base: subs.lh.base.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
            let hh = BlockView(base: subs.hh.base.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
            blockEncode4H(encoder: &encoder, block: hl, parentBlock: nil)
            blockEncode4H(encoder: &encoder, block: lh, parentBlock: nil)
            blockEncode4H(encoder: &encoder, block: hh, parentBlock: nil)
        }
        if br {
            let hl = BlockView(base: subs.hl.base.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
            let lh = BlockView(base: subs.lh.base.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
            let hh = BlockView(base: subs.hh.base.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
            blockEncode4H(encoder: &encoder, block: hl, parentBlock: nil)
            blockEncode4H(encoder: &encoder, block: lh, parentBlock: nil)
            blockEncode4H(encoder: &encoder, block: hh, parentBlock: nil)
        }
    }
}