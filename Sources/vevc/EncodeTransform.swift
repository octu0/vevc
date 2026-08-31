import Foundation

// MARK: - Transform Functions

@inline(__always)
func checkQuadrants16x16(base: UnsafeMutablePointer<Int16>, stride: Int, q0: inout Bool, q1: inout Bool, q2: inout Bool, q3: inout Bool) {
    let zero8 = SIMD8<Int16>(repeating: 0)
    for y in 0..<8 {
        let ptr = base.advanced(by: y * stride)
        if q0 != true {
            let v = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
            if any(v .!= zero8) {
                q0 = true
            }
        }
        if q1 != true {
            let v = UnsafeRawPointer(ptr.advanced(by: 8)).loadUnaligned(as: SIMD8<Int16>.self)
            if any(v .!= zero8) {
                q1 = true
            }
        }
    }
    for y in 8..<16 {
        let ptr = base.advanced(by: y * stride)
        if q2 != true {
            let v = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
            if any(v .!= zero8) {
                q2 = true
            }
        }
        if q3 != true {
            let v = UnsafeRawPointer(ptr.advanced(by: 8)).loadUnaligned(as: SIMD8<Int16>.self)
            if any(v .!= zero8) {
                q3 = true
            }
        }
    }
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
            if any(v .!= zero4) {
                q0 = true
            }
        }
        if q1 != true {
            let v = UnsafeRawPointer(ptr.advanced(by: 4)).loadUnaligned(as: SIMD4<Int16>.self)
            if any(v .!= zero4) {
                q1 = true
            }
        }
    }
    for y in 4..<8 {
        let ptr = base.advanced(by: y * stride)
        if q2 != true {
            let v = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD4<Int16>.self)
            if any(v .!= zero4) {
                q2 = true
            }
        }
        if q3 != true {
            let v = UnsafeRawPointer(ptr.advanced(by: 4)).loadUnaligned(as: SIMD4<Int16>.self)
            if any(v .!= zero4) {
                q3 = true
            }
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

enum EncodeTask32 {
    case encode16
    case split8(Bool, Bool, Bool, Bool)
}

@inline(__always)
func encodePlaneSubbands32(blocks: inout [BlockView], zeroThreshold: Int, parentBlocks: [BlockView]?, colCount: Int, rowCount: Int, history: EntropyHistoryState?, selectModel: ModelSelectorFn, updateHistory: Bool = true) -> [UInt8] {
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
            blockThreshold = if zeroThreshold == 0 { 0 } else { (zeroThreshold * weight) / 1024 }
        } else {
            blockThreshold = zeroThreshold
        }
        let isZero = isEffectivelyZero32(data: blocks[i].base, threshold: blockThreshold)
        if isZero {
            bwFlags.writeBit(true)
            let view = blocks[i]
            let half = 32 / 2
            let base = view.base
            clearBlockRegion(base: base.advanced(by: half), width: half, height: half, stride: 32)
            clearBlockRegion(base: base.advanced(by: half * 32), width: half, height: half, stride: 32)
            clearBlockRegion(base: base.advanced(by: half * 32 + half), width: half, height: half, stride: 32)
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
    
    var encoder = EntropyEncoder()
    var lastVal: Int16 = 0
    
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
                encodeSubbands32WithoutParent(task: task, encoder: &encoder, subs: subs, lastVal: &lastVal)
            }
        }
        encoder.flush()
        var out = bwFlags.bytes
        out.append(contentsOf: encoder.getData(selectModel: selectModel, history: history, updateHistory: updateHistory))
        return out
    }
    
    for (i, task) in tasks {
        let view = blocks[i]
        let subs = getSubbands32(view: view)
        encodeSubbands32WithoutParent(task: task, encoder: &encoder, subs: subs, lastVal: &lastVal)
    }
    encoder.flush()
    var out = bwFlags.bytes
    out.append(contentsOf: encoder.getData(selectModel: selectModel, history: history, updateHistory: updateHistory))
    return out
}

@inline(__always)
func encodePlaneSubbands32WithSkipMap(blocks: inout [BlockView], zeroThreshold: Int, parentBlocks: [BlockView]?, colCount: Int, rowCount: Int, isSkip: [Bool], isTreez: [Bool]? = nil, history: EntropyHistoryState?, selectModel: ModelSelectorFn, updateHistory: Bool = true) -> ([UInt8], [Bool]) {
    var bwFlags = BypassWriter()
    var tasks: [(Int, EncodeTask32)] = []
    tasks.reserveCapacity(blocks.count)
    
    let useSpatialWeight = 1 < colCount && 1 < rowCount
    
    var zeroCount = 0
    var isZeroFlags = [Bool](repeating: true, count: blocks.count)
    for i in blocks.indices {
        if isSkip[i] {
            let view = blocks[i]
            let half = 32 / 2
            let base = view.base
            clearBlockRegion(base: base.advanced(by: half), width: half, height: half, stride: 32)
            clearBlockRegion(base: base.advanced(by: half * 32), width: half, height: half, stride: 32)
            clearBlockRegion(base: base.advanced(by: half * 32 + half), width: half, height: half, stride: 32)
            zeroCount += 1
            isZeroFlags[i] = true
            continue
        }
        if let tz = isTreez, tz[i] {
            let view = blocks[i]
            let half = 32 / 2
            let base = view.base
            clearBlockRegion(base: base.advanced(by: half), width: half, height: half, stride: 32)
            clearBlockRegion(base: base.advanced(by: half * 32), width: half, height: half, stride: 32)
            clearBlockRegion(base: base.advanced(by: half * 32 + half), width: half, height: half, stride: 32)
            zeroCount += 1
            isZeroFlags[i] = true
            continue
        }
        let blockThreshold: Int
        if useSpatialWeight {
            let col = i % colCount
            let row = i / colCount
            let weight = spatialWeight(blockCol: col, blockRow: row, colCount: colCount, rowCount: rowCount)
            if zeroThreshold == 0 {
                blockThreshold = 0
            } else {
                blockThreshold = (zeroThreshold * weight) / 1024
            }
        } else {
            blockThreshold = zeroThreshold
        }
        let isZero = isEffectivelyZero32(data: blocks[i].base, threshold: blockThreshold)
        isZeroFlags[i] = isZero
        if isZero {
            bwFlags.writeBit(true)
            let view = blocks[i]
            let half = 32 / 2
            let base = view.base
            clearBlockRegion(base: base.advanced(by: half), width: half, height: half, stride: 32)
            clearBlockRegion(base: base.advanced(by: half * 32), width: half, height: half, stride: 32)
            clearBlockRegion(base: base.advanced(by: half * 32 + half), width: half, height: half, stride: 32)
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
    
    var encoder = EntropyEncoder()
    var lastVal: Int16 = 0
    
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
                encodeSubbands32WithoutParent(task: task, encoder: &encoder, subs: subs, lastVal: &lastVal)
            }
        }
        encoder.flush()
        var out = bwFlags.bytes
        out.append(contentsOf: encoder.getData(selectModel: selectModel, history: history, updateHistory: updateHistory))
        return (out, isZeroFlags)
    }
    
    for (i, task) in tasks {
        let view = blocks[i]
        let subs = getSubbands32(view: view)
        encodeSubbands32WithoutParent(task: task, encoder: &encoder, subs: subs, lastVal: &lastVal)
    }
    encoder.flush()
    var out = bwFlags.bytes
    out.append(contentsOf: encoder.getData(selectModel: selectModel, history: history, updateHistory: updateHistory))
    return (out, isZeroFlags)
}

enum EncodeTask16 {
    case encode8
    case split4(Bool, Bool, Bool, Bool)
}

@inline(__always)
func encodePlaneSubbands16(blocks: inout [BlockView], zeroThreshold: Int, parentBlocks: [BlockView]?, colCount: Int, rowCount: Int, history: EntropyHistoryState?, selectModel: ModelSelectorFn, updateHistory: Bool = true) -> [UInt8] {
    var bwFlags = BypassWriter()
    var tasks: [(Int, EncodeTask16)] = []
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
            blockThreshold = if zeroThreshold == 0 { 0 } else { (zeroThreshold * weight) / 1024 }
        } else {
            blockThreshold = zeroThreshold
        }
        if isEffectivelyZero16(data: blocks[i].base, threshold: blockThreshold) {
            bwFlags.writeBit(true)
            let view = blocks[i]
            let half = 16 / 2
            let base = view.base
            clearBlockRegion(base: base.advanced(by: half), width: half, height: half, stride: 16)
            clearBlockRegion(base: base.advanced(by: half * 16), width: half, height: half, stride: 16)
            clearBlockRegion(base: base.advanced(by: half * 16 + half), width: half, height: half, stride: 16)
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
    
    var encoder = EntropyEncoder()
    
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
        encoder.flush()
        var out = bwFlags.bytes
        out.append(contentsOf: encoder.getData(selectModel: selectModel, history: history, updateHistory: updateHistory))
        return out
    }
    
    for (i, task) in tasks {
        let view = blocks[i]
        let subs = getSubbands16(view: view)
        encodeSubbands16WithoutParent(task: task, encoder: &encoder, subs: subs)
    }
    encoder.flush()
    var out = bwFlags.bytes
    out.append(contentsOf: encoder.getData(selectModel: selectModel, history: history, updateHistory: updateHistory))
    return out
}

@inline(__always)
func encodePlaneSubbands16WithSkipMap(blocks: inout [BlockView], zeroThreshold: Int, parentBlocks: [BlockView]?, colCount: Int, rowCount: Int, isSkip: [Bool], isTreez: [Bool]? = nil, history: EntropyHistoryState?, selectModel: ModelSelectorFn, updateHistory: Bool = true) -> ([UInt8], [Bool]) {
    var bwFlags = BypassWriter()
    var tasks: [(Int, EncodeTask16)] = []
    tasks.reserveCapacity(blocks.count)

    let useSpatialWeight = 1 < colCount && 1 < rowCount

    var zeroCount = 0
    var isZeroFlags = [Bool](repeating: true, count: blocks.count)
    for i in blocks.indices {
        if isSkip[i] {
            let view = blocks[i]
            let half = 16 / 2
            let base = view.base
            clearBlockRegion(base: base.advanced(by: half), width: half, height: half, stride: 16)
            clearBlockRegion(base: base.advanced(by: half * 16), width: half, height: half, stride: 16)
            clearBlockRegion(base: base.advanced(by: half * 16 + half), width: half, height: half, stride: 16)
            zeroCount += 1
            isZeroFlags[i] = true
            continue
        }
        if let tz = isTreez, tz[i] {
            let view = blocks[i]
            let half = 16 / 2
            let base = view.base
            clearBlockRegion(base: base.advanced(by: half), width: half, height: half, stride: 16)
            clearBlockRegion(base: base.advanced(by: half * 16), width: half, height: half, stride: 16)
            clearBlockRegion(base: base.advanced(by: half * 16 + half), width: half, height: half, stride: 16)
            zeroCount += 1
            isZeroFlags[i] = true
            continue
        }
        let blockThreshold: Int
        if useSpatialWeight {
            let col = i % colCount
            let row = i / colCount
            let weight = spatialWeight(blockCol: col, blockRow: row, colCount: colCount, rowCount: rowCount)
            if zeroThreshold == 0 {
                blockThreshold = 0
            } else {
                blockThreshold = (zeroThreshold * weight) / 1024
            }
        } else {
            blockThreshold = zeroThreshold
        }
        let isZero = isEffectivelyZero16(data: blocks[i].base, threshold: blockThreshold)
        isZeroFlags[i] = isZero
        if isZero {
            bwFlags.writeBit(true)
            let view = blocks[i]
            let half = 16 / 2
            let base = view.base
            clearBlockRegion(base: base.advanced(by: half), width: half, height: half, stride: 16)
            clearBlockRegion(base: base.advanced(by: half * 16), width: half, height: half, stride: 16)
            clearBlockRegion(base: base.advanced(by: half * 16 + half), width: half, height: half, stride: 16)
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
    
    var encoder = EntropyEncoder()
    
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
        encoder.flush()
        var out = bwFlags.bytes
        out.append(contentsOf: encoder.getData(selectModel: selectModel, history: history, updateHistory: updateHistory))
        return (out, isZeroFlags)
    }
    
    for (i, task) in tasks {
        let view = blocks[i]
        let subs = getSubbands16(view: view)
        encodeSubbands16WithoutParent(task: task, encoder: &encoder, subs: subs)
    }
    encoder.flush()
    var out = bwFlags.bytes
    out.append(contentsOf: encoder.getData(selectModel: selectModel, history: history, updateHistory: updateHistory))
    return (out, isZeroFlags)
}

@inline(__always)
func encodePlaneBaseSubbands8(blocks: inout [BlockView], zeroThreshold: Int, selectModel: ModelSelectorFn = unifiedSelectModel, isProfile2: Bool = false) -> [UInt8] {
    var bwFlags = BypassWriter()
    var nonZeroIndices: [Int] = []
    
    for i in blocks.indices {
        let isZero = isEffectivelyZeroBase4(data: blocks[i].base, threshold: zeroThreshold)
        if isZero {
            bwFlags.writeBit(true)
            bwFlags.writeBit(false)
            let b = blocks[i]
            clearBlockRegion(base: b.base, width: b.width, height: b.height, stride: b.stride)
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
    
    var encoder = EntropyEncoder()
    var lastVal: Int16 = 0
    
    var nzCur = 0
    let nzCount = nonZeroIndices.count
    for i in blocks.indices {
        if nzCur < nzCount && nonZeroIndices[nzCur] == i {
            nzCur += 1

            let view = blocks[i]
            let subs = getSubbands8(view: view)
            blockEncodeDPCM4(encoder: &encoder, block: subs.ll, lastVal: &lastVal)
            blockEncode4V(encoder: &encoder, block: subs.hl)
            blockEncode4H(encoder: &encoder, block: subs.lh)
            blockEncode4H(encoder: &encoder, block: subs.hh)
        } else {
            lastVal = 0
        }
    }
    
    encoder.flush()
    var out = bwFlags.bytes
    out.append(contentsOf: encoder.getData(selectModel: selectModel))
    return out
}

@inline(__always)
func encodePlaneBaseSubbands8PFrame(blocks: inout [BlockView], zeroThreshold: Int, history: EntropyHistoryState? = nil, selectModel: ModelSelectorFn = unifiedSelectModel, updateHistory: Bool = true) -> [UInt8] {
    var bwFlags = BypassWriter()
    var nonZeroIndices: [Int] = []
    
    for i in blocks.indices {
        let isZero = isEffectivelyZeroBase4PFrame(data: blocks[i].base, threshold: zeroThreshold)
        if isZero {
            bwFlags.writeBit(true)
            bwFlags.writeBit(false)
            let b = blocks[i]
            clearBlockRegion(base: b.base, width: b.width, height: b.height, stride: b.stride)
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
    
    var encoder = EntropyEncoder()
    
    var nzCur = 0
    let nzCount = nonZeroIndices.count
    for i in blocks.indices {
        if nzCur < nzCount && nonZeroIndices[nzCur] == i {
            nzCur += 1

            let view = blocks[i]
            let subs = getSubbands8(view: view)
            blockEncode4H(encoder: &encoder, block: subs.ll)
            blockEncode4V(encoder: &encoder, block: subs.hl)
            blockEncode4H(encoder: &encoder, block: subs.lh)
            blockEncode4H(encoder: &encoder, block: subs.hh)
        }
    }
    
    encoder.flush()
    var out = bwFlags.bytes
    out.append(contentsOf: encoder.getData(selectModel: selectModel, history: history, updateHistory: updateHistory))
    return out
}

@inline(__always)
func copyLLCoeffs(from view: BlockView, to dst: UnsafeMutablePointer<Int16>) {
    let base = view.base
    dst[0] = base[0]
    dst[1] = base[1]
    dst[2] = base[2]
    dst[3] = base[3]
    dst[4] = base[8]
    dst[5] = base[9]
    dst[6] = base[10]
    dst[7] = base[11]
    dst[8] = base[16]
    dst[9] = base[17]
    dst[10] = base[18]
    dst[11] = base[19]
    dst[12] = base[24]
    dst[13] = base[25]
    dst[14] = base[26]
    dst[15] = base[27]
}

@inline(__always)
func encodePlaneBaseSubbands8PFrameWithSkipMap(
    blocks: inout [BlockView],
    colCount: Int,
    qstep: Int32,
    zeroThreshold: Int,
    isSkip: [Bool],
    isTreez: [Bool]? = nil,
    isLuma: Bool = true,
    history: EntropyHistoryState?,
    selectModel: ModelSelectorFn,
    updateHistory: Bool = true,
    // Owned by the encoder (one instance per LayersEncodeActor) and reused for
    // every frame. nil for planes that can never take the model path (chroma),
    // which is what keeps the non-model path allocation-free.
    workspace ws: rANSContextWorkspace?
) -> (data: [UInt8], isZeroFlags: [Bool], hasRANSContext: Bool) {
    var nonZeroIndices: [Int] = []
    nonZeroIndices.reserveCapacity(blocks.count)
    var isZeroFlags = [Bool](repeating: true, count: blocks.count)

    for i in blocks.indices {
        if isSkip[i] {
            let b = blocks[i]
            clearBlockRegion(base: b.base, width: b.width, height: b.height, stride: b.stride)
            isZeroFlags[i] = true
            continue
        }
        if let tz = isTreez {
            if i < tz.count {
                if tz[i] {
                    let b = blocks[i]
                    clearBlockRegion(base: b.base, width: b.width, height: b.height, stride: b.stride)
                    isZeroFlags[i] = true
                    continue
                }
            }
        }
        let isZero = isEffectivelyZeroBase4PFrame(data: blocks[i].base, threshold: zeroThreshold)
        isZeroFlags[i] = isZero
        if isZero {
            let b = blocks[i]
            clearBlockRegion(base: b.base, width: b.width, height: b.height, stride: b.stride)
        } else {
            nonZeroIndices.append(i)
        }
    }

    let zeroCount = blocks.count - nonZeroIndices.count
    debugLog({
        let zeroPermyriad = (zeroCount * 10000) / max(1, blocks.count)
        let rateStr = "\(zeroPermyriad / 100).\(zeroPermyriad / 10 % 10)"
        return "    [BaseSubbands] blocks=\(blocks.count) zeroBlocks=\(zeroCount) zeroRate=\(rateStr)%"
    }())

    let canUseModel = isLuma && (nonZeroIndices.isEmpty != true) && (ws != nil)

    guard canUseModel, let ws = ws else {
        var bwFlags = BypassWriter()
        for i in blocks.indices {
            if isSkip[i] { continue }
            if let tz = isTreez {
                if i < tz.count {
                    if tz[i] { continue }
                }
            }
            let isZ = isZeroFlags[i]
            bwFlags.writeBit(isZ)
        }
        bwFlags.flush()

        var encoder = EntropyEncoder()
        var nzCur = 0
        let nzCount = nonZeroIndices.count
        for i in blocks.indices {
            if nzCur < nzCount && nonZeroIndices[nzCur] == i {
                nzCur += 1
                let view = blocks[i]
                let subs = getSubbands8(view: view)
                blockEncode4H(encoder: &encoder, block: subs.ll)
                blockEncode4V(encoder: &encoder, block: subs.hl)
                blockEncode4H(encoder: &encoder, block: subs.lh)
                blockEncode4H(encoder: &encoder, block: subs.hh)
            }
        }
        encoder.flush()

        var out = bwFlags.bytes
        out.append(contentsOf: encoder.getData(selectModel: selectModel, history: history, updateHistory: updateHistory))
        return (out, isZeroFlags, false)
    }

    // 1. Candidate A: Base Encoding
    var encoderBase = EntropyEncoder()
    var nzCurBase = 0
    let nzCountBase = nonZeroIndices.count
    for i in blocks.indices {
        if nzCurBase < nzCountBase && nonZeroIndices[nzCurBase] == i {
            nzCurBase += 1
            let view = blocks[i]
            let subs = getSubbands8(view: view)
            blockEncode4H(encoder: &encoderBase, block: subs.ll)
            blockEncode4V(encoder: &encoderBase, block: subs.hl)
            blockEncode4H(encoder: &encoderBase, block: subs.lh)
            blockEncode4H(encoder: &encoderBase, block: subs.hh)
        }
    }
    encoderBase.flush()

    var bwFlagsBase = BypassWriter()
    for i in blocks.indices {
        if isSkip[i] { continue }
        if let tz = isTreez {
            if i < tz.count {
                if tz[i] { continue }
            }
        }
        let isZ = isZeroFlags[i]
        bwFlagsBase.writeBit(isZ)
    }
    bwFlagsBase.flush()

    let baseData = encoderBase.getData(selectModel: selectModel, history: history, updateHistory: false)
    let baseTotalBytes = bwFlagsBase.bytes.count + baseData.count

    // 2. Candidate B: rANSContext Plane Encoding
    ws.resetPlaneEncoder()

    // (a) Escapes in forward order
    var fIdx = 0
    let nzCount = nonZeroIndices.count
    while fIdx < nzCount {
        let bIdx = nonZeroIndices[fIdx]
        ws.cArr.withUnsafeMutableBufferPointer { cPtr in
            copyLLCoeffs(from: blocks[bIdx], to: cPtr.baseAddress!)
        }
        var pos = 4
        while pos < 16 {
            let val = ws.cArr[pos]
            if val < -64 {
                ws.planeEncodeEscape(val: val)
            } else {
                if 64 < val {
                    ws.planeEncodeEscape(val: val)
                }
            }
            pos += 1
        }
        fIdx += 1
    }

    // (b) rANS Symbols in backward order with spatial context wiring
    var rBlockIdx = nzCount - 1
    while 0 <= rBlockIdx {
        let bIdx = nonZeroIndices[rBlockIdx]
        let blockY = bIdx / colCount
        let blockX = bIdx % colCount

        ws.cArr.withUnsafeMutableBufferPointer { cPtr in
            copyLLCoeffs(from: blocks[bIdx], to: cPtr.baseAddress!)
        }

        var hasTop = false
        if 0 < blockY {
            ws.topBuf.withUnsafeMutableBufferPointer { topPtr in
                copyLLCoeffs(from: blocks[bIdx - colCount], to: topPtr.baseAddress!)
            }
            hasTop = true
        }

        var hasLeft = false
        if 0 < blockX {
            ws.leftBuf.withUnsafeMutableBufferPointer { leftPtr in
                copyLLCoeffs(from: blocks[bIdx - 1], to: leftPtr.baseAddress!)
            }
            hasLeft = true
        }

        var rPos = 15
        while 4 <= rPos {
            let val = ws.cArr[rPos]
            let (mu, invScale): (Int32, Int32)
            switch true {
            case hasTop && hasLeft:
                (mu, invScale) = withUnsafePointers(ws.cArr, ws.topBuf, ws.leftBuf) { cPtr, topPtr, leftPtr in
                    ws.predict(pos: rPos, blockCoeffs: cPtr, topCoeffs: topPtr, leftCoeffs: leftPtr, tempCoeffs: nil, isPFrame: true, plane: 0, qstep: qstep)
                }
            case hasTop:
                (mu, invScale) = withUnsafePointers(ws.cArr, ws.topBuf) { cPtr, topPtr in
                    ws.predict(pos: rPos, blockCoeffs: cPtr, topCoeffs: topPtr, leftCoeffs: nil, tempCoeffs: nil, isPFrame: true, plane: 0, qstep: qstep)
                }
            case hasLeft:
                (mu, invScale) = withUnsafePointers(ws.cArr, ws.leftBuf) { cPtr, leftPtr in
                    ws.predict(pos: rPos, blockCoeffs: cPtr, topCoeffs: nil, leftCoeffs: leftPtr, tempCoeffs: nil, isPFrame: true, plane: 0, qstep: qstep)
                }
            default:
                (mu, invScale) = withUnsafePointers(ws.cArr) { cPtr in
                    ws.predict(pos: rPos, blockCoeffs: cPtr, topCoeffs: nil, leftCoeffs: nil, tempCoeffs: nil, isPFrame: true, plane: 0, qstep: qstep)
                }
            }
            ws.buildCDF(muQ12: mu, invScaleQ12: invScale)

            let sym: Int
            if val < -64 {
                sym = 129
            } else {
                if 64 < val {
                    sym = 129
                } else {
                    sym = Int(val + 64)
                }
            }

            let freq = ws.freqs[sym]
            let cumFreq = ws.cumFreqs[sym]
            ws.planeEncodeSymbol(sym: sym, freq: freq, cumFreq: cumFreq)
            rPos -= 1
        }
        rBlockIdx -= 1
    }
    let modelBytes = ws.finalizePlaneEncoder()

    var bwFlagsModel = BypassWriter()
    for i in blocks.indices {
        if isSkip[i] { continue }
        if let tz = isTreez {
            if i < tz.count {
                if tz[i] { continue }
            }
        }
        let isZ = isZeroFlags[i]
        bwFlagsModel.writeBit(isZ)
    }
    bwFlagsModel.flush()

    var encoderModel = EntropyEncoder()
    var nzCurModel = 0
    for i in blocks.indices {
        if nzCurModel < nzCount && nonZeroIndices[nzCurModel] == i {
            nzCurModel += 1
            let view = blocks[i]
            let subs = getSubbands8(view: view)
            blockEncode4HHead(encoder: &encoderModel, block: subs.ll)
            blockEncode4V(encoder: &encoderModel, block: subs.hl)
            blockEncode4H(encoder: &encoderModel, block: subs.lh)
            blockEncode4H(encoder: &encoderModel, block: subs.hh)
        }
    }
    encoderModel.flush()

    let modelEntropyData = encoderModel.getData(selectModel: selectModel, history: history, updateHistory: false)
    let modelTotalBytes = bwFlagsModel.bytes.count + 4 + modelBytes.count + modelEntropyData.count

    // 3. Plane-level Rate-Distortion Comparison
    //
    // Both candidates are serialized exactly once, above, with the history
    // update withheld. The winner emits the bytes it was measured with and then
    // applies its own update, so the size decision and the emitted bytes can
    // never diverge.
    if modelTotalBytes < baseTotalBytes {
        var bufModel = bwFlagsModel.bytes
        let mByteCount = UInt32(modelBytes.count)
        bufModel.append(UInt8(truncatingIfNeeded: (mByteCount >> 24) & 0xFF))
        bufModel.append(UInt8(truncatingIfNeeded: (mByteCount >> 16) & 0xFF))
        bufModel.append(UInt8(truncatingIfNeeded: (mByteCount >> 8) & 0xFF))
        bufModel.append(UInt8(truncatingIfNeeded: mByteCount & 0xFF))
        bufModel.append(contentsOf: modelBytes)
        bufModel.append(contentsOf: modelEntropyData)
        if updateHistory {
            encoderModel.commitDeferredHistory(to: history)
        }
        return (bufModel, isZeroFlags, true)
    } else {
        var bufBase = bwFlagsBase.bytes
        bufBase.append(contentsOf: baseData)
        if updateHistory {
            encoderBase.commitDeferredHistory(to: history)
        }
        return (bufBase, isZeroFlags, false)
    }
}


// MARK: - Dedicated Subband Process Functions

@inline(__always)
func encodeSubbands32WithParent(
    task: EncodeTask32,
    encoder: inout EntropyEncoder,
    subs: Subbands,
    parentHL: BlockView,
    parentLH: BlockView,
    parentHH: BlockView
) {
    switch task {
    case .encode16:
        blockEncode16VWithParent(encoder: &encoder, block: subs.hl, parentBlock: parentHL)
        blockEncode16HWithParent(encoder: &encoder, block: subs.lh, parentBlock: parentLH)
        blockEncode16HWithParent(encoder: &encoder, block: subs.hh, parentBlock: parentHH)
    case .split8(let tl, let tr, let bl, let br):
        if tl {
            let pbHL = BlockView(base: parentHL.base, width: 4, height: 4, stride: parentHL.stride)
            let pbLH = BlockView(base: parentLH.base, width: 4, height: 4, stride: parentLH.stride)
            let pbHH = BlockView(base: parentHH.base, width: 4, height: 4, stride: parentHH.stride)
            let hl = BlockView(base: subs.hl.base, width: 8, height: 8, stride: 32)
            let lh = BlockView(base: subs.lh.base, width: 8, height: 8, stride: 32)
            let hh = BlockView(base: subs.hh.base, width: 8, height: 8, stride: 32)
            blockEncode8HWithParent(encoder: &encoder, block: hl, parentBlock: pbHL)
            blockEncode8HWithParent(encoder: &encoder, block: lh, parentBlock: pbLH)
            blockEncode8HWithParent(encoder: &encoder, block: hh, parentBlock: pbHH)
        } else {
            clearBlockRegion(base: subs.hl.base, width: 8, height: 8, stride: 32)
            clearBlockRegion(base: subs.lh.base, width: 8, height: 8, stride: 32)
            clearBlockRegion(base: subs.hh.base, width: 8, height: 8, stride: 32)
        }
        if tr {
            let pbHL = BlockView(base: parentHL.base.advanced(by: 4), width: 4, height: 4, stride: parentHL.stride)
            let pbLH = BlockView(base: parentLH.base.advanced(by: 4), width: 4, height: 4, stride: parentLH.stride)
            let pbHH = BlockView(base: parentHH.base.advanced(by: 4), width: 4, height: 4, stride: parentHH.stride)
            let hl = BlockView(base: subs.hl.base.advanced(by: 8), width: 8, height: 8, stride: 32)
            let lh = BlockView(base: subs.lh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
            let hh = BlockView(base: subs.hh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
            blockEncode8HWithParent(encoder: &encoder, block: hl, parentBlock: pbHL)
            blockEncode8HWithParent(encoder: &encoder, block: lh, parentBlock: pbLH)
            blockEncode8HWithParent(encoder: &encoder, block: hh, parentBlock: pbHH)
        } else {
            clearBlockRegion(base: subs.hl.base.advanced(by: 8), width: 8, height: 8, stride: 32)
            clearBlockRegion(base: subs.lh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
            clearBlockRegion(base: subs.hh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
        }
        if bl {
            let pbHL = BlockView(base: parentHL.base.advanced(by: 4 * parentHL.stride), width: 4, height: 4, stride: parentHL.stride)
            let pbLH = BlockView(base: parentLH.base.advanced(by: 4 * parentLH.stride), width: 4, height: 4, stride: parentLH.stride)
            let pbHH = BlockView(base: parentHH.base.advanced(by: 4 * parentHH.stride), width: 4, height: 4, stride: parentHH.stride)
            let hl = BlockView(base: subs.hl.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
            let lh = BlockView(base: subs.lh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
            let hh = BlockView(base: subs.hh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
            blockEncode8HWithParent(encoder: &encoder, block: hl, parentBlock: pbHL)
            blockEncode8HWithParent(encoder: &encoder, block: lh, parentBlock: pbLH)
            blockEncode8HWithParent(encoder: &encoder, block: hh, parentBlock: pbHH)
        } else {
            clearBlockRegion(base: subs.hl.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
            clearBlockRegion(base: subs.lh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
            clearBlockRegion(base: subs.hh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
        }
        if br {
            let pbHL = BlockView(base: parentHL.base.advanced(by: 4 * parentHL.stride + 4), width: 4, height: 4, stride: parentHL.stride)
            let pbLH = BlockView(base: parentLH.base.advanced(by: 4 * parentLH.stride + 4), width: 4, height: 4, stride: parentLH.stride)
            let pbHH = BlockView(base: parentHH.base.advanced(by: 4 * parentHH.stride + 4), width: 4, height: 4, stride: parentHH.stride)
            let hl = BlockView(base: subs.hl.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
            let lh = BlockView(base: subs.lh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
            let hh = BlockView(base: subs.hh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
            blockEncode8HWithParent(encoder: &encoder, block: hl, parentBlock: pbHL)
            blockEncode8HWithParent(encoder: &encoder, block: lh, parentBlock: pbLH)
            blockEncode8HWithParent(encoder: &encoder, block: hh, parentBlock: pbHH)
        } else {
            clearBlockRegion(base: subs.hl.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
            clearBlockRegion(base: subs.lh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
            clearBlockRegion(base: subs.hh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
        }
    }
}

@inline(__always)
func encodeSubbands32WithoutParent(
    task: EncodeTask32,
    encoder: inout EntropyEncoder,
    subs: Subbands,
    lastVal: inout Int16
) {
    switch task {
    case .encode16:
        blockEncodeDPCM16(encoder: &encoder, block: subs.ll, lastVal: &lastVal)
        blockEncode16V(encoder: &encoder, block: subs.hl)
        blockEncode16H(encoder: &encoder, block: subs.lh)
        blockEncode16H(encoder: &encoder, block: subs.hh)
    case .split8(let tl, let tr, let bl, let br):
        if tl {
            let ll = BlockView(base: subs.ll.base, width: 8, height: 8, stride: 32)
            let hl = BlockView(base: subs.hl.base, width: 8, height: 8, stride: 32)
            let lh = BlockView(base: subs.lh.base, width: 8, height: 8, stride: 32)
            let hh = BlockView(base: subs.hh.base, width: 8, height: 8, stride: 32)
            blockEncodeDPCM8(encoder: &encoder, block: ll, lastVal: &lastVal)
            blockEncode8H(encoder: &encoder, block: hl)
            blockEncode8H(encoder: &encoder, block: lh)
            blockEncode8H(encoder: &encoder, block: hh)
        } else {
            clearBlockRegion(base: subs.hl.base, width: 8, height: 8, stride: 32)
            clearBlockRegion(base: subs.lh.base, width: 8, height: 8, stride: 32)
            clearBlockRegion(base: subs.hh.base, width: 8, height: 8, stride: 32)
        }
        if tr {
            let ll = BlockView(base: subs.ll.base.advanced(by: 8), width: 8, height: 8, stride: 32)
            let hl = BlockView(base: subs.hl.base.advanced(by: 8), width: 8, height: 8, stride: 32)
            let lh = BlockView(base: subs.lh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
            let hh = BlockView(base: subs.hh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
            blockEncodeDPCM8(encoder: &encoder, block: ll, lastVal: &lastVal)
            blockEncode8H(encoder: &encoder, block: hl)
            blockEncode8H(encoder: &encoder, block: lh)
            blockEncode8H(encoder: &encoder, block: hh)
        } else {
            clearBlockRegion(base: subs.hl.base.advanced(by: 8), width: 8, height: 8, stride: 32)
            clearBlockRegion(base: subs.lh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
            clearBlockRegion(base: subs.hh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
        }
        if bl {
            let ll = BlockView(base: subs.ll.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
            let hl = BlockView(base: subs.hl.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
            let lh = BlockView(base: subs.lh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
            let hh = BlockView(base: subs.hh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
            blockEncodeDPCM8(encoder: &encoder, block: ll, lastVal: &lastVal)
            blockEncode8H(encoder: &encoder, block: hl)
            blockEncode8H(encoder: &encoder, block: lh)
            blockEncode8H(encoder: &encoder, block: hh)
        } else {
            clearBlockRegion(base: subs.hl.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
            clearBlockRegion(base: subs.lh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
            clearBlockRegion(base: subs.hh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
        }
        if br {
            let ll = BlockView(base: subs.ll.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
            let hl = BlockView(base: subs.hl.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
            let lh = BlockView(base: subs.lh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
            let hh = BlockView(base: subs.hh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
            blockEncodeDPCM8(encoder: &encoder, block: ll, lastVal: &lastVal)
            blockEncode8H(encoder: &encoder, block: hl)
            blockEncode8H(encoder: &encoder, block: lh)
            blockEncode8H(encoder: &encoder, block: hh)
        } else {
            clearBlockRegion(base: subs.hl.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
            clearBlockRegion(base: subs.lh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
            clearBlockRegion(base: subs.hh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
        }
    }
}

@inline(__always)
func encodeSubbands16WithParent(
    task: EncodeTask16,
    encoder: inout EntropyEncoder,
    subs: Subbands,
    parentHL: BlockView,
    parentLH: BlockView,
    parentHH: BlockView
) {
    switch task {
    case .encode8:
        blockEncode8VWithParent(encoder: &encoder, block: subs.hl, parentBlock: parentHL)
        blockEncode8HWithParent(encoder: &encoder, block: subs.lh, parentBlock: parentLH)
        blockEncode8HWithParent(encoder: &encoder, block: subs.hh, parentBlock: parentHH)
    case .split4(let tl, let tr, let bl, let br):
        if tl {
            let pbHL = BlockView(base: parentHL.base, width: 2, height: 2, stride: parentHL.stride)
            let pbLH = BlockView(base: parentLH.base, width: 2, height: 2, stride: parentLH.stride)
            let pbHH = BlockView(base: parentHH.base, width: 2, height: 2, stride: parentHH.stride)
            let hl = BlockView(base: subs.hl.base, width: 4, height: 4, stride: 16)
            let lh = BlockView(base: subs.lh.base, width: 4, height: 4, stride: 16)
            let hh = BlockView(base: subs.hh.base, width: 4, height: 4, stride: 16)
            blockEncode4HWithParent(encoder: &encoder, block: hl, parentBlock: pbHL)
            blockEncode4HWithParent(encoder: &encoder, block: lh, parentBlock: pbLH)
            blockEncode4HWithParent(encoder: &encoder, block: hh, parentBlock: pbHH)
        }
        if tr {
            let pbHL = BlockView(base: parentHL.base.advanced(by: 2), width: 2, height: 2, stride: parentHL.stride)
            let pbLH = BlockView(base: parentLH.base.advanced(by: 2), width: 2, height: 2, stride: parentLH.stride)
            let pbHH = BlockView(base: parentHH.base.advanced(by: 2), width: 2, height: 2, stride: parentHH.stride)
            let hl = BlockView(base: subs.hl.base.advanced(by: 4), width: 4, height: 4, stride: 16)
            let lh = BlockView(base: subs.lh.base.advanced(by: 4), width: 4, height: 4, stride: 16)
            let hh = BlockView(base: subs.hh.base.advanced(by: 4), width: 4, height: 4, stride: 16)
            blockEncode4HWithParent(encoder: &encoder, block: hl, parentBlock: pbHL)
            blockEncode4HWithParent(encoder: &encoder, block: lh, parentBlock: pbLH)
            blockEncode4HWithParent(encoder: &encoder, block: hh, parentBlock: pbHH)
        }
        if bl {
            let pbHL = BlockView(base: parentHL.base.advanced(by: 2 * parentHL.stride), width: 2, height: 2, stride: parentHL.stride)
            let pbLH = BlockView(base: parentLH.base.advanced(by: 2 * parentLH.stride), width: 2, height: 2, stride: parentLH.stride)
            let pbHH = BlockView(base: parentHH.base.advanced(by: 2 * parentHH.stride), width: 2, height: 2, stride: parentHH.stride)
            let hl = BlockView(base: subs.hl.base.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
            let lh = BlockView(base: subs.lh.base.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
            let hh = BlockView(base: subs.hh.base.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
            blockEncode4HWithParent(encoder: &encoder, block: hl, parentBlock: pbHL)
            blockEncode4HWithParent(encoder: &encoder, block: lh, parentBlock: pbLH)
            blockEncode4HWithParent(encoder: &encoder, block: hh, parentBlock: pbHH)
        }
        if br {
            let pbHL = BlockView(base: parentHL.base.advanced(by: 2 * parentHL.stride + 2), width: 2, height: 2, stride: parentHL.stride)
            let pbLH = BlockView(base: parentLH.base.advanced(by: 2 * parentLH.stride + 2), width: 2, height: 2, stride: parentLH.stride)
            let pbHH = BlockView(base: parentHH.base.advanced(by: 2 * parentHH.stride + 2), width: 2, height: 2, stride: parentHH.stride)
            let hl = BlockView(base: subs.hl.base.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
            let lh = BlockView(base: subs.lh.base.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
            let hh = BlockView(base: subs.hh.base.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
            blockEncode4HWithParent(encoder: &encoder, block: hl, parentBlock: pbHL)
            blockEncode4HWithParent(encoder: &encoder, block: lh, parentBlock: pbLH)
            blockEncode4HWithParent(encoder: &encoder, block: hh, parentBlock: pbHH)
        }
    }
}

@inline(__always)
func encodeSubbands16WithoutParent(
    task: EncodeTask16,
    encoder: inout EntropyEncoder,
    subs: Subbands
) {
    switch task {
    case .encode8:
        blockEncode8V(encoder: &encoder, block: subs.hl)
        blockEncode8H(encoder: &encoder, block: subs.lh)
        blockEncode8H(encoder: &encoder, block: subs.hh)
    case .split4(let tl, let tr, let bl, let br):
        if tl {
            let hl = BlockView(base: subs.hl.base, width: 4, height: 4, stride: 16)
            let lh = BlockView(base: subs.lh.base, width: 4, height: 4, stride: 16)
            let hh = BlockView(base: subs.hh.base, width: 4, height: 4, stride: 16)
            blockEncode4H(encoder: &encoder, block: hl)
            blockEncode4H(encoder: &encoder, block: lh)
            blockEncode4H(encoder: &encoder, block: hh)
        }
        if tr {
            let hl = BlockView(base: subs.hl.base.advanced(by: 4), width: 4, height: 4, stride: 16)
            let lh = BlockView(base: subs.lh.base.advanced(by: 4), width: 4, height: 4, stride: 16)
            let hh = BlockView(base: subs.hh.base.advanced(by: 4), width: 4, height: 4, stride: 16)
            blockEncode4H(encoder: &encoder, block: hl)
            blockEncode4H(encoder: &encoder, block: lh)
            blockEncode4H(encoder: &encoder, block: hh)
        }
        if bl {
            let hl = BlockView(base: subs.hl.base.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
            let lh = BlockView(base: subs.lh.base.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
            let hh = BlockView(base: subs.hh.base.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
            blockEncode4H(encoder: &encoder, block: hl)
            blockEncode4H(encoder: &encoder, block: lh)
            blockEncode4H(encoder: &encoder, block: hh)
        }
        if br {
            let hl = BlockView(base: subs.hl.base.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
            let lh = BlockView(base: subs.lh.base.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
            let hh = BlockView(base: subs.hh.base.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
            blockEncode4H(encoder: &encoder, block: hl)
            blockEncode4H(encoder: &encoder, block: lh)
            blockEncode4H(encoder: &encoder, block: hh)
        }
    }
}