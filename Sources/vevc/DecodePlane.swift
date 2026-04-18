enum DecodeTask32 {
    case skip
    case decode16
    case split8(Bool, Bool, Bool, Bool)
}

@inline(__always)
fileprivate func decodePlaneSubbands32BlockView(buf: UnsafeBufferPointer<UInt8>, pool: BlockViewPool, blockCount: Int, parentBlocks: [BlockView]?) throws -> [BlockView] {
    guard let base = buf.baseAddress else { return [] }
    let count = buf.count
    var blocks: [BlockView] = []
    blocks.reserveCapacity(blockCount)
    for _ in 0..<blockCount {
        blocks.append(pool.get(width: 32, height: 32))
    }
    
    var brFlags = BypassReader(base: base, count: count)
    var tasks: [(Int, DecodeTask32)] = []
    tasks.reserveCapacity(blockCount)
    for i in 0..<blockCount {
        let isZero = brFlags.readBit()
        if isZero {
            tasks.append((i, .skip))
        } else {
            let mbType = brFlags.readBit()
            if mbType {
                let tlZero = brFlags.readBit()
                if tlZero != true { let _ = brFlags.readBit() }
                
                let trZero = brFlags.readBit()
                if trZero != true { let _ = brFlags.readBit() }
                
                let blZero = brFlags.readBit()
                if blZero != true { let _ = brFlags.readBit() }
                
                let brZero = brFlags.readBit()
                if brZero != true { let _ = brFlags.readBit() }
                
                tasks.append((i, .split8(
                    tlZero != true, 
                    trZero != true, 
                    blZero != true, 
                    brZero != true
                )))
            } else {
                tasks.append((i, .decode16))
            }
        }
    }
    
    let consumed = brFlags.consumedBytes
    guard consumed <= count else { throw DecodeError.insufficientData }
    
    var decoder = try EntropyDecoder(base: base, count: count, startOffset: consumed)
    
    let half = 32 / 2
    
    for (i, task) in tasks {
        let decodeAction = { (parentHL: BlockView?, parentLH: BlockView?, parentHH: BlockView?) throws in
            let view = blocks[i]
            let hlBase = view.base.advanced(by: half)
            let lhBase = view.base.advanced(by: half * 32)
            let hhBase = view.base.advanced(by: half * 32 + half)
            
            switch task {
            case .skip:
                break
            case .decode16:
                let hlView = BlockView(base: hlBase, width: half, height: half, stride: 32)
                try blockDecode16V(decoder: &decoder, block: hlView, parentBlock: parentHL)
                
                let lhView = BlockView(base: lhBase, width: half, height: half, stride: 32)
                try blockDecode16H(decoder: &decoder, block: lhView, parentBlock: parentLH)
                
                let hhView = BlockView(base: hhBase, width: half, height: half, stride: 32)
                try blockDecode16H(decoder: &decoder, block: hhView, parentBlock: parentHH)
            case .split8(let tl, let tr, let bl, let br):
                if tl {
                    var pbHL: BlockView? = nil
                    if let p = parentHL { pbHL = BlockView(base: p.base, width: 4, height: 4, stride: p.stride) }
                    var pbLH: BlockView? = nil
                    if let p = parentLH { pbLH = BlockView(base: p.base, width: 4, height: 4, stride: p.stride) }
                    var pbHH: BlockView? = nil
                    if let p = parentHH { pbHH = BlockView(base: p.base, width: 4, height: 4, stride: p.stride) }
                    let hl = BlockView(base: hlBase, width: 8, height: 8, stride: 32)
                    let lh = BlockView(base: lhBase, width: 8, height: 8, stride: 32)
                    let hh = BlockView(base: hhBase, width: 8, height: 8, stride: 32)
                    try blockDecode8H(decoder: &decoder, block: hl, parentBlock: pbHL)
                    try blockDecode8H(decoder: &decoder, block: lh, parentBlock: pbLH)
                    try blockDecode8H(decoder: &decoder, block: hh, parentBlock: pbHH)
                }
                if tr {
                    var pbHL: BlockView? = nil
                    if let p = parentHL { pbHL = BlockView(base: p.base.advanced(by: 4), width: 4, height: 4, stride: p.stride) }
                    var pbLH: BlockView? = nil
                    if let p = parentLH { pbLH = BlockView(base: p.base.advanced(by: 4), width: 4, height: 4, stride: p.stride) }
                    var pbHH: BlockView? = nil
                    if let p = parentHH { pbHH = BlockView(base: p.base.advanced(by: 4), width: 4, height: 4, stride: p.stride) }
                    let hl = BlockView(base: hlBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    let lh = BlockView(base: lhBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    let hh = BlockView(base: hhBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    try blockDecode8H(decoder: &decoder, block: hl, parentBlock: pbHL)
                    try blockDecode8H(decoder: &decoder, block: lh, parentBlock: pbLH)
                    try blockDecode8H(decoder: &decoder, block: hh, parentBlock: pbHH)
                }
                if bl {
                    var pbHL: BlockView? = nil
                    if let p = parentHL { pbHL = BlockView(base: p.base.advanced(by: 4 * p.stride), width: 4, height: 4, stride: p.stride) }
                    var pbLH: BlockView? = nil
                    if let p = parentLH { pbLH = BlockView(base: p.base.advanced(by: 4 * p.stride), width: 4, height: 4, stride: p.stride) }
                    var pbHH: BlockView? = nil
                    if let p = parentHH { pbHH = BlockView(base: p.base.advanced(by: 4 * p.stride), width: 4, height: 4, stride: p.stride) }
                    let hl = BlockView(base: hlBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    let lh = BlockView(base: lhBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    let hh = BlockView(base: hhBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    try blockDecode8H(decoder: &decoder, block: hl, parentBlock: pbHL)
                    try blockDecode8H(decoder: &decoder, block: lh, parentBlock: pbLH)
                    try blockDecode8H(decoder: &decoder, block: hh, parentBlock: pbHH)
                }
                if br {
                    var pbHL: BlockView? = nil
                    if let p = parentHL { pbHL = BlockView(base: p.base.advanced(by: 4 * p.stride + 4), width: 4, height: 4, stride: p.stride) }
                    var pbLH: BlockView? = nil
                    if let p = parentLH { pbLH = BlockView(base: p.base.advanced(by: 4 * p.stride + 4), width: 4, height: 4, stride: p.stride) }
                    var pbHH: BlockView? = nil
                    if let p = parentHH { pbHH = BlockView(base: p.base.advanced(by: 4 * p.stride + 4), width: 4, height: 4, stride: p.stride) }
                    let hl = BlockView(base: hlBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    let lh = BlockView(base: lhBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    let hh = BlockView(base: hhBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    try blockDecode8H(decoder: &decoder, block: hl, parentBlock: pbHL)
                    try blockDecode8H(decoder: &decoder, block: lh, parentBlock: pbLH)
                    try blockDecode8H(decoder: &decoder, block: hh, parentBlock: pbHH)
                }
            }
        }
        
        if let pBlocks = parentBlocks, i < pBlocks.count {
            let pBlock = pBlocks[i]
            let pView = pBlock
            let pSubs = getSubbands16(view: pView)
            try decodeAction(pSubs.hl, pSubs.lh, pSubs.hh)
        } else {
            try decodeAction(nil, nil, nil)
        }
    }

    return blocks
}

@inline(__always)
func decodePlaneSubbands32(data: [UInt8], pool: BlockViewPool, blockCount: Int, parentBlocks: [BlockView]?) throws -> [BlockView] {
    return try data.withUnsafeBufferPointer { buf -> [BlockView] in
        return try decodePlaneSubbands32BlockView(buf: buf, pool: pool, blockCount: blockCount, parentBlocks: parentBlocks)
    }
}


enum DecodeTask16 {
    case skip
    case decode8
    case split4(Bool, Bool, Bool, Bool)
}

@inline(__always)
fileprivate func decodePlaneSubbands16BlockView(buf: UnsafeBufferPointer<UInt8>, pool: BlockViewPool, blockCount: Int, parentBlocks: [BlockView]?) throws -> [BlockView] {
    guard let base = buf.baseAddress else { return [] }
    let count = buf.count
    var blocks: [BlockView] = []
    blocks.reserveCapacity(blockCount)
    for _ in 0..<blockCount {
        blocks.append(pool.get(width: 16, height: 16))
    }
    
    var brFlags = BypassReader(base: base, count: count)
    var tasks: [(Int, DecodeTask16)] = []
    tasks.reserveCapacity(blockCount)
    for i in 0..<blockCount {
        if count < brFlags.consumedBytes {
            throw DecodeError.outOfBits
        }
        let isZero = brFlags.readBit()
        if isZero {
            tasks.append((i, .skip))
        } else {
            let mbType = brFlags.readBit()
            if mbType {
                let tlZero = brFlags.readBit()
                if tlZero != true { let _ = brFlags.readBit() }
                
                let trZero = brFlags.readBit()
                if trZero != true { let _ = brFlags.readBit() }
                
                let blZero = brFlags.readBit()
                if blZero != true { let _ = brFlags.readBit() }
                
                let brZero = brFlags.readBit()
                if brZero != true { let _ = brFlags.readBit() }
                
                tasks.append((i, .split4(
                    tlZero != true, 
                    trZero != true, 
                    blZero != true, 
                    brZero != true,
                )))
            } else {
                tasks.append((i, .decode8))
            }
        }
    }
    
    let consumed = brFlags.consumedBytes
    guard consumed <= count else { throw DecodeError.insufficientData }
    
    var decoder = try EntropyDecoder(base: base, count: count, startOffset: consumed)
    
    let half = 16 / 2

    for (i, task) in tasks {
        let decodeAction = { (parentHL: BlockView?, parentLH: BlockView?, parentHH: BlockView?) throws in
            let view = blocks[i]
            let hlBase = view.base.advanced(by: half)
            let lhBase = view.base.advanced(by: half * 16)
            let hhBase = view.base.advanced(by: half * 16 + half)
        
        switch task {
        case .skip:
            break
        case .decode8:
            let hlView = BlockView(base: hlBase, width: half, height: half, stride: 16)
            try blockDecode8V(decoder: &decoder, block: hlView, parentBlock: parentHL)
            
            let lhView = BlockView(base: lhBase, width: half, height: half, stride: 16)
            try blockDecode8H(decoder: &decoder, block: lhView, parentBlock: parentLH)
            
            let hhView = BlockView(base: hhBase, width: half, height: half, stride: 16)
            try blockDecode8H(decoder: &decoder, block: hhView, parentBlock: parentHH)
        case .split4(let tl, let tr, let bl, let br):
            if tl {
                var pbHL: BlockView? = nil
                if let p = parentHL { pbHL = BlockView(base: p.base, width: 2, height: 2, stride: p.stride) }
                var pbLH: BlockView? = nil
                if let p = parentLH { pbLH = BlockView(base: p.base, width: 2, height: 2, stride: p.stride) }
                var pbHH: BlockView? = nil
                if let p = parentHH { pbHH = BlockView(base: p.base, width: 2, height: 2, stride: p.stride) }
                let hl = BlockView(base: hlBase, width: 4, height: 4, stride: 16)
                let lh = BlockView(base: lhBase, width: 4, height: 4, stride: 16)
                let hh = BlockView(base: hhBase, width: 4, height: 4, stride: 16)
                try blockDecode4H(decoder: &decoder, block: hl, parentBlock: pbHL)
                try blockDecode4H(decoder: &decoder, block: lh, parentBlock: pbLH)
                try blockDecode4H(decoder: &decoder, block: hh, parentBlock: pbHH)
            }
            if tr {
                var pbHL: BlockView? = nil
                if let p = parentHL { pbHL = BlockView(base: p.base.advanced(by: 2), width: 2, height: 2, stride: p.stride) }
                var pbLH: BlockView? = nil
                if let p = parentLH { pbLH = BlockView(base: p.base.advanced(by: 2), width: 2, height: 2, stride: p.stride) }
                var pbHH: BlockView? = nil
                if let p = parentHH { pbHH = BlockView(base: p.base.advanced(by: 2), width: 2, height: 2, stride: p.stride) }
                let hl = BlockView(base: hlBase.advanced(by: 4), width: 4, height: 4, stride: 16)
                let lh = BlockView(base: lhBase.advanced(by: 4), width: 4, height: 4, stride: 16)
                let hh = BlockView(base: hhBase.advanced(by: 4), width: 4, height: 4, stride: 16)
                try blockDecode4H(decoder: &decoder, block: hl, parentBlock: pbHL)
                try blockDecode4H(decoder: &decoder, block: lh, parentBlock: pbLH)
                try blockDecode4H(decoder: &decoder, block: hh, parentBlock: pbHH)
            }
            if bl {
                var pbHL: BlockView? = nil
                if let p = parentHL { pbHL = BlockView(base: p.base.advanced(by: 2 * p.stride), width: 2, height: 2, stride: p.stride) }
                var pbLH: BlockView? = nil
                if let p = parentLH { pbLH = BlockView(base: p.base.advanced(by: 2 * p.stride), width: 2, height: 2, stride: p.stride) }
                var pbHH: BlockView? = nil
                if let p = parentHH { pbHH = BlockView(base: p.base.advanced(by: 2 * p.stride), width: 2, height: 2, stride: p.stride) }
                let hl = BlockView(base: hlBase.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
                let lh = BlockView(base: lhBase.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
                let hh = BlockView(base: hhBase.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
                try blockDecode4H(decoder: &decoder, block: hl, parentBlock: pbHL)
                try blockDecode4H(decoder: &decoder, block: lh, parentBlock: pbLH)
                try blockDecode4H(decoder: &decoder, block: hh, parentBlock: pbHH)
            }
            if br {
                var pbHL: BlockView? = nil
                if let p = parentHL { pbHL = BlockView(base: p.base.advanced(by: 2 * p.stride + 2), width: 2, height: 2, stride: p.stride) }
                var pbLH: BlockView? = nil
                if let p = parentLH { pbLH = BlockView(base: p.base.advanced(by: 2 * p.stride + 2), width: 2, height: 2, stride: p.stride) }
                var pbHH: BlockView? = nil
                if let p = parentHH { pbHH = BlockView(base: p.base.advanced(by: 2 * p.stride + 2), width: 2, height: 2, stride: p.stride) }
                let hl = BlockView(base: hlBase.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
                let lh = BlockView(base: lhBase.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
                let hh = BlockView(base: hhBase.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
                try blockDecode4H(decoder: &decoder, block: hl, parentBlock: pbHL)
                try blockDecode4H(decoder: &decoder, block: lh, parentBlock: pbLH)
                try blockDecode4H(decoder: &decoder, block: hh, parentBlock: pbHH)
            }
        }
            }
        
        if let pBlocks = parentBlocks, i < pBlocks.count {
            let pBlock = pBlocks[i]
            let pView = pBlock
            let pSubs = getSubbands8(view: pView)
            try decodeAction(pSubs.hl, pSubs.lh, pSubs.hh)
        } else {
            try decodeAction(nil, nil, nil)
        }
    }

    return blocks
}

@inline(__always)
func decodePlaneSubbands16(data: [UInt8], pool: BlockViewPool, blockCount: Int, parentBlocks: [BlockView]?) throws -> [BlockView] {
    return try data.withUnsafeBufferPointer { buf -> [BlockView] in
        return try decodePlaneSubbands16BlockView(buf: buf, pool: pool, blockCount: blockCount, parentBlocks: parentBlocks)
    }
}

@inline(__always)
fileprivate func decodePlaneSubbands8BlockView(buf: UnsafeBufferPointer<UInt8>, pool: BlockViewPool, blockCount: Int, parentImage: Image16?, dx: Int, planeType: Int) throws -> [BlockView] {
    guard let base = buf.baseAddress else { return [] }
    let count = buf.count
    var blocks: [BlockView] = []
    blocks.reserveCapacity(blockCount)
    for _ in 0..<blockCount {
        blocks.append(pool.get(width: 8, height: 8))
    }
    
    var brFlags = BypassReader(base: base, count: count)
    var nonZeroIndices: [Int] = []
    nonZeroIndices.reserveCapacity(blockCount)
    for i in 0..<blockCount {
        let isZero = brFlags.readBit()
        let _ = brFlags.readBit()
        if isZero != true {
            nonZeroIndices.append(i)
        }
    }
    
    let consumed = brFlags.consumedBytes
    guard consumed <= count else { throw DecodeError.insufficientData }
    
    var decoder = try EntropyDecoder(base: base, count: count, startOffset: consumed)
    
    let half = 8 / 2

    for i in nonZeroIndices {
        let colCount = (dx + 8 - 1) / 8
        let r = i / colCount
        let c = i % colCount
        let pbX = c * 4
        let pbY = r * 4
        
        var parentBlock2D: BlockView? = nil
        if let pImg = parentImage {
            switch planeType {
            case 0: parentBlock2D = pImg.getY(x: pbX, y: pbY, size: 4, pool: pool)
            case 1: parentBlock2D = pImg.getCb(x: pbX, y: pbY, size: 4, pool: pool)
            default: parentBlock2D = pImg.getCr(x: pbX, y: pbY, size: 4, pool: pool)
            }
        }
        
        let decodeAction = { (parentBlock: BlockView?) throws in
            let view = blocks[i]
            let hlView = BlockView(base: view.base.advanced(by: half), width: half, height: half, stride: 8)
            try blockDecode4V(decoder: &decoder, block: hlView, parentBlock: parentBlock)
            
            let lhView = BlockView(base: view.base.advanced(by: half * 8), width: half, height: half, stride: 8)
            try blockDecode4H(decoder: &decoder, block: lhView, parentBlock: parentBlock)
            
            let hhView = BlockView(base: view.base.advanced(by: half * 8 + half), width: half, height: half, stride: 8)
            try blockDecode4H(decoder: &decoder, block: hhView, parentBlock: parentBlock)
        }
        
        if let pb2d = parentBlock2D {
            let pb = pb2d
            try decodeAction(pb)
            pool.put(pb2d) // return temp block to pool
        } else {
            try decodeAction(nil)
        }
    }

    return blocks
}

@inline(__always)
func decodePlaneSubbands8(data: [UInt8], pool: BlockViewPool, blockCount: Int, parentImage: Image16?, dx: Int, planeType: Int) throws -> [BlockView] {
    return try data.withUnsafeBufferPointer { buf -> [BlockView] in
        return try decodePlaneSubbands8BlockView(buf: buf, pool: pool, blockCount: blockCount, parentImage: parentImage, dx: dx, planeType: planeType)
    }
}

@inline(__always)
fileprivate func decodePlaneBaseSubbands8BlockView(buf: UnsafeBufferPointer<UInt8>, pool: BlockViewPool, blockCount: Int, isIFrame: Bool) throws -> [BlockView] {
    guard let base = buf.baseAddress else { return [] }
    let count = buf.count
    var blocks: [BlockView] = []
    blocks.reserveCapacity(blockCount)
    for _ in 0..<blockCount {
        blocks.append(pool.get(width: 8, height: 8))
    }
    
    var brFlags = BypassReader(base: base, count: count)
    var nonZeroIndices: [Int] = []
    nonZeroIndices.reserveCapacity(blockCount)
    for i in 0..<blockCount {
        let isZero = brFlags.readBit()
        let _ = brFlags.readBit()
        if isZero != true {
            nonZeroIndices.append(i)
        }
    }
    
    let consumed = brFlags.consumedBytes
    guard consumed <= count else { throw DecodeError.insufficientData }
    
    var decoder = try EntropyDecoder(base: base, count: count, startOffset: consumed)
    
    let half = 8 / 2

    var lastVal: Int16 = 0
    var nzCur = 0
    let nzCount = nonZeroIndices.count
    for i in 0..<blockCount {
        if nzCur < nzCount && nonZeroIndices[nzCur] == i {
            nzCur += 1
            let view = blocks[i]
            let llView = BlockView(base: view.base, width: half, height: half, stride: 8)
            if isIFrame {
                try blockDecodeDPCM4(decoder: &decoder, block: llView, lastVal: &lastVal)
            } else {
                try blockDecode4H(decoder: &decoder, block: llView, parentBlock: nil)
            }
            
            let hlView = BlockView(base: view.base.advanced(by: half), width: half, height: half, stride: 8)
            try blockDecode4V(decoder: &decoder, block: hlView, parentBlock: nil)
            
            let lhView = BlockView(base: view.base.advanced(by: half * 8), width: half, height: half, stride: 8)
            try blockDecode4H(decoder: &decoder, block: lhView, parentBlock: nil)
            
            let hhView = BlockView(base: view.base.advanced(by: half * 8 + half), width: half, height: half, stride: 8)
            try blockDecode4H(decoder: &decoder, block: hhView, parentBlock: nil)
        } else {
            if isIFrame { lastVal = 0 }
        }
    }

    return blocks
}

@inline(__always)
func decodePlaneBaseSubbands8(data: [UInt8], pool: BlockViewPool, blockCount: Int, isIFrame: Bool) throws -> [BlockView] {
    return try data.withUnsafeBufferPointer { buf -> [BlockView] in
        return try decodePlaneBaseSubbands8BlockView(buf: buf, pool: pool, blockCount: blockCount, isIFrame: isIFrame)
    }
}

enum DecodeTaskBase32 {
    case skip
    case decode16
    case split8(Bool, Bool, Bool, Bool)
}

@inline(__always)
fileprivate func decodePlaneBaseSubbands32BlockView(buf: UnsafeBufferPointer<UInt8>, pool: BlockViewPool, blockCount: Int) throws -> [BlockView] {
    guard let base = buf.baseAddress else { return [] }
    let count = buf.count
    var blocks: [BlockView] = []
    blocks.reserveCapacity(blockCount)
    for _ in 0..<blockCount {
        blocks.append(pool.get(width: 32, height: 32))
    }
    
    var brFlags = BypassReader(base: base, count: count)
    var tasks: [(Int, DecodeTaskBase32)] = []
    tasks.reserveCapacity(blockCount)
    for i in 0..<blockCount {
        if count < brFlags.consumedBytes {
            throw DecodeError.outOfBits
        }
        let isZero = brFlags.readBit()
        if isZero {
            tasks.append((i, .skip))
        } else {
            let mbType = brFlags.readBit()
            if mbType {
                let tlZero = brFlags.readBit()
                if tlZero != true { let _ = brFlags.readBit() }
                
                let trZero = brFlags.readBit()
                if trZero != true { let _ = brFlags.readBit() }
                
                let blZero = brFlags.readBit()
                if blZero != true { let _ = brFlags.readBit() }
                
                let brZero = brFlags.readBit()
                if brZero != true { let _ = brFlags.readBit() }
                
                tasks.append((i, .split8(
                    tlZero != true, 
                    trZero != true, 
                    blZero != true, 
                    brZero != true,
                )))
            } else {
                tasks.append((i, .decode16))
            }
        }
    }
    
    let consumed = brFlags.consumedBytes
    guard consumed <= count else { throw DecodeError.insufficientData }
    
    var decoder = try EntropyDecoder(base: base, count: count, startOffset: consumed)
    
    let half = 32 / 2

    var lastVal: Int16 = 0
    for (i, task) in tasks {
        let view = blocks[i]
        let llBase = view.base
        let hlBase = view.base.advanced(by: half)
        let lhBase = view.base.advanced(by: half * 32)
        let hhBase = view.base.advanced(by: half * 32 + half)
        
        switch task {
        case .skip:
            lastVal = 0
        case .decode16:
            let llView = BlockView(base: llBase, width: half, height: half, stride: 32)
            try blockDecodeDPCM16(decoder: &decoder, block: llView, lastVal: &lastVal)
            
            let hlView = BlockView(base: hlBase, width: half, height: half, stride: 32)
            try blockDecode16V(decoder: &decoder, block: hlView, parentBlock: nil)
            
            let lhView = BlockView(base: lhBase, width: half, height: half, stride: 32)
            try blockDecode16H(decoder: &decoder, block: lhView, parentBlock: nil)
            
            let hhView = BlockView(base: hhBase, width: half, height: half, stride: 32)
            try blockDecode16H(decoder: &decoder, block: hhView, parentBlock: nil)
            
        case .split8(let tl, let tr, let bl, let br):
            if tl {
                let ll = BlockView(base: llBase, width: 8, height: 8, stride: 32)
                let hl = BlockView(base: hlBase, width: 8, height: 8, stride: 32)
                let lh = BlockView(base: lhBase, width: 8, height: 8, stride: 32)
                let hh = BlockView(base: hhBase, width: 8, height: 8, stride: 32)
                try blockDecodeDPCM8(decoder: &decoder, block: ll, lastVal: &lastVal)
                try blockDecode8H(decoder: &decoder, block: hl, parentBlock: nil)
                try blockDecode8H(decoder: &decoder, block: lh, parentBlock: nil)
                try blockDecode8H(decoder: &decoder, block: hh, parentBlock: nil)
            }
            if tr {
                let ll = BlockView(base: llBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                let hl = BlockView(base: hlBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                let lh = BlockView(base: lhBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                let hh = BlockView(base: hhBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                try blockDecodeDPCM8(decoder: &decoder, block: ll, lastVal: &lastVal)
                try blockDecode8H(decoder: &decoder, block: hl, parentBlock: nil)
                try blockDecode8H(decoder: &decoder, block: lh, parentBlock: nil)
                try blockDecode8H(decoder: &decoder, block: hh, parentBlock: nil)
            }
            if bl {
                let ll = BlockView(base: llBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                let hl = BlockView(base: hlBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                let lh = BlockView(base: lhBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                let hh = BlockView(base: hhBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                try blockDecodeDPCM8(decoder: &decoder, block: ll, lastVal: &lastVal)
                try blockDecode8H(decoder: &decoder, block: hl, parentBlock: nil)
                try blockDecode8H(decoder: &decoder, block: lh, parentBlock: nil)
                try blockDecode8H(decoder: &decoder, block: hh, parentBlock: nil)
            }
            if br {
                let ll = BlockView(base: llBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                let hl = BlockView(base: hlBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                let lh = BlockView(base: lhBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                let hh = BlockView(base: hhBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                try blockDecodeDPCM8(decoder: &decoder, block: ll, lastVal: &lastVal)
                try blockDecode8H(decoder: &decoder, block: hl, parentBlock: nil)
                try blockDecode8H(decoder: &decoder, block: lh, parentBlock: nil)
                try blockDecode8H(decoder: &decoder, block: hh, parentBlock: nil)
            }
        }
    }

    return blocks
}

@inline(__always)
func decodePlaneBaseSubbands32(data: [UInt8], pool: BlockViewPool, blockCount: Int) throws -> [BlockView] {
    return try data.withUnsafeBufferPointer { buf -> [BlockView] in
        return try decodePlaneBaseSubbands32BlockView(buf: buf, pool: pool, blockCount: blockCount)
    }
}

@inline(__always)
fileprivate func decodeCascadedPlaneSubbands32BlockView(buf: UnsafeBufferPointer<UInt8>, blocks: inout [BlockView]) throws {
    guard let base = buf.baseAddress else { return }
    let count = buf.count
    var bwFlags = BypassReader(base: base, count: count)
    var tasks: [(Int, Bool)] = []
    tasks.reserveCapacity(blocks.count)
    
    for i in blocks.indices {
        let isZero = bwFlags.readBit()
        if isZero {
            // blocks from pool are guaranteed zero
            tasks.append((i, true))
        } else {
            tasks.append((i, false))
        }
    }
    
    let consumed = bwFlags.consumedBytes
    guard consumed <= count else { throw DecodeError.insufficientData }
    var decoder = try EntropyDecoder(base: base, count: count, startOffset: consumed)
    var lastVal: Int16 = 0
    
    for (i, skip) in tasks {
        if skip {
            lastVal = 0
            continue
        }
        
        let view = blocks[i]
        let hl1 = BlockView(base: view.base.advanced(by: 16), width: 16, height: 16, stride: view.stride)
        let lh1 = BlockView(base: view.base.advanced(by: 16 * view.stride), width: 16, height: 16, stride: view.stride)
        let hh1 = BlockView(base: view.base.advanced(by: 16 * view.stride + 16), width: 16, height: 16, stride: view.stride)
        
        let hl2 = BlockView(base: view.base.advanced(by: 8), width: 8, height: 8, stride: view.stride)
        let lh2 = BlockView(base: view.base.advanced(by: 8 * view.stride), width: 8, height: 8, stride: view.stride)
        let hh2 = BlockView(base: view.base.advanced(by: 8 * view.stride + 8), width: 8, height: 8, stride: view.stride)
        
        let ll3 = BlockView(base: view.base, width: 4, height: 4, stride: view.stride)
        let hl3 = BlockView(base: view.base.advanced(by: 4), width: 4, height: 4, stride: view.stride)
        let lh3 = BlockView(base: view.base.advanced(by: 4 * view.stride), width: 4, height: 4, stride: view.stride)
        let hh3 = BlockView(base: view.base.advanced(by: 4 * view.stride + 4), width: 4, height: 4, stride: view.stride)
        
        let m_ll3 = ll3, m_hl3 = hl3, m_lh3 = lh3, m_hh3 = hh3
        try blockDecodeDPCM4(decoder: &decoder, block: m_ll3, lastVal: &lastVal)
        try blockDecode4H(decoder: &decoder, block: m_hl3, parentBlock: nil)
        try blockDecode4H(decoder: &decoder, block: m_lh3, parentBlock: nil)
        try blockDecode4H(decoder: &decoder, block: m_hh3, parentBlock: nil)
        
        let m_hl2 = hl2, m_lh2 = lh2, m_hh2 = hh2
        try blockDecode8H(decoder: &decoder, block: m_hl2, parentBlock: nil)
        try blockDecode8H(decoder: &decoder, block: m_lh2, parentBlock: nil)
        try blockDecode8H(decoder: &decoder, block: m_hh2, parentBlock: nil)
        
        let m_hl1 = hl1, m_lh1 = lh1, m_hh1 = hh1
        try blockDecode16H(decoder: &decoder, block: m_hl1, parentBlock: nil)
        try blockDecode16H(decoder: &decoder, block: m_lh1, parentBlock: nil)
        try blockDecode16H(decoder: &decoder, block: m_hh1, parentBlock: nil)
    }
}

@inline(__always)
func decodeCascadedPlaneSubbands32(data: [UInt8], blocks: inout [BlockView]) throws {
    try data.withUnsafeBufferPointer { buf in
        try decodeCascadedPlaneSubbands32BlockView(buf: buf, blocks: &blocks)
    }
}
