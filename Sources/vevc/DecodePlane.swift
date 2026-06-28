enum DecodeTask32 {
    case skip
    case decode16
    case split8(Bool, Bool, Bool, Bool)
}

@inline(__always)
fileprivate func decodePlaneSubbands32BlockView(buf: UnsafeBufferPointer<UInt8>, pool: BlockViewPool, blockCount: Int) throws -> [BlockView] {
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
    
    var lastVal: Int16 = 0
    
    for (i, task) in tasks {
        let view = blocks[i]
        let llBase = view.base
        let hlBase = view.base.advanced(by: half)
        let lhBase = view.base.advanced(by: half * 32)
        let hhBase = view.base.advanced(by: half * 32 + half)
        
        switch task {
        case .skip:
            break
        case .decode16:
            let llView = BlockView(base: llBase, width: half, height: half, stride: 32)
            try blockDecodeDPCM16(decoder: &decoder, block: llView, lastVal: &lastVal)
            
            let hlView = BlockView(base: hlBase, width: half, height: half, stride: 32)
            try blockDecode16V(decoder: &decoder, block: hlView)
            
            let lhView = BlockView(base: lhBase, width: half, height: half, stride: 32)
            try blockDecode16H(decoder: &decoder, block: lhView)
            
            let hhView = BlockView(base: hhBase, width: half, height: half, stride: 32)
            try blockDecode16H(decoder: &decoder, block: hhView)
        case .split8(let tl, let tr, let bl, let br):
            if tl {
                let ll = BlockView(base: llBase, width: 8, height: 8, stride: 32)
                let hl = BlockView(base: hlBase, width: 8, height: 8, stride: 32)
                let lh = BlockView(base: lhBase, width: 8, height: 8, stride: 32)
                let hh = BlockView(base: hhBase, width: 8, height: 8, stride: 32)
                try blockDecodeDPCM8(decoder: &decoder, block: ll, lastVal: &lastVal)
                try blockDecode8H(decoder: &decoder, block: hl)
                try blockDecode8H(decoder: &decoder, block: lh)
                try blockDecode8H(decoder: &decoder, block: hh)
            }
            if tr {
                let ll = BlockView(base: llBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                let hl = BlockView(base: hlBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                let lh = BlockView(base: lhBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                let hh = BlockView(base: hhBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                try blockDecodeDPCM8(decoder: &decoder, block: ll, lastVal: &lastVal)
                try blockDecode8H(decoder: &decoder, block: hl)
                try blockDecode8H(decoder: &decoder, block: lh)
                try blockDecode8H(decoder: &decoder, block: hh)
            }
            if bl {
                let ll = BlockView(base: llBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                let hl = BlockView(base: hlBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                let lh = BlockView(base: lhBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                let hh = BlockView(base: hhBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                try blockDecodeDPCM8(decoder: &decoder, block: ll, lastVal: &lastVal)
                try blockDecode8H(decoder: &decoder, block: hl)
                try blockDecode8H(decoder: &decoder, block: lh)
                try blockDecode8H(decoder: &decoder, block: hh)
            }
            if br {
                let ll = BlockView(base: llBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                let hl = BlockView(base: hlBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                let lh = BlockView(base: lhBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                let hh = BlockView(base: hhBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                try blockDecodeDPCM8(decoder: &decoder, block: ll, lastVal: &lastVal)
                try blockDecode8H(decoder: &decoder, block: hl)
                try blockDecode8H(decoder: &decoder, block: lh)
                try blockDecode8H(decoder: &decoder, block: hh)
            }
        }
    }

    return blocks
}

@inline(__always)
func decodePlaneSubbands32(data: [UInt8], pool: BlockViewPool, blockCount: Int) throws -> [BlockView] {
    return try data.withUnsafeBufferPointer { buf -> [BlockView] in
        return try decodePlaneSubbands32BlockView(buf: buf, pool: pool, blockCount: blockCount)
    }
}

@inline(__always)
fileprivate func decodePlaneSubbands32BlockViewWithParentBlocks(buf: UnsafeBufferPointer<UInt8>, pool: BlockViewPool, blockCount: Int, parentBlocks: [BlockView]) throws -> [BlockView] {
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
    var lastVal: Int16 = 0
    
    for (i, task) in tasks {
        if i < parentBlocks.count {
            let pSubs = getSubbands16(view: parentBlocks[i])
            let parentHL = pSubs.hl
            let parentLH = pSubs.lh
            let parentHH = pSubs.hh
            let view = blocks[i]
            let hlBase = view.base.advanced(by: half)
            let lhBase = view.base.advanced(by: half * 32)
            let hhBase = view.base.advanced(by: half * 32 + half)
            
            switch task {
            case .skip:
                break
            case .decode16:
                let hlView = BlockView(base: hlBase, width: half, height: half, stride: 32)
                try blockDecode16VWithParentBlock(decoder: &decoder, block: hlView, parentBlock: parentHL)
                
                let lhView = BlockView(base: lhBase, width: half, height: half, stride: 32)
                try blockDecode16HWithParentBlock(decoder: &decoder, block: lhView, parentBlock: parentLH)
                
                let hhView = BlockView(base: hhBase, width: half, height: half, stride: 32)
                try blockDecode16HWithParentBlock(decoder: &decoder, block: hhView, parentBlock: parentHH)
            case .split8(let tl, let tr, let bl, let br):
                if tl {
                    let pbHL = BlockView(base: parentHL.base, width: 4, height: 4, stride: parentHL.stride)
                    let pbLH = BlockView(base: parentLH.base, width: 4, height: 4, stride: parentLH.stride)
                    let pbHH = BlockView(base: parentHH.base, width: 4, height: 4, stride: parentHH.stride)
                    let hl = BlockView(base: hlBase, width: 8, height: 8, stride: 32)
                    let lh = BlockView(base: lhBase, width: 8, height: 8, stride: 32)
                    let hh = BlockView(base: hhBase, width: 8, height: 8, stride: 32)
                    try blockDecode8HWithParentBlock(decoder: &decoder, block: hl, parentBlock: pbHL)
                    try blockDecode8HWithParentBlock(decoder: &decoder, block: lh, parentBlock: pbLH)
                    try blockDecode8HWithParentBlock(decoder: &decoder, block: hh, parentBlock: pbHH)
                }
                if tr {
                    let pbHL = BlockView(base: parentHL.base.advanced(by: 4), width: 4, height: 4, stride: parentHL.stride)
                    let pbLH = BlockView(base: parentLH.base.advanced(by: 4), width: 4, height: 4, stride: parentHL.stride)
                    let pbHH = BlockView(base: parentHH.base.advanced(by: 4), width: 4, height: 4, stride: parentHL.stride)
                    let hl = BlockView(base: hlBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    let lh = BlockView(base: lhBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    let hh = BlockView(base: hhBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    try blockDecode8HWithParentBlock(decoder: &decoder, block: hl, parentBlock: pbHL)
                    try blockDecode8HWithParentBlock(decoder: &decoder, block: lh, parentBlock: pbLH)
                    try blockDecode8HWithParentBlock(decoder: &decoder, block: hh, parentBlock: pbHH)
                }
                if bl {
                    let pbHL = BlockView(base: parentHL.base.advanced(by: 4 * parentHL.stride), width: 4, height: 4, stride: parentHL.stride)
                    let pbLH = BlockView(base: parentLH.base.advanced(by: 4 * parentLH.stride), width: 4, height: 4, stride: parentHL.stride)
                    let pbHH = BlockView(base: parentHH.base.advanced(by: 4 * parentHH.stride), width: 4, height: 4, stride: parentHL.stride)
                    let hl = BlockView(base: hlBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    let lh = BlockView(base: lhBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    let hh = BlockView(base: hhBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    try blockDecode8HWithParentBlock(decoder: &decoder, block: hl, parentBlock: pbHL)
                    try blockDecode8HWithParentBlock(decoder: &decoder, block: lh, parentBlock: pbLH)
                    try blockDecode8HWithParentBlock(decoder: &decoder, block: hh, parentBlock: pbHH)
                }
                if br {
                    let pbHL = BlockView(base: parentHL.base.advanced(by: 4 * parentHL.stride + 4), width: 4, height: 4, stride: parentHL.stride)
                    let pbLH = BlockView(base: parentLH.base.advanced(by: 4 * parentLH.stride + 4), width: 4, height: 4, stride: parentHL.stride)
                    let pbHH = BlockView(base: parentHH.base.advanced(by: 4 * parentHH.stride + 4), width: 4, height: 4, stride: parentHL.stride)
                    let hl = BlockView(base: hlBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    let lh = BlockView(base: lhBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    let hh = BlockView(base: hhBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    try blockDecode8HWithParentBlock(decoder: &decoder, block: hl, parentBlock: pbHL)
                    try blockDecode8HWithParentBlock(decoder: &decoder, block: lh, parentBlock: pbLH)
                    try blockDecode8HWithParentBlock(decoder: &decoder, block: hh, parentBlock: pbHH)
                }
            }
        } else {
            let view = blocks[i]
            let llBase = view.base
            let hlBase = view.base.advanced(by: half)
            let lhBase = view.base.advanced(by: half * 32)
            let hhBase = view.base.advanced(by: half * 32 + half)
            
            switch task {
            case .skip:
                break
            case .decode16:
                let llView = BlockView(base: llBase, width: half, height: half, stride: 32)
                try blockDecodeDPCM16(decoder: &decoder, block: llView, lastVal: &lastVal)
                
                let hlView = BlockView(base: hlBase, width: half, height: half, stride: 32)
                try blockDecode16V(decoder: &decoder, block: hlView)
                
                let lhView = BlockView(base: lhBase, width: half, height: half, stride: 32)
                try blockDecode16H(decoder: &decoder, block: lhView)
                
                let hhView = BlockView(base: hhBase, width: half, height: half, stride: 32)
                try blockDecode16H(decoder: &decoder, block: hhView)
            case .split8(let tl, let tr, let bl, let br):
                if tl {
                    let ll = BlockView(base: llBase, width: 8, height: 8, stride: 32)
                    let hl = BlockView(base: hlBase, width: 8, height: 8, stride: 32)
                    let lh = BlockView(base: lhBase, width: 8, height: 8, stride: 32)
                    let hh = BlockView(base: hhBase, width: 8, height: 8, stride: 32)
                    try blockDecodeDPCM8(decoder: &decoder, block: ll, lastVal: &lastVal)
                    try blockDecode8H(decoder: &decoder, block: hl)
                    try blockDecode8H(decoder: &decoder, block: lh)
                    try blockDecode8H(decoder: &decoder, block: hh)
                }
                if tr {
                    let ll = BlockView(base: llBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    let hl = BlockView(base: hlBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    let lh = BlockView(base: lhBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    let hh = BlockView(base: hhBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    try blockDecodeDPCM8(decoder: &decoder, block: ll, lastVal: &lastVal)
                    try blockDecode8H(decoder: &decoder, block: hl)
                    try blockDecode8H(decoder: &decoder, block: lh)
                    try blockDecode8H(decoder: &decoder, block: hh)
                }
                if bl {
                    let ll = BlockView(base: llBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    let hl = BlockView(base: hlBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    let lh = BlockView(base: lhBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    let hh = BlockView(base: hhBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    try blockDecodeDPCM8(decoder: &decoder, block: ll, lastVal: &lastVal)
                    try blockDecode8H(decoder: &decoder, block: hl)
                    try blockDecode8H(decoder: &decoder, block: lh)
                    try blockDecode8H(decoder: &decoder, block: hh)
                }
                if br {
                    let ll = BlockView(base: llBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    let hl = BlockView(base: hlBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    let lh = BlockView(base: lhBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    let hh = BlockView(base: hhBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    try blockDecodeDPCM8(decoder: &decoder, block: ll, lastVal: &lastVal)
                    try blockDecode8H(decoder: &decoder, block: hl)
                    try blockDecode8H(decoder: &decoder, block: lh)
                    try blockDecode8H(decoder: &decoder, block: hh)
                }
            }
        }
    }

    return blocks
}

@inline(__always)
func decodePlaneSubbands32WithParentBlocks(data: [UInt8], pool: BlockViewPool, blockCount: Int, parentBlocks: [BlockView]) throws -> [BlockView] {
    return try data.withUnsafeBufferPointer { buf -> [BlockView] in
        return try decodePlaneSubbands32BlockViewWithParentBlocks(buf: buf, pool: pool, blockCount: blockCount, parentBlocks: parentBlocks)
    }
}

enum DecodeTask16 {
    case skip
    case decode8
    case split4(Bool, Bool, Bool, Bool)
}

@inline(__always)
fileprivate func decodePlaneSubbands16BlockView(buf: UnsafeBufferPointer<UInt8>, pool: BlockViewPool, blockCount: Int) throws -> [BlockView] {
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
        let view = blocks[i]
        let hlBase = view.base.advanced(by: half)
        let lhBase = view.base.advanced(by: half * 16)
        let hhBase = view.base.advanced(by: half * 16 + half)
        
        switch task {
        case .skip:
            break
        case .decode8:
            let hlView = BlockView(base: hlBase, width: half, height: half, stride: 16)
            try blockDecode8V(decoder: &decoder, block: hlView)
            
            let lhView = BlockView(base: lhBase, width: half, height: half, stride: 16)
            try blockDecode8H(decoder: &decoder, block: lhView)
            
            let hhView = BlockView(base: hhBase, width: half, height: half, stride: 16)
            try blockDecode8H(decoder: &decoder, block: hhView)
        case .split4(let tl, let tr, let bl, let br):
            if tl {
                let hl = BlockView(base: hlBase, width: 4, height: 4, stride: 16)
                let lh = BlockView(base: lhBase, width: 4, height: 4, stride: 16)
                let hh = BlockView(base: hhBase, width: 4, height: 4, stride: 16)
                try blockDecode4H(decoder: &decoder, block: hl)
                try blockDecode4H(decoder: &decoder, block: lh)
                try blockDecode4H(decoder: &decoder, block: hh)
            }
            if tr {
                let hl = BlockView(base: hlBase.advanced(by: 4), width: 4, height: 4, stride: 16)
                let lh = BlockView(base: lhBase.advanced(by: 4), width: 4, height: 4, stride: 16)
                let hh = BlockView(base: hhBase.advanced(by: 4), width: 4, height: 4, stride: 16)
                try blockDecode4H(decoder: &decoder, block: hl)
                try blockDecode4H(decoder: &decoder, block: lh)
                try blockDecode4H(decoder: &decoder, block: hh)
            }
            if bl {
                let hl = BlockView(base: hlBase.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
                let lh = BlockView(base: lhBase.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
                let hh = BlockView(base: hhBase.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
                try blockDecode4H(decoder: &decoder, block: hl)
                try blockDecode4H(decoder: &decoder, block: lh)
                try blockDecode4H(decoder: &decoder, block: hh)
            }
            if br {
                let hl = BlockView(base: hlBase.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
                let lh = BlockView(base: lhBase.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
                let hh = BlockView(base: hhBase.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
                try blockDecode4H(decoder: &decoder, block: hl)
                try blockDecode4H(decoder: &decoder, block: lh)
                try blockDecode4H(decoder: &decoder, block: hh)
            }
        }
    }

    return blocks
}

@inline(__always)
func decodePlaneSubbands16(data: [UInt8], pool: BlockViewPool, blockCount: Int) throws -> [BlockView] {
    return try data.withUnsafeBufferPointer { buf -> [BlockView] in
        return try decodePlaneSubbands16BlockView(buf: buf, pool: pool, blockCount: blockCount)
    }
}

@inline(__always)
fileprivate func decodePlaneSubbands16BlockViewWithParentBlocks(buf: UnsafeBufferPointer<UInt8>, pool: BlockViewPool, blockCount: Int, parentBlocks: [BlockView]) throws -> [BlockView] {
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
        if i < parentBlocks.count {
            let pSubs = getSubbands8(view: parentBlocks[i])
            let parentHL = pSubs.hl
            let parentLH = pSubs.lh
            let parentHH = pSubs.hh
            let view = blocks[i]
            let hlBase = view.base.advanced(by: half)
            let lhBase = view.base.advanced(by: half * 16)
            let hhBase = view.base.advanced(by: half * 16 + half)
        
            switch task {
            case .skip:
                break
            case .decode8:
                let hlView = BlockView(base: hlBase, width: half, height: half, stride: 16)
                try blockDecode8VWithParentBlock(decoder: &decoder, block: hlView, parentBlock: parentHL)
                
                let lhView = BlockView(base: lhBase, width: half, height: half, stride: 16)
                try blockDecode8HWithParentBlock(decoder: &decoder, block: lhView, parentBlock: parentLH)
                
                let hhView = BlockView(base: hhBase, width: half, height: half, stride: 16)
                try blockDecode8HWithParentBlock(decoder: &decoder, block: hhView, parentBlock: parentHH)
            case .split4(let tl, let tr, let bl, let br):
                if tl {
                    let pbHL = BlockView(base: parentHL.base, width: 2, height: 2, stride: parentHL.stride)
                    let pbLH = BlockView(base: parentLH.base, width: 2, height: 2, stride: parentLH.stride)
                    let pbHH = BlockView(base: parentHH.base, width: 2, height: 2, stride: parentHH.stride)
                    let hl = BlockView(base: hlBase, width: 4, height: 4, stride: 16)
                    let lh = BlockView(base: lhBase, width: 4, height: 4, stride: 16)
                    let hh = BlockView(base: hhBase, width: 4, height: 4, stride: 16)
                    try blockDecode4HWithParentBlock(decoder: &decoder, block: hl, parentBlock: pbHL)
                    try blockDecode4HWithParentBlock(decoder: &decoder, block: lh, parentBlock: pbLH)
                    try blockDecode4HWithParentBlock(decoder: &decoder, block: hh, parentBlock: pbHH)
                }
                if tr {
                    let pbHL = BlockView(base: parentHL.base.advanced(by: 2), width: 2, height: 2, stride: parentHL.stride)
                    let pbLH = BlockView(base: parentLH.base.advanced(by: 2), width: 2, height: 2, stride: parentLH.stride)
                    let pbHH = BlockView(base: parentHH.base.advanced(by: 2), width: 2, height: 2, stride: parentHH.stride)
                    let hl = BlockView(base: hlBase.advanced(by: 4), width: 4, height: 4, stride: 16)
                    let lh = BlockView(base: lhBase.advanced(by: 4), width: 4, height: 4, stride: 16)
                    let hh = BlockView(base: hhBase.advanced(by: 4), width: 4, height: 4, stride: 16)
                    try blockDecode4HWithParentBlock(decoder: &decoder, block: hl, parentBlock: pbHL)
                    try blockDecode4HWithParentBlock(decoder: &decoder, block: lh, parentBlock: pbLH)
                    try blockDecode4HWithParentBlock(decoder: &decoder, block: hh, parentBlock: pbHH)
                }
                if bl {
                    let pbHL = BlockView(base: parentHL.base.advanced(by: 2 * parentHL.stride), width: 2, height: 2, stride: parentHL.stride)
                    let pbLH = BlockView(base: parentLH.base.advanced(by: 2 * parentLH.stride), width: 2, height: 2, stride: parentHL.stride)
                    let pbHH = BlockView(base: parentHH.base.advanced(by: 2 * parentHH.stride), width: 2, height: 2, stride: parentHL.stride)
                    let hl = BlockView(base: hlBase.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
                    let lh = BlockView(base: lhBase.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
                    let hh = BlockView(base: hhBase.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
                    try blockDecode4HWithParentBlock(decoder: &decoder, block: hl, parentBlock: pbHL)
                    try blockDecode4HWithParentBlock(decoder: &decoder, block: lh, parentBlock: pbLH)
                    try blockDecode4HWithParentBlock(decoder: &decoder, block: hh, parentBlock: pbHH)
                }
                if br {
                    let pbHL = BlockView(base: parentHL.base.advanced(by: 2 * parentHL.stride + 2), width: 2, height: 2, stride: parentHL.stride)
                    let pbLH = BlockView(base: parentLH.base.advanced(by: 2 * parentLH.stride + 2), width: 2, height: 2, stride: parentHL.stride)
                    let pbHH = BlockView(base: parentHH.base.advanced(by: 2 * parentHH.stride + 2), width: 2, height: 2, stride: parentHL.stride)
                    let hl = BlockView(base: hlBase.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
                    let lh = BlockView(base: lhBase.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
                    let hh = BlockView(base: hhBase.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
                    try blockDecode4HWithParentBlock(decoder: &decoder, block: hl, parentBlock: pbHL)
                    try blockDecode4HWithParentBlock(decoder: &decoder, block: lh, parentBlock: pbLH)
                    try blockDecode4HWithParentBlock(decoder: &decoder, block: hh, parentBlock: pbHH)
                }
            }
        } else {
            let view = blocks[i]
            let hlBase = view.base.advanced(by: half)
            let lhBase = view.base.advanced(by: half * 16)
            let hhBase = view.base.advanced(by: half * 16 + half)
            
            switch task {
            case .skip:
                break
            case .decode8:
                let hlView = BlockView(base: hlBase, width: half, height: half, stride: 16)
                try blockDecode8V(decoder: &decoder, block: hlView)
                
                let lhView = BlockView(base: lhBase, width: half, height: half, stride: 16)
                try blockDecode8H(decoder: &decoder, block: lhView)
                
                let hhView = BlockView(base: hhBase, width: half, height: half, stride: 16)
                try blockDecode8H(decoder: &decoder, block: hhView)
            case .split4(let tl, let tr, let bl, let br):
                if tl {
                    let hl = BlockView(base: hlBase, width: 4, height: 4, stride: 16)
                    let lh = BlockView(base: lhBase, width: 4, height: 4, stride: 16)
                    let hh = BlockView(base: hhBase, width: 4, height: 4, stride: 16)
                    try blockDecode4H(decoder: &decoder, block: hl)
                    try blockDecode4H(decoder: &decoder, block: lh)
                    try blockDecode4H(decoder: &decoder, block: hh)
                }
                if tr {
                    let hl = BlockView(base: hlBase.advanced(by: 4), width: 4, height: 4, stride: 16)
                    let lh = BlockView(base: lhBase.advanced(by: 4), width: 4, height: 4, stride: 16)
                    let hh = BlockView(base: hhBase.advanced(by: 4), width: 4, height: 4, stride: 16)
                    try blockDecode4H(decoder: &decoder, block: hl)
                    try blockDecode4H(decoder: &decoder, block: lh)
                    try blockDecode4H(decoder: &decoder, block: hh)
                }
                if bl {
                    let hl = BlockView(base: hlBase.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
                    let lh = BlockView(base: lhBase.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
                    let hh = BlockView(base: hhBase.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
                    try blockDecode4H(decoder: &decoder, block: hl)
                    try blockDecode4H(decoder: &decoder, block: lh)
                    try blockDecode4H(decoder: &decoder, block: hh)
                }
                if br {
                    let hl = BlockView(base: hlBase.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
                    let lh = BlockView(base: lhBase.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
                    let hh = BlockView(base: hhBase.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
                    try blockDecode4H(decoder: &decoder, block: hl)
                    try blockDecode4H(decoder: &decoder, block: lh)
                    try blockDecode4H(decoder: &decoder, block: hh)
                }
            }
        }
    }

    return blocks
}

@inline(__always)
func decodePlaneSubbands16WithParentBlocks(data: [UInt8], pool: BlockViewPool, blockCount: Int, parentBlocks: [BlockView]) throws -> [BlockView] {
    return try data.withUnsafeBufferPointer { buf -> [BlockView] in
        return try decodePlaneSubbands16BlockViewWithParentBlocks(buf: buf, pool: pool, blockCount: blockCount, parentBlocks: parentBlocks)
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
                try blockDecode4H(decoder: &decoder, block: llView)
            }
            
            let hlView = BlockView(base: view.base.advanced(by: half), width: half, height: half, stride: 8)
            try blockDecode4V(decoder: &decoder, block: hlView)
            
            let lhView = BlockView(base: view.base.advanced(by: half * 8), width: half, height: half, stride: 8)
            try blockDecode4H(decoder: &decoder, block: lhView)
            
            let hhView = BlockView(base: view.base.advanced(by: half * 8 + half), width: half, height: half, stride: 8)
            try blockDecode4H(decoder: &decoder, block: hhView)
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

