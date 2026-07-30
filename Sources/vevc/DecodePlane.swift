enum DecodeTask32 {
    case skip
    case decode16
    case split8(Bool, Bool, Bool, Bool)
}

@inline(__always)
func decodePlaneSubbands32(data: ArraySlice<UInt8>, pool: BlockViewPool, blockCount: Int) throws -> [BlockView] {
    return try data.withUnsafeBufferPointer { buf -> [BlockView] in
        guard let base = buf.baseAddress else { return [] }
        let count = buf.count
        
        var brFlags = BypassReader(base: base, count: count)
        var tasks = pool.getInt16(count: blockCount)
        defer { pool.putInt16(tasks) }
        for i in 0..<blockCount {
            let isZero = brFlags.readBit()
            if isZero {
                tasks[i] = 0
            } else {
                let mbType = brFlags.readBit()
                if mbType {
                    let tlZero = brFlags.readBit()
                    if tlZero != true { brFlags.skipBit() }
                    
                    let trZero = brFlags.readBit()
                    if trZero != true { brFlags.skipBit() }
                    
                    let blZero = brFlags.readBit()
                    if blZero != true { brFlags.skipBit() }
                    
                    let brZero = brFlags.readBit()
                    if brZero != true { brFlags.skipBit() }
                    
                    tasks[i] = 2 + (tlZero != true ? 1 : 0) + (trZero != true ? 2 : 0) + (blZero != true ? 4 : 0) + (brZero != true ? 8 : 0)
                } else {
                    tasks[i] = 1
                }
            }
        }
        
        let consumed = brFlags.consumedBytes
        guard consumed <= count else { throw DecodeError.insufficientData }
        
        var blocks = pool.getBlockViewArray(capacity: blockCount)
        for _ in 0..<blockCount {
            blocks.append(pool.get1024())
        }
        
        var decoder = try EntropyDecoder(base: base, count: count, startOffset: consumed)
        let half = 32 / 2
        var lastVal: Int16 = 0
        for i in 0..<blockCount {
            let task = tasks[i]
            let view = blocks[i]
            let llBase = view.base
            let hlBase = view.base.advanced(by: half)
            let lhBase = view.base.advanced(by: half * 32)
            let hhBase = view.base.advanced(by: half * 32 + half)
            
            if task == 0 { } else if task == 1 { // .decode16
                try blockDecodeDPCM16(decoder: &decoder, ptr: llBase, stride: 32, lastVal: &lastVal)
                try blockDecode16V(decoder: &decoder, ptr: hlBase, stride: 32)
                try blockDecode16H(decoder: &decoder, ptr: lhBase, stride: 32)
                try blockDecode16H(decoder: &decoder, ptr: hhBase, stride: 32)
            } else { // .split8
                let v = task - 2
                let tl = (v & 1) != 0
                let tr = (v & 2) != 0
                let bl = (v & 4) != 0
                let br = (v & 8) != 0
                if tl {
                    try blockDecodeDPCM8(decoder: &decoder, ptr: llBase, stride: 32, lastVal: &lastVal)
                    try blockDecode8H(decoder: &decoder, ptr: hlBase, stride: 32)
                    try blockDecode8H(decoder: &decoder, ptr: lhBase, stride: 32)
                    try blockDecode8H(decoder: &decoder, ptr: hhBase, stride: 32)
                }
                if tr {
                    try blockDecodeDPCM8(decoder: &decoder, ptr: llBase.advanced(by: 8), stride: 32, lastVal: &lastVal)
                    try blockDecode8H(decoder: &decoder, ptr: hlBase.advanced(by: 8), stride: 32)
                    try blockDecode8H(decoder: &decoder, ptr: lhBase.advanced(by: 8), stride: 32)
                    try blockDecode8H(decoder: &decoder, ptr: hhBase.advanced(by: 8), stride: 32)
                }
                if bl {
                    try blockDecodeDPCM8(decoder: &decoder, ptr: llBase.advanced(by: 8 * 32), stride: 32, lastVal: &lastVal)
                    try blockDecode8H(decoder: &decoder, ptr: hlBase.advanced(by: 8 * 32), stride: 32)
                    try blockDecode8H(decoder: &decoder, ptr: lhBase.advanced(by: 8 * 32), stride: 32)
                    try blockDecode8H(decoder: &decoder, ptr: hhBase.advanced(by: 8 * 32), stride: 32)
                }
                if br {
                    try blockDecodeDPCM8(decoder: &decoder, ptr: llBase.advanced(by: 8 * 32 + 8), stride: 32, lastVal: &lastVal)
                    try blockDecode8H(decoder: &decoder, ptr: hlBase.advanced(by: 8 * 32 + 8), stride: 32)
                    try blockDecode8H(decoder: &decoder, ptr: lhBase.advanced(by: 8 * 32 + 8), stride: 32)
                    try blockDecode8H(decoder: &decoder, ptr: hhBase.advanced(by: 8 * 32 + 8), stride: 32)
                }
            }
        }

        return blocks
    }
}

@inline(__always)
func decodePlaneSubbands32WithParentBlocks(data: ArraySlice<UInt8>, pool: BlockViewPool, blockCount: Int, parentBlocks: [BlockView]) throws -> [BlockView] {
    return try data.withUnsafeBufferPointer { buf -> [BlockView] in
        guard let base = buf.baseAddress else { return [] }
        let count = buf.count
        
        var brFlags = BypassReader(base: base, count: count)
        var tasks = pool.getInt16(count: blockCount)
        defer { pool.putInt16(tasks) }
        for i in 0..<blockCount {
            let isZero = brFlags.readBit()
            if isZero {
                tasks[i] = 0
            } else {
                let mbType = brFlags.readBit()
                if mbType {
                    let tlZero = brFlags.readBit()
                    if tlZero != true { brFlags.skipBit() }
                    
                    let trZero = brFlags.readBit()
                    if trZero != true { brFlags.skipBit() }
                    
                    let blZero = brFlags.readBit()
                    if blZero != true { brFlags.skipBit() }
                    
                    let brZero = brFlags.readBit()
                    if brZero != true { brFlags.skipBit() }
                    
                    tasks[i] = 2 + (tlZero != true ? 1 : 0) + (trZero != true ? 2 : 0) + (blZero != true ? 4 : 0) + (brZero != true ? 8 : 0)
                } else {
                    tasks[i] = 1
                }
            }
        }
        
        let consumed = brFlags.consumedBytes
        guard consumed <= count else { throw DecodeError.insufficientData }
        
        var blocks = pool.getBlockViewArray(capacity: blockCount)
        for _ in 0..<blockCount {
            blocks.append(pool.get1024())
        }
        
        var decoder = try EntropyDecoder(base: base, count: count, startOffset: consumed)
        let half = 32 / 2
        var lastVal: Int16 = 0
        
        for i in 0..<blockCount {
            let task = tasks[i]
            if i < parentBlocks.count {
                let pSubs = getSubbands16(view: parentBlocks[i])
                let parentHL = pSubs.hl
                let parentLH = pSubs.lh
                let parentHH = pSubs.hh
                let view = blocks[i]
                let hlBase = view.base.advanced(by: half)
                let lhBase = view.base.advanced(by: half * 32)
                let hhBase = view.base.advanced(by: half * 32 + half)
                
                if task == 0 { } else if task == 1 { // .decode16
                    try blockDecode16VWithParentBlock(decoder: &decoder, ptr: hlBase, stride: 32, parentPtr: UnsafePointer(parentHL.base), parentStride: parentHL.stride)
                    try blockDecode16HWithParentBlock(decoder: &decoder, ptr: lhBase, stride: 32, parentPtr: UnsafePointer(parentLH.base), parentStride: parentLH.stride)
                    try blockDecode16HWithParentBlock(decoder: &decoder, ptr: hhBase, stride: 32, parentPtr: UnsafePointer(parentHH.base), parentStride: parentHH.stride)
                } else { // .split8
                let v = task - 2
                let tl = (v & 1) != 0
                let tr = (v & 2) != 0
                let bl = (v & 4) != 0
                let br = (v & 8) != 0
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
                
                if task == 0 { } else if task == 1 { // .decode16
                    let llView = BlockView(base: llBase, width: half, height: half, stride: 32)
                    try blockDecodeDPCM16(decoder: &decoder, block: llView, lastVal: &lastVal)
                    
                    let hlView = BlockView(base: hlBase, width: half, height: half, stride: 32)
                    try blockDecode16V(decoder: &decoder, block: hlView)
                    
                    let lhView = BlockView(base: lhBase, width: half, height: half, stride: 32)
                    try blockDecode16H(decoder: &decoder, block: lhView)
                    
                    let hhView = BlockView(base: hhBase, width: half, height: half, stride: 32)
                    try blockDecode16H(decoder: &decoder, block: hhView)
                } else { // .split8
                let v = task - 2
                let tl = (v & 1) != 0
                let tr = (v & 2) != 0
                let bl = (v & 4) != 0
                let br = (v & 8) != 0
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
}

enum DecodeTask16 {
    case skip
    case decode8
    case split4(Bool, Bool, Bool, Bool)
}

@inline(__always)
func decodePlaneSubbands16(data: ArraySlice<UInt8>, pool: BlockViewPool, blockCount: Int) throws -> [BlockView] {
    return try data.withUnsafeBufferPointer { buf -> [BlockView] in
        guard let base = buf.baseAddress else { return [] }
        let count = buf.count
        var blocks = pool.getBlockViewArray(capacity: blockCount)
        for _ in 0..<blockCount {
            blocks.append(pool.get256())
        }
        
        var brFlags = BypassReader(base: base, count: count)
        var tasks = pool.getInt16(count: blockCount)
        defer { pool.putInt16(tasks) }
        for i in 0..<blockCount {
            if count < brFlags.consumedBytes {
                throw DecodeError.outOfBits
            }
            let isZero = brFlags.readBit()
            if isZero {
                tasks[i] = 0
            } else {
                let mbType = brFlags.readBit()
                if mbType {
                    let tlZero = brFlags.readBit()
                    if tlZero != true { brFlags.skipBit() }
                    
                    let trZero = brFlags.readBit()
                    if trZero != true { brFlags.skipBit() }
                    
                    let blZero = brFlags.readBit()
                    if blZero != true { brFlags.skipBit() }
                    
                    let brZero = brFlags.readBit()
                    if brZero != true { brFlags.skipBit() }
                    
                    tasks[i] = 2 + (tlZero != true ? 1 : 0) + (trZero != true ? 2 : 0) + (blZero != true ? 4 : 0) + (brZero != true ? 8 : 0)
                } else {
                    tasks[i] = 1
                }
            }
        }
        
        let consumed = brFlags.consumedBytes
        guard consumed <= count else { throw DecodeError.insufficientData }
        
        var decoder = try EntropyDecoder(base: base, count: count, startOffset: consumed)
        let half = 16 / 2
        for i in 0..<blockCount {
            let task = tasks[i]
            let view = blocks[i]
            let hlBase = view.base.advanced(by: half)
            let lhBase = view.base.advanced(by: half * 16)
            let hhBase = view.base.advanced(by: half * 16 + half)
            
            if task == 0 { } else if task == 1 { // .decode8
                try blockDecode8V(decoder: &decoder, ptr: hlBase, stride: 16)
                try blockDecode8H(decoder: &decoder, ptr: lhBase, stride: 16)
                try blockDecode8H(decoder: &decoder, ptr: hhBase, stride: 16)
            } else { // .split4
                let v = task - 2
                let tl = (v & 1) != 0
                let tr = (v & 2) != 0
                let bl = (v & 4) != 0
                let br = (v & 8) != 0
                if tl {
                    try blockDecode4H(decoder: &decoder, ptr: hlBase, stride: 16)
                    try blockDecode4H(decoder: &decoder, ptr: lhBase, stride: 16)
                    try blockDecode4H(decoder: &decoder, ptr: hhBase, stride: 16)
                }
                if tr {
                    try blockDecode4H(decoder: &decoder, ptr: hlBase.advanced(by: 4), stride: 16)
                    try blockDecode4H(decoder: &decoder, ptr: lhBase.advanced(by: 4), stride: 16)
                    try blockDecode4H(decoder: &decoder, ptr: hhBase.advanced(by: 4), stride: 16)
                }
                if bl {
                    try blockDecode4H(decoder: &decoder, ptr: hlBase.advanced(by: 4 * 16), stride: 16)
                    try blockDecode4H(decoder: &decoder, ptr: lhBase.advanced(by: 4 * 16), stride: 16)
                    try blockDecode4H(decoder: &decoder, ptr: hhBase.advanced(by: 4 * 16), stride: 16)
                }
                if br {
                    try blockDecode4H(decoder: &decoder, ptr: hlBase.advanced(by: 4 * 16 + 4), stride: 16)
                    try blockDecode4H(decoder: &decoder, ptr: lhBase.advanced(by: 4 * 16 + 4), stride: 16)
                    try blockDecode4H(decoder: &decoder, ptr: hhBase.advanced(by: 4 * 16 + 4), stride: 16)
                }
            }
        }

        return blocks
    }
}

@inline(__always)
func decodePlaneSubbands16WithParentBlocks(data: ArraySlice<UInt8>, pool: BlockViewPool, blockCount: Int, parentBlocks: [BlockView]) throws -> [BlockView] {
    return try data.withUnsafeBufferPointer { buf -> [BlockView] in
        guard let base = buf.baseAddress else { return [] }
        let count = buf.count
        var blocks = pool.getBlockViewArray(capacity: blockCount)
        for _ in 0..<blockCount {
            blocks.append(pool.get256())
        }
        
        var brFlags = BypassReader(base: base, count: count)
        var tasks = pool.getInt16(count: blockCount)
        defer { pool.putInt16(tasks) }
        for i in 0..<blockCount {
            if count < brFlags.consumedBytes {
                throw DecodeError.outOfBits
            }
            let isZero = brFlags.readBit()
            if isZero {
                tasks[i] = 0
            } else {
                let mbType = brFlags.readBit()
                if mbType {
                    let tlZero = brFlags.readBit()
                    if tlZero != true { brFlags.skipBit() }
                    
                    let trZero = brFlags.readBit()
                    if trZero != true { brFlags.skipBit() }
                    
                    let blZero = brFlags.readBit()
                    if blZero != true { brFlags.skipBit() }
                    
                    let brZero = brFlags.readBit()
                    if brZero != true { brFlags.skipBit() }
                    
                    tasks[i] = 2 + (tlZero != true ? 1 : 0) + (trZero != true ? 2 : 0) + (blZero != true ? 4 : 0) + (brZero != true ? 8 : 0)
                } else {
                    tasks[i] = 1
                }
            }
        }
        
        let consumed = brFlags.consumedBytes
        guard consumed <= count else { throw DecodeError.insufficientData }
        
        var decoder = try EntropyDecoder(base: base, count: count, startOffset: consumed)
        let half = 16 / 2
        for i in 0..<blockCount {
            let task = tasks[i]
            if i < parentBlocks.count {
                let pSubs = getSubbands8(view: parentBlocks[i])
                let parentHL = pSubs.hl
                let parentLH = pSubs.lh
                let parentHH = pSubs.hh
                let view = blocks[i]
                let hlBase = view.base.advanced(by: half)
                let lhBase = view.base.advanced(by: half * 16)
                let hhBase = view.base.advanced(by: half * 16 + half)
            
                if task == 0 { } else if task == 1 { // .decode8
                    try blockDecode8VWithParentBlock(decoder: &decoder, ptr: hlBase, stride: 16, parentPtr: UnsafePointer(parentHL.base), parentStride: parentHL.stride)
                    try blockDecode8HWithParentBlock(decoder: &decoder, ptr: lhBase, stride: 16, parentPtr: UnsafePointer(parentLH.base), parentStride: parentLH.stride)
                    try blockDecode8HWithParentBlock(decoder: &decoder, ptr: hhBase, stride: 16, parentPtr: UnsafePointer(parentHH.base), parentStride: parentHH.stride)
                } else { // .split4
                let v = task - 2
                let tl = (v & 1) != 0
                let tr = (v & 2) != 0
                let bl = (v & 4) != 0
                let br = (v & 8) != 0
                    if tl {
                        try blockDecode4HWithParentBlock(decoder: &decoder, ptr: hlBase, stride: 16, parentPtr: UnsafePointer(parentHL.base), parentStride: parentHL.stride)
                        try blockDecode4HWithParentBlock(decoder: &decoder, ptr: lhBase, stride: 16, parentPtr: UnsafePointer(parentLH.base), parentStride: parentLH.stride)
                        try blockDecode4HWithParentBlock(decoder: &decoder, ptr: hhBase, stride: 16, parentPtr: UnsafePointer(parentHH.base), parentStride: parentHH.stride)
                    }
                    if tr {
                        try blockDecode4HWithParentBlock(decoder: &decoder, ptr: hlBase.advanced(by: 4), stride: 16, parentPtr: UnsafePointer(parentHL.base.advanced(by: 2)), parentStride: parentHL.stride)
                        try blockDecode4HWithParentBlock(decoder: &decoder, ptr: lhBase.advanced(by: 4), stride: 16, parentPtr: UnsafePointer(parentLH.base.advanced(by: 2)), parentStride: parentLH.stride)
                        try blockDecode4HWithParentBlock(decoder: &decoder, ptr: hhBase.advanced(by: 4), stride: 16, parentPtr: UnsafePointer(parentHH.base.advanced(by: 2)), parentStride: parentHH.stride)
                    }
                    if bl {
                        try blockDecode4HWithParentBlock(decoder: &decoder, ptr: hlBase.advanced(by: 4 * 16), stride: 16, parentPtr: UnsafePointer(parentHL.base.advanced(by: 2 * parentHL.stride)), parentStride: parentHL.stride)
                        try blockDecode4HWithParentBlock(decoder: &decoder, ptr: lhBase.advanced(by: 4 * 16), stride: 16, parentPtr: UnsafePointer(parentLH.base.advanced(by: 2 * parentLH.stride)), parentStride: parentLH.stride)
                        try blockDecode4HWithParentBlock(decoder: &decoder, ptr: hhBase.advanced(by: 4 * 16), stride: 16, parentPtr: UnsafePointer(parentHH.base.advanced(by: 2 * parentHH.stride)), parentStride: parentHH.stride)
                    }
                    if br {
                        try blockDecode4HWithParentBlock(decoder: &decoder, ptr: hlBase.advanced(by: 4 * 16 + 4), stride: 16, parentPtr: UnsafePointer(parentHL.base.advanced(by: 2 * parentHL.stride + 2)), parentStride: parentHL.stride)
                        try blockDecode4HWithParentBlock(decoder: &decoder, ptr: lhBase.advanced(by: 4 * 16 + 4), stride: 16, parentPtr: UnsafePointer(parentLH.base.advanced(by: 2 * parentLH.stride + 2)), parentStride: parentLH.stride)
                        try blockDecode4HWithParentBlock(decoder: &decoder, ptr: hhBase.advanced(by: 4 * 16 + 4), stride: 16, parentPtr: UnsafePointer(parentHH.base.advanced(by: 2 * parentHH.stride + 2)), parentStride: parentHH.stride)
                    }
                }
            } else {
                let view = blocks[i]
                let hlBase = view.base.advanced(by: half)
                let lhBase = view.base.advanced(by: half * 16)
                let hhBase = view.base.advanced(by: half * 16 + half)
                
                if task == 0 { } else if task == 1 { // .decode8
                    let hlView = BlockView(base: hlBase, width: half, height: half, stride: 16)
                    try blockDecode8V(decoder: &decoder, block: hlView)
                    
                    let lhView = BlockView(base: lhBase, width: half, height: half, stride: 16)
                    try blockDecode8H(decoder: &decoder, block: lhView)
                    
                    let hhView = BlockView(base: hhBase, width: half, height: half, stride: 16)
                    try blockDecode8H(decoder: &decoder, block: hhView)
                } else { // .split4
                let v = task - 2
                let tl = (v & 1) != 0
                let tr = (v & 2) != 0
                let bl = (v & 4) != 0
                let br = (v & 8) != 0
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
}

@inline(__always)
func decodePlaneBaseSubbands8(data: ArraySlice<UInt8>, pool: BlockViewPool, blockCount: Int, isIFrame: Bool) throws -> [BlockView] {
    return try data.withUnsafeBufferPointer { buf -> [BlockView] in
        guard let base = buf.baseAddress else { return [] }
        let count = buf.count
        var blocks = pool.getBlockViewArray(capacity: blockCount)
        for _ in 0..<blockCount {
            blocks.append(pool.get64())
        }
        
        var brFlags = BypassReader(base: base, count: count)
        var nonZeroIndices: [Int] = []
        nonZeroIndices.reserveCapacity(blockCount)
        for i in 0..<blockCount {
            let isZero = brFlags.readBit()
            brFlags.skipBit()
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
                let base = view.base
                if isIFrame {
                    try blockDecodeDPCM4(decoder: &decoder, ptr: base, stride: 8, lastVal: &lastVal)
                } else {
                    try blockDecode4H(decoder: &decoder, ptr: base, stride: 8)
                }
                
                try blockDecode4V(decoder: &decoder, ptr: base.advanced(by: half), stride: 8)
                try blockDecode4H(decoder: &decoder, ptr: base.advanced(by: half * 8), stride: 8)
                try blockDecode4H(decoder: &decoder, ptr: base.advanced(by: half * 8 + half), stride: 8)
            } else {
                if isIFrame { lastVal = 0 }
            }
        }

        return blocks
    }
}

// MARK: - Convenience overloads for [UInt8]
@inline(__always)
func decodePlaneSubbands32(data: [UInt8], pool: BlockViewPool, blockCount: Int) throws -> [BlockView] {
    try decodePlaneSubbands32(data: data[...], pool: pool, blockCount: blockCount)
}

@inline(__always)
func decodePlaneSubbands32WithParentBlocks(data: [UInt8], pool: BlockViewPool, blockCount: Int, parentBlocks: [BlockView]) throws -> [BlockView] {
    try decodePlaneSubbands32WithParentBlocks(data: data[...], pool: pool, blockCount: blockCount, parentBlocks: parentBlocks)
}

@inline(__always)
func decodePlaneSubbands16(data: [UInt8], pool: BlockViewPool, blockCount: Int) throws -> [BlockView] {
    try decodePlaneSubbands16(data: data[...], pool: pool, blockCount: blockCount)
}

@inline(__always)
func decodePlaneSubbands16WithParentBlocks(data: [UInt8], pool: BlockViewPool, blockCount: Int, parentBlocks: [BlockView]) throws -> [BlockView] {
    try decodePlaneSubbands16WithParentBlocks(data: data[...], pool: pool, blockCount: blockCount, parentBlocks: parentBlocks)
}

@inline(__always)
func decodePlaneBaseSubbands8(data: [UInt8], pool: BlockViewPool, blockCount: Int, isIFrame: Bool) throws -> [BlockView] {
    try decodePlaneBaseSubbands8(data: data[...], pool: pool, blockCount: blockCount, isIFrame: isIFrame)
}
