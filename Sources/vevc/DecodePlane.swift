enum DecodeTask32 {
    case skip
    case decode16
    case split8(Bool, Bool, Bool, Bool)
}

@inline(__always)
func decodePlaneSubbands32(data: [UInt8], blockCount: Int, parentImage: Image16?, dx: Int, planeType: Int) throws -> [Block2D] {
    var blocks: [Block2D] = []
    blocks.reserveCapacity(blockCount)
    for _ in 0..<blockCount {
        blocks.append(Block2D(width: 32, height: 32))
    }
    
    var brFlags = BypassReader(data: data)
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
                    brZero != true,
                )))
            } else {
                tasks.append((i, .decode16))
            }
        }
    }
    
    let consumed = brFlags.consumedBytes
    guard consumed <= data.count else { throw DecodeError.insufficientData }
    let dataSlice = Array(data[consumed...])
    
    var decoder = try EntropyDecoder(data: dataSlice)
    
    let half = 32 / 2
    
    for (i, task) in tasks {
        let colCount = (dx + 32 - 1) / 32
        let r = i / colCount
        let c = i % colCount
        let pbX = c * 16
        let pbY = r * 16
        
        var parentBlock2D: Block2D? = nil
        if let pImg = parentImage {
            switch planeType {
            case 0: parentBlock2D = pImg.getY(x: pbX, y: pbY, size: 16)
            case 1: parentBlock2D = pImg.getCb(x: pbX, y: pbY, size: 16)
            default: parentBlock2D = pImg.getCr(x: pbX, y: pbY, size: 16)
            }
        }
        
        let decodeAction = { (parentBlock: BlockView?) throws in
            try blocks[i].withView { view in
                let hlBase = view.base.advanced(by: half)
                let lhBase = view.base.advanced(by: half * 32)
                let hhBase = view.base.advanced(by: half * 32 + half)
                
                switch task {
                case .skip:
                    break
                case .decode16:
                    var hlView = BlockView(base: hlBase, width: half, height: half, stride: 32)
                    try blockDecode16(decoder: &decoder, block: &hlView, parentBlock: parentBlock)
                    
                    var lhView = BlockView(base: lhBase, width: half, height: half, stride: 32)
                    try blockDecode16(decoder: &decoder, block: &lhView, parentBlock: parentBlock)
                    
                    var hhView = BlockView(base: hhBase, width: half, height: half, stride: 32)
                    try blockDecode16(decoder: &decoder, block: &hhView, parentBlock: parentBlock)
                case .split8(let tl, let tr, let bl, let br):
                    if tl {
                        let pb = parentBlock.map { BlockView(base: $0.base, width: 8, height: 8, stride: 16) }
                        var hl = BlockView(base: hlBase, width: 8, height: 8, stride: 32)
                        var lh = BlockView(base: lhBase, width: 8, height: 8, stride: 32)
                        var hh = BlockView(base: hhBase, width: 8, height: 8, stride: 32)
                        try blockDecode8(decoder: &decoder, block: &hl, parentBlock: pb)
                        try blockDecode8(decoder: &decoder, block: &lh, parentBlock: pb)
                        try blockDecode8(decoder: &decoder, block: &hh, parentBlock: pb)
                    }
                    if tr {
                        let pb = parentBlock.map { BlockView(base: $0.base.advanced(by: 8), width: 8, height: 8, stride: 16) }
                        var hl = BlockView(base: hlBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                        var lh = BlockView(base: lhBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                        var hh = BlockView(base: hhBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                        try blockDecode8(decoder: &decoder, block: &hl, parentBlock: pb)
                        try blockDecode8(decoder: &decoder, block: &lh, parentBlock: pb)
                        try blockDecode8(decoder: &decoder, block: &hh, parentBlock: pb)
                    }
                    if bl {
                        let pb = parentBlock.map { BlockView(base: $0.base.advanced(by: 8 * 16), width: 8, height: 8, stride: 16) }
                        var hl = BlockView(base: hlBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                        var lh = BlockView(base: lhBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                        var hh = BlockView(base: hhBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                        try blockDecode8(decoder: &decoder, block: &hl, parentBlock: pb)
                        try blockDecode8(decoder: &decoder, block: &lh, parentBlock: pb)
                        try blockDecode8(decoder: &decoder, block: &hh, parentBlock: pb)
                    }
                    if br {
                        let pb = parentBlock.map { BlockView(base: $0.base.advanced(by: 8 * 16 + 8), width: 8, height: 8, stride: 16) }
                        var hl = BlockView(base: hlBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                        var lh = BlockView(base: lhBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                        var hh = BlockView(base: hhBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                        try blockDecode8(decoder: &decoder, block: &hl, parentBlock: pb)
                        try blockDecode8(decoder: &decoder, block: &lh, parentBlock: pb)
                        try blockDecode8(decoder: &decoder, block: &hh, parentBlock: pb)
                    }
                }
            }
        }
        
        if var pb2d = parentBlock2D {
            try pb2d.withView { pb in try decodeAction(pb) }
        } else {
            try decodeAction(nil)
        }
    }

    return blocks
}

enum DecodeTask16 {
    case skip
    case decode8
    case split4(Bool, Bool, Bool, Bool)
}

@inline(__always)
func decodePlaneSubbands16(data: [UInt8], blockCount: Int, parentImage: Image16?, dx: Int, planeType: Int) throws -> [Block2D] {
    var blocks: [Block2D] = []
    blocks.reserveCapacity(blockCount)
    for _ in 0..<blockCount {
        blocks.append(Block2D(width: 16, height: 16))
    }
    
    var brFlags = BypassReader(data: data)
    var tasks: [(Int, DecodeTask16)] = []
    tasks.reserveCapacity(blockCount)
    for i in 0..<blockCount {
        if brFlags.consumedBytes > data.count {
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
    guard consumed <= data.count else { throw DecodeError.insufficientData }
    let dataSlice = Array(data[consumed...])
    
    var decoder = try EntropyDecoder(data: dataSlice)
    
    let half = 16 / 2

    for (i, task) in tasks {
        let colCount = (dx + 16 - 1) / 16
        let r = i / colCount
        let c = i % colCount
        let pbX = c * 8
        let pbY = r * 8
        
        var parentBlock2D: Block2D? = nil
        if let pImg = parentImage {
            switch planeType {
            case 0: parentBlock2D = pImg.getY(x: pbX, y: pbY, size: 8)
            case 1: parentBlock2D = pImg.getCb(x: pbX, y: pbY, size: 8)
            default: parentBlock2D = pImg.getCr(x: pbX, y: pbY, size: 8)
            }
        }
        
        let decodeAction = { (parentBlock: BlockView?) throws in
            try blocks[i].withView { view in
                let hlBase = view.base.advanced(by: half)
                let lhBase = view.base.advanced(by: half * 16)
                let hhBase = view.base.advanced(by: half * 16 + half)
            
            switch task {
            case .skip:
                break
            case .decode8:
                var hlView = BlockView(base: hlBase, width: half, height: half, stride: 16)
                try blockDecode8(decoder: &decoder, block: &hlView, parentBlock: parentBlock)
                
                var lhView = BlockView(base: lhBase, width: half, height: half, stride: 16)
                try blockDecode8(decoder: &decoder, block: &lhView, parentBlock: parentBlock)
                
                var hhView = BlockView(base: hhBase, width: half, height: half, stride: 16)
                try blockDecode8(decoder: &decoder, block: &hhView, parentBlock: parentBlock)
            case .split4(let tl, let tr, let bl, let br):
                if tl {
                    let pb = parentBlock.map { BlockView(base: $0.base, width: 4, height: 4, stride: 8) }
                    var hl = BlockView(base: hlBase, width: 4, height: 4, stride: 16)
                    var lh = BlockView(base: lhBase, width: 4, height: 4, stride: 16)
                    var hh = BlockView(base: hhBase, width: 4, height: 4, stride: 16)
                    try blockDecode4(decoder: &decoder, block: &hl, parentBlock: pb)
                    try blockDecode4(decoder: &decoder, block: &lh, parentBlock: pb)
                    try blockDecode4(decoder: &decoder, block: &hh, parentBlock: pb)
                }
                if tr {
                    let pb = parentBlock.map { BlockView(base: $0.base.advanced(by: 4), width: 4, height: 4, stride: 8) }
                    var hl = BlockView(base: hlBase.advanced(by: 4), width: 4, height: 4, stride: 16)
                    var lh = BlockView(base: lhBase.advanced(by: 4), width: 4, height: 4, stride: 16)
                    var hh = BlockView(base: hhBase.advanced(by: 4), width: 4, height: 4, stride: 16)
                    try blockDecode4(decoder: &decoder, block: &hl, parentBlock: pb)
                    try blockDecode4(decoder: &decoder, block: &lh, parentBlock: pb)
                    try blockDecode4(decoder: &decoder, block: &hh, parentBlock: pb)
                }
                if bl {
                    let pb = parentBlock.map { BlockView(base: $0.base.advanced(by: 4 * 8), width: 4, height: 4, stride: 8) }
                    var hl = BlockView(base: hlBase.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
                    var lh = BlockView(base: lhBase.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
                    var hh = BlockView(base: hhBase.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
                    try blockDecode4(decoder: &decoder, block: &hl, parentBlock: pb)
                    try blockDecode4(decoder: &decoder, block: &lh, parentBlock: pb)
                    try blockDecode4(decoder: &decoder, block: &hh, parentBlock: pb)
                }
                if br {
                    let pb = parentBlock.map { BlockView(base: $0.base.advanced(by: 4 * 8 + 4), width: 4, height: 4, stride: 8) }
                    var hl = BlockView(base: hlBase.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
                    var lh = BlockView(base: lhBase.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
                    var hh = BlockView(base: hhBase.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
                    try blockDecode4(decoder: &decoder, block: &hl, parentBlock: pb)
                    try blockDecode4(decoder: &decoder, block: &lh, parentBlock: pb)
                    try blockDecode4(decoder: &decoder, block: &hh, parentBlock: pb)
                }
            }
        }
        }
        
        if var pb2d = parentBlock2D {
            try pb2d.withView { pb in try decodeAction(pb) }
        } else {
            try decodeAction(nil)
        }
    }

    return blocks
}

@inline(__always)
func decodePlaneSubbands8(data: [UInt8], blockCount: Int, parentImage: Image16?, dx: Int, planeType: Int) throws -> [Block2D] {
    var blocks: [Block2D] = []
    blocks.reserveCapacity(blockCount)
    for _ in 0..<blockCount {
        blocks.append(Block2D(width: 8, height: 8))
    }
    
    var brFlags = BypassReader(data: data)
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
    guard consumed <= data.count else { throw DecodeError.insufficientData }
    let dataSlice = Array(data[consumed...])
    
    var decoder = try EntropyDecoder(data: dataSlice)
    
    let half = 8 / 2

    for i in nonZeroIndices {
        let colCount = (dx + 8 - 1) / 8
        let r = i / colCount
        let c = i % colCount
        let pbX = c * 4
        let pbY = r * 4
        
        var parentBlock2D: Block2D? = nil
        if let pImg = parentImage {
            switch planeType {
            case 0: parentBlock2D = pImg.getY(x: pbX, y: pbY, size: 4)
            case 1: parentBlock2D = pImg.getCb(x: pbX, y: pbY, size: 4)
            default: parentBlock2D = pImg.getCr(x: pbX, y: pbY, size: 4)
            }
        }
        
        let decodeAction = { (parentBlock: BlockView?) throws in
            try blocks[i].withView { view in
                var hlView = BlockView(base: view.base.advanced(by: half), width: half, height: half, stride: 8)
                try blockDecode4(decoder: &decoder, block: &hlView, parentBlock: parentBlock)
                
                var lhView = BlockView(base: view.base.advanced(by: half * 8), width: half, height: half, stride: 8)
                try blockDecode4(decoder: &decoder, block: &lhView, parentBlock: parentBlock)
                
                var hhView = BlockView(base: view.base.advanced(by: half * 8 + half), width: half, height: half, stride: 8)
                try blockDecode4(decoder: &decoder, block: &hhView, parentBlock: parentBlock)
            }
        }
        
        if var pb2d = parentBlock2D {
            try pb2d.withView { pb in try decodeAction(pb) }
        } else {
            try decodeAction(nil)
        }
    }

    return blocks
}

@inline(__always)
func decodePlaneBaseSubbands8(data: [UInt8], blockCount: Int) throws -> [Block2D] {
    var blocks: [Block2D] = []
    blocks.reserveCapacity(blockCount)
    for _ in 0..<blockCount {
        blocks.append(Block2D(width: 8, height: 8))
    }
    
    var brFlags = BypassReader(data: data)
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
    guard consumed <= data.count else { throw DecodeError.insufficientData }
    let dataSlice = Array(data[consumed...])
    
    var decoder = try EntropyDecoder(data: dataSlice)
    
    let half = 8 / 2

    var lastVal: Int16 = 0
    var nzCur = 0
    let nzCount = nonZeroIndices.count
    for i in 0..<blockCount {
        if nzCur < nzCount && nonZeroIndices[nzCur] == i {
            nzCur += 1
            try blocks[i].withView { view in
                var llView = BlockView(base: view.base, width: half, height: half, stride: 8)
                try blockDecodeDPCM4(decoder: &decoder, block: &llView, lastVal: &lastVal)
                
                var hlView = BlockView(base: view.base.advanced(by: half), width: half, height: half, stride: 8)
                try blockDecode4(decoder: &decoder, block: &hlView, parentBlock: nil)
                
                var lhView = BlockView(base: view.base.advanced(by: half * 8), width: half, height: half, stride: 8)
                try blockDecode4(decoder: &decoder, block: &lhView, parentBlock: nil)
                
                var hhView = BlockView(base: view.base.advanced(by: half * 8 + half), width: half, height: half, stride: 8)
                try blockDecode4(decoder: &decoder, block: &hhView, parentBlock: nil)
            }
        } else {
            lastVal = 0
        }
    }

    return blocks
}

enum DecodeTaskBase32 {
    case skip
    case decode16
    case split8(Bool, Bool, Bool, Bool)
}

@inline(__always)
func decodePlaneBaseSubbands32(data: [UInt8], blockCount: Int) throws -> [Block2D] {
    var blocks: [Block2D] = []
    blocks.reserveCapacity(blockCount)
    for _ in 0..<blockCount {
        blocks.append(Block2D(width: 32, height: 32))
    }
    
    var brFlags = BypassReader(data: data)
    var tasks: [(Int, DecodeTaskBase32)] = []
    tasks.reserveCapacity(blockCount)
    for i in 0..<blockCount {
        if brFlags.consumedBytes > data.count {
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
    guard consumed <= data.count else { throw DecodeError.insufficientData }
    let dataSlice = Array(data[consumed...])
    
    var decoder = try EntropyDecoder(data: dataSlice)
    
    let half = 32 / 2

    var lastVal: Int16 = 0
    for (i, task) in tasks {
        try blocks[i].withView { view in
            let llBase = view.base
            let hlBase = view.base.advanced(by: half)
            let lhBase = view.base.advanced(by: half * 32)
            let hhBase = view.base.advanced(by: half * 32 + half)
            
            switch task {
            case .skip:
                lastVal = 0
            case .decode16:
                var llView = BlockView(base: llBase, width: half, height: half, stride: 32)
                try blockDecodeDPCM16(decoder: &decoder, block: &llView, lastVal: &lastVal)
                
                var hlView = BlockView(base: hlBase, width: half, height: half, stride: 32)
                try blockDecode16(decoder: &decoder, block: &hlView, parentBlock: nil)
                
                var lhView = BlockView(base: lhBase, width: half, height: half, stride: 32)
                try blockDecode16(decoder: &decoder, block: &lhView, parentBlock: nil)
                
                var hhView = BlockView(base: hhBase, width: half, height: half, stride: 32)
                try blockDecode16(decoder: &decoder, block: &hhView, parentBlock: nil)
                
            case .split8(let tl, let tr, let bl, let br):
                if tl {
                    var ll = BlockView(base: llBase, width: 8, height: 8, stride: 32)
                    var hl = BlockView(base: hlBase, width: 8, height: 8, stride: 32)
                    var lh = BlockView(base: lhBase, width: 8, height: 8, stride: 32)
                    var hh = BlockView(base: hhBase, width: 8, height: 8, stride: 32)
                    try blockDecodeDPCM8(decoder: &decoder, block: &ll, lastVal: &lastVal)
                    try blockDecode8(decoder: &decoder, block: &hl, parentBlock: nil)
                    try blockDecode8(decoder: &decoder, block: &lh, parentBlock: nil)
                    try blockDecode8(decoder: &decoder, block: &hh, parentBlock: nil)
                }
                if tr {
                    var ll = BlockView(base: llBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    var hl = BlockView(base: hlBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    var lh = BlockView(base: lhBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    var hh = BlockView(base: hhBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    try blockDecodeDPCM8(decoder: &decoder, block: &ll, lastVal: &lastVal)
                    try blockDecode8(decoder: &decoder, block: &hl, parentBlock: nil)
                    try blockDecode8(decoder: &decoder, block: &lh, parentBlock: nil)
                    try blockDecode8(decoder: &decoder, block: &hh, parentBlock: nil)
                }
                if bl {
                    var ll = BlockView(base: llBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    var hl = BlockView(base: hlBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    var lh = BlockView(base: lhBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    var hh = BlockView(base: hhBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    try blockDecodeDPCM8(decoder: &decoder, block: &ll, lastVal: &lastVal)
                    try blockDecode8(decoder: &decoder, block: &hl, parentBlock: nil)
                    try blockDecode8(decoder: &decoder, block: &lh, parentBlock: nil)
                    try blockDecode8(decoder: &decoder, block: &hh, parentBlock: nil)
                }
                if br {
                    var ll = BlockView(base: llBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    var hl = BlockView(base: hlBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    var lh = BlockView(base: lhBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    var hh = BlockView(base: hhBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    try blockDecodeDPCM8(decoder: &decoder, block: &ll, lastVal: &lastVal)
                    try blockDecode8(decoder: &decoder, block: &hl, parentBlock: nil)
                    try blockDecode8(decoder: &decoder, block: &lh, parentBlock: nil)
                    try blockDecode8(decoder: &decoder, block: &hh, parentBlock: nil)
                }
            }
        }
    }

    return blocks
}

@inline(__always)
func decodeCascadedPlaneSubbands32(data: [UInt8], blocks: inout [Block2D]) throws {
    var bwFlags = BypassReader(data: data)
    var tasks: [(Int, Bool)] = []
    tasks.reserveCapacity(blocks.count)
    
    for i in blocks.indices {
        let isZero = bwFlags.readBit()
        if isZero {
            blocks[i].withView { $0.clearAll() }
            tasks.append((i, true))
        } else {
            tasks.append((i, false))
        }
    }
    
    let consumed = bwFlags.consumedBytes
    guard consumed <= data.count else { throw DecodeError.insufficientData }
    let entropyData = Array(data[consumed...])
    var decoder = try EntropyDecoder(data: entropyData)
    var lastVal: Int16 = 0
    
    for (i, skip) in tasks {
        if skip {
            lastVal = 0
            continue
        }
        
        try blocks[i].withView { view in
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
            
            var m_ll3 = ll3, m_hl3 = hl3, m_lh3 = lh3, m_hh3 = hh3
            try blockDecodeDPCM4(decoder: &decoder, block: &m_ll3, lastVal: &lastVal)
            try blockDecode4(decoder: &decoder, block: &m_hl3, parentBlock: nil)
            try blockDecode4(decoder: &decoder, block: &m_lh3, parentBlock: nil)
            try blockDecode4(decoder: &decoder, block: &m_hh3, parentBlock: nil)
            
            var m_hl2 = hl2, m_lh2 = lh2, m_hh2 = hh2
            try blockDecode8(decoder: &decoder, block: &m_hl2, parentBlock: nil)
            try blockDecode8(decoder: &decoder, block: &m_lh2, parentBlock: nil)
            try blockDecode8(decoder: &decoder, block: &m_hh2, parentBlock: nil)
            
            var m_hl1 = hl1, m_lh1 = lh1, m_hh1 = hh1
            try blockDecode16(decoder: &decoder, block: &m_hl1, parentBlock: nil)
            try blockDecode16(decoder: &decoder, block: &m_lh1, parentBlock: nil)
            try blockDecode16(decoder: &decoder, block: &m_hh1, parentBlock: nil)
        }
    }
}