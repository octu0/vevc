enum DecodeTask32 {
    case skip
    case decode16
    case split8(Bool, Bool, Bool, Bool)
}

@inline(__always)
func decodePlaneSubbands32(data: [UInt8], blockCount: Int) throws -> [Block2D] {
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
        try blocks[i].withView { view in
            let hlBase = view.base.advanced(by: half)
            let lhBase = view.base.advanced(by: half * 32)
            let hhBase = view.base.advanced(by: half * 32 + half)
            
            switch task {
            case .skip:
                break
            case .decode16:
                var hlView = BlockView(base: hlBase, width: half, height: half, stride: 32)
                try blockDecode16(decoder: &decoder, block: &hlView)
                
                var lhView = BlockView(base: lhBase, width: half, height: half, stride: 32)
                try blockDecode16(decoder: &decoder, block: &lhView)
                
                var hhView = BlockView(base: hhBase, width: half, height: half, stride: 32)
                try blockDecode16(decoder: &decoder, block: &hhView)
            case .split8(let tl, let tr, let bl, let br):
                if tl {
                    var hl = BlockView(base: hlBase, width: 8, height: 8, stride: 32)
                    var lh = BlockView(base: lhBase, width: 8, height: 8, stride: 32)
                    var hh = BlockView(base: hhBase, width: 8, height: 8, stride: 32)
                    try blockDecode8(decoder: &decoder, block: &hl)
                    try blockDecode8(decoder: &decoder, block: &lh)
                    try blockDecode8(decoder: &decoder, block: &hh)
                }
                if tr {
                    var hl = BlockView(base: hlBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    var lh = BlockView(base: lhBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    var hh = BlockView(base: hhBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    try blockDecode8(decoder: &decoder, block: &hl)
                    try blockDecode8(decoder: &decoder, block: &lh)
                    try blockDecode8(decoder: &decoder, block: &hh)
                }
                if bl {
                    var hl = BlockView(base: hlBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    var lh = BlockView(base: lhBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    var hh = BlockView(base: hhBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    try blockDecode8(decoder: &decoder, block: &hl)
                    try blockDecode8(decoder: &decoder, block: &lh)
                    try blockDecode8(decoder: &decoder, block: &hh)
                }
                if br {
                    var hl = BlockView(base: hlBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    var lh = BlockView(base: lhBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    var hh = BlockView(base: hhBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    try blockDecode8(decoder: &decoder, block: &hl)
                    try blockDecode8(decoder: &decoder, block: &lh)
                    try blockDecode8(decoder: &decoder, block: &hh)
                }
            }
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
func decodePlaneSubbands16(data: [UInt8], blockCount: Int) throws -> [Block2D] {
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
        try blocks[i].withView { view in
            let hlBase = view.base.advanced(by: half)
            let lhBase = view.base.advanced(by: half * 16)
            let hhBase = view.base.advanced(by: half * 16 + half)
            
            switch task {
            case .skip:
                break
            case .decode8:
                var hlView = BlockView(base: hlBase, width: half, height: half, stride: 16)
                try blockDecode8(decoder: &decoder, block: &hlView)
                
                var lhView = BlockView(base: lhBase, width: half, height: half, stride: 16)
                try blockDecode8(decoder: &decoder, block: &lhView)
                
                var hhView = BlockView(base: hhBase, width: half, height: half, stride: 16)
                try blockDecode8(decoder: &decoder, block: &hhView)
            case .split4(let tl, let tr, let bl, let br):
                if tl {
                    var hl = BlockView(base: hlBase, width: 4, height: 4, stride: 16)
                    var lh = BlockView(base: lhBase, width: 4, height: 4, stride: 16)
                    var hh = BlockView(base: hhBase, width: 4, height: 4, stride: 16)
                    try blockDecode4(decoder: &decoder, block: &hl)
                    try blockDecode4(decoder: &decoder, block: &lh)
                    try blockDecode4(decoder: &decoder, block: &hh)
                }
                if tr {
                    var hl = BlockView(base: hlBase.advanced(by: 4), width: 4, height: 4, stride: 16)
                    var lh = BlockView(base: lhBase.advanced(by: 4), width: 4, height: 4, stride: 16)
                    var hh = BlockView(base: hhBase.advanced(by: 4), width: 4, height: 4, stride: 16)
                    try blockDecode4(decoder: &decoder, block: &hl)
                    try blockDecode4(decoder: &decoder, block: &lh)
                    try blockDecode4(decoder: &decoder, block: &hh)
                }
                if bl {
                    var hl = BlockView(base: hlBase.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
                    var lh = BlockView(base: lhBase.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
                    var hh = BlockView(base: hhBase.advanced(by: 4 * 16), width: 4, height: 4, stride: 16)
                    try blockDecode4(decoder: &decoder, block: &hl)
                    try blockDecode4(decoder: &decoder, block: &lh)
                    try blockDecode4(decoder: &decoder, block: &hh)
                }
                if br {
                    var hl = BlockView(base: hlBase.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
                    var lh = BlockView(base: lhBase.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
                    var hh = BlockView(base: hhBase.advanced(by: 4 * 16 + 4), width: 4, height: 4, stride: 16)
                    try blockDecode4(decoder: &decoder, block: &hl)
                    try blockDecode4(decoder: &decoder, block: &lh)
                    try blockDecode4(decoder: &decoder, block: &hh)
                }
            }
        }
    }

    return blocks
}

@inline(__always)
func decodePlaneSubbands8(data: [UInt8], blockCount: Int) throws -> [Block2D] {
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
        try blocks[i].withView { view in
            var hlView = BlockView(base: view.base.advanced(by: half), width: half, height: half, stride: 8)
            try blockDecode4(decoder: &decoder, block: &hlView)
            
            var lhView = BlockView(base: view.base.advanced(by: half * 8), width: half, height: half, stride: 8)
            try blockDecode4(decoder: &decoder, block: &lhView)
            
            var hhView = BlockView(base: view.base.advanced(by: half * 8 + half), width: half, height: half, stride: 8)
            try blockDecode4(decoder: &decoder, block: &hhView)
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
                try blockDecode4(decoder: &decoder, block: &hlView)
                
                var lhView = BlockView(base: view.base.advanced(by: half * 8), width: half, height: half, stride: 8)
                try blockDecode4(decoder: &decoder, block: &lhView)
                
                var hhView = BlockView(base: view.base.advanced(by: half * 8 + half), width: half, height: half, stride: 8)
                try blockDecode4(decoder: &decoder, block: &hhView)
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
                try blockDecode16(decoder: &decoder, block: &hlView)
                
                var lhView = BlockView(base: lhBase, width: half, height: half, stride: 32)
                try blockDecode16(decoder: &decoder, block: &lhView)
                
                var hhView = BlockView(base: hhBase, width: half, height: half, stride: 32)
                try blockDecode16(decoder: &decoder, block: &hhView)
                
            case .split8(let tl, let tr, let bl, let br):
                if tl {
                    var ll = BlockView(base: llBase, width: 8, height: 8, stride: 32)
                    var hl = BlockView(base: hlBase, width: 8, height: 8, stride: 32)
                    var lh = BlockView(base: lhBase, width: 8, height: 8, stride: 32)
                    var hh = BlockView(base: hhBase, width: 8, height: 8, stride: 32)
                    try blockDecodeDPCM8(decoder: &decoder, block: &ll, lastVal: &lastVal)
                    try blockDecode8(decoder: &decoder, block: &hl)
                    try blockDecode8(decoder: &decoder, block: &lh)
                    try blockDecode8(decoder: &decoder, block: &hh)
                }
                if tr {
                    var ll = BlockView(base: llBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    var hl = BlockView(base: hlBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    var lh = BlockView(base: lhBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    var hh = BlockView(base: hhBase.advanced(by: 8), width: 8, height: 8, stride: 32)
                    try blockDecodeDPCM8(decoder: &decoder, block: &ll, lastVal: &lastVal)
                    try blockDecode8(decoder: &decoder, block: &hl)
                    try blockDecode8(decoder: &decoder, block: &lh)
                    try blockDecode8(decoder: &decoder, block: &hh)
                }
                if bl {
                    var ll = BlockView(base: llBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    var hl = BlockView(base: hlBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    var lh = BlockView(base: lhBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    var hh = BlockView(base: hhBase.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                    try blockDecodeDPCM8(decoder: &decoder, block: &ll, lastVal: &lastVal)
                    try blockDecode8(decoder: &decoder, block: &hl)
                    try blockDecode8(decoder: &decoder, block: &lh)
                    try blockDecode8(decoder: &decoder, block: &hh)
                }
                if br {
                    var ll = BlockView(base: llBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    var hl = BlockView(base: hlBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    var lh = BlockView(base: lhBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    var hh = BlockView(base: hhBase.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                    try blockDecodeDPCM8(decoder: &decoder, block: &ll, lastVal: &lastVal)
                    try blockDecode8(decoder: &decoder, block: &hl)
                    try blockDecode8(decoder: &decoder, block: &lh)
                    try blockDecode8(decoder: &decoder, block: &hh)
                }
            }
        }
    }

    return blocks
}