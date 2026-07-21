import Foundation

// MARK: - Constants

let DCT_C: [Int32] = [
     2896,  2896,  2896,  2896,  2896,  2896,  2896,  2896,
     4017,  3406,  2276,   799,  -799, -2276, -3406, -4017,
     3784,  1567, -1567, -3784, -3784, -1567,  1567,  3784,
     3406,  -799, -4017, -2276,  2276,  4017,   799, -3406,
     2896, -2896, -2896,  2896,  2896, -2896, -2896,  2896,
     2276, -4017,   799,  3406, -3406,  -799,  4017, -2276,
     1567, -3784,  3784, -1567, -1567,  3784, -3784,  1567,
      799, -2276,  3406, -4017,  4017, -3406,  2276,  -799
]

let ZIGZAG: [Int] = [
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
]

// MARK: - Quantization

@inline(__always)
func quantize(val: Int16, step: Int) -> Int16 {
    let v = Int32(val)
    let s = Int32(step)
    if v >= 0 {
        return Int16((v + (s >> 1)) / s)
    } else {
        return Int16((v - (s >> 1)) / s)
    }
}

@inline(__always)
func dequantize(qval: Int16, step: Int) -> Int16 {
    return Int16(truncatingIfNeeded: Int32(qval) &* Int32(step))
}

// MARK: - DCT Functions

@inline(__always)
func forwardDCT8x8(block: UnsafeMutablePointer<Int16>, stride: Int) {
    var temp = [Int32](repeating: 0, count: 64)
    
    for i in 0..<8 {
        for j in 0..<8 {
            var sum: Int32 = 0
            for k in 0..<8 {
                sum &+= DCT_C[j * 8 + k] &* Int32(block[i * stride + k])
            }
            temp[i * 8 + j] = (sum + 4096) >> 13
        }
    }
    
    for j in 0..<8 {
        for i in 0..<8 {
            var sum: Int32 = 0
            for k in 0..<8 {
                sum &+= DCT_C[i * 8 + k] &* temp[k * 8 + j]
            }
            block[i * stride + j] = Int16(truncatingIfNeeded: (sum + 4096) >> 13)
        }
    }
}

@inline(__always)
func inverseDCT8x8(block: UnsafeMutablePointer<Int16>, stride: Int) {
    var temp = [Int32](repeating: 0, count: 64)
    
    for j in 0..<8 {
        for i in 0..<8 {
            var sum: Int32 = 0
            for k in 0..<8 {
                sum &+= DCT_C[k * 8 + i] &* Int32(block[k * stride + j])
            }
            temp[i * 8 + j] = (sum + 4096) >> 13
        }
    }
    
    for i in 0..<8 {
        for j in 0..<8 {
            var sum: Int32 = 0
            for k in 0..<8 {
                sum &+= DCT_C[k * 8 + j] &* temp[i * 8 + k]
            }
            block[i * stride + j] = Int16(truncatingIfNeeded: (sum + 4096) >> 13)
        }
    }
}

// MARK: - Stats

public struct DCTEncodeStats {
    public let dcBytes: Int
    public let acBytes: Int
    public let totalBytes: Int
}

// MARK: - Encode API

/// Byte stream format:
/// [VLQ: DC Bytes Length] [DC Stream Bytes...] [AC Stream Bytes...]
/// Stream lengths natively included inside EntropyEncoder output.
public func encodeL0PlaneDCT(plane: [Int16], width: Int, height: Int, stride: Int, step: Int) -> (bytes: [UInt8], stats: DCTEncodeStats) {
    let bw = (width + 7) / 8
    let bh = (height + 7) / 8
    let paddedWidth = bw * 8
    let paddedHeight = bh * 8
    
    var localPlane = [Int16](repeating: 0, count: paddedWidth * paddedHeight)
    for y in 0..<height {
        for x in 0..<width {
            localPlane[y * paddedWidth + x] = plane[y * stride + x]
        }
        for x in width..<paddedWidth {
            localPlane[y * paddedWidth + x] = plane[y * stride + width - 1]
        }
    }
    for y in height..<paddedHeight {
        for x in 0..<paddedWidth {
            localPlane[y * paddedWidth + x] = localPlane[(height - 1) * paddedWidth + x]
        }
    }
    
    var dcEncoder = EntropyEncoder()
    var acEncoder = EntropyEncoder()
    
    var prevDC = [Int16](repeating: 0, count: bw)
    var zeroRun: UInt32 = 0
    
    localPlane.withUnsafeMutableBufferPointer { ptr in
        let base = ptr.baseAddress!
        
        for by in 0..<bh {
            for bx in 0..<bw {
                let blockBase = base.advanced(by: by * 8 * paddedWidth + bx * 8)
                
                forwardDCT8x8(block: blockBase, stride: paddedWidth)
                
                let rawDC = blockBase[0]
                let qDC = quantize(val: rawDC, step: step)
                blockBase[0] = qDC 
                
                var pred: Int16 = 0
                if bx > 0 {
                    pred = prevDC[bx - 1]
                } else if by > 0 {
                    pred = prevDC[bx]
                }
                let diff = qDC - pred
                prevDC[bx] = qDC
                
                dcEncoder.addPair(run: 0, val: diff, context: 4)
                
                for i in 1..<64 {
                    let idx = ZIGZAG[i]
                    let row = idx / 8
                    let col = idx % 8
                    
                    let val = blockBase[row * paddedWidth + col]
                    let qVal = quantize(val: val, step: step)
                    
                    if qVal == 0 {
                        zeroRun += 1
                    } else {
                        acEncoder.addPair(run: zeroRun, val: qVal, context: 0)
                        zeroRun = 0
                    }
                }
            }
        }
    }
    acEncoder.addTrailingZeros(zeroRun)
    
    let dcBytes = dcEncoder.getData(selectModel: StaticEntropyModel.selectModel)
    let acBytes = acEncoder.getData(selectModel: StaticEntropyModel.selectModel)
    
    var outBytes = [UInt8]()
    writeVLQSize(&outBytes, dcBytes.count)
    outBytes.append(contentsOf: dcBytes)
    outBytes.append(contentsOf: acBytes)
    
    let stats = DCTEncodeStats(dcBytes: dcBytes.count, acBytes: acBytes.count, totalBytes: outBytes.count)
    return (bytes: outBytes, stats: stats)
}

// MARK: - Decode API

public func decodeL0PlaneDCT(bytes: [UInt8], width: Int, height: Int, step: Int) throws -> [Int16] {
    var offset = 0
    let dcBytesLength = try readVLQSizeFromBytes(bytes, offset: &offset)
    
    let dcStream = Array(bytes[offset..<offset+dcBytesLength])
    offset += dcBytesLength
    let acStream = Array(bytes[offset..<bytes.count])
    
    return try dcStream.withUnsafeBufferPointer { dcPtr in
        return try acStream.withUnsafeBufferPointer { acPtr in
            var dcDecoder = try EntropyDecoder(base: dcPtr.baseAddress!, count: dcPtr.count)
            var acDecoder = try EntropyDecoder(base: acPtr.baseAddress!, count: acPtr.count)
            
            let bw = (width + 7) / 8
            let bh = (height + 7) / 8
            let paddedWidth = bw * 8
            let paddedHeight = bh * 8
            
            var localPlane = [Int16](repeating: 0, count: paddedWidth * paddedHeight)
            var prevDC = [Int16](repeating: 0, count: bw)
            
            let totalAC = bw * bh * 63
            var allAC = [Int16](repeating: 0, count: totalAC)
            var acIndex = 0
            while acIndex < totalAC {
                let pair = acDecoder.readPair(context: 0)
                if pair.run == 0 && pair.val == 0 {
                    break
                }
                acIndex += Int(pair.run)
                if acIndex < totalAC {
                    allAC[acIndex] = dequantize(qval: pair.val, step: step)
                    acIndex += 1
                }
            }
            
            var acOffset = 0
            
            for by in 0..<bh {
                for bx in 0..<bw {
                    let dcPair = dcDecoder.readPair(context: 4)
                    let diff = dcPair.val
                    var pred: Int16 = 0
                    if bx > 0 {
                        pred = prevDC[bx - 1]
                    } else if by > 0 {
                        pred = prevDC[bx]
                    }
                    let qDC = pred + diff
                    prevDC[bx] = qDC
                    
                    var block = [Int16](repeating: 0, count: 64)
                    block[0] = dequantize(qval: qDC, step: step)
                    
                    for i in 1..<64 {
                        let idx = ZIGZAG[i]
                        block[idx] = allAC[acOffset]
                        acOffset += 1
                    }
                    
                    block.withUnsafeMutableBufferPointer { ptr in
                        inverseDCT8x8(block: ptr.baseAddress!, stride: 8)
                    }
                    
                    for row in 0..<8 {
                        for col in 0..<8 {
                            localPlane[(by * 8 + row) * paddedWidth + (bx * 8 + col)] = block[row * 8 + col]
                        }
                    }
                }
            }
            
            var outPlane = [Int16](repeating: 0, count: width * height)
            for y in 0..<height {
                for x in 0..<width {
                    outPlane[y * width + x] = localPlane[y * paddedWidth + x]
                }
            }
            
            return outPlane
        }
    }
}
