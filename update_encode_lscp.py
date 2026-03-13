import re

with open("Sources/vevc/Encode.swift", "r") as f:
    content = f.read()

# Modify blockEncode
old_blockEncode = """@inline(__always)
func blockEncode(encoder: inout CABACEncoder, block: BlockView, size: Int, ctxSig: inout ContextModel, ctxSign: inout ContextModel, ctxMag: inout [ContextModel]) {
    for y in 0..<size {
        let ptr = block.rowPointer(y: y)
        for x in 0..<size {
            encodeCoeff(val: ptr[x], encoder: &encoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag)
        }
    }
}"""

new_blockEncode = """@inline(__always)
func blockEncode(encoder: inout CABACEncoder, block: BlockView, size: Int, ctxSig: inout ContextModel, ctxSign: inout ContextModel, ctxMag: inout [ContextModel]) {
    var lscpX = -1
    var lscpY = -1
    for y in stride(from: size - 1, through: 0, by: -1) {
        let ptr = block.rowPointer(y: y)
        for x in stride(from: size - 1, through: 0, by: -1) {
            if ptr[x] != 0 {
                lscpX = x
                lscpY = y
                break
            }
        }
        if lscpX != -1 { break }
    }

    if lscpX == -1 {
        encoder.encodeBypass(binVal: 0)
        return
    }
    encoder.encodeBypass(binVal: 1)

    encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
    encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

    for y in 0...lscpY {
        let ptr = block.rowPointer(y: y)
        let endX = (y == lscpY) ? lscpX : (size - 1)
        for x in 0...endX {
            encodeCoeff(val: ptr[x], encoder: &encoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag)
        }
    }
}"""

content = content.replace(old_blockEncode, new_blockEncode)

# Modify blockEncodeDPCM
old_blockEncodeDPCM = """@inline(__always)
func blockEncodeDPCM(encoder: inout CABACEncoder, block: BlockView, size: Int, lastVal: inout Int16, ctxSig: inout ContextModel, ctxSign: inout ContextModel, ctxMag: inout [ContextModel]) {
    let ptr0 = block.rowPointer(y: 0)

    let diff00 = ptr0[0] - lastVal
    encodeCoeff(val: diff00, encoder: &encoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag)

    for x in 1..<size {
        let diff = ptr0[x] - ptr0[x - 1]
        encodeCoeff(val: diff, encoder: &encoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag)
    }

    for y in 1..<size {
        let ptr = block.rowPointer(y: y)
        let ptrPrev = block.rowPointer(y: y - 1)

        let diffY0 = ptr[0] - ptrPrev[0]
        encodeCoeff(val: diffY0, encoder: &encoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag)

        for x in 1..<size {
            let a = Int(ptr[x - 1])
            let b = Int(ptrPrev[x])
            let c = Int(ptrPrev[x - 1])
            let predicted: Int16
            if c >= a && c >= b {
                predicted = Int16(min(a, b))
            } else if c <= a && c <= b {
                predicted = Int16(max(a, b))
            } else {
                predicted = Int16(a + b - c)
            }
            let diff = ptr[x] - predicted
            encodeCoeff(val: diff, encoder: &encoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag)
        }
    }
    lastVal = block.rowPointer(y: size - 1)[size - 1]
}"""

new_blockEncodeDPCM = """@inline(__always)
func blockEncodeDPCM(encoder: inout CABACEncoder, block: BlockView, size: Int, lastVal: inout Int16, ctxSig: inout ContextModel, ctxSign: inout ContextModel, ctxMag: inout [ContextModel]) {
    var diffs = [Int16](repeating: 0, count: size * size)

    let ptr0 = block.rowPointer(y: 0)
    diffs[0] = ptr0[0] - lastVal
    for x in 1..<size {
        diffs[x] = ptr0[x] - ptr0[x - 1]
    }
    for y in 1..<size {
        let ptr = block.rowPointer(y: y)
        let ptrPrev = block.rowPointer(y: y - 1)
        diffs[y * size + 0] = ptr[0] - ptrPrev[0]
        for x in 1..<size {
            let a = Int(ptr[x - 1])
            let b = Int(ptrPrev[x])
            let c = Int(ptrPrev[x - 1])
            let predicted: Int16
            if c >= a && c >= b {
                predicted = Int16(min(a, b))
            } else if c <= a && c <= b {
                predicted = Int16(max(a, b))
            } else {
                predicted = Int16(a + b - c)
            }
            diffs[y * size + x] = ptr[x] - predicted
        }
    }

    var lscpIdx = -1
    for i in stride(from: diffs.count - 1, through: 0, by: -1) {
        if diffs[i] != 0 {
            lscpIdx = i
            break
        }
    }

    if lscpIdx == -1 {
        encoder.encodeBypass(binVal: 0)
    } else {
        encoder.encodeBypass(binVal: 1)
        let lscpX = lscpIdx % size
        let lscpY = lscpIdx / size
        encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
        encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)

        for i in 0...lscpIdx {
            encodeCoeff(val: diffs[i], encoder: &encoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag)
        }
    }

    lastVal = block.rowPointer(y: size - 1)[size - 1]
}"""

content = content.replace(old_blockEncodeDPCM, new_blockEncodeDPCM)

with open("Sources/vevc/Encode.swift", "w") as f:
    f.write(content)
