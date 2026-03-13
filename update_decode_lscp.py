import re

with open("Sources/vevc/Decode.swift", "r") as f:
    content = f.read()

# Modify blockDecode
old_blockDecode = """@inline(__always)
func blockDecode(decoder: inout CABACDecoder, block: inout BlockView, size: Int, ctxSig: inout ContextModel, ctxSign: inout ContextModel, ctxMag: inout [ContextModel]) throws {
    for y in 0..<size {
        let ptr = block.rowPointer(y: y)
        for x in 0..<size {
            ptr[x] = try decodeCoeff(decoder: &decoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag)
        }
    }
}"""

new_blockDecode = """@inline(__always)
func blockDecode(decoder: inout CABACDecoder, block: inout BlockView, size: Int, ctxSig: inout ContextModel, ctxSign: inout ContextModel, ctxMag: inout [ContextModel]) throws {
    let hasNonZero = try decoder.decodeBypass()
    if hasNonZero == 0 {
        for y in 0..<size {
            let ptr = block.rowPointer(y: y)
            for x in 0..<size {
                ptr[x] = 0
            }
        }
        return
    }

    let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
    let lscpY = Int(try decodeExpGolomb(decoder: &decoder))

    for y in 0..<size {
        let ptr = block.rowPointer(y: y)
        if y > lscpY {
            for x in 0..<size { ptr[x] = 0 }
            continue
        }
        let endX = (y == lscpY) ? lscpX : (size - 1)
        for x in 0...endX {
            ptr[x] = try decodeCoeff(decoder: &decoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag)
        }
        for x in (endX + 1)..<size {
            ptr[x] = 0
        }
    }
}"""

content = content.replace(old_blockDecode, new_blockDecode)

# Modify blockDecodeDPCM
old_blockDecodeDPCM = """@inline(__always)
func blockDecodeDPCM(decoder: inout CABACDecoder, block: inout BlockView, size: Int, lastVal: inout Int16, ctxSig: inout ContextModel, ctxSign: inout ContextModel, ctxMag: inout [ContextModel]) throws {
    let ptr0 = block.rowPointer(y: 0)

    let diff00 = try decodeCoeff(decoder: &decoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag)
    ptr0[0] = diff00 + lastVal

    for x in 1..<size {
        let diff = try decodeCoeff(decoder: &decoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag)
        ptr0[x] = diff + ptr0[x - 1]
    }

    for y in 1..<size {
        let ptr = block.rowPointer(y: y)
        let ptrPrev = block.rowPointer(y: y - 1)

        let diffY0 = try decodeCoeff(decoder: &decoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag)
        ptr[0] = diffY0 + ptrPrev[0]

        for x in 1..<size {
            let diff = try decodeCoeff(decoder: &decoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag)

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
            ptr[x] = diff + predicted
        }
    }
    lastVal = block.rowPointer(y: size - 1)[size - 1]
}"""

new_blockDecodeDPCM = """@inline(__always)
func blockDecodeDPCM(decoder: inout CABACDecoder, block: inout BlockView, size: Int, lastVal: inout Int16, ctxSig: inout ContextModel, ctxSign: inout ContextModel, ctxMag: inout [ContextModel]) throws {
    let hasNonZero = try decoder.decodeBypass()
    var lscpIdx = -1
    if hasNonZero == 1 {
        let lscpX = Int(try decodeExpGolomb(decoder: &decoder))
        let lscpY = Int(try decodeExpGolomb(decoder: &decoder))
        lscpIdx = lscpY * size + lscpX
    }

    let ptr0 = block.rowPointer(y: 0)

    let diff00 = (0 <= lscpIdx) ? try decodeCoeff(decoder: &decoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag) : 0
    ptr0[0] = diff00 + lastVal

    for x in 1..<size {
        let idx = x
        let diff = (idx <= lscpIdx) ? try decodeCoeff(decoder: &decoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag) : 0
        ptr0[x] = diff + ptr0[x - 1]
    }

    for y in 1..<size {
        let ptr = block.rowPointer(y: y)
        let ptrPrev = block.rowPointer(y: y - 1)

        let idx0 = y * size
        let diffY0 = (idx0 <= lscpIdx) ? try decodeCoeff(decoder: &decoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag) : 0
        ptr[0] = diffY0 + ptrPrev[0]

        for x in 1..<size {
            let idx = y * size + x
            let diff = (idx <= lscpIdx) ? try decodeCoeff(decoder: &decoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag) : 0

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
            ptr[x] = diff + predicted
        }
    }
    lastVal = block.rowPointer(y: size - 1)[size - 1]
}"""

content = content.replace(old_blockDecodeDPCM, new_blockDecodeDPCM)

with open("Sources/vevc/Decode.swift", "w") as f:
    f.write(content)
