import re

with open("Sources/vevc/Encode.swift", "r") as f:
    content = f.read()

# Replace blockEncode
old_encode = """@inline(__always)
func encodeCoeff(val: Int16, encoder: inout CABACEncoder, ctxSig: inout ContextModel, ctxSign: inout ContextModel, ctxMag: inout [ContextModel]) {
    if val == 0 {
        encoder.encodeBin(binVal: 0, ctx: &ctxSig)
        return
    }
    encoder.encodeBin(binVal: 1, ctx: &ctxSig)

    let signBit: UInt8
    if val <= -1 {
        signBit = 1
    } else {
        signBit = 0
    }
    let absVal = UInt32(abs(Int(val)))

    encoder.encodeBin(binVal: signBit, ctx: &ctxSign)

    let magMinus1 = absVal &- 1
    let numBins = min(magMinus1, 7)
    for i in 0..<numBins {
        encoder.encodeBin(binVal: 1, ctx: &ctxMag[Int(i)])
    }
    if magMinus1 < 7 {
        encoder.encodeBin(binVal: 0, ctx: &ctxMag[Int(numBins)])
    } else {
        let rem = magMinus1 &- 7
        encodeExpGolomb(val: rem, encoder: &encoder)
    }
}"""

new_encode = """@inline(__always)
func encodeCoeff(val: Int16, encoder: inout CABACEncoder, ctxSig: inout ContextModel, ctxSign: inout ContextModel, ctxMag: inout [ContextModel]) {
    if val == 0 {
        encoder.encodeBin(binVal: 0, ctx: &ctxSig)
        return
    }
    encoder.encodeBin(binVal: 1, ctx: &ctxSig)

    let signBit: UInt8
    if val <= -1 {
        signBit = 1
    } else {
        signBit = 0
    }
    let absVal = UInt32(abs(Int(val)))

    encoder.encodeBin(binVal: signBit, ctx: &ctxSign)

    let magMinus1 = absVal &- 1
    let numBins = min(magMinus1, 7)
    ctxMag.withUnsafeMutableBufferPointer { ptr in
        guard let base = ptr.baseAddress else { return }
        for i in 0..<numBins {
            encoder.encodeBin(binVal: 1, ctx: &base[Int(i)])
        }
        if magMinus1 < 7 {
            encoder.encodeBin(binVal: 0, ctx: &base[Int(numBins)])
        }
    }

    if magMinus1 >= 7 {
        let rem = magMinus1 &- 7
        encodeExpGolomb(val: rem, encoder: &encoder)
    }
}"""

content = content.replace(old_encode, new_encode)

with open("Sources/vevc/Encode.swift", "w") as f:
    f.write(content)
