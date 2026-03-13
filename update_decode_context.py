import re

with open("Sources/vevc/Decode.swift", "r") as f:
    content = f.read()

old_decode = """@inline(__always)
func decodeCoeff(decoder: inout CABACDecoder, ctxSig: inout ContextModel, ctxSign: inout ContextModel, ctxMag: inout [ContextModel]) throws -> Int16 {
    let sig = try decoder.decodeBin(ctx: &ctxSig)
    if sig == 0 {
        return 0
    }

    let signBit = try decoder.decodeBin(ctx: &ctxSign)

    var mag: UInt32 = 1
    for i in 0..<7 {
        let bit = try decoder.decodeBin(ctx: &ctxMag[Int(i)])
        if bit == 0 {
            break
        }
        mag += 1
    }

    if mag == 8 {
        let rem = try decodeExpGolomb(decoder: &decoder)
        mag += rem
    }

    let sVal = Int16(mag)
    if signBit == 1 {
        return -1 * sVal
    }
    return sVal
}"""

new_decode = """@inline(__always)
func decodeCoeff(decoder: inout CABACDecoder, ctxSig: inout ContextModel, ctxSign: inout ContextModel, ctxMag: inout [ContextModel]) throws -> Int16 {
    let sig = try decoder.decodeBin(ctx: &ctxSig)
    if sig == 0 {
        return 0
    }

    let signBit = try decoder.decodeBin(ctx: &ctxSign)

    var mag: UInt32 = 1
    try ctxMag.withUnsafeMutableBufferPointer { ptr in
        guard let base = ptr.baseAddress else { return }
        for i in 0..<7 {
            let bit = try decoder.decodeBin(ctx: &base[Int(i)])
            if bit == 0 {
                break
            }
            mag += 1
        }
    }

    if mag == 8 {
        let rem = try decodeExpGolomb(decoder: &decoder)
        mag += rem
    }

    let sVal = Int16(mag)
    if signBit == 1 {
        return -1 * sVal
    }
    return sVal
}"""

content = content.replace(old_decode, new_decode)

with open("Sources/vevc/Decode.swift", "w") as f:
    f.write(content)
