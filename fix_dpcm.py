import re

with open("Sources/vevc/Encode.swift", "r") as f:
    content = f.read()

# Refactor blockEncodeDPCM to compute LSCP without allocating an array, then encode using the same math.
old_blockEncodeDPCM = """@inline(__always)
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


new_blockEncodeDPCM = """@inline(__always)
func blockEncodeDPCM(encoder: inout CABACEncoder, block: BlockView, size: Int, lastVal: inout Int16, ctxSig: inout ContextModel, ctxSign: inout ContextModel, ctxMag: inout [ContextModel]) {
    var lscpIdx = -1

    // Pass 1: find LSCP
    let ptr0 = block.rowPointer(y: 0)
    if ptr0[0] - lastVal != 0 { lscpIdx = 0 }
    for x in 1..<size {
        if ptr0[x] - ptr0[x - 1] != 0 { lscpIdx = max(lscpIdx, x) }
    }

    for y in 1..<size {
        let ptr = block.rowPointer(y: y)
        let ptrPrev = block.rowPointer(y: y - 1)
        if ptr[0] - ptrPrev[0] != 0 { lscpIdx = max(lscpIdx, y * size + 0) }

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
            if ptr[x] - predicted != 0 { lscpIdx = max(lscpIdx, y * size + x) }
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

        // Pass 2: encode up to LSCP
        var currentIdx = 0
        let diff00 = ptr0[0] - lastVal
        encodeCoeff(val: diff00, encoder: &encoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag)

        for x in 1..<size {
            currentIdx += 1
            if currentIdx > lscpIdx { break }
            let diff = ptr0[x] - ptr0[x - 1]
            encodeCoeff(val: diff, encoder: &encoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag)
        }

        for y in 1..<size {
            if currentIdx >= lscpIdx { break }
            let ptr = block.rowPointer(y: y)
            let ptrPrev = block.rowPointer(y: y - 1)

            currentIdx += 1
            if currentIdx > lscpIdx { break }
            let diffY0 = ptr[0] - ptrPrev[0]
            encodeCoeff(val: diffY0, encoder: &encoder, ctxSig: &ctxSig, ctxSign: &ctxSign, ctxMag: &ctxMag)

            for x in 1..<size {
                currentIdx += 1
                if currentIdx > lscpIdx { break }
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
    }

    lastVal = block.rowPointer(y: size - 1)[size - 1]
}"""

content = content.replace(old_blockEncodeDPCM, new_blockEncodeDPCM)

with open("Sources/vevc/Encode.swift", "w") as f:
    f.write(content)
