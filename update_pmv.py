import re

with open("Sources/vevc/Motion.swift", "r") as f:
    content = f.read()

# Add calculatePMV to Motion.swift
pmv_func = """
public func calculatePMV(mvs: MotionVectors, mbX: Int, mbY: Int, mbCols: Int) -> (dx: Int, dy: Int) {
    let hasLeft = mbX > 0
    let hasTop = mbY > 0
    let hasTopRight = mbY > 0 && mbX < mbCols - 1

    let idxLeft = hasLeft ? (mbY * mbCols + (mbX - 1)) : -1
    let idxTop = hasTop ? ((mbY - 1) * mbCols + mbX) : -1
    let idxTopRight = hasTopRight ? ((mbY - 1) * mbCols + (mbX + 1)) : -1

    var count = 0
    if hasLeft { count += 1 }
    if hasTop { count += 1 }
    if hasTopRight { count += 1 }

    if count == 0 {
        return (0, 0)
    } else if count == 1 {
        let idx = hasLeft ? idxLeft : (hasTop ? idxTop : idxTopRight)
        return (mvs.dx[idx], mvs.dy[idx])
    } else if count == 2 {
        var dxSum = 0
        var dySum = 0
        if hasLeft { dxSum += mvs.dx[idxLeft]; dySum += mvs.dy[idxLeft] }
        if hasTop { dxSum += mvs.dx[idxTop]; dySum += mvs.dy[idxTop] }
        if hasTopRight { dxSum += mvs.dx[idxTopRight]; dySum += mvs.dy[idxTopRight] }
        return (dxSum / 2, dySum / 2)
    } else {
        let lx = mvs.dx[idxLeft]; let ly = mvs.dy[idxLeft]
        let tx = mvs.dx[idxTop]; let ty = mvs.dy[idxTop]
        let rx = mvs.dx[idxTopRight]; let ry = mvs.dy[idxTopRight]

        let minX = min(lx, min(tx, rx))
        let maxX = max(lx, max(tx, rx))
        let pmvX = lx + tx + rx - minX - maxX

        let minY = min(ly, min(ty, ry))
        let maxY = max(ly, max(ty, ry))
        let pmvY = ly + ty + ry - minY - maxY

        return (pmvX, pmvY)
    }
}
"""

if "calculatePMV" not in content:
    content += pmv_func
    with open("Sources/vevc/Motion.swift", "w") as f:
        f.write(content)

with open("Sources/vevc/Encode.swift", "r") as f:
    content = f.read()

# Update encode MV
old_mv_encode = """            for mvIdx in 0..<mvs.dx.count {
                let dx = mvs.dx[mvIdx]
                let dy = mvs.dy[mvIdx]
                if dx == 0 && dy == 0 {
                    mvBw.encodeBin(binVal: 0, ctx: &ctxDx)
                } else {
                    mvBw.encodeBin(binVal: 1, ctx: &ctxDx)

                    let sx: UInt8
                    if dx <= -1 {
                        sx = 1
                    } else {
                        sx = 0
                    }
                    mvBw.encodeBypass(binVal: sx)
                    let mx = UInt32(abs(dx))
                    encodeExpGolomb(val: mx, encoder: &mvBw)

                    let sy: UInt8
                    if dy <= -1 {
                        sy = 1
                    } else {
                        sy = 0
                    }
                    mvBw.encodeBypass(binVal: sy)
                    let my = UInt32(abs(dy))
                    encodeExpGolomb(val: my, encoder: &mvBw)
                }
            }"""

new_mv_encode = """            let mbSize = 32
            let mbCols = (curr.width + mbSize - 1) / mbSize
            for mvIdx in 0..<mvs.dx.count {
                let mbX = mvIdx % mbCols
                let mbY = mvIdx / mbCols
                let pmv = calculatePMV(mvs: mvs, mbX: mbX, mbY: mbY, mbCols: mbCols)
                let mvdX = mvs.dx[mvIdx] - pmv.dx
                let mvdY = mvs.dy[mvIdx] - pmv.dy

                if mvdX == 0 && mvdY == 0 {
                    mvBw.encodeBin(binVal: 0, ctx: &ctxDx)
                } else {
                    mvBw.encodeBin(binVal: 1, ctx: &ctxDx)

                    let sx: UInt8
                    if mvdX <= -1 {
                        sx = 1
                    } else {
                        sx = 0
                    }
                    mvBw.encodeBypass(binVal: sx)
                    let mx = UInt32(abs(mvdX))
                    encodeExpGolomb(val: mx, encoder: &mvBw)

                    let sy: UInt8
                    if mvdY <= -1 {
                        sy = 1
                    } else {
                        sy = 0
                    }
                    mvBw.encodeBypass(binVal: sy)
                    let my = UInt32(abs(mvdY))
                    encodeExpGolomb(val: my, encoder: &mvBw)
                }
            }"""

content = content.replace(old_mv_encode, new_mv_encode)

with open("Sources/vevc/Encode.swift", "w") as f:
    f.write(content)


with open("Sources/vevc/Decode.swift", "r") as f:
    content = f.read()

# Update decode MV
old_mv_decode = """            for i in 0..<mvsCount {
                let isSig = try mvBr.decodeBin(ctx: &ctxDx)
                if isSig == 0 {
                    mvs.dx[i] = 0
                    mvs.dy[i] = 0
                } else {
                    let sx = try mvBr.decodeBypass()
                    let mx = try decodeExpGolomb(decoder: &mvBr)

                    let dx: Int
                    if sx == 1 {
                        dx = -1 * Int(mx)
                    } else {
                        dx = Int(mx)
                    }

                    let sy = try mvBr.decodeBypass()
                    let my = try decodeExpGolomb(decoder: &mvBr)

                    let dy: Int
                    if sy == 1 {
                        dy = -1 * Int(my)
                    } else {
                        dy = Int(my)
                    }

                    mvs.dx[i] = dx
                    mvs.dy[i] = dy
                }
            }"""

new_mv_decode = """            let mbSize = 32
            // We need width to compute mbCols. We can infer width from previous frame.
            guard let prevWidth = prevReconstructed?.width else { throw DecodeError.invalidHeader }
            let mbCols = (prevWidth + mbSize - 1) / mbSize

            for i in 0..<mvsCount {
                let mbX = i % mbCols
                let mbY = i / mbCols
                let pmv = calculatePMV(mvs: mvs, mbX: mbX, mbY: mbY, mbCols: mbCols)

                let isSig = try mvBr.decodeBin(ctx: &ctxDx)
                if isSig == 0 {
                    mvs.dx[i] = pmv.dx
                    mvs.dy[i] = pmv.dy
                } else {
                    let sx = try mvBr.decodeBypass()
                    let mx = try decodeExpGolomb(decoder: &mvBr)

                    let mvdX: Int
                    if sx == 1 {
                        mvdX = -1 * Int(mx)
                    } else {
                        mvdX = Int(mx)
                    }

                    let sy = try mvBr.decodeBypass()
                    let my = try decodeExpGolomb(decoder: &mvBr)

                    let mvdY: Int
                    if sy == 1 {
                        mvdY = -1 * Int(my)
                    } else {
                        mvdY = Int(my)
                    }

                    mvs.dx[i] = mvdX + pmv.dx
                    mvs.dy[i] = mvdY + pmv.dy
                }
            }"""

content = content.replace(old_mv_decode, new_mv_decode)

with open("Sources/vevc/Decode.swift", "w") as f:
    f.write(content)
