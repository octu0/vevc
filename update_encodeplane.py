import re

with open("Sources/vevc/EncodePlane.swift", "r") as f:
    content = f.read()

# Replace encodePlaneLayer
old_layer = """    let bufY = encodePlaneSubbands(blocks: &subBlocksY, size: size, zeroThreshold: zeroThreshold)
    let bufCb = encodePlaneSubbands(blocks: &subBlocksCb, size: size, zeroThreshold: zeroThreshold)
    let bufCr = encodePlaneSubbands(blocks: &subBlocksCr, size: size, zeroThreshold: zeroThreshold)
    debugLog("  [Layer \(layer)] Y=\(bufY.count) Cb=\(bufCb.count) Cr=\(bufCr.count) bytes")"""

new_layer = """    async let taskBufY = encodePlaneSubbands(blocks: &subBlocksY, size: size, zeroThreshold: zeroThreshold)
    async let taskBufCb = encodePlaneSubbands(blocks: &subBlocksCb, size: size, zeroThreshold: zeroThreshold)
    async let taskBufCr = encodePlaneSubbands(blocks: &subBlocksCr, size: size, zeroThreshold: zeroThreshold)

    let bufY = await taskBufY
    let bufCb = await taskBufCb
    let bufCr = await taskBufCr
    debugLog("  [Layer \(layer)] Y=\(bufY.count) Cb=\(bufCb.count) Cr=\(bufCr.count) bytes")"""

content = content.replace(old_layer, new_layer)

# Replace encodePlaneBase
old_base = """    let bufY = encodePlaneBaseSubbands(blocks: &subBlocksY, size: size, zeroThreshold: zeroThreshold)
    let bufCb = encodePlaneBaseSubbands(blocks: &subBlocksCb, size: size, zeroThreshold: zeroThreshold)
    let bufCr = encodePlaneBaseSubbands(blocks: &subBlocksCr, size: size, zeroThreshold: zeroThreshold)
    debugLog("  [Layer \(layer)/Base] Y=\(bufY.count) Cb=\(bufCb.count) Cr=\(bufCr.count) bytes")"""

new_base = """    async let taskBufY = encodePlaneBaseSubbands(blocks: &subBlocksY, size: size, zeroThreshold: zeroThreshold)
    async let taskBufCb = encodePlaneBaseSubbands(blocks: &subBlocksCb, size: size, zeroThreshold: zeroThreshold)
    async let taskBufCr = encodePlaneBaseSubbands(blocks: &subBlocksCr, size: size, zeroThreshold: zeroThreshold)

    let bufY = await taskBufY
    let bufCb = await taskBufCb
    let bufCr = await taskBufCr
    debugLog("  [Layer \(layer)/Base] Y=\(bufY.count) Cb=\(bufCb.count) Cr=\(bufCr.count) bytes")"""

content = content.replace(old_base, new_base)

with open("Sources/vevc/EncodePlane.swift", "w") as f:
    f.write(content)
