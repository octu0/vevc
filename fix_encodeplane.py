import re

with open("Sources/vevc/EncodePlane.swift", "r") as f:
    content = f.read()

# Replace async let calls to remove & and inout dependency
old_layer = """    async let taskBufY = encodePlaneSubbands(blocks: &subBlocksY, size: size, zeroThreshold: zeroThreshold)
    async let taskBufCb = encodePlaneSubbands(blocks: &subBlocksCb, size: size, zeroThreshold: zeroThreshold)
    async let taskBufCr = encodePlaneSubbands(blocks: &subBlocksCr, size: size, zeroThreshold: zeroThreshold)"""

new_layer = """    async let taskBufY = encodePlaneSubbands(blocks: subBlocksY, size: size, zeroThreshold: zeroThreshold)
    async let taskBufCb = encodePlaneSubbands(blocks: subBlocksCb, size: size, zeroThreshold: zeroThreshold)
    async let taskBufCr = encodePlaneSubbands(blocks: subBlocksCr, size: size, zeroThreshold: zeroThreshold)"""

content = content.replace(old_layer, new_layer)

old_base = """    async let taskBufY = encodePlaneBaseSubbands(blocks: &subBlocksY, size: size, zeroThreshold: zeroThreshold)
    async let taskBufCb = encodePlaneBaseSubbands(blocks: &subBlocksCb, size: size, zeroThreshold: zeroThreshold)
    async let taskBufCr = encodePlaneBaseSubbands(blocks: &subBlocksCr, size: size, zeroThreshold: zeroThreshold)"""

new_base = """    async let taskBufY = encodePlaneBaseSubbands(blocks: subBlocksY, size: size, zeroThreshold: zeroThreshold)
    async let taskBufCb = encodePlaneBaseSubbands(blocks: subBlocksCb, size: size, zeroThreshold: zeroThreshold)
    async let taskBufCr = encodePlaneBaseSubbands(blocks: subBlocksCr, size: size, zeroThreshold: zeroThreshold)"""

content = content.replace(old_base, new_base)

with open("Sources/vevc/EncodePlane.swift", "w") as f:
    f.write(content)

with open("Sources/vevc/Encode.swift", "r") as f:
    encode_content = f.read()

# Remove inout from signatures in Encode.swift
old_sig1 = "func encodePlaneSubbands(blocks: inout [Block2D], size: Int, zeroThreshold: Int) -> [UInt8] {"
new_sig1 = "func encodePlaneSubbands(blocks: [Block2D], size: Int, zeroThreshold: Int) -> [UInt8] {"
encode_content = encode_content.replace(old_sig1, new_sig1)

old_sig2 = "func encodePlaneBaseSubbands(blocks: inout [Block2D], size: Int, zeroThreshold: Int) -> [UInt8] {"
new_sig2 = "func encodePlaneBaseSubbands(blocks: [Block2D], size: Int, zeroThreshold: Int) -> [UInt8] {"
encode_content = encode_content.replace(old_sig2, new_sig2)

# Fix isEffectivelyZero and isEffectivelyZeroBase to not require inout block views if possible, or copy blocks inside the function
# Actually, isEffectivelyZero modifies hl/lh/hh (sets to 0 if effectively zero).
# If we remove inout from blocks array, we can't mutate the blocks in-place directly without making a copy.
# Since we only pass the blocks to CABAC encoder after this and don't reuse them in the caller, we can make `blocks` mutable locally.
old_sig1_impl = "func encodePlaneSubbands(blocks: [Block2D], size: Int, zeroThreshold: Int) -> [UInt8] {"
new_sig1_impl = "func encodePlaneSubbands(blocks: [Block2D], size: Int, zeroThreshold: Int) -> [UInt8] {\n    var blocks = blocks"
encode_content = encode_content.replace(old_sig1_impl, new_sig1_impl)

old_sig2_impl = "func encodePlaneBaseSubbands(blocks: [Block2D], size: Int, zeroThreshold: Int) -> [UInt8] {"
new_sig2_impl = "func encodePlaneBaseSubbands(blocks: [Block2D], size: Int, zeroThreshold: Int) -> [UInt8] {\n    var blocks = blocks"
encode_content = encode_content.replace(old_sig2_impl, new_sig2_impl)

with open("Sources/vevc/Encode.swift", "w") as f:
    f.write(encode_content)
