# Profile 0x02 (P-skip/LTR) Design Specification

## 1. Overview and Profile Positioning
Profile 0x02 introduces P-skip and Long-Term Reference (LTR) capabilities to the vevc codec. This profile is designed specifically for content with large static areas, such as game streaming, particularly at low bitrates. It is an opt-in profile; Profile 0x01 remains the default. Profile 0x02 provides better visual quality, lower bitrate, and faster encoding/decoding speeds for supported content, but may reduce quality at higher bitrates due to the overhead of signaling the skip map and conservative skip decisions.

## 2. Functional Specification

### 2.1. Skip Decision Logic
The skip decision compares the current frame's source image against the previous frame's source image.
- **Decision Granularity:** The base decision block size is 32x32 pixels. However, the evaluation is performed on 16x16 sub-blocks. 
- **Sub-block Evaluation:** For a 32x32 block to be marked as skipped, all of its four 16x16 sub-blocks (along with their corresponding chroma regions) must satisfy the skip threshold.
- **Threshold:** The skip condition requires the Sum of Absolute Differences (SAD) between the current and previous source blocks to be less than or equal to `blockThreshold`.
- **Environment Override:** The per-pixel threshold is configurable via the `skipThreshold` parameter in the `VEVCEncoder` initializer, which defaults to 2. The environment variable `VEVC_SKIP_THRESH` serves as a development override and is evaluated only once during initialization.
- **Edge Blocks:** The calculation uses `computeZeroSAD16x16` for full 16x16 blocks and falls back to `computeZeroSADSubBlock` for partial blocks at the frame edges. There are no additional guard conditions for the skip decision (the only requirement is that all sub-blocks satisfy the threshold).

### 2.2. Skip Modes and LTR References
- **Skip Modes:** A block can be assigned one of three modes, defined in `BlockMode`:
  - `inter` (0): Standard inter-prediction block.
  - `skip_prev` (1): Skipped block referencing the immediately previous reconstructed frame.
  - `skip_ltr` (2): Skipped block referencing the Long-Term Reference (LTR) frame.
- **Mode Selection:** If a block passes the skip decision, its block position counter (`staticCounters`) is incremented. If the counter equals the current GOP position (`gopPosition`), the block is marked as `skip_ltr`. Otherwise, it is marked as `skip_prev`. If a block fails the decision, its counter is reset to 0, and it is marked as `inter`.
- **LTR Reference:** The LTR reference (`ltrInput` / `firstRecon`) is the reconstructed image of the first frame (I-frame) of the current Group of Pictures (GOP). It is updated only at keyframes.

### 2.3. Bitstream Elements
- **Frame Header:** For `pFrame` in Profile 0x02, the frame header (`VEVCFrameHeader`) includes `skipMapSize` (4 bytes, UInt32BE) and a `hasRefDir` flag (0x10 bit in the frame type byte).
- **Skip Map Serialization:** The `skipMap` is encoded directly into the bitstream, taking `skipMapSize` bytes.
- **Motion Vectors (MV):** MVs are encoded alongside the `skipMap`. For skipped blocks, MV encoding is completely omitted from the bitstream. The decoder symmetrically restores the zero motion vectors by checking the `skipMap`.
- **Reference Direction:** If `hasRefDir` is true, an additional array of bits (`refDirBuf`) is serialized to indicate the reference direction (LTR or previous). The size is specified by `refDirSize`.

### 2.4. Decoder Restoration Rules
During reconstruction in `DecodeSpatial.swift` (`decodeSpatialLayers`), the decoder performs a final skip copy to restore the skipped blocks:
- **`skip_ltr`:** The decoder copies pixels from `nextPd` (which holds the LTR reference, usually the I-frame).
- **`skip_prev`:** The decoder copies pixels from `predictedPd` (the immediately previous reconstructed frame).
The copy operates at the highest available resolution (Layer 2, Layer 1, or Layer 0), using functions like `copyBlockPointer` for safe inner blocks and `copyBlockSafe` for edge blocks.

## 3. Encoder/Decoder Reconstruction Symmetry (Invariants)
To maintain a closed-loop design, the following invariants are strictly enforced:

1. **Reconstruction Bit-Exactness:** The reconstructed image used by the encoder as the reference for the next frame MUST perfectly bit-match the reconstructed image produced by the decoder across all frames and all planes.
2. **Processing Order:** Both the encoder and decoder must follow the exact same processing sequence:
   - Reconstruction (Dequantization + Inverse DWT).
   - Motion Compensation (MC) - for skip blocks, this is a 1:1 raw copy from the reference.
   - Deblocking Filter.
   - **Final Skip Copy:** (Crucial Step) Skipped blocks are overwritten with the raw reference pixels (`skip_prev` from the previous frame, `skip_ltr` from the LTR frame).
3. **Post-Deblocking Skip Copy:** The final skip copy occurs **after** the deblocking filter. This ensures that skipped block pixels are identical to the raw reference pixels. The encoder applies this copy in `encodeSpatialLayers` immediately after `applyDeblockingFilter32` / `applyDeblockingFilterChroma16`. The decoder applies it at the end of `decodeSpatialLayers`. (Historically, the encoder kept deblocked pixels in the reference for skipped blocks, causing drift at the skip/inter boundaries).
4. **Validation:** This bit-exactness is permanently verified by the `Profile0x02FixtureTests.swift` reconstruction tests. **WARNING: Modifying reconstruction, deblocking, or MC in only one side (encoder or decoder) will inevitably break this test.**

## 4. Skip Block Computation Bypass and Structural Constraints

**Constraint:** Computational bypass for skipped blocks is ONLY permitted in Layer 32. It is structurally prohibited in Layer 16 and Base 8.

- **Entropy Context Dependency:** The parent layer entropy context (e.g., `blockDecode4HParent`, `blockDecode16HParent`) reads the parent block's buffer **after** the reconstruction is executed. Because `dequantize` and inverse DWT modify the block buffers in-place, skipping the reconstruction alters the buffer contents.
- **rANS Synchronization:** If Layer 16 or Base 8 reconstruction is bypassed on only one side (e.g., encoder but not decoder), their entropy contexts will diverge, causing the rANS codec to lose synchronization and emit `lscp out of range` errors. **This is a structural constraint that cannot be fixed by adjusting indices or patching code.**
- **Layer 32 Bypass:** Because Layer 32 blocks do not serve as parents to any other layer, their reconstruction state does not affect any entropy context. Thus, skipped Layer 32 blocks can safely bypass LL subband reading, dequantization, inverse DWT, and write-back entirely.
  - Encoder: `reconstructPlaneLayer32Y`, `reconstructPlaneLayer32Cb`, `reconstructPlaneLayer32Cr`.
  - Decoder: `decodeLayer32ProcessYWithSkipMap`, `decodeLayer32ProcessCbWithSkipMap`, `decodeLayer32ProcessCrWithSkipMap`.
- **Skip Map Indexing:** The `skipMap` indices have a 1:1 grid correspondence across all layers (L2 = 32x32, L1 = 16x16, Base = 8x8), as the number of matrix elements is identical. The self-layer grid index (`row * colCount + col`) must be used directly. **Applying `/2` or `/4` transformations to the index is incorrect.**
- **Additional Optimizations:**
  - For skipped blocks, motion search and forward transform input scanning are bypassed.
  - The LTR reference subband pyramid (downscaled images for ME) remains invariant within a GOP and is cached, only recomputed at keyframes.
  - The decoder parallelizes the reconstruction process within a frame (entropy decoding remains sequential due to intra-plane dependencies).

## 5. Relationship with Deblocking Filter

1. **Edge Guarding:** Both the encoder and decoder pass the `skipMap` (and MVs) into the deblocking functions (`applyDeblockingFilter32`, `applyDeblockingFilterChroma16`). Neither side filters edges where both sides share the same non-inter mode.
2. **Computational Savings:** This skip-guard in the deblocking filter **does not affect the final output**, because the skipped block pixels are subsequently overwritten by the final skip copy anyway. Its sole purpose is to reduce computational cost (saving CPU cycles).
3. **Boundary Guards:** The deblocking edge loop includes boundary guards to exclude out-of-bounds reads when plane dimensions are multiples of 16 plus 1. Historically, out-of-bounds reading caused non-deterministic encoding outputs.

## 6. Performance Characteristics and Known Limitations

Based on user benchmarks conducted on 2026-07-30 (Comparing Profile 0x02 to Profile 0x01 with full layer decoding):

| Condition | Size | SSIM | Enc fps | Dec fps |
|---|---|---|---|---|
| miko1 300k | -14.4% | +0.0023 | +11.4% | +36.1% |
| miko1 800k | -2.6% | -0.0063 | +6.4% | +28.9% |
| ToS 300k | +1.6% | -0.0025 | -4.4% | +10.4% |
| ToS 800k | -9.6% | -0.0049 | -2.1% | +3.4% |

**Conclusions:**
1. For the target use case (game streaming with large static regions at low bitrates, e.g., `miko1 300k`), Profile 0x02 significantly outperforms 0x01 in all metrics: reduced bitrate (size), improved quality (SSIM), and increased encoding/decoding speed.
2. **Known Limitations:** At high bitrates, SSIM may be lower than 0x01 (a trade-off for bitrate reduction). For content with high motion and almost no skipped blocks (e.g., `ToS`), Profile 0x02 incurs a slight penalty due to the signaling overhead of the skip map.
3. Profile 0x02 is opt-in and should be selected when the content characteristics match its design goals.

## 7. Future Improvements
- **QStep-Conditional Skip:** Implement a more conservative skip decision at high bitrates (currently unimplemented).
