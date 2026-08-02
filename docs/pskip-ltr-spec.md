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

> [!IMPORTANT]
> **Implementation Status Note (as of 2026-08-02 investigation):** The `skip_ltr` selection described above (history verification via `staticCounters == gopPosition`) is **NOT implemented** in the current encoder. The implementation selects `skip_ltr` based solely on `allSubBlocksMatchLtr` (source-vs-LTR-source SAD below threshold) without any history check, and `skip_prev` uses an undocumented additional guard `staticCounters[i] > 3`. This divergence is the structural root cause of the artifact documented in Section 8. See Section 8 for the full root-cause analysis and the remediation plan.

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
- **Restore spec-compliant `skip_ltr` history verification:** Reintroduce `staticCounters == gopPosition` gating for `skip_ltr` (see Section 8).
- **Reconstruction-divergence check for `skip_prev`:** Add an encoder-side SAD check between the current source and the previous *reconstructed* frame so that bucket-relayed stale references are detected and the block falls back to `inter` (see Section 8).

## 8. Known Artifacts and Root-Cause Analysis (Investigation of 2026-08-02)

This section documents two **distinct** visual artifacts observed in `miko1.y4m` encoded/decoded with Profile 0x02 default settings, their separately identified root causes, and the algorithmic/logic findings that led to those conclusions. The two phenomena look similar ("ghosting / color corruption") but originate from **different defects** and must be addressed separately.

### 8.1. The Two Observed Phenomena (Distinct Causes)

**Phenomenon 1 — Frame 174 color corruption & ghosting (`skip_ltr` misjudgment):**
- Layer0/1 (low-resolution decode): severe rainbow color corruption around the batter/catcher/pitcher regions.
- Layer2 (full resolution): ghosting / afterimage at the same locations.
- Observed occurrence range: frames 155–176.

**Phenomenon 2 — Frame 1157 ghosting (`skip_prev` bucket relay):**
- In the batting-selection UI card area, content from several frames earlier remains visible as a translucent afterimage — as if the LTR reference were being displayed indefinitely.

### 8.2. Investigation Method and Established Facts

The following was established through static analysis (code reading) and dynamic verification (instrumentation + reproduction experiments) across multiple investigation agents:

- The `skipMap` is **100% bit-identical between encoder and decoder** — not a transport/protocol issue.
- No bug in the SAD computation itself: all `skip_ltr` blocks in the corrupted region had SAD values below threshold (measured 42–381 vs. threshold 768).
- The LTR reconstruction (GOP-first I-frame, ~Frame 150) is **completely intact** — the reference source is not corrupted.
- At every decoder processing stage (post-decodeBase8, post-MC, post-clamp, post-deblocking, post-final-skip-copy), chroma values are **not physically corrupted** (no clamp leak, no uninitialized memory, no out-of-bounds access).

### 8.3. Root Cause of Phenomenon 1: `skip_ltr` Misjudgment (Missing History Verification)

**Algorithmic defect:** In the current implementation (`Sources/vevc/EncodeSpatial.swift`), `skip_ltr` is selected based **only** on `allSubBlocksMatchLtr` (current-source vs. LTR-source SAD ≤ threshold). The spec's history verification (`staticCounters == gopPosition`, §2.2) is **not implemented**.

Consequences:
1. Even in moving-object regions, a block is marked `skip_ltr` whenever the current source happens to be similar to the LTR source.
2. A `skip_ltr` block is reconstructed as a 1:1 copy of the LTR reference and its residual is ignored (see `MC.swift`).
3. As time passes from the GOP head (LTR, e.g. Frame 150), moving objects shift, yet the stale LTR pixels are copied over the regions they now occupy.
4. At Layer2 this appears as ghosting. At Layer0/1 the chroma gap between the intended moving-object color (e.g. Cb≈+40, skin) and the copied past-background color (Cb≈-4, ground) saturates R/G/B to 0/255 during YUV→RGB conversion, producing the rainbow corruption.

**Quantitative instrumentation evidence:** At frame 174, `skip_ltr` accounted for an anomalous **75% (1535/2040)** of all blocks, with nearly the entire central moving-object region marked `skip_ltr`.

### 8.4. Root Cause of Phenomenon 2: `skip_prev` Bucket Relay (Reconstruction Divergence Never Checked)

**Algorithmic defect:** The skip decision uses only source-vs-source SAD (`sadPrevIn`, `sadLtrIn`) and **never verifies divergence between the copy-source *reconstructed* images (previous reconstructed frame / LTR reconstruction) and the current source.**

Ghosting mechanism:
1. In slowly-changing regions (UI fades, gentle motion), the inter-frame source SAD (`sadPrevIn`) stays below threshold every frame.
2. Initially the block is close to the LTR source, so `skip_ltr` is chosen and the LTR reconstruction (GOP head) is copied.
3. As change accumulates and the LTR-source SAD exceeds threshold, `skip_ltr` becomes ineligible — but since `sadPrevIn` is still small, the block transitions to `skip_prev` (once `staticCounters > 3`).
4. `skip_prev` copies the *previous reconstructed frame*, which — having just been `skip_ltr` — still holds the LTR image.
5. As long as skipping continues, no residual is added, so the reconstruction never tracks the changing source; the stale LTR image is bucket-relayed forward and appears as a persistent afterimage.
6. It only clears on a large change that pushes `sadPrevIn` over threshold, or on a keyframe.

**Note on bit-exactness:** Encoder and decoder reconstructions are fully bit-exact (verified by `Profile0x02FixtureTests`). This is a **design-level** defect, not a decoder bug.

### 8.5. spec-vs-Implementation Divergences (Discovered 2026-08-02)

- **§2.2 `skip_ltr` promotion:** spec mandates promotion via `staticCounters == gopPosition`; implementation uses only `allSubBlocksMatchLtr` (no history verification).
- **`skip_prev` guard:** implementation adds `staticCounters[i] > 3`, which is not documented in the spec.

### 8.6. Remediation Plan and Evaluated Alternatives

Planned direction:
- Restore spec-compliant history verification for `skip_ltr` (`staticCounters == gopPosition AND allSubBlocksMatchLtr`) so moving-object regions are never marked `skip_ltr` (fixes Phenomenon 1 at its root, with no extra SAD cost).
- Add an encoder-side "current source vs. previous reconstruction" SAD check for `skip_prev` so divergence is detected and the block falls back to `inter` (fixes Phenomenon 2).

Evaluated and rejected alternatives (measured on `miko1.y4m`, Profile 0x02):
- **Reconstruction-divergence check on *both* `skip_ltr` and `skip_prev` (scale=1):** Effective against ghosting (SSIM min +0.0086, PSNR avg +0.21 dB) but far too aggressive — Encode −40.4% fps, Decode −53.4% fps, size +9.1%, Inter ratio jumped 22.2% → 83.7%. Rejected.
- **Relaxing the reconstruction-check threshold (scale ×2/×3/×4):** Relaxing increases `skip_ltr` ratio (26% → 35% → 45%) and the artifacts return at every scale; Layer0 chroma corruption was never eliminated. A single absolute SAD threshold cannot separate gentle change from true divergence. Rejected.
- **Layer0 (Base8) 4x4 fine-grained skip decision:** Skip decision moved to Layer0 LL resolution with 4x4 sub-block evaluation to detect partial moving objects, plus `staticCounters ≤ 60` cap. **Failed and rejected** for three reasons: (1) the per-frame DWT LL extraction for skip decision (current + prev-reconstructed) duplicated the main pipeline's DWT work and *reduced* encode speed instead of improving it (cache incomplete); (2) the "all 4x4 sub-blocks must match" condition was so strict that skip ratio collapsed to Prev 0.36% / LTR 3.47% / Inter 96.17%; (3) critically, **even with skip nearly eliminated, the Layer0 chroma corruption remained** — proving the Layer0 corruption is *not* caused by skip misjudgment alone (see §8.8).

### 8.7. Final Design Currently in the Working Tree (Adopted 2026-08-02)

The working tree contains the **adopted** design, verified as the best balance:
- `skip_ltr` = `allSubBlocksMatchLtr && staticCounters[i] == gopPosition` (spec-compliant history verification restored).
- `skip_prev` = `allSubBlocksMatchPrev && allSubBlocksMatchPrevRecon && (3 < staticCounters[i])` (adds the reconstruction-divergence check via `sadPrevRecon`, threshold `blockThreshold * reconThresholdScale`, env `VEVC_RECON_THRESH_SCALE` default 1).

Measured results vs. the unfixed baseline (`miko1.y4m`, Profile 0x02): skip ratio LTR 58.85% / Prev ~2% / Inter ~39%, SSIM avg/min ≈ 0.9033–0.9050 / 0.7564–0.7594 (min improved from 0.7480), encode/decode fps within run-load variance of baseline. Frame 1157 Layer2 ghosting resolved; Frame 174 Layer2 ghosting reduced (pitcher region still slightly residual).

### 8.8. Outstanding Issue: Frame 174 Layer0 Chroma Corruption (Separate Root Cause)

**Status: UNRESOLVED, tracked as a separate issue.** The Frame 174 Layer0 rainbow chroma corruption persists across *all* skip-decision variants tested (unfixed, spec-compliant `skip_ltr`, reconstruction checks on both/neither, and the near-elimination of skip in the Layer0 4x4 attempt). Because eliminating skip did not remove it, the corruption is **not solely a skip-decision problem**. Current leading hypothesis, consistent with the user's "the two phenomena are separate" intuition:
- It is a **decoder-side Layer0 (low-resolution) decode-path issue**, not an encoder skip-judgment issue. At `maxLayer=0`, no Layer1/2 high-frequency residual is decoded, so a skip block's chroma is force-copied at low resolution without contour correction; the chroma gap then saturates R/G/B to 0/255 in YUV→RGB.
- A possible contributing factor is **chroma phase (chroma-siting) misalignment** during Layer0 down-sampling / motion-compensation copy, which would shift colors independently of skip decisions.
- **Next step (not yet done):** identify whether the corrupting Layer0 blocks are `skip` or `inter` blocks, and verify the Layer0 decode-path chroma handling (down-sampling reference, final skip copy coordinates, MC chroma phase) with targeted instrumentation.
