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
- **LTR Reference:** The LTR reference (`ltrInput` / `firstRecon`) begins as the reconstructed image of the first frame (I-frame) of the current GOP. Under periodic LTR refresh (configured via the file header `gop` field, default 12; 0 disables periodic refresh), when `0 < gop && framesSinceKeyframe % gop == 0`, the LTR reference is promoted to the current/immediately previous reconstructed frame, resetting the block static counters and history verification distance (`ltrAge`).

> [!IMPORTANT]
> **Implementation Status Note (as of 2026-08-02 investigation):** The `skip_ltr` selection described above (history verification via `staticCounters == gopPosition`) is **NOT implemented** in the current encoder. The implementation selects `skip_ltr` based solely on `allSubBlocksMatchLtr` (source-vs-LTR-source SAD below threshold) without any history check, and `skip_prev` uses an undocumented additional guard `staticCounters[i] > 3`. This divergence is the structural root cause of the artifact documented in Section 8. See Section 8 for the full root-cause analysis and the remediation plan.

### 2.3. Bitstream Elements
- **Frame Header:** For `pFrame` in Profile 0x02, the frame header (`VEVCFrameHeader`) includes `skipMapSize` (4 bytes, UInt32BE) and a `hasRefDir` flag (0x10 bit in the frame type byte).
- **Skip Map Serialization:** The `skipMap` is encoded directly into the bitstream, taking `skipMapSize` bytes. It starts with a 1-byte mode flag (`0x00` = Raw RLE via bypass, `0x01` = rANS-encoded RLE tokens with compressed frequency tables). The encoder cost-selects the smaller representation.
- **Motion Vectors (MV):** MVs are encoded alongside the `skipMap`. For skipped blocks, MV encoding is completely omitted from the bitstream. The decoder symmetrically restores the zero motion vectors by checking the `skipMap`.
- **Reference Direction:** If `hasRefDir` is true, an additional array of bits (`refDirBuf`) is serialized to indicate the reference direction (LTR or previous). In Profile 0x02, refDir bits are packed in ascending order for `inter` blocks only (`ceil(interCount / 8)` bytes, or 0 bytes if `interCount == 0`), omitting skipped blocks. The size is specified by `refDirSize`.
- **Skip-Conditional BlockFlags (2026-08-19):** the per-plane BlockFlags bitmaps (isZero at L2/L1, isZero+split at L0) carry bits for non-skip blocks only; skip blocks' flags are implied by the all-zero rule. This removed the dominant container fixed cost (measured 1,576 B/frame = 713 kbps even with all-zero coefficients; −341 kbps at miko 500k with ~53% average skip). Same inter-only packing pattern as refDir.
- **MV Spatial Prediction (2026-08-19):** Profile 0x02 P-frame MV data begins with a 1-byte mode flag (`0x00` = raw values, `0x01` = spatial prediction residuals via median(A, B, C)). The encoder cost-selects the smaller payload. Decoders restore reconstructed MVs sequentially to feed subsequent spatial predictions; skipped neighbor blocks use (0,0).
- **Weighted-Prediction Offsets (2026-08-18):** Profile 0x02 P-frame headers carry two mandatory Int8 fields (`lumaOffset`, `chromaOffset`) placed after `refDirSize` and before the layer sizes (layer-dropping readers never reach past them; a stream without the bytes fails to parse). The decoder forms `P′ = P + offset` immediately after MC and before clamp/deblock at every prediction site (full-resolution plane, LL2/LL1 slot predictions, L0 chain) on **prev-referenced inter blocks only** — skip blocks, intra blocks (MV sentinel 32767), and LTR-referenced blocks (`refDir` set) are excluded, because the offset is estimated against the prev reference. The same value applies at all layers (the LeGall 5/3 lowpass is mean-preserving). The encoder (`estimateLumaOffset`, SAD.swift) signals a nonzero offset only when the per-32px-block mean shift is spatially uniform (block-mean std ≤ 12): a localized flash fails the gate — at saturated qsteps a misapplied global offset cannot be repaired by the residual (measured −2〜−3 dB on flash-exit frames). Measured on miko 500k: fade frames +0.3〜+1.1 dB, non-fade content inert (ToS signals zero offsets; the bitstream differs only by the 2 header bytes/P-frame).
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
4. **Function-variant identity:** every reconstruction-path call (deblock, MC, layer reconstruction) must invoke the **identical function variant with identical parameters** on both sides. Three real asymmetries of this class were found and fixed in 2026-08: the encoder's P-frame deblock used the plain variants while the decoder used the intra-boundary-enhanced + skip-gated ones (95d7994); the encoder reconstructed skip-block regions that the decoder leaves zero until the final skip copy, diverging deblock edge reads at mixed inter|skip edges (9d491b4); the encoder's I-frame chroma deblock used a different function than the decoder's mvs==nil branch (9d491b4). Each diverged the reconstructions by ±1 and, once the entropy contexts started depending on reconstruction state, desynchronized the backward-adaptive tables.
5. **Validation:** This bit-exactness is permanently verified by the `Profile0x02FixtureTests.swift` reconstruction tests, and by `L0HistoryDiagTests` which locksteps the full reconstruction, the L0 reference chain, and every backward-adaptive history state between encoder and decoder on real 1080p content. **WARNING: Modifying reconstruction, deblocking, or MC in only one side (encoder or decoder) will inevitably break these tests.**
6. **L0 closed loop (normative, Profile 0x02):** Base8 codes `r0 = LL2(source) − MC_L0(L0_ref)` (One-Pyramid Wave 1, default since 2026-08-17) and both sides maintain a quarter-resolution L0 reference chain that closes over the bitstream alone — the `maxLayer=0` output is bit-exact with the encoder's L0 reconstruction. Applies to every P-frame with an L0 reference from a preceding I-frame; the semantics are fixed by the profile, not signaled per-stream. Measured cost at miko 500k: +6.7% size / −0.26 dB full-resolution, accepted for the layer-drop capability.

## 4. Skip Block Computation Bypass (All Layers)

**Current rule (since 2026-08-17, One-Pyramid Waves 2+4):** skipped blocks bypass computation at **every** layer, on both sides. This is legal because profile 0x02 codes its AC entropy contexts **parent-free** (`ParentFreeContext.swift`): no layer's entropy decode reads another layer's block buffers, so reconstruction state can no longer desynchronize the rANS streams.

- **Normative basis:** skip blocks carry all-zero coefficients by construction (the encoder zeroes their residual before the DWT). Dequantization and the integer LeGall lifting both map zero to zero, so "reconstruct" and "bypass" produce identical pixels for these blocks.
- **Encoder bypass:** `extractSingleTransformBlocks32/16/Base8` skip the block read, zero-scan, DWT and quantization for skip blocks; `reconstructPlaneLayer32*` skip their reconstruction. Coded streams are unchanged (verified SHA1-identical).
- **Decoder bypass:** `decodeLayer32Process*WithSkipMap` and `decodeLayer16Process*WithSkipMap` skip reconstruction entirely (layer1 only when layer2 is present — otherwise layer1 is the display output and every block reconstructs); `decodeBase8Process*WithSkipMap` copy the cleared block views without dequant/IDWT (bit-exact at every `maxLayer`). Decoded outputs are unchanged (verified SHA1-identical).
- **Skip Map Indexing:** luma grids of all layers are `ceil(dx/32) × ceil(dy/32)` and map 1:1 to the skip map — use the self-layer grid index (`row * colCount + col`) directly; `/2` or `/4` transformations are incorrect. Layer1/Layer2 **chroma** blocks are 1:1 with *each other* and use the same direct index; only Base8 chroma blocks span 2×2 skip-map entries geometrically and bypass only when all four are non-inter.
- **Additional Optimizations:**
  - For skipped blocks, motion search and forward transform input scanning are bypassed.
  - The LTR reference subband pyramid (downscaled images for ME) remains invariant within a GOP and is cached, only recomputed at keyframes.
  - The 9 coefficient streams (3 layers × 3 planes) are independent and can entropy-decode in parallel (`parallelEntropy`, latency mode). Under GOP-parallel throughput decoding the fan-out only adds overhead, so the GOP-parallel `Decoder` uses the sequential order.

**Historical constraint (superseded):** until 2026-08-14, profile 0x02 shared the parent-conditioned entropy contexts with profile 0x01 — the layer1/Base8 block buffers were read (post in-place dequant/IDWT) as the next layer's rANS context, so bypassing their reconstruction on one side desynchronized the codec (`lscp out of range`). The parent-free context change (measured smaller streams on every configuration under the backward-adaptive tables) removed that dependency; profile 0x01 still has it, and keeps the layer-32-only bypass rule.

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

**Update (2026-08-17, remote 3-run, miko1 1801f 1080p60 @500k):** encode 9.41 ms/frame (target ≤10 met; was 11.4 before the reconstruction-symmetry fixes and the all-layer skip bypass), full decode 1.39 ms/frame (target ≤1.10 still open), layer-1-only decode 0.44 ms, layer-0-only 0.15 ms. PSNR 31.11 / SSIM min 0.7617.

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

### 8.8. Frame 174 Layer0 Chroma Corruption — Root Cause Identified and FIXED (2026-08-02)

**Status: ROOT CAUSE IDENTIFIED, FIX APPLIED (decoder `DecodeSpatial.swift`).**

**Investigation findings (dynamic instrumentation):**
- The corrupting Layer0 blocks were **99.7% `inter` blocks** (not `skip`), proving this is a **decoder Layer0 decode-path bug**, completely independent of skip decisions. This confirmed the user's "the two phenomena are separate" intuition.
- Chroma first corrupts **immediately after Layer0 motion compensation**: after `decodeBase8` the residual chroma is normal, but after Layer0 MC the Cr value exceeds the signed 8-bit range (reaching +181), then `clampPlane` saturates it to −128/+127, which shows up as rainbow noise in YUV→RGB.

**Root cause — double right-shift in Layer0 Chroma MC:**
- MVs are searched and stored at **Layer0 (480×270) quarter-pixel precision**.
- At `maxLayer=0` decode, `DecodeSpatial.swift` passed `mvShift: 2` to the Chroma MC path. `scaledMV(>>2)` therefore pre-shifted the MV, and then `subMCBlockChroma16` (which is hard-coded to additionally apply `>>3` / `&7` assuming 1/8-pixel input) applied a *second* shift — a total of `>>5` (divide by 32). The Chroma reference coordinates collapsed to ~1/4 of their true location, so completely unrelated chroma pixels were added to the Base8 residual, overflowing the signed range and saturating on clamp.
- **Why Luma was unaffected:** `addMCBlockLuma32` applies `>>2` / `&3` (quarter-pixel assumption), which composes correctly with `scaledMV(>>2)`. Only Chroma (hard-coded `>>3`) suffered the double shift.
- **Layer1 is NOT affected:** at Layer1, `mvShift: 1` gives `scaledMV(>>1)` + `subMCBlockChroma16(>>3)` = `>>4`, which is exactly correct for the Layer1 chroma plane resolution.

**Fix (decoder only, minimal):** in `DecodeSpatial.swift`'s Layer0 MC path, **Chroma `mvShift` changed from `2` to `0`** (both `applyScaledBidirectionalMotionCompensationChroma` and `applyScaledMotionCompensationChroma`); **Luma stays at `mvShift: 2`**. Luma/Layer1/Layer2 are untouched.

**Verified result:** the severe rainbow corruption at Frame 174 Layer0 is eliminated; Luma stays intact; Layer1/2 unaffected. Stream size / PSNR / SSIM are bit-identical to before the fix (decoder-only change, encoder output unchanged); all 32 tests PASS; `Profile0x02FixtureTests` unaffected (they exercise `maxLayer=2`).

**Remaining (separate, minor):** a faint pitcher-region ghost still visible at Layer0/1. This is **not** the chroma MC bug — it comes from `skip_ltr`/`skip_prev` reference copying combined with the structural fact that `maxLayer=0/1` decode reads no high-frequency residual to correct the copied reference, plus the operator mismatch between the encoder's full-resolution loop and the decoder's layer0 MC approximation. **Update 2026-08-14: the layer0 half of this is structurally solved by the L0 closed loop (One-Pyramid Wave 1, §3 invariant 6) — the `maxLayer=0` output becomes bit-exact with the encoder. Update 2026-08-17: the loop is now the Profile 0x02 default (the +6.7% size / −0.26 dB cost at 500k was accepted after visual comparison; the former `enableL0Loop` flag and `VEVC_L0LOOP` env are removed).**

---

## 9. Performance Knowledge (Accumulated from 2026-08-02 Investigations)

This section records hard-won performance facts: where the compute cost actually is, and where a careless change silently makes things *slower* (or breaks them). Read this before attempting any optimization work on Profile 0x02.

### 9.1. Compute-Heavy Hot Spots

- **Skip-decision SAD computation** (`EncodeSpatial.swift`, full-resolution 32×32 block / 16×16 sub-block evaluation): ≈ **9.40 M Ops/frame** baseline. Adding a reconstruction-divergence SAD check (`sadPrevRecon`) roughly **doubles** this — which is why the adopted design applies the check to `skip_prev` *only* (not `skip_ltr`).
- **DWT LL extraction** (`extractSingleTransformSubband32` / `extractSingleTransformSubband16`): expensive. Re-running it per frame *just* for a side computation (e.g. a Layer0 skip decision) duplicates the main pipeline's DWT work and **reduces** encode speed. Any such use must share / cache with ME (`cachedNextSub1`, `cachedNextSub2`) — this is exactly why the Layer0 4×4 skip-decision attempt failed on speed (§8.6).
- **Decoder per-layer pipeline**: Layer0/1/2 MC (`applyScaled*MotionCompensation*`), deblocking (`applyDeblockingFilter*`), final skip copy (`copyBlockPointer` / `copyBlockSafe`).
- **MC sub-pixel FIR interpolation** (`addMCBlockLuma32`, `subMCBlockChroma16`): heavy fractional-pixel filtering on inter blocks.

### 9.2. Scene-Cut Detection (2026-08-18, encoder-only)

- A hard cut crossed by a P-frame at saturated qstep leaves un-repairable ghosts of the previous scene (miko 548→549: megaphone/text outlines persisted for a full GOP), and the LTR keeps pointing at the old scene until the next periodic I. `detectSceneCut` (SAD.swift) separates cuts from flashes/fades by **sign mix**: a cut replaces content so per-32px-block mean diffs go both ways (548→549 minority-side fraction 0.169), a flash/fade shifts one way (all measured flash/fade transitions: 0.000). MAD alone cannot make this call (cut 39.7 vs flash 38.5).
- A cut-driven I-frame does **not** reset the keyint grid phase (`framesSinceKeyframe` restarts at `frameIndex % keyint`, staticCounters start from the same base): resetting shifted every later I and cost min-SSIM −0.05 on miko when the flash plateau lost its adjacent I.
- `sceneChangeThreshold` above `maxEstimateFastSAD` (765) disables all scene detection — deterministic tests rely on this (L0BitExactTests passes 1e9).
- Measured (miko 500k): two cuts detected (214, 549 — both verified true cuts), ghosts eliminated, cut-GOP SSIM +0.025, global avg +0.0013, size −14 KB; ToS bit-identical (no cuts); miko 2500k min +0.0036.

### 9.3. σ-Normalized Adaptive Quantization (2026-08-18, encoder-only)

- SSIM weights a coded error by 1/(2σ²+C) (local-variance masking). Per-32px-block source-luma variance (`computeBlockActivityMap`, SAD.swift — grid 1:1 with the skip map) classifies blocks; TEXTURED blocks quantize layer-2/1 luma HL/LH/HH with a widened dead zone (`qMidTextured`/`qHighTextured`, Quant.swift). The signaled step never changes, so the bitstream format and decoder are untouched; `VEVC_AQ_BIAS=0` reproduces the fixed-dead-zone bitstream exactly.
- **Saturation-ramped** like the qHigh extension (zero at baseStep ≤ 2048, full at 4096): unramped widening measured miko 2500k avg −0.0016 (real detail loss at fine steps); ramped, the 2500k bitstream is byte-identical to no-AQ.
- **The FLAT side gives nothing**: biasing flat blocks toward round-to-nearest costs bits with no SSIM return (swept at multiple thresholds). All gain is on the textured side — dropping masked near-dead-zone coefficients lets rate control reinvest the bits across frames.
- **Variance cannot separate text from texture**: HUD text strokes are high-variance but unmasked. Full-strength widening (1.0 step, σ² ≥ 600) measured best on metrics (min +0.0030 / −16.6% size) but visibly eroded scoreboard text; HH-only widening avoids the text but loses the entire gain (saturated HH is already dead — everything lives in HL/LH). Adopted: 0.25-step delta at σ² ≥ 1600, which only drops coefficients near the dead-zone edge — text strokes (large coefficients) are untouched, verified visually on frames 619/549.
- Measured (miko 500k): min-SSIM +0.0018, avg +0.0006, size −4.5%; ToS and miko 2500k bit-identical.

### 9.4. Average-Bitrate Rate Control (2026-08-19, encoder-only)

- `-b` tracks an average-bitrate target: `plannedBitrate = specified × 1.3` (layer allowance — the stream carries 3 decodable layers; the specified value prices the full-resolution layer). Mechanisms, all in RateController.swift: (1) cumulative consumed-vs-planned balance feeds the next GOP budget, clamped to `[baseGOPBits/4, baseGOPBits×2]` so one overrun spreads its recovery over several GOPs; (2) when the in-GOP consumption pace exceeds plan (`budgetRatioQ8 < 256`), the distortion-equalization blend and the EMA smoothing are skipped — budget-derived qstep correction takes priority; (3) when the budget is tight (`budgetRatioQ8 < 192`), maxStep releases from the baseStep anchor to 16384 (the table clamps at 4096, so this means "as coarse as the format allows"); (4) I-frames use closed-loop proportional prediction from the previous I-frame's (bits, step) sample (`calculateIFrameQStep`) — the open-loop `estimateQuantization` measured 5× too fine on live content (332KB against a 65KB budget); the stream-start I-frame has no sample and uses a conservative absolute seed `max(512, estimate)` (real step 32), corrected from the second I-frame on.
- Measured tracking (miko_700): 4000k → 1.320×, 8000k → 1.308×, 16000k → 1.324× (all ≤1.35× gate); 2500k → 1.357× (accepted, at the edge of the quantizer-limited minimum); 500k/1100k clamp at the minimum output size (~2.8 Mbps on this content, down from ~4.0 Mbps pre-ABR because pace-priority sustains coarser steps and the I-frame loop cut I spend from 43.9% to ~24% of bytes). ToS 500k: 862 → 641 kbps.
- Iso-rate quality (both ≈10.4 Mbps actual): SSIM min 0.4734 → 0.4768 (+0.0034), avg −0.0076 — rate-led allocation does not cost worst-frame quality.
- **BASELINE RESET (user decision 2026-08-19): all pre-ABR quality measurements at "-b 500" were actually measured at ~4.0 Mbps output and are NOT comparable to post-ABR "-b 500" runs (~2.8 Mbps output). The ABR behavior is the new baseline; do not compare min/avg SSIM across this boundary.**

### 9.5. Pitfalls — Changes That Silently Make It Slower (or Break It)

- **Reconstruction check on *both* `skip_ltr` and `skip_prev`:** −40% encode / −53% decode fps, +9% size. **Rejected.** Apply to `skip_prev` only.
- **Layer0 4×4 fine-grained skip decision:** per-frame duplicate DWT LL extraction → *slower* despite lower per-op count; skip ratio collapsed to ~3% → larger size. **Rejected.**
- **`skipMap` granularity must stay at 32×32** (1:1 grid across layers). Changing it forces bitstream + decoder changes; judge any such change as high-cost.
- **MV scale conversion is Luma/Chroma-asymmetric:** `scaledMV(mvShift)` + `addMCBlockLuma32(>>2/&3)` vs `subMCBlockChroma16(>>3/&7)`. **`mvShift` must be validated separately for Luma and Chroma** — the Layer0 chroma corruption (§8.8) was a double-shift from assuming one value fit both. Never change `mvShift` without measuring both planes at every layer.
- **Decoder-only changes** (e.g. the §8.8 chroma MC fix) do **not** change encoder output: stream size / PSNR / SSIM stay bit-identical, and `Profile0x02FixtureTests` (which exercise `maxLayer=2`) are unaffected. Verify which side a change touches before predicting its metric impact.
- **Benchmark variance:** encode/decode fps fluctuate with machine load (observed Enc 72.9–99.7 fps for the *same* code). Always compare against a same-session baseline (`compare -y4m … -vevc-only -profile 2`), never across days/machines.
