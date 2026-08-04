# Profile 0x02 Update: Design Specification (B' — "One Pyramid, Two Loops")

Status: DRAFT for review (2026-08-04). Applies to Profile 0x02 only; no backward compatibility retained (profile is in development).

## 1. Goals

- Quality/speed above H.264; within one step of HEVC for the target content class (game streaming, large static regions, low bitrate).
- Performance targets (remote reference machine): encode ≤ 10 ms/frame, decode ≤ 1.10 ms/frame (1080p, Profile 0x02, full-layer decode). Current: 11.4 / 1.53.
- Structural fix for L0/L1 moving-object chroma speckle (no corrective residual exists at L0 today).
- Design principle: components must compose ("harmony") — one decision point should serve quality, speed, and parallelism simultaneously.

## 2. Measured evidence driving this design (2026-08-04, mba, reports in `.tmp/`)

| # | Question | Result | Consequence |
|---|---|---|---|
| 1 | Can detail subbands (L1/L2 = 55% of bits) drop temporal prediction? | **No.** miko1 +102%, prod +165% total size | Full-res MC and temporal detail prediction are retained |
| 2 | Can the skip decision move to LL2 resolution? | **No** for target content. miko1 agreement 36–47% | Full-res SAD skip decision retained (already optimized) |
| 3a | Cost of removing parent-layer entropy context? | **+0.21–0.37%** size (8-context replacement) | Parent dependency removed — unlocks §5, §6 |
| 3b | 4-way interleaved rANS throughput | **3.22×** decode | Adopted |
| — | 9-stream split overhead | 36 B/frame | Adopted |

## 3. Architecture overview

Two nested closed loops share one DWT pyramid per frame:

```
                 ┌─ skip decision (full-res SAD, lazy)     [unchanged]
input frame ─────┤─ ME on LL pyramid                       [unchanged]
                 └─ DWT pyramid (computed once, shared)
L0 loop (NEW):    LL2(source) − MC_L0(L0_ref) → q → rANS → L0_recon   (bit-exact at L0)
Full loop:        details(source − MC_full(full_ref)) → q → rANS      (as today, levels 1 and 0)
                  LL2 slot of full reconstruction := L0_recon − LL2(P)  (see §4)
```

- References: the decoder/encoder both maintain a small **L0 reference plane** (LL2-sized recon) in addition to the full-res reference. LTR keeps both forms.
- Entropy: 9 independent rANS streams (3 layers × 3 planes), 4-way interleaved, same-layer contexts (§6).
- Skip blocks bypass dequant/IDWT/entropy at **all** layers on both sides (§5).

## 4. The two-loop nesting (core design)

Let `P = MC_full(full_ref)` (pixel-domain full-res prediction, as today) and `R = source − P`.

Today Base8 codes `LL2(R)`; the L0-only decoder approximates `LL2(R) + MC_L0(...)` with mismatched operators → speckle.

**Change:** Base8 codes `r0 = LL2(source) − MC_L0(L0_ref)` instead.

- L0-only decoder: `L0_recon = MC_L0(L0_ref) + deq(r0)` — self-contained, **bit-exact with encoder** (speckle structurally eliminated; L0 deblock/skip-copy defined identically on both sides).
- Full decoder: needs the LL2 coefficient slot value `LL2(R) = LL2(source) − LL2(P) ≈ L0_recon − LL2(P)`. It computes `LL2(P)` by a low-pass-only 2-level analysis of `P` (inter blocks only; skip blocks are copied and bypass everything). Then IDWT with decoded details and add `P` as today.
- Extra cost: LL-only analysis of `P` on both sides ≈ 1.25 low-pass passes over inter blocks only (~few % at current inter ratios; bounded).
- Risk: `r0` may cost slightly more bits than `LL2(R)` (MC_L0 predicts LL2 a bit worse than LL2∘MC_full). **Gate: measure in Wave 1; budget +2% total size.**
- Fallback (Option C, additive): keep coding `LL2(R)` and add a small L0 correction stream only where encoder-simulated L0 drift exceeds a threshold. Lower risk, slightly worse harmony; switch if Wave-1 gate fails.

## 5. Skip-block bypass at all layers

Enabled by §6 (no parent context → reconstruction state of a skipped block is never read as context).

- Encoder: for `skip_prev`/`skip_ltr` blocks — no MC subtract, no DWT, no quant, no entropy, at L2/L1/L0 alike. Final skip copy provides pixels; L0_ref slot copied from reference L0 plane.
- Decoder: symmetric bypass; skip copy at each decodable layer from the layer-matched reference plane (L0 copies from L0_ref — removes today's approximation for skip blocks at L0/L1 as well).
- Expected effect at miko1-class content (~60% skip): large cut into encoder DWT (28% of enc) and decoder L1/L2 decode (35% of dec); deblocking already skip-guarded.

## 6. Entropy coding (rANS) redesign

- **Contexts (8)**: 3 binary features — co-located LL2 coefficient nonzero, left-neighbor block nonzero-count>0, previous coefficient nonzero. No inter-layer references. Measured cost: +0.21–0.37%.
- **Streams**: 9 independent streams per frame `(layer ∈ {0,1,2}) × (plane ∈ {Y,Cb,Cr})`, each with a 4-byte VLQ size header (36 B/frame). Frame header keeps per-layer totals for fast maxLayer truncation.
- **Interleave**: 4-way rANS within each stream (3.22× decode throughput measured).
- **Decode schedule**: all 9 streams entropy-decode in parallel; per-layer reconstruction proceeds as data arrives (L0 first for earliest preview).

## 7. Bitstream changes (DataLayout)

```
[FrameHeader v2]
  frameType(1) | skipMapSize(4) | mvsSize(4) | refDirSize(4)
  streamCount(1)=9 | per-stream: {layerPlaneId(1), size(VLQ)}
[skipMap][mvs][refDirs]
[stream 0: L0/Y] [stream 1: L0/Cb] ... [stream 8: L2/Cr]
```
- L0 streams carry `r0` residuals (§4). Detail streams carry level-1/level-0 subbands as today.
- No compatibility with previous Profile 0x02 bitstreams (version byte bump in profile signaling).

## 8. Performance budget (from measured stage costs)

| Stage | Enc today | Enc after | Dec today | Dec after |
|---|---|---|---|---|
| Skip SAD + ME | 4.3 ms (38%) | 4.3 (kept; already optimized) | — | — |
| DWT/quant | 3.2 (28%) | ~1.6 (skip bypass) | in layers | ~half |
| rANS | 1.7 (15%) | ~0.9 (4-way, parallel) | serialized today | parallel ×3.22 |
| MC | 1.4 (12%) | ~1.0 (skip-aware + L0 loop) | 0.46 (30%) | ~0.35 |
| Deblock | 0.5 | 0.5 | 0.31 (20%) | 0.31 |
| **Total** | **11.4** | **≈ 8.5–9** | **1.53** | **≈ 0.9–1.1** |

Remaining enc gap to well-below-10 is Phase-3 material (inter-pipeline SoA/arena — orthogonal, unchanged plan).

## 9. Implementation waves (each with a hard verification gate)

| Wave | Content | Gate (must pass before next wave) |
|---|---|---|
| 1 | L0 closed loop (§4) behind the new version byte; L0_ref plane maintenance both sides | L0 bit-exact enc/dec (shasum); total size ≤ +2%; visual check by coordinator (f174 etc.); all tests |
| 2 | rANS context replacement + 9 streams + headers (§6, §7) | size within +0.5% of Wave-1; enc/dec outputs bit-exact vs Wave-1 semantics; round-trip fixtures |
| 3 | 4-way interleave + parallel entropy decode | throughput gain on-target; outputs bit-exact vs Wave 2 |
| 4 | Skip bypass at all layers, both sides (§5) | outputs bit-exact vs Wave 3 (bypass is lossless by construction); enc/dec ms/frame vs budget |
| 5 | Docs: README.md, docs/DataLayout.md, docs/pskip-ltr-spec.md rewrite (constraints §4-old removed, new invariants) | docs reviewed by user |

Verification protocol per wave: implement → self-verify → independent critical verifier → coordinator visual/numeric check (strict image rules) → user-approved commit. Rebuild after every source restore (stale-binary rule). Same-session interleaved benchmarks only.

## 10. Documentation updates (mandatory, user directive)

- `README.md`: profile 0x02 feature description, decode parallelism, preview-exactness claim.
- `docs/DataLayout.md`: FrameHeader v2 + 9-stream layout (§7).
- `docs/pskip-ltr-spec.md`: §3 invariants extended with L0 loop; §4 parent-context constraint REMOVED (superseded by §6 contexts — keep a historical note); §8 speckle sections updated to "structurally fixed by L0 loop".
