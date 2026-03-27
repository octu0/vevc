# vevc


> [!IMPORTANT]
> Work In Progress


**vevc** is a high-speed video compression format, extending the high-efficiency, multi-resolution image format [veif](https://github.com/octu0/veif) (velocity image format) with Temporal DWT, Spatial 2D-DWT, and SIMD-optimized entropy coding.

![figure0](docs/fig0.jpg)

## Features

1. **Full Spatio-Temporal DWT Architecture**
   - **No Block-based Prediction**: Unlike conventional codecs (H.264/HEVC) that rely heavily on complex block-matching (Motion Vectors) and intra-frame prediction, `vevc` employs a unified DWT approach across all three dimensions (time + 2D space). 
   - **Temporal DWT**: LeGall 5/3 1D-DWT across frames (e.g., GOP=4). It transforms the time axis directly, smoothly decomposing temporal changes into low-frequency (smooth motion) and high-frequency (details) subband frames. This eliminates the need for expensive motion estimation.
   - **Spatial DWT**: LeGall 5/3 2D-DWT with multi-resolution Layers (Layer 0, 1, 2) inherited from `veif`. Each temporal subband frame is cleanly decomposed into spatial frequency layers.
   - **SIMD8-Optimized**: Both Temporal lifting and Spatial DWT use `SIMD8<Int16>` vectorization for massively parallel pixel processing with `UnsafeRawPointer.load/storeBytes` direct memory access.

2. **Multi-Resolution Design**
   - At decode time, you can extract specific spatial resolutions from a single file depending on your needs. This enables flexible, highly efficient video delivery suited to network bandwidth and device capabilities without storing multiple video files.

   **Extraction Patterns (assuming a 1080p source):**

   | Target Use Case           | Spatial (`-maxLayer`) | Result Output            | Server-Side Action             |
   | :------------------------ | :-------------------- | :----------------------- | :----------------------------- |
   | **Max Quality (Archive)** | `2` (Layer 0,1,2)     | 1080p                    | No extraction (transfer as is) |
   | **Medium (Preview)**      | `1` (Layer 0,1)       | 540p                     | Skip Layer 2                   |
   | **Ultra Low (Thumbnail)** | `0` (Layer 0 only)    | 270p                     | Skip Layer 1, 2               |

3. **Acceleration via Concurrency & SIMD**
   - Temporal subband frames are encoded/decoded in parallel (4-way `TaskGroup`).
   - Spatial DWT, plane matching, shifting, and difference calculations are fully vectorized.
   - Temporal DWT lifting is SIMD8-optimized with scalar tail handling.

---

## Data Layout

`vevc` encodes video using Temporal GOP (Group of Pictures) of 4 frames, processed through a temporal-spatial wavelet pipeline.

**Bitstream Structure:**

```
                           VEVC File Structure
+-------------------+------------+-----------------+-----+-------------+
| Magic 'VEVC' (4B) | Metadata   | GOP (0..3)      | ... | GOP (tail)  |
+-------------------+------------+-----------------+-----+-------------+

    Metadata (Profile 1)
+---------------------------------------------+
| Metadata Size (2B) | Profile Version(1B)    |
+------------+-------+-----+------------------+----------+----------------+
| Width (2B) | Height (2B) | Color Gamut (1B) | FPS (2B) | Timescale (1B) |
+------------+-------------+------------------+----------+----------------+
  Color Gamut: 0x01=BT.709, 0x02=BT.2020
  Timescale:   0x00=1000ms, 0x01=90000hz

    Temporal GOP (GOP=4, nLow=2) or I-Frame GOP (GOP=1, nLow=0)
+----------------+-----------------+-------------+
| Data Size(4B)  | GOP Size X (4B) | nLow X (2B) |
+----------------+-------------+---+-------------+--+
| F0 len (4B) | F0 (Low0 spatial)  | F1 len (4B) | F1 (Low1 spatial)  |
+-------------+--------------------+-------------+--------------------+
| F2 len (4B) | F2 (High0 spatial) | F3 len (4B) | F3 (High1 spatial) |
+-------------+--------------------+-------------+--------------------+

    Spatial Frame (3 Layers structure)
    +---------------------------------------------------------+
    | L0 len (4B)  | Layer 0 Payload (8x8 base)               |
    +--------------+------------------------------------------+
    | L1 len (4B)  | Layer 1 Payload (16x16 refinement)       |
    +--------------+------------------------------------------+
    | L2 len (4B)  | Layer 2 Payload (32x32 refinement)       |
    +--------------+------------------------------------------+

        Layer Payload
        +-------------+-----------+
        | qtY (2B)    | qtC (2B)  |
        +-------------+-----------+
        | Y len (4B)  | Y data    |
        +-------------+-----------+
        | Cb len (4B) | Cb data   |
        +-------------+-----------+
        | Cr len (4B) | Cr data   |
        +-------------+-----------+
```

---

## Performance

*(Tested with Tears of Steel 1080p, 1802 frames, target 500 kbps)*

### Speed & Size

![speed_size](docs/speed_size.png)

SW: Software, HWA: Hardware Acceleration

### PSNR

![psnr](docs/psnr.png)

### SSIM

![ssim](docs/ssim.png)

| Codec | Encode (ms/f) | Decode (ms/f) | Size (KB) | SSIM Avg | SSIM Min |
|-------|---------------|---------------|-----------|----------|----------|
| **VEVC (Layers)** | **1.53** | **0.96** | **24,256** | **0.9070** | **0.8298** |
| H.264 HWA | 2.54 | 0.30 | 1,856 | 0.9282 | 0.8375 |
| HEVC HWA | 2.68 | 0.29 | 1,859 | 0.9508 | 0.8698 |
| HEVC SW | 14.23 | 0.26 | 1,736 | 0.9649 | 0.9399 |
| MJPEG | 0.68 | 1.30 | 159,189 | 0.9875 | 0.9787 |

---

## Entropy Coding: Interleaved rANS

`vevc` uses **Interleaved 4-way rANS (Asymmetric Numeral Systems)** for entropy coding. rANS provides near-optimal compression and enables SIMD-parallel decoding, unlike CABAC which is inherently serial.

### Architecture

```
DWT Coefficients
       │
       ▼
  Zero-Run RLE         ┌─── Raw Mode (≤32 non-zero coeffs)
  (run, value) pairs ──┤
       │               └─── rANS Mode  
       ▼                      │
  ValueTokenizer              ├── runModel (zero-run tokens)
  token + bypass bits         ├── valModel (value tokens)
       │                      └── 4-way Interleaved stream
       ▼
  Interleaved 4-way rANS Encoder
  (4 independent states, shared stream)
```

### Key Components

| File | Role |
|------|------|
| `rANS.swift` | Core rANS encoder/decoder, Interleaved 4-way variants, Bypass I/O, probability model with O(1) LUT |
| `EntropyCodec.swift` | `VevcEncoder` / `VevcDecoder`: Zero-Run RLE, raw fallback, compressed freq tables |
| `ValueTokenizer.swift` | Token/bypass decomposition for signed/unsigned values |
| `rANSCompressor.swift` | Standalone rANS compression for generic byte data |

### Optimizations

- **Interleaved 4-way**: 4 independent rANS states decoded in round-robin, enabling future SIMD4 parallelism
- **O(1) Token Lookup**: 16384-entry LUT for instant cumulative-frequency → token resolution
- **Zero-Run RLE**: DWT zero coefficients compressed as run-length tokens
- **Raw Fallback**: Blocks with ≤32 non-zero coefficients skip rANS overhead entirely
- **Compressed Frequency Tables**: Bitmap-based encoding reduces table size from 32B to ~10B

---

## CLI Usage

The `vevc` package includes command-line tools: `vevc-enc` (encoder) and `vevc-dec` (decoder).

### Encode (`vevc-enc`)

Takes a `y4m` format file as input and outputs the encoded `vevc` binary file. Standard Input (`-`) is also supported for piping.

```bash
$ swift run -c release vevc-enc -i input.y4m -o out.vevc
```

- `-i <path|->`: Specifies the input `.y4m` file path or standard input (`-`).
- `-o <path|->`: Specifies the output `.vevc` file path or standard output (`-`).
- `-b <kilobit>`: Specifies the target bitrate (desired compression ratio/quality) in kilobit per second.
- `-keyint <keyint>`: Specifies the keyframe interval (maximum GOP size, automatically falls back to I-Frame for scene changes or end of stream).
- `-zeroThreshold <threshold>`: Sets the threshold for treating DWT coefficients as zero (reduces size by aggressively skipping noise).
- `-sceneThreshold <sad>`: Sets the SAD threshold for scene change detection (forces an I-frame when temporal changes are too massive).

### Decode (`vevc-dec`)

Takes a `vevc` format file as input and outputs the decoded `y4m` video stream. Standard I/O (`-`) is also supported.

```bash
$ swift run -c release vevc-dec -i output.vevc -o output.y4m
```

**Multi-Resolution Options**:

- `-i <path|->`: Specifies the input `.vevc` file path or standard input (`-`).
- `-o <path|->`: Specifies the output `.y4m` file path or standard output (`-`).
- `-maxLayer <0-2>`: Specifies the maximum level of spatial layers to decode.
  - `0`: 1/4 size (for rough thumbnails)
  - `1`: 1/2 size (for previews)
  - `2`: Original size (default)

---

## Internals

The core components of the implementation consist of the following files:

- `TemporalDWT`: SIMD8-optimized LeGall 5/3 temporal wavelet transform across GOP=4 frames. Produces temporal low/high subband frames for improved compression.
- `DWT`: Spatial LeGall 5/3 2D-DWT with SIMD-optimized lifting steps. Supports both 4-element (temporal) and 8-element (spatial) transforms.
- `Encode` / `Plane`: The encoding flow that uses plane data (`PlaneData`) to process temporal subband frames and individual I-Frames through the spatial 2D-DWT and entropy coding pipeline.
- `rANS` / `EntropyCodec`: Interleaved 4-way rANS entropy coding engine with adaptive token-based probability modeling, O(1) LUT decoding, and raw fallback for sparse data.

## License

MIT
