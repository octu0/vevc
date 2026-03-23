# vevc


> [!IMPORTANT]
> Work In Progress


**vevc** is a high-speed video compression format, extending the high-efficiency, multi-resolution image format [veif](https://github.com/octu0/veif) (velocity image format) with Pixel-Domain Macroblock-based Motion Estimation (MBME) and Residual 2D-DWT.

![figure0](docs/fig0.jpg)

## Features

1. **Pixel-Domain Prediction & 2D-DWT**
   - **Spatial**: LeGall 5/3 2D-DWT (Supports multiple resolutions via Layer 0, 1, 2) similar to `veif`.
   - **Temporal**: Macroblock-based Motion Estimation (MBME) to predict pixel movement using robust 16x16 blocks, followed by Interleaved rANS entropy coding and 2D-DWT applied only to the residual (difference) frame, achieving ultra-fast decode speeds.
   - **Quadtree Variable Macroblocks**: Dynamically splits blocks (e.g., from 32x32 down to 8x8) based on localized subband coefficient variance and SAD, selectively skipping flat regions to significantly optimize processing speed and file size.

2. **Multi-Resolution Design**
   - At decode time, you can extract specific spatial resolutions from a single file depending on your needs. This enables flexible, highly efficient video delivery suited to network bandwidth and device capabilities without storing multiple video files.

   **Extraction Patterns (assuming a 1080p source):**

   | Target Use Case           | Spatial (`-maxLayer`) | Result Output            | Server-Side Action             |
   | :------------------------ | :-------------------- | :----------------------- | :----------------------------- |
   | **Max Quality (Archive)** | `2` (Layer 0,1,2)     | 1080p                    | No extraction (transfer as is) |
   | **Medium (Preview)**      | `1` (Layer 0,1)       | 540p                     | Skip Layer 2                   |
   | **Ultra Low (Thumbnail)** | `0` (Layer 0 only)    | 270p                     | Skip Layer 1, 2               |

3. **Acceleration via Concurrency & SIMD**
   - Critical operations such as Plane matching, shifting, and difference calculations are fully parallelized and vectorized.

---

## Data Layout

`vevc` performs encoding using an I-Frame (Intra-coded) and P-Frame (Predicted) structure.

**Infrastructure Efficiency:** By leveraging the hierarchical spatial layers, servers can generate streams for different visual quality levels simply by skipping (demuxing) unnecessary outer layers without any re-encoding overhead.

```
                                     VEVC File Structure
+--------------------------------------------------------------------------------+
|                                  Container (VEVC)                              |
+--------------------------------------------------------------------------------+
|       I-Frame           |       P-Frame 1         |       P-Frame 2         | ...
+-------------------------+-------------------------+-------------------------+

                                     Frame Structure
+---------------------------------------------------------------------------------------------+
|     Magic (4B)     |   rANS Encoded MVs (P-Frame) |           Spatial Data            |
|  'VEVI' or 'VEVP'  |  RLE + rANS Motion Vectors   |                                   |
+--------------------+--------------------------------+-----------------------------------+
                                                |
                                                v
    +-----------------------------------------------------------------------------------------------+
    |                                        Spatial Layers                                         |
    +-----------------------------------+-----------------------------------+-----------------------+
    |              Layer 0              |              Layer 1              |        Layer 2        |
    +-----------------+-----------------+-----------------+-----------------+-------------+---------+
    | Header & Metdata|     Payload     | Header & Metdata|     Payload     | Header/Meta | Payload |
    |   'VEVC' + 0    |   (Y, Cb, Cr)   |   'VEVC' + 1    |   (Y, Cb, Cr)   | 'VEVC' + 2  | (Y,C,C) |
    +-----------------+-----------------+-----------------+-----------------+-------------+---------+
```

- **I-Frame (`VEVI`)**: The base keyframe encoded as a standalone 2D-DWT image.
- **P-Frame (`VEVP`)**: The predicted frame, containing rANS encoded Motion Vectors relative to the previous frame, followed by the encoded spatial layers of the **residual** (the difference after prediction).

Spatial information (image resolution) is organized hierarchically as Layer 0 to 2 (from `veif`) inside the frame data.

---

## Performance

*(Tested with 640x480, 60 frames, target 500 kbps)*

### Speed & Size

![speed_size](docs/speed_size.png)

SW: Software, HWA: Hardware Acceleration

### PSNR

![psnr](docs/psnr.png)

### SSIM

![ssim](docs/ssim.png)

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
- `-one`: Enables single-layer (Layer 0) mode, bypassing multi-resolution overhead for maximum encoding/decoding speed and optimal compression for fixed-resolution targets.
- `-keyint <keyint>`: Specifies the keyframe interval (GOP size).
- `-zeroThreshold <threshold>`: Sets the threshold for treating DWT coefficients as zero (reduces size).
- `-sceneThreshold <sad>`: Sets the SAD threshold for scene change detection (forces I-frame).

### Decode (`vevc-dec`)

Takes a `vevc` format file as input and outputs the decoded `y4m` video stream. Standard I/O (`-`) is also supported.

```bash
$ swift run -c release vevc-dec -i output.vevc -o output.y4m
```

**Multi-Resolution / Multi-Framerate Options**:

- `-i <path|->`: Specifies the input `.vevc` file path or standard input (`-`).
- `-o <path|->`: Specifies the output `.y4m` file path or standard output (`-`).
- `-one`: Decodes the stream assuming it was encoded with the single-layer (Layer 0) mode.
- `-maxLayer <0-2>`: Specifies the maximum level of spatial layers to decode.
  - `0`: 1/4 size (for rough thumbnails)
  - `1`: 1/2 size (for previews)
  - `2`: Original size (default)
- `-maxFrames <1|2|4>`: Decodes the specified sub-sampled framerate by skipping inter frames.

---

## Internals

The core components of the implementation consist of the following files:

- `Motion`: Macroblock-based Motion Estimation (MBME) utilizing 16x16 block searches to estimate accurate localized motion vectors between frames, significantly reducing prediction residual.
- `Encode` / `Plane`: The encoding flow that uses plane data (`PlaneData`) to process I-Frames and P-Frames. P-Frames generate a residual plane which is then passed to the Spatial 2D-DWT and entropy encoded via Interleaved rANS with Zero-Run RLE.
- `rANS` / `EntropyCodec`: Interleaved 4-way rANS entropy coding engine with adaptive token-based probability modeling, O(1) LUT decoding, and raw fallback for sparse data.

## License

MIT
