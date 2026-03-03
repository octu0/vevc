# vevc


> [!IMPORTANT]
> Work In Progress


**vevc** is a high-speed video compression format, extending the high-efficiency, multi-resolution image format [veif](https://github.com/octu0/veif) (velocity image format) with Pixel-Domain Global Motion Compensation (GMC) and Residual 2D-DWT.

![figure0](docs/fig0.jpg)

## Features

1. **Pixel-Domain Prediction & 2D-DWT**
   - **Spatial**: LeGall 5/3 2D-DWT (Supports multiple resolutions via Layer 0, 1, 2) similar to `veif`.
   - **Temporal**: Global Motion Compensation (GMC) to predict pixel movement, followed by 2D-DWT applied only to the residual (difference) frame, achieving ultra-fast decode speeds.

2. **Multi-Resolution Design**
   - At decode time, you can extract specific spatial resolutions from a single file depending on your needs. This enables flexible, highly efficient video delivery suited to network bandwidth and device capabilities without storing multiple video files.

   **Extraction Patterns (assuming a 1080p source):**

   | Target Use Case           | Spatial (`-maxLayer`) | Result Output            | Server-Side Action             |
   | :------------------------ | :-------------------- | :----------------------- | :----------------------------- |
   | **Max Quality (Archive)** | `2` (Layer 0,1,2)     | 1080p                    | No extraction (transfer as is) |
   | **Medium (Preview)**      | `1` (Layer 0,1)       | 540p                     | Skip Layer 2                   |
   | **Ultra Low (Thumbnail)** | `0` (Layer 0 only)    | 270p                     | Skip Layer 1, 2                |

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
+-----------------------------------------------------------------------------------+
|     Magic (4B)     |  Motion Vector (P-Frame) |           Spatial Data            |
|  'VEVI' or 'VEVP'  |      dx (2B), dy (2B)    |                                   |
+--------------------+--------------------------+-----------------------------------+
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
- **P-Frame (`VEVP`)**: The predicted frame, containing the Global Motion Vector relative to the previous frame, followed by the encoded spatial layers of the **residual** (the difference after prediction).

Spatial information (image resolution) is organized hierarchically as Layer 0 to 2 (from `veif`) inside the frame data.

---

## CLI Usage

The `vevc` package includes command-line tools: `vevc-enc` (encoder) and `vevc-dec` (decoder).

### Encode (`vevc-enc`)

Specify multiple PNG images (e.g., a sequence of files) to encode them into a single `vevc` binary file.

```bash
$ swift run -c release vevc-enc -o out.vevc docs/sample_frames/frame_0*.png
```

- `-bitrate`: Specifies the target bitrate (desired compression ratio/quality).
- `-o`: Specifies the output `.vevc` file path.

### Decode (`vevc-dec`)

Takes a `vevc` format file as input and outputs the decoded PNG images into a specified directory.

```bash
$ swift run -c release vevc-dec -i output.vevc -o .out/
```

**Multi-Resolution / Multi-Framerate Options**:

- `-maxLayer <0-2>`: Specifies the maximum level of spatial layers to decode.
  - `0`: 1/4 size (for rough thumbnails)
  - `1`: 1/2 size (for previews)
  - `2`: Original size (default)

---

## Internals

The core components of the implementation consist of the following files:

- `GMC`: Fast Global Motion Compensation using downscaled (Layer 0 equivalent) reference frames to estimate a single dominant motion vector between frames.
- `Encode` / `Plane`: The encoding flow that uses plane data (`PlaneData`) to process I-Frames and P-Frames. P-Frames generate a residual plane which is then passed to the Spatial 2D-DWT encoder.

## License

MIT
