# vevc


> [!IMPORTANT]
> Work In Progress


**vevc** is a 3D-DWT based video compression format, extending the high-efficiency, multi-resolution image format [veif](https://github.com/octu0/veif) (velocity image format) with Temporal 1D-DWT.

![figure0](docs/fig0.jpg)

## Features

1. **3D-Discrete Wavelet Transform (3D-DWT)**
   - **Spatial**: LeGall 5/3 2D-DWT (Supports multiple resolutions via Layer 0, 1, 2) similar to `veif`.
   - **Temporal**: 1D LeGall 5/3 Transform. It leverages the temporal correlation (similarities between consecutive frames) to increase compression efficiency.

2. **Multi-Resolution & Multi-Framerate Design**
   - At decode time, you can extract specific spatial resolutions and framerates from a single file depending on your needs. This enables flexible, highly efficient video delivery suited to network bandwidth and device capabilities without storing multiple video files.

   **Extraction Patterns (assuming a 1080p / 60fps source):**

   | Target Use Case           | Spatial (`-maxLayer`) | Temporal (`-maxFrames`) | Result Output            | Server-Side Action             |
   | :------------------------ | :-------------------- | :---------------------- | :----------------------- | :----------------------------- |
   | **Max Quality (Archive)** | `2` (Layer 0,1,2)     | `4` (LL, LH, H0, H1)    | 1080p / 60fps            | No extraction (transfer as is) |
   | **High Quality (Perf)**   | `2` (Layer 0,1,2)     | `2` (LL, LH only)       | 1080p / 30fps            | Skip H0, H1                    |
   | **Medium (Preview)**      | `1` (Layer 0,1)       | `2` (LL, LH only)       | 540p / 30fps             | Skip Layer 2 and H0, H1        |
   | **Medium-Low Quality**    | `1` (Layer 0,1)       | `1` (LL only)           | 540p / 15fps             | Skip Layer 2, LH, H0, H1       |
   | **Low Quality**           | `0` (Layer 0 only)    | `2` (LL, LH only)       | 270p / 30fps             | Skip Layer 1, 2, H0, H1        |
   | **Ultra Low (Thumbnail)** | `0` (Layer 0 only)    | `1` (LL only)           | 270p / 15fps             | Extract minimum data only      |

3. **Acceleration via SIMD & Concurrency**
   - The temporal Lift53 Transform is vectorized for high performance using SIMD (SIMD8).
   - Spatial layer processing of frequencies and blocks is executed in parallel using Swift Concurrency (`async/await`, `TaskGroup`).

---

## Data Layout

`vevc` performs encoding in units of GOPs (Group of Pictures), which bundle multiple frames together. The current default implementation uses GOP=4 (4-frame units).

**Infrastructure Efficiency:** By leveraging this file structure, servers can generate streams for different quality levels simply by skipping (demuxing) unnecessary chunks without any re-encoding overhead.

```
                                     VEVC File Structure
+--------------------------------------------------------------------------------+
|                                  Container (VEVC)                              |
+--------------------------------------------------------------------------------+
|       GOP Chunk 1       |       GOP Chunk 2       |       GOP Chunk 3       | ...
+-------------------------+-------------------------+-------------------------+

                                     GOP Chunk Structure (GOP=4)
+-------------------------------------------------------------------------------------------------------------------+
| Magic (4B)    |           LL           |           LH           |           H0           |           H1           |
| 'V''E''L' + 4 |    (Average Base)      |    (High Freq)         |   (Diff Frame)         |   (Diff Frame)         |
+---------------+------------------------+------------------------+------------------------+------------------------+
             |                                                                                                   |
             v                                                                                                   |
    +-------------------------------------------------------------------------------------------------------+    |
    |                                   Spatial Layers (inside each Temporal Band)                          | <--+
    +-----------------------------------+-----------------------------------+-------------------------------+
    |              Layer 0              |              Layer 1              |             Layer 2           |
    +-----------------+-----------------+-----------------+-----------------+-----------------+-------------+
    | Header & Metdata|     Payload     | Header & Metdata|     Payload     | Header & Metdata|   Payload   |
    |   'VEVC' + 0    |   (Y, Cb, Cr)   |   'VEVC' + 1    |   (Y, Cb, Cr)   |   'VEVC' + 2    | (Y, Cb, Cr) |
    +-----------------+-----------------+-----------------+-----------------+-----------------+-------------+
```

- **LL (Low-Low)**: The temporal average component across the entire GOP (4 frames). Can be used as a base thumbnail for the static image.
- **LH (Low-High)**: The temporal difference component between the first 2 frames and the last 2 frames.
- **H0**, **H1**: The differences between frames (motion or noise information) for the first half (Frame 0-1) and the second half (Frame 2-3), respectively.

Spatial information (image resolution) is further organized hierarchically as Layer 0 to 2 (from `veif`) inside each data chunk: LL, LH, H0, and H1.

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
- `-maxFrames <1|2|4>`: Specifies the number of temporal frames to extract from a single GOP.
  - `1`: Decode LL component only (1 frame/GOP)
  - `2`: Use LL and LH components (2 frames/GOP)
  - `4`: Use all components to restore full video (4 frames/GOP, default)

---

## Internals

The core components of the implementation consist of the following files:

- `TemporalDWT`: 1D Lift53 Transform and inverse transform logic across the temporal axis, fully utilizing SIMD.
- `Encode` / `Plane`: The encoding flow that uses plane data (`PlaneData`) to split and zero-pad the input images into GOP sizes, performing a two-stage encode: Temporal DWT -> Spatial DWT.

## License

MIT
