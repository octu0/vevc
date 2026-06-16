# VEVC Data Layout Specification (v2)

This document provides a detailed explanation of the internal structure and format specifications of the VEVC (Video Encoding with Visual Clarity) encoded bitstream. 

This specification serves as a reference for implementing custom decoders, for instance in pure JavaScript without WebAssembly. All data is serialized in **Big Endian (BE)**.

---

## 1. File Level Structure

A VEVC bitstream begins with a file header (containing magic bytes and metadata), followed by multiple frame packets.

### 1.1. Magic Header & Metadata Size
The beginning of the file must always start with the following byte sequence.

| Field Name | Size | Description |
|---|---|---|
| Magic Header | 4 Bytes | `0x56`, `0x45`, `0x56`, `0x43` ("VEVC") |
| Metadata Size | 2 Bytes (UInt16BE) | The byte size of the File Metadata payload that immediately follows. |

### 1.2. File Metadata Payload
The metadata payload occupies exactly the number of bytes specified by `Metadata Size`.

| Field Name | Size | Description |
|---|---|---|
| Profile Version | 1 Byte | Currently always `0x01`. |
| Width | 2 Bytes (UInt16BE) | The pixel width of the video. |
| Height | 2 Bytes (UInt16BE) | The pixel height of the video. |
| Color Gamut | 1 Byte | Color gamut flag. Currently fixed to `0x01` (BT.709). |
| Framerate | 2 Bytes (UInt16BE) | Video framerate (e.g., 30, 60). |
| Timescale | 1 Byte | Currently always `0x00`. |
| Table Flag | 1 Byte | `0x00`: Use built-in static tables (no table data follows). `0x01`: Custom learned tables follow in compressed format (reserved for future use). When `0x00`, the decoder uses the hardcoded `StaticRANSModels` as the fallback baseline. |

---

## 2. Frame Packet Structure

Immediately following the File Metadata, "Frame Packets" are stored sequentially until the end of the file. Each frame begins with a 1-byte `Frame Status (Flag)`.

| Frame Status (1 Byte) | Frame Type | Has Payload? |
|---|---|---|
| `0x00` | **P-Frame** (Predicted, forward-only) | Yes |
| `0x01` | **Copy-Frame** (Skip) | **No** (Directly copies the previously reconstructed frame) |
| `0x02` | **I-Frame** (Keyframe) | Yes |
| `0x10` | **P-Frame** (Bidirectional, with RefDir) | Yes |

> [!NOTE]
> The Frame Status byte encodes both the frame type (lower 4 bits) and the `hasRefDir` flag (bit 4, `0x10`). When bit 4 is set, the frame includes a Reference Direction bitset in the payload. This flag is only meaningful for P-Frames (`0x00`), yielding `0x10` for bidirectional P-Frames.

### 2.1. Frame Header (For I-Frames and P-Frames)
If the Status is not `0x01` (Copy-Frame), the following size information is stored next. These dictate the bounds of the payloads that follow.

| Field Name | Size | Description |
|---|---|---|
| MVs Size | 4 Bytes (UInt32BE) | Byte size of the following MV Data payload. |
| Layer0 Size | 4 Bytes (UInt32BE) | Byte size of the Base Layer (8x8) payload. |
| Layer1 Size | 4 Bytes (UInt32BE) | Byte size of Enhancement Layer 1 (16x16) payload. |
| Layer2 Size | 4 Bytes (UInt32BE) | Byte size of Enhancement Layer 2 (32x32) payload. |

> [!IMPORTANT]
> **Derived Fields (not stored in header)**:
> - **MVs Count**: Derived from frame dimensions at the Base8 (L0) resolution after 2 DWT stages: `l1 = ceil(dim/2)`, `l0 = ceil(l1/2)`, then `ceil(l0_width / 8) × ceil(l0_height / 8)`. Width and height are available from the File Metadata.
> - **RefDir Size**: Derived from MVs Count: `ceil(mvsCount / 8)`. Only present when the `hasRefDir` flag (bit 4 of Frame Status) is set.

> [!TIP]
> **Scalable Bitstream (Droppable Layers)**
> The VEVC bitstream is designed for O(1) server-side resolution and bitrate scaling. Because each layer's payload is completely independent in the bitstream, external tools (like `vevc-splitter`) can instantly drop `Layer1` and `Layer2` to reduce bitrate and resolution without re-encoding. This is done by simply setting `Layer1 Size` and/or `Layer2 Size` to `0` in the header, and stripping those bytes from the Frame Payload.
>
> **Multi-Resolution Motion Compensation**: Motion Vectors are stored in Layer 0 (Base8) quarter-pixel precision. During decoding, each layer scales the MVs appropriately:
> - **Layer 0** (Base8): MVs used as-is (×1), applied with 8×8 Luma / 4×4 Chroma block size.
> - **Layer 1** (Level16): MVs scaled ×2, applied with 16×16 Luma / 8×8 Chroma block size.
> - **Layer 2** (Level32): MVs scaled ×4, applied with 32×32 Luma / 16×16 Chroma block size.
>
> This "Encode Once, Route Anywhere" design means each resolution tier independently performs motion compensation using the same shared MV data, drastically reducing both compute and data overhead.

### 2.2. Frame Payload
Data is stored continuously according to the sizes specified (or derived from) the header.

1. **MV Data** (`MVs Size` bytes)
2. **RefDir Data** (`ceil(mvsCount / 8)` bytes, only present when `hasRefDir` is set): A bitset of flags used for bidirectional prediction.
3. **Layer0 Data** (`Layer0 Size` bytes)
4. **Layer1 Data** (`Layer1 Size` bytes)
5. **Layer2 Data** (`Layer2 Size` bytes)

---

## 3. Motion Vector Precision

Motion Vectors in VEVC are encoded and stored at **Layer 0 (Base8) quarter-pixel precision**. This means that a single set of MVs is shared across all spatial layers, with each layer applying a scale factor during motion compensation:

| Layer | MV Scale | Luma Block Size | Chroma Block Size | Effective Precision |
|---|---|---|---|---|
| Layer 0 (Base8) | ×1 | 8×8 | 4×4 | Quarter-pixel of Base8 |
| Layer 1 (Level16) | ×2 | 16×16 | 8×8 | Quarter-pixel of Level16 |
| Layer 2 (Level32) | ×4 | 32×32 | 16×16 | Quarter-pixel of Level32 |

This design eliminates the need for separate MV computation at each resolution, enabling constant-time resolution scaling.

---

## 4. Layer Data Structure

The internal structure for `Layer0`, `Layer1`, and `Layer2` is identical. Each layer retains residual information (entropy-coded DWT subband data) for the Y, Cb, and Cr planes.

| Field Name | Size | Description |
|---|---|---|
| Quantization Step Y | 2 Bytes (UInt16BE) | Base quantization step for the Y plane. |
| Quantization Step CbCr | 2 Bytes (UInt16BE) | Base quantization step for the Cb/Cr planes. |
| Y Payload Size | VLQ | Byte size of the Y Payload Data. |
| **Y Payload Data** | (Y Payload Size bytes) | See section 3.1 below. |
| Cb Payload Size | VLQ | Byte size of the Cb Payload Data. |
| **Cb Payload Data** | (Cb Payload Size bytes) | |
| Cr Payload Size | VLQ | Byte size of the Cr Payload Data. |
| **Cr Payload Data** | (Cr Payload Size bytes) | |

### 3.1. Plane Payload (Y / Cb / Cr)
The data for each plane consists of a single **unified entropy stream** that encodes all four DWT subbands (LL, HL, LH, HH) together using 5 rANS contexts.

The payload consists of:
1. **Block Flags** (variable, bypass bitstream): Per-block zero/split decision flags.
2. **Unified Entropy Data**: A single `EntropyEncoder` output containing interleaved LL (DPCM, context 4) and HL/LH/HH (AC, contexts 0-3) coefficients.

> [!NOTE]
> In previous versions, each subband had its own `[Size (4B)][Data]` pair, totaling 16 bytes of size prefixes per plane. The unified stream eliminates this overhead entirely.

---

> [!NOTE]
> **Motion Compensation at Decode Time**
> For P-Frames, each layer's decoded residual data must have motion compensation applied *after* inverse DWT reconstruction. The decoder fetches pixels from the reference frame (scaled to the target layer's resolution) using the scaled MVs, and adds them to the decoded residual to produce the final reconstructed frame. This is performed independently at whatever resolution tier the decoder is targeting (`maxLayer`).

## 5. Entropy Coded Data Structure

The unified entropy stream is coded using a combination of Bypass (raw bitstreams) and rANS (Asymmetric Numeral Systems) models.

| Field Name | Size | Description |
|---|---|---|
| Metadata Bypass Size | VLQ | Size of the Bypass data. |
| Metadata Bypass Data | (Variable) | The raw Bypass bitstream. |
| Coefficient Count | VLQ | Total number of coefficients to be decoded. |

If `Coefficient Count` is 0, the data terminates here. If it is 1 or greater, it is followed by one of the formats below.

### 4.1. Raw Mode (Uncompressed / Low Pair Count)
This format is used when the number of coefficients is very small (<= 32 pairs).
- `Flags` (1 Byte): Value is exactly `0x80`.
- `Raw Data Size` (VLQ)
- `Raw Data` (Bypass Bitstream of `Raw Data Size` bytes)
  - Zero-runs and coefficient tokens are written directly bit-by-bit.

### 4.2. rANS Mode
This is the standard mode used for the vast majority of blocks.
- `Flags` (1 Byte):
  - Bit 6 (`0x40`): Indicates the use of Static Tables (decoder uses the built-in models instead of reading dynamic tables).
  - Bit 5 (`0x20`): Reserved (must be 0).
  - Bit 4 (`0x10`): Indicates Merged Context mode. When set, a single rANS model (1 Run + 1 Val table) is used for all 5 contexts instead of separate context-specific models. Only 2 frequency tables follow in the stream instead of 10.
  - Bit 0 (`0x01`): Indicates the presence of Trailing Zeros (zero-runs that extend to the end of the block).
- `Total Pair Entries` (VLQ): Total number of Run/Value pairs.
  - *Note: The 4-lane elements boundaries (chunk starts) are dynamically reconstructed from this value by the decoder using integer division and modulo (`Total Pair Entries / 4`), eliminating fixed header overhead.*
- **Dynamic Frequency Tables (Conditional)**:
  - Present **only if** Bit 6 of `Flags` is `0` (dynamic tables).
  - If Bit 4 (`0x10`, Merged Context) is set: 2 compressed frequency tables (1 Run + 1 Val) follow.
  - If Bit 4 is `0` (5-context mode): 10 compressed frequency tables (Run + Val for each of the 5 contexts) follow.
- `Lane Bypass Data` (4 Chunks):
  - Repeated 4 times for each lane: `Bypass Size` (VLQ) followed by the `Bypass Data`.
- `rANS Bitstream` (All remaining bytes):
  - An interleaved 4-way rANS bitstream. This bitstream must be decoded in reverse order from the end.

> [!NOTE]
> **Cost-Based Model Selection**
> The encoder uses a cost-based model selection algorithm to choose the optimal model for each plane's unified stream. The encoder estimates the total bit cost (data bits + header overhead) for three options:
> 1. **Static 5-context**: Uses the built-in models (no header overhead). Contexts 0-3 use AC models, context 4 uses the DPCM model.
> 2. **Dynamic 5-context**: Builds models from the actual data, writes 10 frequency tables to the stream.
> 3. **Dynamic merged**: Merges all context statistics into a single model, writes only 2 frequency tables.
>
> The option with the lowest estimated total cost is selected. This ensures that dynamic tables are only used when the compression benefit outweighs the header overhead cost.

> [!NOTE]
> **Tokenization and Exponential-Golomb Encoding**
> In VEVC, residual coefficients are not directly compressed by rANS. Instead, they are separated into a **Token** (which is compressed using rANS) and **Bypass Bits** (which are stored uncompressed in the `Bypass Data`).
> 
> VEVC employs a custom Exponential-Golomb style encoding for values:
> - **Small Values (Magnitude <= 15)**: The exact value is mapped to a unique Token (0 to 31). No bypass bits are required.
> - **Large Values (Magnitude >= 16)**: The value is mapped to a Token (32 to 63) indicating the bit-length of the value. The remaining bits (the magnitude remainder and the sign bit) are appended to the `Bypass Data`.
>
> This hybrid approach ensures that the highly-probable small values are densely packed into the rANS models, while large outliers do not inflate the frequency tables.

> [!NOTE]
> **Context Modeling (5 Contexts)**
> VEVC uses 5 probability model contexts within each unified stream:
> - **Context 0-3 (AC)**: Used for HL, LH, HH subband coefficients. The context index is selected based on surrounding block statistics (parent zero, previous value magnitude).
> - **Context 4 (DPCM)**: Used exclusively for LL subband coefficients, which are DPCM-encoded (differential pulse-code modulation). This context captures the prediction residual distribution.
>
> The encoder may also select a merged single-context mode when the context-specific distributions are similar enough that the header savings outweigh the compression loss.

