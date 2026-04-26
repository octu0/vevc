# VEVC Data Layout Specification (v1)

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
| Static rANS Models | 1530 Bytes | 10 pre-trained static rANS models (Run/Val models for normal and DPCM contexts). These models are strictly required to be present in the metadata payload as a fallback baseline. |

---

## 2. Frame Packet Structure

Immediately following the File Metadata, "Frame Packets" are stored sequentially until the end of the file. Each frame begins with a 1-byte `Frame Status (Flag)`.

| Frame Status (1 Byte) | Frame Type | Has Payload? |
|---|---|---|
| `0x00` | **P-Frame** (Predicted) | Yes |
| `0x01` | **Copy-Frame** (Skip) | **No** (Directly copies the previously reconstructed frame) |
| `0x02` | **I-Frame** (Keyframe) | Yes |

### 2.1. Frame Header (For I-Frames and P-Frames)
If the Status is `0x00` or `0x02`, the following size information is stored next. These dictate the bounds of the payloads that follow.

| Field Name | Size | Description |
|---|---|---|
| MVs Count | 4 Bytes (UInt32BE) | Number of Motion Vectors. Usually 0 for I-Frames. |
| MVs Size | 4 Bytes (UInt32BE) | Byte size of the following MV Data payload. |
| RefDir Size | 4 Bytes (UInt32BE) | Byte size of the Reference Direction flags payload. |
| Layer0 Size | 4 Bytes (UInt32BE) | Byte size of the Base Layer (8x8) payload. |
| Layer1 Size | 4 Bytes (UInt32BE) | Byte size of Enhancement Layer 1 (16x16) payload. |
| Layer2 Size | 4 Bytes (UInt32BE) | Byte size of Enhancement Layer 2 (32x32) payload. |

> [!TIP]
> **Scalable Bitstream (Droppable Layers)**
> The VEVC bitstream is designed for O(1) server-side resolution and bitrate scaling. Because each layer's payload is completely independent in the bitstream, external tools (like `vevc-splitter`) can instantly drop `Layer1` and `Layer2` to reduce bitrate and resolution without re-encoding. This is done by simply setting `Layer1 Size` and/or `Layer2 Size` to `0` in the header, and stripping those bytes from the Frame Payload.

### 2.2. Frame Payload
Data is stored continuously according to the sizes specified in the header.

1. **MV Data** (`MVs Size` bytes)
2. **RefDir Data** (`RefDir Size` bytes): A bitset of flags used for bidirectional prediction.
3. **Layer0 Data** (`Layer0 Size` bytes)
4. **Layer1 Data** (`Layer1 Size` bytes)
5. **Layer2 Data** (`Layer2 Size` bytes)

---

## 3. Layer Data Structure

The internal structure for `Layer0`, `Layer1`, and `Layer2` is identical. Each layer retains residual information (entropy-coded DWT subband data) for the Y, Cb, and Cr planes.

| Field Name | Size | Description |
|---|---|---|
| Quantization Step Y | 2 Bytes (UInt16BE) | Base quantization step for the Y plane. |
| Quantization Step CbCr | 2 Bytes (UInt16BE) | Base quantization step for the Cb/Cr planes. |
| Y Payload Size | 4 Bytes (UInt32BE) | Byte size of the Y Payload Data. |
| **Y Payload Data** | (Y Payload Size bytes) | See section 3.1 below. |
| Cb Payload Size | 4 Bytes (UInt32BE) | Byte size of the Cb Payload Data. |
| **Cb Payload Data** | (Cb Payload Size bytes) | |
| Cr Payload Size | 4 Bytes (UInt32BE) | Byte size of the Cr Payload Data. |
| **Cr Payload Data** | (Cr Payload Size bytes) | |

### 3.1. Plane Payload (Y / Cb / Cr)
The data for each plane consists of the entropy-encoded data of the four DWT subbands (LL, HL, LH, HH) concatenated sequentially.

1. `LL Size` (UInt32BE) + `LL Data`
2. `HL Size` (UInt32BE) + `HL Data`
3. `LH Size` (UInt32BE) + `LH Data`
4. `HH Size` (UInt32BE) + `HH Data`

---

## 4. Entropy Coded Data Structure

The interior of each subband data (`LL Data`, `HL Data`, etc.) is entropy-coded using a combination of Bypass (raw bitstreams) and rANS (Asymmetric Numeral Systems) models.

| Field Name | Size | Description |
|---|---|---|
| Metadata Bypass Size | 4 Bytes (UInt32BE) | Size of the Bypass data. |
| Metadata Bypass Data | (Variable) | The raw Bypass bitstream. |
| Coefficient Count | 4 Bytes (UInt32BE) | Total number of coefficients to be decoded. |

If `Coefficient Count` is 0, the data for this subband terminates here. If it is 1 or greater, it is followed by one of the formats below.

### 4.1. Raw Mode (Uncompressed / Low Pair Count)
This format is used when the number of coefficients is very small (<= 32 pairs).
- `Flags` (1 Byte): Value is exactly `0x80`.
- `Raw Data Size` (4 Bytes UInt32BE)
- `Raw Data` (Bypass Bitstream of `Raw Data Size` bytes)
  - Zero-runs and coefficient tokens are written directly bit-by-bit.

### 4.2. rANS Mode
This is the standard mode used for the vast majority of blocks.
- `Flags` (1 Byte):
  - Bit 6 (`0x40`): Indicates the use of Static Tables (decoder uses the models from the file header instead of reading dynamic tables).
  - Bit 5 (`0x20`): Indicates the use of DPCM Tables. When this flag is set, the block exclusively uses the DPCM-specific context models (`dpcmRunModel` and `dpcmValModel`).
  - Bit 0 (`0x01`): Indicates the presence of Trailing Zeros (zero-runs that extend to the end of the block).
- `Total Pair Entries` (4 Bytes UInt32BE): Total number of Run/Value pairs.
- `Chunk Sizes` (4 Bytes x 4 = 16 Bytes): Number of elements per chunk for 4-lane parallel decoding.
- **Dynamic Frequency Tables (Conditional)**:
  - Present **only if** Bit 6 of `Flags` is `0` (dynamic tables). Contains compressed frequency tables for Run and Value tokens for each context.
- `Lane Bypass Data` (4 Chunks):
  - Repeated 4 times for each lane: `Bypass Size` (UInt32BE) followed by the `Bypass Data`.
- `rANS Bitstream` (All remaining bytes):
  - An interleaved 4-way rANS bitstream. This bitstream must be decoded in reverse order from the end.

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
> **DPCM and Context Modeling**
> VEVC uses multiple probability models (Contexts) depending on the spatial characteristics of the block.
> - **Normal Mode**: Uses 4 separate Contexts (Context 0-3) based on the surrounding block statistics.
> - **DPCM Mode**: For highly directional textures, a single specialized DPCM context is used to model the prediction residuals. When the DPCM flag (`0x20`) is set, the decoder routes all pairs to the DPCM Run/Val rANS models.
