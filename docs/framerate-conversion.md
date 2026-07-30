# Frame Rate (Cadence) Conversion

VEVC supports pre-encoding frame rate conversion (also known as cadence conversion) via dropping or duplicating frames.

## Mechanism

VEVC is fundamentally a video codec, not a media container. It does not carry presentation timestamps (PTS) or decoding timestamps (DTS) in its bitstream. Instead, it relies on a constant frame rate (CFR) model where the `framerate` header specifies the display rate.

Frame rate conversion is implemented strictly as a pre-processing step in the encoder wrapper (`vevc-enc` and `compare`). 
The core encoder and decoder (`Sources/vevc/`) are completely unaware of this conversion.

When `inFps` != `outFps`:
- The converter accumulates `outFps` for every input frame.
- If the accumulator >= `inFps`, it outputs the frame (possibly multiple times).
- If the accumulator < `inFps`, the input frame is dropped.
- Duplicated frames are sent to the encoder as the exact same `YCbCrImage` instance. 
- The encoder naturally encodes duplicated frames using the `CopyFrame` mechanism, costing only 6 bytes per duplicate without modifying the bitstream format.

## Usage

You can specify the input and output framerates in the CLI tools:

```bash
# vevc-enc
swift run -c release vevc-enc -i input.y4m -o output.vevc -in-fps 24 -framerate 30

# compare
swift run -c release compare -y4m input.y4m -in-fps 60 -framerate 30
```

- `-in-fps <fps>`: The framerate of the input sequence. If omitted, the FPS from the Y4M header is used.
- `-framerate <fps>`: The target output framerate. If omitted, the input framerate is used (no conversion).

## Limitations

- **CFR Only**: Only Constant Frame Rate inputs are supported. Variable Frame Rate (VFR) inputs will not be converted smoothly and may drift.
- **Integer Math**: The conversion uses purely integer arithmetic (e.g., `59.94` is typically rounded down to `59` when parsed from standard Y4M headers, unless explicitly overridden).
- **No Blending**: Frames are strictly dropped or duplicated (nearest-neighbor style cadence). There is no motion interpolation or frame blending.
