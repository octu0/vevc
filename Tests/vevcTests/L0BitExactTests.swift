import XCTest
@testable import vevc

/// One-Pyramid Wave 1 gate: the quarter-resolution L0 reconstruction chain
/// must be bit-exact across (a) the encoder's closed loop, (b) the full
/// decoder's maintained chain, and (c) the standalone maxLayer==0 decoder.
final class L0BitExactTests: XCTestCase {
    private func makeFrame(index: Int, width: Int, height: Int) -> YCbCrImage {
        var y = [UInt8](repeating: 0, count: width * height)
        let cw = (width + 1) / 2
        let ch = (height + 1) / 2
        var cb = [UInt8](repeating: 128, count: cw * ch)
        var cr = [UInt8](repeating: 128, count: cw * ch)
        for yy in 0..<height {
            for xx in 0..<width {
                if yy < height / 2 {
                    // moving diagonal gradient (exercises inter blocks + MVs)
                    y[yy * width + xx] = UInt8(truncatingIfNeeded: (xx &+ yy &+ index &* 7) &* 3)
                } else {
                    // static textured half (exercises skip_prev / skip_ltr)
                    y[yy * width + xx] = UInt8(truncatingIfNeeded: (xx ^ yy) &* 5)
                }
            }
        }
        for yy in 0..<ch {
            for xx in 0..<cw {
                let moving = yy < ch / 2
                cb[yy * cw + xx] = UInt8(truncatingIfNeeded: 128 &+ ((xx &+ (moving ? index &* 3 : 0)) % 32))
                cr[yy * cw + xx] = UInt8(truncatingIfNeeded: 128 &- ((yy &+ (moving ? index &* 2 : 0)) % 32))
            }
        }
        var img = YCbCrImage(width: width, height: height, ratio: .ratio420, fps: nil)
        img.yPlane = y
        img.cbPlane = cb
        img.crPlane = cr
        return img
    }

    private func assertPlanesEqual(_ a: PlaneData420?, _ b: PlaneData420?, _ label: String) {
        guard let a, let b else {
            XCTFail("\(label): missing L0 reference (enc: \(a != nil), dec: \(b != nil))")
            return
        }
        XCTAssertEqual(a.width, b.width, "\(label) width")
        XCTAssertEqual(a.height, b.height, "\(label) height")
        XCTAssertEqual(a.y, b.y, "\(label) Y plane")
        XCTAssertEqual(a.cb, b.cb, "\(label) Cb plane")
        XCTAssertEqual(a.cr, b.cr, "\(label) Cr plane")
    }

    func testL0ChainBitExactAcrossEncoderAndDecoders() async throws {
        try await runL0ChainCheck(width: 128, height: 96, frames: 13, qstep: 800)
    }

    /// Dimensions whose quarter-resolution planes have partial 8x8 blocks
    /// (200x120 → L0 50x30) plus rate-controlled per-frame qsteps and a
    /// mid-stream scene change — the failure surface the fixed-qstep,
    /// block-aligned variant does not reach.
    func testL0ChainBitExactEdgeBlocksAndRateControl() async throws {
        try await runL0ChainCheck(width: 200, height: 120, frames: 13, qstep: nil, sceneChangeAt: 8)
    }

    private func runL0ChainCheck(width: Int, height: Int, frames: Int, qstep: Int?, sceneChangeAt: Int? = nil) async throws {
        let pool = BlockViewPool()
        let enc = LayersEncodeActor(
            width: width, height: height, maxbitrate: 300, framerate: 30,
            zeroThreshold: 0, keyint: 6, sceneChangeThreshold: 1_000_000_000,
            pool: pool, qstep: qstep, profile: 0x02
        )
        let decFull = StreamingDecoderActor(maxLayer: 2, width: width, height: height, profile: 0x02)
        let decL0 = StreamingDecoderActor(maxLayer: 0, width: width, height: height, profile: 0x02)

        for i in 0..<frames {
            let offset = (sceneChangeAt != nil && sceneChangeAt! <= i) ? 500 : 0
            let input = makeFrame(index: i + offset, width: width, height: height)
            let chunk = try await enc.encodeFrame(image: input, forceKeyFrame: i == sceneChangeAt)
            let fullImg = try await decFull.decodeNextFrame(chunk: chunk)
            let l0Img = try await decL0.decodeNextFrame(chunk: chunk)
            XCTAssertNotNil(fullImg, "frame \(i) full decode")
            XCTAssertNotNil(l0Img, "frame \(i) L0-only decode")

            // (a) == (b): encoder closed loop vs full decoder chain, bit-exact.
            let encPrev = await enc.l0State.prev
            let decPrev = await decFull.l0State.prev
            assertPlanesEqual(encPrev, decPrev, "frame \(i) L0 prev")
            let encLtr = await enc.l0State.ltr
            let decLtr = await decFull.l0State.ltr
            assertPlanesEqual(encLtr, decLtr, "frame \(i) L0 ltr")

            // (b) == (c): the standalone L0-only decoder output must be the
            // rendering of the very same chain state.
            if let ref = decPrev, let l0 = l0Img {
                let rendered = ref.toYCbCr()
                XCTAssertEqual(rendered.yPlane, l0.yPlane, "frame \(i) L0-only Y")
                XCTAssertEqual(rendered.cbPlane, l0.cbPlane, "frame \(i) L0-only Cb")
                XCTAssertEqual(rendered.crPlane, l0.crPlane, "frame \(i) L0-only Cr")
            }

            // Full-decode sanity: the LL2 slot substitution must leave the
            // full reconstruction close to the input (catches any analysis
            // chain mismatch, which would collapse quality catastrophically).
            if let full = fullImg {
                var sad = 0
                for j in 0..<full.yPlane.count {
                    sad += abs(Int(full.yPlane[j]) - Int(input.yPlane[j]))
                }
                // Catastrophe detector, not a quality gate: an analysis-chain
                // (T) mismatch collapses the reconstruction to MAD 50+, while
                // legitimate coarse quantization of this synthetic content
                // stays well under 20 even at starved bitrates.
                let mad = Double(sad) / Double(full.yPlane.count)
                XCTAssertLessThan(mad, 20.0, "frame \(i) full-decode luma MAD")
            }
        }
    }
}
