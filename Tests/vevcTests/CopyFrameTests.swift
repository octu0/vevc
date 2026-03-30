import Testing
@testable import vevc

// MARK: - Copy Frame Tests
//
// Intent: When consecutive frames are identical (SAD=0), the encoder should
// emit a "copy frame" signal (FrameLen=0) instead of encoding the full frame.
// This saves significant data for videos with duplicate frames, such as
// 24fps content upconverted to 60fps where ~60% of frames are duplicates.
//
// Design:
// - Encoder: Compare current frame pixels to previous frame. If identical, write FrameLen=0.
// - Decoder: When FrameLen=0, reuse the previous decoded frame (previous PlaneData420).
// - Bitstream: FrameLen=0 is a valid, unambiguous signal since real encoded frames always have data.

struct CopyFrameTests {

    // MARK: - Helpers

    /// Create a simple solid-color YCbCrImage for testing
    private func makeSolidImage(width: Int, height: Int, y: UInt8, cb: UInt8, cr: UInt8) -> YCbCrImage {
        var img = YCbCrImage(width: width, height: height)
        for i in 0..<img.yPlane.count { img.yPlane[i] = y }
        for i in 0..<img.cbPlane.count { img.cbPlane[i] = cb }
        for i in 0..<img.crPlane.count { img.crPlane[i] = cr }
        return img
    }

    // MARK: - Test: Identical frames produce smaller output than unique frames

    /// Intent: Encoding N identical frames should produce significantly smaller data
    /// than encoding N unique frames, because duplicate frames are encoded as copy
    /// signals (FrameLen=0 = 4 bytes per duplicate).
    @Test func identicalFramesProduceSmallerOutput() async throws {
        let width = 64
        let height = 64

        // Scenario 1: 5 identical frames
        let img = makeSolidImage(width: width, height: height, y: 128, cb: 128, cr: 128)
        let identicalImages = [img, img, img, img, img]

        let identicalEncoder = VEVCEncoder(
            width: width, height: height,
            maxbitrate: 5_000_000,
            framerate: 60,
            zeroThreshold: 0,
            keyint: 60,
            sceneChangeThreshold: 32
        )
        let identicalBytes = try await identicalEncoder.encodeToData(images: identicalImages)

        // Scenario 2: 5 different frames
        let img1 = makeSolidImage(width: width, height: height, y: 100, cb: 128, cr: 128)
        let img2 = makeSolidImage(width: width, height: height, y: 120, cb: 128, cr: 128)
        let img3 = makeSolidImage(width: width, height: height, y: 140, cb: 128, cr: 128)
        let img4 = makeSolidImage(width: width, height: height, y: 160, cb: 128, cr: 128)
        let img5 = makeSolidImage(width: width, height: height, y: 180, cb: 128, cr: 128)
        let differentImages = [img1, img2, img3, img4, img5]

        let differentEncoder = VEVCEncoder(
            width: width, height: height,
            maxbitrate: 5_000_000,
            framerate: 60,
            zeroThreshold: 0,
            keyint: 60,
            sceneChangeThreshold: 32
        )
        let differentBytes = try await differentEncoder.encodeToData(images: differentImages)

        let identicalSize = identicalBytes.count
        let differentSize = differentBytes.count
        print("  Identical 5 frames: \(identicalSize) bytes")
        print("  Different 5 frames: \(differentSize) bytes")
        print("  Ratio: \(String(format: "%.1f", Double(identicalSize) / Double(differentSize) * 100))%")

        // Identical frames should be at least 50% smaller
        // (1 I-frame + 4 copy frames vs 1 I-frame + 4 P-frames)
        #expect(identicalSize < differentSize, "Identical frames should produce less data")
        #expect(Double(identicalSize) < Double(differentSize) * 0.5,
               "Identical frames should be at least 50% smaller")
    }

    // MARK: - Test: Roundtrip with copy frames produces correct output

    /// Intent: Encoding then decoding a sequence with duplicate frames should
    /// produce output frames that match the input (within lossy codec tolerance).
    @Test func copyFrameRoundtrip() async throws {
        let width = 64
        let height = 64

        // Create a sequence: [unique, duplicate, unique, duplicate, duplicate]
        let img1 = makeSolidImage(width: width, height: height, y: 100, cb: 128, cr: 128)
        let img2 = makeSolidImage(width: width, height: height, y: 200, cb: 128, cr: 128)
        let images = [img1, img1, img2, img2, img2]

        let copyEncoder = VEVCEncoder(
            width: width, height: height,
            maxbitrate: 5_000_000,
            framerate: 60,
            zeroThreshold: 0,
            keyint: 60,
            sceneChangeThreshold: 32
        )
        let encoded = try await copyEncoder.encodeToData(images: images)

        let decoded = try await Decoder().decode(data: encoded)
        #expect(decoded.count == 5, "Should decode 5 frames, got \(decoded.count)")

        // Frames 0 and 1 should be similar (both derived from img1)
        let y0 = decoded[0].yPlane
        let y1 = decoded[1].yPlane
        let diff01 = zip(y0, y1).map { abs(Int($0) - Int($1)) }.reduce(0, +) / y0.count
        #expect(diff01 == 0, "Frames 0 and 1 should be identical (copy frame), avg diff = \(diff01)")

        // Frames 2, 3, 4 should be similar (all derived from img2)
        let y2 = decoded[2].yPlane
        let y3 = decoded[3].yPlane
        let y4 = decoded[4].yPlane
        let diff23 = zip(y2, y3).map { abs(Int($0) - Int($1)) }.reduce(0, +) / y2.count
        let diff34 = zip(y3, y4).map { abs(Int($0) - Int($1)) }.reduce(0, +) / y3.count
        #expect(diff23 == 0, "Frames 2 and 3 should be identical (copy frame), avg diff = \(diff23)")
        #expect(diff34 == 0, "Frames 3 and 4 should be identical (copy frame), avg diff = \(diff34)")
    }

    // MARK: - Test: GOP size count is preserved with copy frames

    /// Intent: The GOP should report the correct number of frames even when
    /// some are copy frames. The decoder must produce the right number of output frames.
    @Test func gopFrameCountPreservedWithCopyFrames() async throws {
        let width = 64
        let height = 64
        let img = makeSolidImage(width: width, height: height, y: 128, cb: 128, cr: 128)

        // Test with various numbers of identical frames
        for count in [2, 3, 5, 10] {
            let images = [YCbCrImage](repeating: img, count: count)

            let countEncoder = VEVCEncoder(
                width: width, height: height,
                maxbitrate: 5_000_000,
                framerate: 60,
                zeroThreshold: 0,
                keyint: 60,
                sceneChangeThreshold: 32
            )
            let encoded = try await countEncoder.encodeToData(images: images)

            let decoded = try await Decoder().decode(data: encoded)
            #expect(decoded.count == count,
                   "GOP with \(count) identical frames should decode to \(count) frames, got \(decoded.count)")
        }
    }
}
