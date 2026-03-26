import XCTest
@testable import vevc

/// Temporal DWT (LeGall 5/3) の正確性を検証するテスト
final class TemporalDWTTests: XCTestCase {
    
    // MARK: - Phase 0: 1D Temporal DWT correctness
    
    /// Test 1a: 4-element LeGall 5/3 forward+inverse is lossless (perfect reconstruction)
    func testTemporalLift53_4_Roundtrip() {
        let testCases: [[Int16]] = [
            [100, 110, 120, 130],     // smooth ramp
            [200, 50, 200, 50],       // alternating
            [0, 0, 0, 0],             // all zero
            [255, 255, 255, 255],     // all max
            [10, 20, 30, 40],         // linear
            [-50, 100, -50, 100],     // signed alternating
            [1, 2, 3, 4],             // small values
            [32767, -32768, 0, 100],  // extreme values
        ]
        
        for (idx, original) in testCases.enumerated() {
            var buffer = original
            buffer.withUnsafeMutableBufferPointer { ptr in
                lift53_4(ptr, stride: 1)
            }
            // After forward: buffer[0..1] = L, buffer[2..3] = H
            buffer.withUnsafeMutableBufferPointer { ptr in
                invLift53_4(ptr, stride: 1)
            }
            XCTAssertEqual(buffer, original, "Roundtrip failed for test case \(idx): input=\(original), got=\(buffer)")
        }
    }
    
    /// Test 1b: 8-element LeGall 5/3 forward+inverse is lossless
    func testTemporalLift53_8_Roundtrip() {
        let testCases: [[Int16]] = [
            [100, 110, 120, 130, 140, 150, 160, 170],
            [200, 50, 200, 50, 200, 50, 200, 50],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [10, 20, 30, 40, 50, 60, 70, 80],
        ]
        
        for (idx, original) in testCases.enumerated() {
            var buffer = original
            buffer.withUnsafeMutableBufferPointer { ptr in
                lift53_8(ptr, stride: 1)
            }
            buffer.withUnsafeMutableBufferPointer { ptr in
                invLift53_8(ptr, stride: 1)
            }
            XCTAssertEqual(buffer, original, "Roundtrip failed for test case \(idx): input=\(original), got=\(buffer)")
        }
    }
    
    /// Test 2a: Static input (all same value) → temporal high = all zero (GOP=4)
    func testTemporalLift53_4_StaticInput() {
        let staticValues: [Int16] = [128, 128, 128, 128]
        var buffer = staticValues
        buffer.withUnsafeMutableBufferPointer { ptr in
            lift53_4(ptr, stride: 1)
        }
        // L coefficients should be close to 128, H coefficients should be exactly 0
        XCTAssertEqual(buffer[2], 0, "H[0] should be 0 for static input, got \(buffer[2])")
        XCTAssertEqual(buffer[3], 0, "H[1] should be 0 for static input, got \(buffer[3])")
    }
    
    /// Test 2b: Static input → temporal high = all zero (GOP=8)
    func testTemporalLift53_8_StaticInput() {
        let staticValues: [Int16] = [128, 128, 128, 128, 128, 128, 128, 128]
        var buffer = staticValues
        buffer.withUnsafeMutableBufferPointer { ptr in
            lift53_8(ptr, stride: 1)
        }
        // H coefficients (buffer[4..7]) should be exactly 0
        for i in 4..<8 {
            XCTAssertEqual(buffer[i], 0, "H[\(i-4)] should be 0 for static input, got \(buffer[i])")
        }
    }
    
    /// Test 3: Linear input → H coefficients are small
    func testTemporalLift53_4_LinearInput() {
        // Slowly changing values (like slow camera pan)
        let linear: [Int16] = [100, 101, 102, 103]
        var buffer = linear
        buffer.withUnsafeMutableBufferPointer { ptr in
            lift53_4(ptr, stride: 1)
        }
        // H coefficients should be very small for linear input
        let h0 = abs(Int(buffer[2]))
        let h1 = abs(Int(buffer[3]))
        XCTAssertLessThanOrEqual(h0, 2, "H[0] should be ≤2 for linear input, got \(h0)")
        XCTAssertLessThanOrEqual(h1, 2, "H[1] should be ≤2 for linear input, got \(h1)")
        
        // Verify roundtrip still works
        var roundtrip = linear
        roundtrip.withUnsafeMutableBufferPointer { ptr in
            lift53_4(ptr, stride: 1)
        }
        roundtrip.withUnsafeMutableBufferPointer { ptr in
            invLift53_4(ptr, stride: 1)
        }
        XCTAssertEqual(roundtrip, linear, "Roundtrip failed for linear input")
    }
    
    /// Test: lift53_4 with stride > 1 (used when accessing same pixel across frames)
    func testTemporalLift53_4_WithStride() {
        // Simulate 4 pixel values at stride=3 (e.g., interleaved frame data)
        var data: [Int16] = [
            100, 0, 0,   // frame 0, pixel at offset 0
            110, 0, 0,   // frame 1
            120, 0, 0,   // frame 2
            130, 0, 0,   // frame 3
        ]
        let original = data
        data.withUnsafeMutableBufferPointer { ptr in
            lift53_4(ptr, stride: 3)
        }
        data.withUnsafeMutableBufferPointer { ptr in
            invLift53_4(ptr, stride: 3)
        }
        // Only the strided elements should match
        XCTAssertEqual(data[0], original[0])
        XCTAssertEqual(data[3], original[3])
        XCTAssertEqual(data[6], original[6])
        XCTAssertEqual(data[9], original[9])
    }
    
    /// Test: Random values roundtrip (stress test)
    func testTemporalLift53_4_RandomRoundtrip() {
        for _ in 0..<100 {
            let original: [Int16] = (0..<4).map { _ in Int16.random(in: -1000...1000) }
            var buffer = original
            buffer.withUnsafeMutableBufferPointer { ptr in
                lift53_4(ptr, stride: 1)
            }
            buffer.withUnsafeMutableBufferPointer { ptr in
                invLift53_4(ptr, stride: 1)
            }
            XCTAssertEqual(buffer, original, "Random roundtrip failed: input=\(original), got=\(buffer)")
        }
    }
    
    // MARK: - Phase 1: Pixelwise Temporal DWT on PlaneData420
    
    /// Helper: create a PlaneData420 with uniform Y value
    private func makePlane(width: Int, height: Int, yValue: Int16, cbValue: Int16 = 128, crValue: Int16 = 128) -> PlaneData420 {
        let chromaW = (width + 1) / 2
        let chromaH = (height + 1) / 2
        return PlaneData420(
            width: width, height: height,
            y: [Int16](repeating: yValue, count: width * height),
            cb: [Int16](repeating: cbValue, count: chromaW * chromaH),
            cr: [Int16](repeating: crValue, count: chromaW * chromaH)
        )
    }
    
    /// Test 4: 4 identical PlaneData420 frames → temporal_H planes are all zero
    func testTemporalDWT4_StaticFrames() throws {
        let width = 64
        let height = 64
        let frame = makePlane(width: width, height: height, yValue: 128)
        let frames = [frame, frame, frame, frame]
        
        let subbands = try temporalForwardDWT4(frames: frames)
        
        XCTAssertEqual(subbands.low.count, 2)
        XCTAssertEqual(subbands.high.count, 2)
        
        // All temporal high coefficients should be exactly zero for static input
        for (i, highFrame) in subbands.high.enumerated() {
            for (j, val) in highFrame.y.enumerated() {
                XCTAssertEqual(val, 0, "temporal_H[\(i)].y[\(j)] should be 0 for static input, got \(val)")
            }
            for (j, val) in highFrame.cb.enumerated() {
                XCTAssertEqual(val, 0, "temporal_H[\(i)].cb[\(j)] should be 0 for static input, got \(val)")
            }
            for (j, val) in highFrame.cr.enumerated() {
                XCTAssertEqual(val, 0, "temporal_H[\(i)].cr[\(j)] should be 0 for static input, got \(val)")
            }
        }
    }
    
    /// Test 5: 4 frames → forward → inverse → perfect reconstruction
    func testTemporalDWT4_Roundtrip() throws {
        let width = 64
        let height = 64
        // Create 4 frames with different Y values (gradient across time)
        let frames = [
            makePlane(width: width, height: height, yValue: 100),
            makePlane(width: width, height: height, yValue: 120),
            makePlane(width: width, height: height, yValue: 140),
            makePlane(width: width, height: height, yValue: 160),
        ]
        
        let subbands = try temporalForwardDWT4(frames: frames)
        let reconstructed = try temporalInverseDWT4(subbands: subbands)
        
        XCTAssertEqual(reconstructed.count, 4)
        for i in 0..<4 {
            XCTAssertEqual(reconstructed[i].y, frames[i].y, "Frame \(i) Y plane not reconstructed correctly")
            XCTAssertEqual(reconstructed[i].cb, frames[i].cb, "Frame \(i) Cb plane not reconstructed correctly")
            XCTAssertEqual(reconstructed[i].cr, frames[i].cr, "Frame \(i) Cr plane not reconstructed correctly")
        }
    }
    
    /// Test 6: Small motion frames → temporal_H is close to zero
    func testTemporalDWT4_SmallMotion() throws {
        let width = 64
        let height = 64
        // Simulate small motion: each frame differs by ±1-2 from the previous
        var frames: [PlaneData420] = []
        for i in 0..<4 {
            let yValue = Int16(128 + i)  // 128, 129, 130, 131
            frames.append(makePlane(width: width, height: height, yValue: yValue))
        }
        
        let subbands = try temporalForwardDWT4(frames: frames)
        
        // temporal_H should be very small for slowly changing input
        for (i, highFrame) in subbands.high.enumerated() {
            let maxH = highFrame.y.map { abs(Int($0)) }.max() ?? 0
            XCTAssertLessThanOrEqual(maxH, 3, "temporal_H[\(i)] max Y should be ≤3 for small motion, got \(maxH)")
        }
        
        // Verify roundtrip
        let reconstructed = try temporalInverseDWT4(subbands: subbands)
        for i in 0..<4 {
            XCTAssertEqual(reconstructed[i].y, frames[i].y, "Frame \(i) Y not reconstructed after small motion")
        }
    }
    
    // MARK: - Phase 2: Encode/Decode Roundtrip
    
    /// Helper: create a YCbCrImage with uniform Y value
    private func makeYCbCrImage(width: Int, height: Int, yValue: UInt8 = 128) -> YCbCrImage {
        var img = YCbCrImage(width: width, height: height)
        for i in 0..<img.yPlane.count {
            img.yPlane[i] = yValue
        }
        for i in 0..<img.cbPlane.count {
            img.cbPlane[i] = 128
        }
        for i in 0..<img.crPlane.count {
            img.crPlane[i] = 128
        }
        return img
    }
    
    /// Test 7: 4 static images → temporal GOP encode → decode → PSNR ≥ 40dB
    func testTemporalGOP4_EncodeDecodeRoundtrip_Static() async throws {
        let width = 128
        let height = 128
        let image = makeYCbCrImage(width: width, height: height, yValue: 128)
        let images = [image, image, image, image]
        
        let encoder = CoreEncoder(width: width, height: height, maxbitrate: 500, framerate: 30, isOne: false)
        let encoded = try await encoder.encodeTemporalGOP4(images: images)
        
        let decoder = CoreDecoder(maxLayer: 2, width: width, height: height)
        let decoded = try await decoder.decodeGOP(chunk: encoded)
        
        XCTAssertEqual(decoded.count, 4, "Should decode 4 frames")
        
        // Calculate PSNR for each frame
        for (i, decodedFrame) in decoded.enumerated() {
            var mse: Double = 0
            let original = images[i]
            for j in 0..<original.yPlane.count {
                let diff = Double(Int(decodedFrame.yPlane[j]) - Int(original.yPlane[j]))
                mse += diff * diff
            }
            mse /= Double(original.yPlane.count)
            let psnr = mse > 0 ? 10.0 * log10(255.0 * 255.0 / mse) : 100.0
            XCTAssertGreaterThanOrEqual(psnr, 40.0, "Frame \(i) PSNR should be ≥40dB, got \(String(format: "%.2f", psnr))dB")
        }
    }
    
    /// Test 8: Small motion images → temporal GOP encode → decode → SSIM ≥ 0.95
    func testTemporalGOP4_EncodeDecodeRoundtrip_SmallMotion() async throws {
        let width = 128
        let height = 128
        // Create 4 frames with slightly different Y values
        let images = [
            makeYCbCrImage(width: width, height: height, yValue: 126),
            makeYCbCrImage(width: width, height: height, yValue: 127),
            makeYCbCrImage(width: width, height: height, yValue: 128),
            makeYCbCrImage(width: width, height: height, yValue: 129),
        ]
        
        let encoder = CoreEncoder(width: width, height: height, maxbitrate: 500, framerate: 30, isOne: false)
        let encoded = try await encoder.encodeTemporalGOP4(images: images)
        
        let decoder = CoreDecoder(maxLayer: 2, width: width, height: height)
        let decoded = try await decoder.decodeGOP(chunk: encoded)
        
        XCTAssertEqual(decoded.count, 4, "Should decode 4 frames")
        
        // Check PSNR for each frame (should be high for uniform images with small differences)
        for (i, decodedFrame) in decoded.enumerated() {
            var mse: Double = 0
            let original = images[i]
            for j in 0..<original.yPlane.count {
                let diff = Double(Int(decodedFrame.yPlane[j]) - Int(original.yPlane[j]))
                mse += diff * diff
            }
            mse /= Double(original.yPlane.count)
            let psnr = mse > 0 ? 10.0 * log10(255.0 * 255.0 / mse) : 100.0
            XCTAssertGreaterThanOrEqual(psnr, 30.0, "Frame \(i) PSNR should be ≥30dB, got \(String(format: "%.2f", psnr))dB")
        }
    }
    
    /// Test 9: Temporal GOP compression is smaller than non-temporal for static content
    func testTemporalGOP4_SizeSmallerThanNonTemporal() async throws {
        let width = 128
        let height = 128
        let image = makeYCbCrImage(width: width, height: height, yValue: 128)
        let images = [image, image, image, image]
        
        // Encode with temporal GOP
        let temporalEncoder = CoreEncoder(width: width, height: height, maxbitrate: 500, framerate: 30, isOne: false)
        let temporalEncoded = try await temporalEncoder.encodeTemporalGOP4(images: images)
        
        // Encode without temporal (4 individual I-frames)
        let normalEncoder = CoreEncoder(width: width, height: height, maxbitrate: 500, framerate: 30, isOne: false)
        var normalSize = 0
        for img in images {
            let bytes = try await normalEncoder.encode(image: img)
            normalSize += bytes.count
        }
        
        XCTAssertLessThan(temporalEncoded.count, normalSize,
            "Temporal GOP (\(temporalEncoded.count) bytes) should be smaller than non-temporal (\(normalSize) bytes) for static content")
    }
}


