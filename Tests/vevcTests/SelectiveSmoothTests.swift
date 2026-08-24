import XCTest
@testable import vevc

final class SelectiveSmoothTests: XCTestCase {

    func testSmoothResidualBlock8Flat() {
        var block = [Int16](repeating: 42, count: 64)
        block.withUnsafeMutableBufferPointer { buf in
            guard let ptr = buf.baseAddress else { return }
            smoothResidualBlock8(base: ptr, stride: 8)
        }
        for v in block {
            XCTAssertEqual(v, 42)
        }
    }

    func testSmoothResidualBlock16Flat() {
        var block = [Int16](repeating: 100, count: 256)
        block.withUnsafeMutableBufferPointer { buf in
            guard let ptr = buf.baseAddress else { return }
            smoothResidualBlock16(base: ptr, stride: 16)
        }
        for v in block {
            XCTAssertEqual(v, 100)
        }
    }

    func testSmoothResidualBlock32Flat() {
        var block = [Int16](repeating: -50, count: 1024)
        block.withUnsafeMutableBufferPointer { buf in
            guard let ptr = buf.baseAddress else { return }
            smoothResidualBlock32(base: ptr, stride: 32)
        }
        for v in block {
            XCTAssertEqual(v, -50)
        }
    }

    func testSmoothResidualBlock8SmoothsNoise() {
        var block = [Int16](repeating: 0, count: 64)
        // 中心部にインパルスノイズ
        block[4 * 8 + 4] = 64
        block.withUnsafeMutableBufferPointer { buf in
            guard let ptr = buf.baseAddress else { return }
            smoothResidualBlock8(base: ptr, stride: 8)
        }
        // (4,4) の重みは 1/4 なので 64/4 = 16 に減衰するはず
        XCTAssertEqual(block[4 * 8 + 4], 16)
        // 隣接画素 (4,5), (5,4) などに 1/8 (8) が分配される
        XCTAssertEqual(block[4 * 8 + 5], 8)
        XCTAssertEqual(block[5 * 8 + 4], 8)
    }

    func testSmoothResidualBlock16SmoothsNoise() {
        var block = [Int16](repeating: 0, count: 256)
        block[8 * 16 + 8] = 64
        block.withUnsafeMutableBufferPointer { buf in
            guard let ptr = buf.baseAddress else { return }
            smoothResidualBlock16(base: ptr, stride: 16)
        }
        XCTAssertEqual(block[8 * 16 + 8], 16)
        XCTAssertEqual(block[8 * 16 + 9], 8)
        XCTAssertEqual(block[9 * 16 + 8], 8)
    }

    func testSmoothResidualBlock32SmoothsNoise() {
        var block = [Int16](repeating: 0, count: 1024)
        block[16 * 32 + 16] = 64
        block.withUnsafeMutableBufferPointer { buf in
            guard let ptr = buf.baseAddress else { return }
            smoothResidualBlock32(base: ptr, stride: 32)
        }
        XCTAssertEqual(block[16 * 32 + 16], 16)
        XCTAssertEqual(block[16 * 32 + 17], 8)
        XCTAssertEqual(block[17 * 32 + 16], 8)
    }

    func testDefaultSmoothIsBitExact() async throws {
        let width = 64
        let height = 64
        var frames = [YCbCrImage]()
        for f in 0..<10 {
            var img = YCbCrImage(width: width, height: height, ratio: .ratio420)
            for y in 0..<height {
                for x in 0..<width {
                    img.yPlane[y * width + x] = UInt8((x * 3 + y * 2 + f * 5) % 256)
                }
            }
            for cy in 0..<(height / 2) {
                for cx in 0..<(width / 2) {
                    img.cbPlane[cy * (width / 2) + cx] = 128
                    img.crPlane[cy * (width / 2) + cx] = 128
                }
            }
            frames.append(img)
        }

        // デフォルト初期化
        let encDefault = VEVCEncoder(width: width, height: height, qstep: 16, framerate: 30, profile: 0x02)
        let bytesDefault = try await encDefault.encodeToData(images: frames)

        // 明示的に smooth: 0 を指定
        let encExplicitZero = VEVCEncoder(width: width, height: height, qstep: 16, framerate: 30, profile: 0x02, smoothL2: 0, smoothL1: 0, smoothL0: 0)
        let bytesExplicitZero = try await encExplicitZero.encodeToData(images: frames)

        XCTAssertEqual(bytesDefault, bytesExplicitZero, "Default settings must produce bit-exact output to explicit zero flags")
    }

    func testSmoothFlagsRoundtrip() async throws {
        let width = 64
        let height = 64
        var frames = [YCbCrImage]()
        for f in 0..<10 {
            var img = YCbCrImage(width: width, height: height, ratio: .ratio420)
            for y in 0..<height {
                for x in 0..<width {
                    img.yPlane[y * width + x] = UInt8((x * 4 + y * 4 + f * 10) % 256)
                }
            }
            for cy in 0..<(height / 2) {
                for cx in 0..<(width / 2) {
                    img.cbPlane[cy * (width / 2) + cx] = 128
                    img.crPlane[cy * (width / 2) + cx] = 128
                }
            }
            frames.append(img)
        }

        let enc = VEVCEncoder(width: width, height: height, qstep: 16, framerate: 30, profile: 0x02, smoothL2: 1, smoothL1: 2, smoothL0: 1)
        let bytes = try await enc.encodeToData(images: frames)
        XCTAssertFalse(bytes.isEmpty)

        let decoder = Decoder(maxLayer: 2)
        let decoded = try await decoder.decode(data: bytes)
        XCTAssertEqual(decoded.count, 10)
    }
}
