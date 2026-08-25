import XCTest
@testable import vevc

final class SelectiveSmoothTests: XCTestCase {

    func testContinuousPlaneSmoothFlat() {
        let width = 64
        let height = 64
        let count = width * height
        let src = [Int16](repeating: 42, count: count)
        var dst = [Int16](repeating: 0, count: count)
        var temp = [Int16](repeating: 0, count: count)
        let activityMap = [BlockActivityClass](repeating: .flat, count: (width / 32) * (height / 32))

        withUnsafePointers(src, mut: &dst) { sPtr, dPtr in
            withUnsafePointers(mut: &temp) { tPtr in
                smoothResidualPlaneContinuous(src: sPtr, dst: dPtr, temp: tPtr, width: width, height: height, activityMap: activityMap, stride: width)
            }
        }

        for v in dst {
            XCTAssertEqual(v, 42)
        }
    }

    func testContinuousPlaneSmoothImpulse() {
        let width = 64
        let height = 64
        let count = width * height
        var src = [Int16](repeating: 0, count: count)
        var dst = [Int16](repeating: 0, count: count)
        var temp = [Int16](repeating: 0, count: count)
        let activityMap = [BlockActivityClass](repeating: .flat, count: (width / 32) * (height / 32))

        // 中心 (32, 32) に 64 のインパルス
        src[32 * width + 32] = 64

        withUnsafePointers(src, mut: &dst) { sPtr, dPtr in
            withUnsafePointers(mut: &temp) { tPtr in
                smoothResidualPlaneContinuous(src: sPtr, dst: dPtr, temp: tPtr, width: width, height: height, activityMap: activityMap, stride: width)
            }
        }

        // 3-tap separable [1, 2, 1]/4 により、中心は 64 * (2/4) * (2/4) = 16
        XCTAssertEqual(dst[32 * width + 32], 16)
        // 水平・垂直隣接は 64 * (2/4) * (1/4) = 8
        XCTAssertEqual(dst[32 * width + 33], 8)
        XCTAssertEqual(dst[33 * width + 32], 8)
        // 対角隣接は 64 * (1/4) * (1/4) = 4
        XCTAssertEqual(dst[33 * width + 33], 4)
    }

    func testContinuousPlaneSmoothTexturedProtection() {
        let width = 64
        let height = 64
        let count = width * height
        var src = [Int16](repeating: 0, count: count)
        var dst = [Int16](repeating: 0, count: count)
        var temp = [Int16](repeating: 0, count: count)
        
        // (0,0) ブロックは textured、(1,0) ブロックは flat
        var activityMap = [BlockActivityClass](repeating: .flat, count: (width / 32) * (height / 32))
        activityMap[0] = .textured

        // (16, 16) にインパルス (textured ブロック内)
        src[16 * width + 16] = 100
        // (48, 48) にインパルス (flat ブロック内)
        src[48 * width + 48] = 64

        withUnsafePointers(src, mut: &dst) { sPtr, dPtr in
            withUnsafePointers(mut: &temp) { tPtr in
                smoothResidualPlaneContinuous(src: sPtr, dst: dPtr, temp: tPtr, width: width, height: height, activityMap: activityMap, stride: width)
            }
        }

        // textured ブロック (0,0) 内は原画残差が復帰するので 100 のまま
        XCTAssertEqual(dst[16 * width + 16], 100)
        // flat ブロック (1,1) 内は平滑化されて 64/4 = 16
        XCTAssertEqual(dst[48 * width + 48], 16)
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

        // 明示的に smooth: 1 を指定
        let encExplicitDefault = VEVCEncoder(width: width, height: height, qstep: 16, framerate: 30, profile: 0x02, smooth: 1)
        let bytesExplicitDefault = try await encExplicitDefault.encodeToData(images: frames)

        XCTAssertEqual(bytesDefault, bytesExplicitDefault, "Default settings must produce bit-exact output to explicit smooth: 1")
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

        let enc = VEVCEncoder(width: width, height: height, qstep: 16, framerate: 30, profile: 0x02, smooth: 1)
        let bytes = try await enc.encodeToData(images: frames)
        XCTAssertFalse(bytes.isEmpty)

        let decoder = Decoder(maxLayer: 2)
        let decoded = try await decoder.decode(data: bytes)
        XCTAssertEqual(decoded.count, 10)
    }
}
