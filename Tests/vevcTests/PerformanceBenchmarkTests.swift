import XCTest
@testable import vevc

final class PerformanceBenchmarkTests: XCTestCase {
    func testEncodePerformance() {
        let _images = (0..<10).map { i -> YCbCrImage in
            var img = YCbCrImage(width: 512, height: 512)
            for y in 0..<512 {
                for x in 0..<512 {
                    img.yPlane[y * 512 + x] = UInt8((x + y) % 256)
                }
            }
            return img
        }

        measure {
            let exp = expectation(description: "encode")
            let images = _images
            Task {
                _ = try! await vevc.encodeOne(images: images, maxbitrate: 5000 * 1024, zeroThreshold: 0, keyint: 10, sceneChangeThreshold: 100)
                exp.fulfill()
            }
            wait(for: [exp], timeout: 10.0)
        }
    }
}
