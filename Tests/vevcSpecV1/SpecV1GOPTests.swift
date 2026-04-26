import XCTest
@testable import vevc

final class SpecV1GOPTests: XCTestCase {
    
    let y4mPath = "Tests/vevcSpecV1/testdata/spec_1080p_60f.y4m"
    var allFrames: [YCbCrImage] = []
    
    override func setUp() async throws {
        if allFrames.isEmpty {
            let fileHandle = try FileHandle(forReadingFrom: URL(fileURLWithPath: y4mPath))
            let reader = try Y4MReader(fileHandle: fileHandle)
            var frames = [YCbCrImage]()
            while let frame = try reader.readFrame() {
                frames.append(frame)
            }
            allFrames = frames
            // Append one more frame to make it 61 for testing
            if let last = frames.last {
                allFrames.append(last)
            }
        }
    }
    
    private func runE2E(frameCount: Int) async throws {
        guard frameCount <= allFrames.count else {
            XCTFail("Not enough frames")
            return
        }
        
        let frames = Array(allFrames.prefix(frameCount))
        let width = frames[0].width
        let height = frames[0].height
        
        // Encode
        let encoder = VEVCEncoder(
            width: width,
            height: height,
            maxbitrate: 1500 * 1000,
            zeroThreshold: 0,
            keyint: 4,
            sceneChangeThreshold: 10
        )
        let bitstream = try await encoder.encodeToData(images: frames)
        
        // Decode
        let decoder = Decoder(maxLayer: 2, maxConcurrency: 4)
        let decodedFrames = try await decoder.decode(data: [UInt8](bitstream))
        
        XCTAssertEqual(decodedFrames.count, frameCount, "Decoded frame count should match input frame count")
        
        // Basic check
        for frame in decodedFrames {
            XCTAssertEqual(frame.width, width)
            XCTAssertEqual(frame.height, height)
            XCTAssertEqual(frame.yPlane.count, width * height)
        }
    }
    
    func testGOPLessThan4() async throws {
        try await runE2E(frameCount: 3)
    }
    
    func testGOPEqual4() async throws {
        try await runE2E(frameCount: 4)
    }
    
    func testGOP59() async throws {
        try await runE2E(frameCount: 59)
    }
    
    func testGOP61() async throws {
        try await runE2E(frameCount: 61)
    }
}
