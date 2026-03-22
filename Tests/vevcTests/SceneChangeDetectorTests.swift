import XCTest
@testable import vevc

final class SceneChangeDetectorTests: XCTestCase {
    
    func createSolidPlane(width: Int, height: Int, value: Int16) -> PlaneData420 {
        let data = [Int16](repeating: value, count: width * height)
        return PlaneData420(width: width, height: height, y: data, cb: [Int16](repeating: 128, count: width * height / 4), cr: [Int16](repeating: 128, count: width * height / 4))
    }

    func testDetectNoSceneChange() {
        // Two planes that are perfectly identical
        let plane1 = createSolidPlane(width: 256, height: 256, value: 50)
        let plane2 = createSolidPlane(width: 256, height: 256, value: 50)
        
        let detector = SceneChangeDetector(threshold: 32)
        let isSceneCut = detector.isSceneChanged(prev: plane1, curr: plane2)
        XCTAssertFalse(isSceneCut, "Identical frames should not trigger scene cut")
    }
    
    func testDetectNoSceneChangeMinorDiff() {
        // Two planes that have minor differences (e.g. slight movement or noise)
        let plane1 = createSolidPlane(width: 256, height: 256, value: 50)
        var plane2 = createSolidPlane(width: 256, height: 256, value: 50)
        // Add minor differences
        for i in 0..<plane1.y.count {
            if i % 10 == 0 {
                plane2.y[i] = 60
            }
        }
        
        let detector = SceneChangeDetector(threshold: 32)
        let isSceneCut = detector.isSceneChanged(prev: plane1, curr: plane2)
        XCTAssertFalse(isSceneCut, "Minor differences should not trigger scene cut")
    }
    
    func testDetectSceneChange() {
        // Two completely different planes (e.g. cut from black to white)
        let plane1 = createSolidPlane(width: 256, height: 256, value: 10)
        let plane2 = createSolidPlane(width: 256, height: 256, value: 200)
        
        let detector = SceneChangeDetector(threshold: 32)
        let isSceneCut = detector.isSceneChanged(prev: plane1, curr: plane2)
        XCTAssertTrue(isSceneCut, "Completely different frames should trigger scene cut")
    }
}
