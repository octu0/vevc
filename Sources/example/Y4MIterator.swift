import Foundation
import vevc

class Y4MIterator {
    let fileHandle: FileHandle
    let reader: Y4MReader
    let isStdin: Bool
    var converter: FrameRateConverter?
    var pendingImg: ImageInput?
    var pendingCount = 0
    let maxFrames: Int?
    var frameCount = 0

    init(path: String, config: Config) throws {
        if path == "-" {
            self.fileHandle = FileHandle.standardInput
            self.isStdin = true
        } else {
            guard let f = FileHandle(forReadingAtPath: path) else {
                throw NSError(domain: "Y4MIterator", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to open \(path)"])
            }
            self.fileHandle = f
            self.isStdin = false
        }
        self.reader = try Y4MReader(fileHandle: self.fileHandle)
        self.maxFrames = config.maxFrames
        if let inFps = config.inFps, inFps != config.framerate {
            self.converter = FrameRateConverter(inFps: inFps, outFps: config.framerate)
        }
    }
    
    deinit {
        if !isStdin {
            fileHandle.closeFile()
        }
    }
    
    func next() throws -> ImageInput? {
        if let maxF = maxFrames, maxF <= frameCount { return nil }
        
        if pendingCount > 0 {
            pendingCount -= 1
            frameCount += 1
            return pendingImg
        }
        
        guard let img = try reader.readFrame() else { return nil }
        let input = ImageInput(vevcImage: img, width: reader.width, height: reader.height)
        
        if converter != nil {
            let count = converter!.repeatCount()
            if count > 0 {
                pendingImg = input
                pendingCount = count - 1
                frameCount += 1
                return input
            } else {
                return try next()
            }
        } else {
            frameCount += 1
            return input
        }
    }
}
