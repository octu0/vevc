import Foundation
import Combine
import CoreGraphics
import vevc

@MainActor
class PlayerViewModel: ObservableObject {
    @Published var isPlaying: Bool = false
    @Published var currentFrameIndex: Double = 0.0
    @Published var totalFrames: Double = 0.0
    @Published var isLoading: Bool = false
    @Published var statusMessage: String = ""
    
    @Published var bitrate: Int = 1000 {
        didSet {
            if bitrate < 100 {
                bitrate = 100
            }
            if 8000 < bitrate {
                bitrate = 8000
            }
        }
    }
    
    @Published var currentLayer0Image: CGImage?
    @Published var currentLayer1Image: CGImage?
    @Published var currentLayer2Image: CGImage?
    
    @Published var videoWidth: CGFloat = 800
    @Published var videoHeight: CGFloat = 400
    var videoFps: Double = 30.0
    private var streamingTask: Task<Void, Never>?
    
    func play() {
        isPlaying = true
    }
    
    func pause() {
        isPlaying = false
    }
    
    func loadFile(url: URL) {
        isLoading = true
        statusMessage = "Loading..."
        let currentBitrate = bitrate
        
        streamingTask?.cancel()
        
        streamingTask = Task.detached(priority: .userInitiated) { [weak self] in
            guard let self = self else { return }
            do {
                if url.pathExtension.lowercased() == "y4m" {
                    await MainActor.run { self.statusMessage = "Starting Y4M Streaming..." }
                    try await self.streamY4M(url: url, bitrate: currentBitrate)
                } else {
                    await MainActor.run { self.statusMessage = "Starting VEVC Streaming..." }
                    try await self.streamVEVC(url: url)
                }
            } catch {
                if Task.isCancelled != true {
                    await MainActor.run {
                        self.statusMessage = "Error: \(error.localizedDescription)"
                        self.isLoading = false
                        self.isPlaying = false
                    }
                }
            }
        }
    }
    
    private func streamY4M(url: URL, bitrate: Int) async throws {
        guard let fileHandle = try? FileHandle(forReadingFrom: url) else {
            throw NSError(domain: "Player", code: 1, userInfo: [NSLocalizedDescriptionKey: "Cannot open file"])
        }
        defer { try? fileHandle.close() }
        
        let y4mReader = try Y4MReader(fileHandle: fileHandle)
        var fps: Double = 30.0
        if y4mReader.fpsHeader.starts(with: "F") {
            let parts = y4mReader.fpsHeader.dropFirst().split(separator: ":")
            if parts.count == 2, let num = Double(parts[0]), let den = Double(parts[1]), 0.0 < den {
                fps = num / den
            } else if parts.count == 1, let num = Double(parts[0]) {
                fps = num
            }
            if fps <= 0.0 { fps = 30.0 }
        }
        
        let encoder = VEVCEncoder(
            width: y4mReader.width,
            height: y4mReader.height,
            maxbitrate: bitrate * 1000,
            framerate: Int(round(fps)),
            zeroThreshold: 3,
            keyint: 30,
            sceneChangeThreshold: 32
        )
        
        let decoder0 = StreamingDecoderActor(maxLayer: 0, width: y4mReader.width, height: y4mReader.height)
        let decoder1 = StreamingDecoderActor(maxLayer: 1, width: y4mReader.width, height: y4mReader.height)
        let decoder2 = StreamingDecoderActor(maxLayer: 2, width: y4mReader.width, height: y4mReader.height)
        
        let frameInterval = 1.0 / fps
        var frameIndex: Double = 0.0
        
        await MainActor.run {
            self.videoFps = fps
            self.videoWidth = CGFloat(y4mReader.width)
            self.videoHeight = CGFloat(y4mReader.height)
            self.currentFrameIndex = 0.0
            self.totalFrames = 0.0
            self.isLoading = false
            self.isPlaying = true
            self.statusMessage = String(format: "Playing (%.2f fps)", fps)
        }
        
        let bufferCapacity = (y4mReader.width >= 1920) ? 15 : 30
        let frameBuffer = FrameBuffer(maxCapacity: bufferCapacity)
        
        let producerTask = Task {
            do {
                while let image = try y4mReader.readFrame() {
                    try Task.checkCancellation()
                    let chunk = try await encoder.encode(image: image)
                    
                    let l0Chunk = try splitFrameChunk(chunk, maxLayer: 0)
                    let l1Chunk = try splitFrameChunk(chunk, maxLayer: 1)
                    let l2Chunk = try splitFrameChunk(chunk, maxLayer: 2)
                    
                    async let dec0 = decoder0.decodeNextFrame(chunk: l0Chunk)
                    async let dec1 = decoder1.decodeNextFrame(chunk: l1Chunk)
                    async let dec2 = decoder2.decodeNextFrame(chunk: l2Chunk)
                    
                    let (img0, img1, img2) = try await (dec0, dec1, dec2)
                    
                    let cgImg0 = try img0.flatMap { try createCGImage(from: $0) }
                    let cgImg1 = try img1.flatMap { try createCGImage(from: $0) }
                    let cgImg2 = try img2.flatMap { try createCGImage(from: $0) }
                    
                    try await frameBuffer.enqueue(FrameData(l0: cgImg0, l1: cgImg1, l2: cgImg2))
                }
            } catch {
                print("Producer error: \(error)")
            }
            await frameBuffer.finish()
        }
        
        var nextFrameTime = Date()
        var isBuffering = false
        
        while true {
            try Task.checkCancellation()
            
            while await MainActor.run(resultType: Bool.self, body: { self.isPlaying }) != true {
                try await Task.sleep(nanoseconds: 100_000_000)
                try Task.checkCancellation()
                nextFrameTime = Date() // Reset nextFrameTime when unpaused
            }
            
            let count = await frameBuffer.count
            if count == 0 {
                let finished = await frameBuffer.finished
                if finished { break }
                
                isBuffering = true
                await MainActor.run { self.statusMessage = "Buffering..." }
                try await Task.sleep(nanoseconds: 100_000_000)
                nextFrameTime = Date()
                continue
            }
            
            if isBuffering && count < bufferCapacity / 2 {
                let finished = await frameBuffer.finished
                if finished != true {
                    try await Task.sleep(nanoseconds: 100_000_000)
                    nextFrameTime = Date()
                    continue
                }
            }
            
            isBuffering = false
            await MainActor.run { self.statusMessage = String(format: "Playing (%.2f fps)", fps) }
            
            guard let frame = try await frameBuffer.dequeue() else {
                break
            }
            
            await MainActor.run {
                self.currentLayer0Image = frame.l0
                self.currentLayer1Image = frame.l1
                self.currentLayer2Image = frame.l2
                self.currentFrameIndex = frameIndex
                self.totalFrames = frameIndex + 1.0
            }
            frameIndex += 1.0
            
            let now = Date()
            if nextFrameTime < now {
                nextFrameTime = now
            }
            let delay = nextFrameTime.timeIntervalSince(now)
            if 0.0 < delay {
                try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            }
            nextFrameTime = nextFrameTime.addingTimeInterval(frameInterval)
        }
        
        producerTask.cancel()
        
        await MainActor.run {
            self.isPlaying = false
        }
    }
    
    private func readUInt16BE(_ bytes: [UInt8], offset: inout Int) throws -> UInt16 {
        guard offset + 1 < bytes.count else { throw NSError(domain: "Player", code: 2, userInfo: nil) }
        let val = (UInt16(bytes[offset]) << 8) | UInt16(bytes[offset+1])
        offset += 2
        return val
    }
    
    private func splitFrameChunk(_ chunk: [UInt8], maxLayer: Int) throws -> [UInt8] {
        var offset = 0
        if offset + 4 <= chunk.count && chunk[offset] == 0x56 && chunk[offset+1] == 0x45 && chunk[offset+2] == 0x56 && chunk[offset+3] == 0x43 {
            offset += 4
            let metadataSize = Int(try readUInt16BE(chunk, offset: &offset))
            offset += metadataSize
        }
        
        if chunk.count <= offset { return [] }
        
        let flagByte = chunk[offset]
        let frameTypeBits = flagByte & 0x0F
        let hasRefDir = (flagByte & 0x10) != 0
        guard let fType = VEVCFrameHeader.FrameType(rawValue: frameTypeBits) else {
            throw NSError(domain: "Player", code: 3, userInfo: [NSLocalizedDescriptionKey: "Invalid frame type"])
        }
        
        if fType == .copyFrame {
            return [flagByte]
        }
        
        guard offset + 21 <= chunk.count else {
            throw NSError(domain: "Player", code: 4, userInfo: [NSLocalizedDescriptionKey: "Chunk too small"])
        }
        
        @inline(__always)
        func readU32(_ base: Int) -> Int {
            return Int(UInt32(chunk[base]) << 24 | UInt32(chunk[base+1]) << 16 | UInt32(chunk[base+2]) << 8 | UInt32(chunk[base+3]))
        }
        
        let mvsSize    = readU32(offset + 1)
        let refDirSize = readU32(offset + 5)
        let layer0Size = readU32(offset + 9)
        let layer1Size = readU32(offset + 13)
        let layer2Size = readU32(offset + 17)
        
        let newLayer1Size = if 1 <= maxLayer { layer1Size } else { 0 }
        let newLayer2Size = if 2 <= maxLayer { layer2Size } else { 0 }
        
        let newHeader = VEVCFrameHeader(
            frameType: fType,
            hasRefDir: hasRefDir,
            mvsSize: mvsSize,
            refDirSize: refDirSize,
            layer0Size: layer0Size,
            layer1Size: newLayer1Size,
            layer2Size: newLayer2Size
        )
        
        var output = [UInt8]()
        output.append(contentsOf: newHeader.serialize())
        
        var payloadOffset = offset + 21
        if 0 < mvsSize {
            output.append(contentsOf: chunk[payloadOffset ..< payloadOffset + mvsSize])
            payloadOffset += mvsSize
        }
        if 0 < refDirSize {
            output.append(contentsOf: chunk[payloadOffset ..< payloadOffset + refDirSize])
            payloadOffset += refDirSize
        }
        if 0 < layer0Size {
            output.append(contentsOf: chunk[payloadOffset ..< payloadOffset + layer0Size])
            payloadOffset += layer0Size
        }
        if 0 < layer1Size {
            if 1 <= maxLayer {
                output.append(contentsOf: chunk[payloadOffset ..< payloadOffset + layer1Size])
            }
            payloadOffset += layer1Size
        }
        if 0 < layer2Size {
            if 2 <= maxLayer {
                output.append(contentsOf: chunk[payloadOffset ..< payloadOffset + layer2Size])
            }
            payloadOffset += layer2Size
        }
        
        return output
    }
    
    private func streamVEVC(url: URL) async throws {
        let data = try Data(contentsOf: url)
        let vevcData = [UInt8](data)
        
        var offset = 0
        var headerChunk: [UInt8]? = nil
        
        if offset + 4 <= vevcData.count && vevcData[offset] == 0x56 && vevcData[offset+1] == 0x45 && vevcData[offset+2] == 0x56 && vevcData[offset+3] == 0x43 {
            let headerStart = offset
            offset += 4
            let metadataSize = Int(try readUInt16BE(vevcData, offset: &offset))
            offset += metadataSize
            if offset <= vevcData.count {
                headerChunk = Array(vevcData[headerStart..<offset])
            }
        }
        
        var width = 0
        var height = 0
        var fps: Double = 30.0
        if let h = headerChunk {
            var hOffset = 0
            let fh = try VEVCFileHeader.deserialize(from: h, offset: &hOffset)
            width = fh.width
            height = fh.height
            fps = Double(fh.framerate)
        }
        
        let decoder0 = StreamingDecoderActor(maxLayer: 0, width: width, height: height)
        let decoder1 = StreamingDecoderActor(maxLayer: 1, width: width, height: height)
        let decoder2 = StreamingDecoderActor(maxLayer: 2, width: width, height: height)
        
        let frameInterval = 1.0 / fps
        var frameIndex: Double = 0.0
        
        await MainActor.run {
            self.videoFps = fps
            self.videoWidth = CGFloat(width)
            self.videoHeight = CGFloat(height)
            self.currentFrameIndex = 0.0
            self.totalFrames = 0.0
            self.isLoading = false
            self.isPlaying = true
            self.statusMessage = String(format: "Playing (%.2f fps)", fps)
        }
        
        let bufferCapacity = (width >= 1920) ? 15 : 30
        let frameBuffer = FrameBuffer(maxCapacity: bufferCapacity)
        
        let producerTask = Task {
            do {
                while offset < vevcData.count {
                    try Task.checkCancellation()
                    
                    let chunkStart = offset
                    let frameHeader = try VEVCFrameHeader.deserialize(from: vevcData, offset: &offset)
                    offset += frameHeader.payloadSize
                    
                    if vevcData.count < offset { break }
                    let chunk = Array(vevcData[chunkStart..<offset])
                    
                    let l0Chunk = try splitFrameChunk(chunk, maxLayer: 0)
                    let l1Chunk = try splitFrameChunk(chunk, maxLayer: 1)
                    let l2Chunk = try splitFrameChunk(chunk, maxLayer: 2)
                    
                    async let dec0 = decoder0.decodeNextFrame(chunk: l0Chunk)
                    async let dec1 = decoder1.decodeNextFrame(chunk: l1Chunk)
                    async let dec2 = decoder2.decodeNextFrame(chunk: l2Chunk)
                    
                    let (img0, img1, img2) = try await (dec0, dec1, dec2)
                    
                    let cgImg0 = try img0.flatMap { try createCGImage(from: $0) }
                    let cgImg1 = try img1.flatMap { try createCGImage(from: $0) }
                    let cgImg2 = try img2.flatMap { try createCGImage(from: $0) }
                    
                    try await frameBuffer.enqueue(FrameData(l0: cgImg0, l1: cgImg1, l2: cgImg2))
                }
            } catch {
                print("Producer error: \(error)")
            }
            await frameBuffer.finish()
        }
        
        var nextFrameTime = Date()
        var isBuffering = false
        
        while true {
            try Task.checkCancellation()
            
            while await MainActor.run(resultType: Bool.self, body: { self.isPlaying }) != true {
                try await Task.sleep(nanoseconds: 100_000_000)
                try Task.checkCancellation()
                nextFrameTime = Date() // Reset nextFrameTime when unpaused
            }
            
            let count = await frameBuffer.count
            if count == 0 {
                let finished = await frameBuffer.finished
                if finished { break }
                
                isBuffering = true
                await MainActor.run { self.statusMessage = "Buffering..." }
                try await Task.sleep(nanoseconds: 100_000_000)
                nextFrameTime = Date()
                continue
            }
            
            if isBuffering && count < bufferCapacity / 2 {
                let finished = await frameBuffer.finished
                if finished != true {
                    try await Task.sleep(nanoseconds: 100_000_000)
                    nextFrameTime = Date()
                    continue
                }
            }
            
            isBuffering = false
            await MainActor.run { self.statusMessage = String(format: "Playing (%.2f fps)", fps) }
            
            guard let frame = try await frameBuffer.dequeue() else {
                break
            }
            
            await MainActor.run {
                self.currentLayer0Image = frame.l0
                self.currentLayer1Image = frame.l1
                self.currentLayer2Image = frame.l2
                self.currentFrameIndex = frameIndex
                self.totalFrames = frameIndex + 1.0
            }
            frameIndex += 1.0
            
            let now = Date()
            if nextFrameTime < now {
                nextFrameTime = now
            }
            let delay = nextFrameTime.timeIntervalSince(now)
            if 0.0 < delay {
                try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            }
            nextFrameTime = nextFrameTime.addingTimeInterval(frameInterval)
        }
        
        producerTask.cancel()
        
        await MainActor.run {
            self.isPlaying = false
        }
    }
    
    func currentCGImage(for layer: Int) -> CGImage? {
        if layer == 0 {
            return currentLayer0Image
        } else if layer == 1 {
            return currentLayer1Image
        } else if layer == 2 {
            return currentLayer2Image
        }
        return nil
    }
}

// MARK: - Buffering Support

struct FrameData: @unchecked Sendable {
    let l0: CGImage?
    let l1: CGImage?
    let l2: CGImage?
}

actor FrameBuffer {
    private var frames: [FrameData] = []
    private var isFinished = false
    private let maxCapacity: Int
    
    init(maxCapacity: Int) {
        self.maxCapacity = maxCapacity
    }
    
    func enqueue(_ frame: FrameData) async throws {
        while frames.count >= maxCapacity {
            try Task.checkCancellation()
            try await Task.sleep(nanoseconds: 10_000_000) // 10ms wait
        }
        frames.append(frame)
    }
    
    func dequeue() async throws -> FrameData? {
        while frames.isEmpty {
            try Task.checkCancellation()
            if isFinished { return nil }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        return frames.removeFirst()
    }
    
    func finish() {
        isFinished = true
    }
    
    var finished: Bool { isFinished }
    
    var count: Int { frames.count }
}
