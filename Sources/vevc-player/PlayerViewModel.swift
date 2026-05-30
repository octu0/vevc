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
    
    @Published var bitrate: Int = 500 {
        didSet {
            if bitrate < 100 {
                bitrate = 100
            }
            if 8000 < bitrate {
                bitrate = 8000
            }
        }
    }
    
    var layer0Frames: [CGImage] = []
    var layer1Frames: [CGImage] = []
    var layer2Frames: [CGImage] = []
    
    // Actual video FPS extracted from the decoded stream
    var videoFps: Int = 30
    
    private var timer: Timer?
    
    func play() {
        if layer0Frames.isEmpty { return }
        isPlaying = true
        timer?.invalidate()
        let interval = 1.0 / Double(videoFps)
        timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self = self else { return }
                if self.isPlaying != true { return }
                
                let nextIndex = Int(self.currentFrameIndex) + 1
                if nextIndex < Int(self.totalFrames) {
                    self.currentFrameIndex = Double(nextIndex)
                } else {
                    self.currentFrameIndex = 0.0
                    self.pause()
                }
            }
        }
    }
    
    func pause() {
        isPlaying = false
        timer?.invalidate()
        timer = nil
    }
    
    func loadFile(url: URL) {
        isLoading = true
        statusMessage = "Loading..."
        let currentBitrate = bitrate
        
        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self = self else { return }
            do {
                // 1. Obtain VEVC data (encode from y4m or read vevc directly)
                let vevcData: [UInt8]
                if url.pathExtension.lowercased() == "y4m" {
                    await MainActor.run { self.statusMessage = "Encoding Y4M to VEVC..." }
                    vevcData = try await Self.encodeY4M(url: url, bitrate: currentBitrate)
                } else {
                    await MainActor.run { self.statusMessage = "Reading VEVC file..." }
                    let data = try Data(contentsOf: url)
                    vevcData = [UInt8](data)
                }
                
                // 2. Split VEVC stream into per-layer variants using the library API
                await MainActor.run { self.statusMessage = "Splitting Layer 0..." }
                let layer0Data = try splitVEVCStream(input: vevcData, maxLayer: 0).data
                
                await MainActor.run { self.statusMessage = "Splitting Layer 0+1..." }
                let layer01Data = try splitVEVCStream(input: vevcData, maxLayer: 1).data
                
                // 3. Decode each split stream
                //    (splitter already stripped higher layers, so the decoder processes all available layers)
                await MainActor.run { self.statusMessage = "Decoding Layer 0..." }
                let l0 = try await Self.decodeVEVC(data: layer0Data)
                
                await MainActor.run { self.statusMessage = "Decoding Layer 0+1..." }
                let l1 = try await Self.decodeVEVC(data: layer01Data)
                
                await MainActor.run { self.statusMessage = "Decoding Layer 0+1+2..." }
                let l2 = try await Self.decodeVEVC(data: vevcData)
                
                // Extract FPS from decoded frames (use layer2 as canonical source)
                let detectedFps = l2.fps
                
                await MainActor.run {
                    self.layer0Frames = l0.images
                    self.layer1Frames = l1.images
                    self.layer2Frames = l2.images
                    
                    self.videoFps = detectedFps
                    self.totalFrames = Double(l2.images.count)
                    self.currentFrameIndex = 0.0
                    self.statusMessage = "Ready (\(detectedFps) fps)"
                    self.isLoading = false
                    self.play()
                }
            } catch {
                await MainActor.run {
                    self.statusMessage = "Error: \(error.localizedDescription)"
                    self.isLoading = false
                }
            }
        }
    }
    
    // Static method to avoid MainActor isolation issues in Task.detached
    private static func encodeY4M(url: URL, bitrate: Int) async throws -> [UInt8] {
        guard let fileHandle = try? FileHandle(forReadingFrom: url) else {
            throw NSError(domain: "Player", code: 1, userInfo: [NSLocalizedDescriptionKey: "Cannot open file"])
        }
        defer { try? fileHandle.close() }
        
        let y4mReader = try Y4MReader(fileHandle: fileHandle)
        var fps = 30
        if y4mReader.fpsHeader.starts(with: "F") {
            let parts = y4mReader.fpsHeader.dropFirst().split(separator: ":")
            if parts.count == 2, let num = Int(parts[0]), let den = Int(parts[1]), 0 < den {
                fps = num / den
                if fps == 0 { fps = 30 }
            }
        }
        
        let encoder = VEVCEncoder(
            width: y4mReader.width,
            height: y4mReader.height,
            maxbitrate: bitrate * 1000,
            framerate: fps,
            zeroThreshold: 3,
            keyint: 30,
            sceneChangeThreshold: 32
        )
        
        var encodedData: [UInt8] = []
        while let image = try y4mReader.readFrame() {
            let chunk = try await encoder.encode(image: image)
            encodedData.append(contentsOf: chunk)
        }
        
        return encodedData
    }
    
    // Returns decoded images and their FPS
    private static func decodeVEVC(data: [UInt8]) async throws -> (images: [CGImage], fps: Int) {
        let decoder = Decoder(maxLayer: 2)
        let images = try await decoder.decode(data: data)
        var cgImages: [CGImage] = []
        var fps = 30
        for img in images {
            if let imgFps = img.fps {
                fps = imgFps
            }
            let cgImg = try createCGImage(from: img)
            cgImages.append(cgImg)
        }
        return (cgImages, fps)
    }
    
    func currentCGImage(for layer: Int) -> CGImage? {
        let idx = Int(currentFrameIndex)
        if layer == 0 {
            if idx < layer0Frames.count { return layer0Frames[idx] }
        } else if layer == 1 {
            if idx < layer1Frames.count { return layer1Frames[idx] }
        } else if layer == 2 {
            if idx < layer2Frames.count { return layer2Frames[idx] }
        }
        return nil
    }
}
