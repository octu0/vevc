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
    
    var layer0Frames: [CGImage] = []
    var layer1Frames: [CGImage] = []
    var layer2Frames: [CGImage] = []
    
    private var timer: Timer?
    
    func play() {
        if layer0Frames.isEmpty { return }
        isPlaying = true
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self = self else { return }
                if self.isPlaying == false { return }
                
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
        
        Task.detached(priority: .userInitiated) {
            do {
                let fileData: [UInt8]
                if url.pathExtension.lowercased() == "y4m" {
                    await MainActor.run { self.statusMessage = "Encoding Y4M to VEVC..." }
                    fileData = try await self.encodeY4M(url: url)
                } else {
                    await MainActor.run { self.statusMessage = "Reading VEVC file..." }
                    let data = try Data(contentsOf: url)
                    fileData = [UInt8](data)
                }
                
                await MainActor.run { self.statusMessage = "Decoding Layer 0..." }
                let l0 = try await self.decodeVEVC(data: fileData, maxLayer: 0)
                
                await MainActor.run { self.statusMessage = "Decoding Layer 1..." }
                let l1 = try await self.decodeVEVC(data: fileData, maxLayer: 1)
                
                await MainActor.run { self.statusMessage = "Decoding Layer 2..." }
                let l2 = try await self.decodeVEVC(data: fileData, maxLayer: 2)
                
                await MainActor.run {
                    self.layer0Frames = l0
                    self.layer1Frames = l1
                    self.layer2Frames = l2
                    
                    self.totalFrames = Double(l0.count)
                    self.currentFrameIndex = 0.0
                    self.statusMessage = "Ready"
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
    
    private func encodeY4M(url: URL) async throws -> [UInt8] {
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
            maxbitrate: 500 * 1000,
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
    
    private func decodeVEVC(data: [UInt8], maxLayer: Int) async throws -> [CGImage] {
        let decoder = Decoder(maxLayer: maxLayer)
        let images = try await decoder.decode(data: data)
        var cgImages: [CGImage] = []
        for img in images {
            let cgImg = try createCGImage(from: img)
            cgImages.append(cgImg)
        }
        return cgImages
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
