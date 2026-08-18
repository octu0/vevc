import Foundation
import SwiftUI
import Combine
import CoreGraphics
import QuartzCore
import vevc

@MainActor
public final class CompareViewModel: ObservableObject {
    @Published public var isPlaying: Bool = false
    @Published public var currentFrameIndex: Double = 0.0 {
        didSet {
            updateCurrentImages()
        }
    }
    @Published public var totalFrames: Double = 0.0
    @Published public var fps: Double = 30.0
    @Published public var isLoading: Bool = false
    @Published public var statusMessage: String = ""
    
    @Published public var bitrate: Int = 1000 {
        didSet {
            if bitrate < 100 { bitrate = 100 }
            if 15000 < bitrate { bitrate = 15000 }
        }
    }
    @Published public var profile: UInt8 = 0x01
    @Published public var isLooping: Bool = true
    @Published public var playbackSpeed: Double = 1.0
    @Published public var currentFileName: String = ""
    
    // Decoded results for each codec
    @Published public var vevcResult: CodecResultData?
    @Published public var h264Result: CodecResultData?
    @Published public var hevcResult: CodecResultData?
    
    // Rendered CGImages for current frame
    @Published public var currentVEVCImage: CGImage?
    @Published public var currentH264Image: CGImage?
    @Published public var currentHEVCImage: CGImage?
    @Published public var currentOrigImage: CGImage?
    
    // Stored 720p original frames
    private var orig720pFrames: [YCbCrImage] = []
    private var orig720pCGImages: [CGImage?] = []
    
    private var playbackTask: Task<Void, Never>?
    private var processingTask: Task<Void, Never>?
    
    public init(initialBitrate: Int = 1000, initialProfile: UInt8 = 0x01) {
        self.bitrate = initialBitrate
        self.profile = initialProfile
    }
    
    // MARK: - Playback Control (Wall-clock Precision)
    
    public func play() {
        guard 0.0 < totalFrames else { return }
        isPlaying = true
        startPlaybackLoop()
    }
    
    public func pause() {
        isPlaying = false
        playbackTask?.cancel()
        playbackTask = nil
    }
    
    public func togglePlay() {
        if isPlaying {
            pause()
        } else {
            play()
        }
    }
    
    public func stepForward() {
        pause()
        if currentFrameIndex + 1.0 < totalFrames {
            currentFrameIndex += 1.0
        } else if isLooping {
            currentFrameIndex = 0.0
        }
    }
    
    public func stepBackward() {
        pause()
        if 0.0 < currentFrameIndex {
            currentFrameIndex -= 1.0
        } else if isLooping && 0.0 < totalFrames {
            currentFrameIndex = totalFrames - 1.0
        }
    }
    
    public func seekToStart() {
        pause()
        currentFrameIndex = 0.0
    }
    
    public func seekToEnd() {
        pause()
        if 0.0 < totalFrames {
            currentFrameIndex = totalFrames - 1.0
        }
    }
    
    private func startPlaybackLoop() {
        playbackTask?.cancel()
        
        let targetFps = self.fps
        let totalCount = Int(self.totalFrames)
        guard 0 < totalCount else { return }
        
        let initialIndex = self.currentFrameIndex
        
        playbackTask = Task { [weak self] in
            guard let self = self else { return }
            
            var effectiveSpeed = await MainActor.run { self.playbackSpeed }
            var effectiveFps = max(1.0, targetFps * effectiveSpeed)
            var baseTime = CACurrentMediaTime() - (initialIndex / effectiveFps)
            
            while true {
                try? Task.checkCancellation()
                if Task.isCancelled { break }
                
                let isPlayingNow = await MainActor.run { self.isPlaying }
                if isPlayingNow != true { break }
                
                let currentSpeed = await MainActor.run { self.playbackSpeed }
                if currentSpeed != effectiveSpeed {
                    // Update speed smoothly
                    let currentFrame = await MainActor.run { self.currentFrameIndex }
                    effectiveSpeed = currentSpeed
                    effectiveFps = max(1.0, targetFps * effectiveSpeed)
                    baseTime = CACurrentMediaTime() - (currentFrame / effectiveFps)
                }
                
                let now = CACurrentMediaTime()
                let elapsed = max(0.0, now - baseTime)
                let calculatedFrame = Int(elapsed * effectiveFps)
                
                if calculatedFrame < totalCount {
                    await MainActor.run {
                        if Int(self.currentFrameIndex) != calculatedFrame {
                            self.currentFrameIndex = Double(calculatedFrame)
                        }
                    }
                } else {
                    let looping = await MainActor.run { self.isLooping }
                    if looping {
                        baseTime = CACurrentMediaTime()
                        await MainActor.run { self.currentFrameIndex = 0.0 }
                    } else {
                        await MainActor.run {
                            self.currentFrameIndex = Double(totalCount - 1)
                            self.pause()
                        }
                        break
                    }
                }
                
                let nextFrameNum = calculatedFrame + 1
                let nextTargetTime = baseTime + (Double(nextFrameNum) / effectiveFps)
                let sleepDuration = nextTargetTime - CACurrentMediaTime()
                if 0.001 < sleepDuration {
                    try? await Task.sleep(nanoseconds: UInt64(sleepDuration * 1_000_000_000))
                } else {
                    // Yield to prevent pegging the thread if falling behind
                    await Task.yield()
                }
            }
        }
    }
    
    // MARK: - Image Rendering
    
    private func updateCurrentImages() {
        let idx = Int(currentFrameIndex)
        
        // VEVC
        if let vevcFrames = vevcResult?.frames, idx < vevcFrames.count {
            currentVEVCImage = vevcFrames[idx].cgImage
        } else {
            currentVEVCImage = nil
        }
        
        // H.264
        if let h264Frames = h264Result?.frames, idx < h264Frames.count {
            currentH264Image = h264Frames[idx].cgImage
        } else {
            currentH264Image = nil
        }
        
        // HEVC
        if let hevcFrames = hevcResult?.frames, idx < hevcFrames.count {
            currentHEVCImage = hevcFrames[idx].cgImage
        } else {
            currentHEVCImage = nil
        }
        
        // Original 720p
        if idx < orig720pCGImages.count {
            currentOrigImage = orig720pCGImages[idx]
        } else {
            currentOrigImage = nil
        }
    }
    
    // MARK: - Loading & Encoding Pipeline
    
    public func loadFile(url: URL) {
        pause()
        isLoading = true
        statusMessage = "Opening Y4M file..."
        currentFileName = url.lastPathComponent
        
        processingTask?.cancel()
        processingTask = Task.detached(priority: .userInitiated) { [weak self] in
            guard let self = self else { return }
            do {
                guard let fileHandle = try? FileHandle(forReadingFrom: url) else {
                    throw NSError(domain: "ComparePlayer", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to open \(url.path)"])
                }
                defer { try? fileHandle.close() }
                
                let y4mReader = try Y4MReader(fileHandle: fileHandle)
                var detectedFps: Double = 30.0
                if y4mReader.fpsHeader.starts(with: "F") {
                    let parts = y4mReader.fpsHeader.dropFirst().split(separator: ":")
                    if parts.count == 2, let num = Double(parts[0]), let den = Double(parts[1]), 0.0 < den {
                        detectedFps = num / den
                    } else if parts.count == 1, let num = Double(parts[0]) {
                        detectedFps = num
                    }
                    if detectedFps <= 0.0 { detectedFps = 30.0 }
                }
                
                let parsedFps = detectedFps
                await MainActor.run {
                    self.fps = parsedFps
                    self.statusMessage = String(format: "Reading and resizing frames to 720p (detected %.2f fps)...", parsedFps)
                }
                
                var frames720p: [YCbCrImage] = []
                var origCGs: [CGImage?] = []
                var frameCounter = 0
                while let origFrame = try y4mReader.readFrame() {
                    try Task.checkCancellation()
                    let resized = resizeYCbCrImage720p(image: origFrame)
                    frames720p.append(resized)
                    origCGs.append(try? createCGImage(from: resized))
                    frameCounter += 1
                    if frameCounter % 15 == 0 {
                        let count = frameCounter
                        await MainActor.run {
                            self.statusMessage = "Resizing to 720p: \(count) frames loaded"
                        }
                    }
                }
                
                guard frames720p.isEmpty != true else {
                    throw NSError(domain: "ComparePlayer", code: 2, userInfo: [NSLocalizedDescriptionKey: "No video frames found in Y4M"])
                }
                
                await MainActor.run {
                    self.orig720pFrames = frames720p
                    self.orig720pCGImages = origCGs
                    self.totalFrames = Double(frames720p.count)
                }
                
                try await self.runEncodingPipeline()
            } catch {
                if Task.isCancelled != true {
                    await MainActor.run {
                        self.statusMessage = "Error: \(error.localizedDescription)"
                        self.isLoading = false
                    }
                }
            }
        }
    }
    
    public func reencode() {
        guard orig720pFrames.isEmpty != true else { return }
        pause()
        isLoading = true
        statusMessage = "Re-encoding with Bitrate \(bitrate) kbps..."
        
        processingTask?.cancel()
        processingTask = Task.detached(priority: .userInitiated) { [weak self] in
            guard let self = self else { return }
            do {
                try await self.runEncodingPipeline()
            } catch {
                if Task.isCancelled != true {
                    await MainActor.run {
                        self.statusMessage = "Error: \(error.localizedDescription)"
                        self.isLoading = false
                    }
                }
            }
        }
    }
    
    private func runEncodingPipeline() async throws {
        let frames = orig720pFrames
        let targetBitrate = bitrate
        let targetFps = Int(round(fps))
        let targetProfile = profile
        
        // 1. VEVC
        let vevc = try await runVEVCPipeline(
            images: frames,
            bitrate: targetBitrate,
            fps: targetFps,
            profile: targetProfile,
            onProgress: { msg in
                Task { @MainActor in self.statusMessage = msg }
            }
        )
        try Task.checkCancellation()
        
        // 2. H.264
        let h264 = try await runH264Pipeline(
            images: frames,
            bitrate: targetBitrate,
            fps: targetFps,
            onProgress: { msg in
                Task { @MainActor in self.statusMessage = msg }
            }
        )
        try Task.checkCancellation()
        
        // 3. HEVC
        let hevc = try await runHEVCPipeline(
            images: frames,
            bitrate: targetBitrate,
            fps: targetFps,
            onProgress: { msg in
                Task { @MainActor in self.statusMessage = msg }
            }
        )
        try Task.checkCancellation()
        
        await MainActor.run {
            self.vevcResult = vevc
            self.h264Result = h264
            self.hevcResult = hevc
            self.isLoading = false
            self.statusMessage = String(format: "Ready - %d frames @ 720p (%.2f fps)", frames.count, self.fps)
            self.currentFrameIndex = 0.0
            self.updateCurrentImages()
            self.play()
        }
    }
}
