import SwiftUI
import UniformTypeIdentifiers

public struct CompareView: View {
    @ObservedObject var viewModel: CompareViewModel
    @State private var isFilePickerPresented = false
    
    public init(viewModel: CompareViewModel) {
        self.viewModel = viewModel
    }
    
    public var body: some View {
        VStack(spacing: 0) {
            // Header Bar
            headerBar
            
            Divider()
            
            // Main Video Area
            if viewModel.isLoading {
                loadingView
            } else if viewModel.totalFrames == 0 {
                emptyView
            } else {
                videoComparisonArea
            }
            
            Divider()
            
            // Bottom Sequence & Control Bar
            controlPanel
        }
        .frame(minWidth: 950, minHeight: 550)
    }
    
    // MARK: - Header Bar
    
    private var headerBar: some View {
        HStack(spacing: 16) {
            HStack(spacing: 6) {
                Image(systemName: "square.split.3x1.fill")
                    .foregroundColor(.accentColor)
                Text("Codec Quality Comparison (720p)")
                    .font(.headline)
            }
            
            if !viewModel.currentFileName.isEmpty {
                Text("•")
                    .foregroundColor(.secondary)
                Text(viewModel.currentFileName)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .lineLimit(1)
            }
            
            if 0.0 < viewModel.totalFrames {
                Text(String(format: "• %d frames @ %.2f fps", Int(viewModel.totalFrames), viewModel.fps))
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            if !viewModel.statusMessage.isEmpty {
                Text(viewModel.statusMessage)
                    .font(.callout)
                    .foregroundColor(viewModel.isLoading ? .accentColor : .secondary)
            }
            
            Button(action: { isFilePickerPresented = true }) {
                Label("Open File", systemImage: "folder")
            }
            .fileImporter(
                isPresented: $isFilePickerPresented,
                allowedContentTypes: [UTType.data, UTType.movie, UTType(filenameExtension: "y4m") ?? .data]
            ) { result in
                switch result {
                case .success(let url):
                    viewModel.loadFile(url: url)
                case .failure(let error):
                    print("File import error: \(error)")
                }
            }
            .disabled(viewModel.isLoading)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(Color(NSColor.windowBackgroundColor))
    }
    
    // MARK: - Video Comparison Area
    
    private var videoComparisonArea: some View {
        GeometryReader { geo in
            let availableWidth = geo.size.width - 32
            let paneWidth = max(280, (availableWidth - 32) / 3.0)
            let paneHeight = paneWidth * (9.0 / 16.0)
            
            ScrollView([.horizontal, .vertical]) {
                HStack(alignment: .top, spacing: 16) {
                    // 1. VEVC (Layer 2)
                    videoPane(
                        title: "VEVC (Layer 2)",
                        image: viewModel.currentVEVCImage,
                        result: viewModel.vevcResult,
                        width: paneWidth,
                        height: paneHeight,
                        accentColor: .blue
                    )
                    
                    // 2. H.264 (Baseline, CABAC, no B-frame)
                    videoPane(
                        title: "H.264 (Baseline, CABAC, no-B)",
                        image: viewModel.currentH264Image,
                        result: viewModel.h264Result,
                        width: paneWidth,
                        height: paneHeight,
                        accentColor: .green
                    )
                    
                    // 3. H.265 (HEVC)
                    videoPane(
                        title: "H.265 (HEVC)",
                        image: viewModel.currentHEVCImage,
                        result: viewModel.hevcResult,
                        width: paneWidth,
                        height: paneHeight,
                        accentColor: .orange
                    )
                }
                .padding(16)
                .frame(minWidth: geo.size.width, minHeight: geo.size.height)
            }
        }
    }
    
    @ViewBuilder
    private func videoPane(
        title: String,
        image: CGImage?,
        result: CodecResultData?,
        width: CGFloat,
        height: CGFloat,
        accentColor: Color
    ) -> some View {
        let currentIdx = Int(viewModel.currentFrameIndex)
        let frameData = (result != nil && currentIdx < result!.frames.count) ? result!.frames[currentIdx] : nil
        
        VStack(alignment: .leading, spacing: 8) {
            // Pane Header
            HStack {
                Circle()
                    .fill(accentColor)
                    .frame(width: 8, height: 8)
                Text(title)
                    .font(.headline)
                    .fontWeight(.semibold)
                Spacer()
                if let res = result {
                    Text(String(format: "Avg: %.0f kbps", res.avgBitrateKbps))
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            // Frame Image (720p 16:9)
            ZStack {
                RoundedRectangle(cornerRadius: 6)
                    .fill(Color.black)
                    .frame(width: width, height: height)
                
                if let cgImage = image {
                    Image(cgImage, scale: 1.0, label: Text(title))
                        .resizable()
                        .aspectRatio(16.0 / 9.0, contentMode: .fit)
                        .frame(width: width, height: height)
                } else {
                    ProgressView()
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(Color.secondary.opacity(0.2), lineWidth: 1)
            )
            
            // Metrics Footer
            HStack(spacing: 12) {
                if let fd = frameData {
                    Label(String(format: "PSNR: %.2f dB", fd.psnr), systemImage: "chart.line.uptrend.xyaxis")
                        .font(.system(.caption, design: .monospaced))
                    
                    Label(String(format: "SSIM: %.4f", fd.ssim), systemImage: "sparkles")
                        .font(.system(.caption, design: .monospaced))
                    
                    Spacer()
                    
                    Text(String(format: "%.1f KB", Double(fd.frameSizeBytes) / 1024.0))
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(.secondary)
                } else {
                    Text("Quality metrics ready on playback")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .padding(.horizontal, 4)
        }
        .frame(width: width)
    }
    
    // MARK: - Loading & Empty Views
    
    private var loadingView: some View {
        VStack(spacing: 16) {
            Spacer()
            ProgressView()
                .scaleEffect(1.5)
            Text(viewModel.statusMessage)
                .font(.headline)
                .foregroundColor(.secondary)
            Text("Encoding & Decoding VEVC, H.264, and H.265 at 720p...")
                .font(.subheadline)
                .foregroundColor(.secondary)
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
    
    private var emptyView: some View {
        VStack(spacing: 20) {
            Spacer()
            Image(systemName: "square.split.3x1.fill")
                .font(.system(size: 64))
                .foregroundColor(.secondary)
            Text("No Video File Loaded")
                .font(.title2)
                .fontWeight(.medium)
            Text("Select a Y4M video file to start 720p quality comparison across VEVC (Layer 2), H.264, and H.265.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 480)
            
            Button(action: { isFilePickerPresented = true }) {
                Label("Open File...", systemImage: "folder")
                    .font(.headline)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
            }
            .buttonStyle(.borderedProminent)
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
    
    // MARK: - Sequence & Control Panel
    
    private var controlPanel: some View {
        VStack(spacing: 12) {
            // Sequence / Seek Bar
            HStack(spacing: 12) {
                let current = Int(viewModel.currentFrameIndex)
                let total = max(1, Int(viewModel.totalFrames))
                
                Text(String(format: "Frame: %d / %d", current, total))
                    .font(.system(.subheadline, design: .monospaced))
                    .frame(width: 150, alignment: .leading)
                
                if 1.0 < viewModel.totalFrames {
                    Slider(
                        value: Binding(
                            get: { viewModel.currentFrameIndex },
                            set: {
                                viewModel.pause()
                                viewModel.currentFrameIndex = $0
                            }
                        ),
                        in: 0...Double(viewModel.totalFrames - 1),
                        step: 1
                    )
                    .disabled(viewModel.isLoading)
                } else {
                    Slider(value: .constant(0.0), in: 0...1)
                        .disabled(true)
                }
                
                let curTimeSec = Double(current) / max(viewModel.fps, 1.0)
                let totTimeSec = Double(total) / max(viewModel.fps, 1.0)
                Text(String(format: "%02d:%05.2f / %02d:%05.2f", Int(curTimeSec / 60), curTimeSec.truncatingRemainder(dividingBy: 60), Int(totTimeSec / 60), totTimeSec.truncatingRemainder(dividingBy: 60)))
                    .font(.system(.subheadline, design: .monospaced))
                    .foregroundColor(.secondary)
                    .frame(width: 150, alignment: .trailing)
            }
            
            // Playback and Encoding Controls
            HStack(spacing: 20) {
                // Playback Navigation
                HStack(spacing: 8) {
                    Button(action: { viewModel.seekToStart() }) {
                        Image(systemName: "backward.end.fill")
                    }
                    .help("Jump to Start")
                    
                    Button(action: { viewModel.stepBackward() }) {
                        Image(systemName: "backward.frame.fill")
                    }
                    .help("Step Backward (-1 Frame)")
                    
                    Button(action: { viewModel.togglePlay() }) {
                        Image(systemName: viewModel.isPlaying ? "pause.fill" : "play.fill")
                            .font(.title2)
                    }
                    .buttonStyle(.borderedProminent)
                    .help(viewModel.isPlaying ? "Pause" : "Play")
                    
                    Button(action: { viewModel.stepForward() }) {
                        Image(systemName: "forward.frame.fill")
                    }
                    .help("Step Forward (+1 Frame)")
                    
                    Button(action: { viewModel.seekToEnd() }) {
                        Image(systemName: "forward.end.fill")
                    }
                    .help("Jump to End")
                    
                    Toggle(isOn: $viewModel.isLooping) {
                        Image(systemName: "repeat")
                    }
                    .toggleStyle(.button)
                    .help("Loop Playback")
                }
                .disabled(viewModel.totalFrames == 0 || viewModel.isLoading)
                
                Divider()
                    .frame(height: 24)
                
                // Playback Speed
                HStack(spacing: 6) {
                    Text("Speed:")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    Picker("", selection: $viewModel.playbackSpeed) {
                        Text("0.25x").tag(0.25)
                        Text("0.5x").tag(0.5)
                        Text("1.0x").tag(1.0)
                        Text("2.0x").tag(2.0)
                    }
                    .labelsHidden()
                    .pickerStyle(.segmented)
                    .frame(width: 180)
                }
                .disabled(viewModel.totalFrames == 0 || viewModel.isLoading)
                
                Spacer()
                
                // Bitrate Controls
                HStack(spacing: 8) {
                    Text("Bitrate:")
                        .font(.subheadline)
                        .fontWeight(.medium)
                    
                    Text("\(viewModel.bitrate) kbps")
                        .font(.system(.subheadline, design: .monospaced))
                        .frame(width: 80, alignment: .trailing)
                    
                    Slider(
                        value: Binding(
                            get: { Double(viewModel.bitrate) },
                            set: { viewModel.bitrate = Int($0) }
                        ),
                        in: 200...8000,
                        step: 100
                    )
                    .frame(width: 140)
                    
                    Picker("Profile", selection: $viewModel.profile) {
                        Text("P1").tag(UInt8(1))
                        Text("P2").tag(UInt8(2))
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 80)
                    
                    Button(action: { viewModel.reencode() }) {
                        Label("Re-encode", systemImage: "arrow.triangle.2.circlepath")
                    }
                    .disabled(viewModel.totalFrames == 0 || viewModel.isLoading)
                }
            }
        }
        .padding(16)
        .background(Color(NSColor.windowBackgroundColor))
    }
}
