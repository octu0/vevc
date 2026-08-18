import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    let args: PlayerArguments
    
    @State private var mode: AppMode
    @StateObject private var layerPlayerViewModel = PlayerViewModel()
    @StateObject private var compareViewModel: CompareViewModel
    
    enum AppMode: String, CaseIterable, Identifiable {
        case compare = "720p Codec Compare"
        case layer = "VEVC Layer Player"
        
        var id: String { rawValue }
    }
    
    init(args: PlayerArguments = PlayerArguments()) {
        self.args = args
        self._mode = State(initialValue: args.isCompareMode ? .compare : .layer)
        self._compareViewModel = StateObject(wrappedValue: CompareViewModel(
            initialBitrate: args.bitrate,
            initialProfile: args.profile
        ))
    }
    
    var body: some View {
        VStack(spacing: 0) {
            // Mode Selector Bar
            HStack {
                Picker("Mode", selection: $mode) {
                    ForEach(AppMode.allCases) { m in
                        Text(m.rawValue).tag(m)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 350)
                
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(Color(NSColor.windowBackgroundColor))
            
            Divider()
            
            // Content View based on mode
            if mode == .compare {
                CompareView(viewModel: compareViewModel)
            } else {
                layerPlayerView
            }
        }
        .onAppear {
            if let path = args.inputPath {
                let url = URL(fileURLWithPath: path)
                if mode == .compare {
                    compareViewModel.loadFile(url: url)
                } else {
                    layerPlayerViewModel.loadFile(url: url)
                }
            }
        }
    }
    
    // MARK: - Layer Player View
    
    @State private var isLayerFilePickerPresented = false
    
    private var layerPlayerView: some View {
        VStack {
            if layerPlayerViewModel.isLoading {
                Spacer()
                ProgressView(layerPlayerViewModel.statusMessage)
                    .padding()
                Spacer()
            } else {
                ScrollView([.horizontal, .vertical]) {
                    if layerPlayerViewModel.videoWidth > layerPlayerViewModel.videoHeight {
                        VStack(alignment: .center, spacing: 16) {
                            layerVideoPane(title: "Layer 0", layerIndex: 0, weight: 1.0)
                            layerVideoPane(title: "Layer 0+1", layerIndex: 1, weight: 2.0)
                            layerVideoPane(title: "Layer 0+1+2", layerIndex: 2, weight: 4.0)
                        }
                        .padding()
                    } else {
                        HStack(alignment: .bottom, spacing: 16) {
                            layerVideoPane(title: "Layer 0", layerIndex: 0, weight: 1.0)
                            layerVideoPane(title: "Layer 0+1", layerIndex: 1, weight: 2.0)
                            layerVideoPane(title: "Layer 0+1+2", layerIndex: 2, weight: 4.0)
                        }
                        .padding()
                    }
                }
            }
            
            Divider()
            
            VStack {
                if 0.0 < layerPlayerViewModel.totalFrames {
                    Text("Frame: \(Int(layerPlayerViewModel.currentFrameIndex))")
                        .font(.headline)
                        .padding(.bottom, 4)
                }
                
                HStack {
                    HStack(spacing: 8) {
                        Text("Bitrate: \(layerPlayerViewModel.bitrate) kbps")
                            .font(.callout)
                        Slider(value: Binding(
                            get: { Double(layerPlayerViewModel.bitrate) },
                            set: { layerPlayerViewModel.bitrate = Int($0) }
                        ), in: 100...8000)
                        .frame(width: 150)
                    }
                    
                    HStack(spacing: 8) {
                        Text("Profile")
                        Toggle("1", isOn: Binding(
                            get: { layerPlayerViewModel.profile == 1 },
                            set: { if $0 { layerPlayerViewModel.profile = 1 } }
                        ))
                        .toggleStyle(.checkbox)
                        
                        Toggle("2", isOn: Binding(
                            get: { layerPlayerViewModel.profile == 2 },
                            set: { if $0 { layerPlayerViewModel.profile = 2 } }
                        ))
                        .toggleStyle(.checkbox)
                    }
                    .frame(width: 150)
                    .disabled(layerPlayerViewModel.isLoading)
                    
                    Button("Open File") {
                        isLayerFilePickerPresented = true
                    }
                    .fileImporter(isPresented: $isLayerFilePickerPresented, allowedContentTypes: [UTType.data, UTType.movie, UTType(filenameExtension: "y4m")!, UTType(filenameExtension: "vevc")!]) { result in
                        switch result {
                        case .success(let url):
                            layerPlayerViewModel.loadFile(url: url)
                        case .failure(let error):
                            print(error)
                        }
                    }
                    .disabled(layerPlayerViewModel.isLoading)
                    
                    Spacer()
                    
                    Button(action: {
                        if layerPlayerViewModel.isPlaying {
                            layerPlayerViewModel.pause()
                        } else {
                            layerPlayerViewModel.play()
                        }
                    }) {
                        let icon = if layerPlayerViewModel.isPlaying { "pause.fill" } else { "play.fill" }
                        Image(systemName: icon)
                            .font(.title)
                    }
                    .disabled(layerPlayerViewModel.totalFrames <= 0.0 || layerPlayerViewModel.isLoading)
                    
                    Spacer()
                }
                .padding()
            }
        }
        .frame(minWidth: 800, minHeight: 400)
    }
    
    @ViewBuilder
    private func layerVideoPane(title: String, layerIndex: Int, weight: CGFloat) -> some View {
        VStack {
            Text(title)
                .font(.headline)
            
            if let cgImage = layerPlayerViewModel.currentCGImage(for: layerIndex) {
                Image(cgImage, scale: 1.0, label: Text(title))
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: max(160, layerPlayerViewModel.videoWidth * (weight / 4.0)))
                    .background(Color.black)
            } else {
                Rectangle()
                    .fill(Color.black)
                    .frame(
                        width: max(160, layerPlayerViewModel.videoWidth * (weight / 4.0)),
                        height: max(90, layerPlayerViewModel.videoHeight * (weight / 4.0))
                    )
                    .aspectRatio(contentMode: .fit)
            }
        }
    }
}
