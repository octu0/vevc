import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var viewModel = PlayerViewModel()
    @State private var isFilePickerPresented = false
    
    var body: some View {
        VStack {
            if viewModel.isLoading {
                Spacer()
                ProgressView(viewModel.statusMessage)
                    .padding()
                Spacer()
            } else {
                HStack(alignment: .bottom, spacing: 16) {
                    // Layer 0: quarter resolution
                    videoPane(title: "Layer 0", layerIndex: 0, weight: 1.0)
                    // Layer 0+1: half resolution
                    videoPane(title: "Layer 0+1", layerIndex: 1, weight: 2.0)
                    // Layer 0+1+2: full resolution
                    videoPane(title: "Layer 0+1+2", layerIndex: 2, weight: 4.0)
                }
                .padding()
            }
            
            Divider()
            
            VStack {
                if 0.0 < viewModel.totalFrames {
                    Text("Frame: \(Int(viewModel.currentFrameIndex))")
                        .font(.headline)
                        .padding(.bottom, 4)
                }
                
                HStack {
                    HStack(spacing: 8) {
                        Text("Bitrate: \(viewModel.bitrate) kbps")
                            .font(.callout)
                        Slider(value: Binding(
                            get: { Double(viewModel.bitrate) },
                            set: { viewModel.bitrate = Int($0) }
                        ), in: 100...8000)
                        .frame(width: 150)
                    }
                    
                    Button("Open File") {
                        isFilePickerPresented = true
                    }
                    .fileImporter(isPresented: $isFilePickerPresented, allowedContentTypes: [UTType.data, UTType.movie, UTType(filenameExtension: "y4m")!, UTType(filenameExtension: "vevc")!]) { result in
                        switch result {
                        case .success(let url):
                            viewModel.loadFile(url: url)
                        case .failure(let error):
                            print(error)
                        }
                    }
                    .disabled(viewModel.isLoading)
                    
                    Spacer()
                    
                    Button(action: {
                        if viewModel.isPlaying {
                            viewModel.pause()
                        } else {
                            viewModel.play()
                        }
                    }) {
                        Image(systemName: viewModel.isPlaying ? "pause.fill" : "play.fill")
                            .font(.title)
                    }
                    .disabled(viewModel.totalFrames <= 0.0 || viewModel.isLoading)
                    
                    Spacer()
                }
                .padding()
            }
        }
        // 1.5x width to accommodate Layer 2 (1.0) + Layer 1 (0.5) + Layer 0 (0.25) horizontally.
        // We set ideal size so the window can grow with the video content.
        .frame(minWidth: 800, minHeight: 400)
        .frame(idealWidth: viewModel.videoWidth * 1.75, idealHeight: viewModel.videoHeight + 200)
    }
    
    @ViewBuilder
    private func videoPane(title: String, layerIndex: Int, weight: CGFloat) -> some View {
        VStack {
            Text(title)
                .font(.headline)
            
            if let cgImage = viewModel.currentCGImage(for: layerIndex) {
                Image(cgImage, scale: 1.0, label: Text(title))
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    // Assign relative layout width based on the layer's expected resolution proportion
                    // Layer 2 gets weight 4.0, Layer 1 gets 2.0, Layer 0 gets 1.0
                    .frame(width: viewModel.videoWidth * (weight / 4.0))
                    .background(Color.black)
            } else {
                Rectangle()
                    .fill(Color.black)
                    .frame(width: viewModel.videoWidth * (weight / 4.0), height: viewModel.videoHeight * (weight / 4.0))
                    .aspectRatio(contentMode: .fit)
            }
        }
    }
}
