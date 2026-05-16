import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var viewModel = PlayerViewModel()
    @State private var isFilePickerPresented = false
    
    var body: some View {
        VStack {
            if viewModel.isLoading {
                ProgressView(viewModel.statusMessage)
                    .padding()
            } else {
                VStack(alignment: .leading, spacing: 8) {
                    // Layer 0: 1x scale (quarter resolution)
                    videoPane(title: "Layer 0", layerIndex: 0, scaleFactor: 1)
                    // Layer 0+1: 2x scale (half resolution)
                    videoPane(title: "Layer 0+1", layerIndex: 1, scaleFactor: 2)
                    // Layer 0+1+2: 4x scale (full resolution)
                    videoPane(title: "Layer 0+1+2", layerIndex: 2, scaleFactor: 4)
                }
                .padding()
                
                VStack {
                    if 0.0 < viewModel.totalFrames {
                        Slider(value: $viewModel.currentFrameIndex, in: 0...max(0, viewModel.totalFrames - 1), step: 1.0) { editing in
                            if editing {
                                viewModel.pause()
                            }
                        }
                        
                        Text("Frame: \(Int(viewModel.currentFrameIndex)) / \(Int(viewModel.totalFrames - 1))")
                    }
                    
                    HStack {
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
                        .disabled(viewModel.totalFrames <= 0.0)
                        
                        Spacer()
                    }
                    .padding()
                }
            }
        }
        .frame(minWidth: 800, minHeight: 400)
    }
    
    @ViewBuilder
    private func videoPane(title: String, layerIndex: Int, scaleFactor: Int) -> some View {
        VStack {
            Text(title)
                .font(.headline)
            
            if let cgImage = viewModel.currentCGImage(for: layerIndex) {
                let baseWidth = CGFloat(cgImage.width)
                let baseHeight = CGFloat(cgImage.height)
                // Display at proportional size relative to layer0 (scaleFactor = 1)
                // Use a fixed base unit derived from layer0 size to keep ratios consistent
                Image(cgImage, scale: 1.0, label: Text(title))
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(
                        width: baseWidth * CGFloat(scaleFactor) / 4.0,
                        height: baseHeight * CGFloat(scaleFactor) / 4.0
                    )
                    .background(Color.black)
            } else {
                Rectangle()
                    .fill(Color.black)
                    .frame(width: CGFloat(60 * scaleFactor), height: CGFloat(25 * scaleFactor))
            }
        }
    }
}
