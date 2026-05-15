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
                HStack(spacing: 16) {
                    videoPane(title: "Layer 0", layerIndex: 0)
                    videoPane(title: "Layer 0+1", layerIndex: 1)
                    videoPane(title: "Layer 0+1+2", layerIndex: 2)
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
    private func videoPane(title: String, layerIndex: Int) -> some View {
        VStack {
            Text(title)
                .font(.headline)
            
            if let cgImage = viewModel.currentCGImage(for: layerIndex) {
                Image(cgImage, scale: 1.0, label: Text(title))
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color.black)
            } else {
                Rectangle()
                    .fill(Color.black)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }
}
