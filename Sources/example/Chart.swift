import Foundation
import SwiftUI
import Charts
import AppKit

struct CodecBenchmarkResult {
    let name: String
    let encTimeMs: Double
    let decTimeMs: Double
    let sizeKB: Double
    let avgPSNR: Double?
    let avgSSIM: Double?
}

@available(macOS 13.0, *)
struct SpeedSizeChart: View {
    let results: [CodecBenchmarkResult]
    
    var body: some View {
        let maxTime = results.map { max($0.encTimeMs, $0.decTimeMs) }.max() ?? 1.0
        let maxSize = results.map { $0.sizeKB }.max() ?? 1.0
        let ratio = maxSize > 0 ? (maxTime / maxSize) : 1.0
        
        VStack(alignment: .leading) {
            Text("Speed & Size Benchmark")
                .font(.title)
                .padding()
            
            Chart(results, id: \.name) { res in
                // Using BarMarks for Encoding and Decoding Times
                BarMark(
                    x: .value("Codec", res.name),
                    y: .value("Time (ms)", res.encTimeMs)
                )
                .foregroundStyle(by: .value("Type", "Encode Time"))
                .position(by: .value("Type", "Encode Time"))
                
                BarMark(
                    x: .value("Codec", res.name),
                    y: .value("Time (ms)", res.decTimeMs)
                )
                .foregroundStyle(by: .value("Type", "Decode Time"))
                .position(by: .value("Type", "Decode Time"))
                
                // Normalized Size
                let normalizedSize = res.sizeKB * ratio
                PointMark(
                    x: .value("Codec", res.name),
                    y: .value("Time (ms)", normalizedSize)
                )
                .foregroundStyle(by: .value("Type", "Size (KB)"))
                .symbol(.circle)
                .symbolSize(100)
                .annotation(position: .top) {
                    Text(String(format: "%.1f KB", res.sizeKB))
                        .font(.caption)
                        .padding(2)
                        .background(Color.white.opacity(0.8))
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading) { value in
                    AxisGridLine()
                    AxisTick()
                    AxisValueLabel {
                        if let time = value.as(Double.self) {
                            Text(String(format: "%.0f ms", time))
                        }
                    }
                }
                AxisMarks(position: .trailing) { value in
                    AxisGridLine()
                    AxisTick()
                    AxisValueLabel {
                        if let time = value.as(Double.self) {
                            let size = time / ratio
                            Text(String(format: "%.0f KB", size))
                        }
                    }
                }
            }
            .chartForegroundStyleScale([
                "Encode Time": Color.blue,
                "Decode Time": Color.green,
                "Size (KB)": Color.purple
            ])
            .frame(width: 800, height: 500)
            .padding()
        }
        .background(Color.white)
    }
}

@available(macOS 13.0, *)
struct PsnrChart: View {
    let results: [CodecBenchmarkResult]
    
    var body: some View {
        let validResults = results.filter { $0.avgPSNR != nil }
        
        VStack(alignment: .leading) {
            Text("PSNR Benchmark (Higher is better)")
                .font(.title)
                .padding()
            
            Chart(validResults, id: \.name) { res in
                BarMark(
                    x: .value("Codec", res.name),
                    y: .value("PSNR (dB)", res.avgPSNR!)
                )
                .foregroundStyle(.blue)
                .annotation(position: .top) {
                    Text(String(format: "%.2f", res.avgPSNR!))
                        .font(.caption)
                }
            }
            .chartYScale(domain: .automatic(includesZero: false))
            .frame(width: 800, height: 500)
            .padding()
        }
        .background(Color.white)
    }
}

@available(macOS 13.0, *)
struct SsimChart: View {
    let results: [CodecBenchmarkResult]
    
    var body: some View {
        let validResults = results.filter { $0.avgSSIM != nil }
        
        VStack(alignment: .leading) {
            Text("SSIM Benchmark (Closer to 1.0 is better)")
                .font(.title)
                .padding()
            
            Chart(validResults, id: \.name) { res in
                BarMark(
                    x: .value("Codec", res.name),
                    y: .value("SSIM", res.avgSSIM!)
                )
                .foregroundStyle(.green)
                .annotation(position: .top) {
                    Text(String(format: "%.4f", res.avgSSIM!))
                        .font(.caption)
                }
            }
            .chartYScale(domain: .automatic(includesZero: false))
            .frame(width: 800, height: 500)
            .padding()
        }
        .background(Color.white)
    }
}

@available(macOS 13.0, *)
@MainActor
func generateAndSaveCharts(results: [CodecBenchmarkResult], outDir: String = "docs") {
    // Speed & Size Chart
    let speedSizeView = SpeedSizeChart(results: results)
    let speedSizeRenderer = ImageRenderer(content: speedSizeView)
    speedSizeRenderer.scale = 2.0
    if let nsImage = speedSizeRenderer.nsImage,
       let tiffData = nsImage.tiffRepresentation,
       let bitmapImage = NSBitmapImageRep(data: tiffData),
       let pngData = bitmapImage.representation(using: .png, properties: [:]) {
        let path = URL(fileURLWithPath: "\(outDir)/speed_size.png")
        try? pngData.write(to: path)
        print("Saved \(path.path)")
    }
    
    // PSNR Chart
    if results.contains(where: { $0.avgPSNR != nil }) {
        let psnrView = PsnrChart(results: results)
        let psnrRenderer = ImageRenderer(content: psnrView)
        psnrRenderer.scale = 2.0
        if let nsImage = psnrRenderer.nsImage,
           let tiffData = nsImage.tiffRepresentation,
           let bitmapImage = NSBitmapImageRep(data: tiffData),
           let pngData = bitmapImage.representation(using: .png, properties: [:]) {
            let path = URL(fileURLWithPath: "\(outDir)/psnr.png")
            try? pngData.write(to: path)
            print("Saved \(path.path)")
        }
    }

    // SSIM Chart
    if results.contains(where: { $0.avgSSIM != nil }) {
        let ssimView = SsimChart(results: results)
        let ssimRenderer = ImageRenderer(content: ssimView)
        ssimRenderer.scale = 2.0
        if let nsImage = ssimRenderer.nsImage,
           let tiffData = nsImage.tiffRepresentation,
           let bitmapImage = NSBitmapImageRep(data: tiffData),
           let pngData = bitmapImage.representation(using: .png, properties: [:]) {
            let path = URL(fileURLWithPath: "\(outDir)/ssim.png")
            try? pngData.write(to: path)
            print("Saved \(path.path)")
        }
    }
}
