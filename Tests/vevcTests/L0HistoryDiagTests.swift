import XCTest
import Foundation
@testable import vevc

/// Lockstep gate on real content (requires .tmp/miko_60.y4m; skips if absent):
/// after every frame, the encoder and decoder must agree bit-exactly on
/// (a) the full-resolution reconstruction, (b) the L0 reference chain, and
/// (c) every backward-adaptive history state. Any divergence in (a) or (b)
/// poisons (c) through the entropy contexts and eventually desyncs a
/// history-mode (0x20) stream — see the One-Pyramid Wave-1 postmortem.
final class L0HistoryDiagTests: XCTestCase {
    func testHistoryConsistencyOnRealContent() async throws {
        let path = FileManager.default.currentDirectoryPath + "/.tmp/miko_60.y4m"
        guard let fh = FileHandle(forReadingAtPath: path) else {
            throw XCTSkip("miko_60.y4m not present")
        }
        let reader = try Y4MReader(fileHandle: fh)
        let pool = BlockViewPool()
        let enc = LayersEncodeActor(
            width: reader.width, height: reader.height, maxbitrate: 2_500_000, framerate: 60,
            zeroThreshold: 0, keyint: 30, sceneChangeThreshold: 2000,
            pool: pool, qstep: nil, profile: 0x02, enableL0Loop: true)
        let dec = StreamingDecoderActor(maxLayer: 2, width: reader.width, height: reader.height, profile: 0x02, enableL0Loop: true)

        var frame = 0
        while let img = try reader.readFrame(), frame < 40 {
            let chunk = try await enc.encodeFrame(image: img)
            do {
                _ = try await dec.decodeNextFrame(chunk: chunk)
            } catch {
                XCTFail("frame \(frame): decode threw \(error)")
                break
            }

            // Full-resolution reconstruction drift (must be zero: closed loop)
            if let er = await enc.previousReconstructed, let dr = await dec.previousReconstructed {
                for (name, a, b) in [("Y", er.y, dr.y), ("Cb", er.cb, dr.cb), ("Cr", er.cr, dr.cr)] {
                    if a != b {
                        let idx = zip(a, b).enumerated().first(where: { $1.0 != $1.1 })!.offset
                        print("DIAG recon drift frame \(frame) \(name): first diff at \(idx) enc=\(a[idx]) dec=\(b[idx]) (of \(a.count))")
                        break
                    }
                }
            }
            // L0 chain drift (must be zero: bit-exact loop)
            if let el = await enc.l0State.prev, let dl = await dec.l0State.prev {
                for (name, a, b) in [("Y", el.y, dl.y), ("Cb", el.cb, dl.cb), ("Cr", el.cr, dl.cr)] {
                    if a != b {
                        let idx = zip(a, b).enumerated().first(where: { $1.0 != $1.1 })!.offset
                        print("DIAG l0 drift frame \(frame) \(name): first diff at \(idx) enc=\(a[idx]) dec=\(b[idx]) (of \(a.count))")
                        break
                    }
                }
            }

            let eh = await enc.entropyHistories
            let dh = await dec.entropyHistories
            guard let eh, let dh else { continue }
            var diverged = false
            for layer in 0..<3 {
                for plane in 0..<3 {
                    let es = eh.stream(layer: layer, plane: plane)
                    let ds = dh.stream(layer: layer, plane: plane)
                    if es.primed != ds.primed {
                        XCTFail("frame \(frame) L\(layer)/P\(plane): primed enc=\(es.primed) dec=\(ds.primed)")
                        diverged = true
                        continue
                    }
                    for c in 0..<entropyContextCount {
                        for t in 0..<64 {
                            if es.runCounts[c][t] != ds.runCounts[c][t] {
                                XCTFail("frame \(frame) L\(layer)/P\(plane) ctx\(c) runTok\(t): enc=\(es.runCounts[c][t]) dec=\(ds.runCounts[c][t])")
                                diverged = true
                            }
                            if es.valCounts[c][t] != ds.valCounts[c][t] {
                                XCTFail("frame \(frame) L\(layer)/P\(plane) ctx\(c) valTok\(t): enc=\(es.valCounts[c][t]) dec=\(ds.valCounts[c][t])")
                                diverged = true
                            }
                        }
                    }
                }
            }
            if diverged {
                for plane in 1..<3 {
                    let es = eh.stream(layer: 2, plane: plane)
                    let ds = dh.stream(layer: 2, plane: plane)
                    for c in 0..<entropyContextCount {
                        let er = es.runCounts[c].reduce(0, +)
                        let dr = ds.runCounts[c].reduce(0, +)
                        let ev = es.valCounts[c].reduce(0, +)
                        let dv = ds.valCounts[c].reduce(0, +)
                        print("DIAG L2/P\(plane) ctx\(c): run enc=\(er) dec=\(dr) | val enc=\(ev) dec=\(dv)")
                    }
                }
                break
            }
            frame += 1
        }
        XCTAssertGreaterThan(frame, 30, "should have processed most frames")
    }
}
