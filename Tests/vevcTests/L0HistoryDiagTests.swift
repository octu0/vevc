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
            pool: pool, qstep: nil, profile: 0x02
        )
        let dec = StreamingDecoderActor(maxLayer: 2, width: reader.width, height: reader.height, profile: 0x02)

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
                        XCTFail("frame \(frame) recon drift \(name): first diff at \(idx) enc=\(a[idx]) dec=\(b[idx])")
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
                        XCTFail("frame \(frame) l0 drift \(name): first diff at \(idx) enc=\(a[idx]) dec=\(b[idx])")
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
        XCTAssertLessThan(30, frame, "should have processed most frames")
    }

    func testProductionDefaultConsistency60f() async throws {
        let path = FileManager.default.currentDirectoryPath + "/.tmp/miko_60.y4m"
        guard let fh = FileHandle(forReadingAtPath: path) else {
            throw XCTSkip("miko_60.y4m not present")
        }
        let reader = try Y4MReader(fileHandle: fh)
        let pool = BlockViewPool()
        let enc = LayersEncodeActor(
            width: reader.width,
            height: reader.height,
            maxbitrate: 500_000,
            framerate: 60,
            zeroThreshold: 4,
            keyint: 30,
            sceneChangeThreshold: 500,
            pool: pool,
            qstep: nil,
            profile: 0x02,
            skipThreshold: 2,
            reconThresholdScale: 1,
            gop: 12,
            l2Cadence: 4,
            l1Cadence: 2,
            l0Cadence: 1,
            motionMaskingPx: 2,
            smooth: 1,
            temporalLayers: 1,
            skipModel: 1
        )
        let dec = StreamingDecoderActor(maxLayer: 2, width: reader.width, height: reader.height, profile: 0x02)

        var frame = 0
        var totalReconMismatches = 0
        var totalL0Mismatches = 0

        while let img = try reader.readFrame(), frame < 60 {
            let chunk = try await enc.encodeFrame(image: img)
            do {
                _ = try await dec.decodeNextFrame(chunk: chunk)
            } catch {
                XCTFail("frame \(frame): decode threw \(error)")
                break
            }

            var frameReconDiff = 0
            var frameL0Diff = 0

            // Full-resolution reconstruction drift (must be zero: closed loop)
            if let er = await enc.previousReconstructed, let dr = await dec.previousReconstructed {
                for (name, a, b) in [("Y", er.y, dr.y), ("Cb", er.cb, dr.cb), ("Cr", er.cr, dr.cr)] {
                    if a != b {
                        totalReconMismatches += 1
                        frameReconDiff += 1
                        let idx = zip(a, b).enumerated().first(where: { $1.0 != $1.1 })!.offset
                        print("DIAG recon mismatch frame=\(frame) plane=\(name) index=\(idx) enc=\(a[idx]) dec=\(b[idx]) (total=\(a.count))")
                    }
                }
            }

            // L0 chain drift (must be zero: bit-exact loop)
            if let el = await enc.l0State.prev, let dl = await dec.l0State.prev {
                for (name, a, b) in [("Y", el.y, dl.y), ("Cb", el.cb, dl.cb), ("Cr", el.cr, dl.cr)] {
                    if a != b {
                        totalL0Mismatches += 1
                        frameL0Diff += 1
                        let idx = zip(a, b).enumerated().first(where: { $1.0 != $1.1 })!.offset
                        print("DIAG l0 mismatch frame=\(frame) plane=\(name) index=\(idx) enc=\(a[idx]) dec=\(b[idx]) (total=\(a.count))")
                    }
                }
            }

            if frameReconDiff == 0 && frameL0Diff == 0 {
                print("FRAME \(frame): BITEXACT MATCH (recon: Y/Cb/Cr OK, L0: Y/Cb/Cr OK)")
            } else {
                print("FRAME \(frame): MISMATCH (reconDiff=\(frameReconDiff), l0Diff=\(frameL0Diff))")
            }


            frame += 1
        }

        print("BITEXACT_SUMMARY: frames=\(frame) recon_mismatches=\(totalReconMismatches) l0_mismatches=\(totalL0Mismatches)")
        XCTAssertEqual(frame, 60, "should have processed all 60 frames")
        XCTAssertEqual(totalReconMismatches, 0, "recon mismatches must be zero")
        XCTAssertEqual(totalL0Mismatches, 0, "L0 mismatches must be zero")
    }

    /// Same lockstep procedure as testProductionDefaultConsistency60f, with the
    /// periodic skip refresh enabled and the longer key interval it targets
    /// (#28). Forcing a long-skipping block back to inter rewrites its skip map
    /// entry, motion vector and ref dir after the learned skip decision, so the
    /// encoder's reconstruction and the decoder's must still agree frame by
    /// frame.
    func testSkipRefreshConsistency60fKeyint120() async throws {
        let path = FileManager.default.currentDirectoryPath + "/.tmp/miko_60.y4m"
        guard let fh = FileHandle(forReadingAtPath: path) else {
            throw XCTSkip("miko_60.y4m not present")
        }
        let reader = try Y4MReader(fileHandle: fh)
        let pool = BlockViewPool()
        let enc = LayersEncodeActor(
            width: reader.width,
            height: reader.height,
            maxbitrate: 500_000,
            framerate: 60,
            zeroThreshold: 4,
            keyint: 120,
            sceneChangeThreshold: 500,
            pool: pool,
            qstep: nil,
            profile: 0x02,
            skipThreshold: 2,
            reconThresholdScale: 1,
            gop: 12,
            l2Cadence: 4,
            l1Cadence: 2,
            l0Cadence: 1,
            motionMaskingPx: 2,
            smooth: 1,
            temporalLayers: 1,
            skipModel: 1,
            skipRefresh: 8
        )
        let dec = StreamingDecoderActor(maxLayer: 2, width: reader.width, height: reader.height, profile: 0x02)

        var frame = 0
        var totalReconMismatches = 0
        var totalL0Mismatches = 0

        while let img = try reader.readFrame(), frame < 60 {
            let chunk = try await enc.encodeFrame(image: img)
            do {
                _ = try await dec.decodeNextFrame(chunk: chunk)
            } catch {
                XCTFail("frame \(frame): decode threw \(error)")
                break
            }

            if let er = await enc.previousReconstructed, let dr = await dec.previousReconstructed {
                for (name, a, b) in [("Y", er.y, dr.y), ("Cb", er.cb, dr.cb), ("Cr", er.cr, dr.cr)] {
                    if a != b {
                        totalReconMismatches += 1
                        let idx = zip(a, b).enumerated().first(where: { $1.0 != $1.1 })!.offset
                        print("DIAG recon mismatch frame=\(frame) plane=\(name) index=\(idx) enc=\(a[idx]) dec=\(b[idx]) (total=\(a.count))")
                    }
                }
            }

            if let el = await enc.l0State.prev, let dl = await dec.l0State.prev {
                for (name, a, b) in [("Y", el.y, dl.y), ("Cb", el.cb, dl.cb), ("Cr", el.cr, dl.cr)] {
                    if a != b {
                        totalL0Mismatches += 1
                        let idx = zip(a, b).enumerated().first(where: { $1.0 != $1.1 })!.offset
                        print("DIAG l0 mismatch frame=\(frame) plane=\(name) index=\(idx) enc=\(a[idx]) dec=\(b[idx]) (total=\(a.count))")
                    }
                }
            }

            frame += 1
        }

        let stats = await enc.skipRefreshStats()
        print("SKIPREFRESH_BITEXACT_SUMMARY: frames=\(frame) recon_mismatches=\(totalReconMismatches) l0_mismatches=\(totalL0Mismatches) forced=\(stats.forced) examined=\(stats.examined)")
        XCTAssertEqual(frame, 60, "should have processed all 60 frames")
        XCTAssertEqual(totalReconMismatches, 0, "recon mismatches must be zero")
        XCTAssertEqual(totalL0Mismatches, 0, "L0 mismatches must be zero")
        XCTAssertLessThan(0, stats.forced, "the refresh pass must have forced blocks back to inter")
    }

    /// Same lockstep procedure with the quality floor enabled and keyint acting
    /// as an upper bound (#31). A floor-fired I frame arrives at a position the
    /// decoder never predicts, so the frame-type signalling and both
    /// reconstruction chains must still agree frame by frame.
    func testQualityFloorConsistency60fKeyint120() async throws {
        let path = FileManager.default.currentDirectoryPath + "/.tmp/miko_60.y4m"
        guard let fh = FileHandle(forReadingAtPath: path) else {
            throw XCTSkip("miko_60.y4m not present")
        }
        let reader = try Y4MReader(fileHandle: fh)
        let pool = BlockViewPool()
        let enc = LayersEncodeActor(
            width: reader.width,
            height: reader.height,
            maxbitrate: 500_000,
            framerate: 60,
            zeroThreshold: 4,
            keyint: 120,
            sceneChangeThreshold: 500,
            pool: pool,
            qstep: nil,
            profile: 0x02,
            skipThreshold: 2,
            reconThresholdScale: 1,
            gop: 12,
            l2Cadence: 4,
            l1Cadence: 2,
            l0Cadence: 1,
            motionMaskingPx: 2,
            smooth: 1,
            temporalLayers: 1,
            skipModel: 1,
            iqFloor: 200
        )
        let dec = StreamingDecoderActor(maxLayer: 2, width: reader.width, height: reader.height, profile: 0x02)

        var frame = 0
        var totalReconMismatches = 0
        var totalL0Mismatches = 0

        while let img = try reader.readFrame(), frame < 60 {
            let chunk = try await enc.encodeFrame(image: img)
            do {
                _ = try await dec.decodeNextFrame(chunk: chunk)
            } catch {
                XCTFail("frame \(frame): decode threw \(error)")
                break
            }

            if let er = await enc.previousReconstructed, let dr = await dec.previousReconstructed {
                for (name, a, b) in [("Y", er.y, dr.y), ("Cb", er.cb, dr.cb), ("Cr", er.cr, dr.cr)] {
                    if a != b {
                        totalReconMismatches += 1
                        let idx = zip(a, b).enumerated().first(where: { $1.0 != $1.1 })!.offset
                        print("DIAG recon mismatch frame=\(frame) plane=\(name) index=\(idx) enc=\(a[idx]) dec=\(b[idx]) (total=\(a.count))")
                    }
                }
            }

            if let el = await enc.l0State.prev, let dl = await dec.l0State.prev {
                for (name, a, b) in [("Y", el.y, dl.y), ("Cb", el.cb, dl.cb), ("Cr", el.cr, dl.cr)] {
                    if a != b {
                        totalL0Mismatches += 1
                        let idx = zip(a, b).enumerated().first(where: { $1.0 != $1.1 })!.offset
                        print("DIAG l0 mismatch frame=\(frame) plane=\(name) index=\(idx) enc=\(a[idx]) dec=\(b[idx]) (total=\(a.count))")
                    }
                }
            }

            frame += 1
        }

        let census = await enc.frameCensus()
        print("IQFLOOR_BITEXACT_SUMMARY: frames=\(frame) recon_mismatches=\(totalReconMismatches) l0_mismatches=\(totalL0Mismatches) floorI=\(census.iFloor) forcedI=\(census.iForced) sceneI=\(census.iScene) firings=\(census.firings.count)")
        XCTAssertEqual(frame, 60, "should have processed all 60 frames")
        XCTAssertEqual(totalReconMismatches, 0, "recon mismatches must be zero")
        XCTAssertEqual(totalL0Mismatches, 0, "L0 mismatches must be zero")
        XCTAssertLessThan(0, census.iFloor, "the floor must have fired at least one early I frame")
    }
}

