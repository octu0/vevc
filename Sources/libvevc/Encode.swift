import Foundation
import vevc

final class CAPIEncoderContext: @unchecked Sendable {
    let encoder: VEVCEncoder
    let width: Int
    let height: Int
    
    let resultPtr: UnsafeMutablePointer<vevc_enc_result_t>
    var dataPtr: UnsafeMutablePointer<UInt8>?
    var dataCapacity: Int = 0
    
    init(width: Int, height: Int, maxbitrate: Int, framerate: Int, zeroThreshold: Int, keyint: Int, sceneChangeThreshold: Int) {
        self.encoder = VEVCEncoder(
            width: width, height: height, maxbitrate: maxbitrate, framerate: framerate,
            zeroThreshold: zeroThreshold, keyint: keyint, sceneChangeThreshold: sceneChangeThreshold
        )
        self.width = width
        self.height = height
        
        self.resultPtr = UnsafeMutablePointer<vevc_enc_result_t>.allocate(capacity: 1)
        self.resultPtr.pointee.status = VEVC_OK
        self.resultPtr.pointee.data = nil
        self.resultPtr.pointee.size = 0
        self.resultPtr.pointee.is_iframe = 0
        self.resultPtr.pointee.is_copyframe = 0
    }
    
    deinit {
        if let data = dataPtr {
            data.deallocate()
        }
        resultPtr.deallocate()
    }
    
    func encode(img: YCbCrImage) throws -> [UInt8]? {
        let semaphore = DispatchSemaphore(value: 0)
        final class Box: @unchecked Sendable { 
            var result: [UInt8]?
            var error: Error?
        }
        let box = Box()
        
        let targetEncoder = self.encoder
        
        Task {
            do {
                box.result = try await targetEncoder.encode(image: img)
            } catch {
                box.error = error
            }
            semaphore.signal()
        }
        semaphore.wait()
        
        if let err = box.error {
            throw err
        }
        
        return box.result
    }
}

private func copyPlane(src: UnsafePointer<UInt8>, stride: Int, width: Int, height: Int) -> [UInt8] {
    if stride == width {
        return Array(UnsafeBufferPointer(start: src, count: width * height))
    } else {
        var plane = [UInt8](repeating: 0, count: width * height)
        for y in 0..<height {
            let dstOffset = y * width
            let srcOffset = y * stride
            for x in 0..<width {
                plane[dstOffset + x] = src.advanced(by: srcOffset + x).pointee
            }
        }
        return plane
    }
}

@_cdecl("vevc_enc_create")
public func vevc_enc_create(param: UnsafeRawPointer) -> UnsafeMutableRawPointer? {
    let p = param.load(as: vevc_enc_param_t.self)
    let ctx = CAPIEncoderContext(
        width: Int(p.width),
        height: Int(p.height),
        maxbitrate: Int(p.maxbitrate),
        framerate: Int(p.framerate),
        zeroThreshold: Int(p.zero_threshold),
        keyint: Int(p.keyint),
        sceneChangeThreshold: Int(p.scene_change_threshold)
    )
    return Unmanaged.passRetained(ctx).toOpaque()
}

@_cdecl("vevc_enc_encode")
public func vevc_enc_encode(enc: UnsafeMutableRawPointer, imgb: UnsafeRawPointer) -> UnsafeMutableRawPointer? {
    let ctx = Unmanaged<CAPIEncoderContext>.fromOpaque(enc).takeUnretainedValue()
    let src = imgb.load(as: vevc_enc_imgb_t.self)
    
    var img = YCbCrImage(width: ctx.width, height: ctx.height, fps: ctx.encoder.framerate)
    
    if let yBase = src.y {
        img.yPlane = copyPlane(src: yBase, stride: Int(src.stride_y), width: ctx.width, height: ctx.height)
    }
    if let uBase = src.u {
        img.cbPlane = copyPlane(src: uBase, stride: Int(src.stride_u), width: ctx.width / 2, height: ctx.height / 2)
    }
    if let vBase = src.v {
        img.crPlane = copyPlane(src: vBase, stride: Int(src.stride_v), width: ctx.width / 2, height: ctx.height / 2)
    }
    
    let res = ctx.resultPtr
    res.pointee.status = VEVC_OK
    res.pointee.data = nil
    res.pointee.size = 0
    res.pointee.is_iframe = 0
    res.pointee.is_copyframe = 0
    
    let bytes: [UInt8]
    do {
        if let b = try ctx.encode(img: img) {
            bytes = b
        } else {
            res.pointee.status = VEVC_ERR
            return UnsafeMutableRawPointer(res)
        }
    } catch {
        res.pointee.status = VEVC_ERR
        return UnsafeMutableRawPointer(res)
    }
    
    if bytes.isEmpty != true {
        var isCopy = false
        var isIFrame = false
        var offset = 0
        if 4 <= bytes.count && bytes[0] == 0x56 && bytes[1] == 0x45 && bytes[2] == 0x56 && bytes[3] == 0x43 {
            if 6 <= bytes.count {
                let metaSize = Int(bytes[4]) << 8 | Int(bytes[5])
                offset = 6 + metaSize
            }
        }
        if offset < bytes.count {
            let flag = bytes[offset]
            if flag == 0x01 {
                isCopy = true
            } else if flag == 0x00 {
                isIFrame = true
            }
        }
        res.pointee.is_iframe = if isIFrame { 1 } else { 0 }
        res.pointee.is_copyframe = if isCopy { 1 } else { 0 }
        
        if ctx.dataCapacity < bytes.count {
            if let old = ctx.dataPtr { old.deallocate() }
            let newCap = max(bytes.count, ctx.dataCapacity * 2)
            ctx.dataCapacity = newCap
            ctx.dataPtr = UnsafeMutablePointer<UInt8>.allocate(capacity: newCap)
        }
        let buffer = ctx.dataPtr!
        bytes.withUnsafeBufferPointer { ptr in
            buffer.update(from: ptr.baseAddress!, count: bytes.count)
        }
        res.pointee.data = buffer
        res.pointee.size = bytes.count
    }
    
    return UnsafeMutableRawPointer(res)
}

@_cdecl("vevc_enc_flush")
public func vevc_enc_flush(enc: UnsafeMutableRawPointer) -> UnsafeMutableRawPointer? {
    let ctx = Unmanaged<CAPIEncoderContext>.fromOpaque(enc).takeUnretainedValue()
    let res = ctx.resultPtr
    res.pointee.status = VEVC_OK
    res.pointee.data = nil
    res.pointee.size = 0
    res.pointee.is_iframe = 0
    res.pointee.is_copyframe = 0
    return UnsafeMutableRawPointer(res)
}

@_cdecl("vevc_enc_destroy")
public func vevc_enc_destroy(enc: UnsafeMutableRawPointer) {
    Unmanaged<CAPIEncoderContext>.fromOpaque(enc).release()
}
