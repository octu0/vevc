import Foundation
import vevc

final class CAPIDecoderContext: @unchecked Sendable {
    let actor: StreamingDecoderActor
    let width: Int
    let height: Int
    
    let resultPtr: UnsafeMutablePointer<vevc_dec_result_t>
    var yPtr: UnsafeMutablePointer<UInt8>?
    var uPtr: UnsafeMutablePointer<UInt8>?
    var vPtr: UnsafeMutablePointer<UInt8>?
    var yCapacity: Int = 0
    var cCapacity: Int = 0
    
    init(maxLayer: Int, width: Int, height: Int) {
        self.actor = StreamingDecoderActor(maxLayer: maxLayer, width: width, height: height)
        self.width = width
        self.height = height
        
        self.resultPtr = UnsafeMutablePointer<vevc_dec_result_t>.allocate(capacity: 1)
        self.resultPtr.pointee.status = VEVC_OK
        self.resultPtr.pointee.y = nil
        self.resultPtr.pointee.u = nil
        self.resultPtr.pointee.v = nil
        self.resultPtr.pointee.width = 0
        self.resultPtr.pointee.height = 0
        self.resultPtr.pointee.stride_y = 0
        self.resultPtr.pointee.stride_u = 0
        self.resultPtr.pointee.stride_v = 0
    }
    
    deinit {
        if let y = yPtr { y.deallocate() }
        if let u = uPtr { u.deallocate() }
        if let v = vPtr { v.deallocate() }
        resultPtr.deallocate()
    }
    
    func decode(chunk: [UInt8]) throws -> YCbCrImage? {
        let semaphore = DispatchSemaphore(value: 0)
        final class Box: @unchecked Sendable { 
            var result: YCbCrImage?
            var error: Error?
        }
        let box = Box()
        let targetActor = self.actor
        
        Task {
            do {
                box.result = try await targetActor.decodeNextFrame(chunk: chunk)
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

@_cdecl("vevc_dec_create")
public func vevc_dec_create(max_layer: Int32, max_concurrency: Int32, width: Int32, height: Int32) -> UnsafeMutableRawPointer? {
    let ctx = CAPIDecoderContext(
        maxLayer: Int(max_layer),
        width: Int(width),
        height: Int(height)
    )
    return Unmanaged.passRetained(ctx).toOpaque()
}

@_cdecl("vevc_dec_decode")
public func vevc_dec_decode(dec: UnsafeMutableRawPointer, data: UnsafePointer<UInt8>, size: Int) -> UnsafeMutableRawPointer? {
    let ctx = Unmanaged<CAPIDecoderContext>.fromOpaque(dec).takeUnretainedValue()
    let chunk = Array(UnsafeBufferPointer(start: data, count: size))
    
    let res = ctx.resultPtr
    res.pointee.status = VEVC_OK
    res.pointee.y = nil
    res.pointee.u = nil
    res.pointee.v = nil
    res.pointee.width = 0
    res.pointee.height = 0
    res.pointee.stride_y = 0
    res.pointee.stride_u = 0
    res.pointee.stride_v = 0
    
    var decodeChunk = chunk
    var offset = 0
    if 4 <= chunk.count && chunk[0] == 0x56 && chunk[1] == 0x45 && chunk[2] == 0x56 && chunk[3] == 0x43 {
        if 6 <= chunk.count {
            let metaSize = Int(chunk[4]) << 8 | Int(chunk[5])
            offset = 6 + metaSize
            if offset < chunk.count {
                decodeChunk = Array(chunk[offset..<chunk.count])
            } else {
                decodeChunk = []
            }
        }
    }
    
    guard decodeChunk.isEmpty != true else {
        return UnsafeMutableRawPointer(res)
    }
    
    do {
        if let img = try ctx.decode(chunk: decodeChunk) {
            res.pointee.width = Int32(img.width)
            res.pointee.height = Int32(img.height)
            res.pointee.stride_y = Int32(img.width)
            res.pointee.stride_u = Int32(img.width / 2)
            res.pointee.stride_v = Int32(img.width / 2)
            
            let ySize = img.width * img.height
            let cSize = (img.width / 2) * (img.height / 2)
            
            if ctx.yCapacity < ySize {
                if let old = ctx.yPtr { old.deallocate() }
                let newCap = max(ySize, ctx.yCapacity * 2)
                ctx.yCapacity = newCap
                ctx.yPtr = UnsafeMutablePointer<UInt8>.allocate(capacity: newCap)
            }
            if ctx.cCapacity < cSize {
                if let oldU = ctx.uPtr { oldU.deallocate() }
                if let oldV = ctx.vPtr { oldV.deallocate() }
                let newCap = max(cSize, ctx.cCapacity * 2)
                ctx.cCapacity = newCap
                ctx.uPtr = UnsafeMutablePointer<UInt8>.allocate(capacity: newCap)
                ctx.vPtr = UnsafeMutablePointer<UInt8>.allocate(capacity: newCap)
            }
            
            let yBuf = ctx.yPtr!
            let uBuf = ctx.uPtr!
            let vBuf = ctx.vPtr!
            
            img.yPlane.withUnsafeBufferPointer { ptr in
                if let base = ptr.baseAddress { yBuf.update(from: base, count: ySize) }
            }
            img.cbPlane.withUnsafeBufferPointer { ptr in
                if let base = ptr.baseAddress { uBuf.update(from: base, count: cSize) }
            }
            img.crPlane.withUnsafeBufferPointer { ptr in
                if let base = ptr.baseAddress { vBuf.update(from: base, count: cSize) }
            }
            
            res.pointee.y = yBuf
            res.pointee.u = uBuf
            res.pointee.v = vBuf
        } else {
            res.pointee.status = VEVC_ERR
        }
    } catch {
        res.pointee.status = VEVC_ERR
    }
    
    return UnsafeMutableRawPointer(res)
}

@_cdecl("vevc_dec_destroy")
public func vevc_dec_destroy(dec: UnsafeMutableRawPointer) {
    Unmanaged<CAPIDecoderContext>.fromOpaque(dec).release()
}
