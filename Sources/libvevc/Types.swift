import Foundation

public let VEVC_OK: Int32 = 0
public let VEVC_ERR: Int32 = -1

@frozen
public struct vevc_enc_param_t {
    public var width: Int32
    public var height: Int32
    public var maxbitrate: Int32
    public var framerate: Int32
    public var zero_threshold: Int32
    public var keyint: Int32
    public var scene_change_threshold: Int32
    public var max_concurrency: Int32
}

@frozen
public struct vevc_enc_imgb_t {
    public var y: UnsafeMutablePointer<UInt8>?
    public var u: UnsafeMutablePointer<UInt8>?
    public var v: UnsafeMutablePointer<UInt8>?
    public var stride_y: Int32
    public var stride_u: Int32
    public var stride_v: Int32
}

@frozen
public struct vevc_enc_result_t {
    public var data: UnsafeMutablePointer<UInt8>?
    public var size: Int
    public var is_iframe: Int32
    public var is_copyframe: Int32
    public var status: Int32
}

@frozen
public struct vevc_dec_result_t {
    public var y: UnsafeMutablePointer<UInt8>?
    public var u: UnsafeMutablePointer<UInt8>?
    public var v: UnsafeMutablePointer<UInt8>?
    public var width: Int32
    public var height: Int32
    public var stride_y: Int32
    public var stride_u: Int32
    public var stride_v: Int32
    public var status: Int32
}
