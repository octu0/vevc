package main

import (
	"fmt"
	"runtime"
	"unsafe"
)

/*
#cgo CFLAGS: -I../../Sources/libvevc/include
#cgo LDFLAGS: -L../../.build/release -llibvevc
#include "encode.h"
#include "decode.h"
*/
import "C"

func main() {
	fmt.Println("Starting VEVC Go CGO verification...")

	param := C.vevc_enc_param_t{
		width:                  64,
		height:                 64,
		maxbitrate:             1000,
		framerate:              30,
		zero_threshold:         3,
		keyint:                 30,
		scene_change_threshold: 10,
		max_concurrency:        1,
	}

	enc := C.vevc_enc_create(&param)
	if enc == nil {
		fmt.Println("Failed to create encoder")
		return
	}
	defer C.vevc_enc_destroy(enc)
	fmt.Println("Encoder created successfully")

	ySize := 64 * 64
	uvSize := 32 * 32

	yBuf := make([]byte, ySize)
	uBuf := make([]byte, uvSize)
	vBuf := make([]byte, uvSize)

	for i := 0; i < ySize; i++ {
		yBuf[i] = 128
	}
	for i := 0; i < uvSize; i++ {
		uBuf[i] = 128
		vBuf[i] = 128
	}

	pinner := new(runtime.Pinner)
	pinner.Pin(&yBuf[0])
	pinner.Pin(&uBuf[0])
	pinner.Pin(&vBuf[0])
	defer pinner.Unpin()

	imgb := C.vevc_enc_imgb_t{
		y:        (*C.uint8_t)(unsafe.Pointer(&yBuf[0])),
		u:        (*C.uint8_t)(unsafe.Pointer(&uBuf[0])),
		v:        (*C.uint8_t)(unsafe.Pointer(&vBuf[0])),
		stride_y: 64,
		stride_u: 32,
		stride_v: 32,
	}

	encRes := C.vevc_enc_encode(enc, &imgb)
	if encRes.status != C.VEVC_OK {
		fmt.Printf("Encoding failed: %d\n", encRes.status)
		return
	}
	fmt.Printf("Encoded frame: size = %d bytes, iframe = %d, copyframe = %d\n", encRes.size, encRes.is_iframe, encRes.is_copyframe)

	dumpSize := 20
	if int(encRes.size) < dumpSize {
		dumpSize = int(encRes.size)
	}

	// Dump helper
	dataSlice := unsafe.Slice((*byte)(unsafe.Pointer(encRes.data)), encRes.size)
	fmt.Print("Encoded data hex dump: ")
	for i := 0; i < dumpSize; i++ {
		fmt.Printf("%02x ", dataSlice[i])
	}
	fmt.Println()

	// Initialize Decoder
	dec := C.vevc_dec_create(2, 1, 64, 64)
	if dec == nil {
		fmt.Println("Failed to create decoder")
		return
	}
	defer C.vevc_dec_destroy(dec)
	fmt.Println("Decoder created successfully")

	// Decode
	decRes := C.vevc_dec_decode(dec, (*C.uint8_t)(unsafe.Pointer(encRes.data)), C.size_t(encRes.size))
	if decRes.status != C.VEVC_OK {
		fmt.Printf("Decoding failed: %d\n", decRes.status)
		return
	}

	fmt.Printf("Decoded successfully: resolution = %dx%d, stride_y = %d\n", decRes.width, decRes.height, decRes.stride_y)

	// Flush encoder (to verify flush api as well)
	flushRes := C.vevc_enc_flush(enc)
	if flushRes.status != C.VEVC_OK {
		fmt.Printf("Flushing failed: %d\n", flushRes.status)
	}

	fmt.Println("VEVC Go CGO verification passed!")
}
