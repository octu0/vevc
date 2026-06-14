#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "encode.h"
#include "decode.h"

int main() {
    printf("Starting VEVC C API verification...\n");

    vevc_enc_param_t param = {
        .width = 64,
        .height = 64,
        .maxbitrate = 1000,
        .framerate = 30,
        .zero_threshold = 3,
        .keyint = 30,
        .scene_change_threshold = 10,
        .max_concurrency = 1
    };

    VEVC_ENC enc = vevc_enc_create(&param);
    if (!enc) {
        fprintf(stderr, "Failed to create encoder\n");
        return 1;
    }
    printf("Encoder created successfully\n");

    // YUV420 buffers
    size_t y_size = 64 * 64;
    size_t uv_size = 32 * 32;
    uint8_t* y_buffer = malloc(y_size);
    uint8_t* u_buffer = malloc(uv_size);
    uint8_t* v_buffer = malloc(uv_size);

    if (!y_buffer || !u_buffer || !v_buffer) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }

    // Fill YUV with dummy pattern (flat gray)
    memset(y_buffer, 128, y_size);
    memset(u_buffer, 128, uv_size);
    memset(v_buffer, 128, uv_size);

    vevc_enc_imgb_t imgb = {
        .y = y_buffer,
        .u = u_buffer,
        .v = v_buffer,
        .stride_y = 64,
        .stride_u = 32,
        .stride_v = 32
    };

    vevc_enc_result_t* enc_res = vevc_enc_encode(enc, &imgb);
    if (enc_res->status != VEVC_OK) {
        fprintf(stderr, "Encoding failed: %d\n", enc_res->status);
        free(y_buffer); free(u_buffer); free(v_buffer);
        vevc_enc_destroy(enc);
        return 1;
    }
    printf("Encoded frame: size = %zu bytes, iframe = %d, copyframe = %d\n", enc_res->size, enc_res->is_iframe, enc_res->is_copyframe);

    // Hex dump first 20 bytes
    printf("Encoded data hex dump: ");
    for (size_t idx = 0; idx < (enc_res->size < 20 ? enc_res->size : 20); idx++) {
        printf("%02x ", enc_res->data[idx]);
    }
    printf("\n");

    // Initialize Decoder
    VEVC_DEC dec = vevc_dec_create(2, 1, 64, 64);
    if (!dec) {
        fprintf(stderr, "Failed to create decoder\n");
        free(y_buffer); free(u_buffer); free(v_buffer);
        vevc_enc_destroy(enc);
        return 1;
    }
    printf("Decoder created successfully\n");

    // Decode encoded data
    vevc_dec_result_t* dec_res = vevc_dec_decode(dec, enc_res->data, enc_res->size);
    if (dec_res->status != VEVC_OK) {
        fprintf(stderr, "Decoding failed: %d\n", dec_res->status);
        free(y_buffer); free(u_buffer); free(v_buffer);
        vevc_enc_destroy(enc);
        vevc_dec_destroy(dec);
        return 1;
    }

    printf("Decoded successfully: resolution = %dx%d, stride_y = %d\n", dec_res->width, dec_res->height, dec_res->stride_y);

    // Flush encoder
    vevc_enc_result_t* flush_res = vevc_enc_flush(enc);
    if (flush_res->status != VEVC_OK) {
        fprintf(stderr, "Flushing failed: %d\n", flush_res->status);
    }

    // Clean up
    free(y_buffer); free(u_buffer); free(v_buffer);
    vevc_enc_destroy(enc);
    vevc_dec_destroy(dec);
    printf("VEVC C API verification passed!\n");
    return 0;
}
