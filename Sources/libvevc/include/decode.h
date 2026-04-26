#ifndef LIBVEVC_DECODE_H
#define LIBVEVC_DECODE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VEVC_OK 0
#define VEVC_ERR -1

typedef void* VEVC_DEC;

typedef struct {
    uint8_t* y;
    uint8_t* u;
    uint8_t* v;
    int width;
    int height;
    int stride_y;
    int stride_u;
    int stride_v;
    int status;
} vevc_dec_result_t;

VEVC_DEC vevc_dec_create(int max_layer, int max_concurrency, int width, int height);

vevc_dec_result_t* vevc_dec_decode(VEVC_DEC dec, const uint8_t* data, size_t size);

void vevc_dec_destroy(VEVC_DEC dec);

#ifdef __cplusplus
}
#endif
#endif
