#ifndef LIBVEVC_ENCODE_H
#define LIBVEVC_ENCODE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VEVC_OK 0
#define VEVC_ERR -1

typedef void* VEVC_ENC;

typedef struct {
    int width;
    int height;
    int maxbitrate;
    int framerate;
    int zero_threshold;
    int keyint;
    int scene_change_threshold;
    int max_concurrency;
} vevc_enc_param_t;

typedef struct {
    uint8_t* y;
    uint8_t* u;
    uint8_t* v;
    int stride_y;
    int stride_u;
    int stride_v;
} vevc_enc_imgb_t;

typedef struct {
    uint8_t* data;
    size_t size;
    int is_iframe;
    int is_copyframe;
    int status;
} vevc_enc_result_t;

VEVC_ENC vevc_enc_create(const vevc_enc_param_t* param);

vevc_enc_result_t* vevc_enc_encode(VEVC_ENC enc, const vevc_enc_imgb_t* imgb);

vevc_enc_result_t* vevc_enc_flush(VEVC_ENC enc);

void vevc_enc_destroy(VEVC_ENC enc);

#ifdef __cplusplus
}
#endif
#endif
