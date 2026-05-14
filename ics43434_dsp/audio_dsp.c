#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "audio_dsp.h"

#define SAMPLE_RATE 48000
#define DC_BLOCK_R 0.9998f
#define AGC_TARGET_LEVEL 0.25f
#define AGC_MAX_GAIN 40.0f

typedef struct {
    float dc_x_prev, dc_y_prev;
    float pre_gain, agc_gain, agc_env;
    float last_rms_in, last_rms_out, last_agc_gain, last_gate_gain;
} DspState;

DspContext *dsp_create(float pre_gain_db) {
    DspState *s = (DspState *)calloc(1, sizeof(DspState));
    s->pre_gain = powf(10.0f, pre_gain_db / 20.0f);
    s->agc_gain = 1.0f;
    return (DspContext *)s;
}

void dsp_set_pre_gain_db(DspContext *ctx, float db) {
    ((DspState *)ctx)->pre_gain = powf(10.0f, db / 20.0f);
}

int dsp_process(DspContext *ctx, float *buf, int n_frames) {
    DspState *s = (DspState *)ctx;
    float sum_sq_in = 0, sum_sq_out = 0;
    for (int i = 0; i < n_frames; i++) {
        float x = buf[i];
        sum_sq_in += x * x;
        float y = x - s->dc_x_prev + DC_BLOCK_R * s->dc_y_prev;
        s->dc_x_prev = x; s->dc_y_prev = y;
        x = y * s->pre_gain;
        float absx = fabsf(x);
        s->agc_env = 0.9998f * s->agc_env + 0.0002f * absx;
        if (s->agc_env > 1e-6f) {
            float target = AGC_TARGET_LEVEL / s->agc_env;
            if (target > AGC_MAX_GAIN) target = AGC_MAX_GAIN;
            s->agc_gain = 0.999f * s->agc_gain + 0.001f * target;
        }
        x *= s->agc_gain;
        buf[i] = (x > 1.0f) ? 1.0f : (x < -1.0f ? -1.0f : x);
        sum_sq_out += buf[i] * buf[i];
    }
    s->last_rms_in = sqrtf(sum_sq_in / n_frames);
    s->last_rms_out = sqrtf(sum_sq_out / n_frames);
    s->last_agc_gain = s->agc_gain;
    return 0;
}

// ĐÂY LÀ HÀM CÒN THIẾU KHIẾN ÔNG BỊ LỖI:
void dsp_get_stats(DspContext *ctx, DspStats *out) {
    DspState *s = (DspState *)ctx;
    out->rms_in_db = 20.0f * log10f(s->last_rms_in + 1e-12f);
    out->rms_out_db = 20.0f * log10f(s->last_rms_out + 1e-12f);
    out->agc_gain_db = 20.0f * log10f(s->last_agc_gain + 1e-12f);
    out->gate_gain = 1.0f;
}

void dsp_destroy(DspContext *ctx) { free(ctx); }
