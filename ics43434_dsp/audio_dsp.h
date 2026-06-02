#ifndef AUDIO_DSP_H
#define AUDIO_DSP_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void DspContext;

typedef struct {
    float rms_in_db;
    float rms_out_db;
    float agc_gain_db;
    float gate_gain;
} DspStats;

DspContext *dsp_create(float pre_gain_db);
void        dsp_destroy(DspContext *ctx);
void        dsp_set_pre_gain_db(DspContext *ctx, float db);
void        dsp_set_gate_enabled(DspContext *ctx, int enabled);
void        dsp_set_agc_enabled(DspContext *ctx, int enabled);
int         dsp_process(DspContext *ctx, float *buf, int n_frames);
void        dsp_get_stats(DspContext *ctx, DspStats *stats);

#ifdef __cplusplus
}
#endif

#endif
