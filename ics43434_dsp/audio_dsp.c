/*
 * audio_dsp.c  —  Real-time DSP pipeline for ICS-43434 on Raspberry Pi 4
 *                 Optimised for 1.5 m pickup range
 *
 * At 1.5 m the ICS-43434 incurs only ~3.5 dB of inverse-square-law
 * attenuation relative to its 1 m datasheet reference.  This lets us:
 *   • Use a lighter pre-gain (+24 dB vs +28 dB at 2.5 m)
 *   • Tighten AGC time constants — less pumping, more transparent
 *   • Lower the AGC max-gain ceiling — less amplified noise floor
 *   • Sharpen the noise gate — cleaner silences with better SNR
 *
 * Chain (in order):
 *   1. DC-offset removal   (1st-order IIR high-pass @ ~3 Hz)
 *   2. Fixed pre-gain      (configurable dB boost, default +24 dB)
 *   3. Adaptive Noise Gate (suppresses silence floors cleanly)
 *   4. AGC                 (keeps perceived loudness stable at 1.5 m)
 *   5. Soft Limiter        (prevents clipping at peak moments)
 *
 * Build as shared lib:
 *   gcc -O2 -march=armv8-a -fPIC -shared -o libaudio_dsp.so audio_dsp.c -lm
 *
 * The ICS-43434 outputs 24-bit I2S; ALSA typically presents it as
 * S32_LE (left-justified in the upper 24 bits).  Samples arrive here
 * as normalised float32 in [-1.0, 1.0].
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "audio_dsp.h"

/* ─── Constants ──────────────────────────────────────────────────────────── */

#define SAMPLE_RATE          48000
#define MAX_CHANNELS         1

/* DC-block coefficient: closer to 1.0 → lower corner frequency.
   At 48 kHz, 0.9998 gives ~3 Hz corner — removes slow thermal drift
   without touching any audible content above 20 Hz.               */
#define DC_BLOCK_R           0.9998f

/* AGC time constants — tightened for 1.5 m (better SNR, less compensation needed)
 *   ATTACK  3 ms   (was  5 ms) : reacts faster to sudden loud events
 *   RELEASE 100 ms (was 150 ms): less pumping; noise floor is ~4 dB cleaner
 *   MAX_GAIN 20x  (was  40x)  : lower ceiling prevents over-amplifying room noise;
 *                               at 1.5 m we only need ~3.5 dB distance compensation */
#define AGC_ATTACK_MS        3.0f
#define AGC_RELEASE_MS       100.0f
#define AGC_TARGET_LEVEL     0.25f   /* RMS target (about -12 dBFS)            */
#define AGC_MIN_GAIN         0.5f
#define AGC_MAX_GAIN         20.0f   /* sufficient headroom for 1.5 m          */

/* Noise gate — tighter at 1.5 m because SNR is better
 *   THRESHOLD 0.0010 (was 0.0015): ~-60 dBFS; stronger signal vs noise floor
 *   RELEASE   60 ms  (was  80 ms): snappier closure, less room noise bleed    */

#define GATE_THRESHOLD       0.0010f /* ~-60 dBFS                              */
#define GATE_ATTACK_MS       2.0f
#define GATE_RELEASE_MS      60.0f

/* Soft limiter knee (linear, 0-1) */
#define LIMITER_THRESHOLD    0.85f
#define LIMITER_CEILING      0.999f

/* ─── Internal state ─────────────────────────────────────────────────────── */

typedef struct {
    /* Stage enable flags (1 = active, 0 = bypassed).
     * Default: both on (full processing chain).
     *   gate off + agc off  -> transparent amplifier (DC block + pre-gain + limiter)
     *   gate on  + agc off  -> amp + clean silences (recommended for distance) */
    int gate_enabled;
    int agc_enabled;

    /* DC block */
    float dc_x_prev;
    float dc_y_prev;

    /* Pre-gain (linear) */
    float pre_gain;

    /* AGC */
    float agc_gain;
    float agc_env;          /* smoothed RMS envelope          */
    float agc_attack;       /* per-sample attack coeff        */
    float agc_release;      /* per-sample release coeff       */

    /* Noise gate */
    float gate_env;
    float gate_gain;        /* 0.0 (closed) … 1.0 (open)     */
    float gate_attack;
    float gate_release;

    /* Diagnostics (written each process() call) */
    float last_rms_in;
    float last_rms_out;
    float last_agc_gain;
    float last_gate_gain;
} DspState;

/* ─── Helpers ────────────────────────────────────────────────────────────── */

static inline float ms_to_coeff(float ms, float sr)
{
    /* 1-pole IIR smoothing coefficient from time constant in ms */
    return expf(-1000.0f / (ms * sr));
}

static inline float db_to_linear(float db)
{
    return powf(10.0f, db / 20.0f);
}

static inline float linear_to_db(float lin)

{
    return 20.0f * log10f(lin + 1e-12f);
}

/* Soft-knee limiter — transparent below threshold, cubic saturation above */
static inline float soft_limit(float x)
{
    float ax = fabsf(x);
    if (ax <= LIMITER_THRESHOLD)
        return x;

    /* Map [threshold … ∞) → [threshold … ceiling) via cubic */
    float t  = (ax - LIMITER_THRESHOLD) / (1.0f - LIMITER_THRESHOLD);
    float t2 = t * t;
    float t3 = t2 * t;
    /* Hermite blend — smooth S-curve, zero derivative at t=1 */
    float compressed = LIMITER_THRESHOLD
                     + (LIMITER_CEILING - LIMITER_THRESHOLD)
                       * (3.0f * t2 - 2.0f * t3);
    return (x >= 0.0f) ? compressed : -compressed;
}

/* ─── Public API ─────────────────────────────────────────────────────────── */

DspContext *dsp_create(float pre_gain_db)
{
    DspState *s = (DspState *)calloc(1, sizeof(DspState));
    if (!s) return NULL;

    float sr = (float)SAMPLE_RATE;

    /* Full chain on by default */
    s->gate_enabled  = 1;
    s->agc_enabled   = 1;

    s->pre_gain      = db_to_linear(pre_gain_db);
    s->agc_gain      = 1.0f;
    s->agc_env       = 0.0f;
    s->agc_attack    = ms_to_coeff(AGC_ATTACK_MS,   sr);
    s->agc_release   = ms_to_coeff(AGC_RELEASE_MS,  sr);

    s->gate_env      = 0.0f;
    s->gate_gain     = 0.0f;
    s->gate_attack   = ms_to_coeff(GATE_ATTACK_MS,  sr);
    s->gate_release  = ms_to_coeff(GATE_RELEASE_MS, sr);

    return (DspContext *)s;
}

void dsp_destroy(DspContext *ctx)
{
    free(ctx);
}

void dsp_set_pre_gain_db(DspContext *ctx, float db)
{
    DspState *s = (DspState *)ctx;
    s->pre_gain = db_to_linear(db);
}

void dsp_set_gate_enabled(DspContext *ctx, int enabled)
{
    DspState *s = (DspState *)ctx;
    s->gate_enabled = enabled ? 1 : 0;
}

void dsp_set_agc_enabled(DspContext *ctx, int enabled)
{
    DspState *s = (DspState *)ctx;
    s->agc_enabled = enabled ? 1 : 0;
}

/*
 * dsp_process()
 *
 * in/out   : interleaved float32 samples, mono, normalised [-1, 1]
 * n_frames : number of audio frames (= samples for mono)
 *
 * Processes in-place.  Returns 0 on success.
 */
int dsp_process(DspContext *ctx, float *buf, int n_frames)
{
    if (!ctx || !buf || n_frames <= 0) return -1;
    DspState *s = (DspState *)ctx;

    float sum_sq_in  = 0.0f;
    float sum_sq_out = 0.0f;

    for (int i = 0; i < n_frames; i++) {
        float x = buf[i];

        /* ── 1. Accumulate input RMS ──────────────────────────────────── */
        sum_sq_in += x * x;

        /* ── 2. DC block (1st-order IIR high-pass) ───────────────────── */
        float y = x - s->dc_x_prev + DC_BLOCK_R * s->dc_y_prev;
        s->dc_x_prev = x;
        s->dc_y_prev = y;

        x = y;

        /* ── 3. Fixed pre-gain ────────────────────────────────────────── */
        x *= s->pre_gain;

        /* ── 4. Noise gate ────────────────────────────────────────────── */
        if (s->gate_enabled) {
        float ax = fabsf(x);
        /* Envelope follower on the pre-gain signal */
        if (ax > s->gate_env)
            s->gate_env = s->gate_attack  * s->gate_env + (1.0f - s->gate_attack)  * ax;
        else
            s->gate_env = s->gate_release * s->gate_env + (1.0f - s->gate_release) * ax;

        /* Compute target gate gain */
        float target_gate = (s->gate_env >= GATE_THRESHOLD) ? 1.0f : 0.0f;

        /* Smooth the gate open/close to avoid clicks */
        float gate_coeff = (target_gate > s->gate_gain)
                         ? s->gate_attack : s->gate_release;
        s->gate_gain = gate_coeff * s->gate_gain
                     + (1.0f - gate_coeff) * target_gate;
        x *= s->gate_gain;
        }  /* end gate */

        /* ── 5. AGC ───────────────────────────────────────────────────── */
        if (s->agc_enabled) {
        float absx = fabsf(x);
        /* Peak-following envelope (asymmetric) */
        if (absx > s->agc_env)
            s->agc_env = s->agc_attack  * s->agc_env + (1.0f - s->agc_attack)  * absx;
        else
            s->agc_env = s->agc_release * s->agc_env + (1.0f - s->agc_release) * absx;

        /* Drive gain toward target */
        if (s->agc_env > 1e-6f) {
            float desired_gain = AGC_TARGET_LEVEL / s->agc_env;
            /* Clamp */
            if (desired_gain < AGC_MIN_GAIN) desired_gain = AGC_MIN_GAIN;
            if (desired_gain > AGC_MAX_GAIN) desired_gain = AGC_MAX_GAIN;

            /* Smooth gain changes (same attack/release) */
            float gain_coeff = (desired_gain < s->agc_gain)
                             ? s->agc_attack : s->agc_release;
            s->agc_gain = gain_coeff * s->agc_gain
                        + (1.0f - gain_coeff) * desired_gain;
        }
        x *= s->agc_gain;
        }  /* end agc */

        /* ── 6. Soft limiter ──────────────────────────────────────────── */
        x = soft_limit(x);

        buf[i] = x;
        sum_sq_out += x * x;
    }

    /* ── Update diagnostics ─────────────────────────────────────────── */
    float inv_n = 1.0f / (float)n_frames;
    s->last_rms_in   = sqrtf(sum_sq_in  * inv_n);
    s->last_rms_out  = sqrtf(sum_sq_out * inv_n);
    s->last_agc_gain = s->agc_gain;
    s->last_gate_gain= s->gate_gain;

    return 0;
}

void dsp_get_stats(DspContext *ctx, DspStats *out)
{
    if (!ctx || !out) return;
    DspState *s = (DspState *)ctx;
    out->rms_in_db    = linear_to_db(s->last_rms_in);
    out->rms_out_db   = linear_to_db(s->last_rms_out);
    out->agc_gain_db  = linear_to_db(s->last_agc_gain);
    out->gate_gain    = s->last_gate_gain;
}
