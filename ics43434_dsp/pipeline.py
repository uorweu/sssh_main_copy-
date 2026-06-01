#!/usr/bin/env python3
"""
pipeline.py  —  ICS-43434 → DSP → AI sound-recognition pipeline
                Raspberry Pi 4 | Python 3.9+

Architecture
────────────
  ALSA capture thread  ──►  DSP (C lib via ctypes)  ──►  AI inference queue
                                                     ──►  optional WAV logging

The ICS-43434 is a 24-bit I2S microphone.  The Pi's I2S driver presents
it as S32_LE at 48 kHz; the upper 24 bits carry valid audio data.

Install deps (once):
    sudo apt install libasound2-dev libportaudio2
    pip install sounddevice numpy scipy

Build the C library first:
    cd ics43434_dsp
    gcc -O2 -march=armv8-a -fPIC -shared -o libaudio_dsp.so audio_dsp.c -lm
"""

import ctypes
import logging
import os
import queue
import struct
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    from scipy import signal as sig
except ImportError:
    raise SystemExit("Install scipy:  pip install scipy")

try:
    import sounddevice as sd
except ImportError:
    raise SystemExit("Install sounddevice:  pip install sounddevice")

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ics43434")

# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Hardware
    device_name: str  = "default"        # ALSA device; or index from sd.query_devices()
    sample_rate: int  = 48_000           # ICS-43434 native rate
    channels: int     = 1
    blocksize: int    = 2048             # frames per callback (~43 ms @ 48kHz)

    # DSP
    # Pre-gain: 24 dB for 1.5 m (only ~3.5 dB distance loss to compensate).
    # Raise to 28 dB if the source is consistently quieter than speech level.
    pre_gain_db: float = 24.0
    lib_path: str      = "./libaudio_dsp.so"

    # Sound-level metering (the project's headline output: "how loud is the room?")
    # ICS-43434 datasheet: -26 dBFS at 94 dB SPL @ 1 kHz.
    # dB_SPL = dBFS + (ref_spl - sensitivity) = dBFS + 120.
    # Trim spl_calibration_offset with a reference meter if you have one.
    mic_sensitivity_dbfs: float = -26.0
    spl_ref_db:           float = 94.0
    spl_calibration_offset: float = 0.0
    spl_a_weighting:      bool  = True   # A-weight to match human hearing (recommended)

    # AI inference chunking
    # Feeds the model a 1-second window, stepped every 0.5 s (50% overlap)
    inference_window_s: float  = 1.0
    inference_step_s:   float  = 0.5

    # Optional WAV logging (set to None to disable)
    log_wav_path: Optional[str] = None   # e.g. "/tmp/capture.wav"

    # Monitoring — print stats every N seconds (0 = off)
    stats_interval_s: float = 5.0

# ─── C library bindings ───────────────────────────────────────────────────────

class _DspStats(ctypes.Structure):
    _fields_ = [
        ("rms_in_db",   ctypes.c_float),
        ("rms_out_db",  ctypes.c_float),
        ("agc_gain_db", ctypes.c_float),
        ("gate_gain",   ctypes.c_float),
    ]


class DspLibrary:
    """Thin ctypes wrapper around libaudio_dsp.so"""

    def __init__(self, lib_path: str, pre_gain_db: float):
        if not Path(lib_path).exists():
            raise FileNotFoundError(
                f"DSP library not found: {lib_path}\n"
                "Build with:  gcc -O2 -march=armv8-a -fPIC -shared "
                "-o libaudio_dsp.so audio_dsp.c -lm"
            )
        self._lib = ctypes.CDLL(lib_path)

        # Prototype declarations
        self._lib.dsp_create.restype  = ctypes.c_void_p
        self._lib.dsp_create.argtypes = [ctypes.c_float]

        self._lib.dsp_destroy.restype  = None
        self._lib.dsp_destroy.argtypes = [ctypes.c_void_p]

        self._lib.dsp_set_pre_gain_db.restype  = None
        self._lib.dsp_set_pre_gain_db.argtypes = [ctypes.c_void_p, ctypes.c_float]

        self._lib.dsp_process.restype  = ctypes.c_int
        self._lib.dsp_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]

        self._lib.dsp_get_stats.restype  = None
        self._lib.dsp_get_stats.argtypes = [ctypes.c_void_p,
                                             ctypes.POINTER(_DspStats)]

        self._ctx = self._lib.dsp_create(ctypes.c_float(pre_gain_db))
        if not self._ctx:
            raise RuntimeError("dsp_create() returned NULL — out of memory?")

        log.info("DSP library loaded, pre-gain=%.1f dB", pre_gain_db)

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Process float32 mono ndarray in-place and return it.
        The array must be C-contiguous.
        """
        assert samples.dtype == np.float32
        samples = np.ascontiguousarray(samples)
        ptr = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ret = self._lib.dsp_process(self._ctx, ptr, ctypes.c_int(len(samples)))
        if ret != 0:
            log.warning("dsp_process() returned %d", ret)
        return samples

    def set_pre_gain_db(self, db: float):
        self._lib.dsp_set_pre_gain_db(self._ctx, ctypes.c_float(db))

    def get_stats(self) -> _DspStats:
        stats = _DspStats()
        self._lib.dsp_get_stats(self._ctx, ctypes.byref(stats))
        return stats

    def __del__(self):
        if hasattr(self, "_ctx") and self._ctx:
            self._lib.dsp_destroy(self._ctx)


# ─── Sound-level meter ──────────────────────────────────────────────────────────

class SplMeter:
    """
    Measures calibrated, A-weighted sound pressure level (dB SPL) from the
    RAW microphone signal — i.e. BEFORE pre-gain/gate/AGC. This matters: the
    DSP chain deliberately distorts levels (AGC normalises loudness), so SPL
    must be read from the untouched input or the "how loud is the room?"
    number would be meaningless.

    Calibration
    ───────────
    The ICS-43434 outputs -26 dBFS at 94 dB SPL @ 1 kHz. sounddevice presents
    the 24-bit (left-justified S32) samples as float32 in [-1, 1] where
    full-scale = 0 dBFS. So:
        dB_SPL = dBFS + (spl_ref_db - sensitivity_dbfs) + calibration_offset
               = dBFS + 120 (+ trim)
    If you own a reference sound-level meter, play a steady tone, compare, and
    put the difference in `spl_calibration_offset`.
    """

    def __init__(self, sample_rate, sensitivity_dbfs=-26.0, ref_spl=94.0,
                 calibration_offset=0.0, a_weighting=True):
        self._offset = ref_spl - sensitivity_dbfs + calibration_offset
        self._a_weighting = a_weighting
        if a_weighting:
            self._sos = self._design_a_weighting(sample_rate)
            self._zi = sig.sosfilt_zi(self._sos)

    @staticmethod
    def _design_a_weighting(sr):
        """IEC 61672 A-weighting as a digital biquad cascade."""
        f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
        A1000 = 1.9997
        nums = [(2 * np.pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
        dens = np.convolve(
            [1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
            [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2],
        )
        dens = np.convolve(np.convolve(dens, [1, 2 * np.pi * f3]),
                           [1, 2 * np.pi * f2])
        b, a = sig.bilinear(nums, dens, sr)
        return sig.tf2sos(b, a)

    def measure(self, raw_block: np.ndarray) -> float:
        """raw_block: UNPROCESSED float32 mono. Returns dB SPL (A-weighted if enabled)."""
        if self._a_weighting:
            weighted, self._zi = sig.sosfilt(self._sos, raw_block, zi=self._zi)
            rms = np.sqrt(np.mean(weighted * weighted) + 1e-20)
        else:
            rms = np.sqrt(np.mean(raw_block * raw_block) + 1e-20)
        if rms < 1e-10:
            return 0.0
        return float(20.0 * np.log10(rms) + self._offset)


# ─── Inference buffer ─────────────────────────────────────────────────────────

class OverlapBuffer:
    """
    Accumulates processed audio and emits overlapping windows for inference.

    window_samples : model sees this many frames each call
    step_samples   : advance by this many frames between calls (overlap = window-step)
    """

    def __init__(self, window_samples: int, step_samples: int):
        self._window = window_samples
        self._step   = step_samples
        self._buf    = np.zeros(window_samples, dtype=np.float32)
        self._filled = 0

    def push(self, chunk: np.ndarray):
        """
        Feed audio; yields complete windows whenever enough data accumulates.
        """
        pos = 0
        while pos < len(chunk):
            space = self._window - self._filled
            take  = min(space, len(chunk) - pos)
            self._buf[self._filled : self._filled + take] = chunk[pos : pos + take]
            self._filled += take
            pos          += take

            if self._filled == self._window:
                yield self._buf.copy()
                # Slide: discard 'step' samples, keep the overlap tail
                keep = self._window - self._step
                self._buf[:keep] = self._buf[self._step:]
                self._filled     = keep


# ─── WAV writer ───────────────────────────────────────────────────────────────

class WavLogger:
    def __init__(self, path: str, sample_rate: int):
        self._wf = wave.open(path, "wb")
        self._wf.setnchannels(1)
        self._wf.setsampwidth(2)          # 16-bit PCM
        self._wf.setframerate(sample_rate)
        log.info("WAV logging → %s", path)

    def write(self, samples: np.ndarray):
        pcm = (samples * 32767.0).clip(-32768, 32767).astype(np.int16)
        self._wf.writeframes(pcm.tobytes())

    def close(self):
        self._wf.close()


# ─── Main pipeline ────────────────────────────────────────────────────────────

class AudioPipeline:
    """
    Ties together:
      • sounddevice capture (runs in its own ALSA thread)
      • DSP C library (called from the capture callback — low latency)
      • Inference queue (AI model runs in a dedicated thread)
      • Optional WAV logging

    Usage
    ─────
        def my_model(window: np.ndarray):
            # window is float32 mono, 48 kHz, ~1 second
            # run your classifier here
            ...

        pipeline = AudioPipeline(cfg, inference_callback=my_model)
        pipeline.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pipeline.stop()
    """

    def __init__(self, cfg: Config, inference_callback: Callable[[np.ndarray], None]):
        self._cfg      = cfg
        self._callback = inference_callback
        self._dsp      = DspLibrary(cfg.lib_path, cfg.pre_gain_db)
        self._spl_meter = SplMeter(
            sample_rate        = cfg.sample_rate,
            sensitivity_dbfs   = cfg.mic_sensitivity_dbfs,
            ref_spl            = cfg.spl_ref_db,
            calibration_offset = cfg.spl_calibration_offset,
            a_weighting        = cfg.spl_a_weighting,
        )
        self._infer_q: queue.Queue = queue.Queue(maxsize=16)
        self._stop_evt = threading.Event()

        window_samp = int(cfg.sample_rate * cfg.inference_window_s)
        step_samp   = int(cfg.sample_rate * cfg.inference_step_s)
        self._overlap_buf = OverlapBuffer(window_samp, step_samp)

        self._wav_logger: Optional[WavLogger] = None
        if cfg.log_wav_path:
            self._wav_logger = WavLogger(cfg.log_wav_path, cfg.sample_rate)

        self._stats_lock  = threading.Lock()
        self._last_stats  = None
        self._last_spl_dba = 0.0
        self._frame_count = 0

    # ── ALSA callback (called from sounddevice's internal thread) ─────────────

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            log.warning("ALSA status: %s", status)

        # indata shape: (frames, channels) — float32 in [-1, 1]
        mono = indata[:, 0].copy()      # extract mono channel

        # Measure room SPL from the RAW signal, BEFORE the DSP touches levels.
        # (The AGC normalises loudness, so SPL read post-DSP would be a lie.)
        spl_dba = self._spl_meter.measure(mono)

        # Run DSP in-place (this is the audio the model will see)
        mono = self._dsp.process(mono)

        # Log stats snapshot
        with self._stats_lock:
            self._last_stats  = self._dsp.get_stats()
            self._last_spl_dba = spl_dba
            self._frame_count += frames

        # Feed WAV logger
        if self._wav_logger:
            self._wav_logger.write(mono)

        # Emit inference windows (non-blocking put — drop if queue full)
        for window in self._overlap_buf.push(mono):
            try:
                self._infer_q.put_nowait(window)
            except queue.Full:
                log.debug("Inference queue full — dropped window")

    # ── Inference thread ──────────────────────────────────────────────────────

    def _inference_worker(self):
        log.info("Inference thread started")
        while not self._stop_evt.is_set():
            try:
                window = self._infer_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._callback(window)
            except Exception:
                log.exception("Error in inference callback")
        log.info("Inference thread stopped")

    # ── Stats reporter ────────────────────────────────────────────────────────

    def _stats_reporter(self):
        while not self._stop_evt.is_set():
            time.sleep(self._cfg.stats_interval_s)
            with self._stats_lock:
                s = self._last_stats
                spl = self._last_spl_dba
                fc = self._frame_count
            if s and self._cfg.stats_interval_s > 0:
                weight = "dBA" if self._cfg.spl_a_weighting else "dBZ"
                log.info(
                    "ROOM LEVEL: %.1f %s  |  DSP in=%.1f dBFS out=%.1f dBFS  "
                    "AGC=%.1f dB  gate=%.2f  frames=%d",
                    spl, weight,
                    s.rms_in_db, s.rms_out_db, s.agc_gain_db, s.gate_gain, fc,
                )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        log.info(
            "Opening capture device '%s' @ %d Hz, block=%d",
            self._cfg.device_name,
            self._cfg.sample_rate,
            self._cfg.blocksize,
        )
        self._stream = sd.InputStream(
            device        = self._cfg.device_name,
            samplerate    = self._cfg.sample_rate,
            channels      = self._cfg.channels,
            blocksize     = self._cfg.blocksize,
            dtype         = "float32",
            callback      = self._audio_callback,
            latency       = "low",
        )
        self._stream.start()

        self._infer_thread = threading.Thread(
            target=self._inference_worker, daemon=True, name="InferWorker"
        )
        self._infer_thread.start()

        if self._cfg.stats_interval_s > 0:
            self._stats_thread = threading.Thread(
                target=self._stats_reporter, daemon=True, name="StatsReporter"
            )
            self._stats_thread.start()

        log.info("Pipeline running.  Ctrl+C to stop.")

    def stop(self):
        log.info("Stopping pipeline …")
        self._stop_evt.set()
        self._stream.stop()
        self._stream.close()
        self._infer_thread.join(timeout=2.0)
        if self._wav_logger:
            self._wav_logger.close()
        log.info("Pipeline stopped.")

    def set_pre_gain_db(self, db: float):
        """Hot-swap the pre-gain while running."""
        self._dsp.set_pre_gain_db(db)
        log.info("Pre-gain updated → %.1f dB", db)

    def get_spl_dba(self) -> float:
        """
        Current room sound level in dB SPL (A-weighted if enabled).
        This is the value the database/web modules will publish later.
        Thread-safe.
        """
        with self._stats_lock:
            return self._last_spl_dba


# ─── Example AI model stub ────────────────────────────────────────────────────

def example_ai_model(window: np.ndarray):
    """
    Replace this function with your actual model.

    window : float32 ndarray, shape (sample_rate * window_s,)
             Normalised audio, ready for your classifier.

    Typical integration patterns:
        • yamnet / tflite: convert to mel-spectrogram first
        • whisper:         window is already suitable for 16 kHz after resample
        • custom CNN:      standardise (window - mean) / std per window
    """
    rms = float(np.sqrt(np.mean(window ** 2)))
    peak = float(np.max(np.abs(window)))
    # --- Replace below with real inference ---
    log.debug("AI window | RMS=%.4f  peak=%.4f  samples=%d", rms, peak, len(window))


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ICS-43434 DSP + AI pipeline")
    parser.add_argument("--device",    default="default",   help="ALSA device name or index")
    parser.add_argument("--gain",      type=float, default=24.0, help="Pre-gain in dB (default 24, optimised for 1.5 m)")
    parser.add_argument("--log-wav",   default=None,        help="Path to save processed WAV")
    parser.add_argument("--list-devs", action="store_true", help="List audio devices and exit")
    args = parser.parse_args()

    if args.list_devs:
        print(sd.query_devices())
        raise SystemExit(0)

    cfg = Config(
        device_name    = args.device,
        pre_gain_db    = args.gain,
        log_wav_path   = args.log_wav,
        stats_interval_s = 5.0,
    )

    pipeline = AudioPipeline(cfg, inference_callback=example_ai_model)
    pipeline.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print()
    finally:
        pipeline.stop()
