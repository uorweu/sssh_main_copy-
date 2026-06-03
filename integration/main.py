#!/usr/bin/env python3
"""
main.py  —  SSSH Phase 4: System Integration

Wires together every prior phase into one running edge service:

    ICS-43434 (I2S)
        │  48 kHz S32_LE
        ▼
    pipeline.py  ──►  audio_dsp (C lib)        [Phase 3: range extension]
        │  processed 48 kHz mono windows (1 s, 50% overlap)
        ▼
    yamnet_classifier.py  ──►  fine-tuned YAMNet  [Phase 1: classification]
        │  top class + score
        ▼
    noise_analyzer.py  ──►  dB + duration + class  [Phase 4: "what is noise"]
        │  noise decision + rolling density + suitability
        ▼
    influx_writer.py  ──►  InfluxDB on laptop  [Phase 5: feeds Grafana]

Run:
    python3 main.py --config config.yaml
    python3 main.py --list-devs
    python3 main.py --dry-run            # no InfluxDB, just print decisions
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np

# Phase 3 pipeline (your DSP capture engine)
from pipeline import AudioPipeline, Config as PipelineConfig, sd

# Phase 4 components
from yamnet_classifier import YamnetClassifier
from noise_analyzer import NoiseAnalyzer, NoiseConfig
from influx_writer import InfluxWriter, InfluxConfig

log = logging.getLogger("sssh")


# ─── Config loading ────────────────────────────────────────────────────────────

DEFAULTS = {
    "audio": {
        "device": "default",
        "pre_gain_db": 24.0,
        "gate_enabled": True,
        "agc_enabled": True,
        "inference_window_s": 1.0,
        "inference_step_s": 0.5,
        "lib_path": "./libaudio_dsp.so",
        "log_wav_path": None,
    },
    "model": {
        "model_path": "~/sssh_main/yamnet",
        "labels_path": None,
        "top_k": 3,
    },
    "noise": {
        "min_confidence": 0.30,
        "spl_offset_db": 0.0,
        "spl_threshold_db": -35.0,
        "min_event_duration_s": 0.7,
        "density_window_s": 30.0,
    },
    "influx": {
        "url": "http://192.168.1.50:8086",
        "token": "CHANGE_ME",
        "org": "sssh",
        "bucket": "noise",
        "device": "pi4-01",
        "location": "library-floor2",
        "flush_interval_s": 2.0,
    },
}


def load_config(path: str | None) -> dict:
    cfg = {k: dict(v) for k, v in DEFAULTS.items()}
    if not path:
        return cfg
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        for section, values in user.items():
            if section in cfg and isinstance(values, dict):
                cfg[section].update(values)
            else:
                cfg[section] = values
        log.info("Loaded config from %s", path)
    except FileNotFoundError:
        log.warning("Config %s not found — using defaults", path)
    except ImportError:
        log.warning("pyyaml not installed — using defaults (pip install pyyaml)")
    return cfg


# ─── The integrated service ─────────────────────────────────────────────────────

class SsshService:
    def __init__(self, cfg: dict, dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run

        # 1. Classifier (Phase 1 model) ---------------------------------------
        m = cfg["model"]
        log.info("Loading YAMNet model: %s", m["model_path"])
        self.classifier = YamnetClassifier(
            model_path=m["model_path"],
            labels_path=m.get("labels_path"),
            top_k=int(m.get("top_k", 3)),
        )

        # 2. Noise analyzer (Phase 4 logic) -----------------------------------
        n = cfg["noise"]
        self.analyzer = NoiseAnalyzer(NoiseConfig(
            min_confidence=float(n["min_confidence"]),
            spl_offset_db=float(n["spl_offset_db"]),
            spl_threshold_db=float(n["spl_threshold_db"]),
            min_event_duration_s=float(n["min_event_duration_s"]),
            density_window_s=float(n["density_window_s"]),
            step_s=float(cfg["audio"]["inference_step_s"]),
        ))

        # 3. InfluxDB writer (Phase 5 sink) -----------------------------------
        self.influx = None
        if not dry_run:
            i = cfg["influx"]
            self.influx = InfluxWriter(InfluxConfig(
                url=i["url"], token=i["token"], org=i["org"],
                bucket=i["bucket"], device=i["device"],
                location=i["location"],
                flush_interval_s=float(i.get("flush_interval_s", 2.0)),
            ))

        # 4. DSP capture pipeline (Phase 3) -----------------------------------
        a = cfg["audio"]
        self.pipe_cfg = PipelineConfig(
            device_name=a["device"],
            pre_gain_db=float(a["pre_gain_db"]),
            gate_enabled=bool(a["gate_enabled"]),
            agc_enabled=bool(a["agc_enabled"]),
            inference_window_s=float(a["inference_window_s"]),
            inference_step_s=float(a["inference_step_s"]),
            lib_path=a["lib_path"],
            log_wav_path=a.get("log_wav_path"),
            stats_interval_s=10.0,
        )

        # Shared "latest" DSP stats so the inference callback can read dBFS.
        self._latest_rms_out_db = -120.0

        self.pipeline = AudioPipeline(
            self.pipe_cfg, inference_callback=self._on_window)

        # Wrap the DSP stats hook so we capture rms_out_db per block.
        self._wrap_stats()

    def _wrap_stats(self):
        """
        The pipeline updates DSP stats inside its audio callback. We piggy-back
        by reading them through the pipeline's stats lock just before inference.
        """
        pipe = self.pipeline

        def latest_rms_db():
            with pipe._stats_lock:
                s = pipe._last_stats
            return s.rms_out_db if s else -120.0

        self._latest_rms_db_fn = latest_rms_db

    # ── Called once per inference window (from pipeline's infer thread) ────────

    def _on_window(self, window: np.ndarray):
        ts = time.time()
        rms_out_db = self._latest_rms_db_fn()

        # Phase 1: classify
        preds = self.classifier.classify(window)
        if not preds:
            return
        top_idx, top_label, top_score = preds[0]

        # Phase 4: decide noise (dB + duration + class) + density
        result = self.analyzer.update(
            top_label=top_label,
            top_score=top_score,
            rms_out_db=rms_out_db,
            ts=ts,
        )
        suit = NoiseAnalyzer.suitability(result.noise_density)

        # Phase 5: ship to InfluxDB (or print in dry-run)
        if self.dry_run:
            flag = "NOISE" if result.is_noise else "     "
            ev = "*" if result.event_active else " "
            log.info("[%s%s] %-18s p=%.2f  spl=%5.1f dB  density=%.2f (%s)",
                     flag, ev, top_label[:18], top_score,
                     result.spl_db, result.noise_density, suit)
        else:
            self.influx.record(result, suit)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self):
        self.pipeline.start()
        log.info("SSSH service running (dry_run=%s). Ctrl+C to stop.",
                 self.dry_run)

    def stop(self):
        self.pipeline.stop()
        if self.influx:
            self.influx.close()
        log.info("SSSH service stopped.")


# ─── Entry point ────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(description="SSSH Phase 4 integrated service")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--list-devs", action="store_true", help="List audio devices and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run classifier+analyzer but print instead of writing to InfluxDB")
    ap.add_argument("--model", help="Override model path")
    ap.add_argument("--device", help="Override ALSA device")
    args = ap.parse_args()

    if args.list_devs:
        print(sd.query_devices())
        return

    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["model_path"] = args.model
    if args.device:
        cfg["audio"]["device"] = args.device

    svc = SsshService(cfg, dry_run=args.dry_run)
    svc.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print()
    finally:
        svc.stop()


if __name__ == "__main__":
    main()
