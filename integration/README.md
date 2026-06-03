# SSSH — Phase 4: System Integration

This bundle wires every phase into one running edge service on the Raspberry Pi 4:

```
ICS-43434 ─► pipeline.py + libaudio_dsp.so ─► YAMNet ─► noise analyzer ─► InfluxDB ─► Grafana
              (Phase 3 range extension)      (Phase 1)   (Phase 4)        (Phase 5)
```

## Files

| File | Role |
|------|------|
| `pipeline.py`            | Phase 3 — ALSA capture + DSP C library (unchanged) |
| `audio_dsp.c/.h`, `Makefile`, `libaudio_dsp.so` | Phase 3 — the DSP (rebuild on the Pi) |
| `yamnet_classifier.py`   | Loads the fine-tuned YAMNet, resamples 48k→16k, classifies |
| `noise_analyzer.py`      | **Defines "noise"**: dB level + duration + class → density |
| `influx_writer.py`       | Batches metrics to InfluxDB on the laptop |
| `main.py`                | Integration entrypoint that wires it all together |
| `config.yaml`            | All tunables in one place |
| `requirements.txt`       | Python deps |

## Setup on the Pi

```bash
sudo apt install libasound2-dev libportaudio2
cd ~/sssh_main            # wherever you put these files
python3 -m pip install -r requirements.txt

# rebuild the DSP lib natively (the shipped .so is aarch64 already, but to be safe):
make
```

Install the model runtime that matches your file:
- SavedModel / `.h5` / `.keras` → `pip install tensorflow`
- `.tflite` (lighter, recommended) → `pip install tflite-runtime`

The code **auto-detects** the model format, so `~/sssh_main/yamnet` works whether
it's a SavedModel, an HDF5 file, or a TFLite file (it sniffs the file's magic bytes).

## Configure

Edit `config.yaml`. The fields you must set are marked `←`:
- `model.model_path` → `~/sssh_main/yamnet`
- `model.labels_path` → your 30-class label list (one per line, or YAMNet CSV)
- `influx.url` → your **laptop's** IP + InfluxDB port (e.g. `http://192.168.1.50:8086`)
- `influx.token`, `influx.org`, `influx.bucket` → from your InfluxDB setup

## Run

```bash
# 1. Find the mic device index
python3 main.py --list-devs

# 2. Dry run — no database, prints noise decisions live (use this to tune thresholds)
python3 main.py --dry-run

# 3. Full service — streams metrics to InfluxDB
python3 main.py --config config.yaml
```

## How "noise" is decided (Phase 4 core)

A 1-second window is flagged as **noise** only when all three hold:
1. **Loud enough** — estimated SPL ≥ `spl_threshold_db`
2. **Disruptive class** — top YAMNet class is in `disruptive_classes` (speech,
   laughter, crowd, music…) with confidence ≥ `min_confidence`
3. **Sustained** — the run persists ≥ `min_event_duration_s` (short blips are
   treated as ambient and ignored)

Only sustained, confirmed events feed the headline **noise density** — the
fraction of the last `density_window_s` seconds that were disruptive. That single
0–1 number, plus a `suitability` bucket (quiet/moderate/busy/loud), is what
Grafana shows so users can decide whether the space is good for studying.

## Calibration note (important)

The DSP reports loudness in **dBFS** (relative to full scale), not real-world SPL.
To show true dB-A, measure the room with a reference SPL meter and set
`noise.spl_offset_db` so that `SPL ≈ dBFS + spl_offset_db`. Until then, treat the
SPL/threshold values as a **relative** scale (still fine for density trends).

## InfluxDB schema (for building Grafana panels)

- measurement: `sssh_noise`
- tags: `device`, `location`
- fields: `noise_density`, `spl_db`, `rms_out_db`, `top_score`,
  `is_noise`, `event_active`, `top_label`, `suitability`

Headline Grafana panel: time series of `noise_density` (0–1) per `location`.
