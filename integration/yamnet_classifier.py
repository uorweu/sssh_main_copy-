#!/usr/bin/env python3
"""
yamnet_classifier.py  —  Fine-tuned YAMNet inference wrapper for the SSSH device

Responsibilities
────────────────
  • Load the fine-tuned YAMNet model, auto-detecting its format:
        - TensorFlow SavedModel directory
        - Keras .h5 / .keras file
        - TFLite .tflite file
    (Your model lives at ~/sssh_main/yamnet — a *file*, not a folder.)
  • Support a **two-stage TFLite pipeline** when both a YAMNet base model
    (yamnet.tflite) and a custom classification head (sssh_head.tflite) are
    provided.  In this mode raw 16 kHz audio is first passed through the
    YAMNet base to extract 1024-dim embedding vectors, which are then
    averaged over frames and fed into the head for final classification.
  • Resample the 48 kHz pipeline window down to 16 kHz mono (YAMNet's native rate).
  • Run inference and return the top class index, label and score.

YAMNet contract
───────────────
  YAMNet expects a 1-D float32 waveform in [-1, 1] at 16 kHz.  A standard
  YAMNet (TF-Hub) returns (scores, embeddings, log_mel_spectrogram) where
  scores is shape (num_frames, num_classes).  A fine-tuned head may instead
  return a single (num_classes,) / (1, num_classes) vector.  Both are handled.

Two-stage TFLite pipeline
─────────────────────────
  When yamnet_model_path is provided:
    1. yamnet.tflite  — full YAMNet base: raw 16 kHz audio → (scores, embeddings, spectrogram)
    2. sssh_head.tflite — custom head: averaged 1024-dim embeddings → 11-class scores

Labels
──────
  Provide a CSV with one label per line (or the YAMNet-style class_map.csv
  with an index,mid,display_name header).  Point to it via Config.labels_path.
  If omitted, classes are reported as "class_<index>".
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger("yamnet")

YAMNET_SAMPLE_RATE = 16_000


# ─── Label loading ──────────────────────────────────────────────────────────

def load_labels(labels_path: Optional[str]) -> Optional[List[str]]:
    """
    Load class labels. Accepts either:
      • a plain text file, one label per line, or
      • a CSV with a header containing a 'display_name' column (YAMNet style).
    Returns None if no path given (caller falls back to "class_<i>").
    """
    if not labels_path:
        return None
    p = Path(labels_path).expanduser()
    if not p.exists():
        log.warning("Labels file not found: %s — using numeric class ids", p)
        return None

    rows = p.read_text(encoding="utf-8").splitlines()
    if not rows:
        return None

    # YAMNet-style CSV: index,mid,display_name
    if "," in rows[0] and "display_name" in rows[0].lower():
        labels = []
        reader = csv.DictReader(rows)
        name_key = next((k for k in reader.fieldnames or []
                         if k and k.lower() == "display_name"), None)
        for r in reader:
            labels.append(r[name_key] if name_key else "")
        log.info("Loaded %d labels (YAMNet CSV) from %s", len(labels), p)
        return labels

    # Plain one-per-line (strip optional "idx,label" form)
    labels = []
    for line in rows:
        line = line.strip()
        if not line:
            continue
        if "," in line:
            line = line.split(",", 1)[1].strip()
        labels.append(line)
    log.info("Loaded %d labels from %s", len(labels), p)
    return labels


# ─── Resampling 48k → 16k ─────────────────────────────────────────────────────

def resample_48k_to_16k(window: np.ndarray) -> np.ndarray:
    """
    Decimate 48 kHz → 16 kHz (exact 3:1).  Uses scipy.resample_poly when
    available (anti-aliased, best quality); otherwise falls back to a simple
    averaging decimator that still removes most aliasing.
    """
    window = np.ascontiguousarray(window, dtype=np.float32)
    try:
        from scipy.signal import resample_poly
        out = resample_poly(window, up=1, down=3).astype(np.float32)
        return out
    except Exception:
        # Fallback: trim to a multiple of 3 and average each triplet.
        n = (len(window) // 3) * 3
        if n == 0:
            return window[:0]
        return window[:n].reshape(-1, 3).mean(axis=1).astype(np.float32)


# ─── Classifier ───────────────────────────────────────────────────────────────

class YamnetClassifier:
    """
    Wraps a fine-tuned YAMNet model behind a single .classify(window) call.

    Parameters
    ──────────
      model_path       : path to SavedModel dir, .h5/.keras file, or .tflite file
      labels_path      : optional label list (see load_labels)
      top_k            : how many top predictions to return
      yamnet_model_path: optional path to the full YAMNet base TFLite model
                         (yamnet.tflite).  When provided, enables two-stage
                         inference: base model extracts embeddings, then the
                         head model (model_path) classifies them.
      classify_min_db  : optional minimum RMS dB threshold.  When set,
                         classify() will skip inference if the provided rms_db
                         is below this value, returning 'Unidentified'.
    """

    def __init__(self, model_path: str, labels_path: Optional[str] = None,
                 top_k: int = 3, yamnet_model_path: Optional[str] = None,
                 classify_min_db: Optional[float] = None):
        self.model_path = str(Path(model_path).expanduser())
        self.labels = load_labels(labels_path)
        self.top_k = top_k
        self.classify_min_db = classify_min_db
        self._backend = None      # "tf" | "tflite"
        self._model = None
        self._tflite = None
        self._in_details = None
        self._out_details = None

        # Two-stage pipeline: YAMNet base model
        self._yamnet_model_path = (
            str(Path(yamnet_model_path).expanduser()) if yamnet_model_path
            else None
        )
        self._yamnet_tflite = None
        self._yamnet_in_details = None
        self._yamnet_out_details = None

        self._load()

    # ── Model loading / format detection ─────────────────────────────────────

    def _load(self):
        p = Path(self.model_path)
        suffix = p.suffix.lower()

        # TFLite ------------------------------------------------------------
        if suffix == ".tflite":
            self._load_tflite(p)
            # Load the YAMNet base model if provided (two-stage pipeline)
            if self._yamnet_model_path:
                self._load_yamnet_base(Path(self._yamnet_model_path))
            return

        # Keras file --------------------------------------------------------
        if suffix in (".h5", ".keras"):
            self._load_keras(p)
            return

        # SavedModel directory ---------------------------------------------
        if p.is_dir() and (p / "saved_model.pb").exists():
            self._load_savedmodel(p)
            return

        # File with no/unknown extension (your case: ~/sssh_main/yamnet).
        # Sniff the first bytes: TFLite files start with a "TFL3" magic at
        # offset 4; otherwise assume a Keras/SavedModel loadable by TF.
        if p.is_file():
            head = p.read_bytes()[:8] if p.stat().st_size >= 8 else b""
            if b"TFL3" in head:
                log.info("Detected TFLite magic in %s", p)
                self._load_tflite(p)
                # Load the YAMNet base model if provided (two-stage pipeline)
                if self._yamnet_model_path:
                    self._load_yamnet_base(Path(self._yamnet_model_path))
                return
            # HDF5 files start with \x89HDF
            if head[:4] == b"\x89HDF":
                log.info("Detected HDF5 (Keras) format in %s", p)
                self._load_keras(p)
                return
            # Last resort: try Keras loader (handles .keras zip too)
            log.info("Unknown extension for %s — trying Keras loader", p)
            self._load_keras(p)
            return

        raise FileNotFoundError(f"Could not locate / identify model at {p}")

    def _load_savedmodel(self, p: Path):
        import tensorflow as tf
        log.info("Loading SavedModel: %s", p)
        self._model = tf.saved_model.load(str(p))
        self._backend = "tf"

    def _load_keras(self, p: Path):
        import tensorflow as tf
        log.info("Loading Keras model: %s", p)
        self._model = tf.keras.models.load_model(str(p), compile=False)
        self._backend = "tf"

    def _load_tflite(self, p: Path):
        try:
            import tflite_runtime.interpreter as tflite
            self._tflite = tflite.Interpreter(model_path=str(p))
        except ImportError:
            import tensorflow as tf
            self._tflite = tf.lite.Interpreter(model_path=str(p))
        self._tflite.allocate_tensors()
        self._in_details = self._tflite.get_input_details()
        self._out_details = self._tflite.get_output_details()
        self._backend = "tflite"
        log.info("Loaded TFLite model: %s (input %s)",
                 p, self._in_details[0]["shape"])

    def _load_yamnet_base(self, p: Path):
        """Load the full YAMNet base TFLite model for two-stage inference."""
        if not p.exists():
            log.warning("YAMNet base model not found: %s — falling back to "
                        "single-stage mode", p)
            return
        try:
            import tflite_runtime.interpreter as tflite
            self._yamnet_tflite = tflite.Interpreter(model_path=str(p))
        except ImportError:
            import tensorflow as tf
            self._yamnet_tflite = tf.lite.Interpreter(model_path=str(p))
        self._yamnet_tflite.allocate_tensors()
        self._yamnet_in_details = self._yamnet_tflite.get_input_details()
        self._yamnet_out_details = self._yamnet_tflite.get_output_details()
        log.info("Two-stage mode: YAMNet base → custom head")
        log.info("  YAMNet base loaded: %s (input %s, %d outputs)",
                 p, self._yamnet_in_details[0]["shape"],
                 len(self._yamnet_out_details))

    # ── Inference ─────────────────────────────────────────────────────────────

    def _run_tf(self, wav16: np.ndarray) -> np.ndarray:
        """Returns a 1-D scores vector (num_classes,)."""
        import tensorflow as tf
        x = tf.convert_to_tensor(wav16, dtype=tf.float32)

        try:
            out = self._model(x)                  # raw YAMNet signature
        except Exception:
            # Keras functional model usually wants a batch dim.
            out = self._model(tf.expand_dims(x, 0))

        # YAMNet hub returns a tuple (scores, embeddings, spectrogram)
        if isinstance(out, (tuple, list)):
            out = out[0]
        scores = np.asarray(out)
        if scores.ndim == 2:
            # (frames, classes) → mean over frames; (1, classes) → squeeze
            scores = scores.mean(axis=0) if scores.shape[0] > 1 else scores[0]
        return scores.astype(np.float32)

    def _extract_embeddings(self, wav16: np.ndarray) -> np.ndarray:
        """
        Run the YAMNet base model on 16 kHz audio and extract embeddings.

        YAMNet TFLite models vary in their output layout:
          • 3 outputs (TF Hub style): scores (N,521), embeddings (N,1024), spectrogram (N,64)
          • 2 outputs: scores + embeddings (from export_yamnet_with_embeddings.py)
          • 1 output: may be scores (N,521) only

        Returns a single averaged embedding/feature vector.
        """
        inp = self._yamnet_in_details[0]
        x = wav16.astype(np.float32)

        # YAMNet base expects a 1-D waveform
        if x.ndim > 1:
            x = x.flatten()

        self._yamnet_tflite.resize_tensor_input(inp["index"], list(x.shape))
        self._yamnet_tflite.allocate_tensors()
        self._yamnet_tflite.set_tensor(inp["index"], x)
        self._yamnet_tflite.invoke()

        # Collect all output tensors and log their shapes for debugging
        all_outputs = []
        for i, out_detail in enumerate(self._yamnet_out_details):
            tensor = self._yamnet_tflite.get_tensor(out_detail["index"])
            all_outputs.append(tensor)
            if not hasattr(self, '_yamnet_shapes_logged'):
                log.info("  YAMNet output[%d]: shape=%s dtype=%s",
                         i, tensor.shape, tensor.dtype)
        self._yamnet_shapes_logged = True

        # Strategy 1: Find the tensor with last dimension == 1024 (embeddings)
        embeddings = None
        for tensor in all_outputs:
            if tensor.ndim == 2 and tensor.shape[-1] == 1024:
                embeddings = tensor
                break

        # Strategy 2: If only 1 output, use it directly (whatever it is).
        # The head model was likely trained to accept this output shape.
        if embeddings is None and len(all_outputs) == 1:
            log.info("YAMNet has single output (shape=%s) — using it directly "
                     "as input to the head model", all_outputs[0].shape)
            embeddings = all_outputs[0]

        # Strategy 3: Multiple outputs but none are 1024-dim.
        # Pick the one with the largest last dimension (most likely embeddings).
        if embeddings is None:
            best = max(all_outputs, key=lambda t: t.shape[-1] if t.ndim >= 1 else 0)
            log.warning("Could not identify 1024-dim embeddings output; "
                        "using output with shape %s (largest last dim)", best.shape)
            embeddings = best

        # Average over frames → single vector
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 2:
            embeddings = np.mean(embeddings, axis=0)

        return embeddings

    def _run_head_tflite(self, embedding: np.ndarray) -> np.ndarray:
        """
        Run the classification head TFLite model on a single embedding vector.
        The head expects input shape (1, 1024) or (1024,).
        """
        inp = self._in_details[0]
        want = list(inp["shape"])
        x = embedding.astype(np.float32)

        # Match the head model's expected rank
        if len(want) == 2:
            x = x.reshape(1, -1)

        self._tflite.resize_tensor_input(inp["index"], list(x.shape))
        self._tflite.allocate_tensors()
        self._tflite.set_tensor(inp["index"], x)
        self._tflite.invoke()

        out = self._tflite.get_tensor(self._out_details[0]["index"])
        scores = np.asarray(out)
        if scores.ndim == 2:
            scores = scores.mean(axis=0) if scores.shape[0] > 1 else scores[0]
        return scores.astype(np.float32)

    def _run_tflite(self, wav16: np.ndarray) -> np.ndarray:
        # Two-stage pipeline: YAMNet base → embeddings → head
        if self._yamnet_tflite is not None:
            embedding = self._extract_embeddings(wav16)
            return self._run_head_tflite(embedding)

        # Single-stage fallback (backward compatible)
        inp = self._in_details[0]
        want = list(inp["shape"])
        x = wav16.astype(np.float32)
        # Match the model's expected rank (some expect (N,), some (1, N)).
        if len(want) == 2:
            x = x.reshape(1, -1)
        # If a fixed input length is required, pad/trim to it.
        if want and want[-1] not in (-1, None) and want[-1] != x.shape[-1]:
            target = want[-1]
            if x.shape[-1] > target:
                x = x[..., :target]
            else:
                pad = target - x.shape[-1]
                x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)])
        self._tflite.resize_tensor_input(inp["index"], list(x.shape))
        self._tflite.allocate_tensors()
        self._tflite.set_tensor(inp["index"], x)
        self._tflite.invoke()
        out = self._tflite.get_tensor(self._out_details[0]["index"])
        scores = np.asarray(out)
        if scores.ndim == 2:
            scores = scores.mean(axis=0) if scores.shape[0] > 1 else scores[0]
        return scores.astype(np.float32)

    def classify(self, window_48k: np.ndarray,
                 rms_db: Optional[float] = None
                 ) -> List[Tuple[int, str, float]]:
        """
        Classify one 48 kHz mono window.
        Returns a list of (class_index, label, score), highest score first,
        length == top_k (or fewer if the model has fewer classes).

        If classify_min_db was set and rms_db is provided and below that
        threshold, classification is skipped and [(0, 'Unidentified', 0.0)]
        is returned to avoid classifying quiet ambient noise.
        """
        # Gate on minimum dB level
        if (self.classify_min_db is not None
                and rms_db is not None
                and rms_db < self.classify_min_db):
            return [(0, 'Unidentified', 0.0)]

        wav16 = resample_48k_to_16k(window_48k)
        if wav16.size == 0:
            return []

        if self._backend == "tf":
            scores = self._run_tf(wav16)
        else:
            scores = self._run_tflite(wav16)

        k = min(self.top_k, scores.shape[-1])
        top_idx = np.argsort(scores)[-k:][::-1]
        results = []
        for idx in top_idx:
            label = (self.labels[idx]
                     if self.labels and idx < len(self.labels)
                     else "Unidentified")
            results.append((int(idx), label, float(scores[idx])))
        return results

