#!/usr/bin/env python3
"""
noise_analyzer.py  —  Defines what counts as "noise" for the SSSH project

Per the project brief, a sound is disruptive (real "noise") when it combines:
  • dB LEVEL    — loud enough to be heard over the room floor
  • DURATION    — sustained, not a momentary blip (short ambient sounds are ignored)
  • CLASS       — belongs to a disruptive class (human voice / distinct sounds),
                  not benign ambience (HVAC hum, distant traffic, etc.)

This module turns a stream of per-window classifications + RMS levels into:
  • a per-window NoiseEvent decision (is this window disruptive right now?)
  • a rolling NOISE DENSITY metric (fraction of recent time that was disruptive)
    which is the headline number the website/Grafana will show.

dBFS → SPL note
───────────────
The DSP reports RMS in dBFS (full-scale relative).  Converting to true SPL
(dB-A) needs a one-time calibration against a reference SPL meter, because it
depends on the mic sensitivity, the pre-gain, and the ICS-43434 itself.
`spl_offset_db` is that calibration constant:  SPL ≈ dBFS + spl_offset_db.
Until calibrated, treat the SPL column as relative, not absolute.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence, Tuple


# ─── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class NoiseConfig:
    # A window's top class must clear this confidence to be trusted.
    min_confidence: float = 0.30

    # Loudness gate.  Compared against estimated SPL (dBFS + spl_offset_db).
    # Set spl_offset_db from calibration; default 0 → SPL == dBFS (relative).
    spl_offset_db: float = 0.0
    spl_threshold_db: float = -35.0      # below this = effectively quiet

    # Duration gate.  A disruptive class must persist at least this long
    # (across consecutive windows) before it's logged as a real event.
    # Short blips below this are treated as ambient and ignored.
    min_event_duration_s: float = 0.7

    # Class policy.  Either list disruptive classes, or list benign ones.
    # Matching is case-insensitive substring against the model's labels.
    # If disruptive_classes is non-empty it takes priority (allow-list);
    # otherwise everything that is NOT in benign_classes is disruptive.
    disruptive_classes: List[str] = field(default_factory=lambda: [
        "speech", "conversation", "shout", "yell", "laughter",
        "child", "crowd", "music", "singing", "phone",
    ])
    benign_classes: List[str] = field(default_factory=lambda: [
        "silence", "ambient", "air conditioning", "hum",
        "wind", "rustle", "white noise",
    ])

    # Rolling density window (seconds) used for the headline metric.
    density_window_s: float = 30.0

    # Seconds represented by one inference step (must match pipeline step).
    step_s: float = 0.5


# ─── Per-window result ──────────────────────────────────────────────────────

@dataclass
class WindowResult:
    ts: float                    # epoch seconds
    top_label: str
    top_score: float
    rms_out_db: float            # dBFS from DSP
    spl_db: float                # estimated SPL
    is_loud: bool
    is_disruptive_class: bool
    is_noise: bool               # loud AND disruptive class AND confident
    # filled in once duration is confirmed:
    event_active: bool = False   # part of a confirmed sustained event
    noise_density: float = 0.0   # rolling fraction [0..1] at this moment


# ─── Analyzer ─────────────────────────────────────────────────────────────────

class NoiseAnalyzer:
    """
    Stateful, single-threaded analyzer. Feed it one window at a time via
    .update(...); it returns a WindowResult with the noise decision and the
    current rolling density.
    """

    def __init__(self, cfg: NoiseConfig):
        self.cfg = cfg
        self._disruptive = [c.lower() for c in cfg.disruptive_classes]
        self._benign = [c.lower() for c in cfg.benign_classes]

        # Track a candidate ongoing event.
        self._run_start: Optional[float] = None   # ts the current noise run began
        self._run_confirmed = False

        # Rolling history of (ts, is_noise_confirmed) for density.
        self._history: Deque[Tuple[float, bool]] = deque()

    # ── Class policy ──────────────────────────────────────────────────────────

    def _class_is_disruptive(self, label: str) -> bool:
        l = label.lower()
        if self._disruptive:                       # allow-list mode
            return any(d in l for d in self._disruptive)
        return not any(b in l for b in self._benign)  # deny-list mode

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self,
               top_label: str,
               top_score: float,
               rms_out_db: float,
               ts: Optional[float] = None) -> WindowResult:
        cfg = self.cfg
        ts = time.time() if ts is None else ts

        spl_db = rms_out_db + cfg.spl_offset_db
        is_loud = spl_db >= cfg.spl_threshold_db
        confident = top_score >= cfg.min_confidence
        is_disruptive_class = confident and self._class_is_disruptive(top_label)

        # Instantaneous "this window looks like noise"
        is_noise = is_loud and is_disruptive_class

        # ── Duration tracking ────────────────────────────────────────────────
        event_active = False
        if is_noise:
            if self._run_start is None:
                self._run_start = ts
                self._run_confirmed = False
            run_len = ts - self._run_start
            if run_len >= cfg.min_event_duration_s:
                self._run_confirmed = True
            event_active = self._run_confirmed
        else:
            # Run broken — reset.
            self._run_start = None
            self._run_confirmed = False

        # ── Rolling density ────────────────────────────────────────────────────
        # Only confirmed (sustained) noise counts toward density.
        self._history.append((ts, event_active))
        cutoff = ts - cfg.density_window_s
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        if self._history:
            noisy = sum(1 for _, n in self._history if n)
            density = noisy / len(self._history)
        else:
            density = 0.0

        return WindowResult(
            ts=ts,
            top_label=top_label,
            top_score=top_score,
            rms_out_db=rms_out_db,
            spl_db=spl_db,
            is_loud=is_loud,
            is_disruptive_class=is_disruptive_class,
            is_noise=is_noise,
            event_active=event_active,
            noise_density=density,
        )

    # ── Convenience: classify a "suitability" bucket for the website ──────────

    @staticmethod
    def suitability(density: float) -> str:
        """Map noise density → human-friendly label for the UI."""
        if density < 0.10:
            return "quiet"
        if density < 0.30:
            return "moderate"
        if density < 0.60:
            return "busy"
        return "loud"
