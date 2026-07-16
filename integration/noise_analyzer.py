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

FSM States
──────────
  QUIET            — ambient, nothing disruptive
  CANDIDATE        — potential noise detected, gathering evidence
  CONFIRMED_NOISE  — active confirmed noise event
  COOLDOWN         — noise paused, grace period before ending the event
  (EVENT_END)      — pseudo-transition from COOLDOWN → QUIET that emits an
                     EventSummary
"""

from __future__ import annotations

import enum
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


# ─── dB reference thresholds (for Grafana / UI display) ────────────────────────
# These are informational constants; the FSM itself uses spl_threshold_db.
DB_GREEN_MAX = 55       # < 55 dB SPL  → green  (quiet)
DB_YELLOW_MAX = 70      # 55-70 dB SPL → yellow (moderate)
                        # > 70 dB SPL  → red    (loud)


# ─── FSM States ────────────────────────────────────────────────────────────────

class State(enum.Enum):
    """Finite-state-machine states for the noise analyzer."""
    QUIET = "QUIET"
    CANDIDATE = "CANDIDATE"
    CONFIRMED_NOISE = "CONFIRMED_NOISE"
    COOLDOWN = "COOLDOWN"


# ─── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class NoiseConfig:
    # A window's top class must clear this confidence to be trusted.
    min_confidence: float = 0.30

    # Loudness gate.  Compared against estimated SPL (dBFS + spl_offset_db).
    # Set spl_offset_db from calibration; default 0 → SPL == dBFS (relative).
    spl_offset_db: float = 0.0
    spl_yellow_threshold_db: float = 55.0
    spl_threshold_db: float = 75.0      # below this = effectively quiet

    # Duration gate.  A disruptive class must persist at least this long
    # (across consecutive windows) before it's logged as a real event.
    # Short blips below this are treated as ambient and ignored.
    min_event_duration_s: float = 0.7

    # Class policy.  List disruptive classes (exact match against model labels).
    # Only classes in this list trigger noise events.
    # These match the 11 custom finetuned YAMNet classes.
    disruptive_classes: List[str] = field(default_factory=lambda: [
        "A01_Normal_Speech",
        "A04_Laughter",
    ])
    benign_classes: List[str] = field(default_factory=lambda: [
        "A05_Cough",
        "B01_Footsteps_Hard_Floor",
        "B02_Chair_Drag",
        "C01_Page_Sound",
        "C02_Pen_Writing",
        "D01_Keyboard_Typing",
        "D02_Mouse_Click",
        "E01_AC_Hum",
        "I01_Silence_Room_Tone",
    ])

    # Rolling density window (seconds) used for the headline metric.
    density_window_s: float = 30.0

    # Seconds represented by one inference step (must match pipeline step).
    step_s: float = 0.5

    # Cooldown grace period: how long to wait after noise stops before
    # ending an event.  Bridges brief silences within a single noise event.
    cooldown_s: float = 2.0

    # How often (seconds) to emit a heartbeat window in QUIET state.
    heartbeat_interval_s: float = 30.0

    # How often (seconds) to emit data while in CONFIRMED_NOISE state.
    moderate_send_interval_s: float = 5.0
    noise_send_interval_s: float = 2.0

    # Fallback: if ANY sound stays above spl_threshold_db (Red) for this many
    # seconds continuously, treat it as noise regardless of classification.
    # Handles construction, alarms, or other loud unidentified sounds.
    unidentified_loud_timeout_s: float = 10.0


# ─── Event summary (emitted on EVENT_END transition) ───────────────────────────

@dataclass
class EventSummary:
    """Aggregate statistics for a completed noise event."""
    start_time: float          # epoch when the event started
    end_time: float            # epoch when the event ended
    duration_s: float
    avg_spl_db: float
    peak_spl_db: float
    avg_density: float
    dominant_class: str        # class that appeared most often during the event
    class_counts: Dict[str, int]   # {class_name: count}


# ─── Per-window result ──────────────────────────────────────────────────────

@dataclass
class WindowResult:
    ts: float                    # epoch seconds
    top_label: str
    top_score: float
    rms_out_db: float            # dBFS from DSP
    spl_db: float                # estimated SPL
    is_moderate: bool
    is_loud: bool
    is_disruptive_class: bool
    is_noise: bool               # moderate/loud AND disruptive class AND confident
    # filled in once duration is confirmed:
    event_active: bool = False   # part of a confirmed sustained event
    noise_density: float = 0.0   # rolling fraction [0..1] at this moment
    # FSM additions:
    state: str = "QUIET"         # current FSM state name
    should_send: bool = False    # whether this window should be sent to InfluxDB
    event_summary: Optional[EventSummary] = None  # populated on event end


# ─── Analyzer ─────────────────────────────────────────────────────────────────

class NoiseAnalyzer:
    """
    Stateful, single-threaded analyzer. Feed it one window at a time via
    .update(...); it returns a WindowResult with the noise decision and the
    current rolling density.

    Internally runs a 4-state FSM:
        QUIET → CANDIDATE → CONFIRMED_NOISE → COOLDOWN → QUIET (event end)
    """

    def __init__(self, cfg: NoiseConfig):
        self.cfg = cfg
        self._disruptive = [c.lower() for c in cfg.disruptive_classes]
        self._benign = [c.lower() for c in cfg.benign_classes]

        # FSM state
        self._state: State = State.QUIET

        # Track a candidate ongoing event.
        self._run_start: Optional[float] = None   # ts the current noise run began
        self._cooldown_start: Optional[float] = None  # ts cooldown began

        # Event accumulator (active during CANDIDATE / CONFIRMED_NOISE / COOLDOWN)
        self._event_start: Optional[float] = None
        self._event_spl_values: List[float] = []
        self._event_density_values: List[float] = []
        self._event_class_counts: Counter = Counter()

        # Rolling history of (ts, is_noise_confirmed) for density.
        self._history: Deque[Tuple[float, bool]] = deque()

        # Timing for send gating
        self._last_heartbeat_ts: float = 0.0
        self._last_noise_send_ts: float = 0.0

        # Track last sent SPL color zone ("green"/"yellow"/"red") so we can
        # force-send whenever the zone changes, even inside the same FSM state.
        self._last_sent_spl_zone: str = "green"

        # Fallback tracker: how long has a loud-but-unclassified sound persisted?
        self._loud_unidentified_start: Optional[float] = None

    # ── Class policy ──────────────────────────────────────────────────────────

    def _spl_zone(self, spl_db: float) -> str:
        """Return the color zone for a given SPL value."""
        if spl_db >= self.cfg.spl_threshold_db:
            return "red"
        elif spl_db >= self.cfg.spl_yellow_threshold_db:
            return "yellow"
        return "green"

    def _class_is_disruptive(self, label: str) -> bool:
        l = label.lower()
        if self._disruptive:                       # allow-list mode
            return any(d in l for d in self._disruptive)
        return not any(b in l for b in self._benign)  # deny-list mode

    # ── Event accumulator helpers ─────────────────────────────────────────────

    def _start_event(self, ts: float) -> None:
        """Begin accumulating data for a new noise event."""
        self._event_start = ts
        self._event_spl_values = []
        self._event_density_values = []
        self._event_class_counts = Counter()

    def _accumulate(self, label: str, spl_db: float, density: float) -> None:
        """Add one window's data to the running event accumulator."""
        self._event_spl_values.append(spl_db)
        self._event_density_values.append(density)
        self._event_class_counts[label] += 1

    def _finalize_event(self, end_ts: float) -> EventSummary:
        """Build an EventSummary from the accumulated data and reset."""
        spl_vals = self._event_spl_values or [0.0]
        density_vals = self._event_density_values or [0.0]
        class_counts = dict(self._event_class_counts)
        dominant = self._event_class_counts.most_common(1)
        dominant_class = dominant[0][0] if dominant else ""

        summary = EventSummary(
            start_time=self._event_start or end_ts,
            end_time=end_ts,
            duration_s=end_ts - (self._event_start or end_ts),
            avg_spl_db=sum(spl_vals) / len(spl_vals),
            peak_spl_db=max(spl_vals),
            avg_density=sum(density_vals) / len(density_vals),
            dominant_class=dominant_class,
            class_counts=class_counts,
        )

        # Reset accumulator
        self._event_start = None
        self._event_spl_values = []
        self._event_density_values = []
        self._event_class_counts = Counter()
        return summary

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self,
               top_label: str,
               top_score: float,
               rms_out_db: float,
               ts: Optional[float] = None) -> WindowResult:
        cfg = self.cfg
        ts = time.time() if ts is None else ts

        spl_db = rms_out_db + cfg.spl_offset_db
        is_moderate = spl_db >= cfg.spl_yellow_threshold_db and spl_db < cfg.spl_threshold_db
        is_loud = spl_db >= cfg.spl_threshold_db
        confident = top_score >= cfg.min_confidence
        
        # If the AI is confused (low confidence), it's an unidentified sound
        if not confident:
            top_label = "Unidentified"
            
        is_disruptive_class = confident and self._class_is_disruptive(top_label)

        is_moderate_noise = is_moderate and is_disruptive_class
        is_red_noise = is_loud and is_disruptive_class

        # ── Fallback: sustained loud unidentified noise ───────────────────────
        # If the sound is loud (Red zone) but NOT a known disruptive class,
        # track how long it has been going. If it exceeds the timeout,
        # force it to be treated as noise (e.g. construction, alarms).
        if is_loud and not is_disruptive_class:
            if self._loud_unidentified_start is None:
                self._loud_unidentified_start = ts
            elif (ts - self._loud_unidentified_start) >= cfg.unidentified_loud_timeout_s:
                # Override: treat as red noise
                is_red_noise = True
                is_disruptive_class = True  # so FSM treats it properly
        else:
            # Reset the tracker: either not loud, or it IS a known class
            self._loud_unidentified_start = None

        # Instantaneous "this window looks like noise" (either moderate or loud)
        is_noise = is_moderate_noise or is_red_noise

        # ── FSM transitions ──────────────────────────────────────────────────
        event_active = False
        event_summary: Optional[EventSummary] = None
        prev_state = self._state

        if self._state == State.QUIET:
            if is_noise:
                self._state = State.CANDIDATE
                self._run_start = ts
                self._start_event(ts)

        elif self._state == State.CANDIDATE:
            if is_noise:
                run_len = ts - (self._run_start or ts)
                if run_len >= cfg.min_event_duration_s:
                    self._state = State.CONFIRMED_NOISE
                    event_active = True
            else:
                # Noise stopped before confirmation — discard.
                self._state = State.QUIET
                self._run_start = None
                # Reset the event accumulator (never confirmed)
                self._event_start = None
                self._event_spl_values = []
                self._event_density_values = []
                self._event_class_counts = Counter()

        elif self._state == State.CONFIRMED_NOISE:
            if is_noise:
                event_active = True
            else:
                # Noise paused — enter cooldown grace period.
                self._state = State.COOLDOWN
                self._cooldown_start = ts

        elif self._state == State.COOLDOWN:
            if is_noise:
                # Noise resumed within cooldown — go back to confirmed.
                self._state = State.CONFIRMED_NOISE
                self._cooldown_start = None
                event_active = True
            else:
                cooldown_elapsed = ts - (self._cooldown_start or ts)
                if cooldown_elapsed >= cfg.cooldown_s:
                    # Cooldown expired → EVENT_END transition → QUIET
                    self._state = State.QUIET
                    self._cooldown_start = None
                    self._run_start = None

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

        # ── Accumulate event data during active states ─────────────────────────
        if self._state in (State.CANDIDATE, State.CONFIRMED_NOISE, State.COOLDOWN):
            self._accumulate(top_label, spl_db, density)

        # ── Finalize event on COOLDOWN → QUIET transition ─────────────────────
        if prev_state == State.COOLDOWN and self._state == State.QUIET:
            event_summary = self._finalize_event(ts)

        # ── should_send logic ─────────────────────────────────────────────────
        should_send = False
        state_changed = (self._state != prev_state)
        current_zone = self._spl_zone(spl_db)
        zone_changed = (current_zone != self._last_sent_spl_zone)

        if state_changed or zone_changed:
            # Always send immediately on any FSM state transition OR whenever
            # the SPL crosses a color boundary (green↔yellow, yellow↔red).
            # This captures brief spikes even when the FSM stays in QUIET.
            should_send = True

        elif event_summary is not None:
            # Event just ended — always send the summary.
            should_send = True

        elif self._state == State.QUIET:
            # Heartbeat: send every 20s so Grafana shows the AI classification
            # (e.g. "people present but quiet") even when volume is low.
            if (ts - self._last_heartbeat_ts) >= cfg.heartbeat_interval_s:
                should_send = True

        elif self._state == State.CANDIDATE:
            should_send = False

        elif self._state == State.CONFIRMED_NOISE:
            if is_red_noise:
                if (ts - self._last_noise_send_ts) >= cfg.noise_send_interval_s:
                    should_send = True
            else:
                # Moderate noise
                if (ts - self._last_noise_send_ts) >= cfg.moderate_send_interval_s:
                    should_send = True

        elif self._state == State.COOLDOWN:
            should_send = False

        # Update send timestamps when we actually send.
        if should_send:
            self._last_sent_spl_zone = current_zone
            if self._state == State.QUIET:
                self._last_heartbeat_ts = ts
            elif self._state == State.CONFIRMED_NOISE:
                self._last_noise_send_ts = ts

        # Decide display state
        display_state = self._state.value
        if self._state == State.CONFIRMED_NOISE and is_moderate_noise:
            display_state = "MODERATE_NOISE"

        return WindowResult(
            ts=ts,
            top_label=top_label,
            top_score=top_score,
            rms_out_db=rms_out_db,
            spl_db=spl_db,
            is_moderate=is_moderate,
            is_loud=is_loud,
            is_disruptive_class=is_disruptive_class,
            is_noise=is_noise,
            event_active=event_active,
            noise_density=density,
            state=display_state,
            should_send=should_send,
            event_summary=event_summary,
        )

    # ── Convenience: classify a "suitability" bucket for the website ──────────

    @staticmethod
    def suitability(density: float, spl_db: float = 0.0,
                    yellow_threshold: float = 55.0,
                    red_threshold: float = 75.0) -> str:
        """Map noise density + current SPL → human-friendly label for the UI.

        Uses whichever is *worse* between the rolling density and the
        instantaneous SPL level so the display never says 'quiet' while
        the room is actually loud.
        """
        # Density-based label
        _levels = ("quiet", "moderate", "busy", "loud")
        if density < 0.10:
            density_level = 0
        elif density < 0.30:
            density_level = 1
        elif density < 0.60:
            density_level = 2
        else:
            density_level = 3

        # SPL-based label (instantaneous)
        if spl_db >= red_threshold:
            spl_level = 3   # loud
        elif spl_db >= yellow_threshold:
            spl_level = 1   # moderate
        else:
            spl_level = 0   # quiet

        return _levels[max(density_level, spl_level)]

