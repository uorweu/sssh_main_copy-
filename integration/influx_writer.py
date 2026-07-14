#!/usr/bin/env python3
"""
influx_writer.py  —  Sends SSSH noise metrics to InfluxDB (on the laptop)

Uses the InfluxDB 2.x client (influxdb-client) talking to a remote InfluxDB
over the network.  Points are batched and flushed periodically so we don't
do a network round-trip on every 0.5 s window.

If the laptop/InfluxDB is unreachable, points are buffered in memory (bounded)
and retried on the next flush, so brief network drops don't lose data or stall
the audio pipeline.

Schema
──────
  measurement: "sssh_noise"
  tags:        device, location
  fields:
        noise_density   (float, 0..1)   ← headline metric for Grafana
        spl_db          (float)         ← estimated SPL (relative until calibrated)
        rms_out_db      (float)
        top_score       (float)
        is_noise        (int 0/1)
        event_active    (int 0/1)
        top_label       (string)
        suitability     (string)        ← quiet/moderate/busy/loud

  measurement: "sssh_events"            ← FSM-based noise event summaries
  tags:        device, location, dominant_class
  fields:
        start_time      (float, epoch)
        end_time        (float, epoch)
        duration_s      (float)
        avg_spl_db      (float)
        peak_spl_db     (float)
        avg_density     (float)
        dominant_class  (string)
        class_counts    (string, JSON-encoded dict)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

log = logging.getLogger("influx")


@dataclass
class InfluxConfig:
    url: str = "http://100.65.252.55:8086"   # ← your laptop's IP / InfluxDB port
    token: str = "HXHIrhYAe5sE-37tLnUQ34jshooUstA9BcuLCy0NzTofpjkaOwsh7yEWm4tRc5Sb48VxJ8Mt5g82epG58eBDyg=="                # InfluxDB API token
    org: str = "Edge_computing"
    bucket: str = "Noises"
    device: str = "pi4-01"                  # tag: which device
    location: str = "library-floor2"        # tag: where it is
    flush_interval_s: float = 2.0
    max_buffer: int = 5000                  # cap on offline-buffered points


class InfluxWriter:
    """
    Thread-safe point writer with background flushing.
    Call .record(result, suitability) from the inference thread; a background
    thread batches and ships points to InfluxDB.
    """

    def __init__(self, cfg: InfluxConfig):
        self.cfg = cfg
        self._buf: Deque = deque(maxlen=cfg.max_buffer)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._client = None
        self._write_api = None
        self._Point = None
        self._connect()

        self._thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="InfluxFlush")
        self._thread.start()

    def _connect(self):
        try:
            from influxdb_client import InfluxDBClient, Point
            from influxdb_client.client.write_api import SYNCHRONOUS
            self._Point = Point
            self._client = InfluxDBClient(
                url=self.cfg.url, token=self.cfg.token, org=self.cfg.org,
                timeout=5_000)
            self._write_api = self._client.write_api(write_options=SYNCHRONOUS)
            log.info("InfluxDB client ready → %s (bucket=%s)",
                     self.cfg.url, self.cfg.bucket)
        except ImportError:
            log.error("influxdb-client not installed: pip install influxdb-client")
            raise
        except Exception as e:
            # Don't crash the pipeline if Influx is down at boot — buffer instead.
            log.warning("InfluxDB connect failed (%s) — will buffer & retry", e)

    # ── Public API ────────────────────────────────────────────────────────────

    def record(self, result, suitability: str):
        """Queue one WindowResult for writing. Non-blocking."""
        p = self._Point("sssh_noise") if self._Point else None
        if p is None:
            # Client lib missing — shouldn't happen after _connect, but guard.
            return
        p = (p.tag("device", self.cfg.device)
              .tag("location", self.cfg.location)
              .field("noise_density", float(result.noise_density))
              .field("spl_db", float(result.spl_db))
              .field("rms_out_db", float(result.rms_out_db))
              .field("top_score", float(result.top_score))
              .field("is_noise", int(result.is_noise))
              .field("event_active", int(result.event_active))
              .field("top_label", str(result.top_label))
              .field("state", str(result.state))
              .field("suitability", str(suitability))
              .time(int(result.ts * 1e9)))   # ns precision
        with self._lock:
            self._buf.append(p)

    def record_event(self, event_summary, device: Optional[str] = None,
                     location: Optional[str] = None):
        """Queue one FSM EventSummary for writing to the sssh_events measurement.

        Parameters
        ----------
        event_summary : EventSummary
            A completed noise-event summary produced by the FSM, containing
            start_time, end_time, duration_s, avg_spl_db, peak_spl_db,
            avg_density, dominant_class, and class_counts.
        device : str, optional
            Override ``self.cfg.device`` for the device tag.
        location : str, optional
            Override ``self.cfg.location`` for the location tag.
        """
        p = self._Point("sssh_events") if self._Point else None
        if p is None:
            return

        dev = device or self.cfg.device
        loc = location or self.cfg.location

        # Encode class_counts dict as a compact JSON string for storage.
        counts_json = json.dumps(
            event_summary.class_counts, separators=(",", ":"))

        p = (p.tag("device", dev)
              .tag("location", loc)
              .tag("dominant_class", str(event_summary.dominant_class))
              .field("start_time", float(event_summary.start_time))
              .field("end_time", float(event_summary.end_time))
              .field("duration_s", float(event_summary.duration_s))
              .field("avg_spl_db", float(event_summary.avg_spl_db))
              .field("peak_spl_db", float(event_summary.peak_spl_db))
              .field("avg_density", float(event_summary.avg_density))
              .field("dominant_class", str(event_summary.dominant_class))
              .field("class_counts", counts_json)
              .time(int(event_summary.end_time * 1e9)))   # ns precision
        with self._lock:
            self._buf.append(p)
        log.info("Queued sssh_event: %s (%.1fs, peak=%.1f dB)",
                 event_summary.dominant_class, event_summary.duration_s,
                 event_summary.peak_spl_db)

    # ── Background flush ──────────────────────────────────────────────────────

    def _flush_loop(self):
        while not self._stop.is_set():
            time.sleep(self.cfg.flush_interval_s)
            self._flush_once()

    def _flush_once(self):
        with self._lock:
            if not self._buf:
                return
            batch = list(self._buf)

        if self._write_api is None:
            self._connect()                  # retry connecting
            if self._write_api is None:
                return                        # still down — keep buffering

        try:
            self._write_api.write(bucket=self.cfg.bucket,
                                  org=self.cfg.org, record=batch)
            with self._lock:
                # Drop exactly what we shipped (newer points may have arrived).
                for _ in range(len(batch)):
                    if self._buf:
                        self._buf.popleft()
            log.debug("Flushed %d points to InfluxDB", len(batch))
        except Exception as e:
            log.warning("InfluxDB write failed (%s) — %d points buffered",
                        e, len(batch))

    def close(self):
        self._stop.set()
        self._flush_once()
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
        log.info("InfluxWriter closed")
