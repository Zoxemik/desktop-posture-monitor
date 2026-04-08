from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


def _sanitize_filename(value: str) -> str:
    cleaned = []
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
        elif char in (" ", "-", "_"):
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "session"


class TelemetryLogger:
    """
    Compact session logger focused on posture strain and movement.

    Runtime telemetry is intentionally small and easy to analyze:
    - timeline.csv: one compact row per processed frame,
    - events.csv: sparse high-level events,
    - summary.json: aggregated session-level metrics,
    - session_meta.json: config snapshot and basic metadata.
    """

    TIMELINE_FIELDNAMES = [
        "wall_clock_iso",
        "monotonic_seconds",
        "timestamp_ms",
        "frame_index",
        "pose_state",
        "zone",
        "total_score",
        "movement_score",
        "dominant_issue",
        "baseline_generation",
    ]

    EVENT_FIELDNAMES = [
        "wall_clock_iso",
        "monotonic_seconds",
        "timestamp_ms",
        "frame_index",
        "kind",
        "zone",
        "total_score",
        "movement_score",
        "dominant_issue",
        "details",
        "baseline_generation",
    ]

    def __init__(
        self,
        data_dir: Path,
        app_name: str,
        enabled: bool = True,
        flush_interval_seconds: float = 1.0,
    ) -> None:
        self._enabled = enabled
        self._flush_interval_seconds = max(0.1, float(flush_interval_seconds))
        self._last_flush_time = time.monotonic()
        self._timeline_file = None
        self._events_file = None
        self._timeline_writer: csv.DictWriter | None = None
        self._events_writer: csv.DictWriter | None = None
        self._session_dir: Path | None = None

        self._summary = {
            "processed_frames": 0,
            "avg_total_score": 0.0,
            "max_total_score": 0.0,
            "avg_movement_score": 0.0,
            "max_movement_score": 0.0,
            "time_in_green_s": 0.0,
            "time_in_yellow_s": 0.0,
            "time_in_red_s": 0.0,
            "time_in_unreliable_s": 0.0,
            "time_in_no_person_s": 0.0,
            "time_in_calibration_s": 0.0,
            "time_in_paused_s": 0.0,
            "posture_alerts": 0,
            "stillness_alerts": 0,
            "repositions": 0,
            "auto_recalibrations": 0,
            "manual_recalibrations": 0,
            "baseline_refreshes": 0,
        }
        self._total_score_sum = 0.0
        self._movement_score_sum = 0.0
        self._last_timestamp_ms: int | None = None
        self._last_zone: str | None = None
        self._max_posture_alert_count = 0
        self._max_stillness_alert_count = 0
        self._max_reposition_count = 0

        if not enabled:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"{_sanitize_filename(app_name)}_{timestamp}"

        self._session_dir = data_dir / "telemetry" / session_name
        self._session_dir.mkdir(parents=True, exist_ok=True)

        self._timeline_file = (self._session_dir / "timeline.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        )
        self._events_file = (self._session_dir / "events.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        )

        self._timeline_writer = csv.DictWriter(
            self._timeline_file,
            fieldnames=self.TIMELINE_FIELDNAMES,
            extrasaction="ignore",
        )
        self._events_writer = csv.DictWriter(
            self._events_file,
            fieldnames=self.EVENT_FIELDNAMES,
            extrasaction="ignore",
        )

        self._timeline_writer.writeheader()
        self._events_writer.writeheader()

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    def write_session_metadata(self, config_snapshot: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> None:
        if not self._enabled or self._session_dir is None:
            return

        payload: dict[str, Any] = {
            "created_at_iso": datetime.now().isoformat(timespec="seconds"),
            "config": dict(config_snapshot),
        }
        if extra:
            payload.update(dict(extra))

        metadata_path = self._session_dir / "session_meta.json"
        metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def log_frame(self, row: Mapping[str, Any]) -> None:
        if not self._enabled or self._timeline_writer is None:
            return

        normalized = self._normalize_row(row, self.TIMELINE_FIELDNAMES)
        self._timeline_writer.writerow(normalized)
        self._update_summary_from_frame(row)
        self._maybe_flush()

    def log_event(self, row: Mapping[str, Any]) -> None:
        if not self._enabled or self._events_writer is None:
            return

        normalized = self._normalize_row(row, self.EVENT_FIELDNAMES)
        self._events_writer.writerow(normalized)
        self._update_summary_from_event(row)
        self._maybe_flush(force=True)

    def close(self) -> None:
        if self._enabled:
            self._write_summary_file()

        for handle in (self._timeline_file, self._events_file):
            if handle is None:
                continue
            try:
                handle.flush()
                handle.close()
            except Exception:
                pass

    def _normalize_row(self, row: Mapping[str, Any], fieldnames: list[str]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for field in fieldnames:
            value = row.get(field, "")
            if value is None:
                value = ""
            normalized[field] = value
        return normalized

    def _update_summary_from_frame(self, row: Mapping[str, Any]) -> None:
        self._summary["processed_frames"] += 1

        total_score = float(row.get("total_score", 0.0) or 0.0)
        movement_score = float(row.get("movement_score", 0.0) or 0.0)
        self._total_score_sum += total_score
        self._movement_score_sum += movement_score
        self._summary["max_total_score"] = max(float(self._summary["max_total_score"]), total_score)
        self._summary["max_movement_score"] = max(float(self._summary["max_movement_score"]), movement_score)

        timestamp_ms = int(row.get("timestamp_ms", row.get("stream_timestamp_ms", 0)) or 0)
        zone = str(row.get("zone", "unknown") or "unknown")

        if self._last_timestamp_ms is not None and self._last_zone is not None:
            delta_seconds = max(0.0, (timestamp_ms - self._last_timestamp_ms) / 1000.0)
            self._add_zone_duration(self._last_zone, delta_seconds)

        self._last_timestamp_ms = timestamp_ms
        self._last_zone = zone

        self._max_posture_alert_count = max(
            self._max_posture_alert_count,
            int(row.get("posture_alert_count", 0) or 0),
        )
        self._max_stillness_alert_count = max(
            self._max_stillness_alert_count,
            int(row.get("stillness_alert_count", 0) or 0),
        )
        self._max_reposition_count = max(
            self._max_reposition_count,
            int(row.get("reposition_count", 0) or 0),
        )

    def _update_summary_from_event(self, row: Mapping[str, Any]) -> None:
        kind = str(row.get("kind", "") or "")
        details = str(row.get("details", row.get("message", "")) or "")

        if kind in {"posture", "posture_alert"}:
            self._summary["posture_alerts"] += 1
        elif kind in {"stillness", "stillness_alert"}:
            self._summary["stillness_alerts"] += 1
        elif kind == "baseline_ready":
            self._summary["baseline_refreshes"] += 1
        elif kind == "recalibration_started":
            if "auto" in details:
                self._summary["auto_recalibrations"] += 1
            else:
                self._summary["manual_recalibrations"] += 1

    def _add_zone_duration(self, zone: str, delta_seconds: float) -> None:
        if zone == "green":
            self._summary["time_in_green_s"] += delta_seconds
        elif zone == "yellow":
            self._summary["time_in_yellow_s"] += delta_seconds
        elif zone == "red":
            self._summary["time_in_red_s"] += delta_seconds
        elif zone == "unreliable":
            self._summary["time_in_unreliable_s"] += delta_seconds
        elif zone == "no_person":
            self._summary["time_in_no_person_s"] += delta_seconds
        elif zone == "calibration":
            self._summary["time_in_calibration_s"] += delta_seconds
        elif zone == "paused":
            self._summary["time_in_paused_s"] += delta_seconds

    def _write_summary_file(self) -> None:
        if self._session_dir is None:
            return

        processed_frames = int(self._summary["processed_frames"])
        if processed_frames > 0:
            self._summary["avg_total_score"] = self._total_score_sum / processed_frames
            self._summary["avg_movement_score"] = self._movement_score_sum / processed_frames
        else:
            self._summary["avg_total_score"] = 0.0
            self._summary["avg_movement_score"] = 0.0

        self._summary["posture_alerts"] = max(
            int(self._summary["posture_alerts"]),
            self._max_posture_alert_count,
        )
        self._summary["stillness_alerts"] = max(
            int(self._summary["stillness_alerts"]),
            self._max_stillness_alert_count,
        )
        self._summary["repositions"] = self._max_reposition_count

        duration_s = 0.0
        if self._last_timestamp_ms is not None:
            duration_s = max(0.0, self._last_timestamp_ms / 1000.0)

        payload = {
            "created_at_iso": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(duration_s, 3),
            "processed_frames": processed_frames,
            "avg_total_score": round(float(self._summary["avg_total_score"]), 6),
            "max_total_score": round(float(self._summary["max_total_score"]), 6),
            "avg_movement_score": round(float(self._summary["avg_movement_score"]), 6),
            "max_movement_score": round(float(self._summary["max_movement_score"]), 6),
            "time_in_green_s": round(float(self._summary["time_in_green_s"]), 3),
            "time_in_yellow_s": round(float(self._summary["time_in_yellow_s"]), 3),
            "time_in_red_s": round(float(self._summary["time_in_red_s"]), 3),
            "time_in_unreliable_s": round(float(self._summary["time_in_unreliable_s"]), 3),
            "time_in_no_person_s": round(float(self._summary["time_in_no_person_s"]), 3),
            "time_in_calibration_s": round(float(self._summary["time_in_calibration_s"]), 3),
            "time_in_paused_s": round(float(self._summary["time_in_paused_s"]), 3),
            "posture_alerts": int(self._summary["posture_alerts"]),
            "stillness_alerts": int(self._summary["stillness_alerts"]),
            "repositions": int(self._summary["repositions"]),
            "auto_recalibrations": int(self._summary["auto_recalibrations"]),
            "manual_recalibrations": int(self._summary["manual_recalibrations"]),
            "baseline_refreshes": int(self._summary["baseline_refreshes"]),
        }

        summary_path = self._session_dir / "summary.json"
        summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _maybe_flush(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_flush_time) < self._flush_interval_seconds:
            return

        self._last_flush_time = now
        for handle in (self._timeline_file, self._events_file):
            if handle is None:
                continue
            try:
                handle.flush()
            except Exception:
                pass