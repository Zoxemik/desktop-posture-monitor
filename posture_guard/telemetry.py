from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional


TRACKING_ZONES = {"green", "yellow", "red"}


def _sanitize_filename(value: str) -> str:
    cleaned = []
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
        elif char in (" ", "-", "_"):
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "session"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _bool_to_csv(value: bool) -> str:
    return "true" if value else "false"


class TelemetryLogger:
    """
    Small analytics-focused session logger.

    The logger keeps CSV files readable and useful for charts:
    - samples.csv: low-frequency line chart data,
    - segments.csv: continuous posture/pose intervals,
    - alerts.csv: generated alerts with delivered/suppressed information,
    - session.json: final session summary and config metadata.
    """

    SAMPLE_FIELDNAMES = [
        "time_s",
        "clock",
        "pose_state",
        "zone",
        "total_score",
        "movement_score",
        "bad_duration_s",
        "static_duration_s",
        "dominant_issue",
        "notification_mode",
    ]

    SEGMENT_FIELDNAMES = [
        "start_s",
        "end_s",
        "duration_s",
        "pose_state",
        "zone",
        "dominant_issue",
        "avg_total_score",
        "max_total_score",
        "avg_movement_score",
        "is_real_bad_posture",
        "notification_mode",
    ]

    ALERT_FIELDNAMES = [
        "time_s",
        "clock",
        "alert_type",
        "zone",
        "total_score",
        "movement_score",
        "dominant_issue",
        "bad_duration_s",
        "static_duration_s",
        "notification_mode",
        "delivered",
    ]

    def __init__(
        self,
        data_dir: Path,
        app_name: str,
        enabled: bool = True,
        flush_interval_seconds: float = 1.0,
        sample_interval_seconds: float = 5.0,
        real_bad_posture_min_seconds: float = 5.0,
    ) -> None:
        self._enabled = enabled
        self._flush_interval_seconds = max(0.1, float(flush_interval_seconds))
        self._sample_interval_seconds = max(0.25, float(sample_interval_seconds))
        self._real_bad_posture_min_seconds = max(0.0, float(real_bad_posture_min_seconds))
        self._last_flush_time = time.monotonic()

        self._samples_file = None
        self._segments_file = None
        self._alerts_file = None
        self._samples_writer: Optional[csv.DictWriter] = None
        self._segments_writer: Optional[csv.DictWriter] = None
        self._alerts_writer: Optional[csv.DictWriter] = None
        self._session_dir: Optional[Path] = None

        self._session_payload: dict[str, Any] = {}
        self._first_monotonic_seconds: Optional[float] = None
        self._session_started_at_iso: Optional[str] = None
        self._last_sample_time_s: Optional[float] = None
        self._last_frame_row: Optional[dict[str, Any]] = None
        self._last_frame_time_s: Optional[float] = None

        self._segment_key: Optional[tuple[str, str, str, str]] = None
        self._segment_start_s = 0.0
        self._segment_duration_s = 0.0
        self._segment_total_score_sum = 0.0
        self._segment_movement_score_sum = 0.0
        self._segment_max_total_score = 0.0

        self._tracking_s = 0.0
        self._zone_durations = {
            "green": 0.0,
            "yellow": 0.0,
            "red": 0.0,
            "calibration": 0.0,
            "no_person": 0.0,
            "unreliable": 0.0,
            "paused": 0.0,
            "startup": 0.0,
            "error": 0.0,
            "unknown": 0.0,
        }
        self._pose_state_durations: dict[str, float] = {}
        self._notification_mode_durations: dict[str, float] = {}
        self._tracking_total_score_time_sum = 0.0
        self._tracking_movement_score_time_sum = 0.0
        self._max_total_score = 0.0
        self._max_movement_score = 0.0
        self._real_bad_posture_s = 0.0
        self._longest_bad_posture_s = 0.0
        self._bad_posture_segments = 0
        self._dominant_issue_bad_durations: dict[str, float] = {}

        self._posture_alerts_generated = 0
        self._posture_alerts_delivered = 0
        self._stillness_alerts_generated = 0
        self._stillness_alerts_delivered = 0
        self._max_reposition_count = 0
        self._baseline_refreshes = 0
        self._auto_recalibrations = 0
        self._manual_recalibrations = 0
        self._sample_count = 0
        self._segment_count = 0
        self._alert_count = 0

        if not enabled:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"{_sanitize_filename(app_name)}_{timestamp}"

        self._session_dir = data_dir / "telemetry" / session_name
        self._session_dir.mkdir(parents=True, exist_ok=True)

        self._samples_file = (self._session_dir / "samples.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        )
        self._segments_file = (self._session_dir / "segments.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        )
        self._alerts_file = (self._session_dir / "alerts.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        )

        self._samples_writer = csv.DictWriter(
            self._samples_file,
            fieldnames=self.SAMPLE_FIELDNAMES,
            extrasaction="ignore",
        )
        self._segments_writer = csv.DictWriter(
            self._segments_file,
            fieldnames=self.SEGMENT_FIELDNAMES,
            extrasaction="ignore",
        )
        self._alerts_writer = csv.DictWriter(
            self._alerts_file,
            fieldnames=self.ALERT_FIELDNAMES,
            extrasaction="ignore",
        )

        self._samples_writer.writeheader()
        self._segments_writer.writeheader()
        self._alerts_writer.writeheader()

    @property
    def session_dir(self) -> Optional[Path]:
        return self._session_dir

    def write_session_metadata(
        self,
        config_snapshot: Mapping[str, Any],
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not self._enabled:
            return

        self._session_payload = {
            "created_at_iso": datetime.now().isoformat(timespec="seconds"),
            "config": dict(config_snapshot),
        }
        if extra:
            self._session_payload.update(dict(extra))

    def log_frame(self, row: Mapping[str, Any]) -> None:
        if not self._enabled or self._samples_writer is None:
            return

        current_row = dict(row)
        current_monotonic = _safe_float(current_row.get("monotonic_seconds"), time.monotonic())
        current_time_s = self._get_relative_time_s(current_monotonic, current_row)

        if self._last_frame_row is not None and self._last_frame_time_s is not None:
            previous_row = self._last_frame_row
            previous_time_s = self._last_frame_time_s
            delta_seconds = max(0.0, current_time_s - previous_time_s)

            if self._segment_key is None:
                self._start_segment(previous_row, previous_time_s)

            self._add_duration_to_current_segment(previous_row, delta_seconds)
            self._add_duration_to_summary(previous_row, delta_seconds)

            previous_key = self._build_segment_key(previous_row)
            current_key = self._build_segment_key(current_row)
            if current_key != previous_key:
                self._close_current_segment(current_time_s)
                self._start_segment(current_row, current_time_s)
        else:
            self._start_segment(current_row, current_time_s)

        self._max_reposition_count = max(
            self._max_reposition_count,
            _safe_int(current_row.get("reposition_count")),
        )
        self._write_sample_if_due(current_row, current_time_s)

        self._last_frame_row = current_row
        self._last_frame_time_s = current_time_s
        self._maybe_flush()

    def log_alert(self, row: Mapping[str, Any]) -> None:
        if not self._enabled or self._alerts_writer is None:
            return

        current_row = dict(row)
        current_monotonic = _safe_float(current_row.get("monotonic_seconds"), time.monotonic())
        current_time_s = self._get_relative_time_s(current_monotonic, current_row)
        alert_type = str(current_row.get("alert_type", current_row.get("kind", "")) or "")
        delivered = bool(current_row.get("delivered", False))

        if alert_type == "posture":
            self._posture_alerts_generated += 1
            if delivered:
                self._posture_alerts_delivered += 1
        elif alert_type == "stillness":
            self._stillness_alerts_generated += 1
            if delivered:
                self._stillness_alerts_delivered += 1

        self._alerts_writer.writerow(
            self._normalize_row(
                {
                    "time_s": round(current_time_s, 3),
                    "clock": current_row.get("clock", current_row.get("wall_clock_iso", "")),
                    "alert_type": alert_type,
                    "zone": current_row.get("zone", ""),
                    "total_score": round(_safe_float(current_row.get("total_score")), 6),
                    "movement_score": round(_safe_float(current_row.get("movement_score")), 6),
                    "dominant_issue": current_row.get("dominant_issue", ""),
                    "bad_duration_s": round(_safe_float(current_row.get("bad_duration_s")), 3),
                    "static_duration_s": round(_safe_float(current_row.get("static_duration_s")), 3),
                    "notification_mode": current_row.get("notification_mode", "on"),
                    "delivered": _bool_to_csv(delivered),
                },
                self.ALERT_FIELDNAMES,
            )
        )
        self._alert_count += 1
        self._maybe_flush(force=True)

    def log_event(self, row: Mapping[str, Any]) -> None:
        """
        Keep sparse non-alert events inside session.json counters.

        Zone changes are represented by segments.csv, so they are intentionally
        not written as a separate CSV anymore.
        """
        if not self._enabled:
            return

        kind = str(row.get("kind", "") or "")
        details = str(row.get("details", row.get("message", "")) or "")

        if kind == "baseline_ready":
            self._baseline_refreshes += 1
        elif kind == "recalibration_started":
            if "auto" in details:
                self._auto_recalibrations += 1
            else:
                self._manual_recalibrations += 1

    def close(self) -> None:
        if self._enabled:
            if self._last_frame_time_s is not None:
                self._close_current_segment(self._last_frame_time_s)
            self._write_session_file()

        for handle in (self._samples_file, self._segments_file, self._alerts_file):
            if handle is None:
                continue
            try:
                handle.flush()
                handle.close()
            except Exception:
                pass

    def _get_relative_time_s(self, monotonic_seconds: float, row: Mapping[str, Any]) -> float:
        if self._first_monotonic_seconds is None:
            self._first_monotonic_seconds = monotonic_seconds
            self._session_started_at_iso = str(row.get("clock", row.get("wall_clock_iso", "")) or "")
        return max(0.0, monotonic_seconds - self._first_monotonic_seconds)

    def _write_sample_if_due(self, row: Mapping[str, Any], time_s: float) -> None:
        if self._samples_writer is None:
            return

        if (
            self._last_sample_time_s is not None
            and (time_s - self._last_sample_time_s) < self._sample_interval_seconds
        ):
            return

        self._last_sample_time_s = time_s
        self._samples_writer.writerow(
            self._normalize_row(
                {
                    "time_s": round(time_s, 3),
                    "clock": row.get("clock", row.get("wall_clock_iso", "")),
                    "pose_state": row.get("pose_state", "unknown"),
                    "zone": row.get("zone", "unknown"),
                    "total_score": round(_safe_float(row.get("total_score")), 6),
                    "movement_score": round(_safe_float(row.get("movement_score")), 6),
                    "bad_duration_s": round(_safe_float(row.get("bad_duration_s")), 3),
                    "static_duration_s": round(_safe_float(row.get("static_duration_s")), 3),
                    "dominant_issue": row.get("dominant_issue", ""),
                    "notification_mode": row.get("notification_mode", "on"),
                },
                self.SAMPLE_FIELDNAMES,
            )
        )
        self._sample_count += 1

    def _build_segment_key(self, row: Mapping[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(row.get("pose_state", "unknown") or "unknown"),
            str(row.get("zone", "unknown") or "unknown"),
            str(row.get("dominant_issue", "") or ""),
            str(row.get("notification_mode", "on") or "on"),
        )

    def _start_segment(self, row: Mapping[str, Any], start_s: float) -> None:
        self._segment_key = self._build_segment_key(row)
        self._segment_start_s = max(0.0, float(start_s))
        self._segment_duration_s = 0.0
        self._segment_total_score_sum = 0.0
        self._segment_movement_score_sum = 0.0
        self._segment_max_total_score = 0.0

    def _add_duration_to_current_segment(self, row: Mapping[str, Any], delta_seconds: float) -> None:
        if self._segment_key is None or delta_seconds <= 0.0:
            return

        total_score = _safe_float(row.get("total_score"))
        movement_score = _safe_float(row.get("movement_score"))

        self._segment_duration_s += delta_seconds
        self._segment_total_score_sum += total_score * delta_seconds
        self._segment_movement_score_sum += movement_score * delta_seconds
        self._segment_max_total_score = max(self._segment_max_total_score, total_score)

    def _add_duration_to_summary(self, row: Mapping[str, Any], delta_seconds: float) -> None:
        if delta_seconds <= 0.0:
            return

        pose_state = str(row.get("pose_state", "unknown") or "unknown")
        zone = str(row.get("zone", "unknown") or "unknown")
        notification_mode = str(row.get("notification_mode", "on") or "on")
        total_score = _safe_float(row.get("total_score"))
        movement_score = _safe_float(row.get("movement_score"))

        self._pose_state_durations[pose_state] = self._pose_state_durations.get(pose_state, 0.0) + delta_seconds
        self._zone_durations[zone] = self._zone_durations.get(zone, 0.0) + delta_seconds
        self._notification_mode_durations[notification_mode] = (
            self._notification_mode_durations.get(notification_mode, 0.0) + delta_seconds
        )

        if pose_state == "tracking":
            self._tracking_s += delta_seconds
            self._tracking_total_score_time_sum += total_score * delta_seconds
            self._tracking_movement_score_time_sum += movement_score * delta_seconds
            self._max_total_score = max(self._max_total_score, total_score)
            self._max_movement_score = max(self._max_movement_score, movement_score)

    def _close_current_segment(self, end_s: float) -> None:
        if self._segments_writer is None or self._segment_key is None:
            return

        pose_state, zone, dominant_issue, notification_mode = self._segment_key
        duration_s = max(0.0, self._segment_duration_s)
        if duration_s <= 0.0:
            self._segment_key = None
            return

        avg_total_score = self._segment_total_score_sum / duration_s
        avg_movement_score = self._segment_movement_score_sum / duration_s
        is_real_bad_posture = (
            pose_state == "tracking"
            and zone == "red"
            and duration_s >= self._real_bad_posture_min_seconds
        )

        if is_real_bad_posture:
            self._real_bad_posture_s += duration_s
            self._longest_bad_posture_s = max(self._longest_bad_posture_s, duration_s)
            self._bad_posture_segments += 1
            if dominant_issue:
                self._dominant_issue_bad_durations[dominant_issue] = (
                    self._dominant_issue_bad_durations.get(dominant_issue, 0.0) + duration_s
                )

        self._segments_writer.writerow(
            self._normalize_row(
                {
                    "start_s": round(self._segment_start_s, 3),
                    "end_s": round(max(end_s, self._segment_start_s), 3),
                    "duration_s": round(duration_s, 3),
                    "pose_state": pose_state,
                    "zone": zone,
                    "dominant_issue": dominant_issue,
                    "avg_total_score": round(avg_total_score, 6),
                    "max_total_score": round(self._segment_max_total_score, 6),
                    "avg_movement_score": round(avg_movement_score, 6),
                    "is_real_bad_posture": _bool_to_csv(is_real_bad_posture),
                    "notification_mode": notification_mode,
                },
                self.SEGMENT_FIELDNAMES,
            )
        )
        self._segment_count += 1
        self._segment_key = None

    def _write_session_file(self) -> None:
        if self._session_dir is None:
            return

        duration_s = 0.0
        if self._last_frame_time_s is not None:
            duration_s = max(0.0, self._last_frame_time_s)

        tracking_s = max(0.0, self._tracking_s)
        avg_total_score = self._tracking_total_score_time_sum / tracking_s if tracking_s > 0.0 else 0.0
        avg_movement_score = self._tracking_movement_score_time_sum / tracking_s if tracking_s > 0.0 else 0.0

        green_s = self._zone_durations.get("green", 0.0)
        yellow_s = self._zone_durations.get("yellow", 0.0)
        red_s = self._zone_durations.get("red", 0.0)
        tracking_coverage_percent = (tracking_s / duration_s * 100.0) if duration_s > 0.0 else 0.0
        real_bad_posture_percent = (self._real_bad_posture_s / tracking_s * 100.0) if tracking_s > 0.0 else 0.0

        top_dominant_issue = ""
        if self._dominant_issue_bad_durations:
            top_dominant_issue = max(
                self._dominant_issue_bad_durations,
                key=self._dominant_issue_bad_durations.get,
            )

        notification_mode = "unknown"
        config = self._session_payload.get("config", {})
        if isinstance(config, dict):
            notification_mode = "on" if bool(config.get("notifications_enabled", True)) else "off"

        payload = {
            **self._session_payload,
            "session_started_at_iso": self._session_started_at_iso,
            "finished_at_iso": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(duration_s, 3),
            "notification_mode_at_start": notification_mode,
            "sample_count": self._sample_count,
            "segment_count": self._segment_count,
            "alert_count": self._alert_count,
            "tracking_s": round(tracking_s, 3),
            "tracking_coverage_percent": round(tracking_coverage_percent, 3),
            "green_s": round(green_s, 3),
            "yellow_s": round(yellow_s, 3),
            "red_s": round(red_s, 3),
            "green_percent_of_tracking": round((green_s / tracking_s * 100.0) if tracking_s > 0.0 else 0.0, 3),
            "yellow_percent_of_tracking": round((yellow_s / tracking_s * 100.0) if tracking_s > 0.0 else 0.0, 3),
            "red_percent_of_tracking": round((red_s / tracking_s * 100.0) if tracking_s > 0.0 else 0.0, 3),
            "real_bad_posture_s": round(self._real_bad_posture_s, 3),
            "real_bad_posture_percent": round(real_bad_posture_percent, 3),
            "longest_bad_posture_s": round(self._longest_bad_posture_s, 3),
            "bad_posture_segments": self._bad_posture_segments,
            "avg_total_score": round(avg_total_score, 6),
            "max_total_score": round(self._max_total_score, 6),
            "avg_movement_score": round(avg_movement_score, 6),
            "max_movement_score": round(self._max_movement_score, 6),
            "top_dominant_issue": top_dominant_issue,
            "dominant_issue_bad_durations_s": {
                key: round(value, 3)
                for key, value in sorted(
                    self._dominant_issue_bad_durations.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            },
            "posture_alerts_generated": self._posture_alerts_generated,
            "posture_alerts_delivered": self._posture_alerts_delivered,
            "stillness_alerts_generated": self._stillness_alerts_generated,
            "stillness_alerts_delivered": self._stillness_alerts_delivered,
            "repositions": self._max_reposition_count,
            "baseline_refreshes": self._baseline_refreshes,
            "auto_recalibrations": self._auto_recalibrations,
            "manual_recalibrations": self._manual_recalibrations,
            "pose_state_durations_s": {
                key: round(value, 3)
                for key, value in sorted(self._pose_state_durations.items())
            },
            "notification_mode_durations_s": {
                key: round(value, 3)
                for key, value in sorted(self._notification_mode_durations.items())
            },
            "zone_durations_s": {
                key: round(value, 3)
                for key, value in sorted(self._zone_durations.items())
                if value > 0.0
            },
        }

        session_path = self._session_dir / "session.json"
        session_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _normalize_row(self, row: Mapping[str, Any], fieldnames: list[str]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for field in fieldnames:
            value = row.get(field, "")
            if value is None:
                value = ""
            normalized[field] = value
        return normalized

    def _maybe_flush(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_flush_time) < self._flush_interval_seconds:
            return

        self._last_flush_time = now
        for handle in (self._samples_file, self._segments_file, self._alerts_file):
            if handle is None:
                continue
            try:
                handle.flush()
            except Exception:
                pass
