from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    # General application settings.
    app_name: str = "Posture Guard"
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    inference_fps: float = 12.0

    # Resource path.
    model_path: str = "models/pose_landmarker_full.task"

    # Startup and runtime behavior.
    preview_enabled: bool = True
    start_paused: bool = False
    tray_enabled: bool = True
    notifications_enabled: bool = True
    sound_enabled: bool = True
    toast_notifications_enabled: bool = True

    # Calibration.
    calibration_seconds: float = 10.0
    calibration_min_samples: int = 15

    # Neutral range thresholds relative to the calibrated baseline.
    head_forward_delta_m: float = 0.040
    torso_angle_delta_deg: float = 12.0
    neck_drop_delta_m: float = 0.025
    shoulder_tilt_delta_deg: float = 6.0
    head_tilt_delta_deg: float = 7.0
    screen_approach_delta: float = 0.075

    # Alert policy.
    posture_alert_after_seconds: float = 5.0
    posture_alert_cooldown_seconds: float = 15.0
    stillness_reminder_after_seconds: float = 180.0
    stillness_alert_cooldown_seconds: float = 10.0

    # Movement detection thresholds.
    movement_refresh_threshold: float = 0.55
    reposition_threshold: float = 1.20
    reposition_cooldown_seconds: float = 8.0

    # Anti-jitter movement filtering.
    movement_deadband: float = 0.18
    movement_score_smoothing_alpha: float = 0.30
    movement_consecutive_frames_for_refresh: int = 3
    movement_consecutive_frames_for_reposition: int = 4

    # Smoothing and pose filtering.
    landmark_smoothing_alpha: float = 0.30
    metrics_smoothing_alpha: float = 0.25
    min_visibility: float = 0.55
    render_visibility: float = 0.35
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # Ergonomic scoring weights.
    weight_forward_head: float = 0.35
    weight_torso_lean: float = 0.25
    weight_neck_drop: float = 0.15
    weight_shoulder_tilt: float = 0.10
    weight_head_tilt: float = 0.05
    weight_screen_approach: float = 0.10
    green_zone_max_score: float = 0.45
    yellow_zone_max_score: float = 1.00

    # Movement score normalization.
    movement_head_forward_unit: float = 0.008
    movement_torso_angle_unit: float = 3.5
    movement_neck_gap_unit: float = 0.008
    movement_screen_approach_unit: float = 0.025
    movement_shoulder_tilt_unit: float = 4.0
    movement_head_tilt_unit: float = 4.0
    movement_head_side_shift_unit: float = 0.010

    # Notification throttling.
    duplicate_notification_cooldown_seconds: float = 10.0
    notification_queue_size: int = 20

    # Compact session telemetry.
    telemetry_enabled: bool = True
    telemetry_flush_interval_seconds: float = 1.0
    telemetry_sample_interval_seconds: float = 5.0

    # Camera recovery.
    camera_read_failure_warning_threshold: int = 30
    camera_read_failure_reopen_threshold: int = 60
    camera_read_failure_fatal_threshold: int = 180
    camera_reopen_cooldown_seconds: float = 2.0

    # Shutdown behavior.
    shutdown_join_timeout_seconds: float = 5.0

    # Automatic baseline refresh after a meaningful seat reposition.
    auto_recalibration_enabled: bool = True
    auto_recalibration_stability_seconds: float = 2.0
    auto_recalibration_max_score: float = 0.35
    auto_recalibration_min_time_since_baseline_seconds: float = 20.0
    auto_recalibration_cooldown_seconds: float = 30.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        defaults = asdict(cls())
        defaults.update({key: value for key, value in data.items() if key in defaults})
        config = cls(**defaults)
        config.normalize()
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def normalize(self) -> None:
        """
        Keep loaded config values inside safe runtime ranges.

        A broken config.json should not prevent the application from starting.
        Invalid values are replaced with safe defaults or clamped.
        """
        self.app_name = _safe_non_empty_str(self.app_name, _default_value("app_name"))
        self.model_path = _safe_non_empty_str(self.model_path, _default_value("model_path"))

        self.camera_index = _safe_int_min(self.camera_index, 0, _default_value("camera_index"))
        self.frame_width = _safe_int_min(self.frame_width, 320, _default_value("frame_width"))
        self.frame_height = _safe_int_min(self.frame_height, 240, _default_value("frame_height"))
        self.inference_fps = _safe_float_min(self.inference_fps, 1.0, _default_value("inference_fps"))

        self.preview_enabled = _safe_bool(self.preview_enabled, _default_value("preview_enabled"))
        self.start_paused = _safe_bool(self.start_paused, _default_value("start_paused"))
        self.tray_enabled = _safe_bool(self.tray_enabled, _default_value("tray_enabled"))
        self.notifications_enabled = _safe_bool(self.notifications_enabled, _default_value("notifications_enabled"))
        self.sound_enabled = _safe_bool(self.sound_enabled, _default_value("sound_enabled"))
        self.toast_notifications_enabled = _safe_bool(
            self.toast_notifications_enabled,
            _default_value("toast_notifications_enabled"),
        )

        self.calibration_seconds = _safe_float_min(
            self.calibration_seconds,
            1.0,
            _default_value("calibration_seconds"),
        )
        self.calibration_min_samples = _safe_int_min(
            self.calibration_min_samples,
            1,
            _default_value("calibration_min_samples"),
        )

        self.posture_alert_after_seconds = _safe_float_min(
            self.posture_alert_after_seconds,
            0.0,
            _default_value("posture_alert_after_seconds"),
        )
        self.posture_alert_cooldown_seconds = _safe_float_min(
            self.posture_alert_cooldown_seconds,
            0.0,
            _default_value("posture_alert_cooldown_seconds"),
        )
        self.stillness_reminder_after_seconds = _safe_float_min(
            self.stillness_reminder_after_seconds,
            1.0,
            _default_value("stillness_reminder_after_seconds"),
        )
        self.stillness_alert_cooldown_seconds = _safe_float_min(
            self.stillness_alert_cooldown_seconds,
            0.0,
            _default_value("stillness_alert_cooldown_seconds"),
        )

        self.movement_refresh_threshold = _safe_float_min(
            self.movement_refresh_threshold,
            0.0,
            _default_value("movement_refresh_threshold"),
        )
        self.reposition_threshold = _safe_float_min(
            self.reposition_threshold,
            0.0,
            _default_value("reposition_threshold"),
        )
        self.reposition_cooldown_seconds = _safe_float_min(
            self.reposition_cooldown_seconds,
            0.0,
            _default_value("reposition_cooldown_seconds"),
        )
        self.movement_deadband = _safe_float_min(
            self.movement_deadband,
            0.0,
            _default_value("movement_deadband"),
        )
        self.movement_score_smoothing_alpha = _safe_float_clamp(
            self.movement_score_smoothing_alpha,
            0.0,
            1.0,
            _default_value("movement_score_smoothing_alpha"),
        )
        self.movement_consecutive_frames_for_refresh = _safe_int_min(
            self.movement_consecutive_frames_for_refresh,
            1,
            _default_value("movement_consecutive_frames_for_refresh"),
        )
        self.movement_consecutive_frames_for_reposition = _safe_int_min(
            self.movement_consecutive_frames_for_reposition,
            1,
            _default_value("movement_consecutive_frames_for_reposition"),
        )

        self.landmark_smoothing_alpha = _safe_float_clamp(
            self.landmark_smoothing_alpha,
            0.0,
            1.0,
            _default_value("landmark_smoothing_alpha"),
        )
        self.metrics_smoothing_alpha = _safe_float_clamp(
            self.metrics_smoothing_alpha,
            0.0,
            1.0,
            _default_value("metrics_smoothing_alpha"),
        )
        self.min_visibility = _safe_float_clamp(self.min_visibility, 0.0, 1.0, _default_value("min_visibility"))
        self.render_visibility = _safe_float_clamp(self.render_visibility, 0.0, 1.0, _default_value("render_visibility"))
        self.min_pose_detection_confidence = _safe_float_clamp(
            self.min_pose_detection_confidence,
            0.0,
            1.0,
            _default_value("min_pose_detection_confidence"),
        )
        self.min_pose_presence_confidence = _safe_float_clamp(
            self.min_pose_presence_confidence,
            0.0,
            1.0,
            _default_value("min_pose_presence_confidence"),
        )
        self.min_tracking_confidence = _safe_float_clamp(
            self.min_tracking_confidence,
            0.0,
            1.0,
            _default_value("min_tracking_confidence"),
        )

        self.head_forward_delta_m = _safe_float_min(self.head_forward_delta_m, 0.001, _default_value("head_forward_delta_m"))
        self.torso_angle_delta_deg = _safe_float_min(self.torso_angle_delta_deg, 0.1, _default_value("torso_angle_delta_deg"))
        self.neck_drop_delta_m = _safe_float_min(self.neck_drop_delta_m, 0.001, _default_value("neck_drop_delta_m"))
        self.shoulder_tilt_delta_deg = _safe_float_min(self.shoulder_tilt_delta_deg, 0.1, _default_value("shoulder_tilt_delta_deg"))
        self.head_tilt_delta_deg = _safe_float_min(self.head_tilt_delta_deg, 0.1, _default_value("head_tilt_delta_deg"))
        self.screen_approach_delta = _safe_float_min(self.screen_approach_delta, 0.001, _default_value("screen_approach_delta"))

        self.weight_forward_head = _safe_float_min(self.weight_forward_head, 0.0, _default_value("weight_forward_head"))
        self.weight_torso_lean = _safe_float_min(self.weight_torso_lean, 0.0, _default_value("weight_torso_lean"))
        self.weight_neck_drop = _safe_float_min(self.weight_neck_drop, 0.0, _default_value("weight_neck_drop"))
        self.weight_shoulder_tilt = _safe_float_min(self.weight_shoulder_tilt, 0.0, _default_value("weight_shoulder_tilt"))
        self.weight_head_tilt = _safe_float_min(self.weight_head_tilt, 0.0, _default_value("weight_head_tilt"))
        self.weight_screen_approach = _safe_float_min(self.weight_screen_approach, 0.0, _default_value("weight_screen_approach"))
        self.green_zone_max_score = _safe_float_min(self.green_zone_max_score, 0.0, _default_value("green_zone_max_score"))
        self.yellow_zone_max_score = _safe_float_min(self.yellow_zone_max_score, 0.01, _default_value("yellow_zone_max_score"))

        self.movement_head_forward_unit = _safe_float_min(self.movement_head_forward_unit, 0.0001, _default_value("movement_head_forward_unit"))
        self.movement_torso_angle_unit = _safe_float_min(self.movement_torso_angle_unit, 0.0001, _default_value("movement_torso_angle_unit"))
        self.movement_neck_gap_unit = _safe_float_min(self.movement_neck_gap_unit, 0.0001, _default_value("movement_neck_gap_unit"))
        self.movement_screen_approach_unit = _safe_float_min(self.movement_screen_approach_unit, 0.0001, _default_value("movement_screen_approach_unit"))
        self.movement_shoulder_tilt_unit = _safe_float_min(self.movement_shoulder_tilt_unit, 0.0001, _default_value("movement_shoulder_tilt_unit"))
        self.movement_head_tilt_unit = _safe_float_min(self.movement_head_tilt_unit, 0.0001, _default_value("movement_head_tilt_unit"))
        self.movement_head_side_shift_unit = _safe_float_min(self.movement_head_side_shift_unit, 0.0001, _default_value("movement_head_side_shift_unit"))

        self.duplicate_notification_cooldown_seconds = _safe_float_min(
            self.duplicate_notification_cooldown_seconds,
            0.0,
            _default_value("duplicate_notification_cooldown_seconds"),
        )
        self.notification_queue_size = _safe_int_min(
            self.notification_queue_size,
            1,
            _default_value("notification_queue_size"),
        )
        self.notification_queue_size = min(self.notification_queue_size, 500)

        self.telemetry_enabled = _safe_bool(self.telemetry_enabled, _default_value("telemetry_enabled"))
        self.telemetry_flush_interval_seconds = _safe_float_min(
            self.telemetry_flush_interval_seconds,
            0.1,
            _default_value("telemetry_flush_interval_seconds"),
        )
        self.telemetry_sample_interval_seconds = _safe_float_min(
            self.telemetry_sample_interval_seconds,
            0.25,
            _default_value("telemetry_sample_interval_seconds"),
        )

        self.camera_read_failure_warning_threshold = _safe_int_min(
            self.camera_read_failure_warning_threshold,
            1,
            _default_value("camera_read_failure_warning_threshold"),
        )
        self.camera_read_failure_reopen_threshold = _safe_int_min(
            self.camera_read_failure_reopen_threshold,
            1,
            _default_value("camera_read_failure_reopen_threshold"),
        )
        self.camera_read_failure_fatal_threshold = _safe_int_min(
            self.camera_read_failure_fatal_threshold,
            1,
            _default_value("camera_read_failure_fatal_threshold"),
        )
        self.camera_reopen_cooldown_seconds = _safe_float_min(
            self.camera_reopen_cooldown_seconds,
            0.1,
            _default_value("camera_reopen_cooldown_seconds"),
        )
        self.shutdown_join_timeout_seconds = _safe_float_min(
            self.shutdown_join_timeout_seconds,
            0.5,
            _default_value("shutdown_join_timeout_seconds"),
        )

        self.auto_recalibration_enabled = _safe_bool(
            self.auto_recalibration_enabled,
            _default_value("auto_recalibration_enabled"),
        )
        self.auto_recalibration_stability_seconds = _safe_float_min(
            self.auto_recalibration_stability_seconds,
            0.0,
            _default_value("auto_recalibration_stability_seconds"),
        )
        self.auto_recalibration_max_score = _safe_float_min(
            self.auto_recalibration_max_score,
            0.0,
            _default_value("auto_recalibration_max_score"),
        )
        self.auto_recalibration_min_time_since_baseline_seconds = _safe_float_min(
            self.auto_recalibration_min_time_since_baseline_seconds,
            0.0,
            _default_value("auto_recalibration_min_time_since_baseline_seconds"),
        )
        self.auto_recalibration_cooldown_seconds = _safe_float_min(
            self.auto_recalibration_cooldown_seconds,
            0.0,
            _default_value("auto_recalibration_cooldown_seconds"),
        )

        if self.camera_read_failure_reopen_threshold > self.camera_read_failure_fatal_threshold:
            self.camera_read_failure_reopen_threshold = max(1, self.camera_read_failure_fatal_threshold // 2)

        if self.green_zone_max_score >= self.yellow_zone_max_score:
            self.green_zone_max_score = _default_value("green_zone_max_score")
            self.yellow_zone_max_score = _default_value("yellow_zone_max_score")


def load_app_config(config_file: Path) -> AppConfig:
    """
    Load configuration from config.json.
    Invalid config values never prevent application startup.
    """
    if not config_file.exists():
        return AppConfig()

    try:
        raw = json.loads(config_file.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Could not read config file: %s", config_file)
        return AppConfig()

    if not isinstance(raw, dict):
        logger.warning("Config file %s does not contain a JSON object. Defaults will be used.", config_file)
        return AppConfig()

    try:
        return AppConfig.from_dict(raw)
    except Exception:
        logger.exception("Could not normalize config file: %s", config_file)
        return AppConfig()


def save_app_config(config_file: Path, config: AppConfig) -> bool:
    """
    Save configuration to config.json using an atomic replace.
    """
    try:
        config.normalize()
        config_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = config_file.with_name(f"{config_file.name}.tmp.{os.getpid()}")
        temp_file.write_text(
            json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_file.replace(config_file)
        return True
    except Exception:
        logger.exception("Could not save config file: %s", config_file)
        return False


def _default_value(field_name: str) -> Any:
    return AppConfig.__dataclass_fields__[field_name].default


def _safe_non_empty_str(value: Any, default: str) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return str(default)


def _safe_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return bool(default)


def _safe_int_min(value: Any, minimum: int, default: int) -> int:
    try:
        if isinstance(value, bool):
            raise ValueError("bool is not a valid integer config value")
        number = int(value)
    except Exception:
        number = int(default)
    return max(int(minimum), number)


def _safe_float_min(value: Any, minimum: float, default: float) -> float:
    try:
        if isinstance(value, bool):
            raise ValueError("bool is not a valid float config value")
        number = float(value)
    except Exception:
        number = float(default)
    return max(float(minimum), number)


def _safe_float_clamp(value: Any, minimum: float, maximum: float, default: float) -> float:
    try:
        if isinstance(value, bool):
            raise ValueError("bool is not a valid float config value")
        number = float(value)
    except Exception:
        number = float(default)
    return max(float(minimum), min(float(maximum), number))