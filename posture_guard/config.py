from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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

    # Compact session telemetry.
    telemetry_enabled: bool = True
    telemetry_flush_interval_seconds: float = 1.0
    telemetry_sample_interval_seconds: float = 5.0

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

        This is intentionally non-strict: a broken config.json should not prevent
        the application from starting. Values that could crash calculations are
        clamped to safe minimums.
        """
        self.camera_index = max(0, int(self.camera_index))
        self.frame_width = max(320, int(self.frame_width))
        self.frame_height = max(240, int(self.frame_height))
        self.inference_fps = _min_float(self.inference_fps, 1.0)

        self.calibration_seconds = _min_float(self.calibration_seconds, 1.0)
        self.calibration_min_samples = max(1, int(self.calibration_min_samples))

        self.posture_alert_after_seconds = _min_float(self.posture_alert_after_seconds, 0.0)
        self.posture_alert_cooldown_seconds = _min_float(self.posture_alert_cooldown_seconds, 0.0)
        self.stillness_reminder_after_seconds = _min_float(self.stillness_reminder_after_seconds, 1.0)
        self.stillness_alert_cooldown_seconds = _min_float(self.stillness_alert_cooldown_seconds, 0.0)

        self.movement_refresh_threshold = _min_float(self.movement_refresh_threshold, 0.0)
        self.reposition_threshold = _min_float(self.reposition_threshold, 0.0)
        self.reposition_cooldown_seconds = _min_float(self.reposition_cooldown_seconds, 0.0)
        self.movement_deadband = _min_float(self.movement_deadband, 0.0)
        self.movement_score_smoothing_alpha = _clamp_float(self.movement_score_smoothing_alpha, 0.0, 1.0)
        self.movement_consecutive_frames_for_refresh = max(1, int(self.movement_consecutive_frames_for_refresh))
        self.movement_consecutive_frames_for_reposition = max(1, int(self.movement_consecutive_frames_for_reposition))

        self.landmark_smoothing_alpha = _clamp_float(self.landmark_smoothing_alpha, 0.0, 1.0)
        self.metrics_smoothing_alpha = _clamp_float(self.metrics_smoothing_alpha, 0.0, 1.0)
        self.min_visibility = _clamp_float(self.min_visibility, 0.0, 1.0)
        self.render_visibility = _clamp_float(self.render_visibility, 0.0, 1.0)
        self.min_pose_detection_confidence = _clamp_float(self.min_pose_detection_confidence, 0.0, 1.0)
        self.min_pose_presence_confidence = _clamp_float(self.min_pose_presence_confidence, 0.0, 1.0)
        self.min_tracking_confidence = _clamp_float(self.min_tracking_confidence, 0.0, 1.0)

        self.head_forward_delta_m = _min_float(self.head_forward_delta_m, 0.001)
        self.torso_angle_delta_deg = _min_float(self.torso_angle_delta_deg, 0.1)
        self.neck_drop_delta_m = _min_float(self.neck_drop_delta_m, 0.001)
        self.shoulder_tilt_delta_deg = _min_float(self.shoulder_tilt_delta_deg, 0.1)
        self.head_tilt_delta_deg = _min_float(self.head_tilt_delta_deg, 0.1)
        self.screen_approach_delta = _min_float(self.screen_approach_delta, 0.001)

        self.movement_head_forward_unit = _min_float(self.movement_head_forward_unit, 0.0001)
        self.movement_torso_angle_unit = _min_float(self.movement_torso_angle_unit, 0.0001)
        self.movement_neck_gap_unit = _min_float(self.movement_neck_gap_unit, 0.0001)
        self.movement_screen_approach_unit = _min_float(self.movement_screen_approach_unit, 0.0001)
        self.movement_shoulder_tilt_unit = _min_float(self.movement_shoulder_tilt_unit, 0.0001)
        self.movement_head_tilt_unit = _min_float(self.movement_head_tilt_unit, 0.0001)
        self.movement_head_side_shift_unit = _min_float(self.movement_head_side_shift_unit, 0.0001)

        self.duplicate_notification_cooldown_seconds = _min_float(self.duplicate_notification_cooldown_seconds, 0.0)
        self.telemetry_flush_interval_seconds = _min_float(self.telemetry_flush_interval_seconds, 0.1)
        self.telemetry_sample_interval_seconds = _min_float(self.telemetry_sample_interval_seconds, 0.25)

        self.auto_recalibration_stability_seconds = _min_float(self.auto_recalibration_stability_seconds, 0.0)
        self.auto_recalibration_max_score = _min_float(self.auto_recalibration_max_score, 0.0)
        self.auto_recalibration_min_time_since_baseline_seconds = _min_float(
            self.auto_recalibration_min_time_since_baseline_seconds,
            0.0,
        )
        self.auto_recalibration_cooldown_seconds = _min_float(self.auto_recalibration_cooldown_seconds, 0.0)

        if self.green_zone_max_score >= self.yellow_zone_max_score:
            self.green_zone_max_score = 0.45
            self.yellow_zone_max_score = 1.00


def load_app_config(config_file: Path) -> AppConfig:
    """
    Load configuration from config.json.
    If the file is missing or invalid, return safe defaults.
    """
    if not config_file.exists():
        return AppConfig()

    try:
        raw = json.loads(config_file.read_text(encoding="utf-8"))
    except Exception:
        return AppConfig()

    if not isinstance(raw, dict):
        return AppConfig()

    return AppConfig.from_dict(raw)


def save_app_config(config_file: Path, config: AppConfig) -> None:
    """
    Save configuration to config.json.
    """
    try:
        config.normalize()
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(
            json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _min_float(value: Any, minimum: float) -> float:
    try:
        return max(float(value), float(minimum))
    except Exception:
        return float(minimum)


def _clamp_float(value: Any, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except Exception:
        return float(minimum)

    return max(float(minimum), min(float(maximum), number))
