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

    # Resource and data file paths.
    model_path: str = "models/pose_landmarker_full.task"
    stats_path: str = "data/stats.json"
    config_path: str = "data/config.json"
    baseline_path: str = "data/baseline.json"

    # Startup and runtime behavior.
    preview_enabled: bool = False
    start_paused: bool = False
    tray_enabled: bool = True
    sound_enabled: bool = True
    toast_notifications_enabled: bool = True
    overlay_enabled: bool = True
    persist_baseline_between_runs: bool = True

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

    # Static load policy.
    static_load_grace_seconds: float = 120.0

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
    movement_shoulder_width_unit: float = 0.025
    movement_shoulder_tilt_unit: float = 4.0
    movement_head_tilt_unit: float = 4.0
    movement_head_side_shift_unit: float = 0.010

    # Sound configuration.
    # This flag now means:
    # - True: allow the operating system notification sound
    # - False: keep notifications visually silent
    sound_enabled: bool = True

    # Corner overlay.
    overlay_size_px: int = 18
    overlay_margin_px: int = 14
    overlay_alpha: float = 0.90
    overlay_flash_seconds: float = 4.0

    # Notification throttling.
    duplicate_notification_cooldown_seconds: float = 10.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        defaults = asdict(cls())
        defaults.update({key: value for key, value in data.items() if key in defaults})
        return cls(**defaults)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_parent_directory(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)


def read_json_file(file_path: Path, default_value: Any) -> Any:
    if not file_path.exists():
        return default_value

    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return default_value


def write_json_file(file_path: Path, payload: Any) -> None:
    ensure_parent_directory(file_path)
    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_app_config(config_file: Path) -> AppConfig:
    """
    Load the user config if it exists, otherwise return defaults.
    """
    raw = read_json_file(config_file, {})
    if not isinstance(raw, dict):
        return AppConfig()

    return AppConfig.from_dict(raw)


def save_app_config(config_file: Path, config: AppConfig) -> None:
    """
    Persist the application settings to disk.
    """
    write_json_file(config_file, config.to_dict())