from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MonitoringSnapshot:
    monitoring_active: bool
    paused: bool
    preview_enabled: bool
    calibrated: bool
    pose_detected: bool
    reliable_pose: bool
    status_label: str
    info_line: str
    zone: str
    dominant_issue: str
    total_score: float
    static_duration: float
    bad_duration: float
    movement_score: float
    head_delta: float
    torso_delta: float
    neck_drop: float
    shoulder_tilt_delta: float
    head_tilt_delta: float
    screen_approach_delta: float
    posture_alert_count: int
    stillness_alert_count: int
    reposition_count: int
    muted_until_monotonic: float
    loop_latency_ms: float
    baseline_generation: int
    recalibration_reason: str
    camera_label: str


@dataclass
class AlertEvent:
    kind: str
    title: str
    message: str


@dataclass
class AlertPayload:
    title: str
    message: str
