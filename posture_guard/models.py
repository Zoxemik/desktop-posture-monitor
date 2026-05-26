from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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


@dataclass(frozen=True)
class AlertEvent:
    kind: str
    title: str
    message: str


@dataclass(frozen=True)
class AlertPayload:
    title: str
    message: str


@dataclass(frozen=True)
class NotificationQueueResult:
    key: str
    accepted: bool
    reason: str = ""


@dataclass(frozen=True)
class NotificationDeliveryResult:
    key: str
    title: str
    message: str
    queued: bool
    toast_backend_accepted: bool
    sound_played: bool
    fallback_printed: bool
    error_message: Optional[str] = None

    @property
    def user_visible(self) -> bool:
        return self.toast_backend_accepted or self.sound_played or self.fallback_printed