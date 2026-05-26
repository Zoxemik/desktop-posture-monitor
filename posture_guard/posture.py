from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from statistics import median
from typing import Any, Optional, Sequence

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerOptions

from config import AppConfig


EPSILON = 1e-6


class PoseIndex(IntEnum):
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24


IMPORTANT_LANDMARKS = (
    PoseIndex.LEFT_EAR,
    PoseIndex.RIGHT_EAR,
    PoseIndex.LEFT_SHOULDER,
    PoseIndex.RIGHT_SHOULDER,
    PoseIndex.LEFT_HIP,
    PoseIndex.RIGHT_HIP,
)

POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (27, 31), (28, 32),
)

MOVEMENT_SCORE_UNITS = (
    ("head_forward_signed", "movement_head_forward_unit"),
    ("torso_angle_deg", "movement_torso_angle_unit"),
    ("neck_gap", "movement_neck_gap_unit"),
    ("screen_approach_ratio", "movement_screen_approach_unit"),
    ("shoulder_tilt_deg", "movement_shoulder_tilt_unit"),
    ("head_tilt_deg", "movement_head_tilt_unit"),
    ("head_side_shift", "movement_head_side_shift_unit"),
)

ISSUE_LABELS = {
    "none": "",
    "forward_head": "Head too far forward",
    "torso_lean": "Torso leaning forward",
    "neck_drop": "Neck collapsing",
    "shoulder_tilt": "Shoulders uneven",
    "head_tilt": "Head tilted",
    "screen_approach": "Too close to the screen",
}


@dataclass(frozen=True)
class SmoothedLandmark:
    """
    Lightweight landmark container used for stable drawing and metric calculation.
    """

    x: float
    y: float
    z: float
    visibility: float = 1.0
    presence: float = 1.0


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float


def create_landmarker_options(model_path: str, config: AppConfig) -> PoseLandmarkerOptions:
    """
    Create MediaPipe Pose Landmarker options for video processing.
    """
    return PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=config.min_pose_detection_confidence,
        min_pose_presence_confidence=config.min_pose_presence_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
        output_segmentation_masks=False,
    )


def detect_pose(landmarker: Any, frame: Any, timestamp_ms: int) -> Any:
    """
    Run a single pose detection pass for the current frame.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    return landmarker.detect_for_video(mp_image, timestamp_ms)


def draw_text(
    frame: Any,
    text: str,
    line: int,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """
    Draw readable outlined text on top of the preview frame.
    """
    x = 20
    y = 30 + line * 28

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_pose(frame: Any, normalized_landmarks: Sequence[Any], config: AppConfig) -> None:
    """
    Draw a simplified pose skeleton and the most important joints.
    """
    height, width = frame.shape[:2]

    def get_point(index: int) -> tuple[int, int]:
        landmark = normalized_landmarks[index]
        return int(landmark.x * width), int(landmark.y * height)

    for start_index, end_index in POSE_CONNECTIONS:
        if not _has_landmark(normalized_landmarks, start_index):
            continue
        if not _has_landmark(normalized_landmarks, end_index):
            continue
        if not _is_visible(normalized_landmarks[start_index], config.render_visibility):
            continue
        if not _is_visible(normalized_landmarks[end_index], config.render_visibility):
            continue

        cv2.line(
            frame,
            get_point(start_index),
            get_point(end_index),
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    for index, landmark in enumerate(normalized_landmarks):
        if not _is_visible(landmark, config.render_visibility):
            continue

        radius = 6 if index in IMPORTANT_LANDMARKS else 4
        cv2.circle(
            frame,
            get_point(index),
            radius,
            (0, 200, 255),
            -1,
            cv2.LINE_AA,
        )


def smooth_landmarks(
    previous_landmarks: Optional[Sequence[Any]],
    current_landmarks: Optional[Sequence[Any]],
    alpha: float,
) -> Optional[list[SmoothedLandmark]]:
    """
    Apply EMA smoothing to landmarks.

    This reduces visual skeleton jitter, false movement detection and unstable
    posture metrics caused by small frame-to-frame noise.
    """
    if current_landmarks is None:
        return list(previous_landmarks) if previous_landmarks is not None else None

    alpha = _clamp01(alpha)

    if previous_landmarks is None or len(previous_landmarks) != len(current_landmarks):
        return [_to_smoothed_landmark(landmark) for landmark in current_landmarks]

    return [
        SmoothedLandmark(
            x=_ema(current.x, previous.x, alpha),
            y=_ema(current.y, previous.y, alpha),
            z=_ema(current.z, previous.z, alpha),
            visibility=_ema(_get_visibility(current), _get_visibility(previous), alpha),
            presence=_ema(_get_presence(current), _get_presence(previous), alpha),
        )
        for previous, current in zip(previous_landmarks, current_landmarks)
    ]


def compute_metrics(
    normalized_landmarks: Sequence[Any],
    world_landmarks: Sequence[Any],
    config: AppConfig,
) -> Optional[dict[str, float]]:
    """
    Compute posture metrics used by ergonomic evaluation.

    The goal is not to force one perfect pose. The goal is to keep the user
    inside a calibrated neutral range and detect meaningful drift or static
    overload.
    """
    if not _has_required_landmarks(normalized_landmarks, world_landmarks):
        return None

    if get_average_visibility(normalized_landmarks, IMPORTANT_LANDMARKS) < config.min_visibility:
        return None

    left_ear = world_landmarks[PoseIndex.LEFT_EAR]
    right_ear = world_landmarks[PoseIndex.RIGHT_EAR]
    left_shoulder = world_landmarks[PoseIndex.LEFT_SHOULDER]
    right_shoulder = world_landmarks[PoseIndex.RIGHT_SHOULDER]
    left_hip = world_landmarks[PoseIndex.LEFT_HIP]
    right_hip = world_landmarks[PoseIndex.RIGHT_HIP]

    ear_mid = get_midpoint(left_ear, right_ear)
    shoulder_mid = get_midpoint(left_shoulder, right_shoulder)
    hip_mid = get_midpoint(left_hip, right_hip)

    left_ear_n = normalized_landmarks[PoseIndex.LEFT_EAR]
    right_ear_n = normalized_landmarks[PoseIndex.RIGHT_EAR]
    left_shoulder_n = normalized_landmarks[PoseIndex.LEFT_SHOULDER]
    right_shoulder_n = normalized_landmarks[PoseIndex.RIGHT_SHOULDER]

    torso_angle_deg = _compute_torso_angle_deg(shoulder_mid, hip_mid)
    head_forward_signed = shoulder_mid.z - ear_mid.z
    neck_gap = abs(ear_mid.y - shoulder_mid.y)

    screen_approach_ratio = get_distance_2d(
        (left_shoulder_n.x, left_shoulder_n.y),
        (right_shoulder_n.x, right_shoulder_n.y),
    )
    ear_span_ratio = get_distance_2d(
        (left_ear_n.x, left_ear_n.y),
        (right_ear_n.x, right_ear_n.y),
    )

    shoulder_tilt_deg = get_angle_from_horizontal(
        left_shoulder_n.y - right_shoulder_n.y,
        left_shoulder_n.x - right_shoulder_n.x,
    )
    head_tilt_deg = get_angle_from_horizontal(
        left_ear_n.y - right_ear_n.y,
        left_ear_n.x - right_ear_n.x,
    )

    head_side_shift = abs(ear_mid.x - shoulder_mid.x)
    head_yaw_proxy = _compute_head_yaw_proxy(
        ear_span_ratio=ear_span_ratio,
        shoulder_span_ratio=screen_approach_ratio,
    )

    return {
        "head_forward_signed": head_forward_signed,
        "torso_angle_deg": torso_angle_deg,
        "neck_gap": neck_gap,
        "screen_approach_ratio": screen_approach_ratio,
        "ear_span_ratio": ear_span_ratio,
        "head_yaw_proxy": head_yaw_proxy,
        "shoulder_tilt_deg": shoulder_tilt_deg,
        "head_tilt_deg": head_tilt_deg,
        "head_side_shift": head_side_shift,
    }


def build_baseline(calibration_samples: list[dict[str, float]]) -> dict[str, float]:
    """
    Build a stable user baseline from calibration samples.

    Median is more robust than mean for noisy frames.
    """
    if not calibration_samples:
        raise ValueError("Cannot build baseline from an empty calibration sample list.")

    keys = calibration_samples[0].keys()
    return {key: median(sample[key] for sample in calibration_samples) for key in keys}


def smooth_metrics(
    previous_metrics: Optional[dict[str, float]],
    current_metrics: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    """
    Apply exponential moving average to reduce frame-to-frame jitter.
    """
    if previous_metrics is None:
        return dict(current_metrics)

    alpha = _clamp01(alpha)
    return {
        key: _ema(value, previous_metrics[key], alpha)
        for key, value in current_metrics.items()
    }


def compute_movement_score(
    current_metrics: dict[str, float],
    previous_metrics: Optional[dict[str, float]],
    config: AppConfig,
) -> float:
    """
    Estimate how much posture changed since the previous processed frame.
    """
    if previous_metrics is None:
        return 0.0

    score = 0.0
    for metric_key, config_key in MOVEMENT_SCORE_UNITS:
        current_value = current_metrics[metric_key]
        previous_value = previous_metrics[metric_key]
        unit = max(float(getattr(config, config_key)), EPSILON)
        score += abs(current_value - previous_value) / unit

    return score


def evaluate_ergonomics(
    metrics: dict[str, float],
    baseline: dict[str, float],
    config: AppConfig,
) -> dict[str, object]:
    """
    Evaluate posture against the calibrated baseline.
    """
    deltas = _compute_posture_deltas(metrics, baseline)
    issue_scores = _compute_issue_scores(deltas, config)
    issue_flags = _compute_issue_flags(deltas, config)
    total_score = sum(issue_scores.values())
    zone = _classify_zone(total_score, config)

    dominant_issue_key = "none"
    dominant_issue_label = ""
    if total_score > 0.0:
        dominant_issue_key = max(issue_scores, key=issue_scores.get)
        dominant_issue_label = ISSUE_LABELS.get(dominant_issue_key, "")

    return {
        "zone": zone,
        "total_score": total_score,
        "issue_scores": issue_scores,
        "issue_flags": issue_flags,
        "dominant_issue_key": dominant_issue_key,
        "dominant_issue_label": dominant_issue_label,
        "head_delta": deltas["head_delta"],
        "torso_delta": deltas["torso_delta"],
        "neck_drop": deltas["neck_drop"],
        "shoulder_tilt_delta": deltas["shoulder_tilt_delta"],
        "head_tilt_delta": deltas["head_tilt_delta"],
        "screen_approach_delta": deltas["screen_approach_delta"],
    }


def get_average_visibility(landmarks: Sequence[Any], indices: Sequence[int]) -> float:
    values = [_get_visibility(landmarks[index]) for index in indices]
    return sum(values) / max(len(values), 1)


def get_midpoint(point_a: Any, point_b: Any) -> Point3D:
    return Point3D(
        x=(point_a.x + point_b.x) * 0.5,
        y=(point_a.y + point_b.y) * 0.5,
        z=(point_a.z + point_b.z) * 0.5,
    )


def get_distance_2d(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def get_angle_from_horizontal(delta_y: float, delta_x: float) -> float:
    """
    Return signed tilt angle in degrees relative to a horizontal line.

    Positive and negative values represent opposite tilt directions. This keeps
    baseline comparison meaningful for head and shoulder tilt.
    """
    return math.degrees(math.atan2(delta_y, delta_x + EPSILON))


def _compute_torso_angle_deg(shoulder_mid: Point3D, hip_mid: Point3D) -> float:
    torso_delta_y = shoulder_mid.y - hip_mid.y
    torso_delta_z = shoulder_mid.z - hip_mid.z
    return math.degrees(math.atan2(abs(torso_delta_z), abs(torso_delta_y) + EPSILON))


def _compute_head_yaw_proxy(ear_span_ratio: float, shoulder_span_ratio: float) -> float:
    reference_width = max(shoulder_span_ratio, EPSILON)
    normalized_ear_span = min(1.0, ear_span_ratio / reference_width)
    return max(0.0, 1.0 - normalized_ear_span)


def _compute_posture_deltas(
    metrics: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float]:
    return {
        "head_delta": metrics["head_forward_signed"] - baseline["head_forward_signed"],
        "torso_delta": metrics["torso_angle_deg"] - baseline["torso_angle_deg"],
        "neck_drop": baseline["neck_gap"] - metrics["neck_gap"],
        "shoulder_tilt_delta": metrics["shoulder_tilt_deg"] - baseline["shoulder_tilt_deg"],
        "head_tilt_delta": metrics["head_tilt_deg"] - baseline["head_tilt_deg"],
        "screen_approach_delta": metrics["screen_approach_ratio"] - baseline["screen_approach_ratio"],
    }


def _compute_issue_scores(deltas: dict[str, float], config: AppConfig) -> dict[str, float]:
    return {
        "forward_head": _weighted_positive_score(
            deltas["head_delta"],
            config.head_forward_delta_m,
            config.weight_forward_head,
        ),
        "torso_lean": _weighted_positive_score(
            deltas["torso_delta"],
            config.torso_angle_delta_deg,
            config.weight_torso_lean,
        ),
        "neck_drop": _weighted_positive_score(
            deltas["neck_drop"],
            config.neck_drop_delta_m,
            config.weight_neck_drop,
        ),
        "shoulder_tilt": _weighted_absolute_score(
            deltas["shoulder_tilt_delta"],
            config.shoulder_tilt_delta_deg,
            config.weight_shoulder_tilt,
        ),
        "head_tilt": _weighted_absolute_score(
            deltas["head_tilt_delta"],
            config.head_tilt_delta_deg,
            config.weight_head_tilt,
        ),
        "screen_approach": _weighted_positive_score(
            deltas["screen_approach_delta"],
            config.screen_approach_delta,
            config.weight_screen_approach,
        ),
    }


def _compute_issue_flags(deltas: dict[str, float], config: AppConfig) -> dict[str, bool]:
    return {
        "forward_head": deltas["head_delta"] >= config.head_forward_delta_m,
        "torso_lean": deltas["torso_delta"] >= config.torso_angle_delta_deg,
        "neck_drop": deltas["neck_drop"] >= config.neck_drop_delta_m,
        "shoulder_tilt": abs(deltas["shoulder_tilt_delta"]) >= config.shoulder_tilt_delta_deg,
        "head_tilt": abs(deltas["head_tilt_delta"]) >= config.head_tilt_delta_deg,
        "screen_approach": deltas["screen_approach_delta"] >= config.screen_approach_delta,
    }


def _weighted_positive_score(delta: float, threshold: float, weight: float) -> float:
    safe_threshold = max(float(threshold), EPSILON)
    return max(0.0, delta / safe_threshold) * weight

def _weighted_absolute_score(delta: float, threshold: float, weight: float) -> float:
    safe_threshold = max(float(threshold), EPSILON)
    return abs(delta) / safe_threshold * weight

def _classify_zone(total_score: float, config: AppConfig) -> str:
    if total_score < config.green_zone_max_score:
        return "green"
    if total_score < config.yellow_zone_max_score:
        return "yellow"
    return "red"


def _has_required_landmarks(normalized_landmarks: Sequence[Any], world_landmarks: Sequence[Any]) -> bool:
    required_indices = [int(index) for index in IMPORTANT_LANDMARKS]
    required_count = max(required_indices) + 1
    return len(normalized_landmarks) >= required_count and len(world_landmarks) >= required_count


def _has_landmark(landmarks: Sequence[Any], index: int) -> bool:
    return 0 <= index < len(landmarks)


def _is_visible(landmark: Any, minimum_visibility: float) -> bool:
    return _get_visibility(landmark) >= minimum_visibility


def _get_visibility(landmark: Any) -> float:
    return float(getattr(landmark, "visibility", 1.0))


def _get_presence(landmark: Any) -> float:
    return float(getattr(landmark, "presence", 1.0))


def _to_smoothed_landmark(landmark: Any) -> SmoothedLandmark:
    return SmoothedLandmark(
        x=float(landmark.x),
        y=float(landmark.y),
        z=float(landmark.z),
        visibility=_get_visibility(landmark),
        presence=_get_presence(landmark),
    )


def _ema(current: float, previous: float, alpha: float) -> float:
    return (alpha * float(current)) + ((1.0 - alpha) * float(previous))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
