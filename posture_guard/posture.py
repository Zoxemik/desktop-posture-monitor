from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerOptions

from config import AppConfig


IMPORTANT_LANDMARKS = (7, 8, 11, 12, 23, 24)

POSE_CONNECTIONS = [
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
]


@dataclass
class SmoothedLandmark:
    """
    Lightweight landmark container used for stable drawing and metric calculation.
    """
    x: float
    y: float
    z: float
    visibility: float = 1.0
    presence: float = 1.0


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


def detect_pose(landmarker, frame, timestamp_ms: int):
    """
    Run a single pose detection pass for the current frame.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    return landmarker.detect_for_video(mp_image, timestamp_ms)


def draw_text(frame, text: str, line: int, color=(255, 255, 255)) -> None:
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


def draw_pose(frame, normalized_landmarks, config: AppConfig) -> None:
    """
    Draw a simplified pose skeleton and the most important joints.
    """
    height, width = frame.shape[:2]

    def get_point(index: int) -> Tuple[int, int]:
        landmark = normalized_landmarks[index]
        return int(landmark.x * width), int(landmark.y * height)

    for start_index, end_index in POSE_CONNECTIONS:
        if start_index >= len(normalized_landmarks) or end_index >= len(normalized_landmarks):
            continue

        if normalized_landmarks[start_index].visibility < config.render_visibility:
            continue

        if normalized_landmarks[end_index].visibility < config.render_visibility:
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
        if landmark.visibility < config.render_visibility:
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


def get_average_visibility(landmarks, indices: Sequence[int]) -> float:
    values = [landmarks[index].visibility for index in indices]
    return sum(values) / len(values)


def get_midpoint(point_a, point_b) -> Tuple[float, float, float]:
    return (
        (point_a.x + point_b.x) * 0.5,
        (point_a.y + point_b.y) * 0.5,
        (point_a.z + point_b.z) * 0.5,
    )


def get_distance_2d(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def get_angle_from_horizontal(delta_y: float, delta_x: float) -> float:
    """
    Return an absolute tilt angle in degrees relative to a horizontal line.
    """
    return math.degrees(math.atan2(abs(delta_y), abs(delta_x) + 1e-6))


def smooth_landmarks(previous_landmarks, current_landmarks, alpha: float):
    """
    Apply EMA smoothing to landmarks.

    This helps with:
    - visual skeleton jitter,
    - false movement detection,
    - unstable posture metrics from small frame-to-frame noise.
    """
    if current_landmarks is None:
        return previous_landmarks

    if previous_landmarks is None or len(previous_landmarks) != len(current_landmarks):
        return [
            SmoothedLandmark(
                x=float(landmark.x),
                y=float(landmark.y),
                z=float(landmark.z),
                visibility=float(getattr(landmark, "visibility", 1.0)),
                presence=float(getattr(landmark, "presence", 1.0)),
            )
            for landmark in current_landmarks
        ]

    alpha = max(0.0, min(1.0, float(alpha)))
    smoothed = []

    for previous, current in zip(previous_landmarks, current_landmarks):
        smoothed.append(
            SmoothedLandmark(
                x=(alpha * float(current.x)) + ((1.0 - alpha) * float(previous.x)),
                y=(alpha * float(current.y)) + ((1.0 - alpha) * float(previous.y)),
                z=(alpha * float(current.z)) + ((1.0 - alpha) * float(previous.z)),
                visibility=(alpha * float(getattr(current, "visibility", 1.0))) + ((1.0 - alpha) * float(previous.visibility)),
                presence=(alpha * float(getattr(current, "presence", 1.0))) + ((1.0 - alpha) * float(previous.presence)),
            )
        )

    return smoothed


def compute_metrics(normalized_landmarks, world_landmarks, config: AppConfig) -> Optional[Dict[str, float]]:
    """
    Compute the posture metrics that drive ergonomic evaluation.

    The goal is not to force one perfect pose.
    The goal is to keep the user inside a neutral calibrated range
    and detect meaningful drift or static overload.
    """
    if get_average_visibility(normalized_landmarks, IMPORTANT_LANDMARKS) < config.min_visibility:
        return None

    left_ear = world_landmarks[7]
    right_ear = world_landmarks[8]
    left_shoulder = world_landmarks[11]
    right_shoulder = world_landmarks[12]
    left_hip = world_landmarks[23]
    right_hip = world_landmarks[24]

    ear_mid = get_midpoint(left_ear, right_ear)
    shoulder_mid = get_midpoint(left_shoulder, right_shoulder)
    hip_mid = get_midpoint(left_hip, right_hip)

    left_ear_n = normalized_landmarks[7]
    right_ear_n = normalized_landmarks[8]
    left_shoulder_n = normalized_landmarks[11]
    right_shoulder_n = normalized_landmarks[12]

    torso_delta_y = shoulder_mid[1] - hip_mid[1]
    torso_delta_z = shoulder_mid[2] - hip_mid[2]

    torso_angle_deg = math.degrees(math.atan2(abs(torso_delta_z), abs(torso_delta_y) + 1e-6))
    head_forward_abs = abs(ear_mid[2] - shoulder_mid[2])
    neck_gap = abs(ear_mid[1] - shoulder_mid[1])

    shoulder_width = get_distance_2d(
        (left_shoulder_n.x, left_shoulder_n.y),
        (right_shoulder_n.x, right_shoulder_n.y),
    )
    ear_width = get_distance_2d(
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

    head_side_shift = abs(ear_mid[0] - shoulder_mid[0])

    return {
        "head_forward_abs": head_forward_abs,
        "torso_angle_deg": torso_angle_deg,
        "neck_gap": neck_gap,
        "shoulder_width": shoulder_width,
        "ear_width": ear_width,
        "shoulder_tilt_deg": shoulder_tilt_deg,
        "head_tilt_deg": head_tilt_deg,
        "head_side_shift": head_side_shift,
    }


def build_baseline(calibration_samples: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Build a stable user baseline from calibration samples.
    Median is more robust than mean for noisy frames.
    """
    keys = calibration_samples[0].keys()
    return {key: median(sample[key] for sample in calibration_samples) for key in keys}


def smooth_metrics(
    previous_metrics: Optional[Dict[str, float]],
    current_metrics: Dict[str, float],
    alpha: float,
) -> Dict[str, float]:
    """
    Apply exponential moving average to reduce frame-to-frame jitter.
    """
    if previous_metrics is None:
        return dict(current_metrics)

    smoothed: Dict[str, float] = {}
    for key, value in current_metrics.items():
        smoothed[key] = (alpha * value) + ((1.0 - alpha) * previous_metrics[key])

    return smoothed


def compute_movement_score(
    current_metrics: Dict[str, float],
    previous_metrics: Optional[Dict[str, float]],
    config: AppConfig,
) -> float:
    """
    Estimate how much the posture changed since the previous frame.

    The raw score is intentionally interpretable.
    Additional anti-jitter filtering is applied in the monitoring engine.
    """
    if previous_metrics is None:
        return 0.0

    score = 0.0
    score += abs(current_metrics["head_forward_abs"] - previous_metrics["head_forward_abs"]) / config.movement_head_forward_unit
    score += abs(current_metrics["torso_angle_deg"] - previous_metrics["torso_angle_deg"]) / config.movement_torso_angle_unit
    score += abs(current_metrics["neck_gap"] - previous_metrics["neck_gap"]) / config.movement_neck_gap_unit
    score += abs(current_metrics["shoulder_width"] - previous_metrics["shoulder_width"]) / config.movement_shoulder_width_unit
    score += abs(current_metrics["shoulder_tilt_deg"] - previous_metrics["shoulder_tilt_deg"]) / config.movement_shoulder_tilt_unit
    score += abs(current_metrics["head_tilt_deg"] - previous_metrics["head_tilt_deg"]) / config.movement_head_tilt_unit
    score += abs(current_metrics["head_side_shift"] - previous_metrics["head_side_shift"]) / config.movement_head_side_shift_unit

    return score


def evaluate_ergonomics(metrics: Dict[str, float], baseline: Dict[str, float], config: AppConfig) -> Dict[str, object]:
    """
    Evaluate posture against the calibrated baseline.
    """
    head_forward_delta = metrics["head_forward_abs"] - baseline["head_forward_abs"]
    torso_delta = metrics["torso_angle_deg"] - baseline["torso_angle_deg"]
    neck_drop = baseline["neck_gap"] - metrics["neck_gap"]
    shoulder_tilt_delta = metrics["shoulder_tilt_deg"] - baseline["shoulder_tilt_deg"]
    head_tilt_delta = metrics["head_tilt_deg"] - baseline["head_tilt_deg"]
    screen_approach_delta = metrics["shoulder_width"] - baseline["shoulder_width"]

    issue_scores = {
        "forward_head": max(0.0, head_forward_delta / config.head_forward_delta_m) * config.weight_forward_head,
        "torso_lean": max(0.0, torso_delta / config.torso_angle_delta_deg) * config.weight_torso_lean,
        "neck_drop": max(0.0, neck_drop / config.neck_drop_delta_m) * config.weight_neck_drop,
        "shoulder_tilt": max(0.0, shoulder_tilt_delta / config.shoulder_tilt_delta_deg) * config.weight_shoulder_tilt,
        "head_tilt": max(0.0, head_tilt_delta / config.head_tilt_delta_deg) * config.weight_head_tilt,
        "screen_approach": max(0.0, screen_approach_delta / config.screen_approach_delta) * config.weight_screen_approach,
    }

    total_score = sum(issue_scores.values())

    if total_score < config.green_zone_max_score:
        zone = "green"
    elif total_score < config.yellow_zone_max_score:
        zone = "yellow"
    else:
        zone = "red"

    issue_labels = {
        "forward_head": "Head too far forward",
        "torso_lean": "Torso leaning forward",
        "neck_drop": "Neck collapsing",
        "shoulder_tilt": "Shoulders uneven",
        "head_tilt": "Head tilted",
        "screen_approach": "Too close to the screen",
    }

    dominant_issue_key = max(issue_scores, key=issue_scores.get)
    dominant_issue_label = issue_labels[dominant_issue_key]

    issue_flags = {
        "forward_head": head_forward_delta >= config.head_forward_delta_m,
        "torso_lean": torso_delta >= config.torso_angle_delta_deg,
        "neck_drop": neck_drop >= config.neck_drop_delta_m,
        "shoulder_tilt": shoulder_tilt_delta >= config.shoulder_tilt_delta_deg,
        "head_tilt": head_tilt_delta >= config.head_tilt_delta_deg,
        "screen_approach": screen_approach_delta >= config.screen_approach_delta,
    }

    return {
        "zone": zone,
        "total_score": total_score,
        "issue_scores": issue_scores,
        "issue_flags": issue_flags,
        "dominant_issue_key": dominant_issue_key,
        "dominant_issue_label": dominant_issue_label,
        "head_delta": head_forward_delta,
        "torso_delta": torso_delta,
        "neck_drop": neck_drop,
        "shoulder_tilt_delta": shoulder_tilt_delta,
        "head_tilt_delta": head_tilt_delta,
        "screen_approach_delta": screen_approach_delta,
    }