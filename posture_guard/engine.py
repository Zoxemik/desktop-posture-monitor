from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
from mediapipe.tasks.python import vision

from camera import open_camera, read_camera_frame, safe_release_camera
from config import AppConfig
from models import AlertEvent, MonitoringSnapshot, NotificationDeliveryResult, NotificationQueueResult
from paths import get_resource_path
from posture import build_baseline
from posture import compute_metrics
from posture import compute_movement_score
from posture import create_landmarker_options
from posture import detect_pose
from posture import draw_pose
from posture import draw_text
from posture import evaluate_ergonomics
from posture import smooth_landmarks
from posture import smooth_metrics
from telemetry import TelemetryLogger

logger = logging.getLogger(__name__)

AlertCallback = Callable[..., object]


@dataclass
class RuntimeState:
    calibration_samples: list[dict[str, float]] = field(default_factory=list)
    baseline: Optional[dict[str, float]] = None
    smoothed_metrics: Optional[dict[str, float]] = None
    previous_metrics: Optional[dict[str, float]] = None
    smoothed_normalized_landmarks: Optional[list[Any]] = None
    smoothed_world_landmarks: Optional[list[Any]] = None
    latest_metrics: Optional[dict[str, float]] = None
    latest_evaluation: Optional[dict[str, object]] = None
    latest_pose_detected: bool = False
    latest_reliable_pose: bool = False
    calibration_start: float = 0.0
    bad_posture_start: Optional[float] = None
    last_posture_alert_time: float = -10000.0
    last_stillness_alert_time: float = -10000.0
    last_movement_time: Optional[float] = None
    last_reposition_time: Optional[float] = None
    last_movement_score: float = 0.0
    last_raw_movement_score: float = 0.0
    reposition_count: int = 0
    movement_refresh_streak: int = 0
    movement_reposition_streak: int = 0
    filtered_movement_score: float = 0.0
    baseline_generation: int = 0
    baseline_established_at: Optional[float] = None
    auto_recalibration_pending: bool = False
    auto_recalibration_candidate_since: Optional[float] = None
    last_auto_recalibration_time: Optional[float] = None
    last_recalibration_reason: str = "initial_calibration"
    last_logged_zone: Optional[str] = None
    frame_index: int = 0
    last_stream_timestamp_ms: int = 0
    loop_latency_ms: float = 0.0
    posture_alert_fired: bool = False
    stillness_alert_fired: bool = False
    auto_recalibration_started: bool = False
    camera_read_failures: int = 0
    last_camera_reopen_attempt: float = 0.0


@dataclass(frozen=True)
class CalibratedPostureResult:
    status_label: str
    info_line: str
    zone: str
    dominant_issue: str
    total_score: float
    static_duration: float
    head_delta: float
    torso_delta: float
    neck_drop: float
    shoulder_tilt_delta: float
    head_tilt_delta: float
    screen_approach_delta: float
    movement_score: float
    alert_event: Optional[AlertEvent]
    posture_alert_count: int
    stillness_alert_count: int


class MonitoringEngine:
    """
    Background monitoring engine.

    It owns the camera, MediaPipe processing, calibration logic,
    posture evaluation, telemetry logging and alert scheduling.
    """

    def __init__(
        self,
        config: AppConfig,
        data_dir: Path,
        on_snapshot: Optional[Callable[[MonitoringSnapshot], None]] = None,
        on_alert: Optional[AlertCallback] = None,
    ) -> None:
        self._config = config
        self._data_dir = data_dir
        self._on_snapshot = on_snapshot
        self._on_alert = on_alert

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._lock = threading.RLock()
        self._telemetry_lock = threading.RLock()

        self._preview_enabled = config.preview_enabled
        self._recalibration_requested = False
        self._requested_recalibration_reason: Optional[str] = None
        self._muted_until_monotonic = 0.0
        self._latest_snapshot = self._create_initial_snapshot()

        self._latest_preview_frame = None
        self._preview_frame_lock = threading.Lock()
        self._fatal_error_message: Optional[str] = None
        self._camera_label = "Camera not opened yet"

        self._telemetry = TelemetryLogger(
            data_dir=data_dir,
            app_name=config.app_name,
            enabled=config.telemetry_enabled,
            flush_interval_seconds=config.telemetry_flush_interval_seconds,
            sample_interval_seconds=config.telemetry_sample_interval_seconds,
            real_bad_posture_min_seconds=config.posture_alert_after_seconds,
        )
        self._telemetry.write_session_metadata(
            config_snapshot=config.to_dict(),
            extra={"data_dir": str(data_dir)},
        )

        if config.start_paused:
            self._pause_event.set()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._fatal_error_message = None
        self._thread = threading.Thread(target=self._run_loop, name="MonitoringEngine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

        if self._thread is not None and threading.current_thread() is not self._thread:
            timeout = max(0.5, float(self._config.shutdown_join_timeout_seconds))
            self._thread.join(timeout=timeout)

            if self._thread.is_alive():
                self._fatal_error_message = "Monitoring engine shutdown timed out."
                logger.error("Monitoring engine did not stop within %.1f seconds", timeout)
                return

        self.clear_preview_frame()
        self._destroy_preview_window()

        with self._telemetry_lock:
            self._telemetry.close()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_fatal_error(self) -> Optional[str]:
        return self._fatal_error_message

    def pause(self) -> None:
        self._pause_event.set()

    def resume(self) -> None:
        self._pause_event.clear()

    def toggle_preview(self) -> None:
        with self._lock:
            self._preview_enabled = not self._preview_enabled
            preview_enabled = self._preview_enabled

        if not preview_enabled:
            self.clear_preview_frame()

    def hide_preview(self) -> None:
        with self._lock:
            self._preview_enabled = False
        self.clear_preview_frame()

    def request_recalibration(self) -> None:
        with self._lock:
            self._recalibration_requested = True
            self._requested_recalibration_reason = "manual_request"

    def mute_for_seconds(self, seconds: float) -> None:
        with self._lock:
            self._muted_until_monotonic = max(
                self._muted_until_monotonic,
                time.monotonic() + max(0.0, seconds),
            )

    def get_latest_snapshot(self) -> MonitoringSnapshot:
        with self._lock:
            return self._latest_snapshot

    def get_latest_preview_frame(self):
        with self._preview_frame_lock:
            if self._latest_preview_frame is None:
                return None
            return self._latest_preview_frame.copy()

    def update_preview_frame(self, frame) -> None:
        with self._preview_frame_lock:
            self._latest_preview_frame = frame.copy()

    def clear_preview_frame(self) -> None:
        with self._preview_frame_lock:
            self._latest_preview_frame = None

    def handle_preview_key(self, key: int) -> None:
        if key == ord("r"):
            with self._lock:
                self._recalibration_requested = True
                self._requested_recalibration_reason = "manual_request"
        elif key in (ord("p"), ord("q"), 27):
            self.hide_preview()

    def _create_initial_snapshot(self) -> MonitoringSnapshot:
        return MonitoringSnapshot(
            monitoring_active=False,
            paused=self._config.start_paused,
            preview_enabled=self._config.preview_enabled,
            calibrated=False,
            pose_detected=False,
            reliable_pose=False,
            status_label="Starting",
            info_line="",
            zone="unknown",
            dominant_issue="",
            total_score=0.0,
            static_duration=0.0,
            bad_duration=0.0,
            movement_score=0.0,
            head_delta=0.0,
            torso_delta=0.0,
            neck_drop=0.0,
            shoulder_tilt_delta=0.0,
            head_tilt_delta=0.0,
            screen_approach_delta=0.0,
            posture_alert_count=0,
            stillness_alert_count=0,
            reposition_count=0,
            muted_until_monotonic=0.0,
            loop_latency_ms=0.0,
            baseline_generation=0,
            recalibration_reason="initial_calibration",
            camera_label="Camera not opened yet",
        )

    def _create_runtime_state(self) -> RuntimeState:
        return RuntimeState(calibration_start=time.monotonic())

    def _reset_for_recalibration(self, state: RuntimeState, reason: str) -> None:
        now = time.monotonic()
        state.calibration_samples.clear()
        state.baseline = None
        state.smoothed_metrics = None
        state.previous_metrics = None
        state.smoothed_normalized_landmarks = None
        state.smoothed_world_landmarks = None
        state.latest_metrics = None
        state.latest_evaluation = None
        state.latest_pose_detected = False
        state.latest_reliable_pose = False
        state.calibration_start = now
        state.bad_posture_start = None
        state.last_movement_time = None
        state.last_movement_score = 0.0
        state.last_raw_movement_score = 0.0
        state.movement_refresh_streak = 0
        state.movement_reposition_streak = 0
        state.filtered_movement_score = 0.0
        state.auto_recalibration_pending = False
        state.auto_recalibration_candidate_since = None
        state.posture_alert_fired = False
        state.stillness_alert_fired = False
        state.auto_recalibration_started = False
        state.last_logged_zone = None
        state.last_recalibration_reason = reason

        self._log_event(
            kind="recalibration_started",
            title="Recalibration",
            message=reason,
            state=state,
            zone="calibration",
            dominant_issue="",
            total_score=0.0,
        )

    def _run_loop(self) -> None:
        capture: Optional[cv2.VideoCapture] = None

        try:
            model_file = get_resource_path(self._config.model_path)
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")

            state = self._create_runtime_state()
            capture = self._open_camera_for_state(state)

            options = create_landmarker_options(str(model_file), self._config)

            stream_start = time.monotonic()
            last_loop_time = time.monotonic()
            last_processed_frame_time = 0.0
            min_frame_interval = 1.0 / self._config.inference_fps if self._config.inference_fps > 0 else 0.0

            posture_alert_count = 0
            stillness_alert_count = 0

            with vision.PoseLandmarker.create_from_options(options) as landmarker:
                while not self._stop_event.is_set():
                    if self._pause_event.is_set():
                        self._handle_paused_state(state, posture_alert_count, stillness_alert_count)
                        continue

                    now = time.monotonic()
                    if min_frame_interval > 0.0 and (now - last_processed_frame_time) < min_frame_interval:
                        time.sleep(0.002)
                        continue

                    capture = self._ensure_camera_ready(capture, state)
                    if capture is None:
                        time.sleep(0.05)
                        continue

                    read_result = read_camera_frame(capture)
                    if not read_result.ok:
                        capture = self._handle_camera_read_failure(capture, state, read_result.error)
                        continue

                    if state.camera_read_failures > 0:
                        logger.info("Camera recovered after %s failed reads", state.camera_read_failures)
                        state.camera_read_failures = 0

                    frame = cv2.flip(read_result.frame, 1)

                    now = time.monotonic()
                    delta_time = now - last_loop_time
                    last_loop_time = now
                    last_processed_frame_time = now
                    timestamp_ms = int((now - stream_start) * 1000)

                    with self._lock:
                        preview_enabled = self._preview_enabled
                        if self._recalibration_requested:
                            reason = self._requested_recalibration_reason or "manual_request"
                            self._reset_for_recalibration(state, reason)
                            self._recalibration_requested = False
                            self._requested_recalibration_reason = None

                    state.frame_index += 1
                    state.last_stream_timestamp_ms = timestamp_ms

                    processing_start = time.monotonic()
                    result = detect_pose(landmarker, frame, timestamp_ms)

                    snapshot, alert_event = self._process_frame(
                        result=result,
                        now=now,
                        delta_time=delta_time,
                        state=state,
                        posture_alert_count=posture_alert_count,
                        stillness_alert_count=stillness_alert_count,
                        preview_enabled=preview_enabled,
                    )

                    processing_latency_ms = (time.monotonic() - processing_start) * 1000.0
                    state.loop_latency_ms = processing_latency_ms
                    snapshot = replace(snapshot, loop_latency_ms=processing_latency_ms)

                    posture_alert_count = snapshot.posture_alert_count
                    stillness_alert_count = snapshot.stillness_alert_count

                    self._emit_snapshot(snapshot)
                    self._log_frame_telemetry(snapshot, state)

                    if alert_event is not None:
                        if time.monotonic() >= self._get_muted_until():
                            self._emit_alert(alert_event, snapshot, state)
                        else:
                            self._log_suppressed_alert(alert_event, snapshot, state, "muted")

                    if preview_enabled:
                        rendered_frame = self._build_preview_frame(frame, snapshot, state)
                        self.update_preview_frame(rendered_frame)
                    else:
                        self.clear_preview_frame()

        except Exception as exc:
            self._fatal_error_message = str(exc)
            logger.exception("Monitoring engine failed")
            self._log_startup_error(str(exc))
            self._emit_snapshot(self._create_error_snapshot(str(exc)))

        finally:
            safe_release_camera(capture)
            self.clear_preview_frame()

    def _open_camera_for_state(self, state: RuntimeState) -> cv2.VideoCapture:
        camera_result = open_camera(
            self._config.camera_index,
            self._config.frame_width,
            self._config.frame_height,
        )
        self._camera_label = (
            f"Camera index {camera_result.camera_index} "
            f"({camera_result.backend_name}, {camera_result.actual_width}x{camera_result.actual_height})"
        )
        state.camera_read_failures = 0
        state.last_camera_reopen_attempt = time.monotonic()

        self._log_event(
            kind="camera_opened",
            title="Camera opened",
            message=self._camera_label,
            state=state,
            zone="startup",
            dominant_issue="",
            total_score=0.0,
        )
        return camera_result.capture

    def _ensure_camera_ready(
        self,
        capture: Optional[cv2.VideoCapture],
        state: RuntimeState,
    ) -> Optional[cv2.VideoCapture]:
        if capture is not None:
            return capture

        now = time.monotonic()
        if (now - state.last_camera_reopen_attempt) < self._config.camera_reopen_cooldown_seconds:
            return None

        try:
            logger.info("Trying to reopen camera")
            return self._open_camera_for_state(state)
        except Exception as exc:
            state.last_camera_reopen_attempt = now
            state.camera_read_failures += 1
            logger.warning("Camera reopen failed: %s", exc)

            if state.camera_read_failures >= self._config.camera_read_failure_fatal_threshold:
                raise RuntimeError(f"Camera could not be reopened: {exc}") from exc

            return None

    def _handle_camera_read_failure(
        self,
        capture: Optional[cv2.VideoCapture],
        state: RuntimeState,
        error: str,
    ) -> Optional[cv2.VideoCapture]:
        state.camera_read_failures += 1
        failure_count = state.camera_read_failures

        if failure_count == self._config.camera_read_failure_warning_threshold:
            logger.warning("Camera read failures reached warning threshold: %s", failure_count)
            self._log_event(
                kind="camera_read_warning",
                title="Camera warning",
                message=error or "camera read failures",
                state=state,
                zone="camera_warning",
                dominant_issue="",
                total_score=0.0,
            )

        if failure_count >= self._config.camera_read_failure_fatal_threshold:
            raise RuntimeError(f"Camera stopped returning frames after {failure_count} attempts. Last error: {error}")

        now = time.monotonic()
        should_reopen = failure_count >= self._config.camera_read_failure_reopen_threshold
        cooldown_elapsed = (now - state.last_camera_reopen_attempt) >= self._config.camera_reopen_cooldown_seconds

        if should_reopen and cooldown_elapsed:
            logger.warning("Reopening camera after %s failed reads", failure_count)
            safe_release_camera(capture)
            state.last_camera_reopen_attempt = now

            try:
                reopened_capture = self._open_camera_for_state(state)
                self._reset_for_recalibration(state, "camera_reopened")
                return reopened_capture
            except Exception as exc:
                logger.exception("Camera reopen attempt failed")
                state.camera_read_failures = failure_count + 1
                if state.camera_read_failures >= self._config.camera_read_failure_fatal_threshold:
                    raise RuntimeError(f"Camera reopen failed: {exc}") from exc
                return None

        time.sleep(0.02)
        return capture

    def _handle_paused_state(self, state: RuntimeState, posture_alert_count: int, stillness_alert_count: int) -> None:
        state.latest_metrics = None
        state.latest_evaluation = None
        state.latest_pose_detected = False
        state.latest_reliable_pose = False

        snapshot = MonitoringSnapshot(
            monitoring_active=True,
            paused=True,
            preview_enabled=self._preview_enabled,
            calibrated=state.baseline is not None,
            pose_detected=False,
            reliable_pose=False,
            status_label="Paused",
            info_line="Monitoring is paused from the tray menu.",
            zone="paused",
            dominant_issue="",
            total_score=0.0,
            static_duration=0.0,
            bad_duration=0.0,
            movement_score=0.0,
            head_delta=0.0,
            torso_delta=0.0,
            neck_drop=0.0,
            shoulder_tilt_delta=0.0,
            head_tilt_delta=0.0,
            screen_approach_delta=0.0,
            posture_alert_count=posture_alert_count,
            stillness_alert_count=stillness_alert_count,
            reposition_count=state.reposition_count,
            muted_until_monotonic=self._get_muted_until(),
            loop_latency_ms=0.0,
            baseline_generation=state.baseline_generation,
            recalibration_reason=state.last_recalibration_reason,
            camera_label=self._camera_label,
        )

        self._emit_snapshot(snapshot)
        self._log_frame_telemetry(snapshot, state)
        self.clear_preview_frame()
        time.sleep(0.15)

    def _process_frame(
        self,
        result,
        now: float,
        delta_time: float,
        state: RuntimeState,
        posture_alert_count: int,
        stillness_alert_count: int,
        preview_enabled: bool,
    ) -> tuple[MonitoringSnapshot, Optional[AlertEvent]]:
        status_label = "No person detected"
        info_line = "Adjust your position so the upper body is visible."
        alert_event: Optional[AlertEvent] = None

        total_score = 0.0
        static_duration = 0.0
        head_delta = 0.0
        torso_delta = 0.0
        neck_drop = 0.0
        shoulder_tilt_delta = 0.0
        head_tilt_delta = 0.0
        screen_approach_delta = 0.0
        movement_score = state.last_movement_score
        dominant_issue = ""
        zone = "no_person"
        pose_detected = False
        reliable_pose = False

        state.latest_metrics = None
        state.latest_evaluation = None
        state.latest_pose_detected = False
        state.latest_reliable_pose = False
        state.posture_alert_fired = False
        state.stillness_alert_fired = False
        state.auto_recalibration_started = False

        if result.pose_landmarks and result.pose_world_landmarks:
            pose_detected = True
            raw_normalized_landmarks = result.pose_landmarks[0]
            raw_world_landmarks = result.pose_world_landmarks[0]

            state.smoothed_normalized_landmarks = smooth_landmarks(
                state.smoothed_normalized_landmarks,
                raw_normalized_landmarks,
                self._config.landmark_smoothing_alpha,
            )
            state.smoothed_world_landmarks = smooth_landmarks(
                state.smoothed_world_landmarks,
                raw_world_landmarks,
                self._config.landmark_smoothing_alpha,
            )

            normalized_landmarks = state.smoothed_normalized_landmarks
            world_landmarks = state.smoothed_world_landmarks

            raw_metrics = compute_metrics(normalized_landmarks, world_landmarks, self._config)

            if raw_metrics is None:
                status_label = "Pose not reliable"
                info_line = "Tracking confidence is too low. Sit so shoulders and head stay visible."
                zone = "unreliable"
                state.bad_posture_start = None
                state.movement_refresh_streak = 0
                state.movement_reposition_streak = 0

            elif state.baseline is None:
                reliable_pose = True
                state.latest_metrics = dict(raw_metrics)
                status_label, info_line, zone = self._handle_calibration(raw_metrics, now, state)

            else:
                reliable_pose = True
                calibrated_result = self._evaluate_calibrated_posture(
                    raw_metrics=raw_metrics,
                    now=now,
                    delta_time=delta_time,
                    state=state,
                    posture_alert_count=posture_alert_count,
                    stillness_alert_count=stillness_alert_count,
                )
                status_label = calibrated_result.status_label
                info_line = calibrated_result.info_line
                zone = calibrated_result.zone
                dominant_issue = calibrated_result.dominant_issue
                total_score = calibrated_result.total_score
                static_duration = calibrated_result.static_duration
                head_delta = calibrated_result.head_delta
                torso_delta = calibrated_result.torso_delta
                neck_drop = calibrated_result.neck_drop
                shoulder_tilt_delta = calibrated_result.shoulder_tilt_delta
                head_tilt_delta = calibrated_result.head_tilt_delta
                screen_approach_delta = calibrated_result.screen_approach_delta
                movement_score = calibrated_result.movement_score
                alert_event = calibrated_result.alert_event
                posture_alert_count = calibrated_result.posture_alert_count
                stillness_alert_count = calibrated_result.stillness_alert_count
        else:
            state.bad_posture_start = None
            state.movement_refresh_streak = 0
            state.movement_reposition_streak = 0

        state.latest_pose_detected = pose_detected
        state.latest_reliable_pose = reliable_pose

        bad_duration = 0.0
        if state.bad_posture_start is not None:
            bad_duration = now - state.bad_posture_start

        snapshot = MonitoringSnapshot(
            monitoring_active=True,
            paused=False,
            preview_enabled=preview_enabled,
            calibrated=state.baseline is not None,
            pose_detected=pose_detected,
            reliable_pose=reliable_pose,
            status_label=status_label,
            info_line=info_line,
            zone=zone,
            dominant_issue=dominant_issue,
            total_score=total_score,
            static_duration=static_duration,
            bad_duration=bad_duration,
            movement_score=movement_score,
            head_delta=head_delta,
            torso_delta=torso_delta,
            neck_drop=neck_drop,
            shoulder_tilt_delta=shoulder_tilt_delta,
            head_tilt_delta=head_tilt_delta,
            screen_approach_delta=screen_approach_delta,
            posture_alert_count=posture_alert_count,
            stillness_alert_count=stillness_alert_count,
            reposition_count=state.reposition_count,
            muted_until_monotonic=self._get_muted_until(),
            loop_latency_ms=state.loop_latency_ms,
            baseline_generation=state.baseline_generation,
            recalibration_reason=state.last_recalibration_reason,
            camera_label=self._camera_label,
        )
        return snapshot, alert_event

    def _handle_calibration(self, raw_metrics: dict[str, float], now: float, state: RuntimeState) -> tuple[str, str, str]:
        state.calibration_samples.append(raw_metrics)

        remaining = max(0.0, self._config.calibration_seconds - (now - state.calibration_start))
        status_label = "Calibration in progress"
        info_line = f"Sit naturally and keep your upper body visible. Remaining: {remaining:0.1f}s"
        zone = "calibration"

        enough_time_passed = (now - state.calibration_start) >= self._config.calibration_seconds
        enough_samples = len(state.calibration_samples) >= self._config.calibration_min_samples

        if enough_time_passed and enough_samples:
            state.baseline = build_baseline(state.calibration_samples)
            state.smoothed_metrics = None
            state.previous_metrics = None
            state.last_movement_time = now
            state.last_reposition_time = now
            state.last_movement_score = 0.0
            state.last_raw_movement_score = 0.0
            state.filtered_movement_score = 0.0
            state.movement_refresh_streak = 0
            state.movement_reposition_streak = 0
            state.baseline_generation += 1
            state.baseline_established_at = now
            state.auto_recalibration_pending = False
            state.auto_recalibration_candidate_since = None
            state.last_logged_zone = None

            self._log_event(
                kind="baseline_ready",
                title="Baseline ready",
                message=state.last_recalibration_reason,
                state=state,
                zone="calibration",
                dominant_issue="",
                total_score=0.0,
            )

        return status_label, info_line, zone

    def _filter_movement_score(self, raw_score: float, state: RuntimeState) -> float:
        deadband = max(0.0, float(self._config.movement_deadband))
        score_after_deadband = max(0.0, raw_score - deadband)

        previous_filtered = float(state.filtered_movement_score)
        alpha = max(0.0, min(1.0, float(self._config.movement_score_smoothing_alpha)))
        filtered = (alpha * score_after_deadband) + ((1.0 - alpha) * previous_filtered)

        state.filtered_movement_score = filtered
        return filtered

    def _get_pose_state(self, snapshot: MonitoringSnapshot) -> str:
        if snapshot.paused:
            return "paused"
        if not snapshot.calibrated:
            return "calibration"
        if not snapshot.pose_detected:
            return "no_person"
        if not snapshot.reliable_pose:
            return "unreliable"
        return "tracking"

    def _update_movement_timers(self, movement_score: float, now: float, state: RuntimeState) -> None:
        if movement_score >= self._config.movement_refresh_threshold:
            state.movement_refresh_streak += 1
        else:
            state.movement_refresh_streak = 0

        if movement_score >= self._config.reposition_threshold:
            state.movement_reposition_streak += 1
        else:
            state.movement_reposition_streak = 0

        if state.movement_refresh_streak >= self._config.movement_consecutive_frames_for_refresh:
            state.last_movement_time = now
            state.movement_refresh_streak = 0

        if state.movement_reposition_streak >= self._config.movement_consecutive_frames_for_reposition:
            last_reposition_time = state.last_reposition_time
            can_count_reposition = (
                last_reposition_time is None
                or (now - last_reposition_time) >= self._config.reposition_cooldown_seconds
            )
            if can_count_reposition:
                state.last_reposition_time = now
                state.last_movement_time = now
                state.reposition_count += 1
                state.auto_recalibration_pending = True
                state.auto_recalibration_candidate_since = None

            state.movement_reposition_streak = 0

    def _maybe_start_auto_recalibration(
        self,
        now: float,
        movement_score: float,
        evaluation: dict[str, object],
        state: RuntimeState,
    ) -> bool:
        if not self._config.auto_recalibration_enabled:
            return False

        if not state.auto_recalibration_pending:
            return False

        baseline_established_at = state.baseline_established_at
        if baseline_established_at is None:
            return False

        if (now - baseline_established_at) < self._config.auto_recalibration_min_time_since_baseline_seconds:
            return False

        last_auto_recalibration_time = state.last_auto_recalibration_time
        if (
            last_auto_recalibration_time is not None
            and (now - last_auto_recalibration_time) < self._config.auto_recalibration_cooldown_seconds
        ):
            return False

        total_score = float(evaluation["total_score"])
        zone = str(evaluation["zone"])

        if zone != "green" or total_score > self._config.auto_recalibration_max_score:
            state.auto_recalibration_candidate_since = None
            return False

        if movement_score > self._config.movement_deadband:
            state.auto_recalibration_candidate_since = None
            return False

        if state.auto_recalibration_candidate_since is None:
            state.auto_recalibration_candidate_since = now
            return False

        stable_duration = now - state.auto_recalibration_candidate_since
        if stable_duration < self._config.auto_recalibration_stability_seconds:
            return False

        state.last_auto_recalibration_time = now
        self._reset_for_recalibration(state, "auto_recalibration_after_reposition")
        state.auto_recalibration_started = True
        return True

    def _evaluate_calibrated_posture(
        self,
        raw_metrics: dict[str, float],
        now: float,
        delta_time: float,
        state: RuntimeState,
        posture_alert_count: int,
        stillness_alert_count: int,
    ) -> CalibratedPostureResult:
        _ = delta_time

        smoothed_metrics = smooth_metrics(
            state.smoothed_metrics,
            raw_metrics,
            self._config.metrics_smoothing_alpha,
        )

        raw_movement_score = compute_movement_score(smoothed_metrics, state.previous_metrics, self._config)
        movement_score = self._filter_movement_score(raw_movement_score, state)
        state.last_raw_movement_score = raw_movement_score
        state.last_movement_score = movement_score

        self._update_movement_timers(movement_score, now, state)

        evaluation = evaluate_ergonomics(smoothed_metrics, state.baseline, self._config)
        state.smoothed_metrics = smoothed_metrics
        state.previous_metrics = dict(smoothed_metrics)
        state.latest_metrics = dict(smoothed_metrics)
        state.latest_evaluation = dict(evaluation)

        if self._maybe_start_auto_recalibration(now, movement_score, evaluation, state):
            return CalibratedPostureResult(
                status_label="Auto recalibration",
                info_line="Seat position changed. Updating baseline.",
                zone="calibration",
                dominant_issue="",
                total_score=0.0,
                static_duration=0.0,
                head_delta=0.0,
                torso_delta=0.0,
                neck_drop=0.0,
                shoulder_tilt_delta=0.0,
                head_tilt_delta=0.0,
                screen_approach_delta=0.0,
                movement_score=movement_score,
                alert_event=None,
                posture_alert_count=posture_alert_count,
                stillness_alert_count=stillness_alert_count,
            )

        total_score = float(evaluation["total_score"])
        head_delta = float(evaluation["head_delta"])
        torso_delta = float(evaluation["torso_delta"])
        neck_drop = float(evaluation["neck_drop"])
        shoulder_tilt_delta = float(evaluation["shoulder_tilt_delta"])
        head_tilt_delta = float(evaluation["head_tilt_delta"])
        screen_approach_delta = float(evaluation["screen_approach_delta"])
        dominant_issue = str(evaluation["dominant_issue_label"])
        zone = str(evaluation["zone"])

        static_duration = 0.0
        if state.last_movement_time is not None:
            static_duration = max(0.0, now - state.last_movement_time)

        alert_event: Optional[AlertEvent] = None

        if zone == "green":
            status_label = "Neutral range"
            info_line = "You are inside your calibrated neutral zone."
            state.bad_posture_start = None

        elif zone == "yellow":
            status_label = "Drifting from neutral"
            info_line = f"Watch this trend: {dominant_issue}." if dominant_issue else "Small deviation detected."
            state.bad_posture_start = None

        else:
            status_label = "Posture overload"
            info_line = f"Main issue: {dominant_issue}."

            if state.bad_posture_start is None:
                state.bad_posture_start = now

            bad_duration = now - state.bad_posture_start
            can_alert_by_duration = bad_duration >= self._config.posture_alert_after_seconds
            can_alert_by_cooldown = (now - state.last_posture_alert_time) >= self._config.posture_alert_cooldown_seconds

            if can_alert_by_duration and can_alert_by_cooldown:
                state.last_posture_alert_time = now
                posture_alert_count += 1
                state.posture_alert_fired = True
                alert_event = AlertEvent(
                    kind="posture",
                    title="Posture Guard",
                    message=f"Adjust posture: {dominant_issue}.",
                )

        can_send_stillness_reminder = (
            static_duration >= self._config.stillness_reminder_after_seconds
            and (now - state.last_stillness_alert_time) >= self._config.stillness_alert_cooldown_seconds
        )

        if can_send_stillness_reminder:
            state.last_stillness_alert_time = now
            stillness_alert_count += 1
            state.stillness_alert_fired = True

            if alert_event is None:
                alert_event = AlertEvent(
                    kind="stillness",
                    title="Posture Guard",
                    message="Time to change position or move a little.",
                )

        if zone != "red" and static_duration >= self._config.stillness_reminder_after_seconds:
            status_label = "Move reminder"
            info_line = "You stayed too still for too long."

        return CalibratedPostureResult(
            status_label=status_label,
            info_line=info_line,
            zone=zone,
            dominant_issue=dominant_issue,
            total_score=total_score,
            static_duration=static_duration,
            head_delta=head_delta,
            torso_delta=torso_delta,
            neck_drop=neck_drop,
            shoulder_tilt_delta=shoulder_tilt_delta,
            head_tilt_delta=head_tilt_delta,
            screen_approach_delta=screen_approach_delta,
            movement_score=movement_score,
            alert_event=alert_event,
            posture_alert_count=posture_alert_count,
            stillness_alert_count=stillness_alert_count,
        )

    def _build_preview_frame(self, frame, snapshot: MonitoringSnapshot, state: RuntimeState):
        rendered = frame.copy()

        smoothed_landmarks = state.smoothed_normalized_landmarks
        if smoothed_landmarks:
            draw_pose(rendered, smoothed_landmarks, self._config)

        draw_text(rendered, self._config.app_name, 0, (255, 255, 255))
        draw_text(rendered, self._camera_label, 1, (200, 200, 200))

        if not snapshot.calibrated:
            remaining = max(0.0, self._config.calibration_seconds - (time.monotonic() - state.calibration_start))
            draw_text(rendered, f"Status: calibration ({remaining:0.1f}s left)", 2, (0, 255, 255))
            draw_text(
                rendered,
                f"Samples: {len(state.calibration_samples)}/{self._config.calibration_min_samples}",
                3,
                (255, 255, 255),
            )
            draw_text(rendered, "Sit naturally, look at the screen, do not freeze completely.", 4, (220, 220, 220))
            draw_text(rendered, "Controls: R - recalibrate | P/Q/Esc - hide preview", 6, (220, 220, 220))
        else:
            color = (255, 255, 255)
            if snapshot.zone == "green":
                color = (0, 220, 0)
            elif snapshot.zone == "yellow":
                color = (0, 200, 255)
            elif snapshot.zone == "red":
                color = (0, 0, 255)

            draw_text(rendered, f"Status: {snapshot.status_label}", 2, color)
            draw_text(rendered, f"Info: {snapshot.info_line}", 3, (255, 255, 255))
            draw_text(
                rendered,
                f"Strain score: {snapshot.total_score:0.2f} | Static time: {snapshot.static_duration:0.1f}s",
                4,
                (255, 255, 255),
            )
            draw_text(
                rendered,
                f"Head: {snapshot.head_delta:+0.3f} m | Torso: {snapshot.torso_delta:+0.1f} deg | Neck: {snapshot.neck_drop:+0.3f} m",
                5,
                (255, 255, 255),
            )
            draw_text(
                rendered,
                f"Shoulders: {snapshot.shoulder_tilt_delta:+0.1f} deg | Head tilt: {snapshot.head_tilt_delta:+0.1f} deg",
                6,
                (255, 255, 255),
            )
            draw_text(
                rendered,
                f"Screen ratio: {snapshot.screen_approach_delta:+0.3f} | Movement score: {snapshot.movement_score:0.2f}",
                7,
                (255, 255, 255),
            )
            draw_text(
                rendered,
                f"Bad posture: {snapshot.bad_duration:0.1f}s | Posture alerts: {snapshot.posture_alert_count}",
                8,
                (255, 255, 255),
            )
            draw_text(
                rendered,
                f"Move reminders: {snapshot.stillness_alert_count} | Repositions: {snapshot.reposition_count}",
                9,
                (255, 255, 255),
            )
            draw_text(
                rendered,
                f"Loop latency: {snapshot.loop_latency_ms:0.1f} ms | Baseline: {snapshot.baseline_generation}",
                10,
                (255, 255, 255),
            )
            draw_text(rendered, "Controls: R - recalibrate | P/Q/Esc - hide preview", 12, (220, 220, 220))

            if snapshot.zone == "red":
                cv2.rectangle(
                    rendered,
                    (10, 10),
                    (rendered.shape[1] - 10, rendered.shape[0] - 10),
                    (0, 0, 255),
                    4,
                )
                draw_text(rendered, "ALERT: Adjust posture", 13, (0, 0, 255))

        return rendered

    def _destroy_preview_window(self) -> None:
        try:
            cv2.destroyWindow(self._config.app_name)
        except Exception:
            logger.debug("Preview window was already closed or unavailable", exc_info=True)

    def _emit_snapshot(self, snapshot: MonitoringSnapshot) -> None:
        with self._lock:
            self._latest_snapshot = snapshot

        if self._on_snapshot is not None:
            try:
                self._on_snapshot(snapshot)
            except Exception:
                logger.exception("Snapshot callback failed")

    def _emit_alert(self, event: AlertEvent, snapshot: MonitoringSnapshot, state: RuntimeState) -> None:
        if self._on_alert is None:
            self._log_suppressed_alert(event, snapshot, state, "no_alert_callback")
            return

        def on_delivery(result: NotificationDeliveryResult) -> None:
            self._log_alert_telemetry(event, snapshot, state, result)

        try:
            result = self._on_alert(event, on_delivery=on_delivery)
        except TypeError:
            # Compatibility with older callbacks that accepted only AlertEvent.
            try:
                accepted = bool(self._on_alert(event))
            except Exception:
                logger.exception("Alert callback failed")
                self._log_suppressed_alert(event, snapshot, state, "callback_error")
                return

            if not accepted:
                self._log_suppressed_alert(event, snapshot, state, "callback_rejected")
            return
        except Exception:
            logger.exception("Alert callback failed")
            self._log_suppressed_alert(event, snapshot, state, "callback_error")
            return

        if isinstance(result, NotificationQueueResult):
            if not result.accepted:
                self._log_suppressed_alert(event, snapshot, state, result.reason or "queue_rejected")
            return

        if isinstance(result, bool):
            if not result:
                self._log_suppressed_alert(event, snapshot, state, "callback_rejected")
            return

        logger.warning("Alert callback returned unsupported result type: %s", type(result).__name__)

    def _log_suppressed_alert(
        self,
        event: AlertEvent,
        snapshot: MonitoringSnapshot,
        state: RuntimeState,
        reason: str,
    ) -> None:
        result = NotificationDeliveryResult(
            key=event.kind,
            title=event.title,
            message=event.message,
            queued=False,
            toast_backend_accepted=False,
            sound_played=False,
            fallback_printed=False,
            error_message=reason,
        )
        self._log_alert_telemetry(event, snapshot, state, result)

    def _get_muted_until(self) -> float:
        with self._lock:
            return self._muted_until_monotonic

    def _now_iso(self) -> str:
        return datetime.now().isoformat(timespec="milliseconds")

    def _build_telemetry_row(self, snapshot: MonitoringSnapshot, state: RuntimeState) -> dict[str, object]:
        return {
            "clock": self._now_iso(),
            "monotonic_seconds": round(time.monotonic(), 6),
            "timestamp_ms": state.last_stream_timestamp_ms,
            "frame_index": state.frame_index,
            "pose_state": self._get_pose_state(snapshot),
            "zone": snapshot.zone,
            "total_score": round(snapshot.total_score, 6),
            "movement_score": round(snapshot.movement_score, 6),
            "bad_duration_s": round(snapshot.bad_duration, 3),
            "static_duration_s": round(snapshot.static_duration, 3),
            "dominant_issue": snapshot.dominant_issue,
            "notification_mode": "on" if self._config.notifications_enabled else "off",
            "baseline_generation": snapshot.baseline_generation,
            "reposition_count": snapshot.reposition_count,
        }

    def _log_frame_telemetry(self, snapshot: MonitoringSnapshot, state: RuntimeState) -> None:
        previous_zone = state.last_logged_zone
        if previous_zone is not None and previous_zone != snapshot.zone:
            self._log_event(
                kind="zone_changed",
                title="Zone changed",
                message=f"{previous_zone} -> {snapshot.zone}",
                state=state,
                zone=snapshot.zone,
                dominant_issue=snapshot.dominant_issue,
                total_score=snapshot.total_score,
            )

        state.last_logged_zone = snapshot.zone
        with self._telemetry_lock:
            self._telemetry.log_frame(self._build_telemetry_row(snapshot, state))

    def _log_alert_telemetry(
        self,
        event: AlertEvent,
        snapshot: MonitoringSnapshot,
        state: RuntimeState,
        delivery: NotificationDeliveryResult,
    ) -> None:
        with self._telemetry_lock:
            self._telemetry.log_alert(
                {
                    "clock": self._now_iso(),
                    "monotonic_seconds": round(time.monotonic(), 6),
                    "timestamp_ms": state.last_stream_timestamp_ms,
                    "frame_index": state.frame_index,
                    "alert_type": event.kind,
                    "zone": snapshot.zone,
                    "total_score": round(snapshot.total_score, 6),
                    "movement_score": round(snapshot.movement_score, 6),
                    "dominant_issue": snapshot.dominant_issue,
                    "bad_duration_s": round(snapshot.bad_duration, 3),
                    "static_duration_s": round(snapshot.static_duration, 3),
                    "notification_mode": "on" if self._config.notifications_enabled else "off",
                    "queued": delivery.queued,
                    "toast_backend_accepted": delivery.toast_backend_accepted,
                    "sound_played": delivery.sound_played,
                    "fallback_printed": delivery.fallback_printed,
                    "delivered": delivery.user_visible,
                    "delivery_error": delivery.error_message or "",
                }
            )

    def _log_event(
        self,
        kind: str,
        title: str,
        message: str,
        state: RuntimeState,
        zone: str,
        dominant_issue: str,
        total_score: float,
    ) -> None:
        with self._telemetry_lock:
            self._telemetry.log_event(
                {
                    "wall_clock_iso": self._now_iso(),
                    "monotonic_seconds": round(time.monotonic(), 6),
                    "timestamp_ms": state.last_stream_timestamp_ms,
                    "frame_index": state.frame_index,
                    "kind": kind,
                    "zone": zone,
                    "total_score": round(float(total_score), 6),
                    "movement_score": round(float(state.last_movement_score), 6),
                    "dominant_issue": dominant_issue,
                    "details": f"{title}: {message}" if message else title,
                    "baseline_generation": state.baseline_generation,
                }
            )

    def _log_startup_error(self, message: str) -> None:
        state = RuntimeState(last_recalibration_reason="startup_error")
        self._log_event(
            kind="startup_error",
            title="Startup error",
            message=message,
            state=state,
            zone="error",
            dominant_issue="",
            total_score=0.0,
        )

    def _create_error_snapshot(self, message: str) -> MonitoringSnapshot:
        return MonitoringSnapshot(
            monitoring_active=False,
            paused=False,
            preview_enabled=False,
            calibrated=False,
            pose_detected=False,
            reliable_pose=False,
            status_label="Startup error",
            info_line=message,
            zone="error",
            dominant_issue="",
            total_score=0.0,
            static_duration=0.0,
            bad_duration=0.0,
            movement_score=0.0,
            head_delta=0.0,
            torso_delta=0.0,
            neck_drop=0.0,
            shoulder_tilt_delta=0.0,
            head_tilt_delta=0.0,
            screen_approach_delta=0.0,
            posture_alert_count=0,
            stillness_alert_count=0,
            reposition_count=0,
            muted_until_monotonic=0.0,
            loop_latency_ms=0.0,
            baseline_generation=0,
            recalibration_reason="startup_error",
            camera_label=self._camera_label,
        )