from __future__ import annotations

import os

# Reduce native library log noise as much as possible before importing runtime dependencies.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_LOG_LEVEL", "2")

import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2
from mediapipe.tasks.python import vision

from config import AppConfig
from config import load_app_config
from config import save_app_config
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


# ============================================================
# Path helpers
# ============================================================

DATA_DIR_NAME = "data"
CONFIG_FILE_NAME = "config.json"


def get_runtime_directory() -> Path:
    """
    Return the base directory used by the application.

    In source mode this is the directory containing app.py.
    In frozen mode this is the directory containing the executable.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    return Path(__file__).resolve().parent.parent


def get_resource_path(relative_path: str) -> Path:
    """
    Resolve a resource path for both source mode and PyInstaller mode.
    """
    relative = Path(relative_path)

    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative

    return get_runtime_directory() / relative


def get_data_directory() -> Path:
    """
    Return the directory used for mutable runtime files.

    The data directory is stored inside the application directory.
    """
    data_dir = get_runtime_directory() / DATA_DIR_NAME
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_config_file_path() -> Path:
    """
    Store config in data/config.json relative to the app directory.
    """
    return get_data_directory() / CONFIG_FILE_NAME


# ============================================================
# Camera helpers
# ============================================================

@dataclass(frozen=True)
class CameraOpenResult:
    capture: cv2.VideoCapture
    camera_index: int
    backend_name: str


def _get_camera_backend_candidates() -> list[tuple[str, int]]:
    if sys.platform.startswith("win"):
        candidates = [("CAP_DSHOW", cv2.CAP_DSHOW)]

        # CAP_MSMF is not always available in every OpenCV build.
        cap_msmf = getattr(cv2, "CAP_MSMF", None)
        if isinstance(cap_msmf, int):
            candidates.append(("CAP_MSMF", cap_msmf))

        candidates.append(("CAP_ANY", cv2.CAP_ANY))
        return candidates

    return [("CAP_ANY", cv2.CAP_ANY)]


def _get_camera_index_candidates(preferred_index: int) -> list[int]:
    indices = [preferred_index]
    for fallback_index in range(3):
        if fallback_index not in indices:
            indices.append(fallback_index)
    return indices


def open_camera(camera_index: int, frame_width: int, frame_height: int) -> CameraOpenResult:
    """
    Open a working camera with several backend and index fallbacks.

    This is intentionally defensive because laptop webcams can behave differently
    across OpenCV builds and Windows backends.
    """
    last_error = "unknown error"

    for candidate_index in _get_camera_index_candidates(camera_index):
        for backend_name, backend_id in _get_camera_backend_candidates():
            capture = cv2.VideoCapture(candidate_index, backend_id)

            if not capture.isOpened():
                capture.release()
                continue

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Try to grab a frame immediately to verify that the camera really works.
            read_ok = False
            for _ in range(8):
                read_ok, _ = capture.read()
                if read_ok:
                    break
                time.sleep(0.03)

            if read_ok:
                return CameraOpenResult(
                    capture=capture,
                    camera_index=candidate_index,
                    backend_name=backend_name,
                )

            last_error = (
                f"Camera opened but did not return frames "
                f"(index={candidate_index}, backend={backend_name})"
            )
            capture.release()

    raise RuntimeError(
        "Could not open a working camera. "
        f"Last attempt failed with: {last_error}"
    )


def safe_release_camera(capture: Optional[cv2.VideoCapture]) -> None:
    if capture is None:
        return

    try:
        capture.release()
    except Exception:
        pass


# ============================================================
# Notification layer
# ============================================================

@dataclass
class AlertPayload:
    title: str
    message: str


def set_windows_app_user_model_id(app_id: str) -> None:
    """
    Set AppUserModelID to improve toast notification reliability on Windows.
    """
    if not sys.platform.startswith("win"):
        return

    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass


def play_system_notification_sound() -> None:
    """
    Play only the operating system notification sound.
    """
    try:
        if sys.platform.startswith("win"):
            import winsound

            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass


class ToastNotifier:
    """
    Best-effort desktop notification wrapper.

    The important design choice here is that actual toast delivery happens
    in a separate worker thread so monitoring never blocks on a toast backend.
    """

    def __init__(self, app_name: str) -> None:
        self._app_name = app_name
        self._backend = self._create_backend()

    def show(self, title: str, message: str) -> bool:
        if self._backend is None:
            return False

        try:
            self._backend(title, message)
            return True
        except Exception:
            return False

    def _create_backend(self):
        backend = self._try_win11toast()
        if backend is not None:
            return backend

        backend = self._try_win10toast()
        if backend is not None:
            return backend

        backend = self._try_plyer()
        if backend is not None:
            return backend

        return None

    def _try_win11toast(self):
        try:
            from win11toast import toast
        except Exception:
            return None

        def send(title: str, message: str) -> None:
            """
            Keep the notification simple and passive.
            No buttons, no callbacks, no user action required.
            """
            toast(
                title,
                message,
                app_id=self._app_name,
                duration="short",
            )

        return send

    def _try_win10toast(self):
        try:
            from win10toast import ToastNotifier as Win10ToastNotifier
        except Exception:
            return None

        toaster = Win10ToastNotifier()

        def send(title: str, message: str) -> None:
            """
            threaded=True prevents the backend from blocking the caller.
            """
            toaster.show_toast(
                title,
                message,
                duration=4,
                threaded=True,
                icon_path=None,
            )

        return send

    def _try_plyer(self):
        try:
            from plyer import notification
        except Exception:
            return None

        def send(title: str, message: str) -> None:
            notification.notify(
                title=title,
                message=message,
                app_name=self._app_name,
                timeout=4,
            )

        return send


class NotificationManager:
    """
    High-level notification manager used by the monitoring engine.

    All actual notification delivery is serialized through a background worker.
    This prevents toast backends from freezing the monitoring loop.
    """

    def __init__(
        self,
        app_name: str,
        sound_enabled: bool,
        toast_enabled: bool,
    ) -> None:
        self._app_name = app_name
        self._sound_enabled = sound_enabled
        self._toast_enabled = toast_enabled
        self._last_notification_times: dict[str, float] = {}
        self._lock = threading.Lock()

        set_windows_app_user_model_id(f"{app_name}.desktop")
        self._toast = ToastNotifier(app_name) if toast_enabled else None

        self._queue: queue.Queue[Optional[tuple[AlertPayload, bool]]] = queue.Queue()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="NotificationWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def notify(self, key: str, payload: AlertPayload, cooldown_seconds: float) -> None:
        """
        Queue a notification if the per-key cooldown allows it.

        The monitoring thread returns immediately after pushing to the queue.
        """
        now = time.monotonic()

        with self._lock:
            previous = self._last_notification_times.get(key, -1e9)
            if (now - previous) < cooldown_seconds:
                return
            self._last_notification_times[key] = now

        play_sound = self._sound_enabled
        self._queue.put((payload, play_sound))

    def stop(self) -> None:
        try:
            self._queue.put(None)
            self._worker_thread.join(timeout=2.0)
        except Exception:
            pass

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()

            if item is None:
                break

            payload, play_sound = item

            toast_delivered = False
            if self._toast_enabled and self._toast is not None:
                try:
                    toast_delivered = self._toast.show(payload.title, payload.message)
                except Exception:
                    toast_delivered = False

            if play_sound:
                try:
                    play_system_notification_sound()
                except Exception:
                    pass

            if not toast_delivered:
                print(f"[{self._app_name}] {payload.title}: {payload.message}")


# ============================================================
# Tray integration
# ============================================================

@dataclass
class TrayCallbacks:
    pause_monitoring: Callable[[], None]
    resume_monitoring: Callable[[], None]
    toggle_preview: Callable[[], None]
    recalibrate: Callable[[], None]
    mute_for_fifteen_minutes: Callable[[], None]
    open_app_folder: Callable[[], None]
    exit_application: Callable[[], None]


class TrayController:
    """
    Optional system tray integration.

    If pystray or Pillow are missing, the app still runs normally.
    """

    def __init__(self, app_name: str, callbacks: TrayCallbacks) -> None:
        self._app_name = app_name
        self._callbacks = callbacks
        self._icon = None
        self._available = False

        try:
            import pystray
            from PIL import Image, ImageDraw
        except Exception:
            return

        self._pystray = pystray
        self._image_class = Image
        self._image_draw_class = ImageDraw
        self._available = True

    def start(self) -> bool:
        if not self._available:
            return False

        icon_image = self._create_icon_image()
        menu = self._pystray.Menu(
            self._pystray.MenuItem("Pause monitoring", self._on_pause_monitoring),
            self._pystray.MenuItem("Resume monitoring", self._on_resume_monitoring),
            self._pystray.MenuItem("Show / hide preview", self._on_toggle_preview),
            self._pystray.MenuItem("Recalibrate", self._on_recalibrate),
            self._pystray.MenuItem("Mute alerts for 15 minutes", self._on_mute_for_fifteen_minutes),
            self._pystray.MenuItem("Open app folder", self._on_open_app_folder),
            self._pystray.MenuItem("Exit", self._on_exit),
        )

        self._icon = self._pystray.Icon(self._app_name, icon_image, self._app_name, menu)
        self._icon.run_detached()
        return True

    def stop(self) -> None:
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:
                pass

    def _create_icon_image(self):
        image = self._image_class.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = self._image_draw_class.Draw(image)

        draw.rounded_rectangle((8, 8, 56, 56), radius=12, fill=(33, 170, 88, 255))
        draw.rounded_rectangle((40, 40, 56, 56), radius=5, fill=(220, 50, 47, 255))
        return image

    def _on_pause_monitoring(self, icon=None, item=None) -> None:
        self._callbacks.pause_monitoring()

    def _on_resume_monitoring(self, icon=None, item=None) -> None:
        self._callbacks.resume_monitoring()

    def _on_toggle_preview(self, icon=None, item=None) -> None:
        self._callbacks.toggle_preview()

    def _on_recalibrate(self, icon=None, item=None) -> None:
        self._callbacks.recalibrate()

    def _on_mute_for_fifteen_minutes(self, icon=None, item=None) -> None:
        self._callbacks.mute_for_fifteen_minutes()

    def _on_open_app_folder(self, icon=None, item=None) -> None:
        self._callbacks.open_app_folder()

    def _on_exit(self, icon=None, item=None) -> None:
        self._callbacks.exit_application()


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
        on_snapshot: Optional[Callable[["MonitoringSnapshot"], None]] = None,
        on_alert: Optional[Callable[["AlertEvent"], None]] = None,
    ) -> None:
        self._config = config
        self._data_dir = data_dir
        self._on_snapshot = on_snapshot
        self._on_alert = on_alert

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._lock = threading.Lock()

        self._preview_enabled = config.preview_enabled
        self._recalibration_requested = False
        self._requested_recalibration_reason: Optional[str] = None
        self._muted_until_monotonic = 0.0
        self._latest_snapshot = MonitoringSnapshot(
            monitoring_active=False,
            paused=config.start_paused,
            preview_enabled=config.preview_enabled,
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

        self._latest_preview_frame = None
        self._preview_frame_lock = threading.Lock()
        self._fatal_error_message: Optional[str] = None
        self._camera_label = "Camera not opened yet"

        self._telemetry = TelemetryLogger(
            data_dir=data_dir,
            app_name=config.app_name,
            enabled=config.telemetry_enabled,
            flush_interval_seconds=config.telemetry_flush_interval_seconds,
        )
        self._telemetry.write_session_metadata(
            config_snapshot=config.to_dict(),
            extra={
                "data_dir": str(data_dir),
            },
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
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.clear_preview_frame()
        self._destroy_preview_window()
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
            self._destroy_preview_window()

    def request_recalibration(self) -> None:
        with self._lock:
            self._recalibration_requested = True
            self._requested_recalibration_reason = "manual_request"

    def mute_for_seconds(self, seconds: float) -> None:
        self._muted_until_monotonic = max(
            self._muted_until_monotonic,
            time.monotonic() + max(0.0, seconds),
        )

    def get_latest_snapshot(self) -> MonitoringSnapshot:
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
            with self._lock:
                self._preview_enabled = False
            self.clear_preview_frame()
            self._destroy_preview_window()

    def _create_runtime_state(self) -> dict:
        now = time.monotonic()
        return {
            "calibration_samples": [],
            "baseline": None,
            "smoothed_metrics": None,
            "previous_metrics": None,
            "smoothed_normalized_landmarks": None,
            "smoothed_world_landmarks": None,
            "latest_metrics": None,
            "latest_evaluation": None,
            "latest_pose_detected": False,
            "latest_reliable_pose": False,
            "calibration_start": now,
            "bad_posture_start": None,
            "last_posture_alert_time": -10000.0,
            "last_stillness_alert_time": -10000.0,
            "last_movement_time": None,
            "last_reposition_time": None,
            "last_movement_score": 0.0,
            "last_raw_movement_score": 0.0,
            "reposition_count": 0,
            "movement_refresh_streak": 0,
            "movement_reposition_streak": 0,
            "filtered_movement_score": 0.0,
            "baseline_generation": 0,
            "baseline_established_at": None,
            "auto_recalibration_pending": False,
            "auto_recalibration_candidate_since": None,
            "last_auto_recalibration_time": None,
            "last_recalibration_reason": "initial_calibration",
            "last_logged_zone": None,
            "frame_index": 0,
            "last_stream_timestamp_ms": 0,
            "loop_latency_ms": 0.0,
            "posture_alert_fired": False,
            "stillness_alert_fired": False,
            "auto_recalibration_started": False,
        }

    def _reset_for_recalibration(self, state: dict, reason: str) -> None:
        now = time.monotonic()
        state["calibration_samples"] = []
        state["baseline"] = None
        state["smoothed_metrics"] = None
        state["previous_metrics"] = None
        state["smoothed_normalized_landmarks"] = None
        state["smoothed_world_landmarks"] = None
        state["latest_metrics"] = None
        state["latest_evaluation"] = None
        state["latest_pose_detected"] = False
        state["latest_reliable_pose"] = False
        state["calibration_start"] = now
        state["bad_posture_start"] = None
        state["last_movement_time"] = None
        state["last_movement_score"] = 0.0
        state["last_raw_movement_score"] = 0.0
        state["movement_refresh_streak"] = 0
        state["movement_reposition_streak"] = 0
        state["filtered_movement_score"] = 0.0
        state["auto_recalibration_pending"] = False
        state["auto_recalibration_candidate_since"] = None
        state["posture_alert_fired"] = False
        state["stillness_alert_fired"] = False
        state["auto_recalibration_started"] = False
        state["last_logged_zone"] = None
        state["last_recalibration_reason"] = reason

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

            camera_result = open_camera(
                self._config.camera_index,
                self._config.frame_width,
                self._config.frame_height,
            )
            capture = camera_result.capture
            self._camera_label = (
                f"Camera index {camera_result.camera_index} "
                f"({camera_result.backend_name})"
            )

            self._log_event(
                kind="session_started",
                title="Session",
                message=self._camera_label,
                state=state,
                zone="startup",
                dominant_issue="",
                total_score=0.0,
            )

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

                    read_ok, frame = capture.read()
                    if not read_ok:
                        time.sleep(0.02)
                        continue

                    frame = cv2.flip(frame, 1)

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

                    state["frame_index"] += 1
                    state["last_stream_timestamp_ms"] = timestamp_ms

                    processing_start = time.monotonic()
                    result = detect_pose(landmarker, frame, timestamp_ms)

                    snapshot, alert_event = self._process_frame(
                        result=result,
                        now=now,
                        delta_time=delta_time,
                        stream_timestamp_ms=timestamp_ms,
                        state=state,
                        posture_alert_count=posture_alert_count,
                        stillness_alert_count=stillness_alert_count,
                        preview_enabled=preview_enabled,
                    )

                    processing_latency_ms = (time.monotonic() - processing_start) * 1000.0
                    state["loop_latency_ms"] = processing_latency_ms

                    if processing_latency_ms != snapshot.loop_latency_ms:
                        snapshot = MonitoringSnapshot(
                            monitoring_active=snapshot.monitoring_active,
                            paused=snapshot.paused,
                            preview_enabled=snapshot.preview_enabled,
                            calibrated=snapshot.calibrated,
                            pose_detected=snapshot.pose_detected,
                            reliable_pose=snapshot.reliable_pose,
                            status_label=snapshot.status_label,
                            info_line=snapshot.info_line,
                            zone=snapshot.zone,
                            dominant_issue=snapshot.dominant_issue,
                            total_score=snapshot.total_score,
                            static_duration=snapshot.static_duration,
                            bad_duration=snapshot.bad_duration,
                            movement_score=snapshot.movement_score,
                            head_delta=snapshot.head_delta,
                            torso_delta=snapshot.torso_delta,
                            neck_drop=snapshot.neck_drop,
                            shoulder_tilt_delta=snapshot.shoulder_tilt_delta,
                            head_tilt_delta=snapshot.head_tilt_delta,
                            screen_approach_delta=snapshot.screen_approach_delta,
                            posture_alert_count=snapshot.posture_alert_count,
                            stillness_alert_count=snapshot.stillness_alert_count,
                            reposition_count=snapshot.reposition_count,
                            muted_until_monotonic=snapshot.muted_until_monotonic,
                            loop_latency_ms=processing_latency_ms,
                            baseline_generation=snapshot.baseline_generation,
                            recalibration_reason=snapshot.recalibration_reason,
                            camera_label=snapshot.camera_label,
                        )

                    posture_alert_count = snapshot.posture_alert_count
                    stillness_alert_count = snapshot.stillness_alert_count

                    self._emit_snapshot(snapshot)
                    self._log_frame_telemetry(snapshot, state)

                    if alert_event is not None:
                        self._log_event(
                            kind=alert_event.kind,
                            title=alert_event.title,
                            message=alert_event.message,
                            state=state,
                            zone=snapshot.zone,
                            dominant_issue=snapshot.dominant_issue,
                            total_score=snapshot.total_score,
                        )
                        if time.monotonic() >= self._muted_until_monotonic:
                            self._emit_alert(alert_event)

                    if preview_enabled:
                        rendered_frame = self._build_preview_frame(frame, snapshot, state)
                        self.update_preview_frame(rendered_frame)
                    else:
                        self.clear_preview_frame()

        except Exception as exc:
            self._fatal_error_message = str(exc)
            self._log_event(
                kind="startup_error",
                title="Startup error",
                message=str(exc),
                state={
                    "frame_index": 0,
                    "baseline_generation": 0,
                    "last_recalibration_reason": "startup_error",
                },
                zone="error",
                dominant_issue="",
                total_score=0.0,
            )
            error_snapshot = MonitoringSnapshot(
                monitoring_active=False,
                paused=False,
                preview_enabled=False,
                calibrated=False,
                pose_detected=False,
                reliable_pose=False,
                status_label="Startup error",
                info_line=str(exc),
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
            self._emit_snapshot(error_snapshot)
            print(f"[{self._config.app_name}] startup error: {exc}")

        finally:
            safe_release_camera(capture)
            self.clear_preview_frame()

    def _handle_paused_state(self, state: dict, posture_alert_count: int, stillness_alert_count: int) -> None:
        state["latest_metrics"] = None
        state["latest_evaluation"] = None
        state["latest_pose_detected"] = False
        state["latest_reliable_pose"] = False

        snapshot = MonitoringSnapshot(
            monitoring_active=True,
            paused=True,
            preview_enabled=self._preview_enabled,
            calibrated=state["baseline"] is not None,
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
            reposition_count=state["reposition_count"],
            muted_until_monotonic=self._muted_until_monotonic,
            loop_latency_ms=0.0,
            baseline_generation=state["baseline_generation"],
            recalibration_reason=state["last_recalibration_reason"],
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
        stream_timestamp_ms: int,
        state: dict,
        posture_alert_count: int,
        stillness_alert_count: int,
        preview_enabled: bool,
    ) -> tuple[MonitoringSnapshot, Optional[AlertEvent]]:
        _ = stream_timestamp_ms

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
        movement_score = state["last_movement_score"]
        dominant_issue = ""
        zone = "no_person"
        pose_detected = False
        reliable_pose = False

        state["latest_metrics"] = None
        state["latest_evaluation"] = None
        state["latest_pose_detected"] = False
        state["latest_reliable_pose"] = False
        state["posture_alert_fired"] = False
        state["stillness_alert_fired"] = False
        state["auto_recalibration_started"] = False

        if result.pose_landmarks and result.pose_world_landmarks:
            pose_detected = True
            raw_normalized_landmarks = result.pose_landmarks[0]
            raw_world_landmarks = result.pose_world_landmarks[0]

            state["smoothed_normalized_landmarks"] = smooth_landmarks(
                state["smoothed_normalized_landmarks"],
                raw_normalized_landmarks,
                self._config.landmark_smoothing_alpha,
            )
            state["smoothed_world_landmarks"] = smooth_landmarks(
                state["smoothed_world_landmarks"],
                raw_world_landmarks,
                self._config.landmark_smoothing_alpha,
            )

            normalized_landmarks = state["smoothed_normalized_landmarks"]
            world_landmarks = state["smoothed_world_landmarks"]

            raw_metrics = compute_metrics(normalized_landmarks, world_landmarks, self._config)

            if raw_metrics is None:
                status_label = "Pose not reliable"
                info_line = "Tracking confidence is too low. Sit so shoulders and head stay visible."
                zone = "unreliable"
                state["bad_posture_start"] = None
                state["movement_refresh_streak"] = 0
                state["movement_reposition_streak"] = 0

            elif state["baseline"] is None:
                reliable_pose = True
                state["latest_metrics"] = dict(raw_metrics)
                status_label, info_line, zone = self._handle_calibration(raw_metrics, now, state)

            else:
                reliable_pose = True
                (
                    status_label,
                    info_line,
                    zone,
                    dominant_issue,
                    total_score,
                    static_duration,
                    head_delta,
                    torso_delta,
                    neck_drop,
                    shoulder_tilt_delta,
                    head_tilt_delta,
                    screen_approach_delta,
                    movement_score,
                    alert_event,
                    posture_alert_count,
                    stillness_alert_count,
                ) = self._evaluate_calibrated_posture(
                    raw_metrics=raw_metrics,
                    now=now,
                    delta_time=delta_time,
                    state=state,
                    posture_alert_count=posture_alert_count,
                    stillness_alert_count=stillness_alert_count,
                )
        else:
            state["bad_posture_start"] = None
            state["movement_refresh_streak"] = 0
            state["movement_reposition_streak"] = 0

        state["latest_pose_detected"] = pose_detected
        state["latest_reliable_pose"] = reliable_pose

        bad_duration = 0.0
        if state["bad_posture_start"] is not None:
            bad_duration = now - state["bad_posture_start"]

        snapshot = MonitoringSnapshot(
            monitoring_active=True,
            paused=False,
            preview_enabled=preview_enabled,
            calibrated=state["baseline"] is not None,
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
            reposition_count=state["reposition_count"],
            muted_until_monotonic=self._muted_until_monotonic,
            loop_latency_ms=state["loop_latency_ms"],
            baseline_generation=state["baseline_generation"],
            recalibration_reason=state["last_recalibration_reason"],
            camera_label=self._camera_label,
        )
        return snapshot, alert_event

    def _handle_calibration(self, raw_metrics: dict[str, float], now: float, state: dict) -> tuple[str, str, str]:
        state["calibration_samples"].append(raw_metrics)

        remaining = max(
            0.0,
            self._config.calibration_seconds - (now - state["calibration_start"]),
        )
        status_label = "Calibration in progress"
        info_line = f"Sit naturally and keep your upper body visible. Remaining: {remaining:0.1f}s"
        zone = "calibration"

        enough_time_passed = (now - state["calibration_start"]) >= self._config.calibration_seconds
        enough_samples = len(state["calibration_samples"]) >= self._config.calibration_min_samples

        if enough_time_passed and enough_samples:
            state["baseline"] = build_baseline(state["calibration_samples"])
            state["smoothed_metrics"] = None
            state["previous_metrics"] = None
            state["last_movement_time"] = now
            state["last_reposition_time"] = now
            state["last_movement_score"] = 0.0
            state["last_raw_movement_score"] = 0.0
            state["filtered_movement_score"] = 0.0
            state["movement_refresh_streak"] = 0
            state["movement_reposition_streak"] = 0
            state["baseline_generation"] += 1
            state["baseline_established_at"] = now
            state["auto_recalibration_pending"] = False
            state["auto_recalibration_candidate_since"] = None
            state["last_logged_zone"] = None

            self._log_event(
                kind="baseline_ready",
                title="Baseline ready",
                message=state["last_recalibration_reason"],
                state=state,
                zone="calibration",
                dominant_issue="",
                total_score=0.0,
            )

        return status_label, info_line, zone

    def _filter_movement_score(self, raw_score: float, state: dict) -> float:
        """
        Remove small jitter and smooth movement score over time.
        """
        deadband = max(0.0, float(self._config.movement_deadband))
        score_after_deadband = max(0.0, raw_score - deadband)

        previous_filtered = float(state["filtered_movement_score"])
        alpha = max(0.0, min(1.0, float(self._config.movement_score_smoothing_alpha)))
        filtered = (alpha * score_after_deadband) + ((1.0 - alpha) * previous_filtered)

        state["filtered_movement_score"] = filtered
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

    def _update_movement_timers(self, movement_score: float, now: float, state: dict) -> None:
        """
        Refresh stillness timers only after sustained meaningful movement.

        This avoids resetting static time because of tiny landmark jitter.
        """
        if movement_score >= self._config.movement_refresh_threshold:
            state["movement_refresh_streak"] += 1
        else:
            state["movement_refresh_streak"] = 0

        if movement_score >= self._config.reposition_threshold:
            state["movement_reposition_streak"] += 1
        else:
            state["movement_reposition_streak"] = 0

        if state["movement_refresh_streak"] >= self._config.movement_consecutive_frames_for_refresh:
            state["last_movement_time"] = now
            state["movement_refresh_streak"] = 0

        if state["movement_reposition_streak"] >= self._config.movement_consecutive_frames_for_reposition:
            last_reposition_time = state["last_reposition_time"]
            can_count_reposition = (
                last_reposition_time is None
                or (now - last_reposition_time) >= self._config.reposition_cooldown_seconds
            )
            if can_count_reposition:
                state["last_reposition_time"] = now
                state["last_movement_time"] = now
                state["reposition_count"] += 1
                state["auto_recalibration_pending"] = True
                state["auto_recalibration_candidate_since"] = None

            state["movement_reposition_streak"] = 0

    def _maybe_start_auto_recalibration(
        self,
        now: float,
        movement_score: float,
        evaluation: dict[str, object],
        state: dict,
    ) -> bool:
        if not self._config.auto_recalibration_enabled:
            return False

        if not state["auto_recalibration_pending"]:
            return False

        baseline_established_at = state["baseline_established_at"]
        if baseline_established_at is None:
            return False

        if (now - baseline_established_at) < self._config.auto_recalibration_min_time_since_baseline_seconds:
            return False

        last_auto_recalibration_time = state["last_auto_recalibration_time"]
        if (
            last_auto_recalibration_time is not None
            and (now - last_auto_recalibration_time) < self._config.auto_recalibration_cooldown_seconds
        ):
            return False

        total_score = float(evaluation["total_score"])
        zone = str(evaluation["zone"])

        if zone != "green" or total_score > self._config.auto_recalibration_max_score:
            state["auto_recalibration_candidate_since"] = None
            return False

        if movement_score > self._config.movement_deadband:
            state["auto_recalibration_candidate_since"] = None
            return False

        if state["auto_recalibration_candidate_since"] is None:
            state["auto_recalibration_candidate_since"] = now
            return False

        stable_duration = now - state["auto_recalibration_candidate_since"]
        if stable_duration < self._config.auto_recalibration_stability_seconds:
            return False

        state["last_auto_recalibration_time"] = now
        self._reset_for_recalibration(state, "auto_recalibration_after_reposition")
        state["auto_recalibration_started"] = True
        return True

    def _evaluate_calibrated_posture(
        self,
        raw_metrics: dict[str, float],
        now: float,
        delta_time: float,
        state: dict,
        posture_alert_count: int,
        stillness_alert_count: int,
    ) -> tuple[
        str,
        str,
        str,
        str,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        Optional[AlertEvent],
        int,
        int,
    ]:
        _ = delta_time

        smoothed_metrics = smooth_metrics(
            state["smoothed_metrics"],
            raw_metrics,
            self._config.metrics_smoothing_alpha,
        )

        raw_movement_score = compute_movement_score(
            smoothed_metrics,
            state["previous_metrics"],
            self._config,
        )
        movement_score = self._filter_movement_score(raw_movement_score, state)
        state["last_raw_movement_score"] = raw_movement_score
        state["last_movement_score"] = movement_score

        self._update_movement_timers(movement_score, now, state)

        evaluation = evaluate_ergonomics(smoothed_metrics, state["baseline"], self._config)
        state["smoothed_metrics"] = smoothed_metrics
        state["previous_metrics"] = dict(smoothed_metrics)
        state["latest_metrics"] = dict(smoothed_metrics)
        state["latest_evaluation"] = dict(evaluation)

        if self._maybe_start_auto_recalibration(now, movement_score, evaluation, state):
            return (
                "Auto recalibration",
                "Seat position changed. Updating baseline.",
                "calibration",
                "",
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                movement_score,
                None,
                posture_alert_count,
                stillness_alert_count,
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
        last_movement_time = state["last_movement_time"]
        if last_movement_time is not None:
            static_duration = max(0.0, now - last_movement_time)

        alert_event: Optional[AlertEvent] = None

        if zone == "green":
            status_label = "Neutral range"
            info_line = "You are inside your calibrated neutral zone."
            state["bad_posture_start"] = None

        elif zone == "yellow":
            status_label = "Drifting from neutral"
            if dominant_issue:
                info_line = f"Watch this trend: {dominant_issue}."
            else:
                info_line = "Small deviation detected."
            state["bad_posture_start"] = None

        else:
            status_label = "Posture overload"
            info_line = f"Main issue: {dominant_issue}."

            if state["bad_posture_start"] is None:
                state["bad_posture_start"] = now

            bad_duration = now - state["bad_posture_start"]

            can_alert_by_duration = bad_duration >= self._config.posture_alert_after_seconds
            can_alert_by_cooldown = (
                (now - state["last_posture_alert_time"])
                >= self._config.posture_alert_cooldown_seconds
            )

            if can_alert_by_duration and can_alert_by_cooldown:
                state["last_posture_alert_time"] = now
                posture_alert_count += 1
                state["posture_alert_fired"] = True
                alert_event = AlertEvent(
                    kind="posture",
                    title="Posture Guard",
                    message=f"Adjust posture: {dominant_issue}.",
                )

        can_send_stillness_reminder = (
            static_duration >= self._config.stillness_reminder_after_seconds
            and (now - state["last_stillness_alert_time"])
            >= self._config.stillness_alert_cooldown_seconds
        )

        if can_send_stillness_reminder:
            state["last_stillness_alert_time"] = now
            stillness_alert_count += 1
            state["stillness_alert_fired"] = True

            if alert_event is None:
                alert_event = AlertEvent(
                    kind="stillness",
                    title="Posture Guard",
                    message="Time to change position or move a little.",
                )

        if zone != "red" and static_duration >= self._config.stillness_reminder_after_seconds:
            status_label = "Move reminder"
            info_line = "You stayed too still for too long."

        return (
            status_label,
            info_line,
            zone,
            dominant_issue,
            total_score,
            static_duration,
            head_delta,
            torso_delta,
            neck_drop,
            shoulder_tilt_delta,
            head_tilt_delta,
            screen_approach_delta,
            movement_score,
            alert_event,
            posture_alert_count,
            stillness_alert_count,
        )

    def _build_preview_frame(self, frame, snapshot: MonitoringSnapshot, state: dict):
        rendered = frame.copy()

        smoothed_landmarks = state["smoothed_normalized_landmarks"]
        if smoothed_landmarks:
            draw_pose(rendered, smoothed_landmarks, self._config)

        draw_text(rendered, self._config.app_name, 0, (255, 255, 255))
        draw_text(rendered, self._camera_label, 1, (200, 200, 200))

        if not snapshot.calibrated:
            remaining = max(
                0.0,
                self._config.calibration_seconds - (time.monotonic() - state["calibration_start"]),
            )
            draw_text(rendered, f"Status: calibration ({remaining:0.1f}s left)", 2, (0, 255, 255))
            draw_text(
                rendered,
                f"Samples: {len(state['calibration_samples'])}/{self._config.calibration_min_samples}",
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
            pass

    def _emit_snapshot(self, snapshot: MonitoringSnapshot) -> None:
        self._latest_snapshot = snapshot
        if self._on_snapshot is not None:
            try:
                self._on_snapshot(snapshot)
            except Exception:
                pass

    def _emit_alert(self, event: AlertEvent) -> None:
        if self._on_alert is not None:
            try:
                self._on_alert(event)
            except Exception:
                pass

    def _now_iso(self) -> str:
        return datetime.now().isoformat(timespec="milliseconds")

    def _build_telemetry_row(self, snapshot: MonitoringSnapshot, state: dict) -> dict[str, object]:
        return {
            "wall_clock_iso": self._now_iso(),
            "monotonic_seconds": round(time.monotonic(), 6),
            "timestamp_ms": state.get("last_stream_timestamp_ms", 0),
            "frame_index": state.get("frame_index", 0),
            "pose_state": self._get_pose_state(snapshot),
            "zone": snapshot.zone,
            "total_score": round(snapshot.total_score, 6),
            "movement_score": round(snapshot.movement_score, 6),
            "dominant_issue": snapshot.dominant_issue,
            "baseline_generation": snapshot.baseline_generation,
            # Helper values used only for summary aggregation inside telemetry.py.
            "posture_alert_count": snapshot.posture_alert_count,
            "stillness_alert_count": snapshot.stillness_alert_count,
            "reposition_count": snapshot.reposition_count,
            "auto_recalibration_started": bool(state.get("auto_recalibration_started", False)),
        }

    def _log_frame_telemetry(self, snapshot: MonitoringSnapshot, state: dict) -> None:
        previous_zone = state.get("last_logged_zone")
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

        state["last_logged_zone"] = snapshot.zone
        self._telemetry.log_frame(self._build_telemetry_row(snapshot, state))

    def _log_event(
        self,
        kind: str,
        title: str,
        message: str,
        state: dict,
        zone: str,
        dominant_issue: str,
        total_score: float,
    ) -> None:
        self._telemetry.log_event(
            {
                "wall_clock_iso": self._now_iso(),
                "monotonic_seconds": round(time.monotonic(), 6),
                "timestamp_ms": state.get("last_stream_timestamp_ms", 0),
                "frame_index": state.get("frame_index", 0),
                "kind": kind,
                "zone": zone,
                "total_score": round(float(total_score), 6),
                "movement_score": round(float(state.get("last_movement_score", 0.0)), 6),
                "dominant_issue": dominant_issue,
                "details": f"{title}: {message}" if message else title,
                "baseline_generation": state.get("baseline_generation", 0),
            }
        )


# ============================================================
# Top-level application
# ============================================================

class PostureGuardApplication:
    """
    Top-level application object.

    It wires together configuration, monitoring, notifications and the tray menu.
    """

    def __init__(self) -> None:
        self._runtime_dir = get_runtime_directory()
        self._data_dir = get_data_directory()
        self._config_file = get_config_file_path()
        self._config = load_app_config(self._config_file)

        # Keep the mutable data folder one level above the app directory.
        save_app_config(self._config_file, self._config)

        self._exit_event = threading.Event()
        self._latest_snapshot: Optional[MonitoringSnapshot] = None

        self._notifications = NotificationManager(
            app_name=self._config.app_name,
            sound_enabled=self._config.sound_enabled,
            toast_enabled=self._config.toast_notifications_enabled,
        )

        self._engine = MonitoringEngine(
            config=self._config,
            data_dir=self._data_dir,
            on_snapshot=self._on_snapshot,
            on_alert=self._on_alert,
        )

        self._tray = TrayController(
            app_name=self._config.app_name,
            callbacks=TrayCallbacks(
                pause_monitoring=self._engine.pause,
                resume_monitoring=self._engine.resume,
                toggle_preview=self._engine.toggle_preview,
                recalibrate=self._engine.request_recalibration,
                mute_for_fifteen_minutes=self._mute_for_fifteen_minutes,
                open_app_folder=self._open_app_folder,
                exit_application=self.stop,
            ),
        )

    def start(self) -> None:
        self._install_signal_handlers()
        self._engine.start()

        tray_started = False
        if self._config.tray_enabled:
            tray_started = self._tray.start()

        if self._config.tray_enabled and not tray_started:
            print("Tray integration is unavailable. The app will continue without tray controls.")

        if not self._config.preview_enabled:
            print("Preview was disabled in config, so it is being enabled automatically on startup.")
            self._config.preview_enabled = True
            save_app_config(self._config_file, self._config)
            self._engine.toggle_preview()

        print(f"[{self._config.app_name}] running")
        print(f"App folder: {self._runtime_dir}")
        print(f"Data folder: {self._data_dir}")
        print(f"Config file: {self._config_file}")
        print("Camera preview should appear in a separate OpenCV window.")
        print("Controls in preview: R - recalibrate | P/Q/Esc - hide preview")

        try:
            while not self._exit_event.is_set():
                self._pump_preview()
                self._check_engine_health()
                time.sleep(0.01)
        finally:
            self.stop()

    def stop(self) -> None:
        if self._exit_event.is_set():
            return

        self._exit_event.set()
        self._tray.stop()
        self._engine.stop()
        self._notifications.stop()

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def _pump_preview(self) -> None:
        snapshot = self._engine.get_latest_snapshot()

        if not snapshot.preview_enabled:
            try:
                cv2.destroyWindow(self._config.app_name)
            except Exception:
                pass
            return

        frame = self._engine.get_latest_preview_frame()
        if frame is None:
            return

        cv2.imshow(self._config.app_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            self._engine.handle_preview_key(key)

    def _check_engine_health(self) -> None:
        if self._engine.is_running():
            return

        fatal_error = self._engine.get_fatal_error()
        if fatal_error:
            print(f"[{self._config.app_name}] fatal error: {fatal_error}")
        else:
            print(f"[{self._config.app_name}] monitoring engine stopped unexpectedly.")

        self.stop()

    def _on_snapshot(self, snapshot: MonitoringSnapshot) -> None:
        self._latest_snapshot = snapshot

    def _on_alert(self, event: AlertEvent) -> None:
        payload = AlertPayload(
            title=event.title,
            message=event.message,
        )
        self._notifications.notify(
            key=event.kind,
            payload=payload,
            cooldown_seconds=self._config.duplicate_notification_cooldown_seconds,
        )

    def _mute_for_fifteen_minutes(self) -> None:
        self._engine.mute_for_seconds(15 * 60)
        self._notifications.notify(
            key="mute_confirmation",
            payload=AlertPayload(
                title="Posture Guard",
                message="Alerts muted for 15 minutes.",
            ),
            cooldown_seconds=1.0,
        )

    def _open_app_folder(self) -> None:
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(self._runtime_dir))
            elif sys.platform == "darwin":
                import subprocess

                subprocess.Popen(["open", str(self._runtime_dir)])
            else:
                import subprocess

                subprocess.Popen(["xdg-open", str(self._runtime_dir)])
        except Exception:
            pass

    def _install_signal_handlers(self) -> None:
        def handle_signal(signum, frame) -> None:
            self.stop()

        try:
            signal.signal(signal.SIGINT, handle_signal)
            signal.signal(signal.SIGTERM, handle_signal)
        except Exception:
            pass


def run_application() -> None:
    application = PostureGuardApplication()
    application.start()