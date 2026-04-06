from __future__ import annotations

import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
from mediapipe.tasks.python import vision

from config import AppConfig
from config import ensure_parent_directory
from config import load_app_config
from config import read_json_file
from config import save_app_config
from config import write_json_file
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


APP_DATA_FOLDER_NAME = "PostureGuard"


# ============================================================
# Storage helpers
# ============================================================

def get_project_root() -> Path:
    """
    Return the project root in source mode.
    When frozen with PyInstaller, resources are read from _MEIPASS.
    """
    return Path(__file__).resolve().parent.parent


def get_resource_path(relative_path: str) -> Path:
    relative = Path(relative_path)

    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative

    return get_project_root() / relative


def get_writable_base_directory() -> Path:
    """
    Return a user-writable application directory.

    On Windows, LOCALAPPDATA is preferred.
    On other systems, fallback to a hidden folder in the home directory.
    """
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / APP_DATA_FOLDER_NAME

    return Path.home() / f".{APP_DATA_FOLDER_NAME.lower()}"


def get_writable_data_path(relative_path: str) -> Path:
    return get_writable_base_directory() / Path(relative_path)


def load_baseline(file_path: Path) -> Optional[dict[str, float]]:
    data = read_json_file(file_path, None)
    if not isinstance(data, dict):
        return None

    try:
        return {str(key): float(value) for key, value in data.items()}
    except Exception:
        return None


def save_baseline(file_path: Path, baseline: Optional[dict[str, float]]) -> None:
    if baseline is None:
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass
        return

    write_json_file(file_path, baseline)


def save_stats(stats_file: Path, session_summary: dict) -> None:
    """
    Append one finished session to the local JSON stats file.
    If the file is corrupted, recreate a clean structure.
    """
    ensure_parent_directory(stats_file)

    data = read_json_file(stats_file, {"sessions": []})
    if not isinstance(data, dict):
        data = {"sessions": []}

    sessions = data.setdefault("sessions", [])
    if not isinstance(sessions, list):
        data["sessions"] = []
        sessions = data["sessions"]

    sessions.append(session_summary)
    write_json_file(stats_file, data)


# ============================================================
# Camera helpers
# ============================================================

def open_camera(camera_index: int, frame_width: int, frame_height: int) -> cv2.VideoCapture:
    """
    Open the camera using a backend that works well on the current platform.
    DirectShow is often more reliable for built-in laptop webcams on Windows.
    """
    if sys.platform.startswith("win"):
        capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened() and camera_index == 0:
        capture.release()
        capture = cv2.VideoCapture(camera_index)

    if capture.isOpened():
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return capture


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
    color: str = "#ff3b30"


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

    This intentionally avoids any custom/generated alert beep.
    On Windows it uses the standard system sound.
    On other platforms it falls back to the terminal bell.
    """
    try:
        if sys.platform.startswith("win"):
            import winsound
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass


class CornerOverlay:
    """
    A tiny always-on-top square shown in the bottom-left corner of the screen.

    It runs in its own UI thread so the monitoring loop stays responsive.
    """

    def __init__(self, size_px: int, margin_px: int, alpha: float) -> None:
        self._size_px = max(8, int(size_px))
        self._margin_px = max(0, int(margin_px))
        self._alpha = max(0.2, min(1.0, float(alpha)))
        self._command_queue: queue.Queue[tuple[str, Optional[str], Optional[float]]] = queue.Queue()
        self._ready_event = threading.Event()
        self._thread = threading.Thread(target=self._run_ui, name="CornerOverlayUI", daemon=True)
        self._thread.start()
        self._ready_event.wait(timeout=2.0)

    def flash(self, color: str, duration_seconds: float) -> None:
        self._command_queue.put(("flash", color, max(0.5, float(duration_seconds))))

    def stop(self) -> None:
        self._command_queue.put(("stop", None, None))

    def _run_ui(self) -> None:
        try:
            import tkinter as tk
        except Exception:
            self._ready_event.set()
            return

        root = tk.Tk()
        root.withdraw()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", self._alpha)
        root.configure(bg="#ff3b30")

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = self._margin_px
        y = screen_height - self._size_px - self._margin_px
        root.geometry(f"{self._size_px}x{self._size_px}+{x}+{y}")

        try:
            root.wm_attributes("-toolwindow", True)
        except Exception:
            pass

        self._enable_click_through(root)
        self._ready_event.set()

        hide_job: Optional[str] = None

        def hide_overlay() -> None:
            root.withdraw()

        def poll_queue() -> None:
            nonlocal hide_job

            try:
                while True:
                    command, color, duration_seconds = self._command_queue.get_nowait()

                    if command == "stop":
                        root.destroy()
                        return

                    if command == "flash":
                        root.configure(bg=str(color))
                        root.deiconify()
                        root.lift()

                        if hide_job is not None:
                            root.after_cancel(hide_job)
                            hide_job = None

                        hide_job = root.after(int(float(duration_seconds) * 1000), hide_overlay)
            except queue.Empty:
                pass

            root.after(80, poll_queue)

        root.after(80, poll_queue)
        root.mainloop()

    def _enable_click_through(self, root) -> None:
        """
        Make the overlay ignore mouse clicks on Windows.
        """
        if not sys.platform.startswith("win"):
            return

        try:
            import ctypes

            hwnd = root.winfo_id()
            user32 = ctypes.windll.user32
            extended_style = user32.GetWindowLongW(hwnd, -20)
            extended_style |= 0x00000020  # WS_EX_TRANSPARENT
            extended_style |= 0x00000080  # WS_EX_TOOLWINDOW
            user32.SetWindowLongW(hwnd, -20, extended_style)
        except Exception:
            pass


class FallbackPopupNotifier:
    """
    Last-resort popup notifier when native Windows notifications are unavailable.

    This intentionally does not depend on external packages.
    """

    def __init__(self, app_name: str) -> None:
        self._app_name = app_name

    def show(self, title: str, message: str) -> None:
        thread = threading.Thread(
            target=self._show_popup,
            args=(title, message),
            daemon=True,
        )
        thread.start()

    def _show_popup(self, title: str, message: str) -> None:
        try:
            import tkinter as tk
        except Exception:
            return

        root = tk.Tk()
        root.title(title)
        root.attributes("-topmost", True)
        root.resizable(False, False)

        width = 340
        height = 120
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        x = max(0, screen_width - width - 24)
        y = max(0, screen_height - height - 80)

        root.geometry(f"{width}x{height}+{x}+{y}")

        frame = tk.Frame(root, padx=14, pady=12)
        frame.pack(fill="both", expand=True)

        title_label = tk.Label(frame, text=title, font=("Segoe UI", 11, "bold"), anchor="w", justify="left")
        title_label.pack(fill="x")

        message_label = tk.Label(frame, text=message, font=("Segoe UI", 10), anchor="w", justify="left", wraplength=300)
        message_label.pack(fill="both", expand=True, pady=(8, 0))

        root.after(5000, root.destroy)
        root.mainloop()


class ToastNotifier:
    """
    Best-effort desktop notification wrapper.

    The class tries several optional libraries and falls back silently
    when none are available.
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
            toast(title, message, app_id=self._app_name, duration="short")

        return send

    def _try_win10toast(self):
        try:
            from win10toast import ToastNotifier as Win10ToastNotifier
        except Exception:
            return None

        toaster = Win10ToastNotifier()

        def send(title: str, message: str) -> None:
            toaster.show_toast(title, message, duration=5, threaded=True)

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
                timeout=5,
            )

        return send


class NotificationManager:
    """
    High-level notification manager used by the monitoring engine.
    """

    def __init__(
        self,
        app_name: str,
        sound_enabled: bool,
        toast_enabled: bool,
        overlay_enabled: bool,
        overlay_size_px: int,
        overlay_margin_px: int,
        overlay_alpha: float,
        overlay_flash_seconds: float,
    ) -> None:
        self._app_name = app_name
        self._sound_enabled = sound_enabled
        self._toast_enabled = toast_enabled
        self._overlay_enabled = overlay_enabled
        self._overlay_flash_seconds = overlay_flash_seconds
        self._last_notification_times: dict[str, float] = {}
        self._lock = threading.Lock()

        set_windows_app_user_model_id(f"{app_name}.desktop")

        self._toast = ToastNotifier(app_name) if toast_enabled else None
        self._fallback_popup = FallbackPopupNotifier(app_name)
        self._overlay = CornerOverlay(overlay_size_px, overlay_margin_px, overlay_alpha) if overlay_enabled else None

    def notify(self, key: str, payload: AlertPayload, cooldown_seconds: float) -> None:
        """
        Send a notification if the per-key cooldown allows it.
        """
        now = time.monotonic()

        with self._lock:
            previous = self._last_notification_times.get(key, -1e9)
            if (now - previous) < cooldown_seconds:
                return
            self._last_notification_times[key] = now

        toast_delivered = False
        if self._toast_enabled and self._toast is not None:
            toast_delivered = self._toast.show(payload.title, payload.message)

        # If no native toast backend is available, fall back to a popup.
        # Optionally trigger only the operating system notification sound,
        # never a custom/generated beep.
        if not toast_delivered:
            if self._sound_enabled:
                threading.Thread(
                    target=play_system_notification_sound,
                    daemon=True,
                ).start()

            self._fallback_popup.show(payload.title, payload.message)

        if self._overlay_enabled and self._overlay is not None:
            self._overlay.flash(payload.color, self._overlay_flash_seconds)

    def stop(self) -> None:
        if self._overlay is not None:
            self._overlay.stop()


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
    open_data_folder: Callable[[], None]
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
            self._pystray.MenuItem("Open app data folder", self._on_open_data_folder),
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

    def _on_open_data_folder(self, icon=None, item=None) -> None:
        self._callbacks.open_data_folder()

    def _on_exit(self, icon=None, item=None) -> None:
        self._callbacks.exit_application()


# ============================================================
# Monitoring engine
# ============================================================

SnapshotCallback = Callable[["MonitoringSnapshot"], None]
AlertCallback = Callable[["AlertEvent"], None]


@dataclass
class MonitoringSnapshot:
    monitoring_active: bool
    paused: bool
    preview_enabled: bool
    calibrated: bool
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


@dataclass
class AlertEvent:
    kind: str
    title: str
    message: str
    color: str


class MonitoringEngine:
    """
    Background monitoring engine.

    It owns the camera, MediaPipe processing, calibration logic,
    posture evaluation and alert scheduling.
    """

    def __init__(
        self,
        config: AppConfig,
        on_snapshot: Optional[SnapshotCallback] = None,
        on_alert: Optional[AlertCallback] = None,
    ) -> None:
        self._config = config
        self._on_snapshot = on_snapshot
        self._on_alert = on_alert

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._lock = threading.Lock()

        self._preview_enabled = config.preview_enabled
        self._recalibration_requested = False
        self._muted_until_monotonic = 0.0
        self._latest_snapshot = MonitoringSnapshot(
            monitoring_active=False,
            paused=config.start_paused,
            preview_enabled=config.preview_enabled,
            calibrated=False,
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
        )

        if config.start_paused:
            self._pause_event.set()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="MonitoringEngine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._destroy_preview_window()

    def pause(self) -> None:
        self._pause_event.set()

    def resume(self) -> None:
        self._pause_event.clear()

    def toggle_preview(self) -> None:
        with self._lock:
            self._preview_enabled = not self._preview_enabled
            if not self._preview_enabled:
                self._destroy_preview_window()

    def request_recalibration(self) -> None:
        with self._lock:
            self._recalibration_requested = True

    def mute_for_seconds(self, seconds: float) -> None:
        self._muted_until_monotonic = max(
            self._muted_until_monotonic,
            time.monotonic() + max(0.0, seconds),
        )

    def get_latest_snapshot(self) -> MonitoringSnapshot:
        return self._latest_snapshot

    def _create_runtime_state(self, baseline: Optional[dict[str, float]]) -> dict:
        now = time.monotonic()
        return {
            "calibration_samples": [],
            "baseline": baseline,
            "smoothed_metrics": None,
            "previous_metrics": None,
            "previous_raw_landmarks": None,
            "smoothed_normalized_landmarks": None,
            "smoothed_world_landmarks": None,
            "calibration_start": now,
            "bad_posture_start": None,
            "last_posture_alert_time": -10000.0,
            "last_stillness_alert_time": -10000.0,
            "last_movement_time": now if baseline is not None else None,
            "last_reposition_time": now if baseline is not None else None,
            "last_movement_score": 0.0,
            "reposition_count": 0,
            "movement_refresh_streak": 0,
            "movement_reposition_streak": 0,
            "filtered_movement_score": 0.0,
        }

    def _reset_for_recalibration(self, state: dict) -> None:
        now = time.monotonic()
        state["calibration_samples"] = []
        state["baseline"] = None
        state["smoothed_metrics"] = None
        state["previous_metrics"] = None
        state["previous_raw_landmarks"] = None
        state["smoothed_normalized_landmarks"] = None
        state["smoothed_world_landmarks"] = None
        state["calibration_start"] = now
        state["bad_posture_start"] = None
        state["last_movement_time"] = None
        state["last_reposition_time"] = None
        state["last_movement_score"] = 0.0
        state["reposition_count"] = 0
        state["movement_refresh_streak"] = 0
        state["movement_reposition_streak"] = 0
        state["filtered_movement_score"] = 0.0

    def _run_loop(self) -> None:
        model_file = get_resource_path(self._config.model_path)
        stats_file = get_writable_data_path(self._config.stats_path)
        baseline_file = get_writable_data_path(self._config.baseline_path)

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        initial_baseline = None
        if self._config.persist_baseline_between_runs:
            initial_baseline = load_baseline(baseline_file)

        state = self._create_runtime_state(initial_baseline)

        capture = open_camera(
            self._config.camera_index,
            self._config.frame_width,
            self._config.frame_height,
        )
        if not capture.isOpened():
            raise RuntimeError("Could not open the camera.")

        options = create_landmarker_options(str(model_file), self._config)

        stream_start = time.monotonic()
        last_loop_time = time.monotonic()
        session_start_wall = time.strftime("%Y-%m-%d %H:%M:%S")
        last_processed_frame_time = 0.0
        min_frame_interval = 1.0 / self._config.inference_fps if self._config.inference_fps > 0 else 0.0

        bad_posture_total = 0.0
        static_load_total = 0.0
        posture_alert_count = 0
        stillness_alert_count = 0

        try:
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
                            self._reset_for_recalibration(state)
                            if self._config.persist_baseline_between_runs:
                                save_baseline(baseline_file, None)
                            self._recalibration_requested = False

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

                    posture_alert_count = snapshot.posture_alert_count
                    stillness_alert_count = snapshot.stillness_alert_count

                    if state["bad_posture_start"] is not None:
                        bad_posture_total += delta_time if snapshot.zone == "red" else 0.0

                    if snapshot.static_duration >= self._config.static_load_grace_seconds:
                        static_load_total += delta_time

                    self._emit_snapshot(snapshot)

                    if alert_event is not None and time.monotonic() >= self._muted_until_monotonic:
                        self._emit_alert(alert_event)

                    if preview_enabled:
                        self._render_preview_frame(frame, snapshot, state)
                        self._handle_preview_keys()
                    else:
                        self._destroy_preview_window()

        finally:
            safe_release_camera(capture)
            self._destroy_preview_window()

            if self._config.persist_baseline_between_runs:
                save_baseline(baseline_file, state["baseline"])

            session_summary = {
                "started_at": session_start_wall,
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "posture_alert_count": posture_alert_count,
                "stillness_alert_count": stillness_alert_count,
                "total_alert_count": posture_alert_count + stillness_alert_count,
                "reposition_count": state["reposition_count"],
                "bad_posture_total_seconds": round(bad_posture_total, 2),
                "static_load_total_seconds": round(static_load_total, 2),
                "was_calibrated": state["baseline"] is not None,
                "baseline": state["baseline"],
            }
            save_stats(stats_file, session_summary)

    def _handle_paused_state(self, state: dict, posture_alert_count: int, stillness_alert_count: int) -> None:
        snapshot = MonitoringSnapshot(
            monitoring_active=True,
            paused=True,
            preview_enabled=self._preview_enabled,
            calibrated=state["baseline"] is not None,
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
        )
        self._emit_snapshot(snapshot)
        self._destroy_preview_window()
        time.sleep(0.15)

    def _process_frame(
        self,
        result,
        now: float,
        delta_time: float,
        state: dict,
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
        movement_score = state["last_movement_score"]
        dominant_issue = ""
        zone = "no_person"

        if result.pose_landmarks and result.pose_world_landmarks:
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
                status_label, info_line, zone = self._handle_calibration(raw_metrics, now, state)
            else:
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

        bad_duration = 0.0
        if state["bad_posture_start"] is not None:
            bad_duration = now - state["bad_posture_start"]

        snapshot = MonitoringSnapshot(
            monitoring_active=True,
            paused=False,
            preview_enabled=preview_enabled,
            calibrated=state["baseline"] is not None,
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
            state["filtered_movement_score"] = 0.0
            state["movement_refresh_streak"] = 0
            state["movement_reposition_streak"] = 0

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

    def _update_movement_timers(self, movement_score: float, now: float, state: dict) -> None:
        """
        Refresh stillness timers only after sustained meaningful movement.

        This avoids static time being reset by tiny frame-to-frame landmark jitter.
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

            state["movement_reposition_streak"] = 0

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
        state["last_movement_score"] = movement_score

        self._update_movement_timers(movement_score, now, state)

        evaluation = evaluate_ergonomics(smoothed_metrics, state["baseline"], self._config)
        state["smoothed_metrics"] = smoothed_metrics
        state["previous_metrics"] = dict(smoothed_metrics)

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
            info_line = f"Watch this trend: {dominant_issue}."
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
                alert_event = AlertEvent(
                    kind="posture",
                    title="Posture Guard",
                    message=f"Adjust posture: {dominant_issue}.",
                    color="#ff3b30",
                )

        can_send_stillness_reminder = (
            static_duration >= self._config.stillness_reminder_after_seconds
            and (now - state["last_stillness_alert_time"])
            >= self._config.stillness_alert_cooldown_seconds
        )

        if can_send_stillness_reminder:
            state["last_stillness_alert_time"] = now
            stillness_alert_count += 1

            if alert_event is None:
                alert_event = AlertEvent(
                    kind="stillness",
                    title="Posture Guard",
                    message="Time to change position or move a little.",
                    color="#ff3b30",
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

    def _render_preview_frame(self, frame, snapshot: MonitoringSnapshot, state: dict) -> None:
        smoothed_landmarks = state["smoothed_normalized_landmarks"]
        if smoothed_landmarks:
            draw_pose(frame, smoothed_landmarks, self._config)

        draw_text(frame, self._config.app_name, 0, (255, 255, 255))

        if not snapshot.calibrated:
            remaining = max(
                0.0,
                self._config.calibration_seconds - (time.monotonic() - state["calibration_start"]),
            )
            draw_text(frame, f"Status: calibration ({remaining:0.1f}s left)", 1, (0, 255, 255))
            draw_text(
                frame,
                f"Samples: {len(state['calibration_samples'])}/{self._config.calibration_min_samples}",
                2,
                (255, 255, 255),
            )
            draw_text(frame, "Sit naturally, look at the screen, do not freeze completely.", 3, (220, 220, 220))
            draw_text(frame, "Controls: R - recalibrate | P/Q/Esc - hide preview", 5, (220, 220, 220))
        else:
            color = (255, 255, 255)
            if snapshot.zone == "green":
                color = (0, 220, 0)
            elif snapshot.zone == "yellow":
                color = (0, 200, 255)
            elif snapshot.zone == "red":
                color = (0, 0, 255)

            draw_text(frame, f"Status: {snapshot.status_label}", 1, color)
            draw_text(frame, f"Info: {snapshot.info_line}", 2, (255, 255, 255))
            draw_text(
                frame,
                f"Strain score: {snapshot.total_score:0.2f} | Static time: {snapshot.static_duration:0.1f}s",
                3,
                (255, 255, 255),
            )
            draw_text(
                frame,
                f"Head: {snapshot.head_delta:+0.3f} m | Torso: {snapshot.torso_delta:+0.1f} deg | Neck: {snapshot.neck_drop:+0.3f} m",
                4,
                (255, 255, 255),
            )
            draw_text(
                frame,
                f"Shoulders: {snapshot.shoulder_tilt_delta:+0.1f} deg | Head tilt: {snapshot.head_tilt_delta:+0.1f} deg",
                5,
                (255, 255, 255),
            )
            draw_text(
                frame,
                f"Screen approach: {snapshot.screen_approach_delta:+0.3f} | Movement score: {snapshot.movement_score:0.2f}",
                6,
                (255, 255, 255),
            )
            draw_text(
                frame,
                f"Bad posture: {snapshot.bad_duration:0.1f}s | Posture alerts: {snapshot.posture_alert_count}",
                7,
                (255, 255, 255),
            )
            draw_text(
                frame,
                f"Move reminders: {snapshot.stillness_alert_count} | Repositions: {snapshot.reposition_count}",
                8,
                (255, 255, 255),
            )
            draw_text(frame, "Controls: R - recalibrate | P/Q/Esc - hide preview", 10, (220, 220, 220))

            if snapshot.zone == "red":
                cv2.rectangle(
                    frame,
                    (10, 10),
                    (frame.shape[1] - 10, frame.shape[0] - 10),
                    (0, 0, 255),
                    4,
                )
                draw_text(frame, "ALERT: Adjust posture", 12, (0, 0, 255))

        cv2.imshow(self._config.app_name, frame)

    def _handle_preview_keys(self) -> None:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            with self._lock:
                self._recalibration_requested = True
        elif key in (ord("p"), ord("q"), 27):
            with self._lock:
                self._preview_enabled = False
            self._destroy_preview_window()

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


# ============================================================
# Top-level application
# ============================================================

class PostureGuardApplication:
    """
    Top-level application object.

    It wires together configuration, monitoring, notifications and the tray menu.
    """

    def __init__(self) -> None:
        self._data_dir = get_writable_base_directory()
        self._config_file = get_writable_data_path("data/config.json")
        self._config = load_app_config(self._config_file)
        self._config.config_path = "data/config.json"

        save_app_config(self._config_file, self._config)

        self._exit_event = threading.Event()
        self._latest_snapshot: Optional[MonitoringSnapshot] = None

        self._notifications = NotificationManager(
            app_name=self._config.app_name,
            sound_enabled=self._config.sound_enabled,
            toast_enabled=self._config.toast_notifications_enabled,
            overlay_enabled=self._config.overlay_enabled,
            overlay_size_px=self._config.overlay_size_px,
            overlay_margin_px=self._config.overlay_margin_px,
            overlay_alpha=self._config.overlay_alpha,
            overlay_flash_seconds=self._config.overlay_flash_seconds,
        )

        self._engine = MonitoringEngine(
            config=self._config,
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
                open_data_folder=self._open_data_folder,
                exit_application=self.stop,
            ),
        )

    def start(self) -> None:
        self._install_signal_handlers()
        self._engine.start()

        if self._config.tray_enabled:
            self._tray.start()

        print(f"[{self._config.app_name}] running")
        print(f"Data folder: {self._data_dir}")
        print(f"Config file: {self._config_file}")
        if not self._config.preview_enabled:
            print("Preview starts hidden. Use the tray icon to show or hide it.")

        try:
            while not self._exit_event.is_set():
                time.sleep(0.2)
        finally:
            self.stop()

    def stop(self) -> None:
        if self._exit_event.is_set():
            return

        self._exit_event.set()
        self._tray.stop()
        self._engine.stop()
        self._notifications.stop()

    def _on_snapshot(self, snapshot: MonitoringSnapshot) -> None:
        self._latest_snapshot = snapshot

    def _on_alert(self, event: AlertEvent) -> None:
        payload = AlertPayload(
            title=event.title,
            message=event.message,
            color=event.color,
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
                color="#ff9500",
            ),
            cooldown_seconds=1.0,
        )

    def _open_data_folder(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)

        try:
            if sys.platform.startswith("win"):
                os.startfile(str(self._data_dir))
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", str(self._data_dir)])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(self._data_dir)])
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