from __future__ import annotations

import os

# Reduce native library log noise as much as possible before importing runtime dependencies.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_LOG_LEVEL", "2")

import signal
import sys
import threading
import time
from typing import Optional

import cv2

from config import load_app_config, save_app_config
from engine import MonitoringEngine
from models import AlertEvent, AlertPayload, MonitoringSnapshot
from notifications import NotificationManager
from paths import get_config_file_path, get_data_directory, get_runtime_directory
from tray import TrayCallbacks, TrayController


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
                toggle_notifications=self._toggle_notifications,
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

    def _on_alert(self, event: AlertEvent) -> bool:
        if not self._config.notifications_enabled:
            return False

        payload = AlertPayload(
            title=event.title,
            message=event.message,
        )
        return self._notifications.notify(
            key=event.kind,
            payload=payload,
            cooldown_seconds=self._config.duplicate_notification_cooldown_seconds,
        )

    def _toggle_notifications(self) -> None:
        self._config.notifications_enabled = not self._config.notifications_enabled
        save_app_config(self._config_file, self._config)

        if self._config.notifications_enabled:
            self._notifications.notify(
                key="notifications_enabled_confirmation",
                payload=AlertPayload(
                    title="Posture Guard",
                    message="Notifications enabled. Telemetry is still being recorded.",
                ),
                cooldown_seconds=1.0,
            )
            print(f"[{self._config.app_name}] notifications enabled")
        else:
            print(f"[{self._config.app_name}] notifications disabled; telemetry is still being recorded")

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
