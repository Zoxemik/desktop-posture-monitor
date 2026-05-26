from __future__ import annotations

import logging
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
from logging_setup import configure_logging
from models import AlertEvent, AlertPayload, MonitoringSnapshot, NotificationQueueResult
from notifications import NotificationManager
from paths import get_config_file_path, get_data_directory, get_runtime_directory
from tray import TrayCallbacks, TrayController

logger = logging.getLogger(__name__)


class PostureGuardApplication:
    """
    Top-level application object.

    It wires together configuration, monitoring, notifications and the tray menu.
    """

    def __init__(self) -> None:
        self._runtime_dir = get_runtime_directory()
        self._data_dir = get_data_directory()
        self._config_file = get_config_file_path()

        # Logging is initialized before loading config so config errors are visible.
        configure_logging(self._data_dir, "Posture Guard")

        self._config = load_app_config(self._config_file)
        save_app_config(self._config_file, self._config)

        self._exit_event = threading.Event()
        self._latest_snapshot: Optional[MonitoringSnapshot] = None
        self._preview_window_was_visible = False

        self._notifications = NotificationManager(
            app_name=self._config.app_name,
            sound_enabled=self._config.sound_enabled,
            toast_enabled=self._config.toast_notifications_enabled,
            queue_size=self._config.notification_queue_size,
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
            logger.warning("Tray integration is unavailable. The app will continue without tray controls.")

        if not self._config.preview_enabled:
            logger.info("Preview was disabled in config, so it is being enabled automatically on startup.")
            self._config.preview_enabled = True
            save_app_config(self._config_file, self._config)
            self._engine.toggle_preview()

        logger.info("[%s] running", self._config.app_name)
        logger.info("App folder: %s", self._runtime_dir)
        logger.info("Data folder: %s", self._data_dir)
        logger.info("Config file: %s", self._config_file)
        logger.info("Camera preview should appear in a separate OpenCV window.")
        logger.info("Controls in preview: R - recalibrate | P/Q/Esc - hide preview")

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

        logger.info("Stopping application")

        self._tray.stop()
        self._engine.stop()
        self._notifications.stop()

        try:
            cv2.destroyAllWindows()
        except Exception:
            logger.exception("Could not destroy OpenCV windows during application shutdown")

    def _pump_preview(self) -> None:
        snapshot = self._engine.get_latest_snapshot()

        if not snapshot.preview_enabled:
            self._preview_window_was_visible = False
            self._destroy_preview_window()
            return

        frame = self._engine.get_latest_preview_frame()
        if frame is None:
            return

        window_name = self._config.app_name

        try:
            cv2.imshow(window_name, frame)
            self._preview_window_was_visible = True

            # If the user closes the OpenCV window with X, hide preview instead of
            # recreating the window and freezing the UI event loop.
            if self._is_preview_window_closed_by_user(window_name):
                logger.info("Preview window was closed by user")
                self._engine.hide_preview()
                self._preview_window_was_visible = False
                self._destroy_preview_window()
                return

            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                self._engine.handle_preview_key(key)

        except Exception:
            logger.exception("OpenCV preview pump failed")
            self._engine.hide_preview()
            self._preview_window_was_visible = False
            self._destroy_preview_window()

    def _is_preview_window_closed_by_user(self, window_name: str) -> bool:
        if not self._preview_window_was_visible:
            return False

        try:
            return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
        except Exception:
            # Some OpenCV builds throw when the window no longer exists.
            return True

    def _destroy_preview_window(self) -> None:
        try:
            cv2.destroyWindow(self._config.app_name)
        except Exception:
            logger.debug("Preview window was already closed or unavailable", exc_info=True)

    def _check_engine_health(self) -> None:
        if self._engine.is_running():
            return

        fatal_error = self._engine.get_fatal_error()
        if fatal_error:
            logger.error("[%s] fatal error: %s", self._config.app_name, fatal_error)
        else:
            logger.error("[%s] monitoring engine stopped unexpectedly.", self._config.app_name)

        self.stop()

    def _on_snapshot(self, snapshot: MonitoringSnapshot) -> None:
        self._latest_snapshot = snapshot

    def _on_alert(self, event: AlertEvent, on_delivery=None):
        """
        Queue a notification.

        Backward compatible behavior:
        - old engine calls this with only event and receives bool,
        - updated engine calls it with on_delivery and receives NotificationQueueResult.
        """
        if not self._config.notifications_enabled:
            if on_delivery is None:
                return False
            return NotificationQueueResult(
                key=event.kind,
                accepted=False,
                reason="notifications_disabled",
            )

        payload = AlertPayload(
            title=event.title,
            message=event.message,
        )

        result = self._notifications.notify(
            key=event.kind,
            payload=payload,
            cooldown_seconds=self._config.duplicate_notification_cooldown_seconds,
            on_delivery=on_delivery,
        )

        if on_delivery is None:
            return result.accepted

        return result

    def _toggle_notifications(self) -> None:
        self._config.notifications_enabled = not self._config.notifications_enabled
        saved = save_app_config(self._config_file, self._config)

        if not saved:
            logger.warning("Notification setting changed in memory but could not be saved")

        if self._config.notifications_enabled:
            self._notifications.notify(
                key="notifications_enabled_confirmation",
                payload=AlertPayload(
                    title="Posture Guard",
                    message="Notifications enabled. Telemetry is still being recorded.",
                ),
                cooldown_seconds=1.0,
            )
            logger.info("[%s] notifications enabled", self._config.app_name)
        else:
            logger.info("[%s] notifications disabled; telemetry is still being recorded", self._config.app_name)

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
            logger.exception("Could not open app folder: %s", self._runtime_dir)

    def _install_signal_handlers(self) -> None:
        def handle_signal(signum, frame) -> None:
            logger.info("Received signal %s", signum)
            self.stop()

        try:
            signal.signal(signal.SIGINT, handle_signal)
            signal.signal(signal.SIGTERM, handle_signal)
        except Exception:
            logger.exception("Could not install signal handlers")


def run_application() -> None:
    application = PostureGuardApplication()
    application.start()