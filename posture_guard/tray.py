from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class TrayCallbacks:
    pause_monitoring: Callable[[], None]
    resume_monitoring: Callable[[], None]
    toggle_preview: Callable[[], None]
    recalibrate: Callable[[], None]
    toggle_notifications: Callable[[], None]
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
            logger.info("System tray integration is unavailable", exc_info=True)
            return

        self._pystray = pystray
        self._image_class = Image
        self._image_draw_class = ImageDraw
        self._available = True

    def start(self) -> bool:
        if not self._available:
            return False

        try:
            icon_image = self._create_icon_image()
            menu = self._pystray.Menu(
                self._pystray.MenuItem("Pause monitoring", self._on_pause_monitoring),
                self._pystray.MenuItem("Resume monitoring", self._on_resume_monitoring),
                self._pystray.MenuItem("Show / hide preview", self._on_toggle_preview),
                self._pystray.MenuItem("Recalibrate", self._on_recalibrate),
                self._pystray.MenuItem("Enable / disable notifications", self._on_toggle_notifications),
                self._pystray.MenuItem("Open app folder", self._on_open_app_folder),
                self._pystray.MenuItem("Exit", self._on_exit),
            )

            self._icon = self._pystray.Icon(self._app_name, icon_image, self._app_name, menu)
            self._icon.run_detached()
            logger.info("System tray icon started")
            return True
        except Exception:
            logger.exception("Could not start system tray icon")
            return False

    def stop(self) -> None:
        if self._icon is None:
            return

        try:
            self._icon.stop()
        except Exception:
            logger.exception("Could not stop system tray icon cleanly")

    def _create_icon_image(self):
        image = self._image_class.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = self._image_draw_class.Draw(image)

        draw.rounded_rectangle((8, 8, 56, 56), radius=12, fill=(33, 170, 88, 255))
        draw.rounded_rectangle((40, 40, 56, 56), radius=5, fill=(220, 50, 47, 255))
        return image

    def _run_callback(self, name: str, callback: Callable[[], None]) -> None:
        try:
            callback()
        except Exception:
            logger.exception("Tray callback failed: %s", name)

    def _on_pause_monitoring(self, icon=None, item=None) -> None:
        self._run_callback("pause_monitoring", self._callbacks.pause_monitoring)

    def _on_resume_monitoring(self, icon=None, item=None) -> None:
        self._run_callback("resume_monitoring", self._callbacks.resume_monitoring)

    def _on_toggle_preview(self, icon=None, item=None) -> None:
        self._run_callback("toggle_preview", self._callbacks.toggle_preview)

    def _on_recalibrate(self, icon=None, item=None) -> None:
        self._run_callback("recalibrate", self._callbacks.recalibrate)

    def _on_toggle_notifications(self, icon=None, item=None) -> None:
        self._run_callback("toggle_notifications", self._callbacks.toggle_notifications)

    def _on_open_app_folder(self, icon=None, item=None) -> None:
        self._run_callback("open_app_folder", self._callbacks.open_app_folder)

    def _on_exit(self, icon=None, item=None) -> None:
        self._run_callback("exit_application", self._callbacks.exit_application)