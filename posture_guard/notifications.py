from __future__ import annotations

import queue
import sys
import threading
import time
from typing import Optional

from models import AlertPayload


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

    def notify(self, key: str, payload: AlertPayload, cooldown_seconds: float) -> bool:
        """
        Queue a notification if the per-key cooldown allows it.

        Returns True when the notification was accepted for delivery.
        The worker may still fall back to console output if toast delivery fails.
        """
        now = time.monotonic()

        with self._lock:
            previous = self._last_notification_times.get(key, -1e9)
            if (now - previous) < cooldown_seconds:
                return False
            self._last_notification_times[key] = now

        play_sound = self._sound_enabled
        self._queue.put((payload, play_sound))
        return True

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
