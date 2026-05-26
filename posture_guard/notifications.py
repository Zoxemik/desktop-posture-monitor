from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from models import AlertPayload, NotificationDeliveryResult, NotificationQueueResult

logger = logging.getLogger(__name__)

DeliveryCallback = Callable[[NotificationDeliveryResult], None]


@dataclass(frozen=True)
class _NotificationWorkItem:
    key: str
    payload: AlertPayload
    play_sound: bool
    on_delivery: Optional[DeliveryCallback]


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
        logger.exception("Could not set Windows AppUserModelID")


def play_system_notification_sound() -> bool:
    """
    Play only the operating system notification sound.
    """
    try:
        if sys.platform.startswith("win"):
            import winsound

            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        else:
            print("\a", end="", flush=True)
        return True
    except Exception:
        logger.exception("Could not play notification sound")
        return False


class ToastNotifier:
    """
    Best-effort desktop notification wrapper.
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
            logger.exception("Toast backend failed")
            return False

    def _create_backend(self):
        backend = self._try_win11toast()
        if backend is not None:
            logger.info("Using win11toast notification backend")
            return backend

        backend = self._try_win10toast()
        if backend is not None:
            logger.info("Using win10toast notification backend")
            return backend

        backend = self._try_plyer()
        if backend is not None:
            logger.info("Using plyer notification backend")
            return backend

        logger.warning("No desktop toast notification backend is available")
        return None

    def _try_win11toast(self):
        try:
            from win11toast import toast
        except Exception:
            return None

        def send(title: str, message: str) -> None:
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
            # win10toast reports that the backend accepted the toast request.
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

    notify() only reports whether the notification was queued. The worker reports
    final best-effort delivery information through the optional callback.
    """

    def __init__(
        self,
        app_name: str,
        sound_enabled: bool,
        toast_enabled: bool,
        queue_size: int = 20,
    ) -> None:
        self._app_name = app_name
        self._sound_enabled = sound_enabled
        self._toast_enabled = toast_enabled
        self._last_notification_times: dict[str, float] = {}
        self._lock = threading.Lock()
        self._stopped = False

        set_windows_app_user_model_id(f"{app_name}.desktop")
        self._toast = ToastNotifier(app_name) if toast_enabled else None

        self._queue: queue.Queue[Optional[_NotificationWorkItem]] = queue.Queue(maxsize=max(1, int(queue_size)))
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="NotificationWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def notify(
        self,
        key: str,
        payload: AlertPayload,
        cooldown_seconds: float,
        on_delivery: Optional[DeliveryCallback] = None,
    ) -> NotificationQueueResult:
        """
        Queue a notification if the per-key cooldown allows it.
        """
        now = time.monotonic()

        with self._lock:
            if self._stopped:
                return NotificationQueueResult(key=key, accepted=False, reason="notification manager stopped")

            previous = self._last_notification_times.get(key, -1e9)
            if (now - previous) < max(0.0, float(cooldown_seconds)):
                return NotificationQueueResult(key=key, accepted=False, reason="cooldown")

            self._last_notification_times[key] = now

        item = _NotificationWorkItem(
            key=key,
            payload=payload,
            play_sound=self._sound_enabled,
            on_delivery=on_delivery,
        )

        try:
            self._queue.put_nowait(item)
        except queue.Full:
            logger.warning("Notification queue is full. Dropping notification: %s", key)
            self._notify_delivery_callback(
                item,
                NotificationDeliveryResult(
                    key=key,
                    title=payload.title,
                    message=payload.message,
                    queued=False,
                    toast_backend_accepted=False,
                    sound_played=False,
                    fallback_printed=False,
                    error_message="notification queue full",
                ),
            )
            return NotificationQueueResult(key=key, accepted=False, reason="queue_full")

        return NotificationQueueResult(key=key, accepted=True)

    def stop(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True

        try:
            self._queue.put_nowait(None)
        except queue.Full:
            logger.warning("Notification queue was full during shutdown")

        self._worker_thread.join(timeout=2.0)
        if self._worker_thread.is_alive():
            logger.warning("Notification worker did not stop within timeout")

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()

            if item is None:
                break

            toast_backend_accepted = False
            sound_played = False
            fallback_printed = False
            error_message: Optional[str] = None

            if self._toast_enabled and self._toast is not None:
                toast_backend_accepted = self._toast.show(item.payload.title, item.payload.message)

            if item.play_sound:
                sound_played = play_system_notification_sound()

            if not toast_backend_accepted:
                try:
                    print(f"[{self._app_name}] {item.payload.title}: {item.payload.message}")
                    fallback_printed = True
                except Exception as exc:
                    logger.exception("Could not print fallback notification")
                    error_message = str(exc)

            result = NotificationDeliveryResult(
                key=item.key,
                title=item.payload.title,
                message=item.payload.message,
                queued=True,
                toast_backend_accepted=toast_backend_accepted,
                sound_played=sound_played,
                fallback_printed=fallback_printed,
                error_message=error_message,
            )
            self._notify_delivery_callback(item, result)

    def _notify_delivery_callback(self, item: _NotificationWorkItem, result: NotificationDeliveryResult) -> None:
        if item.on_delivery is None:
            return

        try:
            item.on_delivery(result)
        except Exception:
            logger.exception("Notification delivery callback failed")