from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraOpenResult:
    capture: cv2.VideoCapture
    camera_index: int
    backend_name: str
    actual_width: int
    actual_height: int


@dataclass(frozen=True)
class CameraReadResult:
    ok: bool
    frame: Optional[Any]
    error: str = ""


def _get_camera_backend_candidates() -> list[tuple[str, int]]:
    if sys.platform.startswith("win"):
        candidates = [("CAP_DSHOW", cv2.CAP_DSHOW)]

        cap_msmf = getattr(cv2, "CAP_MSMF", None)
        if isinstance(cap_msmf, int):
            candidates.append(("CAP_MSMF", cap_msmf))

        candidates.append(("CAP_ANY", cv2.CAP_ANY))
        return candidates

    return [("CAP_ANY", cv2.CAP_ANY)]


def _get_camera_index_candidates(preferred_index: int) -> list[int]:
    indices = [max(0, int(preferred_index))]
    for fallback_index in range(3):
        if fallback_index not in indices:
            indices.append(fallback_index)
    return indices


def open_camera(camera_index: int, frame_width: int, frame_height: int) -> CameraOpenResult:
    """
    Open a working camera with backend and index fallbacks.
    """
    attempt_errors: list[str] = []

    for candidate_index in _get_camera_index_candidates(camera_index):
        for backend_name, backend_id in _get_camera_backend_candidates():
            capture = cv2.VideoCapture(candidate_index, backend_id)

            if not capture.isOpened():
                safe_release_camera(capture)
                attempt_errors.append(f"index={candidate_index}, backend={backend_name}: could not open")
                continue

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(frame_width))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(frame_height))
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            read_ok = False
            for _ in range(8):
                read_ok, _ = capture.read()
                if read_ok:
                    break
                time.sleep(0.03)

            if read_ok:
                actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                logger.info(
                    "Opened camera index=%s backend=%s resolution=%sx%s",
                    candidate_index,
                    backend_name,
                    actual_width,
                    actual_height,
                )
                return CameraOpenResult(
                    capture=capture,
                    camera_index=candidate_index,
                    backend_name=backend_name,
                    actual_width=actual_width,
                    actual_height=actual_height,
                )

            attempt_errors.append(
                f"index={candidate_index}, backend={backend_name}: opened but did not return frames"
            )
            safe_release_camera(capture)

    details = "; ".join(attempt_errors) if attempt_errors else "no camera attempts were made"
    raise RuntimeError(f"Could not open a working camera. Attempts: {details}")


def read_camera_frame(capture: Optional[cv2.VideoCapture]) -> CameraReadResult:
    if capture is None:
        return CameraReadResult(ok=False, frame=None, error="camera is not initialized")

    try:
        ok, frame = capture.read()
    except Exception as exc:
        logger.exception("Camera read raised an exception")
        return CameraReadResult(ok=False, frame=None, error=str(exc))

    if not ok or frame is None:
        return CameraReadResult(ok=False, frame=None, error="camera did not return a frame")

    return CameraReadResult(ok=True, frame=frame)


def safe_release_camera(capture: Optional[cv2.VideoCapture]) -> None:
    if capture is None:
        return

    try:
        capture.release()
    except Exception:
        logger.exception("Could not release camera cleanly")