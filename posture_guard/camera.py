from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Optional

import cv2

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
