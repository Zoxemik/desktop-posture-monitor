from __future__ import annotations

import sys
from pathlib import Path

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
