from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(data_dir: Path, app_name: str, console_enabled: bool = True) -> None:
    """
    Configure application logging once.

    Logs are written to data/logs/app.log and also to stderr when console output
    is enabled. Calling this function more than once is safe.
    """
    logger = logging.getLogger()
    if getattr(logger, "_posture_guard_configured", False):
        return

    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        logs_dir / "app.log",
        maxBytes=1_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    handlers: list[logging.Handler] = [file_handler]

    if console_enabled:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        handlers.append(console_handler)

    logger.setLevel(logging.INFO)
    for handler in handlers:
        logger.addHandler(handler)

    setattr(logger, "_posture_guard_configured", True)
    logging.getLogger(__name__).info("Logging initialized for %s", app_name)