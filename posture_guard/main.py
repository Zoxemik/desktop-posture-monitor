from __future__ import annotations

import os

# Reduce native library log noise before runtime dependencies are imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_LOG_LEVEL", "2")

from app import run_application


if __name__ == "__main__":
    run_application()
