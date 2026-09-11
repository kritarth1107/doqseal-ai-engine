"""Worker liveness heartbeat for supervisor + /health."""

from __future__ import annotations

import os
import time
from pathlib import Path

_DEFAULT = "/tmp/doqseal-worker-heartbeat"


def heartbeat_path() -> Path:
    return Path(os.getenv("WORKER_HEARTBEAT_FILE", _DEFAULT))


def touch_heartbeat() -> None:
    path = heartbeat_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(int(time.time())), encoding="utf-8")
    except Exception:
        # Never fail the worker because of heartbeat I/O
        pass


def heartbeat_age_seconds() -> float | None:
    path = heartbeat_path()
    try:
        if not path.exists():
            return None
        stamped = float(path.read_text(encoding="utf-8").strip() or "0")
        return max(0.0, time.time() - stamped)
    except Exception:
        return None
