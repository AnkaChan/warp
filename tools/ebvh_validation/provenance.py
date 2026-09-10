"""Capture native sources and environment for each benchmark record."""

import hashlib
import os
import subprocess
from pathlib import Path


def source_record():
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in ("warp/native/bvh.h", "warp/native/mesh.h", "warp/bin/warp.so")
        },
        "requested_clock_mhz": os.environ.get("EBVH_CLOCK_MHZ"),
        "clock_locked": os.environ.get("EBVH_CLOCK_LOCKED") == "1",
    }
