"""Run CP attribution controls and temporal refit validation under one GPU claim."""

import subprocess
import sys
from pathlib import Path


def main():
    logs = Path("/home/horde/Code/AI-Docs/AI-Logs/Newton/tasks/ebvh-validation")
    jobs = []
    for scene, n, nq in (("scatter", 1_000_000, 1_000_000), ("sheet", 996_872, 499_849)):
        jobs.append((f"controls-{scene}", ["cp_controls.py", "--scene", scene, "--triangles", str(n),
                     "--queries", str(nq), "--brute-samples", "32", "--output", str(logs / "2026-09-10-cp-controls.jsonl")]))
    for scene, constructor in (("scatter", "sah"), ("sheet", "sah"), ("scatter", "lbvh")):
        jobs.append((f"temporal-{scene}-{constructor}", ["temporal_cp.py", "--scene", scene, "--constructor", constructor,
                     "--output", str(logs / "2026-09-10-cp-temporal.jsonl")]))
    jobs.append(("aabb-probe-smoke", ["aabb_probe.py", "--primitives", "10000", "--queries", "10000",
                 "--seconds", "0.1", "--repeats", "1", "--batch", "10",
                 "--output", str(logs / "2026-09-10-aabb-probe-smoke.jsonl")]))
    for label, options in jobs:
        print("START", label, flush=True)
        command = [sys.executable, "tools/ebvh_validation/" + options[0], *options[1:]]
        with (logs / f"2026-09-10-{label}.log").open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Failed {label}; see log")
        print("DONE", label, flush=True)


if __name__ == "__main__":
    main()
