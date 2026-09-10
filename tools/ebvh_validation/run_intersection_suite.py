"""Run fixed-clock intersection comparisons in independent processes."""

import os
import subprocess
import sys
from pathlib import Path


def main():
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Use run_claimed.sh")
    scripts = Path(__file__).resolve().parent
    task = Path("/home/horde/Code/AI-Docs/AI-Logs/Newton/tasks/ebvh-validation")
    cases = []
    for n in (10000, 100000, 1000000):
        cases.append(("aabb_probe.py", ["--primitives", str(n), "--queries", "1000000"], "aabb-final"))
    for n in (10000, 1000000):
        for length in (0.1, 20.0):
            cases.append(("ray_benchmark.py", ["--triangles", str(n), "--length", str(length)], "ray-final"))
    for scene, n in (("scatter", 10000), ("scatter", 1000000), ("sheet", 1000000)):
        cases.append(("temporal_aabb.py", ["--scene", scene, "--primitives", str(n)], "aabb-temporal"))
    for index, (script, arguments, suffix) in enumerate(cases):
        command = [sys.executable, str(scripts / script), *arguments,
                   "--output", str(task / f"2026-09-10-{suffix}.jsonl")]
        print("CASE", index + 1, "of", len(cases), command, flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
