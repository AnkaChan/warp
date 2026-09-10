"""Repeat the main comparison at a clock below the observed power-throttling range."""

import subprocess
import sys
from pathlib import Path


def main():
    logs = Path("/home/horde/Code/AI-Docs/AI-Logs/Newton/tasks/ebvh-validation")
    cases = [("scatter", n, 1_000_000) for n in (10000, 100000, 1000000, 10000000)]
    cases += [("sheet", n, nq) for n, nq in ((99458, 50176), (498002, 250000), (996872, 499849))]
    cases += [("scatter", 1000000, 1000000), ("sheet", 996872, 499849)] * 2
    for i, (scene, n, nq) in enumerate(cases):
        label = f"stable-{i}-{scene}-{n}"
        print("START", label, flush=True)
        command = [sys.executable, "tools/ebvh_validation/cp_controls.py", "--scene", scene,
                   "--triangles", str(n), "--queries", str(nq), "--repeats", "1",
                   "--brute-samples", "8", "--output", str(logs / "2026-09-10-cp-stable.jsonl")]
        with (logs / f"2026-09-10-{label}.log").open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Failed {label}")
        print("DONE", label, flush=True)


if __name__ == "__main__":
    main()
