"""Run exact CP matrix in independent processes inside a single GPU claim."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("primary", "sensitivity"), default="primary")
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Use run_claimed.sh")
    log_dir = Path("/home/horde/Code/AI-Docs/AI-Logs/Newton/tasks/ebvh-validation")
    output = log_dir / f"2026-09-10-cp-{args.suite}.jsonl"
    if args.suite == "primary":
        configs = [("scatter", n, 1_000_000, []) for n in (10_000, 100_000, 1_000_000, 10_000_000)]
        configs += [("sheet", n, nq, []) for n, nq in ((99_458, 50_176), (498_002, 250_000), (996_872, 499_849))]
        repeats = 3
    else:
        configs = [(scene, 1_000_000, 250_000, extra) for scene in ("scatter", "sheet")
                   for extra in (["--block-dim", "16"], ["--block-dim", "128"],
                                 ["--leaf-size", "4"], ["--constructor", "lbvh"], ["--unsorted"])]
        repeats = 1
    for repeat in range(repeats):
        order = configs if repeat % 2 == 0 else list(reversed(configs))
        for scene, n, nq, extra in order:
            label = f"{scene}-{n}-r{repeat}-{'-'.join(extra) or 'default'}"
            command = [sys.executable, "tools/ebvh_validation/closest_point.py", "--scene", scene,
                       "--triangles", str(n), "--queries", str(nq), "--repeats", "1",
                       "--brute-samples", "8" if n >= 10_000_000 else "32", "--output", str(output), *extra]
            print("START", label, flush=True)
            with (log_dir / f"2026-09-10-cp-{args.suite}-{label}.log").open("w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Failed: {label}; see per-case log")
            print("DONE", label, flush=True)


if __name__ == "__main__":
    main()
