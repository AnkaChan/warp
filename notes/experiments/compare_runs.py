"""Compare repeated harness runs; reject differing query outputs."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def collect(pattern):
    cells = defaultdict(list)
    for path in sorted(Path(".").glob(pattern)):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["kind"] == "measurement":
                cells[tuple(row["case"])].append(row)
    if not cells:
        raise ValueError(f"No measurements match {pattern}")
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--min-runs", type=int, default=3)
    args = parser.parse_args()
    before, after = collect(args.before), collect(args.after)
    if before.keys() != after.keys():
        raise ValueError("The two inputs do not contain the same benchmark cases")
    print("| Case | Before ms | After ms | Change | Repeat spread before/after |")
    print("|---|---:|---:|---:|---:|")
    for case in sorted(before):
        if min(len(before[case]), len(after[case])) < args.min_runs:
            raise ValueError(f"Fewer than {args.min_runs} complete repetitions for {case}")
        rows = before[case] + after[case]
        if not all(row["sha256"] == rows[0]["sha256"] for row in rows):
            raise ValueError(f"Output mismatch: {case}")
        a = [row["median_ms"] for row in before[case]]
        b = [row["median_ms"] for row in after[case]]
        ma, mb = statistics.median(a), statistics.median(b)
        sa, sb = 100 * (max(a) - min(a)) / ma, 100 * (max(b) - min(b)) / mb
        print(
            f"| {'/'.join(map(str, case))} | {ma:.4f} | {mb:.4f} | {(mb / ma - 1) * 100:+.2f}% | {sa:.2f}% / {sb:.2f}% |"
        )


if __name__ == "__main__":
    main()
