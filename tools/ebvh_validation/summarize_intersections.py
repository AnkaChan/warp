"""Summarize paired intersection timings and verify observed clocks."""

import argparse
import csv
import json
import statistics
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    from pathlib import Path

    directory = Path(args.directory)
    run_ids = set()
    for suffix in ("aabb-final", "ray-final", "aabb-temporal"):
        path = directory / f"2026-09-10-{suffix}.jsonl"
        if not path.exists():
            continue
        groups = defaultdict(list)
        for record in map(json.loads, path.read_text().splitlines()):
            run_ids.add(record["run_id"])
            options = record["args"]
            key = (
                options.get("scene", "scatter"),
                options.get("primitives", options.get("triangles")),
                options["queries"],
                options.get("length"),
            )
            timings = record.get("timings")
            if timings is None:
                timings = [frame["timing"] for frame in record["frames"]]
            means = {
                name: statistics.mean(t[name]["mean_ms"] for t in timings)
                for name in timings[0]
            }
            groups[key].append((record, means))
        print(f"\n{suffix}")
        for key, records in groups.items():
            print("CASE", key, "processes", len(records))
            for name in records[0][1]:
                times = [m[name] for _, m in records if name in m]
                ratios = [m["root"] / m[name] for _, m in records if name in m]
                print(
                    name,
                    "ms",
                    round(statistics.median(times), 6),
                    "speedup",
                    round(statistics.median(ratios), 4),
                    "range",
                    [round(min(ratios), 4), round(max(ratios), 4)],
                    "processes",
                    len(times),
                )
            record, means = records[0]
            if "stackless_fallback_fraction" in record:
                print(
                    "fallback",
                    record["stackless_fallback_fraction"],
                    "mean_depth",
                    record["mean_depth"],
                    "mean_hits",
                    record["mean_hits"],
                )
            if "frames" in record:
                refits = [
                    statistics.mean(f["refit_ms"] for f in r["frames"][1:])
                    for r, _ in records
                ]
                ordinary_refits = [
                    statistics.mean(f["ordinary_refit_ms"] for f in r["frames"][1:])
                    for r, _ in records
                ]
                total_ratios = [
                    sum(
                        f["ordinary_refit_ms"] + f["timing"]["ordinary_root"]["mean_ms"]
                        for f in r["frames"][1:]
                    )
                    / sum(
                        f["refit_ms"] + f["timing"]["fused"]["mean_ms"]
                        for f in r["frames"][1:]
                    )
                    for r, _ in records
                ]
                print(
                    "refit_frames_2_to_25_ms",
                    statistics.median(refits),
                    "ordinary_refit_ms",
                    statistics.median(ordinary_refits),
                    "total_vs_ordinary",
                    statistics.median(total_ratios),
                    "mean_hits",
                    statistics.mean(f["mean_hits"] for f in record["frames"]),
                )
                existing_pipeline_ratios = [
                    sum(
                        f["refit_ms"] + f["timing"]["cached_with_refresh"]["mean_ms"]
                        for f in r["frames"][1:]
                    )
                    / sum(
                        f["refit_ms"] + f["timing"]["fused"]["mean_ms"]
                        for f in r["frames"][1:]
                    )
                    for r, _ in records
                ]
                print(
                    "total_vs_existing_cached_pipeline",
                    statistics.median(existing_pipeline_ratios),
                )
                diagnosed = [
                    f
                    for r, _ in records
                    for f in r["frames"]
                    if "cached_stackless_fraction" in f
                ]
                if diagnosed:
                    print(
                        "mean_cached_fallback_fraction",
                        statistics.mean(
                            f["cached_stackless_fraction"] for f in diagnosed
                        ),
                        "mean_cached_depth",
                        statistics.mean(f["cached_mean_depth"] for f in diagnosed),
                    )
            if "hit_fraction" in record:
                print(
                    "hit_fraction",
                    record["hit_fraction"],
                    "changed_face_fraction",
                    record["changed_face_fraction"],
                )
    for run_id in sorted(run_ids):
        telemetry = directory / f"{run_id}-telemetry.csv"
        with telemetry.open() as stream:
            rows = list(csv.DictReader(stream, skipinitialspace=True))
        active = [r for r in rows if float(r["utilization.gpu [%]"].split()[0]) > 10]
        clocks = sorted({int(r["clocks.current.sm [MHz]"].split()[0]) for r in active})
        memory_clocks = sorted(
            {int(r["clocks.current.memory [MHz]"].split()[0]) for r in active}
        )
        print(
            "TELEMETRY",
            run_id,
            "active_samples",
            len(active),
            "observed_sm_mhz",
            clocks,
            "observed_memory_mhz",
            memory_clocks,
        )


if __name__ == "__main__":
    main()
