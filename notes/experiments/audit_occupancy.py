"""Annotate saved compiler resources with the standalone NVIDIA occupancy calculator."""

import argparse
import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path


def require(condition):
    if not condition:
        raise ValueError("Compiler resource or artifact verification failed")


def extra_resources(log):
    result = {}
    current = None
    for line in log.splitlines():
        entry = re.search(r"Compiling entry function '([^']+)'", line)
        if entry:
            current = entry.group(1)
            result[current] = {}
        if current and "Used " in line and "registers" in line:
            smem = re.search(r"(\d+) bytes smem", line)
            barriers = re.search(r"(?:used )?(\d+) barriers", line)
            result[current].update(
                static_shared_bytes=int(smem.group(1)) if smem else 0,
                barriers=int(barriers.group(1)) if barriers else None,
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--calculator", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    inputs = {}
    cache = {}
    paths = sorted(args.root.glob("*/*/resources.json")) + sorted(args.root.glob("cuda126/*/*/resources.json"))
    for path in paths:
        data = json.loads(path.read_text())
        relative = path.relative_to(args.root)
        version = "cuda126" if relative.parts[0] == "cuda126" else "cuda130"
        variant, module = relative.parts[-3:-1]
        inputs[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for target in data["targets"]:
            if target["status"] != "ok":
                raise ValueError(f"Incomplete or failed compile: {path}: {target['requested_target']}")
            sm = target["actual_architecture"]["sm"]
            require(target["requested_target"] == f"sm_{sm}")
            require(target["ptx"]["target"] == f"sm_{sm}")
            ptx_bytes = (path.parent / target["ptx"]["file"]).read_bytes()
            require(hashlib.sha256(ptx_bytes).hexdigest() == target["ptx"]["sha256"])
            ptx = ptx_bytes.decode()
            require(re.search(r"^\.target\s+(\w+)", ptx, re.MULTILINE).group(1) == f"sm_{sm}")
            require(set(re.findall(r"\.entry\s+(\w+)\s*\(", ptx)) == {r["name"] for r in target["resources"]})
            cubin = (path.parent / target["cubin"]).read_bytes()
            require(hashlib.sha256(cubin).hexdigest() == target["cubin_sha256"])
            require(cubin[:6] == b"\x7fELF\x02\x01")
            flags = struct.unpack_from("<I", cubin, 48)[0]
            require(cubin[8] in (7, 8))
            require((flags & 255 if cubin[8] == 7 else (flags >> 8) & 255) == sm)
            extras = extra_resources((path.parent / target["log"]).read_text())
            for resource in target["resources"]:
                extra = extras[resource["name"]]
                require(extra["static_shared_bytes"] == resource["static_shared_bytes"])
                row = {
                    "compiler": version,
                    "variant": variant,
                    "module": module,
                    "sm": sm,
                    **resource,
                    "barriers_reported": extra["barriers"],
                    "source_sha256": data["source"]["sha256"],
                    "native_headers_sha256": data["native"]["headers_sha256"],
                }
                key = (sm, resource["registers"], resource["static_shared_bytes"], extra["barriers"] or 0)
                if key not in cache:
                    calculated = subprocess.run(
                        [str(args.calculator), *map(str, key)], capture_output=True, text=True, check=False
                    )
                    cache[key] = (
                        json.loads(calculated.stdout)
                        if calculated.returncode == 0
                        else {"unavailable": calculated.stderr.strip()}
                    )
                row["occupancy"] = cache[key]
                rows.append(row)
    comparisons = []
    by_key = {(r["compiler"], r["variant"], r["module"], r["sm"], r["name"]): r for r in rows}
    for row in rows:
        if row["variant"] not in ("candidate", "count-selected"):
            continue
        references = (
            ["pinned-main", "latest-upstream", "point-parent" if row["module"] == "point" else "ray-parent"]
            if row["variant"] == "candidate"
            else ["candidate", "ray-parent", "latest-upstream"]
        )
        for reference in references:
            before = by_key.get((row["compiler"], reference, row["module"], row["sm"], row["name"]))
            if before is None:
                raise ValueError(f"Missing reference {reference} for {row['name']}")
            require(before["source_sha256"] == row["source_sha256"])
            changes = {}
            for field in ("registers", "stack_bytes", "spill_store_bytes", "spill_load_bytes", "static_shared_bytes"):
                if before[field] != row[field]:
                    changes[field] = [before[field], row[field]]
            blocks = [r["occupancy"].get("active_blocks") for r in (before, row)]
            if blocks[0] != blocks[1]:
                changes["active_blocks"] = blocks
            if changes:
                comparisons.append(
                    {
                        "compiler": row["compiler"],
                        "before": reference,
                        "after": row["variant"],
                        "module": row["module"],
                        "sm": row["sm"],
                        "name": row["name"],
                        "changes": changes,
                    }
                )
    args.out.write_text(
        json.dumps({"input_manifests_sha256": inputs, "rows": rows, "comparisons": comparisons}, indent=2) + "\n"
    )
    print(f"Audited {len(inputs)} manifests, {len(rows)} kernel/target rows; {len(comparisons)} changed comparisons")
    for comparison in comparisons:
        blocks = comparison["changes"].get("active_blocks")
        if blocks and blocks[1] < blocks[0]:
            print(json.dumps(comparison))


if __name__ == "__main__":
    main()
