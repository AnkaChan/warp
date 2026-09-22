"""Compare CUDA ELF kernel text sections without relying on whole-CUBIN equality."""

import argparse
import hashlib
import json
import struct
from pathlib import Path


def kernel_text(path):
    data = path.read_bytes()
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError(f"Expected little-endian ELF64: {path}")
    offset = struct.unpack_from("<Q", data, 40)[0]
    entry_size, count, string_index = struct.unpack_from("<HHH", data, 58)
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, offset + i * entry_size) for i in range(count)]
    names_header = sections[string_index]
    names = data[names_header[4] : names_header[4] + names_header[5]]
    result = {}
    for section in sections:
        start = section[0]
        name = names[start : names.index(b"\0", start)].decode()
        if name.startswith(".text."):
            payload = data[section[4] : section[4] + section[5]]
            result[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
    if not result:
        raise ValueError(f"No kernel text sections: {path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--target", default="sm_89")
    args = parser.parse_args()
    rows = []
    for variant in ["closest", "any", "ordered", "count", "sign"]:
        before = args.before / variant / f"{args.target}.cubin"
        after = args.after / variant / f"{args.target}.cubin"
        a, b = kernel_text(before), kernel_text(after)
        rows.append(
            {
                "variant": variant,
                "before": str(before),
                "after": str(after),
                "before_sections": a,
                "after_sections": b,
                "kernel_text_equal": a == b,
            }
        )
    print(json.dumps({"target": args.target, "rows": rows}, indent=2))
    if not all(row["kernel_text_equal"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
