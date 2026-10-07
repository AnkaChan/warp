"""Cross-compile Warp query kernels without initializing CUDA or loading Warp.

Example:
    uv run --no-project notes/experiments/compile_resources.py \
        --source path/to/generated.cu --native warp/native --out notes/resources/main

Pass ``--nvrtc`` for another compiler and repeat ``--cuda-include`` or
``--option=...`` to reproduce additional options from a Warp compilation log.
The output directory must be empty. Unsupported targets remain explicit errors;
the tool never lowers a requested architecture or substitutes PTX for CUBIN.
"""

import argparse
import ctypes
import hashlib
import json
import re
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_NVRTC = Path("/home/horde/.cache/uv/archive-v0/pNJEmc_vesuVb-hiBMK0Q/nvidia/cu13/lib/libnvrtc.so.13")


def parse_ptxas_resources(log):
    """Extract per-function resources without treating missing lines as zero."""
    records = {}
    current = None
    for line in log.splitlines():
        entry = re.search(r"Compiling entry function '([^']+)' for '([^']+)'", line)
        function = re.search(r"Function properties for\s+(.+?)\s*$", line)
        if entry or function:
            name = entry.group(1) if entry else function.group(1).strip("'")
            current = records.setdefault(
                name,
                {
                    "name": name,
                    "ptxas_target": None,
                    "registers": None,
                    "stack_bytes": None,
                    "spill_store_bytes": None,
                    "spill_load_bytes": None,
                    "static_shared_bytes": None,
                },
            )
            if entry:
                current["ptxas_target"] = entry.group(2)
        if current is None:
            continue
        stack = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if stack:
            current.update(
                stack_bytes=int(stack.group(1)),
                spill_store_bytes=int(stack.group(2)),
                spill_load_bytes=int(stack.group(3)),
            )
        registers = re.search(r"Used\s+(\d+)\s+registers", line)
        if registers:
            current["registers"] = int(registers.group(1))
            shared = re.search(r"(\d+) bytes smem", line)
            current["static_shared_bytes"] = int(shared.group(1)) if shared else 0
    return list(records.values())


def cubin_architecture(data):
    """Read CUDA's SM field from ELF e_flags, retaining the complete flags."""
    if len(data) < 52 or data[:4] != b"\x7fELF":
        raise ValueError("NVRTC output is not an ELF CUBIN")
    elf_class, encoding = data[4], data[5]
    if elf_class not in (1, 2) or encoding not in (1, 2):
        raise ValueError("Unsupported ELF class or byte order")
    if elf_class == 2 and len(data) < 64:
        raise ValueError("Truncated ELF64 header")
    endian = "<" if encoding == 1 else ">"
    machine = struct.unpack_from(endian + "H", data, 18)[0]
    if machine != 190:  # EM_CUDA
        raise ValueError(f"ELF machine {machine} is not CUDA (190)")
    flags = struct.unpack_from(endian + "I", data, 48 if elf_class == 2 else 36)[0]
    abi_version = data[8]
    if abi_version not in (7, 8):
        raise ValueError(f"Unknown CUDA ELF ABI version {abi_version}")
    # CUDA 13 emits ABI 8 even for older SMs. LLVM's getNVPTXCPUName()
    # uses the low byte for ABI 7 and bits 8..15 for ABI 8:
    # https://github.com/llvm/llvm-project/blob/main/llvm/lib/Object/ELFObjectFile.cpp
    sm = (flags & 0xFF) if abi_version == 7 else ((flags >> 8) & 0xFF)
    return {
        "elf_class": 64 if elf_class == 2 else 32,
        "elf_machine": machine,
        "elf_abi_version": abi_version,
        "elf_flags": f"0x{flags:08x}",
        "sm": sm,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def native_provenance(native):
    headers = {
        str(path.relative_to(native)): sha256_file(path)
        for path in sorted(native.rglob("*"))
        if path.is_file() and path.suffix in (".h", ".hpp", ".cuh", ".inl")
    }
    commit = subprocess.run(
        ["git", "-C", str(native), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return {
        "path": str(native),
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "headers_sha256": hashlib.sha256(json.dumps(headers, sort_keys=True).encode()).hexdigest(),
        "header_files": headers,
    }


class Nvrtc:
    """Use only compiler APIs; this class does not load the CUDA driver."""

    def __init__(self, path):
        self.lib = ctypes.CDLL(str(path))
        pointer = ctypes.c_void_p
        char_ptr = ctypes.c_char_p
        int_ptr = ctypes.POINTER(ctypes.c_int)
        size_ptr = ctypes.POINTER(ctypes.c_size_t)
        signatures = {
            "nvrtcVersion": [int_ptr, int_ptr],
            "nvrtcGetNumSupportedArchs": [int_ptr],
            "nvrtcGetSupportedArchs": [int_ptr],
            "nvrtcCreateProgram": [
                ctypes.POINTER(pointer),
                char_ptr,
                char_ptr,
                ctypes.c_int,
                ctypes.POINTER(char_ptr),
                ctypes.POINTER(char_ptr),
            ],
            "nvrtcCompileProgram": [pointer, ctypes.c_int, ctypes.POINTER(char_ptr)],
            "nvrtcDestroyProgram": [ctypes.POINTER(pointer)],
            "nvrtcGetProgramLogSize": [pointer, size_ptr],
            "nvrtcGetProgramLog": [pointer, pointer],
            "nvrtcGetCUBINSize": [pointer, size_ptr],
            "nvrtcGetCUBIN": [pointer, pointer],
            "nvrtcGetPTXSize": [pointer, size_ptr],
            "nvrtcGetPTX": [pointer, pointer],
        }
        for name, arguments in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes = arguments
            function.restype = ctypes.c_int
        self.lib.nvrtcGetErrorString.argtypes = [ctypes.c_int]
        self.lib.nvrtcGetErrorString.restype = char_ptr
        major, minor = ctypes.c_int(), ctypes.c_int()
        self.check(self.lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)))
        self.version = [major.value, minor.value]
        # Wheel libraries may live outside the dynamic linker's search path.
        # Preload the companion by its absolute path so NVRTC's later dlopen
        # resolves the same SONAME without modifying the process environment.
        builtins = path.parent / f"libnvrtc-builtins.so.{major.value}.{minor.value}"
        self.builtins = ctypes.CDLL(str(builtins), mode=ctypes.RTLD_GLOBAL) if builtins.is_file() else None
        count = ctypes.c_int()
        self.check(self.lib.nvrtcGetNumSupportedArchs(ctypes.byref(count)))
        architectures = (ctypes.c_int * count.value)()
        self.check(self.lib.nvrtcGetSupportedArchs(architectures))
        self.supported_architectures = list(architectures)

    def error_string(self, result):
        return self.lib.nvrtcGetErrorString(result).decode()

    def check(self, result):
        if result:
            raise RuntimeError(f"{self.error_string(result)} ({result})")

    def buffer(self, program, kind):
        size = ctypes.c_size_t()
        self.check(getattr(self.lib, f"nvrtcGet{kind}Size")(program, ctypes.byref(size)))
        data = ctypes.create_string_buffer(size.value)
        self.check(getattr(self.lib, f"nvrtcGet{kind}")(program, data))
        return data.raw

    def compile(self, source, name, options, emit_ptx=False):
        program = ctypes.c_void_p()
        self.check(self.lib.nvrtcCreateProgram(ctypes.byref(program), source, name.encode(), 0, None, None))
        try:
            encoded_options = (ctypes.c_char_p * len(options))(*(option.encode() for option in options))
            result = self.lib.nvrtcCompileProgram(program, len(options), encoded_options)
            log = self.buffer(program, "ProgramLog").rstrip(b"\0").decode(errors="replace")
            cubin = self.buffer(program, "CUBIN") if result == 0 else None
            ptx = self.buffer(program, "PTX").rstrip(b"\0") if result == 0 and emit_ptx else None
            return result, log, cubin, ptx
        finally:
            self.lib.nvrtcDestroyProgram(ctypes.byref(program))


def parse_target(value):
    match = re.fullmatch(r"(?:sm_)?(\d+)([af]?)", value)
    if not match:
        raise argparse.ArgumentTypeError(f"Invalid CUDA target: {value}")
    return f"sm_{match.group(1)}{match.group(2)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nvrtc", type=Path, default=DEFAULT_NVRTC)
    parser.add_argument("--all-supported", action="store_true", help="Compile every integer target reported by NVRTC")
    parser.add_argument("--emit-ptx", action="store_true", help="Save and check PTX as well as the assembled CUBIN")
    parser.add_argument(
        "--targets", type=parse_target, nargs="+", default=["sm_89", "sm_100", "sm_103", "sm_110", "sm_120", "sm_121"]
    )
    parser.add_argument(
        "--block-size", type=int, default=256, help="Intended runtime block size; does not modify the source"
    )
    parser.add_argument("--cuda-include", type=Path, action="append", default=[])
    parser.add_argument("--fmad", choices=("true", "false"), default="true", help="Match Warp's fuse_fp module option")
    parser.add_argument(
        "--option", action="append", default=[], help="Additional NVRTC option, passed as --option=--flag"
    )
    args = parser.parse_args()
    args.source, args.native, args.out, args.nvrtc = (
        path.resolve() for path in (args.source, args.native, args.out, args.nvrtc)
    )
    if not args.source.is_file() or not (args.native / "builtin.h").is_file():
        parser.error("--source must exist and --native must contain Warp's builtin.h")
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("--out must be empty; use a distinct directory for each source/native variant")
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.source.read_bytes()
    (args.out / "source.cu").write_bytes(source)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "helper_sha256": sha256_file(Path(__file__)),
        "source": {"path": str(args.source), "sha256": hashlib.sha256(source).hexdigest()},
        "native": native_provenance(args.native),
        "compiler": {"library": str(args.nvrtc)},
        "block_size": args.block_size,
        "source_launch_bounds": re.findall(r"__launch_bounds__\s*\(([^)]+)\)", source.decode(errors="replace")),
        "runtime_measured": False,
        "targets": [],
    }

    def save():
        (args.out / "resources.json").write_text(json.dumps(manifest, indent=2) + "\n")

    try:
        compiler = Nvrtc(args.nvrtc)
        manifest["compiler"].update(
            version=compiler.version,
            supported_architectures=compiler.supported_architectures,
            library_sha256=sha256_file(args.nvrtc),
        )
    except (OSError, RuntimeError) as error:
        manifest["compiler"]["error"] = str(error)
        save()
        print(f"NVRTC unavailable: {error}", file=sys.stderr)
        return 1

    targets = [f"sm_{arch}" for arch in compiler.supported_architectures] if args.all_supported else args.targets
    for target in targets:
        target_number = int(re.search(r"\d+", target).group())
        options = [
            f"--gpu-architecture={target}",
            f"--include-path={args.native}",
            "--std=c++17",
            "--define-macro=NDEBUG",
            "--undefine-macro=WP_VERIFY_FP",
            "--define-macro=WP_ENABLE_MATHDX=0",
            f"--fmad={args.fmad}",
            "--device-as-default-execution-space",
            "--extra-device-vectorization",
            "--restrict",
            "--diag-suppress=177,550",
            "--ptxas-options=-v",
        ]
        if compiler.version >= [12, 9]:
            options.append("--Ofast-compile=0")
        if compiler.version >= [13, 0]:
            # A cached CUBIN suppresses PTXAS's verbose resource messages.
            # Keep the investigation independent of NVRTC's shared disk cache.
            options.append("--no-cache")
        options.extend(f"--include-path={path.resolve()}" for path in args.cuda_include)
        options.extend(args.option)
        row = {"requested_target": target, "options": options, "status": "pending"}
        manifest["targets"].append(row)
        print(f"Compiling {target}: {json.dumps(options)}", flush=True)
        started = time.perf_counter()
        try:
            if target_number not in compiler.supported_architectures:
                row.update(status="unsupported", error="Architecture absent from NVRTC's supported architecture list")
                log = row["error"] + "\n"
            else:
                result, log, cubin, ptx = compiler.compile(source, args.source.name, options, args.emit_ptx)
                row["nvrtc_result"] = result
                row["resources"] = parse_ptxas_resources(log)
                if result:
                    row.update(status="compile_error", error=compiler.error_string(result))
                else:
                    cubin_path = args.out / f"{target}.cubin"
                    cubin_path.write_bytes(cubin)
                    row.update(status="ok", cubin=cubin_path.name, cubin_sha256=hashlib.sha256(cubin).hexdigest())
                    row["actual_architecture"] = cubin_architecture(cubin)
                    if row["actual_architecture"]["sm"] != target_number:
                        row.update(status="architecture_mismatch", error="ELF SM differs from requested target")
                    if ptx is not None:
                        ptx_path = args.out / f"{target}.ptx"
                        ptx_path.write_bytes(ptx)
                        ptx_text = ptx.decode()
                        ptx_target = re.search(r"(?m)^\s*\.target\s+(sm_\d+[af]?)", ptx_text)
                        ptx_version = re.search(r"(?m)^\s*\.version\s+([\d.]+)", ptx_text)
                        row["ptx"] = {
                            "file": ptx_path.name,
                            "sha256": hashlib.sha256(ptx).hexdigest(),
                            "target": ptx_target.group(1) if ptx_target else None,
                            "version": ptx_version.group(1) if ptx_version else None,
                            "entries": re.findall(r"\.entry\s+([^\s(]+)", ptx_text),
                        }
                        if row["ptx"]["target"] != target or not row["ptx"]["entries"]:
                            row.update(status="ptx_mismatch", error="PTX target differs or PTX has no entry points")
                    if not row["resources"] or any(item["registers"] is None for item in row["resources"]):
                        row["resource_warning"] = (
                            "Some resource counts are unavailable; inspect the raw compilation log"
                        )
            (args.out / f"{target}.log").write_text(log)
            row["log"] = f"{target}.log"
        except (OSError, RuntimeError, ValueError) as error:
            row.update(status="error", error=str(error))
        row["compile_seconds"] = time.perf_counter() - started
        save()
        print(f"{target}: {row['status']} ({row['compile_seconds']:.2f}s)", flush=True)
    print(args.out / "resources.json", flush=True)
    return int(any(row["status"] != "ok" for row in manifest["targets"]))


if __name__ == "__main__":
    raise SystemExit(main())
