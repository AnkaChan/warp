"""Install the pinned CUDA components needed for the EBVH validation build."""

import hashlib
import json
import pathlib
import subprocess
import urllib.request

VERSION = "12.6.3"
BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
ROOT = pathlib.Path(__file__).resolve().parents[2] / "_build" / "ebvh-cuda-12.6.3"
PACKAGES = ("cuda_nvcc", "cuda_cudart", "cuda_cccl", "cuda_nvrtc", "cuda_cuobjdump")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "redistrib.json"
    urllib.request.urlretrieve(f"{BASE}redistrib_{VERSION}.json", manifest_path)
    manifest = json.loads(manifest_path.read_text())
    for name in PACKAGES:
        package = manifest[name]["linux-x86_64"]
        archive = ROOT / pathlib.Path(package["relative_path"]).name
        if not archive.exists():
            print(f"Downloading {name}: {package['size']} bytes", flush=True)
            urllib.request.urlretrieve(BASE + package["relative_path"], archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != package["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {archive}")
        subprocess.run(["tar", "-xf", str(archive), "--strip-components=1", "-C", str(ROOT)], check=True)
        print(f"Verified and installed {name}: {digest}", flush=True)
    lib64 = ROOT / "lib64"
    if not lib64.exists():
        lib64.symlink_to("lib")
    subprocess.run([str(ROOT / "bin" / "nvcc"), "--version"], check=True)
    print(f"WARP_CUDA_PATH={ROOT}")


if __name__ == "__main__":
    main()
