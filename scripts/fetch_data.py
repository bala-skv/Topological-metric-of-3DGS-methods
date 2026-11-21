"""Fetch only the bicycle files needed by the project.

- Pretrained 3DGS scene: the official INRIA models.zip server is throttled (~20-90 KB/s),
  so we download from a Hugging Face mirror of the same archive and VERIFY each file's
  CRC32 against the CRC recorded in INRIA's own zip directory (read via range requests).
- Mip-NeRF 360 bicycle images (images_4 + sparse/0): range requests into 360_v2.zip.
Writes <out>/SHA256SUMS.

    python scripts/fetch_data.py [--out data] [--skip-images]
"""
import argparse
import hashlib
import pathlib
import subprocess
import zlib

from remotezip import RemoteZip

INRIA = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip"
MIRROR = "https://huggingface.co/johnowhitaker/gaussian_splatter_models/resolve/main/"
M360 = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
SCENE_FILES = ["bicycle/cfg_args", "bicycle/cameras.json",
               "bicycle/point_cloud/iteration_30000/point_cloud.ply"]


def crc32(p):
    c = 0
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            c = zlib.crc32(chunk, c)
    return c & 0xFFFFFFFF


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_scene(dest: pathlib.Path):
    with RemoteZip(INRIA) as z:
        official = {i.filename: (i.file_size, i.CRC) for i in z.infolist() if i.filename in SCENE_FILES}
    for name in SCENE_FILES:
        size, crc = official[name]
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not (target.exists() and target.stat().st_size == size):
            print("fetching", name, f"{size / 1e6:.1f} MB", flush=True)
            subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", str(target), MIRROR + name], check=True)
        got = crc32(target)
        ok = got == crc and target.stat().st_size == size
        print(f"  {name}: size {target.stat().st_size} (official {size}), "
              f"crc32 {got:08x} (official {crc:08x}) -> {'IDENTICAL' if ok else 'MISMATCH'}", flush=True)
        if not ok:
            raise SystemExit(f"{name} does not match the official INRIA archive")


def fetch_images(dest: pathlib.Path):
    dest.mkdir(parents=True, exist_ok=True)
    with RemoteZip(M360) as z:
        for info in z.infolist():
            f = info.filename
            if f.startswith("bicycle/") and not f.endswith("/") and (
                    "/images_4/" in f or "/sparse/" in f or f.endswith("poses_bounds.npy")):
                target = dest / f
                if not (target.exists() and target.stat().st_size == info.file_size):
                    z.extract(info, dest)
    print("images done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--skip-images", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    fetch_scene(out / "pretrained")
    if not a.skip_images:
        fetch_images(out / "mipnerf360")
    lines = [f"{sha256(p)}  {p.stat().st_size:>12}  {p.relative_to(out)}"
             for p in sorted((out / "pretrained").rglob("*")) if p.is_file()]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("done")
