"""Do the crop .ply files the HAC fields were built from (laptop copies, rsync'd from ada) match the
sha256 recorded on ada by scripts/hac_validate_export.py? Closes the chain validated export -> field.

    python scripts/hac_check_copies.py -> results/hac/copy_check.json
"""
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.hac_fields import SOURCES  # noqa: E402
from topogs.io_ply import load_ply  # noqa: E402

REC = {"scaffold": ("data/hac/export/scaffold/validate{tag}.json", "crop_raw_sha256"),
       "hac": ("data/hac/export2/hac/validate{tag}.json", "crop_keepzero_raw_sha256"),
       "hacdrop": ("data/hac/export/hac/validate{tag}.json", "crop_raw_sha256")}


def main():
    out, ok = {}, True
    for m, (rec, key) in REC.items():
        for tag in ("", "_rep"):
            v = json.load(open(rec.format(tag=tag)))["cameras"]
            for cam, c in v.items():
                p = SOURCES[m].format(cam=cam + tag)
                h = hashlib.sha256(load_ply(p).raw.tobytes()).hexdigest()
                out[p] = h == c[key]; ok &= out[p]
    pathlib.Path("results/hac/copy_check.json").write_text(json.dumps(dict(all_match=bool(ok), files=out), indent=2))
    print("all_match", ok, len(out), "files")


if __name__ == "__main__":
    main()
