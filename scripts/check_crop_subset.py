"""(1) --write-box: write the crop-box JSON used by extract_crop_subset.py (from the frozen configs).
(2) check: the crop subset of the ORIGINAL gives exactly the same density fields (all resolutions,
offsets 0 and 0.5) as the full scene. Proves the subset shortcut changes nothing.

    python scripts/check_crop_subset.py --write-box configs/crop_box.json
    python scripts/check_crop_subset.py configs/crop_box.json data/.../point_cloud.ply
"""
import json
import pathlib
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.extract_crop_subset import crop_mask  # noqa: E402
from topogs.density import density_field  # noqa: E402
from topogs.grids import crop_grid  # noqa: E402
from topogs.io_ply import load_ply  # noqa: E402

AN = "configs/analysis.yaml"


def write_box(path):
    an = yaml.safe_load(open(AN)); crop = yaml.safe_load(open(an["crop_config"]))
    b = crop["boxes"][an["box"]]
    R = np.array(crop["frame"]["R_box"]); c = np.array(crop["frame"]["centre"])
    box = dict(box=an["box"], R_box=R.tolist(), centre_world=(c + R.T @ np.array(b["center"])).tolist(),
               half=b["half"], filter_sigma=an["filter_sigma"], cutoff_sigma=an["cutoff_sigma"], margin=0.05)
    pathlib.Path(path).write_text(json.dumps(box, indent=2)); print(box)


def check(box_json, ply):
    an = yaml.safe_load(open(AN)); box = json.load(open(box_json))
    scene = load_ply(ply)
    keep = crop_mask(scene, box)
    sub = scene.subset(keep)
    print("subset", int(keep.sum()), "of", scene.n)
    for res in an["resolutions"]:
        for off in (0.0, 0.5):
            g = crop_grid(an["crop_config"], an["box"], res, offset=off)
            a, na = density_field(scene, g, an["cutoff_sigma"], filter_sigma=an["filter_sigma"], mode=an["filter_mode"])
            b, nb = density_field(sub, g, an["cutoff_sigma"], filter_sigma=an["filter_sigma"], mode=an["filter_mode"])
            print(res, off, "used", na, nb, "max|diff|", float(np.abs(a - b).max()), "bit-identical", bool(np.array_equal(a, b)))
            assert na == nb and np.array_equal(a, b)


if __name__ == "__main__":
    if sys.argv[1] == "--write-box":
        write_box(sys.argv[2])
    else:
        check(sys.argv[1], sys.argv[2])
