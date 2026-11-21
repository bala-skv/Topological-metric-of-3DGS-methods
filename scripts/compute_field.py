"""Compute density fields for one scene on the chosen crop at several resolutions,
save .npy + JSON sidecar, and density-slice figures.

    python scripts/compute_field.py configs/field_original.yaml
"""
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from topogs.grids import compute_and_save, crop_grid  # noqa: E402
from topogs.io_ply import load_ply  # noqa: E402
from topogs.viz import density_panels  # noqa: E402


def main(cfg_path):
    cfg = yaml.safe_load(pathlib.Path(cfg_path).read_text())
    an = yaml.safe_load(pathlib.Path(cfg["analysis_config"]).read_text())
    scene = load_ply(cfg["ply"])
    out = pathlib.Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    for res in cfg["res"]:
        for off in cfg.get("offsets", [0.0]):
            g = crop_grid(an["crop_config"], an["box"], res, offset=off)
            tag = f"{cfg['name']}_r{res}_off{off}"
            field, meta = compute_and_save(scene, g, out / f"{tag}.npy", source=cfg["ply"],
                                           filter_sigma=an["filter_sigma"], mode=an["filter_mode"],
                                           cutoff=an["cutoff_sigma"])
            print(tag, field.shape, f"max {field.max():.3f}", f"{meta['seconds']}s", flush=True)
            if off == 0.0 and cfg.get("figures", True):
                density_panels(field, g, out / f"{tag}_slices.png",
                               title=f"{cfg['name']}, box {an['box']}, {res}^3, filter "
                                     f"{an['filter_mode']} f={an['filter_sigma']:.4g}")


if __name__ == "__main__":
    main(sys.argv[1])
