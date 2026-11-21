"""Run a Scaffold-GS / HAC train.py unchanged, with memory + speed logging (output-neutral).

    python memwrap.py train.py <train args...>        (cwd = repo root)

Wraps GaussianModel.update_learning_rate (called once per iteration in both repos) to print, every
TOPOGS_EVERY iterations: iteration, wall seconds since iteration 1, s/iter over the last window, anchor count,
torch memory_allocated / max_memory_allocated / max_memory_reserved (MiB). Prints the final peaks at exit.

With --data_device cpu both repos crash at the end-of-training render_set: `errormap = (rendering - gt).abs()` with
rendering on cuda and gt = view.original_image on cpu (HAC train.py:423, job 88387). Unless TOPOGS_GT_CUDA=0, we
therefore make Camera.original_image a property: stored on data_device (cpu), returned as a cuda copy once the
camera is constructed. Values are identical; training already did `.cuda()` on it every iteration.
"""
import atexit
import os
import runpy
import sys
import time

import torch

EVERY = int(os.environ.get("TOPOGS_EVERY", "500"))
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))

import scene.gaussian_model as gm  # noqa: E402  (same module object train.py will import)

_orig = gm.GaussianModel.update_learning_rate
_state = {"t0": None, "tlast": None, "ilast": None}


def _mib(x):
    return x / 2 ** 20


def update_learning_rate(self, iteration):
    if iteration == 1 or iteration % EVERY == 0:
        torch.cuda.synchronize()
        now = time.time()
        if _state["t0"] is None:
            _state["t0"] = now
        dt = "" if _state["tlast"] is None else "%.4f" % ((now - _state["tlast"]) / max(1, iteration - _state["ilast"]))
        _state["tlast"], _state["ilast"] = now, iteration
        print("TOPOGS it=%d wall_s=%.1f s_per_it=%s anchors=%d alloc_MiB=%.0f max_alloc_MiB=%.0f max_reserved_MiB=%.0f"
              % (iteration, now - _state["t0"], dt, self._anchor.shape[0], _mib(torch.cuda.memory_allocated()),
                 _mib(torch.cuda.max_memory_allocated()), _mib(torch.cuda.max_memory_reserved())), flush=True)
    return _orig(self, iteration)


gm.GaussianModel.update_learning_rate = update_learning_rate

if os.environ.get("TOPOGS_GT_CUDA", "1") == "1":
    import scene.cameras as sc  # noqa: E402

    def _get_img(self):
        t = self.__dict__["_topogs_img"]
        return t.cuda() if self.__dict__.get("_topogs_ready", False) else t

    def _set_img(self, v):
        self.__dict__["_topogs_img"] = v

    _orig_cam_init = sc.Camera.__init__

    def _cam_init(self, *a, **k):
        _orig_cam_init(self, *a, **k)
        self.__dict__["_topogs_ready"] = True

    sc.Camera.original_image = property(_get_img, _set_img)
    sc.Camera.__init__ = _cam_init
    print("TOPOGS: Camera.original_image kept on data_device, returned on cuda", flush=True)


@atexit.register
def _report():
    if torch.cuda.is_initialized():
        print("TOPOGS FINAL max_memory_allocated_MiB=%.0f max_memory_reserved_MiB=%.0f"
              % (_mib(torch.cuda.max_memory_allocated()), _mib(torch.cuda.max_memory_reserved())), flush=True)


runpy.run_path(script, run_name="__main__")
