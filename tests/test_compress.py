import numpy as np

from topogs.compress import prune_opacity, prune_random, quantize
from topogs.io_ply import make_scene


def _scene(n=1000, seed=1):
    r = np.random.default_rng(seed)
    q = r.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    return make_scene(r.uniform(-5, 5, (n, 3)), np.exp(r.uniform(-6, 0, (n, 3))), q,
                      r.uniform(0.01, 0.99, n))


def test_prune_opacity_removes_exactly_the_lowest():
    s = _scene()
    v, info = prune_opacity(s, 0.3)
    assert v.n == 700 and info["n_out"] == 700
    assert v.opacities.min() >= np.sort(s.opacities)[300] - 1e-7
    # survivors keep file order
    assert len(set(v.raw["x"])) == 700 and np.all(np.isin(v.raw["x"], s.raw["x"]))


def test_prune_random_count_and_determinism():
    s = _scene()
    a, _ = prune_random(s, 0.5, seed=0)
    b, _ = prune_random(s, 0.5, seed=0)
    c, _ = prune_random(s, 0.5, seed=1)
    assert a.n == 500 and np.array_equal(a.raw, b.raw) and not np.array_equal(a.raw, c.raw)


def test_quantize_error_bound_and_levels():
    s = _scene()
    for bits in (8, 16):
        v, info = quantize(s, bits)
        for f, st in info["steps"].items():
            err = np.abs(v.raw[f].astype(np.float64) - s.raw[f].astype(np.float64))
            assert err.max() <= st["step"] / 2 + 1e-5 * max(abs(st["min"]), abs(st["max"]), 1)
            assert len(np.unique(v.raw[f])) <= 2 ** bits
        # untouched properties are bit-identical
        assert np.array_equal(v.raw["opacity"], s.raw["opacity"])
        assert np.array_equal(v.raw["rot_0"], s.raw["rot_0"])
