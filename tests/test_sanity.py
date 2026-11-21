"""Synthetic sanity tests (shapes with known homology).

Units: isotropic Gaussians have sigma = 0.1 and opacity ~1, so peak density ~1.
A bar is "long" if its persistence exceeds LONG_FRAC = 5% of the field's maximum
density (scale-free); shorter bars are ripples between overlapping Gaussians and
are ignored, as Section 6 allows.
"""
import numpy as np
import pytest

from topogs.density import Grid, density_bruteforce, density_field
from topogs.io_ply import load_ply, make_scene, quat_to_rotmat, save_ply
from topogs.topology import bottleneck, count_long, filter_dgm, persistence

SIG = 0.1
OPA = 0.99
LONG_FRAC = 0.05
ID_Q = [1.0, 0.0, 0.0, 0.0]


def iso(means):
    return make_scene(means, SIG, ID_Q, OPA)


def grid64(half=1.0, res=64, **kw):
    return Grid.cubic(np.zeros(3), half, res, **kw)


def ph(scene, grid=None):
    f, _ = density_field(scene, grid or grid64())
    res = persistence(f)
    res["long"] = LONG_FRAC * f.max()
    return f, res


def n_long(res, dim):
    return count_long(res["dgms"][dim], res["long"])


def summary(res):
    # number of components = 1 essential bar + long finite H0 bars
    return (1 + n_long(res, 0), n_long(res, 1), n_long(res, 2))


# ---------------------------------------------------------------- topology

def test_single_gaussian():
    f, res = ph(iso([[0, 0, 0]]))
    assert summary(res) == (1, 0, 0)
    # negation check: the essential H0 bar is born at -max(density)
    assert np.isclose(res["essential"][0][0], -f.max())


def test_two_separated():
    _, res = ph(iso([[-0.5, 0, 0], [0.5, 0, 0]]))
    assert summary(res) == (2, 0, 0)
    bar = filter_dgm(res["dgms"][0], res["long"])[0]
    assert -bar[1] < 0.01, "should merge only at a very low density level"


def test_two_overlapping():
    _, res = ph(iso([[-0.1, 0, 0], [0.1, 0, 0]]))
    assert summary(res) == (1, 0, 0)


def test_ring():
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    pts = np.stack([0.5 * np.cos(t), 0.5 * np.sin(t), 0 * t], axis=1)
    _, res = ph(iso(pts))
    assert summary(res) == (1, 1, 0)


def fibonacci_sphere(n, r):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return r * np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)


def test_sphere_shell():
    _, res = ph(iso(fibonacci_sphere(300, 0.5)))
    assert summary(res) == (1, 0, 1)


def test_line_remove_middle():
    x = np.linspace(-0.6, 0.6, 7)  # spacing 0.2 = 2 sigma -> connected
    pts = np.stack([x, 0 * x, 0 * x], axis=1)
    _, full = ph(iso(pts))
    _, cut = ph(iso(np.delete(pts, 3, axis=0)))
    assert summary(full) == (1, 0, 0)
    assert summary(cut) == (2, 0, 0)
    d = bottleneck(full["dgms"][0], cut["dgms"][0])
    assert d > 0.2, f"bottleneck H0 should jump, got {d:.3f}"


# ---------------------------------------------------------------- geometry

def test_anisotropic_rotation():
    """Long axis along x; rotate 90 deg about z with a (w,x,y,z) quaternion -> long axis along y."""
    scales = [0.3, 0.05, 0.05]
    a = make_scene([[0, 0, 0]], scales, ID_Q, OPA)
    q = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
    b = make_scene([[0, 0, 0]], scales, q, OPA)
    g = grid64()
    fa, _ = density_field(a, g)
    fb, _ = density_field(b, g)
    # elongated along x before rotation: more mass spread in x than y
    mx = (fa > 0.1).any(axis=(1, 2)).sum()
    my = (fa > 0.1).any(axis=(0, 2)).sum()
    assert mx > 3 * my
    np.testing.assert_allclose(fb, fa.transpose(1, 0, 2), atol=1e-6)


def test_rotated_crop_box():
    """A Gaussian placed at +0.3 along the box's first axis must peak there in the grid."""
    ang = np.deg2rad(30)
    Rz = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    R_box = Rz.T  # rows = box axes in world coords
    c = np.array([2.0, -1.0, 0.5])
    p = c + R_box.T @ np.array([0.3, 0, 0])
    g = Grid.cubic(c, 1.0, 64, R_box=R_box)
    f, _ = density_field(iso([p]), g)
    idx = np.unravel_index(np.argmax(f), f.shape)
    peak_u = np.array([g.axes()[a][idx[a]] for a in range(3)])
    assert np.allclose(peak_u, [0.3, 0, 0], atol=g.spacing.max())


def test_fast_vs_bruteforce():
    rng = np.random.default_rng(0)
    n = 40
    q = rng.normal(size=(n, 4))
    scene = make_scene(rng.uniform(-0.8, 0.8, (n, 3)), rng.uniform(0.03, 0.25, (n, 3)),
                       q, rng.uniform(0.05, 0.99, n))
    ang = 0.7
    R_box = quat_to_rotmat(np.array([[np.cos(ang / 2), 0.3, np.sin(ang / 2), 0.1]]) /
                           np.linalg.norm([np.cos(ang / 2), 0.3, np.sin(ang / 2), 0.1]))[0]
    g = Grid([0.1, -0.05, 0.0], [0.9, 0.8, 0.7], (26, 22, 18), R_box=R_box, offset=0.3)
    fast, _ = density_field(scene, g, cutoff=4.0)
    brute = density_bruteforce(scene, g, cutoff=4.0)
    exact = density_bruteforce(scene, g, cutoff=None)
    err = np.abs(fast - brute).max()
    trunc = np.abs(fast - exact).max() / exact.max()
    print(f"\nfast vs brute (same 4-sigma cutoff): max abs diff {err:.2e}")
    print(f"truncation error vs untruncated: max {trunc:.2e} of peak density")
    assert err < 1e-10
    assert trunc < 2e-3


# ---------------------------------------------------------------- I/O

def test_ply_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    n = 500
    s = make_scene(rng.normal(size=(n, 3)), rng.uniform(0.01, 0.5, (n, 3)),
                   rng.normal(size=(n, 4)), rng.uniform(0.01, 0.99, n))
    s.raw["f_dc_0"] = rng.normal(size=n)
    s.raw["f_rest_44"] = rng.normal(size=n)
    p = tmp_path / "rt.ply"
    save_ply(s, p)
    r = load_ply(p)
    assert r.raw.dtype.names == s.raw.dtype.names  # same properties, same order
    for name in s.raw.dtype.names:
        np.testing.assert_array_equal(r.raw[name], s.raw[name])
    np.testing.assert_allclose(r.opacities, s.opacities)


@pytest.mark.skipif("TOPOGS_REAL_PLY" not in __import__("os").environ, reason="set TOPOGS_REAL_PLY")
def test_real_ply_round_trip(tmp_path):
    import os
    s = load_ply(os.environ["TOPOGS_REAL_PLY"])
    save_ply(s, tmp_path / "rt.ply")
    r = load_ply(tmp_path / "rt.ply")
    assert r.raw.dtype == s.raw.dtype
    for name in s.raw.dtype.names:
        np.testing.assert_array_equal(r.raw[name], s.raw[name])


def test_density_matches_textbook_formula():
    """Independent of our quat_to_rotmat / to_box_frame: build Sigma = R S S^T R^T in WORLD
    coordinates with scipy (scalar-last quaternions!) and evaluate o*exp(-1/2 d^T Sigma^-1 d)."""
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(7)
    q_wxyz = rng.normal(size=4)
    q_wxyz /= np.linalg.norm(q_wxyz)
    s = np.array([0.35, 0.12, 0.05])
    mu, o = np.array([0.2, -0.1, 0.15]), 0.8
    scene = make_scene([mu], s, q_wxyz, o)
    R_box = Rotation.from_euler("zyx", [25, -15, 40], degrees=True).as_matrix().T
    c = np.array([0.1, 0.0, -0.05])
    g = Grid(c, [0.8, 0.7, 0.6], (20, 18, 16), R_box=R_box, offset=0.25)
    ours, _ = density_field(scene, g, cutoff=20.0)  # cutoff large enough to be irrelevant
    # .ply stores float32: use the stored values (rounded) so the comparison can be exact
    q32 = np.array([scene.raw[f"rot_{i}"][0] for i in range(4)], float)
    s = np.exp(np.array([scene.raw[f"scale_{i}"][0] for i in range(3)], float))
    o = 1 / (1 + np.exp(-float(scene.raw["opacity"][0])))
    mu = np.array([scene.raw[a][0] for a in "xyz"], float)
    R = Rotation.from_quat(np.r_[q32[1:], q32[0]]).as_matrix()  # scipy wants (x,y,z,w); it normalizes
    Sigma_inv = np.linalg.inv(R @ np.diag(s ** 2) @ R.T)
    X = c + g.points_box().reshape(-1, 3) @ R_box  # x = c + R_box^T u, row form
    d = X - mu
    ref = o * np.exp(-0.5 * np.einsum("ni,ij,nj->n", d, Sigma_inv, d)).reshape(g.shape)
    assert ref.max() > 0.5  # the Gaussian is actually inside the box
    np.testing.assert_allclose(ours, ref, atol=1e-10)


def test_filter_mass_is_convolution():
    """mass-mode filtering = convolving the field with a normalised Gaussian:
    total integral is preserved; peak-mode keeps the peak but widens the blob."""
    s = make_scene([[0, 0, 0]], [0.08, 0.02, 0.005], [0.9, 0.1, -0.3, 0.2], 0.9)
    g = Grid.cubic(np.zeros(3), 0.6, 120)
    dv = np.prod(g.spacing)
    f0 = density_bruteforce(s, g, cutoff=None, filter_sigma=0.02, mode="mass")
    exact_mass = 0.9 * (2 * np.pi) ** 1.5 * 0.08 * 0.02 * 0.005
    assert abs(f0.sum() * dv - exact_mass) / exact_mass < 1e-3
    fp, _ = density_field(s, g, filter_sigma=0.02, mode="peak")
    fb = density_bruteforce(s, g, cutoff=4.0, filter_sigma=0.02, mode="peak")
    assert np.abs(fp - fb).max() < 1e-10
    assert fp.max() > 0.85  # peak preserved (sample near the centre)
