import sys, numpy as np
from plyfile import PlyData
a, b = sys.argv[1], sys.argv[2]
pa, pb = PlyData.read(a), PlyData.read(b)
va, vb = pa['vertex'], pb['vertex']
na = [p.name for p in va.properties]; nb = [p.name for p in vb.properties]
print("count orig", va.count, "dec", vb.count, "same:", va.count == vb.count)
print("props identical (names+order):", na == nb, "| dtypes identical:", [p.val_dtype for p in va.properties] == [p.val_dtype for p in vb.properties])
if na != nb: print("orig", na); print("dec ", nb)
print("decoded header:\n" + "\n".join(str(pb.header).splitlines()[:6]) + "\n...")
for k in ["x", "opacity", "scale_0", "scale_1", "f_dc_0", "f_rest_0", "rot_0"]:
    x, y = np.asarray(va[k], dtype=np.float64), np.asarray(vb[k], dtype=np.float64)
    print(f"{k:9s} orig min/med/max {x.min():9.4f} {np.median(x):9.4f} {x.max():9.4f} | dec {np.nanmin(y):9.4f} {np.nanmedian(y):9.4f} {np.nanmax(y):9.4f} | dec NaN {int(np.isnan(y).sum())} Inf {int(np.isinf(y).sum())}")
allnan = sum(int(np.isnan(np.asarray(vb[k], dtype=np.float64)).sum()) for k in nb)
print("total NaN in decoded:", allnan)
sig = lambda t: 1/(1+np.exp(-t))
print("sigmoid(opacity) mean orig %.4f dec %.4f" % (sig(np.asarray(va['opacity'])).mean(), sig(np.asarray(vb['opacity'])).mean()))
# order check: is the decoded xyz in same order as original?
xa = np.stack([va['x'], va['y'], va['z']], 1)[:5]; xb = np.stack([vb['x'], vb['y'], vb['z']], 1)[:5]
print("first 5 xyz orig", xa.round(3).tolist()); print("first 5 xyz dec ", xb.round(3).tolist())
