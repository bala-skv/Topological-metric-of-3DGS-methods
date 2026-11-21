import numpy as np
from plyfile import PlyData
R = '/home2/bala_skv/topogs-fcgs/'
orig = R + 'data/bicycle/point_cloud.ply'
def header(p):
    with open(p, 'rb') as f:
        h = b''
        while not h.endswith(b'end_header\n'):
            h += f.readline()
    return h
ho = header(orig)
print('orig header bytes', len(ho), 'comments:', [l for l in ho.decode().splitlines() if l.startswith('comment')])
xyz = {}
for l in ['0.0001', '0.0004', '0.0016']:
    p = R + f'results/fcgs/lmd_{l}/point_cloud.ply'
    print(l, 'raw header byte-identical to original:', header(p) == ho)
    v = PlyData.read(p)['vertex']
    xyz[l] = np.stack([np.asarray(v['x']), np.asarray(v['y']), np.asarray(v['z'])], 1)
    del v
print('xyz arrays (incl. order) equal 1e-4 vs 4e-4:', np.array_equal(xyz['0.0001'], xyz['0.0004']), '| 1e-4 vs 16e-4:', np.array_equal(xyz['0.0001'], xyz['0.0016']))
