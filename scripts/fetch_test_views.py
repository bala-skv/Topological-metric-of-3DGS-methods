"""Fetch only the Mip-NeRF 360 bicycle test views (images_4, every 8th by name) by HTTP range
requests into 360_v2.zip, and verify each against the laptop's SHA-256 (configs/test_views_sha256.txt).
Used on ada, where the laptop->ada link is too slow to copy images.

    python scripts/fetch_test_views.py configs/test_views_sha256.txt OUT_DIR
"""
import hashlib
import os
import sys

from remotezip import RemoteZip

M360 = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"


def main(sums, out):
    want = dict(reversed(l.split()) for l in open(sums) if l.strip())
    os.makedirs(out, exist_ok=True)
    with RemoteZip(M360) as z:
        for name, sha in sorted(want.items()):
            data = z.read("bicycle/images_4/" + name)
            got = hashlib.sha256(data).hexdigest()
            if got != sha:
                raise SystemExit("MISMATCH " + name)
            open(os.path.join(out, name), "wb").write(data)
            print("ok", name, flush=True)
    print("all %d verified" % len(want))


if __name__ == "__main__":
    main(*sys.argv[1:3])
