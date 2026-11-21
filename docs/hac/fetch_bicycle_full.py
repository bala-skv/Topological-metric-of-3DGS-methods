"""Fetch Mip-NeRF 360 bicycle full-size images/, poses_bounds.npy and sparse/0/* from 360_v2.zip
by HTTP range requests (remotezip). zipfile verifies each member's CRC32 on read (raises BadZipFile
on mismatch); we additionally re-check zlib.crc32 against the directory entry and the size.

    python fetch_bicycle_full.py OUT_DIR [--list]
"""
import os
import sys
import zlib

from remotezip import RemoteZip

M360 = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"


def wanted(n):
    return (n.startswith("bicycle/images/") and not n.endswith("/")) or n == "bicycle/poses_bounds.npy" \
        or (n.startswith("bicycle/sparse/0/") and not n.endswith("/"))


def main(out, list_only=False):
    with RemoteZip(M360) as z:
        infos = [i for i in z.infolist() if wanted(i.filename)]
        tot = sum(i.file_size for i in infos)
        n_img = sum(i.filename.startswith("bicycle/images/") for i in infos)
        print("members %d (images %d) total %.1f MB" % (len(infos), n_img, tot / 1e6), flush=True)
        if list_only:
            for i in infos:
                if not i.filename.startswith("bicycle/images/"):
                    print(i.filename, i.file_size)
            return
        done = 0
        for i in infos:
            dst = os.path.join(out, i.filename)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst) and os.path.getsize(dst) == i.file_size:
                data = open(dst, "rb").read()
                if zlib.crc32(data) & 0xffffffff == i.CRC:
                    done += 1
                    continue
            data = z.read(i.filename)  # zipfile checks CRC32 here
            assert len(data) == i.file_size and zlib.crc32(data) & 0xffffffff == i.CRC, i.filename
            with open(dst + ".part", "wb") as f:
                f.write(data)
            os.replace(dst + ".part", dst)
            done += 1
            print("ok %d/%d %s %d" % (done, len(infos), i.filename, i.file_size), flush=True)
        print("all %d members verified (CRC32 + size)" % done)


if __name__ == "__main__":
    main(sys.argv[1], "--list" in sys.argv)
