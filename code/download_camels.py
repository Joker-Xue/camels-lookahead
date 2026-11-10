"""Download the CAMELS v1.2 files used in this study from Zenodo record 15529996 into data/raw, with resume and MD5 check.
Usage: python code/download_camels.py   (about 7.6 GB)"""
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data/raw"
BASE = "https://zenodo.org/api/records/15529996/files/{}/content"
FILES = {
    "camels_clim.txt": "67f22592f3fb72c57df81358ce68458b",
    "camels_geol.txt": "f5ce5de53eb1ea2532cda7e3b4813993",
    "camels_soil.txt": "8edb46a363a20b466a4b7105ba633767",
    "camels_topo.txt": "0f6267838c40b1507b64582433bc0b8e",
    "camels_vege.txt": "f40e843defc1e654a800be9fe5fd5090",
    "camels_hydro.txt": "55ebdeb36c42ee7acdb998229c3edb3a",
    "camels_name.txt": "c96491b32c4df55a31bead7ceca7d64b",
    "basin_timeseries_v1p2_metForcing_obsFlow.zip": "8e9a466710e8270b58f01d332a87184f",
    "basin_timeseries_v1p2_modelOutput_daymet.zip": "f2af624b6277b75b3e410d6a0365591a",
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, want, tries=60):
    dst = RAW / name
    for _ in range(tries):
        if dst.exists() and md5(dst) == want:
            print("OK", name)
            return
        have = dst.stat().st_size if dst.exists() else 0
        req = urllib.request.Request(BASE.format(name), headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(dst, "ab" if have and r.status == 206 else "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        except urllib.error.HTTPError as e:
            if e.code == 416:  # local file is complete but corrupt: start over
                dst.unlink()
            else:
                print(f"{name}: {e}; retrying", file=sys.stderr)
                time.sleep(20)
        except OSError as e:
            print(f"{name}: {e}; retrying", file=sys.stderr)
            time.sleep(20)
    raise SystemExit(f"failed to download {name} with the expected MD5")


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for n, m in FILES.items():
        fetch(n, m)
