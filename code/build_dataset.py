"""把 CAMELS 原始 zip 解析成每流域一个 npz（Daymet 强迫、PET、径流 mm/day）。不写入任何属性。"""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
FORC = zipfile.ZipFile(RAW / "basin_timeseries_v1p2_metForcing_obsFlow.zip")
MODEL = zipfile.ZipFile(RAW / "basin_timeseries_v1p2_modelOutput_daymet.zip")
INDEX = pd.date_range("1980-01-01", "2014-12-31", freq="D")


def read_forcing(name):
    lines = FORC.read(name).decode().splitlines()
    area_m2 = float(lines[2])
    rows = np.loadtxt(io.StringIO("\n".join(lines[4:])))
    idx = pd.to_datetime({"year": rows[:, 0], "month": rows[:, 1], "day": rows[:, 2]})
    df = pd.DataFrame(rows[:, 4:], columns=["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"],
                      index=pd.DatetimeIndex(idx))
    return df[~df.index.duplicated()].reindex(INDEX), area_m2


def read_flow(name, area_m2):
    rows = [l.split() for l in FORC.read(name).decode().splitlines() if l.strip()]
    idx = pd.to_datetime([f"{r[1]}-{r[2]}-{r[3]}" for r in rows])
    cfs = np.array([float(r[4]) for r in rows])
    cfs[cfs < 0] = np.nan
    s = pd.Series(cfs * 0.0283168466 * 86400 * 1000 / area_m2, index=idx)
    return s[~s.index.duplicated()].reindex(INDEX)


def read_pet(names):
    """各校准种子的 PET 系数不同，取所有种子的日均值，并另存逐种子长期均值以评估其离散度。"""
    series = []
    for n in names:
        raw = MODEL.read(n)
        if not raw.strip():
            print("空的模型输出，跳过:", n, flush=True)
            continue
        df = pd.read_csv(io.BytesIO(raw), sep=r"\s+")
        idx = pd.to_datetime({"year": df.YR, "month": df.MNTH, "day": df.DY})
        series.append(pd.Series(df.PET.to_numpy(), index=idx))
    all_ = pd.concat(series, axis=1)
    all_ = all_[~all_.index.duplicated()].reindex(INDEX)
    return all_


def main():
    forc = {re.search(r"/(\d{8})_lump_cida", n).group(1): n for n in FORC.namelist()
            if "basin_mean_forcing/daymet" in n and n.endswith("_lump_cida_forcing_leap.txt")
            and "__MACOSX" not in n}
    flow = {re.search(r"/(\d{8})_streamflow", n).group(1): n for n in FORC.namelist()
            if "usgs_streamflow" in n and n.endswith("_streamflow_qc.txt") and "__MACOSX" not in n}
    pet = {}
    for n in MODEL.namelist():
        m = re.search(r"/(\d{8})_(\d+)_model_output\.txt$", n)
        if m and "__MACOSX" not in n:
            pet.setdefault(m.group(1), []).append(n)
    ids = sorted(set(forc) & set(flow) & set(pet))
    print(f"强迫 {len(forc)} 径流 {len(flow)} 模型输出 {len(pet)} 共有 {len(ids)}", flush=True)
    for i, b in enumerate(ids):
        f, area = read_forcing(forc[b])
        q = read_flow(flow[b], area)
        p = read_pet(pet[b])
        np.savez_compressed(OUT / f"{b}.npz", area_m2=area, prcp=f.prcp, srad=f.srad, tmax=f.tmax,
                            tmin=f.tmin, vp=f.vp, dayl=f.dayl, q=q, pet=p.mean(axis=1),
                            pet_seeds=p.to_numpy(dtype=np.float32))
        if i % 100 == 0:
            print(i, b, f"种子数 {p.shape[1]}", flush=True)
    np.save(OUT / "dates.npy", INDEX.to_numpy())
    print("完成", len(ids))


if __name__ == "__main__":
    sys.exit(main())
