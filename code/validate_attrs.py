"""校验：用官方窗口 1989-10-01..2009-09-30 复算气候属性，与 camels_clim.txt 比较。"""
from pathlib import Path

import numpy as np
import pandas as pd

from camels_attrs import CLIM, climate_attributes

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
dates = pd.DatetimeIndex(np.load(P / "dates.npy"))
off = pd.read_csv(ROOT / "data/raw/camels_clim.txt", sep=";", dtype={"gauge_id": str}).set_index("gauge_id")
rows = {}
for f in sorted(P.glob("*.npz")):
    z = np.load(f)
    df = pd.DataFrame({k: z[k] for k in ["prcp", "tmax", "tmin"]}, index=dates)
    df["pet"] = z["pet_seeds"][:, 0]  # 官方 pet_mean 对应第一个校准种子（05）
    rows[f.stem] = climate_attributes(df.loc["1989-10-01":"2009-09-30"])
mine = pd.DataFrame(rows)
mine = mine.T
ids = mine.index.intersection(off.index)
off = off.rename(columns={"frac_snow_daily": "frac_snow"})
out = []
for c in CLIM:
    a, b = mine.loc[ids, c].astype(float), off.loc[ids, c].astype(float)
    ok = a.notna() & b.notna()
    out.append(dict(attr=c, n=int(ok.sum()), pearson=np.corrcoef(a[ok], b[ok])[0, 1],
                    mae=(a[ok] - b[ok]).abs().mean(), median_official=b[ok].median(),
                    max_abs=(a[ok] - b[ok]).abs().max()))
res = pd.DataFrame(out)
print(res.to_string(index=False))
(ROOT / "results").mkdir(exist_ok=True)
res.to_csv(ROOT / "results/attr_validation.csv", index=False)
