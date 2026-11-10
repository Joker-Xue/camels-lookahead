"""生成各实验配置的静态属性表（531 个流域）。训练期、测试期由 experiment.py 统一定义。"""
from pathlib import Path

import numpy as np
import pandas as pd

from camels_attrs import CLIM, climate_attributes
from experiment import CONFIGS, FULL_WINDOW, SPLITS, ROOT, basin_ids

RAW = ROOT / "data/raw"
PROC = ROOT / "data/processed"
OUT = ROOT / "data/attrs"
OUT.mkdir(parents=True, exist_ok=True)
FIXED = {
    "topo": ["elev_mean", "slope_mean", "area_gages2"],
    "vege": ["frac_forest", "lai_max", "lai_diff", "gvf_max", "gvf_diff"],
    "soil": ["soil_depth_pelletier", "soil_depth_statsgo", "soil_porosity", "soil_conductivity",
             "max_water_content", "sand_frac", "silt_frac", "clay_frac"],
    "geol": ["carbonate_rocks_frac", "geol_permeability"],
}
HYDRO = ["q_mean", "runoff_ratio", "q5", "q95", "high_q_freq", "low_q_freq", "zero_q_freq"]


def flow_attributes(df):
    """径流特征，定义同 Addor et al. (2017)；只用有观测的日子。"""
    q = df["q"].dropna()
    m = q.mean()
    years = len(q) / 365.25
    return {"q_mean": m, "runoff_ratio": m / df.loc[q.index, "prcp"].mean(),
            "q5": q.quantile(.05), "q95": q.quantile(.95),
            "high_q_freq": (q >= 9 * q.median()).sum() / years,
            "low_q_freq": (q < .2 * m).sum() / years,
            "zero_q_freq": (q == 0).mean() * 100}


def main():
    ids = basin_ids()
    fixed = pd.concat([pd.read_csv(RAW / f"camels_{k}.txt", sep=";", dtype={"gauge_id": str})
                       .set_index("gauge_id")[v] for k, v in FIXED.items()], axis=1).loc[ids]
    print("固定属性缺失数:", int(fixed.isna().sum().sum()))
    fixed = fixed.fillna(fixed.median())
    off = pd.read_csv(RAW / "camels_clim.txt", sep=";", dtype={"gauge_id": str}).set_index("gauge_id")
    dates = pd.DatetimeIndex(np.load(PROC / "dates.npy"))
    frames = {}
    for b in ids:
        z = np.load(PROC / f"{b}.npz")
        frames[b] = pd.DataFrame({"prcp": z["prcp"], "tmax": z["tmax"], "tmin": z["tmin"],
                                  "pet": z["pet_seeds"][:, 0], "q": z["q"]}, index=dates)

    def table(window, fn):
        return pd.DataFrame({b: fn(frames[b].loc[window[0]:window[1]]) for b in ids}).T

    for split, s in SPLITS.items():
        full = table(FULL_WINDOW, climate_attributes)
        err = (full[CLIM] - off.loc[ids, CLIM]).abs().max().max()
        print(f"{split}: 复算与官方最大绝对差 {err:.2e}")
        clim = {"full": off.loc[ids, CLIM], "train": table(s["train"], climate_attributes),
                "test": table(s["test"], climate_attributes)}
        hydro = {"full": table(FULL_WINDOW, flow_attributes), "train": table(s["train"], flow_attributes),
                 "test": table(s["test"], flow_attributes)}
        for name, cfg in CONFIGS.items():
            parts = [fixed]
            if cfg["clim"]:
                parts.append(clim[cfg["clim"]])
            if cfg["hydro"]:
                parts.append(hydro[cfg["hydro"]])
            t = pd.concat(parts, axis=1).astype(float)
            assert not t.isna().any().any(), (split, name, t.columns[t.isna().any()].tolist())
            t.to_csv(OUT / f"{split}_{name}.csv")
        delta = (clim["train"] - clim["full"]) / clim["full"].std()
        print(split, "训练期属性 vs 官方，差异/跨流域标准差 的中位绝对值:")
        print(delta.abs().median().round(3).to_string())


if __name__ == "__main__":
    main()
