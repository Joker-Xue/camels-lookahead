"""按 Addor et al. (2017) 的定义，在指定时间窗口内计算 9 个 CAMELS 气候属性。"""
import numpy as np
import pandas as pd

CLIM = ["p_mean", "pet_mean", "aridity", "p_seasonality", "frac_snow",
        "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur"]


def _mean_run(mask):
    """连续为 True 的平均长度；没有事件时返回 NaN。"""
    m = np.asarray(mask, dtype=int)
    edges = np.diff(np.concatenate([[0], m, [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return float(np.mean(ends - starts)) if len(starts) else np.nan


def _sine_fit(y, doy, period=365.25):
    """拟合 y = m + a*sin(2*pi*(t-s)/period)，返回 m、振幅 a(>0)、相位 s。"""
    w = 2 * np.pi * doy / period
    X = np.column_stack([np.ones_like(w), np.sin(w), np.cos(w)])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    amp = float(np.hypot(b[1], b[2]))
    # a*sin(w - phi) = a*cos(phi)*sin(w) - a*sin(phi)*cos(w)
    phase = float(np.arctan2(-b[2], b[1]) * period / (2 * np.pi))
    return float(b[0]), amp, phase


def climate_attributes(df):
    """df 须含日期索引和 prcp、tmax、tmin、pet 列（单位 mm/day、摄氏度）。"""
    p = df["prcp"].to_numpy()
    pet = df["pet"].to_numpy()
    t = ((df["tmax"] + df["tmin"]) / 2).to_numpy()
    years = len(df) / 365.25
    p_mean = float(np.nanmean(p))
    pet_mean = float(np.nanmean(pet))
    doy = (df.index - pd.Timestamp("1980-01-01")).days.to_numpy(dtype=float)
    pm, pa, ps = _sine_fit(p, doy)
    _, ta, ts = _sine_fit(t, doy)
    # Woods (2009) 式 14：delta_p*sgn(delta_t)*cos(2*pi*(s_p-s_t)/365.25)；delta_t 取正振幅，相位已含符号
    seasonality = (pa / pm) * np.cos(2 * np.pi * (ps - ts) / 365.25)
    high = p >= 5 * p_mean
    low = p < 1
    return {
        "p_mean": p_mean,
        "pet_mean": pet_mean,
        "aridity": pet_mean / p_mean,
        "p_seasonality": float(seasonality),
        # Addor et al. (2017) 文字写 0 摄氏度，但官方 camels_clim.txt 只有用 1 摄氏度才能精确复现
        "frac_snow": float(p[t < 1].sum() / p.sum()),
        "high_prec_freq": float(high.sum() / years),
        "high_prec_dur": _mean_run(high),
        "low_prec_freq": float(low.sum() / years),
        "low_prec_dur": _mean_run(low),
    }
