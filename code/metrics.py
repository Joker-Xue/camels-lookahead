"""逐流域评估指标：NSE、KGE、FHV（前 2% 流量偏差）、FLV（后 30% 流量偏差）。"""
import numpy as np
import pandas as pd

from experiment import ROOT, SPLITS, basin_ids


def nse(o, s):
    return 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)


def kge(o, s):
    r = np.corrcoef(o, s)[0, 1] if s.std() > 0 else 0.0
    return 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)


def fhv(o, s, h=0.02):
    n = max(int(round(h * len(o))), 1)
    oo, ss = np.sort(o)[::-1][:n], np.sort(s)[::-1][:n]
    return np.sum(ss - oo) / np.sum(oo) * 100


def flv(o, s, l=0.3, eps=1e-6):
    n = max(int(round(l * len(o))), 1)
    oo, ss = np.log(np.maximum(np.sort(o)[:n], eps)), np.log(np.maximum(np.sort(s)[:n], eps))
    oo, ss = oo - oo.min(), ss - ss.min()
    d = np.sum(oo)
    return np.sum(ss - oo) / d * 100 if d != 0 else np.nan


def observations(split):
    dates = pd.DatetimeIndex(np.load(ROOT / "data/processed/dates.npy"))
    s = SPLITS[split]["test"]
    m = (dates >= s[0]) & (dates <= s[1])
    return np.stack([np.load(ROOT / f"data/processed/{b}.npz")["q"][m] for b in basin_ids()])


def evaluate(pred, obs):
    rows = []
    for o, s in zip(obs, pred):
        ok = ~np.isnan(o)
        rows.append(dict(nse=nse(o[ok], s[ok]), kge=kge(o[ok], s[ok]), fhv=fhv(o[ok], s[ok]), flv=flv(o[ok], s[ok])))
    return pd.DataFrame(rows, index=basin_ids())


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        split = f.split("/")[-1].split("_")[0]
        e = evaluate(np.load(f)["pred"], observations(split))
        print(f, "NSE 中位", round(e.nse.median(), 4), "均值", round(e.nse.mean(), 4),
              "KGE 中位", round(e.kge.median(), 4), "NSE<0 流域", int((e.nse < 0).sum()))
