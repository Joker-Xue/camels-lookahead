"""按 PROTOCOL.md 统计。输出 results/analysis.json 和逐流域指标 csv。只读 results/runs 中已完成的运行。"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from experiment import ROOT, basin_ids
from metrics import evaluate, observations

RUNS = ROOT / "results/runs"
OUT = ROOT / "results"
BOOT, BSEED, MARGIN = 10000, 20261102, 0.01
CLIM = ["p_mean", "pet_mean", "aridity", "p_seasonality", "frac_snow",
        "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur"]


def boot_median(d):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    rng = np.random.default_rng(BSEED)
    idx = rng.integers(0, len(d), (BOOT, len(d)))
    m = np.median(d[idx], axis=1)
    return float(np.median(d)), float(np.quantile(m, .025)), float(np.quantile(m, .975))


def verdict(lo, hi):
    if lo > -MARGIN and hi < MARGIN:
        return "等效"
    if lo > MARGIN:
        return "A 明显抬高"
    if hi < -MARGIN:
        return "方向相反"
    return "不确定"


def seeds_for(split, cfg):
    return sorted(int(f.stem.split("_")[2]) for f in RUNS.glob(f"{split}_{cfg}_*.npz"))


def main():
    res = {}
    ids = basin_ids()
    for split in ["S1", "S2"]:
        cfgs = [c for c in "ABCDEF" if seeds_for(split, c)]
        if not cfgs:
            continue
        common = sorted(set.intersection(*[set(seeds_for(split, c)) for c in cfgs]))
        obs = observations(split)
        ens, single = {}, {}
        for c in cfgs:
            preds = [np.load(RUNS / f"{split}_{c}_{s}.npz")["pred"] for s in common]
            ens[c] = evaluate(np.mean(preds, axis=0), obs)
            single[c] = {s: evaluate(p, obs) for s, p in zip(common, preds)}
            ens[c].to_csv(OUT / f"metrics_{split}_{c}_ensemble.csv")
        r = {"seeds": common, "configs": cfgs,
             "median": {c: {m: float(ens[c][m].median()) for m in ens[c]} for c in cfgs},
             "single_nse_median": {c: {s: float(single[c][s].nse.median()) for s in common} for c in cfgs}}
        pairs = [p for p in [("A", "B"), ("A", "C"), ("D", "B"), ("E", "F")] if p[0] in cfgs and p[1] in cfgs]
        cmp_, pv = {}, {}
        for a, b in pairs:
            for m in ["nse", "kge", "fhv", "flv"]:
                d = ens[a][m] - ens[b][m]
                est, lo, hi = boot_median(d)
                cmp_[f"{a}-{b}:{m}"] = dict(median=est, lo=lo, hi=hi)
                if m == "nse":
                    x = d.dropna()
                    x = x[x != 0]
                    pv[f"{a}-{b}"] = float(stats.wilcoxon(x).pvalue)
                    cmp_[f"{a}-{b}:{m}"]["frac_positive"] = float((d > 0).mean())
                    cmp_[f"{a}-{b}:{m}"]["verdict"] = verdict(lo, hi)
        order = sorted(pv, key=pv.get)
        holm, cur = {}, 0.0
        for i, k in enumerate(order):
            cur = max(cur, min(1.0, pv[k] * (len(order) - i)))
            holm[k] = cur
        r["compare"], r["wilcoxon_p"], r["wilcoxon_holm"] = cmp_, pv, holm
        # 配对的单种子结果
        r["per_seed_AB"] = {s: boot_median(single["A"][s].nse - single["B"][s].nse) for s in common} if {"A", "B"} <= set(cfgs) else {}
        # 噪声基线：同配置单模型两两差
        noise = []
        for c in cfgs:
            for s, t in itertools.combinations(common, 2):
                noise.append((single[c][s].nse - single[c][t].nse).abs().median())
        r["noise_abs_median"] = float(np.median(noise)) if noise else None
        # 剂量响应：属性位移幅度
        if {"A", "B"} <= set(cfgs):
            ta = pd.read_csv(ROOT / f"data/attrs/{split}_A.csv", dtype=str, index_col=0).astype(float)
            for t in [ta]:
                t.index = t.index.astype(str).str.zfill(8)
            def attrs(c):
                t = pd.read_csv(ROOT / f"data/attrs/{split}_{c}.csv", index_col=0)
                t.index = t.index.astype(str).str.zfill(8)
                return t.loc[ids, CLIM]
            sd = attrs("A").std()
            dist_ab = (((attrs("A") - attrs("B")) / sd) ** 2).sum(axis=1) ** .5
            sp = stats.spearmanr(dist_ab, (ens["A"].nse - ens["B"].nse).abs())
            r["dose_AB"] = dict(rho=float(sp.statistic), p=float(sp.pvalue), median_dist=float(dist_ab.median()))
            if "D" in cfgs:
                dist_db = (((attrs("D") - attrs("B")) / sd) ** 2).sum(axis=1) ** .5
                sp = stats.spearmanr(dist_db, ens["D"].nse - ens["B"].nse)
                r["dose_DB"] = dict(rho=float(sp.statistic), p=float(sp.pvalue), median_dist=float(dist_db.median()))
            # 分组
            off = attrs("A")
            dif = ens["A"].nse - ens["B"].nse
            grp = pd.qcut(off.aridity, 4, labels=["Q1湿润", "Q2", "Q3", "Q4干旱"])
            r["by_aridity"] = {str(k): boot_median(dif[grp == k]) for k in grp.cat.categories}
            shift = ((attrs("D").p_mean - attrs("B").p_mean) / attrs("B").p_mean).abs() if "D" in cfgs else None
            if shift is not None:
                g2 = pd.qcut(shift, 4, labels=["Q1小", "Q2", "Q3", "Q4大"])
                r["by_pshift"] = {str(k): boot_median(dif[g2 == k]) for k in g2.cat.categories}
        res[split] = r
    (OUT / "analysis.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
