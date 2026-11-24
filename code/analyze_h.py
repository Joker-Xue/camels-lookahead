"""偏离 3 的探索性分析：静态通路敏感性对照 H − B、I − B（S1）。不改动预注册输出。
估计量、集成、bootstrap 与 A − B 相同。输出 results/analysis_h.json。

H − B 判据：区间上限 < −0.01，即设计能检测到静态通路上的实质变化。
I − B 判读：区间在 ±0.01 内 → 支持“流域标签”解释；上限 < −0.01 → 否定；其他 → 不确定。
"""
import json

import numpy as np
from scipy import stats

from analyze import MARGIN, OUT, RUNS, boot_median, seeds_for, verdict
from metrics import evaluate, observations


def main():
    split = "S1"
    cfgs = ["B", "H", "I"]
    seeds = sorted(set.intersection(*[set(seeds_for(split, c)) for c in cfgs]))
    obs = observations(split)
    ens, single = {}, {}
    for c in cfgs:
        preds = [np.load(RUNS / f"{split}_{c}_{s}.npz")["pred"] for s in seeds]
        ens[c] = evaluate(np.mean(preds, axis=0), obs)
        single[c] = {s: evaluate(p, obs) for s, p in zip(seeds, preds)}
        if c != "B":
            ens[c].to_csv(OUT / f"metrics_{split}_{c}_ensemble.csv")
    r = {"seeds": seeds, "median_nse": {c: float(ens[c].nse.median()) for c in cfgs},
         "single_nse_median": {c: {s: float(single[c][s].nse.median()) for s in seeds} for c in cfgs},
         "compare": {}}
    for a in ["H", "I"]:
        for m in ["nse", "kge", "fhv", "flv"]:
            est, lo, hi = boot_median(ens[a][m] - ens["B"][m])
            r["compare"][f"{a}-B:{m}"] = dict(median=est, lo=lo, hi=hi)
        d = (ens[a].nse - ens["B"].nse).dropna()
        c = r["compare"][f"{a}-B:nse"]
        c["frac_positive"] = float((d > 0).mean())
        c["wilcoxon_p"] = float(stats.wilcoxon(d[d != 0]).pvalue)
        c["verdict"] = verdict(c["lo"], c["hi"])
        r[f"per_seed_{a}B"] = {s: boot_median(single[a][s].nse - single["B"][s].nse) for s in seeds}
    r["compare"]["H-B:nse"]["sensitivity_demonstrated"] = r["compare"]["H-B:nse"]["hi"] < -MARGIN
    lo, hi = r["compare"]["I-B:nse"]["lo"], r["compare"]["I-B:nse"]["hi"]
    r["compare"]["I-B:nse"]["identifier_reading"] = (
        "支持" if (lo > -MARGIN and hi < MARGIN) else "否定" if hi < -MARGIN else "不确定")
    (OUT / "analysis_h.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
