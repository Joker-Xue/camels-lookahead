"""偏离 2 的探索性分析：更强的阳性对照 G − F（S1）。不改动 analyze.py 的预注册输出。
估计量、bootstrap、等效界与 E − F 相同；阳性判据：区间下限 > +0.01。输出 results/analysis_g.json。"""
import json

import numpy as np
from scipy import stats

from analyze import MARGIN, OUT, RUNS, boot_median, seeds_for
from metrics import evaluate, observations


def main():
    split = "S1"
    seeds = sorted(set(seeds_for(split, "G")) & set(seeds_for(split, "F")))
    obs = observations(split)
    ens, single = {}, {}
    for c in ["F", "G"]:
        preds = [np.load(RUNS / f"{split}_{c}_{s}.npz")["pred"] for s in seeds]
        ens[c] = evaluate(np.mean(preds, axis=0), obs)
        single[c] = {s: evaluate(p, obs) for s, p in zip(seeds, preds)}
    ens["G"].to_csv(OUT / f"metrics_{split}_G_ensemble.csv")
    r = {"seeds": seeds, "median_nse": {c: float(ens[c].nse.median()) for c in ens}, "compare": {}}
    for m in ["nse", "kge", "fhv", "flv"]:
        est, lo, hi = boot_median(ens["G"][m] - ens["F"][m])
        r["compare"][f"G-F:{m}"] = dict(median=est, lo=lo, hi=hi)
    d = (ens["G"].nse - ens["F"].nse).dropna()
    c = r["compare"]["G-F:nse"]
    c["frac_positive"] = float((d > 0).mean())
    c["wilcoxon_p"] = float(stats.wilcoxon(d[d != 0]).pvalue)
    c["positive_control_passed"] = c["lo"] > MARGIN
    r["per_seed_GF"] = {s: boot_median(single["G"][s].nse - single["F"][s].nse) for s in seeds}
    (OUT / "analysis_g.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
