"""偏离 3 的探索性稳健性分析：把训练随机性（种子）纳入置信区间。不改动预注册输出。

预注册的区间只对流域重抽样，是"给定这几个种子"的条件区间。这里做两层 bootstrap：
每次重抽样先有放回地抽种子（两种配置用同一组种子，保持配对），用抽到的种子组成集成，
再有放回地抽流域，取逐流域 ΔNSE 的中位数。

实现：有放回抽 k 个种子只取决于各种子被抽中的次数（多重集），共 C(2k-1, k) 种。
先对每种多重集算好两配置的加权集成和逐流域 NSE，再按多项分布概率抽多重集、抽流域。
输出 results/analysis_seeds.json。另给出留一种子集成（leave-one-seed-out）的估计范围。
"""
import itertools
import json
from collections import Counter

import numpy as np

from analyze import BOOT, BSEED, MARGIN, OUT, RUNS, seeds_for
from metrics import evaluate, observations

PAIRS = {"S1": [("A", "B"), ("A", "C"), ("D", "B"), ("E", "F"), ("G", "F")],
         "S2": [("A", "B"), ("A", "C"), ("D", "B")]}


def multisets(k):
    """k 个种子有放回抽 k 次的全部多重集，返回 (计数矩阵, 概率)。"""
    from math import factorial
    rows, probs = [], []
    for combo in itertools.combinations_with_replacement(range(k), k):
        c = Counter(combo)
        cnt = np.array([c.get(i, 0) for i in range(k)])
        p = factorial(k) / np.prod([factorial(x) for x in cnt]) / k ** k
        rows.append(cnt)
        probs.append(p)
    return np.array(rows), np.array(probs)


def nse_matrix(pred, obs):
    """pred: (流域, 天)，返回逐流域 NSE（只用有观测的天）。"""
    return evaluate(pred, obs).nse.to_numpy()


def main():
    res = {}
    rng = np.random.default_rng(BSEED)
    for split, pairs in PAIRS.items():
        obs = observations(split)
        cfgs = sorted({c for p in pairs for c in p})
        seeds = sorted(set.intersection(*[set(seeds_for(split, c)) for c in cfgs]))
        k = len(seeds)
        cnt, prob = multisets(k)
        preds = {c: np.stack([np.load(RUNS / f"{split}_{c}_{s}.npz")["pred"] for s in seeds]) for c in cfgs}
        # 每个多重集的加权集成 → 逐流域 NSE，形状 (多重集数, 流域数)
        nse = {}
        for c in cfgs:
            nse[c] = np.stack([nse_matrix(np.tensordot(w / k, preds[c], axes=1), obs) for w in cnt])
        # 留一种子集成
        loo = {c: np.stack([nse_matrix(np.delete(preds[c], i, 0).mean(0), obs) for i in range(k)]) for c in cfgs}
        nb = nse[cfgs[0]].shape[1]
        ms = rng.choice(len(prob), size=BOOT, p=prob)
        bi = rng.integers(0, nb, (BOOT, nb))
        full = int(np.flatnonzero((cnt == 1).all(1))[0])  # 每个种子恰好一次 = 原集成
        r = {"seeds": seeds, "n_multisets": int(len(prob)), "compare": {}}
        for a, b in pairs:
            d = nse[a] - nse[b]                       # (多重集, 流域)
            boot = np.nanmedian(d[ms[:, None], bi], axis=1)
            lo, hi = np.quantile(boot, [.025, .975])
            est = float(np.nanmedian(d[full]))
            # 只对流域重抽样（与预注册相同的条件区间），用同一组流域索引，便于比较宽度
            cond = np.nanmedian(d[full][bi], axis=1)
            clo, chi = np.quantile(cond, [.025, .975])
            loo_est = [float(np.nanmedian(loo[a][i] - loo[b][i])) for i in range(k)]
            r["compare"][f"{a}-{b}"] = dict(
                median=est, lo=float(lo), hi=float(hi),
                basin_only_lo=float(clo), basin_only_hi=float(chi),
                inside_margin=bool(lo > -MARGIN and hi < MARGIN),
                leave_one_seed_out=loo_est)
        res[split] = r
    (OUT / "analysis_seeds.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
