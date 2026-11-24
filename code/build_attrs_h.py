"""偏离 3：静态通路敏感性对照的属性表，从 data/attrs/S1_B.csv 派生，不改动已有表。

H：不用任何静态属性（表中只有流域编号，没有属性列）。
I：把 B 的 27 维属性向量在流域之间整体置换（错排：没有流域保留自己的向量），固定种子 20261124。
"""
import numpy as np
import pandas as pd

from experiment import ROOT, basin_ids

OUT = ROOT / "data/attrs"
PERM_SEED = 20261124


def derangement(n, rng):
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p


def main():
    ids = basin_ids()
    b = pd.read_csv(OUT / "S1_B.csv", dtype={"Unnamed: 0": str}, index_col=0)
    b.index = b.index.astype(str).str.zfill(8)
    b = b.loc[ids]
    pd.DataFrame(index=b.index).to_csv(OUT / "S1_H.csv")
    p = derangement(len(b), np.random.default_rng(PERM_SEED))
    i = pd.DataFrame(b.to_numpy()[p], index=b.index, columns=b.columns)
    i.to_csv(OUT / "S1_I.csv")
    # 检查：每个流域得到的是另一个流域的向量，且向量仍互不相同（可作为流域标识）
    assert (p != np.arange(len(b))).all()
    assert len(np.unique(i.round(10).to_numpy(), axis=0)) == len(i)
    print("H 列数", 0, "I 列数", i.shape[1], "唯一向量数", len(np.unique(i.round(10).to_numpy(), axis=0)))


if __name__ == "__main__":
    main()
