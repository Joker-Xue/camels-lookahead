"""训练一个 LSTM 并保存测试期逐日预测。用法：train.py --split S1 --config A --seed 1"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from experiment import DYNAMIC, HP, ROOT, SPLITS, basin_ids

PROC = ROOT / "data/processed"


class LSTM(nn.Module):
    def __init__(self, n_in, hidden, layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1])).squeeze(-1)


def load(split, config, device):
    ids = basin_ids()
    dates = pd.DatetimeIndex(np.load(PROC / "dates.npy"))
    s = SPLITS[split]
    attrs = pd.read_csv(ROOT / f"data/attrs/{split}_{config}.csv", dtype={"Unnamed: 0": str}, index_col=0)
    attrs.index = attrs.index.astype(str).str.zfill(8)
    attrs = attrs.loc[ids]
    a = attrs.to_numpy(np.float32)
    a = (a - a.mean(0)) / np.maximum(a.std(0), 1e-6)  # 跨流域标准化
    X = np.stack([np.stack([np.load(PROC / f"{b}.npz")[k] for k in DYNAMIC], -1) for b in ids]).astype(np.float32)
    Q = np.stack([np.load(PROC / f"{b}.npz")["q"] for b in ids]).astype(np.float32)
    assert not np.isnan(X).any()
    tr = (dates >= s["train"][0]) & (dates <= s["train"][1])
    te = (dates >= s["test"][0]) & (dates <= s["test"][1])
    # 动态输入和径流的标准化参数只用训练期
    xm, xs = X[:, tr].mean((0, 1)), X[:, tr].std((0, 1))
    qm, qs = np.nanmean(Q[:, tr]), np.nanstd(Q[:, tr])
    X = (X - xm) / xs
    Qn = (Q - qm) / qs
    return dict(ids=ids, dates=dates, tr=tr, te=te, X=torch.from_numpy(X).to(device),
                Q=torch.from_numpy(np.nan_to_num(Qn)).to(device), Qraw=Q, static=torch.from_numpy(a).to(device),
                qm=float(qm), qs=float(qs), n_static=a.shape[1], columns=list(attrs.columns))


def windows(d, basin, end, L):
    """basin、end 为同长度索引张量，返回 (n, L, 输入维度)。"""
    t = end[:, None] - L + 1 + torch.arange(L, device=end.device)[None]
    x = d["X"][basin[:, None], t]
    st = d["static"][basin][:, None].expand(-1, L, -1)
    return torch.cat([x, st], -1)


@torch.no_grad()
def predict(model, d, L, device, bs=2048):
    model.eval()
    te = np.where(d["te"])[0]
    out = np.zeros((len(d["ids"]), len(te)), np.float32)
    for b in range(len(d["ids"])):
        for i in range(0, len(te), bs):
            e = torch.as_tensor(te[i:i + bs], device=device)
            bb = torch.full_like(e, b)
            out[b, i:i + bs] = model(windows(d, bb, e, L)).cpu().numpy()
    return np.maximum(out * d["qs"] + d["qm"], 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-batches", type=int, default=0, help="只用于测速")
    ap.add_argument("--epochs", type=int, default=HP["epochs"])
    ap.add_argument("--outdir", default="results/runs")
    ap.add_argument("--hidden", type=int, default=HP["hidden"])
    ap.add_argument("--batch", type=int, default=HP["batch"])
    ap.add_argument("--frac", type=float, default=1.0, help="每轮随机抽取的样本比例")
    a = ap.parse_args()
    HP.update(hidden=a.hidden, batch=a.batch, frac=a.frac)
    device = torch.device(a.device)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    d = load(a.split, a.config, device)
    L = HP["seq_len"]
    tr_idx = np.where(d["tr"])[0]
    obs = ~np.isnan(d["Qraw"][:, tr_idx])
    bas, pos = np.where(obs)
    bas, ends = bas.astype(np.int64), tr_idx[pos].astype(np.int64)
    n_in = len(DYNAMIC) + d["n_static"]
    model = LSTM(n_in, HP["hidden"], HP["layers"], HP["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"][0])
    crit = nn.MSELoss()
    log, t0, first = [], time.time(), 0
    # 每轮结束保存断点（模型、优化器和全部随机数状态），中断后从下一轮接着训练
    ckpt = ROOT / f"results/ckpt/{a.split}_{a.config}_{a.seed}.pt"
    if ckpt.exists() and not a.max_batches:
        c = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        rng.bit_generator.state = c["np_rng"]
        torch.set_rng_state(c["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(c["cuda_rng"].cpu())
        log, first = c["log"], c["epoch"] + 1
        t0 -= log[-1]["sec"]
        print(f"从断点恢复：已完成 {first} 轮", flush=True)
    for ep in range(first, a.epochs):
        if ep in HP["lr"]:
            for g in opt.param_groups:
                g["lr"] = HP["lr"][ep]
        model.train()
        perm = rng.permutation(len(bas))[:int(len(bas) * a.frac)]
        total, nb = 0.0, 0
        for i in range(0, len(perm), HP["batch"]):
            sel = perm[i:i + HP["batch"]]
            bb = torch.from_numpy(bas[sel]).to(device)
            ee = torch.from_numpy(ends[sel]).to(device)
            x = windows(d, bb, ee, L)
            y = d["Q"][bb, ee]
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), HP["clip"])
            opt.step()
            total += loss.item()
            nb += 1
            if a.max_batches and nb >= a.max_batches:
                break
        log.append(dict(epoch=ep, loss=total / nb, sec=time.time() - t0))
        print(f"{a.split}-{a.config}-{a.seed} 轮 {ep} loss {total / nb:.4f} 累计 {time.time() - t0:.0f}s 批数 {nb}/{len(perm) // HP['batch'] + 1}", flush=True)
        if not a.max_batches:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), np_rng=rng.bit_generator.state,
                            torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state() if device.type == "cuda" else None,
                            log=log, epoch=ep), ckpt.with_suffix(".tmp"))
            ckpt.with_suffix(".tmp").replace(ckpt)
    if a.max_batches:
        print(f"测速 {a.device}: {nb / (time.time() - t0):.1f} 批/秒，样本 {len(bas)}")
        return
    pred = predict(model, d, L, device)
    out = ROOT / a.outdir
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{a.split}_{a.config}_{a.seed}.npz", pred=pred)
    (out / f"{a.split}_{a.config}_{a.seed}.json").write_text(json.dumps(dict(
        split=a.split, config=a.config, seed=a.seed, device=a.device, columns=d["columns"], hp=HP, log=log,
        n_samples=int(len(bas)), torch=torch.__version__), ensure_ascii=False))
    ckpt.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
