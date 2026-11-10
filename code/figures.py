"""论文图件。只读 results/ 与 data/attrs 中已有结果，不重新训练。输出 paper/fig*.pdf|png。"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from analyze import CLIM, boot_median
from experiment import ROOT, basin_ids
from metrics import evaluate, observations

OUT = ROOT / "paper"
OUT.mkdir(exist_ok=True)
RES = ROOT / "results"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e4e3de"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
IDS = basin_ids()


def save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig1_design():
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    y0 = {"S1": 2.0, "S2": 1.0}
    for sp, y in y0.items():
        tr, te = ((1999.75, 2008.75), (1989.75, 1999.75)) if sp == "S1" else ((1989.75, 1999.75), (1999.75, 2009.75))
        ax.add_patch(Rectangle((tr[0], y - .18), tr[1] - tr[0], .36, color=BLUE, lw=0))
        ax.add_patch(Rectangle((te[0], y - .18), te[1] - te[0], .36, color=ORANGE, lw=0))
        ax.text(1988.9, y, sp, ha="right", va="center", fontsize=9, color=INK, weight="bold")
        ax.text(np.mean(tr), y, "train", ha="center", va="center", color="white", fontsize=8)
        ax.text(np.mean(te), y, "test", ha="center", va="center", color="white", fontsize=8)
    ax.add_patch(Rectangle((1989.75, 0.25), 20, .22, facecolor="none", edgecolor=INK, lw=1, hatch="////"))
    ax.text(1999.75, 0.05, "Official CAMELS climate-attribute window (WY1990–2009)", ha="center", va="top",
            fontsize=8, color=INK)
    ax.set_xlim(1988, 2010.5)
    ax.set_ylim(-0.3, 2.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks(range(1990, 2011, 5))
    ax.set_xlabel("Year")
    save(fig, "fig1_design")


def attrs(split, c):
    t = pd.read_csv(ROOT / f"data/attrs/{split}_{c}.csv", index_col=0)
    t.index = t.index.astype(str).str.zfill(8)
    return t.loc[IDS]


def fig2_shift():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, sp in zip(axs, ["S1", "S2"]):
        a, b, d = attrs(sp, "A")[CLIM], attrs(sp, "B")[CLIM], attrs(sp, "D")[CLIM]
        sd = a.std()
        for k, (other, col, lab) in enumerate([(b, BLUE, "training-period (B)"), (d, ORANGE, "test-period (D)")]):
            z = ((other - a) / sd).abs()
            pos = np.arange(len(CLIM)) + (k - .5) * .36
            bp = ax.boxplot([z[c].values for c in CLIM], positions=pos, widths=.3, vert=False, patch_artist=True,
                            showfliers=False, medianprops=dict(color=INK, lw=1), whiskerprops=dict(color=col),
                            capprops=dict(color=col))
            for p in bp["boxes"]:
                p.set(facecolor=col, edgecolor=col, alpha=.75)
            ax.plot([], [], color=col, lw=6, alpha=.75, label=lab)
        ax.set_yticks(range(len(CLIM)))
        ax.set_yticklabels(CLIM)
        ax.invert_yaxis()
        ax.set_title(sp, loc="left", fontsize=9, color=INK)
        ax.set_xlabel("|attribute − official| / cross-basin SD")
        ax.grid(axis="x", color=GRID, lw=.6)
        ax.set_axisbelow(True)
    axs[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(-0.1, -0.22), ncol=2)
    save(fig, "fig2_attribute_shift")


def fig3_forest():
    r = json.loads((RES / "analysis.json").read_text())
    g = json.loads((RES / "analysis_g.json").read_text())
    rows = []
    lab = {"A-B": "A − B  official vs training-period (primary)", "A-C": "A − C  official vs none",
           "D-B": "D − B  test-period vs training-period", "E-F": "E − F  pre-registered positive control"}
    for sp in ["S1", "S2"]:
        for k in ["A-B", "A-C", "D-B", "E-F"]:
            if f"{k}:nse" in r[sp]["compare"]:
                c = r[sp]["compare"][f"{k}:nse"]
                rows.append((sp, lab[k], c["median"], c["lo"], c["hi"], k == "A-B"))
        if sp == "S1":
            c = g["compare"]["G-F:nse"]
            rows.append(("S1", "G − F  strengthened positive control*", c["median"], c["lo"], c["hi"], False))
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.axvspan(-.01, .01, color="#f0efec", lw=0)
    for x in [-.01, .01]:
        ax.axvline(x, color=INK2, lw=.8, ls="--")
    ax.axvline(0, color=INK2, lw=.6)
    y = 0
    ticks, names = [], []
    for i, (sp, name, m, lo, hi, prim) in enumerate(rows):
        if i and rows[i - 1][0] != sp:
            y += .7
        col = BLUE if sp == "S1" else ORANGE
        ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
        ax.plot(m, y, "D" if prim else "o", color=col, ms=6, mec="white", mew=1)
        ax.text(.0215, y, f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]", va="center", fontsize=7, color=INK2)
        ticks.append(y)
        names.append(f"{sp}  {name}")
        y += 1
    ax.set_yticks(ticks)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(-.015, .036)
    ax.set_xticks([-.01, -.005, 0, .005, .01, .015, .02])
    ax.set_xlabel("Median ΔNSE across 531 basins (ensemble), 95% bootstrap CI")
    ax.text(0, -.9, "equivalence margin ±0.01", ha="center", fontsize=7, color=INK2)
    save(fig, "fig3_forest")


def fig4_basins():
    topo = pd.read_csv(ROOT / "data/raw/camels_topo.txt", sep=";", dtype={"gauge_id": str}).set_index("gauge_id").loc[IDS]
    obs = observations("S1")
    ens = {c: evaluate(np.mean([np.load(RES / f"runs/S1_{c}_{s}.npz")["pred"] for s in range(1, 6)], 0), obs)
           for c in "AB"}
    d = ens["A"].nse - ens["B"].nse
    p1, p2 = [evaluate(np.load(RES / f"runs/S1_A_{s}.npz")["pred"], obs).nse for s in (1, 2)]
    b1 = evaluate(np.load(RES / "runs/S1_B_1.npz")["pred"], obs).nse
    noise, single = p1 - p2, p1 - b1
    fig = plt.figure(figsize=(7.0, 2.9))
    ax = fig.add_axes([0.0, 0.08, 0.58, 0.85])
    lim = .03
    sc = ax.scatter(topo.gauge_lon, topo.gauge_lat, c=d.clip(-lim, lim), cmap="RdBu_r", vmin=-lim, vmax=lim,
                    s=9, edgecolors="none")
    ax.set_aspect(1.25)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=.05, pad=.02, aspect=40)
    cb.set_label("ΔNSE (A − B), S1 ensemble; clipped at ±0.03", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("a", loc="left", fontsize=9, weight="bold")
    bx = fig.add_axes([0.67, 0.2, 0.32, 0.7])
    for v, col, lab in [(d, BLUE, "A − B, 5-seed ensembles"), (single, VIOLET, "A − B, single models (seed 1)"),
                         (noise, YELLOW, "A − A, seed 1 vs seed 2")]:
        x = np.sort(v.dropna().values)
        bx.plot(x, np.arange(1, len(x) + 1) / len(x), color=col, lw=2, label=lab)
    bx.axvspan(-.01, .01, color="#f0efec", lw=0, zorder=0)
    bx.set_xlim(-.08, .08)
    bx.set_xlabel("ΔNSE per basin")
    bx.set_ylabel("Cumulative fraction of basins")
    bx.grid(color=GRID, lw=.6)
    bx.legend(frameon=False, fontsize=6.5, loc="lower right")
    bx.set_title("b", loc="left", fontsize=9, weight="bold")
    save(fig, "fig4_basins")
    return {k: (round(float(v.abs().median()), 4), round(float((v.abs() > .01).mean()), 3)) for k, v in [("ens_AB", d), ("single_AB", single), ("seed_noise", noise)]}


if __name__ == "__main__":
    fig1_design()
    fig2_shift()
    fig3_forest()
    print("fig4 |A−B| 中位, |种子差| 中位, |A−B|>0.01 比例, |种子差|>0.01 比例:", fig4_basins())
