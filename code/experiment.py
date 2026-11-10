"""实验定义：时间划分、属性配置、模型超参数。超参数沿用 Kratzert et al. (2019) 的 LSTM。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL_WINDOW = ("1989-10-01", "2009-09-30")
SPLITS = {
    # S1：Kratzert et al. (2019) 的基准划分，测试期早于训练期
    "S1": {"train": ("1999-10-01", "2008-09-30"), "test": ("1989-10-01", "1999-09-30")},
    # S2：严格向前的划分，测试期仍完整落在官方属性窗口内
    "S2": {"train": ("1989-10-01", "1999-09-30"), "test": ("1999-10-01", "2009-09-30")},
}
CONFIGS = {
    "A": {"clim": "full", "hydro": None},    # 官方气候属性（现行做法）
    "B": {"clim": "train", "hydro": None},   # 只用训练期重算（无前视）
    "C": {"clim": None, "hydro": None},      # 不用气候属性
    "D": {"clim": "test", "hydro": None},    # 只用测试期计算（前视上限）
    "E": {"clim": "full", "hydro": "full"},  # 阳性对照：全时段径流特征，含目标泄漏
    "F": {"clim": "train", "hydro": "train"},  # E 的无泄漏对照
    "G": {"clim": "train", "hydro": "test"},   # 偏离 2：更强的阳性对照，只用测试期径流特征（探索性）
}
DYNAMIC = ["prcp", "srad", "tmax", "tmin", "vp"]
HP = dict(seq_len=270, hidden=256, layers=2, dropout=0.4, batch=256, epochs=30, clip=1.0,
          lr={0: 1e-3, 11: 5e-4, 21: 1e-4})


def basin_ids():
    return (ROOT / "data/raw/basin_list_531.txt").read_text().split()
