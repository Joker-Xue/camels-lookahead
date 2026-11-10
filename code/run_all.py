"""按协议顺序运行全部实验，可中断续跑：已有结果的运行会跳过。用法：run_all.py [--workers N]"""
import argparse
import os
import subprocess
import sys
import time

import torch

from experiment import HP, ROOT

DEVICE = "cuda" if torch.cuda.is_available() else "mps"

RUNS = [("S1", c, s) for s in range(1, 6) for c in "ABCDEF"]
RUNS += [("S2", c, s) for s in range(1, 4) for c in "ABCD"]
RUNS += [("S1", "G", s) for s in range(1, 6)]  # 偏离 2：更强的阳性对照（探索性）
# 按种子交错，让两个划分都尽早有完整配对
RUNS.sort(key=lambda r: (r[2], r[0]))


def start(sp, c, s):
    f = open(ROOT / f"logs/run_{sp}_{c}_{s}.log", "a", encoding="utf-8")
    p = subprocess.Popen([sys.executable, str(ROOT / "code/train.py"), "--split", sp, "--config", c, "--seed", str(s),
                          "--device", DEVICE, "--hidden", str(HP["hidden"]), "--batch", str(HP["batch"]), "--frac", "1.0"],
                         stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return p, f, time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1, help="同时运行的任务数")
    a = ap.parse_args()
    if sys.platform == "win32":  # 运行期间阻止系统睡眠，相当于 macOS 的 caffeinate
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    todo = [r for r in RUNS if not (ROOT / f"results/runs/{r[0]}_{r[1]}_{r[2]}.npz").exists()]
    running, failed = {}, False
    while todo or running:
        while todo and len(running) < a.workers and not failed:
            r = todo.pop(0)
            running[r] = start(*r)
        if failed and not running:
            break
        time.sleep(10)
        for r, (p, f, t) in list(running.items()):
            if p.poll() is None:
                continue
            f.close()
            del running[r]
            done = len(RUNS) - len(todo) - len(running)
            print(f"{done}/{len(RUNS)} {'-'.join(map(str, r))} 退出码 {p.returncode} 用时 {(time.time() - t) / 60:.1f} 分钟", flush=True)
            failed |= p.returncode != 0
        if failed:
            todo = []
    print("有任务失败，已停止" if failed else "全部完成", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
