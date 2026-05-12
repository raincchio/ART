import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = "results"
ENV = "HalfCheetah-v4"
SEEDS = [1, 2, 3, 4, 5, 6]
EVAL_FREQ = 1000
WARMUP = 25000

CONDITIONS = {
    "TD7 (baseline)":    ("td7_val200k",           "#4878CF"),
    "TD7 + ART":         ("td7_mrt250_ema200k",    "#D65F5F"),
    "TD7 + RC":          ("td7_rc_ema200k",         "#E8913A"),
    "TD7 + ART + RC":    ("td7_mrt250_rc_ema200k", "#6ACC65"),
}

METRICS = {
    "eval_reward":  "Evaluation Reward",
    "value":        "Q-value",
    "be_error":     "Bellman Error (squared)",
}


def load_condition(folder):
    dfs = []
    for seed in SEEDS:
        path = os.path.join(RESULTS_DIR, folder, f"{ENV}_seed_{seed}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        dfs.append(df)
    if not dfs:
        return None
    min_len = min(len(d) for d in dfs)
    arr = np.stack([d.iloc[:min_len].values for d in dfs])   # (seeds, steps, cols)
    cols = dfs[0].columns.tolist()
    steps = WARMUP + 1000 + np.arange(min_len) * EVAL_FREQ    # first eval at step 26000
    return steps, arr, cols


def smooth(x, w=5):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4))
fig.suptitle(f"HalfCheetah-v4  —  200k steps  (6 seeds, mean ± std)  [scalar EMA bias]", fontsize=12)

for ax, (metric_key, metric_label) in zip(axes, METRICS.items()):
    for label, (folder, color) in CONDITIONS.items():
        result = load_condition(folder)
        if result is None:
            continue
        steps, arr, cols = result
        if metric_key not in cols:
            continue
        idx = cols.index(metric_key)
        data = arr[:, :, idx]          # (seeds, steps)
        mean = data.mean(0)
        std  = data.std(0)
        mean_s = smooth(mean)
        std_s  = smooth(std)
        n_steps = len(steps)
        ax.plot(steps[:n_steps], mean_s[:n_steps], label=label, color=color, lw=1.8)
        ax.fill_between(steps[:n_steps],
                        (mean_s - std_s)[:n_steps],
                        (mean_s + std_s)[:n_steps],
                        alpha=0.15, color=color)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(metric_label)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, f"{ENV}_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
