"""Numbers quoted in the paper that the notebooks do not print, and Figs. 3 (RR quantum) and arrival clustering.

    BORG_CSV=/path/to/borg_traces_data.csv python paper_numbers.py

Uses kernel/workload_seed42.csv (written by cpu-scheduling-algorithm-comparison.ipynb), the
notebook's own policy functions, load_sweep.csv, and the trace CSV (same path variable as the notebooks),
so every value comes from the code and data that produced Tables II and IV.
"""
import ast
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "kernel"))
from analyze import simulator  # noqa: E402

w = pd.read_csv(HERE / "kernel/workload_seed42.csv")
rr = simulator()["round_robin"]


def stretch(w, rho):
    """Same arrival stretching as make_workload(seed, rho) in the notebook."""
    rho0 = w.Burst_Time.sum() / w.Arrival_Time.max()
    return w.assign(Arrival_Time=(w.Arrival_Time * rho0 / rho).round().astype(int))


def peak_local_load(w, frac=0.01):
    """Largest work arriving within any window of `frac` of the arrival span, divided by the window."""
    a, b = w.Arrival_Time.to_numpy(), w.Burst_Time.to_numpy()
    width = frac * a.max()
    return max(b[(a >= t) & (a < t + width)].sum() / width for t in np.unique(a))


rho = w.Burst_Time.sum() / w.Arrival_Time.max()
print(f"Offered load: {w.Burst_Time.sum():,} units over {w.Arrival_Time.max():,} -> rho = {rho:.1f}")
modal = w.Burst_Time.mode()[0]
print(f"Modal burst {modal:,}: {(w.Burst_Time == modal).sum()} of {len(w)} jobs; mean burst {w.Burst_Time.mean():,.0f}")

prod = w.Priority >= 120
print(f"Mean burst, Borg priority >= 120: {w.Burst_Time[prod].mean():,.0f} (n={prod.sum()}); "
      f"< 120: {w.Burst_Time[~prod].mean():,.0f}; Spearman(priority, burst) = "
      f"{spearmanr(w.Priority, w.Burst_Time).statistic:.2f}")
print(f"Spearman(relative deadline, burst) = {spearmanr(w.Deadline - w.Arrival_Time, w.Burst_Time).statistic:.2f}")

size = w.groupby("Arrival_Time").size()
print(f"Jobs sharing an arrival instant: {size[size > 1].sum()} of {len(w)} (largest group {size.max()})")
for r in [0.5, 0.9]:
    print(f"Peak load in 1% windows at rho = {r}: {peak_local_load(stretch(w, r)):.1f}")

print("\nRR cliff (simulator): q -> mean WT (x10^6)")
for q in [5_000, 9_000, 9_990, 10_000, 10_010, 11_000, 15_000, 20_000, 30_000]:
    print(f"  {q:>6,}  {rr(w, quantum=q)['avg_waiting'] / 1e6:.2f}")

# --- What a trace row is (paper Sec. III-A/B) ------------------------------------------------------
# Same cleaning, in the same order, as the notebook's Section 2, so row positions (and the seed-42 draw) match.
t = pd.read_csv(os.environ.get("BORG_CSV", "/kaggle/input/datasets/derrickmwiti/google-2019-cluster-sample/"
                                            "borg_traces_data.csv"),
                usecols=["collection_id", "instance_index", "start_time", "end_time", "average_usage", "failed"])
t = t.dropna(subset=["instance_index", "start_time", "end_time", "average_usage"])
t = t[(t.failed == 0) & (t.end_time > t.start_time)]
t = t.assign(u=t.average_usage.map(lambda x: ast.literal_eval(x)["cpus"] if isinstance(x, str) else np.nan))
t = t.dropna(subset=["u"]).reset_index(drop=True)
t["b"] = (t.u * 100).astype(int) + 1
assert len(t) == 313_216, len(t)
window = (t.end_time - t.start_time) / 1e6            # s; trace times are in microseconds
work = t.u * window                                  # NCU-seconds used in the window (u is in NCUs)
print(f"\nTrace rows: {len(t):,}; b = 1 for {(t.b == 1).mean() * 100:.2f}%; mean b {t.b.mean():.2f}, max {t.b.max()}")
print(f"Usage window (end - start): median {window.median():.0f} s, max {window.max():.0f} s")
print(f"CPU work per row u*(end - start): median {work.median():.2f}, p90 {work.quantile(0.9):.1f}, "
      f"max {work.max():.0f} NCU-s")
print(f"Spearman(b, CPU work) = {spearmanr(t.b, work).statistic:.2f}; "
      f"Spearman(b, window) = {spearmanr(t.b, window).statistic:.2f}")
print(f"Trace span: {(t.start_time.max() - t.start_time.min()) / 86_400e6:.1f} days; rows come from "
      f"{t.groupby(['collection_id', 'instance_index']).ngroups:,} task instances; "
      f"{(t.start_time % 300_000_000 == 0).mean() * 100:.1f}% of windows start on a 5-min boundary")

s = t.sample(500, random_state=42)                   # the rows make_workload(42) draws (depends on row count + seed)
arr = (s.start_time - s.start_time.min()) // 10_000_000
assert sorted(zip(arr, s.b * 10_000)) == sorted(zip(w.Arrival_Time, w.Burst_Time)), "sample != workload CSV"
span_s = (s.start_time.max() - s.start_time.min()) / 1e6
same = s.groupby("start_time").size()
print(f"Sample, raw trace times: {same[same > 1].sum()} rows share an exact start time (largest group {same.max()}); "
      f"the 500 rows come from {s.collection_id.nunique()} collections (Borg jobs)")
print(f"Sample: spans {span_s / 86_400:.1f} days; native load = CPU work / span = "
      f"{work[s.index].sum() / span_s:.2e} NCU ({work[s.index].sum() / span_s * 100:.3f}% of the largest machine)")

UNIT_S = 1e-4                                        # Kernel Baseline: one Time Unit = 100 us of CPU time
laxity = (w.Deadline - w.Arrival_Time) / w.Burst_Time
print(f"Kernel time scale: bursts {w.Burst_Time.min() * UNIT_S:.0f}-{w.Burst_Time.max() * UNIT_S:.0f} s "
      f"(modal {modal * UNIT_S:.0f} s), all arrivals within {w.Arrival_Time.max() * UNIT_S:.1f} s, "
      f"relative deadline {laxity.min():.2f}-{laxity.max():.2f} x burst")

# --- Comparison with Wierman & Harchol-Balter: slowdown Jain below saturation (load_sweep.csv, 5 seeds) ---
ls = pd.read_csv(HERE / "load_sweep.csv")
sj = ls[ls.rho < 1].groupby(["rho", "policy"]).slowdown_jain.median().unstack()
print("\nMedian slowdown Jain below saturation (top three per load):")
for r, row in sj.iterrows():
    top = row.sort_values(ascending=False).head(3)
    print(f"  rho = {r}: " + ", ".join(f"{p} {v:.3f}" for p, v in top.items()))

# --- Figures: one IEEE column (3.5 in = 252 pt), no tight bbox, so 8 pt text prints at 8 pt -----------
import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": ["Liberation Sans", "DejaVu Sans"],
                     "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "figure.constrained_layout.use": True})
LINE, INK = "#2a78d6", "#52514e"


def style(ax):
    ax.grid(linewidth=0.4, alpha=0.5)
    ax.tick_params(length=2, colors=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.5)


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf")
    fig.savefig(HERE / f"{name}.png", dpi=300)


# RR quantum (paper Fig. 3): dense grid over 1..10^6 plus each divisor of the modal burst and 10 below
teeth = [q for k in range(1, 11) for q in (int(np.ceil(modal / k)) - 10, int(np.ceil(modal / k)))]
qs = sorted(set(np.unique(np.logspace(0, 6, 700).astype(int))) | set(teeth))
runs = [rr(w, quantum=q) for q in qs]
wt = [r["avg_waiting"] / 1e6 for r in runs]
cs = [r["context_switches"] for r in runs]
print(f"\nRR dense sweep: {len(qs)} quanta, WT {min(wt):.2f}-{max(wt):.2f} x10^6 "
      f"(worst q = {qs[int(np.argmax(wt))]:,}, best q = {qs[int(np.argmin(wt))]:,}); "
      f"RR = FCFS for q >= {min(q for q, r in zip(qs, runs) if r['context_switches'] == len(w)):,} "
      f"(largest burst {w.Burst_Time.max():,})")
q10 = wt[qs.index(10)]
low = [x for q, x in zip(qs, wt) if q <= 178]
mid = [x for q, x in zip(qs, wt) if 10_000 <= q <= 30_000]
print(f"  q <= 178 ({len(low)} quanta): WT {min(low):.4f}-{max(low):.4f} x10^6, "
      f"within {max(abs(x - q10) for x in low) / q10 * 100:.2f}% of q = 10 ({q10:.4f})")
print(f"  10^4 <= q <= 3x10^4: WT {min(mid):.3f}-{max(mid):.3f} x10^6")
print(f"  80th percentile of the bursts (the quantum rule of Zohora et al.): {np.percentile(w.Burst_Time, 80):,.0f}")
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.1), gridspec_kw={"width_ratios": [1.9, 1]})
ax.plot(qs, wt, color=LINE, linewidth=0.8)
ax.axhline(wt[-1], color=INK, linewidth=0.5, linestyle="--")
ax.text(1.3, wt[-1] - 0.6, "FCFS", color=INK)
for k in (1, 2):
    ax.axvline(modal / k, color=INK, linewidth=0.4, linestyle=":")
ax.text(modal * 1.15, 0.25, "b", color=INK)
ax.text(modal / 2 * 0.87, 0.25, "b/2", color=INK, ha="right")
ax.annotate("worst: q = 9,990", (9_990, 5.41), (6, 5.85), color=INK,
            arrowprops={"arrowstyle": "-", "color": INK, "linewidth": 0.4})
ax.annotate("best: q = b", (10_000, 3.02), (30, 1.4), color=INK,
            arrowprops={"arrowstyle": "-", "color": INK, "linewidth": 0.4})
ax.set_xscale("log")
ax.set_ylim(0, 6.5)
ax.set_xlabel("Quantum q (units, log)")
ax.set_title("Mean waiting time (×10⁶ units)", loc="left", pad=3)
ax2.plot(qs, cs, color=LINE, linewidth=0.8)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax2.set_xlabel("q (units, log)")
ax2.set_title("Context switches", loc="left", pad=3)
for a in (ax, ax2):
    a.set_xticks([1, 1e2, 1e4, 1e6])
    a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    style(a)
save(fig, "rr_sawtooth")

# Arrival clustering: load in a sliding window of 1% of the arrival span, at rho = 0.5
w5 = stretch(w, 0.5)
a, b = w5.Arrival_Time.to_numpy(), w5.Burst_Time.to_numpy()
width = 0.01 * a.max()
starts = np.linspace(0, a.max() - width, 2000)
load = [b[(a >= t) & (a < t + width)].sum() / width for t in starts]
fig, ax = plt.subplots(figsize=(3.5, 1.5))
ax.plot(starts / a.max() * 100, load, color=LINE, linewidth=0.8)
ax.axhline(1, color=INK, linewidth=0.6, linestyle="--")
ax.text(100, 1.08, "one core's capacity", color=INK, ha="right")
ax.set_xlim(0, 100)
ax.set_ylim(bottom=0)
ax.set_xlabel("Window start (% of arrival span)")
ax.set_title(f"Offered load in 1% windows at ρ = 0.5 (peak {max(load):.1f})", loc="left", pad=3)
style(ax)
save(fig, "arrival_clustering")
print("\nFigures: rr_sawtooth.pdf/.png, arrival_clustering.pdf/.png")
