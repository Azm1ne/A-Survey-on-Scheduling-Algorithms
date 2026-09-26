"""Numbers quoted in the paper's Discussion and RR section that the notebooks do not print.

    python paper_numbers.py

Uses kernel/workload_seed42.csv (written by cpu-scheduling-algorithm-comparison.ipynb) and the
notebook's own policy functions, so every value comes from the code that produced Table II.
"""
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

# --- Figures (single column, same style as the notebook figures) ---------------------------------
import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "axes.titlesize": 7,
                     "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
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
    fig.tight_layout(pad=0.3)
    fig.savefig(HERE / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(HERE / f"{name}.png", dpi=300, bbox_inches="tight")


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
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.65), gridspec_kw={"width_ratios": [1.75, 1]})
ax.plot(qs, wt, color=LINE, linewidth=0.8)
ax.axhline(wt[-1], color=INK, linewidth=0.5, linestyle="--")
ax.text(1.3, wt[-1] - 0.45, "FCFS", color=INK, fontsize=6)
for k in (1, 2):
    ax.axvline(modal / k, color=INK, linewidth=0.4, linestyle=":")
ax.text(modal * 1.15, 0.2, "b", color=INK, fontsize=6)
ax.text(modal / 2 * 0.87, 0.2, "b/2", color=INK, ha="right", fontsize=6)
ax.annotate("q = 9,990: worst", (9_990, 5.41), (12, 5.75), color=INK, fontsize=6,
            arrowprops={"arrowstyle": "-", "color": INK, "linewidth": 0.4})
ax.annotate("q = b: best", (10_000, 3.02), (16_000, 1.6), color=INK, fontsize=6,
            arrowprops={"arrowstyle": "-", "color": INK, "linewidth": 0.4})
ax.set_xscale("log")
ax.set_ylim(0, 6.2)
ax.set_xlabel("Quantum q (units, log)")
ax.set_title("Mean waiting time (×10⁶ units)", loc="left", pad=3)
ax2.plot(qs, cs, color=LINE, linewidth=0.8)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Quantum q (units, log)")
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
ax.text(100, 1.08, "one core's capacity", color=INK, ha="right", fontsize=6)
ax.set_xlim(0, 100)
ax.set_ylim(bottom=0)
ax.set_xlabel("Window start (% of arrival span)")
ax.set_title(f"Offered load in 1% windows at ρ = 0.5 (peak {max(load):.1f})", loc="left", pad=3)
style(ax)
save(fig, "arrival_clustering")
print("\nFigures: rr_sawtooth.pdf/.png, arrival_clustering.pdf/.png")
