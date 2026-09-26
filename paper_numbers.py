"""Numbers quoted in the paper that the notebooks do not print, and Figs. 3 (RR quantum) and arrival clustering.

    BORG_CSV=/path/to/borg_traces_data.csv python paper_numbers.py

Uses kernel/workload_seed42.csv (written by cpu-scheduling-algorithm-comparison.ipynb), the
notebook's own policy functions, load_sweep.csv, and the trace CSV (same path variable as the notebooks),
so every value comes from the code and data that produced Tables II and IV.
"""
import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

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
                usecols=["collection_id", "instance_index", "start_time", "end_time", "priority", "average_usage",
                         "failed"])
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

# --- Streams for other seeds and loads: the notebook's own make_workload, on the same cleaned rows -----------
nb = json.loads((HERE / "cpu-scheduling-algorithm-comparison.ipynb").read_text())
ns = simulator()
ns["scheduler_df"] = t.rename(columns={"instance_index": "PID", "start_time": "Arrival_Time", "b": "Burst_Time",
                                       "priority": "Priority"})[["PID", "Arrival_Time", "Burst_Time", "Priority"]]
exec(next("".join(c["source"]) for c in nb["cells"] if "".join(c["source"]).startswith("def make_workload"))
     .split("# Reproducible")[0], ns)
make_workload = ns["make_workload"]
assert np.allclose(make_workload(42)[w.columns], w), "make_workload(42) != kernel/workload_seed42.csv"
POLICIES = {"FCFS": "fcfs", "SJF": "sjf", "RR": "round_robin", "Priority": "priority_sched",
            "MLQ": "multilevel_queue", "MLFQ": "mlfq", "EDF": "edf", "CFS": "cfs"}
SEEDS = range(42, 47)
ls = pd.read_csv(HERE / "load_sweep.csv")


def run_all(w):
    return pd.DataFrame({name: ns[f](w) for name, f in POLICIES.items()}).T


# Why the orderings persist below saturation: arrivals cluster, so many jobs still find the core busy.
# Busy periods are the same for every work-conserving policy; FCFS order gives them in one pass.
print("\nBelow saturation (median over seeds 42-46):")
for r in [0.5, 0.9]:
    busy, queue = [], []
    for seed in SEEDS:
        v = make_workload(seed, r)
        a, b = v.Arrival_Time.to_numpy(), v.Burst_Time.to_numpy()
        f = np.empty_like(a)                                   # FCFS finish times (arrival order)
        end = 0
        for k in range(len(a)):
            end = max(end, a[k]) + b[k]
            f[k] = end
        waiting = np.array([((a <= a[j]) & (f > a[j])).sum() - 1 for j in range(len(a))])  # jobs ahead of j
        busy.append((waiting > 0).mean())
        queue.append(waiting[waiting > 0].mean())
    print(f"  rho = {r}: {np.median(busy) * 100:.0f}% of jobs arrive to a busy core, "
          f"finding {np.median(queue):.1f} jobs ahead on average")

# Deadline-model sensitivity (the trace has no deadlines): only EDF's schedule and the miss ratios depend on them.
DEADLINES = {"tight U(0.1,1)": (0.1, 1.0, True), "paper U(0.5,2)": (0.5, 2.0, True),
             "loose U(2,5)": (2.0, 5.0, True), "size-blind (1+U(0.5,2))*mean": (0.5, 2.0, False)}


def with_deadlines(v, seed, lo, hi, proportional):
    s = np.random.default_rng(seed).uniform(lo, hi, size=len(v))  # same draw order as make_workload
    base = v.Burst_Time if proportional else v.Burst_Time.mean()
    return v.assign(Deadline=v.Arrival_Time + (1 + s) * base)


print("\nDeadline models (EDF WT rank of 8 at rho = 30, seed 42; miss ratios: median over seeds 42-46):")
table2 = run_all(w)
for label, (lo, hi, prop) in DEADLINES.items():
    edf30 = ns["edf"](with_deadlines(w, 42, lo, hi, prop))
    alt = table2.copy()
    alt.loc["EDF"] = pd.Series(edf30)
    taus = [kendalltau(table2[k].astype(float), alt[k].astype(float)).statistic for k in ("avg_waiting", "avg_response")]
    out = (f"  {label:<30} EDF WT {edf30['avg_waiting'] / 1e3:,.1f}x10^3 "
           f"({(1 - edf30['avg_waiting'] / table2.loc['FCFS', 'avg_waiting']) * 100:.1f}% below FCFS),"
           f" tau vs paper model WT {taus[0]:.2f} RT {taus[1]:.2f}")
    for r in [0.5, 0.9]:
        m = pd.DataFrame([run_all(with_deadlines(make_workload(seed, r), seed, lo, hi, prop)).miss_ratio
                          for seed in SEEDS]).median()
        out += f"; rho {r}: EDF miss {m['EDF']:.2f}, best {m.min():.2f} ({m.idxmin()})"
    print(out)


def errdtq(df):
    """ERRDTQ, Zohora et al. [11], Alg. 1: the ready queue is sorted by remaining burst and the quantum QT is its
    80th-percentile element, recomputed when a job arrives or finishes. The shortest job runs for up to QT. An arrival
    preempts the running job only if it still needs more than QT/3 and the shortest queued job needs at most QT/3;
    otherwise the running job keeps executing for QT."""
    jobs = df.sort_values("Arrival_Time").to_dict("records")
    n, i, time, cs = len(jobs), 0, 0, 0
    ready, run, first, done = [], None, {}, []            # ready: [remaining, job]; run: [job, remaining, turn end]

    def qt():
        srq = sorted(x[0] for x in ready)
        return srq[max(1, int(0.8 * len(srq) + 0.5)) - 1]

    def admit():
        nonlocal i
        while i < n and jobs[i]["Arrival_Time"] <= time:
            ready.append([jobs[i]["Burst_Time"], i])
            i += 1

    while len(done) < n:
        admit()
        if run is None:
            if not ready:
                time = jobs[i]["Arrival_Time"]
                continue
            q = qt()
            ready.sort()
            rem, k = ready.pop(0)
            first.setdefault(k, time)
            cs += 1
            run = [k, rem, time + min(rem, q)]
        nxt = jobs[i]["Arrival_Time"] if i < n else float("inf")
        if nxt < run[2]:                                  # an arrival: re-evaluate (Alg. 1, steps 10-22)
            run[1] -= nxt - time
            time = nxt
            admit()
            q = qt()
            if run[1] > q / 3 and min(ready)[0] <= q / 3:
                ready.append([run[1], run[0]])
                run = None
            else:
                run[2] = time + min(run[1], q)
            continue
        run[1] -= run[2] - time
        time = run[2]
        k = run[0]
        if run[1] == 0:
            j = jobs[k]
            done.append({"arrival": j["Arrival_Time"], "start": first[k], "finish": time,
                         "burst": j["Burst_Time"], "deadline": j["Deadline"]})
        else:
            ready.append([run[1], k])
        run = None
    return ns["compute_metrics"](done, context_switches=cs)


e = errdtq(w)
assert e["throughput"] == table2.loc["FCFS", "throughput"]         # work-conserving: same makespan as Table II
same = all(np.isclose(e[k], table2.loc["SJF", k]) for k in ("avg_waiting", "avg_response", "context_switches"))
print(f"\nAt rho = 30 ERRDTQ {'equals' if same else 'differs from'} SJF on WT, RT and CS")
print("\nERRDTQ [11] on the seed-42 stream (units of Table II): "
      f"WT {e['avg_waiting'] / 1e3:.1f}, RT {e['avg_response'] / 1e3:.1f}, CS {e['context_switches'] / 1e3:.1f}, "
      f"J_W {e['wt_jain']:.3f}, J_S {e['slowdown_jain']:.3f}, miss {e['miss_ratio']:.3f}")
for r in [0.5, 0.9]:
    v = pd.DataFrame([{**errdtq(make_workload(seed, r)), "seed": seed} for seed in SEEDS]).median()
    sw = ls[ls.rho == r].groupby("policy").avg_waiting.median()  # load_sweep.csv: the 8 policies, same 5 seeds
    print(f"  rho = {r} (median, 5 seeds): WT {v.avg_waiting:,.0f} (SJF {sw['SJF']:,.0f}, best of 8 {sw.min():,.0f} "
          f"{sw.idxmin()}; ERRDTQ {(1 - v.avg_waiting / sw['SJF']) * 100:.0f}% below SJF), RT {v.avg_response:,.0f}, "
          f"miss {v.miss_ratio:.2f}")

# --- Comparison with Wierman & Harchol-Balter: slowdown Jain below saturation (load_sweep.csv, 5 seeds) ---
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
# Tick labels 1, 100, 10k, 1M: plain 8 pt text, no 5.6 pt superscripts
SI = matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}" if x < 1e3 else f"{x / 1e3:g}k" if x < 1e6 else f"{x / 1e6:g}M")


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
ax.set_title("Mean waiting time (millions of units)", loc="left", pad=3)
ax2.plot(qs, cs, color=LINE, linewidth=0.8)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax2.set_xlabel("q (units, log)")
ax2.set_title("Context switches", loc="left", pad=3)
for a in (ax, ax2):
    a.set_xticks([1, 1e2, 1e4, 1e6])
    a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a.xaxis.set_major_formatter(SI)
    style(a)
ax2.yaxis.set_major_formatter(SI)
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
