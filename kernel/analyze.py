"""Compare a Kernel Baseline campaign (campaign.sh) with the simulator.

    python analyze.py results/<campaign-dir>

Writes runs.csv (per-run metrics) and tau.csv (Rank Agreement per replicate) into that directory
and prints the tables used in the paper. The simulator policies are executed from the notebook
itself, so the comparison always uses the code that produced Table I.
"""
import heapq, json, re, sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent
KEYS = ["avg_waiting", "avg_turnaround", "avg_response"]

# Kernel Counterparts of the validated policies (SJF, MLQ, MLFQ are Simulation-only)
COUNTERPART = {"FCFS": "fifo", "Priority": "fifo-prio", "Round Robin": "rr-q1ms", "EDF": "deadline"}
EEVDF = {"Shipped EEVDF": "eevdf", "Matched-slice EEVDF": "eevdf-slice"}
RR_QUANTA_MS = [1, 2, 5, 10, 20, 50, 100, 1000, 1100]


def simulator():
    """Load the policy functions from the notebook (Section 6 cells)."""
    nb = json.loads((HERE.parent / "cpu-scheduling-algorithm-comparison.ipynb").read_text())
    ns = {"np": np, "pd": pd, "heapq": heapq, "deque": deque}
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and re.match(r"\s*def (fairness_index|fcfs|sjf|round_robin|priority_sched|"
                                                    r"multilevel_queue|mlfq|edf|cfs)\b", src):
            exec(src, ns)
    return ns


def jain(x):
    x = np.asarray(x, float)
    return x.sum() ** 2 / (len(x) * (x ** 2).sum() + 1e-9)


def kernel_metrics(csv, unit_ns):
    d = pd.read_csv(csv)
    tt = (d.finish_ns - d.release_ns) / unit_ns
    wt = tt - d.burst_units
    return {
        "avg_waiting": wt.mean(), "avg_turnaround": tt.mean(),
        "avg_response": ((d.start_ns - d.release_ns) / unit_ns).mean(),
        "context_switches": int((d.nvcsw + d.nivcsw).sum()),
        "wt_jain": jain(wt), "slowdown_jain": jain(tt / d.burst_units),
        "miss_ratio": (tt > d.rel_deadline_units).mean(),
        "cpu_overshoot_pct": ((d.cpu_ns / unit_ns - d.burst_units) / d.burst_units).max() * 100,
    }


def main(run_dir):
    run_dir = Path(run_dir)
    unit_ns = int(re.search(r"unit_ns=(\d+)", (run_dir / "env.txt").read_text()).group(1))
    runs = []
    for f in sorted(run_dir.glob("*_rep*_cpu*.csv")):
        name, rep, cpu = re.match(r"(.+)_rep(\d+)_cpu(\d+)\.csv", f.name).groups()
        runs.append({"run": name, "rep": int(rep), "cpu": int(cpu), **kernel_metrics(f, unit_ns)})
    runs = pd.DataFrame(runs)
    runs.to_csv(run_dir / "runs.csv", index=False)
    n_jobs = len(pd.read_csv(next(run_dir.glob("fifo_rep1_*.csv"))))

    sim = simulator()
    w = pd.read_csv(HERE / "workload_seed42.csv").head(n_jobs)
    sim_policies = {"FCFS": sim["fcfs"], "Priority": sim["priority_sched"], "Round Robin": sim["round_robin"],
                    "EDF": sim["edf"], "CFS": sim["cfs"]}
    sim_res = pd.DataFrame({k: f(w) for k, f in sim_policies.items()}).T
    rr_sim = pd.DataFrame({q: sim["round_robin"](w, quantum=q * 10) for q in RR_QUANTA_MS}).T
    # the kernel's tick-based slice ends a hair before a job's CPU clock, so q ms acts like q*10 - epsilon units
    rr_sim_eps = pd.DataFrame({q: sim["round_robin"](w, quantum=q * 10 - 1) for q in RR_QUANTA_MS}).T

    pd.set_option("display.width", 200)
    fmt = lambda x: f"{x:,.3f}" if abs(x) < 10 else f"{x:,.0f}"
    median = runs.groupby("run").median(numeric_only=True)
    spread = runs.groupby("run").agg(["min", "max"])
    print(f"== Kernel runs ({n_jobs} jobs, median over replicates; Time Units)")
    print(median.drop(columns=["rep", "cpu"]).to_string(float_format=fmt))
    print("\nmax CPU overshoot per run (% of burst):", runs.cpu_overshoot_pct.max().round(4))

    # Rank Agreement: Kendall tau between simulator and kernel orderings of the 5 validated policies
    taus = []
    for label, eevdf_run in EEVDF.items():
        mapping = {**COUNTERPART, "CFS": eevdf_run}
        for rep in sorted(runs.rep.unique()):
            k = runs[runs.rep == rep].set_index("run")
            if not set(mapping.values()) <= set(k.index):
                continue
            for key in KEYS:
                s = [sim_res.loc[p, key] for p in mapping]
                kv = [k.loc[r, key] for r in mapping.values()]
                taus.append({"cfs_counterpart": label, "rep": rep, "metric": key,
                             "tau": kendalltau(s, kv).statistic,
                             "sim_order": " < ".join(sorted(mapping, key=lambda p: sim_res.loc[p, key])),
                             "kernel_order": " < ".join(sorted(mapping, key=lambda p: k.loc[mapping[p], key]))})
    taus = pd.DataFrame(taus)
    taus.to_csv(run_dir / "tau.csv", index=False)
    if len(taus):
        print("\n== Rank Agreement (Kendall tau, simulator vs kernel, 5 policies), per replicate")
        print(taus.pivot_table(index=["cfs_counterpart", "metric"], columns="rep", values="tau").round(3).to_string())
        print("\nOrderings (replicate 1):")
        for _, r in taus[taus.rep == 1].iterrows():
            print(f"  {r.cfs_counterpart:20} {r.metric:15} sim:    {r.sim_order}\n  {'':20} {'':15} kernel: {r.kernel_order}")

    # Trade-off Direction: preemptive policies give lower response time for more context switches
    print("\n== Trade-off Direction (median kernel run vs simulator)")
    checks = [("RT(RR) < RT(FCFS)", "Round Robin", "rr-q1ms", "FCFS", "fifo", "avg_response", "<"),
              ("RT(CFS/EEVDF) < RT(FCFS)", "CFS", "eevdf", "FCFS", "fifo", "avg_response", "<"),
              ("CS(RR) > CS(FCFS)", "Round Robin", "rr-q1ms", "FCFS", "fifo", "context_switches", ">"),
              ("WT-Jain(RR) > WT-Jain(FCFS)", "Round Robin", "rr-q1ms", "FCFS", "fifo", "wt_jain", ">"),
              ("WT(EDF) < WT(FCFS)", "EDF", "deadline", "FCFS", "fifo", "avg_waiting", "<")]
    for text, sa, ka, sb, kb, key, op in checks:
        if ka not in median.index or kb not in median.index:
            continue
        cmp = (lambda a, b: a < b) if op == "<" else (lambda a, b: a > b)
        s_ok = cmp(sim_res.loc[sa, key], sim_res.loc[sb, key])
        k_ok = cmp(median.loc[ka, key], median.loc[kb, key])
        print(f"  {text:30} simulator {'yes' if s_ok else 'NO ':3}  kernel {'yes' if k_ok else 'NO '}"
              f"  -> {'HOLDS' if s_ok and k_ok else 'DIFFERS'}")

    # Paper Table III: median kernel run in Table II's units (WT, RT, CS in 10^3), then tau per EEVDF variant
    rows = {"FCFS": "fifo", "Priority": "fifo-prio", "RR (1 ms)": "rr-q1ms", "EDF": "deadline",
            "Shipped EEVDF": "eevdf", "Matched-slice EEVDF": "eevdf-slice"}
    k3 = lambda x: f"{x / 1e3:.2f}" if x < 1e4 else f"{x / 1e3:.1f}"
    if set(rows.values()) <= set(median.index):
        print("\n== Table III (median of replicates): WT, RT, CS (x10^3), J_W, J_S, miss")
        for label, run in rows.items():
            m = median.loc[run]
            print(f"  {label:20} {k3(m.avg_waiting):>7} {k3(m.avg_response):>7} {m.context_switches / 1e3:>6.1f} "
                  f"{m.wt_jain:.3f} {m.slowdown_jain:.3f} {m.miss_ratio:.3f}")
        if len(taus):
            med_tau = taus.groupby(["cfs_counterpart", "metric"]).tau.median().unstack()
            print("  median tau:\n" + med_tau.round(2).to_string())

    # RR quantum: kernel q in ms = simulator q in units / 10
    quanta = [q for q in RR_QUANTA_MS if f"rr-q{q}ms" in median.index]
    rr = [f"rr-q{q}ms" for q in quanta]
    if rr:
        print("\n== RR quantum (median kernel run vs simulator)")
        t = pd.DataFrame({"q_ms": quanta,
                          "sim_WT": rr_sim.loc[quanta, "avg_waiting"].values,
                          "sim_WT_q-1": rr_sim_eps.loc[quanta, "avg_waiting"].values,
                          "kernel_WT": median.loc[rr, "avg_waiting"].values,
                          "sim_CS": rr_sim.loc[quanta, "context_switches"].values,
                          "kernel_CS": median.loc[rr, "context_switches"].values})
        t["dev_%"] = (t.kernel_WT / t["sim_WT_q-1"] - 1) * 100
        print(t.to_string(index=False, float_format=fmt))
        for col in ["sim_WT", "sim_WT_q-1", "kernel_WT"]:
            print(f"  {col} spread over quanta: {(t[col].max() - t[col].min()) / t[col].mean() * 100:.2f}% of mean")
        fine = t[t.q_ms <= 100]
        print(f"  kernel vs sim_WT_q-1, 1-100 ms: max |dev| {fine['dev_%'].abs().max():.2f}% "
              f"(at {fine.q_ms[fine['dev_%'].abs().idxmax()]} ms)")
        if {1000, 1100} <= set(quanta):
            k = t.set_index("q_ms").kernel_WT
            print(f"  kernel 1.0 s -> 1.1 s: {k[1000]:,.0f} -> {k[1100]:,.0f} ({(k[1100] / k[1000] - 1) * 100:+.1f}%)")

    if "fifo-alone" in median.index:
        a, p = median.loc["fifo-alone", "avg_waiting"], median.loc["fifo", "avg_waiting"]
        lo, hi = spread.loc["fifo", ("avg_waiting", "min")], spread.loc["fifo", ("avg_waiting", "max")]
        print(f"\n== Interference check: FIFO alone WT {a:,.1f} vs parallel lanes median {p:,.1f} "
              f"(range {lo:,.1f}–{hi:,.1f}); difference {(a - p) / p * 100:+.3f}%")


if __name__ == "__main__":
    main(sys.argv[1])
