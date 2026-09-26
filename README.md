# A Survey on Scheduling Algorithms

Code, notebooks and kernel harness for the paper *"From First-Come-First-Served to EEVDF: An Empirical Survey of CPU Scheduling Using Google Borg Cluster Traces"* (IEEE CSDE 2026). The paper reviews eight classical and modern CPU scheduling algorithms and tests them on a workload taken from Google's 2019 Borg cluster trace, first in a simulator and then on a live Linux kernel.

## Why this exists

Most scheduling comparisons run on synthetic, hand-picked job sets. This project uses task-instance data from a production cluster manager, so the arrival pattern and burst sizes come from real machines. The question is simple: does any single scheduling policy win across the board, or does the "best" algorithm depend on what you optimize for?

## What's in this repo

- `cpu-scheduling-algorithm-comparison.ipynb`: FCFS, SJF, Round Robin, Priority, MLQ, MLFQ, EDF and CFS as discrete-event simulations, run on a 500-job Borg sample. It reports waiting, turnaround and response time, throughput, context switches, Jain's index over waiting times and over slowdowns, and the deadline miss ratio. A **load sweep** (Section 9) re-runs everything at offered loads from 0.5 to 30 over five samples. It writes `table1_results.csv`, `load_sweep*.csv`, `scheduling_comparison.pdf` and `kernel/workload_seed42.csv`.
- `rr-time-quantum-sensitivity-multi-core-affinity.ipynb`: sweeps Round Robin's quantum over 25 values from 1 to 10⁶ time units (`rr_quantum_sensitivity.pdf`; the paper uses the denser sweep of `paper_numbers.py`). It also contains a multi-core affinity model, which is **not** part of the paper. Its hit rates are inputs of the model (see the note in the notebook).
- `kernel/`: the Kernel Baseline. `replay.c` replays the same 500-job stream as real processes pinned to one CPU under SCHED_FIFO, SCHED_RR, SCHED_DEADLINE or EEVDF. `campaign.sh` runs every policy (3 replicates, root, about 3.4 h). `analyze.py` compares the kernel runs with the simulator by Kendall τ. Raw results are in `kernel/results/`.
- `paper_numbers.py`: recomputes the numbers the paper quotes outside the notebooks (arrival clustering, tier burst means, the deadline–burst correlation, the RR cliff) from `kernel/workload_seed42.csv` with the notebook's own policy code, and draws `rr_sawtooth.pdf` (the paper's RR figure: 569 quanta from 1 to 10⁶, with context switches) and `arrival_clustering.pdf`. Runs in about 15 seconds.
- `methodology_diagram.py`: draws the paper's pipeline figure (`methodology.pdf`) in the [figures4papers](https://github.com/ChenLiu-1996/figures4papers) style.

## Data and workload model

The [Google 2019 Cluster Sample](https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample) (from the Borg trace described by Tirmazi et al., EuroSys 2020) has 405,894 task-instance rows. Keeping jobs with `failed == 0` and `end_time > start_time` leaves 313,216. A job's burst is `floor(100 × average CPU usage) + 1`. Borg priorities keep the trace's meaning: a **larger** value is more important (production ≥ 120).

The 500-job sample (seed 42) has its arrivals divided by 10⁷ and its bursts multiplied by 10⁴. That puts the **offered load** at ρ = 30.0: work arrives 30 times faster than one core can serve it. At this load every work-conserving policy has the same makespan and throughput and differs only in the order it serves jobs. The load sweep checks that the rankings also hold at ρ ≤ 1. The Borg trace has no deadlines, so EDF uses synthetic, burst-proportional ones: D = (1+s)·burst, s ~ U(0.5, 2).

## What the experiments found

Simulated, 500 jobs, ρ = 30 (WT, RT in ×10³ time units):

| Algorithm | Avg wait | Avg response | Context switches | Waiting-time Jain | Slowdown Jain | Deadline miss |
|---|---|---|---|---|---|---|
| FCFS | 3839.6 | 3839.6 | 0 | 0.743 | 0.686 | 0.994 |
| SJF | **2492.5** | 2492.5 | 500 | 0.707 | 0.771 | 0.996 |
| Round Robin (q=10) | 5018.8 | 2.49 | 799,000 | **0.987** | **0.943** | 1.000 |
| Priority | 4413.1 | 4413.1 | 500 | 0.777 | 0.724 | 0.996 |
| MLQ | 4102.9 | 3528.4 | 322,335 | 0.807 | 0.740 | 1.000 |
| MLFQ | 3845.4 | **0.013** | 1,768 | 0.745 | 0.687 | 0.994 |
| EDF | 2967.1 | 2966.8 | 501 | 0.719 | 0.672 | 0.998 |
| CFS | 5020.7 | 0.272 | 1,597,752 | **0.987** | **0.943** | 1.000 |

- SJF has the lowest waiting time and MLFQ the lowest response time. Round Robin and CFS tie on both fairness indices and pay for it with the most context switches.
- These rankings hold across the load sweep (median Kendall τ at ρ = 0.9 vs ρ = 30: 0.71 for waiting time, 0.93 for response time).
- The Round Robin quantum barely changes waiting time while it is small. Larger quanta give a sawtooth: waiting time drops whenever q divides the most common burst (10⁴ units) and rises just below each such q. The biggest step is at the burst itself: q = 9,990 is the worst quantum (5.41×10⁶) and q = 10⁴ the best (3.02×10⁶, 40% below q = 10). Past the largest burst, RR equals FCFS. Context switches fall as 1/q throughout.
- Kernel Baseline: see `kernel/results/` once the campaign has run.

### Corrections since the first version of this repo

Several numbers from earlier versions changed. The details are in the paper's camera-ready response letter.
- Priority and MLQ now run the **largest** Borg priority first; earlier versions inverted this.
- The CFS row now comes from the corrected vruntime code; the old table was stale.
- EDF deadlines are now burst-proportional.
- The RR study uses the same filtered data as the comparison.
- The "knee" in the RR sweep was the sweep's end point.
- The affinity hit rates (95–97% soft, 6.2% random) are inputs of the model, not findings.

## Running it

```bash
pip install pandas numpy matplotlib scipy nbconvert ipykernel
export BORG_CSV=/path/to/borg_traces_data.csv     # defaults to the Kaggle input path
jupyter nbconvert --to notebook --execute --inplace cpu-scheduling-algorithm-comparison.ipynb
jupyter nbconvert --to notebook --execute --inplace rr-time-quantum-sensitivity-multi-core-affinity.ipynb
python paper_numbers.py                                # needs kernel/workload_seed42.csv from the first notebook

# Kernel Baseline (Linux >= 6.12, root; changes 4 scheduler sysctls and restores them on exit)
cd kernel && gcc -O2 -Wall -o replay replay.c
sudo JOBS=10 ./campaign.sh                            # ~4 min smoke run
sudo nohup ./campaign.sh > campaign.log 2>&1 &        # full run, ~3.4 h
python analyze.py results/<run-dir>
```
