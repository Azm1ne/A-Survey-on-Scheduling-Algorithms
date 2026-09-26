# A Survey on Scheduling Algorithms

Code, notebooks and kernel harness for the paper *"From First-Come-First-Served to EEVDF: An Empirical Survey of CPU Scheduling Using Google Borg Cluster Traces"* (IEEE CSDE 2026). The paper reviews eight classical and modern CPU scheduling algorithms and tests them on a workload taken from Google's 2019 Borg cluster trace, first in a simulator and then on a live Linux kernel.

## Why this exists

Most scheduling comparisons run on synthetic, hand-picked job sets. This project uses task-instance data from a production cluster manager, so the arrival pattern and burst sizes come from real machines. The question is simple: does any single scheduling policy win across the board, or does the "best" algorithm depend on what you optimize for?

## What's in this repo

- `cpu-scheduling-algorithm-comparison.ipynb`: FCFS, SJF, Round Robin, Priority, MLQ, MLFQ, EDF and CFS as discrete-event simulations, run on a 500-job Borg sample. It reports waiting, turnaround and response time, throughput, context switches, Jain's index over waiting times and over slowdowns, and the deadline miss ratio. A **load sweep** (Section 9) re-runs everything at offered loads from 0.5 to 30 over five samples. It writes `table1_results.csv`, `load_sweep*.csv`, `scheduling_comparison.pdf` and `kernel/workload_seed42.csv`.
- `rr-time-quantum-sensitivity-multi-core-affinity.ipynb`: sweeps Round Robin's quantum over 25 values from 1 to 10⁶ time units (`rr_quantum_sensitivity.pdf`; the paper uses the denser sweep of `paper_numbers.py`). It also contains a multi-core affinity model, which is **not** part of the paper. Its hit rates are inputs of the model (see the note in the notebook).
- `kernel/`: the Kernel Baseline. `replay.c` replays the same 500-job stream as real processes pinned to one CPU under SCHED_FIFO, SCHED_RR, SCHED_DEADLINE or EEVDF. `campaign.sh` runs every policy (3 replicates, root, about 3.4 h). `analyze.py` compares the kernel runs with the simulator by Kendall τ. Raw results are in `kernel/results/`.
- `paper_numbers.py`: recomputes every number the paper quotes that the notebooks do not print, with the notebook's own policy and workload code: what a trace row is (usage windows, NCUs, native load), arrival clustering, tier burst means, the deadline–burst correlation, the RR cliff and sawtooth, why the rankings persist below saturation (share of jobs arriving to a busy core), the deadline-model sensitivity (four deadline models), and **ERRDTQ**, the dynamic-quantum Round Robin of Zohora et al. (PLOS ONE 2024, Algorithm 1), run as an existing method. It draws `rr_sawtooth.pdf` (the paper's Fig. 3: 569 quanta from 1 to 10⁶, with context switches) and `arrival_clustering.pdf`. It asserts that its trace sample reproduces `kernel/workload_seed42.csv`. Needs `BORG_CSV`; runs in about 20 seconds.
- `methodology_diagram.py`: draws the paper's pipeline figure (`methodology.pdf`) in the [figures4papers](https://github.com/ChenLiu-1996/figures4papers) style.

## Data and workload model

The [Google 2019 Cluster Sample](https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample) (from the Borg trace described by Tirmazi et al., EuroSys 2020) has 405,894 task-instance rows. Keeping jobs with `failed == 0` and `end_time > start_time` leaves 313,216. A job's burst is `floor(100 × average CPU usage) + 1`. Borg priorities keep the trace's meaning: a **larger** value is more important (production ≥ 120).

The 500-job sample (seed 42) has its arrivals divided by 10⁷ and its bursts multiplied by 10⁴. That puts the **offered load** at ρ = 30.0: work arrives 30 times faster than one core can serve it. At this load every work-conserving policy has the same makespan and throughput and differs only in the order it serves jobs. The load sweep checks that the rankings also hold at ρ ≤ 1. The Borg trace has no deadlines, so EDF uses synthetic, burst-proportional ones: D = (1+s)·burst, s ~ U(0.5, 2).

## What the experiments found

Simulated, 500 jobs, ρ = 30 (WT, RT in ×10³ time units):

| Algorithm | Avg wait | Avg response | Context switches | Waiting-time Jain | Slowdown Jain | Deadline miss |
|---|---|---|---|---|---|---|
| FCFS | 3839.6 | 3839.6 | 500 | 0.743 | 0.686 | 0.994 |
| SJF | **2492.5** | 2492.5 | 500 | 0.707 | 0.771 | 0.996 |
| Round Robin (q=10) | 5018.8 | 2.49 | 799,000 | **0.987** | **0.943** | 1.000 |
| Priority | 4413.1 | 4413.1 | 500 | 0.777 | 0.724 | 0.996 |
| MLQ | 4102.9 | 3528.4 | 322,335 | 0.807 | 0.740 | 1.000 |
| MLFQ | 3845.4 | **0.013** | 1,768 | 0.745 | 0.687 | 0.994 |
| EDF | 2967.1 | 2966.8 | 501 | 0.719 | 0.672 | 0.998 |
| CFS | 5020.7 | 0.272 | 1,597,752 | **0.987** | **0.943** | 1.000 |
| ERRDTQ (Zohora et al.) | 2492.5 | 2492.5 | 500 | 0.707 | 0.771 | 0.996 |

- SJF has the lowest waiting time and MLFQ the lowest response time. Round Robin and CFS tie on both fairness indices and pay for it with the most context switches.
- These rankings hold across the load sweep (median Kendall τ at ρ = 0.9 vs ρ = 30: 0.71 for waiting time, 0.93 for response time).
- The Round Robin quantum barely changes waiting time while it is small. Larger quanta give a sawtooth: waiting time drops whenever q divides the most common burst (10⁴ units) and rises just below each such q. The biggest step is at the burst itself: q = 9,990 is the worst quantum (5.41×10⁶) and q = 10⁴ the best (3.02×10⁶, 40% below q = 10). Past the largest burst, RR equals FCFS. Context switches fall as 1/q throughout.
- ERRDTQ schedules exactly as SJF at ρ = 30; below saturation it waits 18% (ρ = 0.5) and 26% (ρ = 0.9) less than SJF, but it also needs the bursts in advance.
- The waiting-time ordering is the same under four deadline models (tighter, looser and burst-independent deadlines); EDF's lead over FCFS needs burst-proportional deadlines.
- Kernel Baseline (Linux 7.0, `kernel/results/20260926-105859`): the kernel orders FCFS, Priority, RR, EDF and EEVDF exactly as the simulator does on waiting and turnaround time (Kendall τ = 1.0 in all three replicates, for both EEVDF slices), and τ = 0.8–1.0 on response time. SCHED_RR tracks the simulator's sawtooth: 1 s is the worst kernel quantum, and 1.1 s cuts waiting time by 42.5%.

### Corrections since the first version of this repo

Several numbers from earlier versions changed. The details are in the paper's camera-ready response letter.
- Priority and MLQ now run the **largest** Borg priority first; earlier versions inverted this.
- The CFS row now comes from the corrected vruntime code; the old table was stale.
- EDF deadlines are now burst-proportional.
- The RR study uses the same filtered data as the comparison.
- The "knee" in the RR sweep was the sweep's end point.
- The affinity hit rates (95–97% soft, 6.2% random) are inputs of the model, not findings.

## Where each result in the paper comes from

| Paper artifact | Script | Output |
|---|---|---|
| Table II (except the ERRDTQ row), Table IV, Fig. 2, the stream the kernel replays | `cpu-scheduling-algorithm-comparison.ipynb` | `table1_results.csv`, `load_sweep*.csv`, `scheduling_comparison.pdf`, `kernel/workload_seed42.csv` |
| Fig. 3, the ERRDTQ row of Table II, and the numbers of §III-A, §III-B, §IV-D and §V that the notebook does not print | `paper_numbers.py` | stdout, `rr_sawtooth.pdf` |
| Fig. 1 | `methodology_diagram.py` | `methodology.pdf` |
| The kernel runs | `kernel/replay.c`, `kernel/campaign.sh` | `kernel/results/<run>/` |
| Table III and the kernel numbers of §IV-B, §IV-D and §V | `kernel/analyze.py` | stdout, `runs.csv`, `tau.csv` |

## Running it

Tested with Python 3.14.4, pandas 3.0.6, NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2 and nbconvert 7.17.1. The data is `borg_traces_data.csv` from the Kaggle sample linked above (SHA-256 `c3362a78555b7bd10d27a693da4e4c7230ac323359a5bfe0e75340c27b416b0d`). The kernel runs used Linux 7.0.0-31-generic on an AMD Ryzen 5 7535HS (`kernel/results/20260926-105859/env.txt`).

```bash
pip install pandas numpy matplotlib scipy nbconvert ipykernel
export BORG_CSV=/path/to/borg_traces_data.csv     # defaults to the Kaggle input path
jupyter nbconvert --to notebook --execute --inplace cpu-scheduling-algorithm-comparison.ipynb
jupyter nbconvert --to notebook --execute --inplace rr-time-quantum-sensitivity-multi-core-affinity.ipynb
python paper_numbers.py                                # needs BORG_CSV and the first notebook's outputs

# Kernel Baseline (Linux >= 6.12, root; changes 4 scheduler sysctls and restores them on exit)
cd kernel && gcc -O2 -Wall -o replay replay.c
sudo JOBS=10 ./campaign.sh                            # ~4 min smoke run
sudo nohup ./campaign.sh > campaign.log 2>&1 &        # full run, ~3.4 h
python analyze.py results/20260926-105859            # Table III, in about 2 s
```
