# A Survey on Scheduling Algorithms

Code and notebooks for the paper *"From First-Come-First-Served to EEVDF: An Empirical Survey of CPU Scheduling Using Google Borg Cluster Traces."* The paper reviews eight classical and modern CPU scheduling algorithms, then tests them against a real workload pulled from Google's 2019 Borg cluster traces.

## Why this exists

Most scheduling comparisons run on synthetic, hand-picked job sets. This project instead uses actual task-instance data from a production cluster manager, so the arrival patterns, burst sizes, and overlap between jobs come from real machines rather than a textbook example. The question driving the work is simple: does any single scheduling policy actually win across the board, or does the "best" algorithm depend on what you're optimizing for?

## What's in this repo

- `cpu-scheduling-algorithm-comparison.ipynb`: implements FCFS, SJF, Round Robin, Priority, MLQ, MLFQ, EDF, and CFS as discrete-event simulations, then runs all eight against a 500-job Borg sample and compares them on waiting time, turnaround time, response time, throughput, Jain's fairness index, and context switches.
- `rr-time-quantum-sensitivity-multi-core-affinity.ipynb`: two follow-up studies. The first sweeps Round Robin's time quantum from 1 to 500 to see how it trades off context switch overhead against waiting time. The second dispatches 2,000 jobs across 2, 4, 8, and 16 cores under random, soft, and hard affinity policies to measure cache-hit rate and migration overhead.

Both notebooks pull from the [Google 2019 Cluster Sample (Borg traces)](https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample) on Kaggle.

## Data and preprocessing

The raw trace has 405,894 task-instance rows across 34 columns. After dropping failed jobs and rows where the end time doesn't exceed the start time, 313,216 rows remain for the comparative study (the affinity study uses the full 405,894). Burst time is derived from the CPU usage field, and since Borg doesn't record deadlines, EDF gets a synthetic deadline built from arrival time plus burst time plus a random offset. Everything runs with a fixed random seed (42) so the results reproduce.

To force genuine scheduling contention rather than jobs running back to back, inter-arrival gaps are compressed by a factor of 10^7 and burst times are scaled up by 10^4. In the 500-job sample this pushes job overlap to 99.8%.

## What the experiments found

On the 500-job comparison, no algorithm wins on every metric:

| Algorithm | Avg wait (×10³) | Avg turnaround (×10³) | Avg response | Context switches | Fairness |
|---|---|---|---|---|---|
| FCFS | 3839.6 | 3855.6 | 3839.6 | 0 | 0.744 |
| SJF | 2492.5 | 2508.5 | 2492.5 | 500 | 0.707 |
| Round Robin | 5018.8 | 5034.8 | 2.5 | 799,000 | 0.987 |
| Priority | 3366.7 | 3382.7 | 3366.7 | 500 | 0.668 |
| MLQ | 3655.0 | 3671.0 | 3489.3 | 112,397 | 0.728 |
| MLFQ | 3845.4 | 3861.4 | 0.013 | 1,768 | 0.745 |
| EDF | 3226.5 | 3242.5 | 3226.5 | 500 | 0.713 |
| CFS | 5101.4 | 5117.3 | 0.017 | 1,597,748 | 0.989 |

SJF comes out lowest on waiting and turnaround time, which matches the theory since it always knows the shortest job. MLFQ gets the best response time while keeping context switches reasonable, which makes it the strongest general-purpose option in this sample. CFS and Round Robin post the highest fairness scores, but they pay for it with a heavy volume of context switches.

The quantum sweep on Round Robin shows context switches dropping roughly 500-fold as the quantum goes from 1 to 500, while average waiting time barely moves (under 2% across the whole range). A single fixed quantum just can't serve short interactive jobs and long batch jobs equally well, which is part of why CFS and EEVDF moved to self-tuning slice lengths instead.

The affinity study shows soft affinity capturing most of the benefit of pinning tasks to a core (95-97% cache-hit rate) while still leaving the scheduler free to rebalance load, versus random placement's hit rate collapsing to 6.2% at 16 cores.

## Running the notebooks

Both notebooks are plain Jupyter notebooks built around pandas and numpy for the simulation logic and matplotlib for the plots. Download the Borg trace CSV from the Kaggle link above, point the notebook's data path at it, and run all cells top to bottom. No GPU or special hardware is needed since the simulations are lightweight discrete-event loops.
