"""Paper Fig. 1: the study's pipeline, one column wide.

    python methodology_diagram.py   -> methodology.pdf / methodology.png

Style and palette follow figures4papers (https://github.com/ChenLiu-1996/figures4papers):
Liberation Sans (metric-compatible with the repo's Arial), blue for the data path, green for simulation, red for the kernel, neutral for metrics.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
PALETTE = {"blue_main": "#0F4D92", "green_1": "#DDF3DE", "green_3": "#8BCF8B", "red_1": "#F6CFCB",
           "red_strong": "#B64342", "neutral": "#CFCECE", "ink": "#272727"}
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"],
                     "font.size": 8, "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
                     "mathtext.it": "Liberation Sans:italic", "pdf.fonttype": 42, "ps.fonttype": 42})

# One IEEE column (3.5 in = 252 pt), saved without a tight bbox, so 8 pt text prints at 8 pt.
LINE = 0.145                               # height of one 8 pt text line (linespacing 1.3), inches
GAP = 0.14                                 # vertical gap between rows (holds the arrows)


def rows_height(n_lines):
    """Box height for a bold title plus n body lines."""
    return 0.11 + (n_lines + 1) * LINE


h1, h2, h3, h4 = rows_height(3), rows_height(2), rows_height(4), 3 * LINE + 0.1
W = 3.5
H = 0.03 + h4 + GAP + h3 + GAP + h2 + GAP + h1 + 0.03
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(0, H))
ax.set_axis_off()


def box(x, y, w, h, title, body, fill, edge):
    """Rounded box with a bold title line and body lines; (x, y) is the lower-left corner."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor=fill, edgecolor=edge, linewidth=0.8))
    ax.text(x + w / 2, y + h - 0.06, title, ha="center", va="top", fontweight="bold", color=edge)
    ax.text(x + w / 2, y + h - 0.06 - LINE - 0.02, body, ha="center", va="top", color=PALETTE["ink"],
            linespacing=1.3)


def arrow(p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, linewidth=0.7,
                                 color=PALETTE["ink"], shrinkA=0, shrinkB=0))


blue, white = PALETTE["blue_main"], "white"
my = 0.03                                  # row 4: metrics and comparison
by = my + h4 + GAP                         # row 3: simulator and kernel branches
wy = by + h3 + GAP                         # row 2: workload model
top = wy + h2 + GAP                        # row 1: trace -> filter -> job model

box(0.03, top, 1.06, h1, "Borg 2019 trace", "public sample\n405,894 rows\n(usage windows)", white, blue)
box(1.22, top, 1.06, h1, "Filter", "failed = 0,\nend > start\n313,216 rows", white, blue)
box(2.41, top, 1.06, h1, "Job model", r"$b=\lfloor 100u\rfloor+1$" "\nu: CPU in NCUs\nBorg priority", white, blue)
arrow((1.09, top + h1 / 2), (1.22, top + h1 / 2))
arrow((2.28, top + h1 / 2), (2.41, top + h1 / 2))

box(0.03, wy, 3.44, h2, "Workload model (seed 42)",
    r"500 jobs; arrivals $\div$ 10,000,000, bursts $\times$ 10,000 $\rightarrow$ load $\rho=30$" "\n"
    r"synthetic deadlines $D=(1+s)\,b$, $s\sim U(0.5,2)$", PALETTE["neutral"], PALETTE["ink"])
arrow((2.94, top), (2.94, wy + h2))

box(0.03, by, 1.66, h3, "Simulator (one core)",
    "FCFS, SJF, RR, Priority,\nMLQ, MLFQ, EDF, CFS\n" r"load sweep $\rho=0.5$–30" "\n"
    r"RR quantum $q$ = 1 to 1M", PALETTE["green_1"], "#2E7D32")
box(1.81, by, 1.66, h3, "Kernel Baseline (Linux 7.0)",
    "SCHED_FIFO, SCHED_RR,\nSCHED_DEADLINE, EEVDF\n(2 slices); 3 replicates\n"
    "1 unit = 100 µs of CPU", PALETTE["red_1"], PALETTE["red_strong"])
arrow((0.86, wy), (0.86, by + h3))
arrow((2.64, wy), (2.64, by + h3))

ax.add_patch(FancyBboxPatch((0.03, my), 3.44, h4, boxstyle="round,pad=0,rounding_size=0.05",
                            facecolor=white, edgecolor=PALETTE["ink"], linewidth=0.8))
ax.text(W / 2, my + h4 / 2,
        "Waiting, turnaround and response time, context switches,\n"
        "Jain (waiting, slowdown), deadline miss ratio;\n"
        r"simulator vs. kernel by rank agreement (Kendall $\tau$)",
        ha="center", va="center", color=PALETTE["ink"], linespacing=1.3)
arrow((0.86, by), (0.86, my + h4))
arrow((2.64, by), (2.64, my + h4))

fig.savefig(HERE / "methodology.pdf")
fig.savefig(HERE / "methodology.png", dpi=300)
print("Figure: methodology.pdf/.png")
