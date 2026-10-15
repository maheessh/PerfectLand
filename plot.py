"""Draw a training run's learning curve from its logs.

    python plot.py               # writes runs/main/learning_curve.png
    python plot.py --run NAME    # another run (also works while it is still training)
"""

import argparse
import csv
import json
import math

import matplotlib

matplotlib.use("Agg")  # draw straight to a file, no window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator

from lander import RUNS

WINDOW = 100  # attempts per average, and per column

# Outcomes stacked bottom to top. The colors are categorical slots checked for
# color-blind safety in this order, so keep the order if you change them.
GROUPS = [
    ("landed on pad", ("landed on pad",), "#2a78d6"),
    ("landed off pad", ("landed off pad",), "#eb6834"),
    ("timed out", ("timed out",), "#1baf7a"),
    ("crashed / flew off", ("crashed", "flew off screen"), "#eda100"),
]
SURFACE, INK, INK_SOFT, MUTED, GRID, BASELINE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
)


def load_attempts(run):
    """Every finished episode of a run, in the order they ended."""
    episodes = []
    for path in (RUNS / run / "monitor").glob("*.monitor.csv"):
        with open(path) as f:
            t_start = json.loads(f.readline()[1:])["t_start"]
            for row in csv.DictReader(f):
                episodes.append((t_start + float(row["t"]), float(row["r"]), int(row["l"]), row["outcome"]))
    episodes.sort()
    scores = np.array([e[1] for e in episodes])
    frames = sum(e[2] for e in episodes)
    outcomes = [e[3] for e in episodes]
    return scores, frames, outcomes


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0, labelsize=9, pad=6)


def plot_run(run):
    scores, frames, outcomes = load_attempts(run)
    n = len(scores)
    if n < WINDOW:
        print(f"Only {n} attempts logged in {RUNS / run}/; need {WINDOW} to draw a learning curve.")
        return

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 7.6), dpi=150, sharex=True, facecolor=SURFACE,
        gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.55},
    )
    fig.subplots_adjust(left=0.07, right=0.86, top=0.86, bottom=0.08)
    fig.text(0.07, 0.955, "Learning to land", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.07, 0.922, f"Run '{run}': {n:,} attempts, {frames:,} frames of practice",
             fontsize=10, color=INK_SOFT)
    blue = GROUPS[0][2]

    # Top: official score, averaged over the last 100 attempts.
    windows = np.lib.stride_tricks.sliding_window_view(scores, WINDOW)
    x = np.arange(WINDOW, n + 1)
    average = windows.mean(axis=1)
    low, high = np.percentile(windows, [10, 90], axis=1)
    style(top)
    top.fill_between(x, low, high, color=blue, alpha=0.12, linewidth=0)
    top.plot(x, average, color=blue, linewidth=2, solid_capstyle="round", solid_joinstyle="round")
    top.axhline(200, color=INK_SOFT, linewidth=1)
    top.text(WINDOW, 204, "200 = solved", color=INK_SOFT, fontsize=9, va="bottom")
    top.scatter([x[-1]], [average[-1]], s=40, color=blue, edgecolors=SURFACE, linewidths=1, zorder=3)
    top.annotate(f"{average[-1]:.0f}".replace("-", "−"), (x[-1], average[-1]), xytext=(9, 0), textcoords="offset points",
                 va="center", color=INK, fontsize=11, fontweight="bold", annotation_clip=False)
    top.set_title(
        "Official score per attempt: average of the last 100 (line) and the middle 80% (band)",
        loc="left", fontsize=10.5, color=INK, pad=10,
    )

    # Bottom: how each block of attempts ended.
    block = WINDOW * max(1, math.ceil(n / (WINDOW * 50)))  # at most ~50 columns
    blocks = n // block
    style(bottom)
    bottom.set_xlim(0, n)
    centers = (np.arange(blocks) + 0.5) * block
    px_per_attempt = bottom.get_window_extent().width / n
    width = min(0.7 * block, 24 / px_per_attempt)  # columns at most 24 px wide
    stacked = np.zeros(blocks)
    last_column = []
    for label, members, color in GROUPS:
        share = np.array([
            sum(o in members for o in outcomes[b * block:(b + 1) * block]) for b in range(blocks)
        ]) * 100 / block
        bottom.bar(centers, share, bottom=stacked, width=width, color=color,
                   edgecolor=SURFACE, linewidth=1, label=label)
        if blocks and share[-1] >= 6:
            last_column.append((stacked[-1] + share[-1] / 2, f"{label.split(' (')[0]} {share[-1]:.0f}%"))
        stacked += share
    for y, text in last_column:
        bottom.annotate(text, (n, y), xytext=(6, 0), textcoords="offset points",
                        va="center", color=INK_SOFT, fontsize=8.5, annotation_clip=False)
    bottom.set_ylim(0, 100)
    bottom.yaxis.set_major_locator(MultipleLocator(25))
    bottom.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    bottom.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    bottom.set_xlabel("landing attempts", color=INK_SOFT, fontsize=9.5, labelpad=8)
    bottom.set_title(f"How each block of {block:,} attempts ended",
                     loc="left", fontsize=10.5, color=INK, pad=30)
    bottom.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False, fontsize=9,
                  labelcolor=INK_SOFT, handlelength=0.8, handleheight=0.8, borderaxespad=0.4,
                  columnspacing=1.6, handletextpad=0.5)

    path = RUNS / run / "learning_curve.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="main", help="folder under runs/ (default: main)")
    args = parser.parse_args()
    path = plot_run(args.run)
    if path:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
