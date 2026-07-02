"""Plot the degradation benchmark: identified clusters vs separation / noise.

Reads `benchmark_results/summary.csv` (built by `benchmark.py`) and draws a
2x2 grid of line charts — rows: best distance-based solution / best LCA;
columns: the separation sweep (x reversed, easy -> no structure) and the noise
sweep. One line per clustering validity index; y is the number of identified
clusters **excluding singletons**; a reference line marks the true k = 4.

With several seeds, lines show the mean across seeds and translucent dots the
individual seed values.

Usage:
    python plot_benchmark.py [--summary benchmark_results/summary.csv]
                             [--out benchmark_results/benchmark_clusters.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

# Validated categorical palette (dataviz reference, light mode) — fixed order.
SERIES = {
    "silhouette": ("Silhouette", "#2a78d6"),
    "calinski_harabasz": ("Calinski-Harabasz", "#1baf7a"),
    "davies_bouldin": ("Davies-Bouldin", "#eda100"),
    "dunn": ("Dunn (GD43)", "#008300"),
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Fixed y scale so one extreme run cannot distort the whole grid; values
# above Y_MAX are clipped at the axis edge.
Y_MAX = 20

POOLS = [("distance", "Best distance-based solution"), ("LCA", "Best LCA solution")]
SWEEPS = [
    ("separation", "class_sep", "Class separation (easy → no structure)", True),
    ("noise", "flip_y", "Label noise (flip_y)", False),
]
TRUE_K = 4


def draw_panel(ax, sub, xcol, reverse_x):
    """One panel: n effective clusters vs `xcol`, one line per CVI."""
    multi_seed = sub["seed"].nunique() > 1
    for col, (label, color) in SERIES.items():
        s = sub[sub["index"] == col].dropna(subset=["n_clust_effective"])
        if s.empty:
            continue
        mean = s.groupby(xcol)["n_clust_effective"].mean().sort_index()
        ax.plot(
            mean.index, mean.values.clip(max=Y_MAX),
            color=color, lw=2, solid_capstyle="round", solid_joinstyle="round",
            marker="o", ms=8, mec=SURFACE, mew=2, label=label,
        )
        if multi_seed:
            ax.scatter(
                s[xcol], s["n_clust_effective"].clip(upper=Y_MAX),
                color=color, s=14, alpha=0.35, linewidths=0, zorder=1,
            )

    ax.axhline(TRUE_K, color=AXIS, lw=1, zorder=0)
    if reverse_x:
        ax.invert_xaxis()

    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_ylim(0, Y_MAX)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_facecolor(SURFACE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary", type=Path, default=Path("benchmark_results/summary.csv")
    )
    ap.add_argument(
        "--out", type=Path, default=Path("benchmark_results/benchmark_clusters.png")
    )
    args = ap.parse_args()

    df = pd.read_csv(args.summary)

    fig, axes = plt.subplots(
        len(POOLS), len(SWEEPS), figsize=(10, 7), sharey="row", facecolor=SURFACE
    )
    for i, (pool, pool_label) in enumerate(POOLS):
        for j, (sweep, xcol, xlabel, reverse_x) in enumerate(SWEEPS):
            ax = axes[i, j]
            sub = df[(df["pool"] == pool) & (df["sweep"] == sweep)]
            draw_panel(ax, sub, xcol, reverse_x)
            if i == 0:
                ax.set_title(
                    "Separation sweep (no noise)" if sweep == "separation"
                    else f"Noise sweep (class_sep = {sub['class_sep'].iloc[0] if len(sub) else '—'})",
                    fontsize=10, color=INK,
                )
            if i == len(POOLS) - 1:
                ax.set_xlabel(xlabel, fontsize=9, color=INK_2)
            if j == 0:
                ax.set_ylabel(
                    f"{pool_label}\nclusters identified (≥ 2 members)",
                    fontsize=9, color=INK_2,
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=AXIS, lw=1))
    labels.append(f"true k = {TRUE_K}")
    fig.legend(
        handles, labels, loc="lower center", ncol=len(handles), frameon=False,
        fontsize=9, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Number of clusters identified by the best solution per validity index",
        fontsize=12, color=INK, y=0.99,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
