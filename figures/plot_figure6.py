#!/usr/bin/env python3
"""Create the four-panel publication figure for Eraslan PTR validation."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import json
from ptr_data import ROOT, load_inputs, calculate
import numpy as np
from scipy.stats import pearsonr, spearmanr


BLUE = "#2B6CB0"
ORANGE = "#D97706"
GREEN = "#2F855A"
GRAY = "#5B6573"
LIGHT = "#E6EAF0"


def read_tsv(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def panel_label(ax, label: str):
    ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontsize=19,
            fontweight="bold", va="top", ha="right")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-ev3", type=Path, required=True, help="User-downloaded Table EV3 ZIP or extracted Table_EV3.tsv")
    parser.add_argument("--predictions", type=Path, default=ROOT / "figure_data/figure6/predictions")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs/figure6")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "outputs/figures/FigPTR")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--formats", nargs="+", choices=["pdf", "png", "tif"], default=["pdf", "png", "tif"])
    args = parser.parse_args()
    data, tissue_names, metadata = load_inputs(args.table_ev3, args.predictions)
    primary, tissues = calculate(data, tissue_names, args.results_dir, args.bootstrap, args.seed)
    ids, obs, pred0, pred1 = data["c50"]
    ptr = np.nanmedian(obs, axis=1)
    metadata.update(bootstrap=args.bootstrap, seed=args.seed, thresholds=["c50","c70","c90"])
    (args.results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11.5,
        "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
        "axes.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(14.2, 11.4), constrained_layout=False)
    outer = fig.add_gridspec(2, 2, width_ratios=[0.92, 1.35],
                             height_ratios=[0.88, 1.25], wspace=0.26, hspace=0.32)

    # A: workflow
    ax = fig.add_subplot(outer[0, 0]); ax.set_axis_off(); panel_label(ax, "(a)")
    ax.set_title("External PTR validation workflow", loc="left", fontweight="bold", pad=8)
    boxes = [
        (0.50, 0.85, "Eraslan et al. (2019)\n11,575 transcripts × 29 tissues"),
        (0.50, 0.62, "Match to our 9,926 transcripts by Ensembl ID\n5,255 matched transcripts"),
        (0.50, 0.39, "Held-out test predictions for each random seed\nc50, 10 random seeds"),
        (0.50, 0.16, "OOF ensemble\n4,687 transcripts"),
    ]
    for i, (x, y, text) in enumerate(boxes):
        ax.text(x, y, text, ha="center", va="center", transform=ax.transAxes,
                fontsize=11.2, linespacing=1.25,
                bbox=dict(boxstyle="round,pad=0.48", fc="white", ec=GRAY, lw=1.0))
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x, boxes[i + 1][1] + 0.075), xytext=(x, y - 0.075),
                        xycoords=ax.transAxes, arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.1))
    ax.text(0.07, 0.02, "PTR: median log$_{10}$ PTR across tissues", transform=ax.transAxes,
            fontsize=10.2, color=GRAY)
    ax.text(0.07, -0.04, "Prediction: median TE across 78 output heads", transform=ax.transAxes,
            fontsize=10.2, color=GRAY)

    # B: transcript-level density plots with a shared logarithmic count scale
    inner = outer[0, 1].subgridspec(1, 3, width_ratios=[1, 1, 0.07], wspace=0.23)
    density_axes = [fig.add_subplot(inner[0, 0]), fig.add_subplot(inner[0, 1])]
    density_cax = fig.add_subplot(inner[0, 2])
    xpad = 0.04 * (ptr.max() - ptr.min())
    ymin = min(pred0.min(), pred1.min()); ymax = max(pred0.max(), pred1.max())
    ypad = 0.04 * (ymax - ymin)
    gridsize = 40
    density_counts = []
    for pred in (pred0, pred1):
        temporary_hexbin = density_axes[0].hexbin(ptr, pred, gridsize=gridsize, mincnt=1)
        density_counts.extend(temporary_hexbin.get_array().tolist())
        temporary_hexbin.remove()
    density_norm = LogNorm(vmin=1, vmax=max(density_counts))

    density_artist = None
    for j, (sax, pred, title, col) in enumerate([
        (density_axes[0], pred0, "mRNA-only", "#8B0000"),
        (density_axes[1], pred1, "mRNA+T5u", "#8B0000"),
    ]):
        density_artist = sax.hexbin(
            ptr, pred, gridsize=gridsize, mincnt=1, cmap="YlOrRd",
            norm=density_norm, linewidths=0.25, edgecolors="white",
            rasterized=True,
        )
        fit = np.polyfit(ptr, pred, 1); xx = np.linspace(ptr.min(), ptr.max(), 100)
        sax.plot(xx, np.polyval(fit, xx), color=col, lw=1.8)
        r = pearsonr(ptr, pred).statistic
        rho = spearmanr(ptr, pred).statistic
        sax.set_title(title, color=col, fontweight="bold")
        sax.set_xlabel("Median log$_{10}$ PTR")
        if j == 0: sax.set_ylabel("Median predicted TE")
        else: sax.set_yticklabels([])
        sax.text(0.04, 0.95, f"Pearson r = {r:.3f}\nSpearman ρ = {rho:.3f}\nn = 4,687",
                 transform=sax.transAxes, va="top", ha="left", fontsize=10.5,
                 bbox=dict(fc="white", ec="none", alpha=0.86, pad=2.5))
        sax.grid(color=LIGHT, lw=0.6, zorder=0)
        sax.set_xlim(ptr.min() - xpad, ptr.max() + xpad)
        sax.set_ylim(ymin - ypad, ymax + ypad)
    density_cbar = fig.colorbar(density_artist, cax=density_cax)
    density_cbar.set_label("Transcripts per hexagon (log scale)")
    panel_label(density_axes[0], "(b)")

    # C: forest plot of Spearman differences
    ax = fig.add_subplot(outer[1, 0]); panel_label(ax, "(c)")
    ax.set_title("T5u-associated improvement\nacross split thresholds",
                 loc="left", fontweight="bold", linespacing=1.05)
    order = ["c90", "c70", "c50"]
    by = {r["threshold"]: r for r in primary}
    for y, threshold in enumerate(order):
        row = by[threshold]; est = float(row["spearman_difference"])
        lo = float(row["spearman_ci95_low"]); hi = float(row["spearman_ci95_high"])
        ax.plot([lo, hi], [y, y], color=GRAY, lw=2.0, solid_capstyle="round")
        ax.scatter(est, y, s=58, color=GREEN, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(est, y - 0.18, f"{est:.3f} [{lo:.3f}, {hi:.3f}]",
                va="top", ha="center", fontsize=10.5)
    ax.axvline(0, color="black", lw=0.9, ls="--")
    ax.set_yticks(range(3), order); ax.set_ylim(-0.65, 2.65); ax.set_xlim(-0.004, 0.078)
    ax.set_xlabel("Δ Spearman ρ (mRNA+T5u − mRNA-only)")
    ax.grid(axis="x", color=LIGHT, lw=0.7)
    ax.text(0.02, 0.04, f"95% CI: {args.bootstrap:,} cluster-bootstrap replicates",
            transform=ax.transAxes, fontsize=10.2, color=GRAY)

    # D: tissue-level differences for primary c50 analysis
    ax = fig.add_subplot(outer[1, 1]); panel_label(ax, "(d)")
    ax.set_title("Improvement across 29 human tissues (c50)", loc="left", fontweight="bold")
    selected = [r for r in tissues if r["threshold"] == "c50"]
    selected.sort(key=lambda r: float(r["spearman_difference"]))
    names = [r["tissue"] for r in selected]
    values = np.array([float(r["spearman_difference"]) for r in selected])
    yy = np.arange(len(selected))
    ax.hlines(yy, 0, values, color="#AAB2BD", lw=1.2)
    ax.scatter(values, yy, s=28, color=ORANGE, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(0, color="black", lw=0.9, ls="--")
    ax.set_yticks(yy, names); ax.set_ylim(-0.8, len(names) - 0.2)
    ax.set_xlabel("Δ Spearman ρ (mRNA+T5u − mRNA-only)")
    ax.grid(axis="x", color=LIGHT, lw=0.6)
    ax.text(0.98, 0.02, f"{int((values > 0).sum())}/{len(values)} tissues > 0", transform=ax.transAxes,
            ha="right", va="bottom", fontweight="bold", color=ORANGE)

    fig.subplots_adjust(left=0.08, right=0.985, top=0.975, bottom=0.075)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in args.formats:
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 300
        elif suffix == "tif":
            kwargs.update(dpi=1200, pil_kwargs={"compression": "tiff_lzw"})
        fig.savefig(args.output_prefix.with_suffix("." + suffix), **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
