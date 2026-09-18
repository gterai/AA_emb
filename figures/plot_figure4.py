#!/usr/bin/env python3
"""Create an integrated main-text figure for protein-embedding robustness analyses."""

from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "figure_data/figure4"
CONTROL_RESULTS = RESULTS
MAIN_OUTDIR = ROOT / "outputs/figures"
SUPPLEMENTARY_OUTDIR = MAIN_OUTDIR
FORMATS = ["pdf", "png", "svg"]
CONTROL_DIRS = {"protein_cluster_controls", "random_t5u_control", "shuffled_t5u_control"}

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7A5195"
GREY = "#777777"
LIGHT_GREY = "#D9D9D9"


def read_tsv(relative):
    base = CONTROL_RESULTS if Path(relative).parts[0] in CONTROL_DIRS else RESULTS
    return pd.read_csv(base / relative, sep="\t")


def panel_b(ax, metric):
    df = read_tsv("protein_cluster_baseline/per_seed_summary.tsv")
    df = df[df["condition"].isin(["mrna_only", "mrna_t5u"])]
    thresholds = ["c90", "c70", "c50"]
    labels = ["90%", "70%", "50%"]
    offsets = {"mrna_only": -0.13, "mrna_t5u": 0.13}
    colors = {"mrna_only": BLUE, "mrna_t5u": ORANGE}
    names = {"mrna_only": "mRNA-only", "mrna_t5u": "mRNA+T5u"}

    for condition in ["mrna_only", "mrna_t5u"]:
        for i, threshold in enumerate(thresholds):
            vals = df[(df.threshold == threshold) & (df.condition == condition)][f"mean_{metric}"].to_numpy()
            x = np.full(len(vals), i + offsets[condition])
            jitter = np.linspace(-0.035, 0.035, len(vals))
            ax.scatter(x + jitter, vals, s=13, facecolor="white", edgecolor=colors[condition],
                       linewidth=0.7, zorder=2)
            mean = vals.mean()
            sd = vals.std(ddof=1)
            ax.errorbar(i + offsets[condition], mean, yerr=sd, fmt="o", ms=5.5,
                        color=colors[condition], capsize=2.5, lw=1.2, zorder=3,
                        label=names[condition] if i == 0 else None)

    ax.set_title("Protein-similarity-aware evaluation", loc="left", fontsize=10.5, fontweight="bold")
    ax.text(-0.13, 1.06, "(a)", transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax.set_xticks(range(3), labels)
    ax.set_xlabel("Protein sequence identity threshold")
    ax.set_ylabel(f"Mean {metric.capitalize()} correlation")
    ax.set_ylim(0.63, 0.74)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.grid(axis="y", color=LIGHT_GREY, lw=0.6)


def panel_c(ax, metric):
    # Reference length results use train-only log1p scaling for each split/seed.
    controls = read_tsv("protein_cluster_controls/condition_summary.tsv")
    random = read_tsv("random_t5u_control/condition_summary.tsv")
    shuffled = read_tsv("shuffled_t5u_control/condition_summary.tsv")
    value_column = f"{metric}_mean"
    base = controls[(controls.threshold == "c50") & (controls.condition == "mrna_only")][value_column].iloc[0]
    native = controls[(controls.threshold == "c50") & (controls.condition == "mrna_t5u")][value_column].iloc[0]
    full_gain = native - base
    rows = [
        ("Length", controls, "mrna_length"),
        ("AA\ncomposition", controls, "mrna_aacom"),
        ("Dipeptide\ncomposition", controls, "mrna_dipep"),
        ("Random\nencoder", random, "mrna_random_t5u"),
        ("Shuffled\nsequence", shuffled, "mrna_shuffled_t5u"),
        ("Native\nT5u", controls, "mrna_t5u"),
    ]
    values = []
    for _, frame, condition in rows:
        value = frame[(frame.threshold == "c50") & (frame.condition == condition)][value_column].iloc[0]
        values.append(100 * (value - base) / full_gain)

    x = np.arange(len(rows))
    colors = [GREY, GREY, GREY, PURPLE, GREEN, ORANGE]
    bars = ax.bar(x, values, color=colors, width=0.68)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, max(value, 0) + 2.5, f"{value:.0f}%",
                ha="center", va="bottom", fontsize=7.3)
    ax.axhline(100, color=ORANGE, lw=0.8, ls="--")
    ax.set_title("Fraction of the native T5u gain recovered", loc="left", fontsize=10.5, fontweight="bold")
    ax.text(-0.13, 1.06, "(b)", transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax.set_xticks(x, [r[0].replace("\n", " ") for r in rows], fontsize=6.6,
                  rotation=28, ha="right", rotation_mode="anchor")
    ax.set_ylabel(f"Recovered {metric.capitalize()} gain (%)")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", color=LIGHT_GREY, lw=0.6)


def panel_d(ax, metric):
    df = read_tsv("go_slim_holdout/per_seed_summary_final.tsv")
    pivot = df.pivot_table(
        index=["go_slim_name", "seed"],
        columns="condition",
        values=f"mean_{metric}",
    ).reset_index()
    pivot["gain"] = pivot["mrna_t5u"] - pivot["mrna_only"]
    order = ["ribosome", "extracellular region", "RNA binding", "immune system process",
             "plasma membrane", "mitochondrion"]
    summary = pivot.groupby("go_slim_name")["gain"].agg(["mean", "std"]).loc[order]
    y = np.arange(len(order))[::-1]
    ax.errorbar(summary["mean"], y, xerr=summary["std"], fmt="o", color=ORANGE,
                ecolor=ORANGE, capsize=2.5, ms=5.5, lw=1.2)
    ax.axvline(0, color=GREY, lw=0.8)
    display = {"RNA binding": "RNA binding", "immune system process": "Immune-related"}
    ax.set_yticks(y, [display.get(s, s.capitalize()) for s in order], fontsize=7.2)
    ax.set_xlabel(f"{metric.capitalize()} gain: mRNA+T5u $-$ mRNA-only")
    ax.set_title("Generalization to held-out functional classes", loc="left", fontsize=10.5, fontweight="bold")
    ax.text(-0.13, 1.06, "(c)", transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax.set_xlim(0, 0.105)
    ax.grid(axis="x", color=LIGHT_GREY, lw=0.6)


def make_figure(metric):
    outdir = MAIN_OUTDIR if metric == "pearson" else SUPPLEMENTARY_OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 5.25), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.56, wspace=0.42,
                          left=0.09, right=0.98, top=0.94, bottom=0.11)
    panel_b(fig.add_subplot(gs[0, 0]), metric)
    panel_c(fig.add_subplot(gs[0, 1]), metric)
    panel_d(fig.add_subplot(gs[1, :]), metric)

    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    stem = "FigRob" if metric == "pearson" else f"SFigRob_{metric}"
    for suffix in FORMATS:
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 600
        elif suffix == "tif":
            kwargs.update(dpi=1200, pil_kwargs={"compression": "tiff_lzw"})
        fig.savefig(outdir / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def main():
    global RESULTS, CONTROL_RESULTS, MAIN_OUTDIR, SUPPLEMENTARY_OUTDIR, FORMATS
    parser = argparse.ArgumentParser(description="Recreate the three-panel FigRob and SFigRob_spearman using the manuscript plotting layout.")
    parser.add_argument("--results", type=Path, default=ROOT / "figure_data/figure4",
                        help="Summary root for protein-cluster and functional holdout panels; defaults to reference results")
    parser.add_argument("--controls-results", type=Path,
                        help="Summary root for the control panel; defaults to --results. Specify explicitly when using separate reference controls.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/figures")
    parser.add_argument("--formats", nargs="+", choices=["pdf", "png", "svg", "eps", "tif"], default=["pdf", "png", "svg"])
    args = parser.parse_args()
    RESULTS = args.results
    CONTROL_RESULTS = args.controls_results or RESULTS
    MAIN_OUTDIR = SUPPLEMENTARY_OUTDIR = args.output
    FORMATS = args.formats
    required = ["protein_cluster_baseline/per_seed_summary.tsv", "go_slim_holdout/per_seed_summary_final.tsv"]
    for relative in required:
        if not (RESULTS / relative).is_file():
            parser.error(f"Missing figure input: {RESULTS / relative}")
    for directory in sorted(CONTROL_DIRS):
        if not (CONTROL_RESULTS / directory / "condition_summary.tsv").is_file():
            parser.error(f"Missing control summary: {CONTROL_RESULTS / directory / 'condition_summary.tsv'}. Use --controls-results for a separate control-summary root, or training/plot_overview.py for a two-benchmark overview.")
    MAIN_OUTDIR.mkdir(parents=True, exist_ok=True)
    SUPPLEMENTARY_OUTDIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    make_figure("pearson")
    make_figure("spearman")


if __name__ == "__main__":
    main()
