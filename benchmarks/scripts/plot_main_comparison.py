#!/usr/bin/env python3
"""Rebuild the primary benchmark figure for Pearson or Spearman correlation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "reference_metrics/main_comparison"
RESULTS = ROOT / "results/main_comparison"
MAIN = ROOT / "results/figures"
SUPPLEMENTARY = MAIN
FORMATS = ["pdf", "png", "svg"]

TISSUES = [
    ("TE_HEK293T", "HEK293T"),
    ("TE_HeLa", "HeLa"),
    ("TE_HepG2", "HepG2"),
    ("TE_muscle_tissue", "muscle tissue"),
]

CONDITIONS = [
    ("mrna_only", "mRNA", "#2f7d1e"),
    ("t5b_only", "T5b", "#add3df"),
    ("t5u_only", "T5u", "#add3df"),
    ("ank_only", "ank", "#add3df"),
    ("ank3_only", "ank3", "#add3df"),
    ("esm2_only", "esm2", "#add3df"),
    ("esm2l_only", "esm2L", "#add3df"),
    ("mrna_t5b", "mRNA+T5b", "#f5aa35"),
    ("mrna_t5u", "mRNA+T5u", "#f5aa35"),
    ("mrna_ank", "mRNA+ank", "#f5aa35"),
    ("mrna_ank3", "mRNA+ank3", "#f5aa35"),
    ("mrna_esm2", "mRNA+esm2", "#f5aa35"),
    ("mrna_esm2l", "mRNA+esm2L", "#f5aa35"),
    ("mrna_aacom", "mRNA+aacomp", "#ebb3be"),
    ("mrna_dipep", "mRNA+dipep", "#ebb3be"),
]

COMPARISONS = [
    ("mrna_ank", "mrna_ank3"),
    ("mrna_esm2", "mrna_esm2l"),
]


def read_metric(condition: str, seed: int, tissue: str, metric: str) -> float:
    path = RUNS / condition / f"seed_{seed}" / "metrics.tsv"
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["tissue"] == tissue:
                return float(row[metric])
    raise ValueError(f"Tissue {tissue!r} not found in {path}")


def significance(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def add_bracket(ax, left: int, right: int, y: float, height: float, label: str) -> None:
    ax.plot([left, left, right, right], [y, y + height, y + height, y],
            color="black", linewidth=0.65, clip_on=False)
    ax.text((left + right) / 2, y + height * 1.25, label,
            ha="center", va="bottom", fontsize=8.5, fontweight="bold")


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def make_figure(metric: str) -> None:
    values = {
        (tissue, condition): np.asarray(
            [read_metric(condition, seed, tissue, metric) for seed in range(10)]
        )
        for tissue, _ in TISSUES
        for condition, _, _ in CONDITIONS
    }

    tests = []
    for tissue, tissue_label in TISSUES:
        for left, right in COMPARISONS:
            raw_p = float(ttest_rel(values[(tissue, left)], values[(tissue, right)]).pvalue)
            tests.append({
                "tissue": tissue,
                "tissue_label": tissue_label,
                "metric": metric,
                "condition_1": left,
                "condition_2": right,
                "mean_1": values[(tissue, left)].mean(),
                "mean_2": values[(tissue, right)].mean(),
                "mean_difference": values[(tissue, right)].mean() - values[(tissue, left)].mean(),
                "p_raw": raw_p,
            })
    for row in tests:
        row["significance"] = significance(row["p_raw"])

    summaries = []
    for tissue, tissue_label in TISSUES:
        for condition, display, _ in CONDITIONS:
            series = values[(tissue, condition)]
            summaries.append({
                "tissue": tissue,
                "tissue_label": tissue_label,
                "condition": condition,
                "display_label": display,
                "metric": metric,
                "n_seeds": len(series),
                "mean": series.mean(),
                "standard_deviation": series.std(ddof=1),
            })

    write_tsv(
        RESULTS / f"FigMain_{metric}_summary.tsv",
        summaries,
        ["tissue", "tissue_label", "condition", "display_label", "metric",
         "n_seeds", "mean", "standard_deviation"],
    )
    write_tsv(
        RESULTS / f"FigMain_{metric}_paired_ttests.tsv",
        tests,
        ["tissue", "tissue_label", "metric", "condition_1", "condition_2",
         "mean_1", "mean_2", "mean_difference", "p_raw", "significance"],
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.35))
    x = np.arange(len(CONDITIONS))
    panels = ["(a)", "(b)", "(c)", "(d)"]
    for ax, (tissue, tissue_label), panel in zip(axes.flat, TISSUES, panels):
        means = np.asarray([values[(tissue, c)].mean() for c, _, _ in CONDITIONS])
        standard_deviations = np.asarray([values[(tissue, c)].std(ddof=1) for c, _, _ in CONDITIONS])
        ax.bar(
            x, means, yerr=standard_deviations,
            color=[color for _, _, color in CONDITIONS], edgecolor="#333333",
            linewidth=0.55, width=0.74, capsize=2.4,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )
        ax.set_xticks(x, [label for _, label, _ in CONDITIONS], rotation=90)
        ax.set_ylabel(f"{metric.capitalize()} correlation")
        ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.45)
        ax.set_axisbelow(True)
        lower = min(means - standard_deviations)
        upper = max(means + standard_deviations)
        span = max(upper - lower, 0.04)
        # Use an identical scale in all panels so that differences between
        # tissues can be compared without visual distortion.
        ax.set_ylim(0.30, 0.85)
        ax.text(0.02, 0.95, tissue_label, transform=ax.transAxes,
                ha="left", va="top", fontsize=11.5)
        ax.text(-0.11, 1.08, panel, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=11.5)

        tissue_tests = [row for row in tests if row["tissue"] == tissue]
        bracket_y = upper + 0.055 * span
        bracket_height = 0.035 * span
        index = {condition: i for i, (condition, _, _) in enumerate(CONDITIONS)}
        for comparison_number, row in enumerate(tissue_tests):
            add_bracket(
                ax,
                index[row["condition_1"]],
                index[row["condition_2"]],
                bracket_y + comparison_number * 0.005 * span,
                bracket_height,
                row["significance"],
            )

    fig.subplots_adjust(left=0.085, right=0.99, top=0.96, bottom=0.17,
                        wspace=0.28, hspace=0.55)
    outdir = MAIN if metric == "pearson" else SUPPLEMENTARY
    stem = "FigMain" if metric == "pearson" else "SFigMain_spearman"
    outdir.mkdir(parents=True, exist_ok=True)
    for suffix in FORMATS:
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 600
        elif suffix == "tif":
            kwargs.update(dpi=1200, pil_kwargs={"compression": "tiff_lzw"})
        fig.savefig(outdir / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def validate_metrics():
    """Reject missing, duplicate or non-finite plotting inputs before writing files."""
    for condition, _, _ in CONDITIONS:
        for seed in range(10):
            path = RUNS / condition / f"seed_{seed}" / "metrics.tsv"
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            tissues = [row["tissue"] for row in rows]
            if len(rows) != 78 or len(set(tissues)) != 78:
                raise ValueError(f"Expected 78 unique tissue rows in {path}")
            for tissue, _ in TISSUES:
                row = next((r for r in rows if r["tissue"] == tissue), None)
                if row is None or not all(np.isfinite(float(row[m])) for m in ["pearson", "spearman"]):
                    raise ValueError(f"Missing/non-finite figure metric for {tissue} in {path}")


def main() -> None:
    global RUNS, RESULTS, MAIN, SUPPLEMENTARY, FORMATS
    parser = argparse.ArgumentParser(description="Recreate FigMain and SFigMain_spearman from per-tissue reference metrics.")
    parser.add_argument("--metrics", type=Path, default=ROOT / "reference_metrics/main_comparison",
                        help="Root containing condition/seed_N/metrics.tsv files")
    parser.add_argument("--results", type=Path, default=ROOT / "results/main_comparison",
                        help="Directory for generated mean/SD and paired-test tables")
    parser.add_argument("--output", type=Path, default=ROOT / "results/figures")
    parser.add_argument("--formats", nargs="+", choices=["pdf", "png", "svg", "eps", "tif"], default=["pdf", "png", "svg"])
    parser.add_argument("--metric", choices=("pearson", "spearman", "both"), default="both")
    args = parser.parse_args()
    RUNS, RESULTS = args.metrics, args.results
    MAIN = SUPPLEMENTARY = args.output
    FORMATS = args.formats
    validate_metrics()
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7.5,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    metrics = ("pearson", "spearman") if args.metric == "both" else (args.metric,)
    for metric in metrics:
        make_figure(metric)


if __name__ == "__main__":
    main()
