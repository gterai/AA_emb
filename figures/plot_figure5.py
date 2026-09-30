#!/usr/bin/env python3
"""Plot the five-model TE benchmark using run-level mean correlations."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "figure_data/figure5/per_seed_summary.tsv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/figures"
DEFAULT_BASENAME = "FigModel"
N_RUNS = 10
N_TISSUES = 78

MODELS = [
    ("mrna_only", "mRNA-only", "#4daf4a"),
    ("ribonn", "RiboNN", "#7f7f7f"),
    ("5utrbert", "5UTRBERT", "#984ea3"),
    ("mrnabert", "mRNABERT", "#377eb8"),
    ("mrna_t5u", "mRNA+T5u", "#ff7f00"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the main Pearson comparison of five TE models."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", choices=["pdf", "png", "tif"], default=["pdf", "png", "tif"])
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and summarize the input without writing figure files.",
    )
    return parser.parse_args()


def read_values(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    expected_fields = {
        "model_key",
        "seed",
        "n_tissues",
        "mean_tissue_pearson",
        "mean_tissue_spearman",
    }
    missing_fields = expected_fields - set(rows[0] if rows else [])
    if missing_fields:
        raise ValueError(f"Missing columns in {path}: {sorted(missing_fields)}")

    expected_keys = [model_key for model_key, _, _ in MODELS]
    if len(rows) != len(expected_keys) * N_RUNS:
        raise ValueError(
            f"Expected {len(expected_keys) * N_RUNS} rows in {path}, found {len(rows)}"
        )

    values: dict[str, dict[str, np.ndarray]] = {}
    for model_key in expected_keys:
        selected = [row for row in rows if row["model_key"] == model_key]
        selected.sort(key=lambda row: int(row["seed"]))
        seeds = [int(row["seed"]) for row in selected]
        if seeds != list(range(N_RUNS)):
            raise ValueError(f"{model_key}: expected seeds 0--9, found {seeds}")
        if any(int(row["n_tissues"]) != N_TISSUES for row in selected):
            raise ValueError(f"{model_key}: a run does not contain {N_TISSUES} tissues")

        values[model_key] = {}
        for metric in ("pearson", "spearman"):
            array = np.asarray(
                [float(row[f"mean_tissue_{metric}"]) for row in selected],
                dtype=float,
            )
            if not np.isfinite(array).all():
                raise ValueError(f"{model_key}: non-finite {metric} value")
            values[model_key][metric] = array
    return values


def common_limits(values: dict[str, dict[str, np.ndarray]]) -> tuple[float, float]:
    arrays = [
        values[model_key][metric]
        for model_key, _, _ in MODELS
        for metric in ("pearson", "spearman")
    ]
    data_min = min(float(array.min()) for array in arrays)
    data_max = max(float(array.max()) for array in arrays)
    step = 0.05
    lower = math.floor((data_min - 0.015) / step) * step
    upper = math.ceil((data_max + 0.015) / step) * step
    return lower, upper


def plot_panel(
    ax: plt.Axes,
    values: dict[str, dict[str, np.ndarray]],
    metric: str,
    panel: str,
    limits: tuple[float, float],
) -> None:
    x_positions = np.arange(len(MODELS), dtype=float)
    jitter = np.linspace(-0.13, 0.13, N_RUNS)

    for index, (model_key, _, color) in enumerate(MODELS):
        array = values[model_key][metric]
        ax.scatter(
            x_positions[index] + jitter,
            array,
            s=20,
            facecolor=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.72,
            zorder=2,
        )
        ax.errorbar(
            x_positions[index],
            array.mean(),
            yerr=array.std(ddof=1),
            fmt="D",
            markersize=5.5,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.75,
            ecolor="black",
            elinewidth=1.0,
            capsize=3.5,
            capthick=1.0,
            zorder=3,
        )

    ax.set_xticks(x_positions, [label for _, label, _ in MODELS], rotation=28)
    ax.tick_params(axis="x", pad=2)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.set_ylabel(f"Mean {metric.capitalize()} correlation")
    ax.set_ylim(*limits)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.set_axisbelow(True)
    ax.text(
        -0.12,
        1.03,
        panel,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def print_summary(values: dict[str, dict[str, np.ndarray]]) -> None:
    for model_key, label, _ in MODELS:
        pearson = values[model_key]["pearson"]
        spearman = values[model_key]["spearman"]
        print(
            f"{label}: "
            f"Pearson={pearson.mean():.6f} +/- {pearson.std(ddof=1):.6f}; "
            f"Spearman={spearman.mean():.6f} +/- {spearman.std(ddof=1):.6f}"
        )


def main() -> None:
    args = parse_args()
    values = read_values(args.input)
    print(f"validation: PASS (5 models, {N_RUNS} runs, {N_TISSUES} tissues per run)")
    print_summary(values)
    if args.check_only:
        return

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    limits = common_limits(values)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = (
        # Bioinformatics single-column figure (approximately 3.3 inches wide).
        ("pearson", args.output_dir / args.basename, (3.35, 3.55)),
    )
    for metric, stem, size in outputs:
        fig, ax = plt.subplots(1, 1, figsize=size)
        plot_panel(ax, values, metric, "", limits)
        # A single-panel figure does not require an (a)/(b) panel label.
        ax.texts[-1].remove()
        if metric == "pearson":
            fig.subplots_adjust(left=0.20, right=0.98, top=0.97, bottom=0.28)
        else:
            fig.subplots_adjust(left=0.16, right=0.98, top=0.96, bottom=0.22)
        for suffix in args.formats:
            kwargs = {"bbox_inches": "tight"}
            if suffix == "png":
                kwargs["dpi"] = 600
            elif suffix == "tif":
                kwargs.update(dpi=1200, pil_kwargs={"compression": "tiff_lzw"})
            fig.savefig(stem.with_suffix(f".{suffix}"), **kwargs)
        plt.close(fig)



if __name__ == "__main__":
    main()
