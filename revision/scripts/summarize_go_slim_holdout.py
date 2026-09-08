#!/usr/bin/env python3
"""Summarize matched mRNA-only and mRNA+T5u GO-slim holdout results."""

import argparse
import csv
import math
import statistics
from pathlib import Path

from scipy.stats import ttest_rel


CATEGORIES = {
    "GO_0005840": "ribosome",
    "GO_0005886": "plasma membrane",
    "GO_0005576": "extracellular region",
    "GO_0005739": "mitochondrion",
    "GO_0002376": "immune system process",
    "GO_0003723": "RNA binding",
}
CONDITIONS = ("mrna_only", "mrna_t5u")
METRICS = ("pearson", "spearman")


def read_metrics(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 78:
        raise ValueError(f"Expected 78 tissues in {path}, found {len(rows)}")
    return {
        row["tissue"]: {
            "pearson": float(row["pearson"]),
            "spearman": float(row["spearman"]),
            "n_test": int(row["n_test"]),
        }
        for row in rows
    }


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(pvalues) - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def paired_t_pvalue(alternative: list[float], reference: list[float]) -> float:
    return float(ttest_rel(alternative, reference).pvalue)


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    category_seeds = {}
    for category in CATEGORIES:
        complete = [
            seed for seed in args.seeds
            if all(
                (args.run_root / condition / category / f"seed_{seed}" / "metrics.tsv").is_file()
                for condition in CONDITIONS
            )
        ]
        if not args.allow_incomplete and complete != args.seeds:
            missing = sorted(set(args.seeds) - set(complete))
            raise FileNotFoundError(f"{category}: incomplete matched seeds {missing}")
        if len(complete) < 2:
            if args.allow_incomplete:
                continue
            raise ValueError(f"{category}: fewer than two matched seeds")
        category_seeds[category] = complete

    if not category_seeds:
        raise ValueError("No category has at least two matched completed seeds")

    data = {}
    per_seed = []
    for category, seeds in category_seeds.items():
        for seed in seeds:
            for condition in CONDITIONS:
                path = args.run_root / condition / category / f"seed_{seed}" / "metrics.tsv"
                values = read_metrics(path)
                data[(category, seed, condition)] = values
                row = {
                    "go_slim_id": category.replace("_", ":", 1),
                    "go_slim_name": CATEGORIES[category],
                    "seed": seed,
                    "condition": condition,
                }
                for metric in METRICS:
                    finite = [item[metric] for item in values.values() if math.isfinite(item[metric])]
                    row[f"mean_{metric}"] = statistics.fmean(finite)
                    row[f"n_finite_{metric}_tissues"] = len(finite)
                per_seed.append(row)

    indexed = {
        (row["go_slim_id"].replace(":", "_", 1), row["seed"], row["condition"]): row
        for row in per_seed
    }
    condition_summary = []
    comparisons = []
    tissue_comparisons = []
    for category, seeds in category_seeds.items():
        for condition in CONDITIONS:
            row = {
                "go_slim_id": category.replace("_", ":", 1),
                "go_slim_name": CATEGORIES[category],
                "condition": condition,
                "n_seeds": len(seeds),
            }
            for metric in METRICS:
                values = [indexed[(category, seed, condition)][f"mean_{metric}"] for seed in seeds]
                row[f"{metric}_mean"] = statistics.fmean(values)
                row[f"{metric}_sd"] = statistics.stdev(values)
            condition_summary.append(row)

        for metric in METRICS:
            reference = [indexed[(category, seed, "mrna_only")][f"mean_{metric}"] for seed in seeds]
            alternative = [indexed[(category, seed, "mrna_t5u")][f"mean_{metric}"] for seed in seeds]
            differences = [a - b for a, b in zip(alternative, reference)]
            t_pvalue = paired_t_pvalue(alternative, reference)
            comparisons.append(
                {
                    "go_slim_id": category.replace("_", ":", 1),
                    "go_slim_name": CATEGORIES[category],
                    "metric": metric,
                    "n_seeds": len(seeds),
                    "mrna_only_mean": statistics.fmean(reference),
                    "mrna_t5u_mean": statistics.fmean(alternative),
                    "mean_difference": statistics.fmean(differences),
                    "t5u_better_seeds": sum(value > 0 for value in differences),
                    "paired_t_pvalue": t_pvalue,
                }
            )

            tissues = data[(category, seeds[0], "mrna_only")]
            for tissue in tissues:
                reference_tissue = [data[(category, seed, "mrna_only")][tissue][metric] for seed in seeds]
                alternative_tissue = [data[(category, seed, "mrna_t5u")][tissue][metric] for seed in seeds]
                paired = [
                    (a, b) for a, b in zip(alternative_tissue, reference_tissue)
                    if math.isfinite(a) and math.isfinite(b)
                ]
                if len(paired) < 2:
                    continue
                alternative_finite = [a for a, _ in paired]
                reference_finite = [b for _, b in paired]
                t_pvalue = paired_t_pvalue(alternative_finite, reference_finite)
                differences_tissue = [a - b for a, b in paired]
                tissue_comparisons.append(
                    {
                        "go_slim_id": category.replace("_", ":", 1),
                        "go_slim_name": CATEGORIES[category],
                        "tissue": tissue,
                        "metric": metric,
                        "n_paired_seeds": len(paired),
                        "mean_difference": statistics.fmean(differences_tissue),
                        "t5u_better_seeds": sum(value > 0 for value in differences_tissue),
                        "paired_t_pvalue": t_pvalue,
                    }
                )

    # Treat the six functional-class comparisons as one family within each
    # correlation metric. Pearson and Spearman are adjusted separately.
    for metric in METRICS:
        metric_rows = [row for row in comparisons if row["metric"] == metric]
        adjusted = holm_adjust([row["paired_t_pvalue"] for row in metric_rows])
        for row, value in zip(metric_rows, adjusted):
            row["paired_t_holm_pvalue"] = value

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "preliminary" if args.allow_incomplete else "final"
    write_tsv(args.output_dir / f"per_seed_summary_{suffix}.tsv", per_seed)
    write_tsv(args.output_dir / f"condition_summary_{suffix}.tsv", condition_summary)
    write_tsv(args.output_dir / f"comparisons_{suffix}.tsv", comparisons)
    write_tsv(args.output_dir / f"per_tissue_comparisons_{suffix}.tsv", tissue_comparisons)
    print(f"mode: {suffix}")
    print(f"categories summarized: {len(category_seeds)}")
    for category, seeds in category_seeds.items():
        print(f"{category}: matched seeds={seeds}")


if __name__ == "__main__":
    main()
