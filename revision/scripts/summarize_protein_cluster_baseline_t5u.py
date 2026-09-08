#!/usr/bin/env python3
"""Compare mRNA-only and mRNA+T5u on matched protein-cluster splits."""

import argparse
import csv
import statistics
from pathlib import Path

from scipy.stats import ttest_rel, wilcoxon


THRESHOLDS = ("c90", "c70", "c50")
CONDITIONS = ("mrna_only", "mrna_t5u", "t5u_only")
METRICS = ("pearson", "spearman")


def read_metrics(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 78:
        raise ValueError(f"Expected 78 tissues in {path}, found {len(rows)}")
    result = {}
    for row in rows:
        tissue = row["tissue"]
        if tissue in result:
            raise ValueError(f"Duplicate tissue {tissue} in {path}")
        result[tissue] = {
            "pearson": float(row["pearson"]),
            "spearman": float(row["spearman"]),
            "n_test": int(row["n_test"]),
        }
    return result


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    total = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def paired_tests(alternative: list[float], reference: list[float]) -> tuple[float, float]:
    t_pvalue = float(ttest_rel(alternative, reference).pvalue)
    try:
        w_pvalue = float(wilcoxon(alternative, reference).pvalue)
    except ValueError:
        w_pvalue = 1.0
    return t_pvalue, w_pvalue


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
    args = parser.parse_args()

    data = {}
    per_seed = []
    for threshold in THRESHOLDS:
        for seed in args.seeds:
            for condition in CONDITIONS:
                path = args.run_root / condition / threshold / f"seed_{seed}" / "metrics.tsv"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing result: {path}")
                metrics = read_metrics(path)
                data[(threshold, seed, condition)] = metrics
                row = {"threshold": threshold, "seed": seed, "condition": condition}
                for metric in METRICS:
                    row[f"mean_{metric}"] = statistics.fmean(
                        tissue_values[metric] for tissue_values in metrics.values()
                    )
                per_seed.append(row)

    indexed = {
        (row["threshold"], row["seed"], row["condition"]): row for row in per_seed
    }
    condition_summary = []
    overall_comparisons = []
    for threshold in THRESHOLDS:
        for condition in CONDITIONS:
            row = {"threshold": threshold, "condition": condition, "n_seeds": len(args.seeds)}
            for metric in METRICS:
                values = [
                    indexed[(threshold, seed, condition)][f"mean_{metric}"]
                    for seed in args.seeds
                ]
                row[f"{metric}_mean"] = statistics.fmean(values)
                row[f"{metric}_sd"] = statistics.stdev(values)
            condition_summary.append(row)

        for metric in METRICS:
            reference = [
                indexed[(threshold, seed, "mrna_only")][f"mean_{metric}"]
                for seed in args.seeds
            ]
            alternative = [
                indexed[(threshold, seed, "mrna_t5u")][f"mean_{metric}"]
                for seed in args.seeds
            ]
            differences = [alt - ref for alt, ref in zip(alternative, reference)]
            t_pvalue, w_pvalue = paired_tests(alternative, reference)
            overall_comparisons.append(
                {
                    "threshold": threshold,
                    "metric": metric,
                    "n_seeds": len(args.seeds),
                    "mrna_only_mean": statistics.fmean(reference),
                    "mrna_t5u_mean": statistics.fmean(alternative),
                    "mean_difference": statistics.fmean(differences),
                    "relative_improvement_percent": 100
                    * statistics.fmean(differences)
                    / statistics.fmean(reference),
                    "paired_t_pvalue": t_pvalue,
                    "wilcoxon_pvalue": w_pvalue,
                }
            )

    # Pearson and Spearman define separate multiplicity families.
    for metric in METRICS:
        metric_rows = [row for row in overall_comparisons if row["metric"] == metric]
        adjusted = holm_adjust([row["paired_t_pvalue"] for row in metric_rows])
        for row, value in zip(metric_rows, adjusted):
            row["paired_t_holm_pvalue"] = value

    tissue_comparisons = []
    for threshold in THRESHOLDS:
        tissues = list(data[(threshold, args.seeds[0], "mrna_only")])
        for tissue in tissues:
            for metric in METRICS:
                reference = [
                    data[(threshold, seed, "mrna_only")][tissue][metric]
                    for seed in args.seeds
                ]
                alternative = [
                    data[(threshold, seed, "mrna_t5u")][tissue][metric]
                    for seed in args.seeds
                ]
                differences = [alt - ref for alt, ref in zip(alternative, reference)]
                t_pvalue, w_pvalue = paired_tests(alternative, reference)
                tissue_comparisons.append(
                    {
                        "threshold": threshold,
                        "tissue": tissue,
                        "metric": metric,
                        "mrna_only_mean": statistics.fmean(reference),
                        "mrna_t5u_mean": statistics.fmean(alternative),
                        "mean_difference": statistics.fmean(differences),
                        "t5u_better_seeds": sum(diff > 0 for diff in differences),
                        "paired_t_pvalue": t_pvalue,
                        "wilcoxon_pvalue": w_pvalue,
                    }
                )

    for threshold in THRESHOLDS:
        for metric in METRICS:
            selected = [
                row for row in tissue_comparisons
                if row["threshold"] == threshold and row["metric"] == metric
            ]
            adjusted = holm_adjust([row["paired_t_pvalue"] for row in selected])
            for row, value in zip(selected, adjusted):
                row["paired_t_holm_pvalue"] = value

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output_dir / "per_seed_summary.tsv", per_seed)
    write_tsv(args.output_dir / "condition_summary.tsv", condition_summary)
    write_tsv(args.output_dir / "overall_comparisons.tsv", overall_comparisons)
    write_tsv(args.output_dir / "per_tissue_comparisons.tsv", tissue_comparisons)
    print(f"Seeds summarized: {len(args.seeds)}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
