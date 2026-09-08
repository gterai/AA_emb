#!/usr/bin/env python3
"""Create leave-one-GO-slim-category-out splits with protein c50 control."""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


SELECTED_CATEGORIES = (
    "GO:0005840",  # ribosome
    "GO:0005886",  # plasma membrane
    "GO:0005576",  # extracellular region
    "GO:0005739",  # mitochondrion
    "GO:0002376",  # immune system process
    "GO:0003723",  # RNA binding
)


def read_audit(path: Path):
    ordered_base_ids = []
    base_to_versioned = {}
    base_to_gene = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["passes_final_cohort"] != "True":
                continue
            base = row["transcript_id_base"]
            ordered_base_ids.append(base)
            base_to_versioned[base] = row["transcript_id_original"]
            base_to_gene[base] = row["gene_id_base"]
    return ordered_base_ids, base_to_versioned, base_to_gene


def read_c50_clusters(path: Path) -> dict[str, str]:
    result = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["threshold"] == "c50":
                result[row["transcript_id"]] = row["cluster_id"]
    return result


def read_go_slim(path: Path):
    category_genes = defaultdict(set)
    category_names = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            category = row["go_slim_id"]
            category_genes[category].add(row["ensembl_gene_id"])
            category_names[category] = row["go_slim_name"]
    return category_genes, category_names


def split_clusters(cluster_to_ids, seed: int):
    cluster_ids = list(cluster_to_ids)
    random.Random(seed).shuffle(cluster_ids)
    total = sum(len(cluster_to_ids[cluster]) for cluster in cluster_ids)
    targets = [int(total * 0.75), total - int(total * 0.75)]
    assigned = [[], []]
    counts = [0, 0]
    for cluster in cluster_ids:
        members = cluster_to_ids[cluster]
        remaining = [target - count for target, count in zip(targets, counts)]
        destination = max(range(2), key=lambda index: (remaining[index], -counts[index]))
        assigned[destination].extend(members)
        counts[destination] += len(members)
    return assigned, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("transcript_audit_tsv", type=Path)
    parser.add_argument("protein_cluster_membership_tsv", type=Path)
    parser.add_argument("gene_go_slim_membership_tsv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    args = parser.parse_args()

    ordered_ids, versioned, transcript_gene = read_audit(args.transcript_audit_tsv)
    transcript_cluster = read_c50_clusters(args.protein_cluster_membership_tsv)
    category_genes, category_names = read_go_slim(args.gene_go_slim_membership_tsv)
    if set(ordered_ids) != set(transcript_cluster):
        raise ValueError("Transcript audit and c50 cluster identifiers differ")

    summary_rows = []
    for category in SELECTED_CATEGORIES:
        if category not in category_genes:
            raise ValueError(f"Selected GO slim category not found: {category}")
        test_ids = {
            sid for sid in ordered_ids if transcript_gene[sid] in category_genes[category]
        }
        blocked_clusters = {transcript_cluster[sid] for sid in test_ids}
        excluded_neighbors = {
            sid for sid in ordered_ids
            if sid not in test_ids and transcript_cluster[sid] in blocked_clusters
        }
        pool_ids = [
            sid for sid in ordered_ids if transcript_cluster[sid] not in blocked_clusters
        ]
        pool_clusters = defaultdict(list)
        for sid in pool_ids:
            pool_clusters[transcript_cluster[sid]].append(sid)

        for seed in args.seeds:
            (train_ids, validation_ids), counts = split_clusters(pool_clusters, seed)
            labels = {sid: "train" for sid in train_ids}
            labels.update({sid: "validation" for sid in validation_ids})
            labels.update({sid: "test" for sid in test_ids})

            cluster_splits = defaultdict(set)
            for sid, label in labels.items():
                cluster_splits[transcript_cluster[sid]].add(label)
            crossing = sum(len(splits) > 1 for splits in cluster_splits.values())
            if crossing:
                raise RuntimeError(f"{category} seed {seed}: c50 cluster leakage")
            if any(transcript_gene[sid] in category_genes[category] for sid in train_ids + validation_ids):
                raise RuntimeError(f"{category} seed {seed}: held-out category leakage")

            split_dir = args.output_dir / category.replace(":", "_") / f"seed_{seed}"
            split_dir.mkdir(parents=True, exist_ok=True)
            with (split_dir / "class.txt").open("w") as handle:
                for sid in ordered_ids:
                    if sid in labels:
                        handle.write(f"{versioned[sid]} {labels[sid]}\n")

            summary_rows.append(
                {
                    "go_slim_id": category,
                    "go_slim_name": category_names[category],
                    "seed": seed,
                    "n_train": counts[0],
                    "n_validation": counts[1],
                    "n_test": len(test_ids),
                    "n_excluded_c50_neighbors": len(excluded_neighbors),
                    "n_used": len(labels),
                    "c50_clusters_crossing_splits": crossing,
                    "heldout_category_in_train_or_validation": 0,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "split_summary.tsv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"categories: {len(SELECTED_CATEGORIES)}")
    print(f"split files: {len(summary_rows)}")
    print("functional-category and c50 leakage validation: PASS")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
