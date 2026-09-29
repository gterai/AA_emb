# Precomputed results for the paper figures

These files are the saved numerical results used by the plotting scripts.
They contain no embeddings, checkpoints or original source measurements.
You normally do not need to open or modify them.

| Directory | Contents |
| --- | --- |
| `figure3/metrics/` | Per-tissue correlations for 15 models × ten runs |
| `figure3/statistics/` | Expected means, SD and paired-test results |
| `figure4/` | Protein, functional-holdout, Pfam-holdout and control summaries |
| `figure4/metrics/` | Per-tissue results underlying protein and function summaries |
| `figure5/` | Five models × ten run-wise mean correlations |
| `figure6/predictions/` | Transcript IDs and precomputed model prediction values |
| `figure6/statistics/` | Expected correlations and confidence intervals |

Figure 6 additionally requires user-downloaded Table EV3; it is not included.
`sources.json` records which author analyses and manuscript snapshot these
results correspond to. `figure6/metadata.json` records tissue names and expected
sample counts. Source data are checked by `tests/check_data.py`.

Recreate the figures using the commands in [figures/README.md](../figures/README.md).
The plotting scripts write new outputs to `outputs/`, leaving this directory intact.

Figure 4(d) reads `figure4/motif_holdout_training/per_seed_final.tsv`: five
Pfam motifs × ten runs × two models × two correlation measures (200 rows).
It contains aggregate evaluation results only. The script computes the mean
and sample SD of the ten paired gains for each motif.
