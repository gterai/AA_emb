# Testing and reproducibility

Run the lightweight workflow tests from the repository root:

```bash
pip install -r benchmarks/requirements.txt
python -m unittest discover -s benchmarks/tests -v
python benchmarks/scripts/validate_benchmarks.py
```

These checks require no GPU or pretrained model. They cover data checksums,
cohort membership, all 100 fixed partitions, cluster/function leakage, duplicate
ID rejection, run matrix construction, T5u-only ablation arguments and refusal
to overwrite existing run directories.

To check a locally generated input, install the main pipeline dependencies and run:

```bash
python benchmarks/scripts/validate_benchmarks.py --input data/processed/input.pkl.gz
```

An optional `--strict-hashes` checks exact core feature content against the
reference. This is stricter than checking schema compatibility, and can fail
when embedding generation uses a different dtype or execution environment.

## Verification coverage

The supplied benchmark data and workflow have been checked as follows:

- All 210 protein-cluster/functional-holdout reference runs match the supplied split membership and within-subset
  ID order.
- All 60 functional partitions regenerate byte-for-byte from the included metadata.
- Reaggregating the 210 per-tissue metric files reproduces the eight reference
  summary tables at relative tolerance 1e-9 / absolute tolerance 1e-12.
- Figures 3–5 and their Spearman counterparts are checked against the current
  author figure exports. Figure 3 uses the mRNA c80 results and Holm correction
  across eight comparisons per metric. Figure 4 uses training-scaled length
  controls and the manuscript labels. Figure 5 uses matched c80 run summaries.
- mRNA-only, mRNA + T5u and T5u-only each completed a one-epoch CPU smoke test on
  synthetic inputs with 78 finite tissue-level Pearson/Spearman metric pairs.
- Reference feature checks confirmed identical transcript IDs, tissue order and
  content hashes for mRNA input, TE, target masks and T5u across the two benchmarks.

CPU smoke tests and aggregation checks used Python 3.12.13, NumPy 2.4.4,
SciPy 1.17.1, PyTorch 2.12.1 and Matplotlib 3.10.8 on macOS. The reference
training runs used an NVIDIA B200 environment. Smoke tests do not substitute
for complete training: the 210 full GPU jobs and embedding generation pipeline
were not rerun for these workflow checks.

## Primary model comparison figure

The 150 current mRNA c80 per-tissue reference metric tables used for `FigMain` are included in
`reference_metrics/main_comparison/`. `plot_main_comparison.py` checks their
tissue coverage and selected figure values before writing outputs. In the
verification environment, both FigMain/SFigMain_spearman PNG exports match the
current manuscript PNGs pixel-for-pixel. All four generated summary/paired-test
tables, including Holm-adjusted p-values, match the reference
values at relative tolerance 1e-9 / absolute tolerance 1e-12.

## External PTR validation

The FigPTR workflow uses user-downloaded Table EV3 (ZIP or TSV) and compact
reference prediction medians. Tests cover ZIP/TSV equivalence, retention of
log10 values and missing values, duplicate ID rejection and the absence of
source measurements in distributed prediction tables. The full 5,000-replicate
cluster bootstrap and all three output tables were checked against the original
analysis. In the verification environment, the FigPTR PNG matches the original
pixel-for-pixel, as does the PDF rendered at 90 dpi. Source Table EV3 files are
excluded by `.gitignore`.

## Scope checks for the current manuscript

All 150 primary run partitions match the distributed mRNA c80 files, including
within-subset order. Tests cover the primary 150-command run matrix, its protein
branch ablation for mRNA-only, Holm step-down correction, and rejection of
missing/duplicate recent-model seeds. Figures 3, 4 and 5 match the current author
PNG exports pixel-for-pixel. Figure 5's Spearman counterpart is generated from
the same 50 validated rows; exact equality with its older export is not claimed.
Primary full GPU training was not repeated during these packaging checks.
