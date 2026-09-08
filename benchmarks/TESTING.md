# Testing and reproducibility

Run the lightweight workflow tests from the repository root:

```bash
python -m unittest discover -s benchmarks/tests -v
python benchmarks/scripts/validate_benchmarks.py
```

These checks require no GPU or pretrained model. They cover data checksums,
cohort membership, all 90 fixed partitions, cluster/function leakage, duplicate
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
- The manuscript figure workflow recreates `FigRob` and `SFigRob_spearman`.
  In the verification environment, both 600-dpi PNGs match the original manuscript
  PNGs pixel-for-pixel, including when benchmark summaries are reaggregated from
  per-tissue metrics. See [FIGURES.md](FIGURES.md) for font and rendering details.
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

The 150 per-tissue reference metric tables used for `FigMain` are included in
`reference_metrics/main_comparison/`. `plot_main_comparison.py` checks their
tissue coverage and selected figure values before writing outputs. In the
verification environment, both FigMain/SFigMain_spearman PNG exports match the
original PNGs pixel-for-pixel, and the rendered FigMain PDF matches the original
at 120 dpi. All four generated summary/paired-test tables match the reference
values at relative tolerance 1e-9 / absolute tolerance 1e-12.
