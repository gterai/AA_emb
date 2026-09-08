# Recreate paper figures

## Model comparison: FigMain

```bash
python benchmarks/scripts/plot_main_comparison.py
```

This creates `FigMain.{pdf,png,svg}` and `SFigMain_spearman.{pdf,png,svg}` in
`benchmarks/results/figures/`. For only the requested main PDF:

```bash
python benchmarks/scripts/plot_main_comparison.py --metric pearson --formats pdf
```

Inputs are the 150 per-tissue metric tables in
`benchmarks/reference_metrics/main_comparison/{condition}/seed_{0..9}/metrics.tsv`.
These are the results of the primary model-comparison experiment, not the
c50/c70/c90 experiments. The four panels show HEK293T, HeLa, HepG2 and muscle
tissue. Fifteen conditions include mRNA-only, six protein-only models, six
mRNA-plus-protein models, and amino acid/dipeptide composition controls.

The source figure's 2-by-2 layout, model order, colors and common 0.30-0.85 axis
range are preserved. Bars and error bars are the mean and sample SD across ten
seeds. Brackets compare mRNA+Ankh against mRNA+Ankh3 and mRNA+ESM-2 against
mRNA+ESM-2L using paired t-tests. The stars use **unadjusted** p-values:
`* < 0.05`, `** < 0.01`, `*** < 0.001`, otherwise `n.s.`.

The script also writes `FigMain_{pearson,spearman}_summary.tsv` and
`FigMain_{pearson,spearman}_paired_ttests.tsv` to `benchmarks/results/main_comparison/`.
Expected tables are in `benchmarks/reference_results/main_comparison/`.
Use `--metrics`, `--results`, and `--output` to override these locations.
`--formats` supports `pdf png svg eps tif`; `--metric` supports `pearson`,
`spearman`, or `both` (default).

Only metric tables are distributed: no input arrays, embeddings or model weights.
This addition recreates figures and statistics; the training runners continue
to cover protein-cluster and functional-holdout benchmarks only.

## Robustness: FigRob

`plot_benchmarks.py` uses the plotting layout used to create the manuscript's
`FigRob` (Pearson) and `SFigRob_spearman` figures. Figure numbers may differ
between manuscript versions; these names identify the robustness figures.

Run from the repository root:

```bash
pip install -r benchmarks/requirements.txt
python benchmarks/scripts/plot_benchmarks.py --output benchmarks/results/figures
```

By default this reads `benchmarks/reference_results/`. It creates PDF, PNG
(600 dpi) and SVG files. For the original additional EPS and 1200-dpi LZW TIFF
exports, use:

```bash
python benchmarks/scripts/plot_benchmarks.py --formats pdf png svg eps tif
```

## Figure inputs and calculations

| Panel | Input under the summary root | Calculation |
| --- | --- | --- |
| (a) Protein similarity | `protein_cluster_baseline/per_seed_summary.tsv` | mRNA-only and mRNA+T5u, ordered c90/c70/c50; mean and sample SD of ten run-wise tissue means |
| (b) Controls at c50 | `protein_cluster_controls/condition_summary.tsv`, `random_t5u_control/condition_summary.tsv`, `shuffled_t5u_control/condition_summary.tsv` | `100 * (control - mRNA-only) / (native T5u - mRNA-only)` |
| (c) Functional holdouts | `go_slim_holdout/per_seed_summary_final.tsv` | Mean and sample SD of paired within-seed differences, mRNA+T5u minus mRNA-only |

Panel (a) does not include T5u-only, matching the manuscript. Panel (c) plots
gains rather than separate absolute correlations. Both metrics use the same
panel layout, fixed axis ranges and category order as the source figure code.

The three additional control files are author-generated aggregate results, not
embeddings, trained models, sequences or raw measurements. Their checksums are
included in `metadata/checksums.json`. They are used to redraw panel (b); no code
for training those control experiments is added to the two benchmark runners.

The main README also demonstrates reaggregating the per-tissue reference metrics
for panels (a) and (c). That command explicitly selects the original control
summaries with `--controls-results benchmarks/reference_results`. This reproduces
the same figure inputs while making the source of panel (b) visible.

## Plotting your own results

For the paper layout, `--results` must contain all figure summaries, or a
separate complete control-summary root must be supplied with `--controls-results`.
Do not interpret a figure combining new benchmark runs and original reference
controls as a fully retrained experiment. A fresh complete experiment is not
expected to give bitwise-identical training results across environments.

For an overview of only the two supported training workflows, use:

```bash
python benchmarks/scripts/plot_benchmark_overview.py --results benchmarks/results --output benchmarks/results/figures
```

This optional overview has a different layout and includes T5u-only.

## Fonts and rendering

The manuscript source specifies Arial. Install/provide Arial in your plotting
environment for matching typography; Matplotlib may otherwise substitute another
font. Matplotlib versions and PDF metadata can also differ, so identical file
checksums across machines are not required. The data calculations, panel layout,
axis ranges and category order are preserved from the source plotting code.
