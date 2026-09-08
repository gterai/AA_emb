# Recreate FigPTR from Table EV3

This workflow redraws the four-panel external PTR validation figure using a
user-downloaded copy of Table EV3 and the included reference TE predictions.
It requires no GPU, embeddings or model checkpoints. PTR is a protein-abundance
related outcome, not a direct measurement of translation efficiency.

## Obtain Table EV3 yourself

Download **Table EV3** from the supplementary material of:

Eraslan et al. (2019), *Quantification and discovery of sequence determinants of
protein-per-mRNA amount in 29 human tissues*, Molecular Systems Biology 15:e8513.
[Source article and supplementary material](https://doi.org/10.15252/msb.20188513).

Place the downloaded archive in `data/raw/eraslan2019/`. The archive used for the
reference analysis is named `44320_2019_BFMSB188513_MOESM5_ESM.zip` and contains
`Table_EV3/Table_EV3.tsv`. The script accepts either this ZIP or the extracted
TSV. Obtain the source from the publisher; no automatic download is performed.

The entire `data/raw/eraslan2019/` directory, the published archive filename,
and Table EV3 filenames/directories are excluded through `.gitignore`.
Neither source measurements nor intermediate PTR arrays are distributed.

## Generate the figure

Run from the repository root:

```bash
pip install -r benchmarks/requirements.txt
python benchmarks/scripts/plot_ptr_validation.py --table-ev3 data/raw/eraslan2019/44320_2019_BFMSB188513_MOESM5_ESM.zip
```

Or use the extracted TSV:

```bash
python benchmarks/scripts/plot_ptr_validation.py --table-ev3 data/raw/eraslan2019/Table_EV3/Table_EV3.tsv
```

The default outputs are `FigPTR.pdf`, `FigPTR.png` (300 dpi) and `FigPTR.svg` in
`benchmarks/results/figures/`. Add `--formats pdf` to save only PDF, or include
`tif` for a 600-dpi LZW TIFF. `--output-prefix` changes the destination and stem.

The script recomputes the correlations and **5,000 cluster-bootstrap replicates
per threshold**, using seed 20260826. This is more expensive than simply plotting
saved summary tables and may take several minutes on a CPU. Progress is printed
every 1,000 replicates. `--bootstrap` and `--seed` can be changed for development,
but use the defaults to reproduce the figure's confidence intervals.

## Inputs and data handling

- **User input:** Table EV3 provides the already log10-transformed `*_PTR`
  values for 29 tissues. No additional log transform or z-score is applied.
  Missing values (`NA`) remain missing.
- **Reference predictions:** `reference_predictions/ptr_validation/{c50,c70,c90}.tsv`
  contains only versioned transcript IDs and the median predicted TE for each
  of two models. It contains no observed PTR, protein/mRNA measurements,
  embeddings, weights or 78-dimensional latent representations.
- Predictions were originally generated only when the transcript was in a
  held-out test split. Predictions across eligible seeds were averaged per
  output head, then the median across 78 heads was taken. The supplied values
  are these final medians, not the median of per-seed medians.
- **Metadata:** the cohort IDs, protein-cluster memberships and
  `metadata/ptr_reference.json` define the expected source structure and cohorts.
  Version suffixes are removed only for matching Ensembl transcript IDs.
- Table EV3 contains 11,575 transcripts; 5,255 match the model cohort and have
  usable PTR. The OOF cohorts contain 4,687/c50, 4,679/c70 and 4,668/c90 transcripts.
  These counts are checked before the source figure's fixed workflow labels are drawn.

## Panels and generated tables

| Panel | Calculation |
| --- | --- |
| (a) | Dataset matching and held-out prediction workflow |
| (b) | c50 transcript-level scatter: median observed log10 PTR across available tissues versus median predicted TE; least-squares line and Pearson/Spearman correlations |
| (c) | Paired model difference in Spearman correlation at c90/c70/c50, with 95% percentile intervals from protein-cluster bootstrap resampling |
| (d) | Per-tissue Spearman differences at c50, comparing each tissue's observed PTR to the same median-78-head TE prediction |

The generated `figure_primary_correlations.tsv`, `figure_tissue_differences.tsv`,
`primary_cluster_bootstrap.tsv` and source/settings `metadata.json` are saved to
`benchmarks/results/ptr_validation/` (override with `--results-dir`). Expected
aggregate values are supplied in `benchmarks/reference_results/ptr_validation/`
for comparison; the plotting command recomputes them rather than loading them.

The reference prediction medians are sufficient to reproduce these panels, but
not to recreate the complete 29-by-78 correlation matrix or re-ensemble other
seeds. This workflow reproduces evaluation and the figure, not model training.
See [FIGURES.md](FIGURES.md) for the other paper figures.
