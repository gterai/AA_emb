# Figure 6 calculation details

For the basic command and download location, see [the figure guide](README.md#figure-6).

## Measurements and predictions

Table EV3 supplies processed log10 protein-to-mRNA ratios (PTR) for 11,575
major transcripts and 29 tissues. Values are already log transformed; they are
not transformed again. Missing measurements remain missing. Ensembl transcript
versions are removed for matching, and duplicate stable IDs are rejected.
5,255 transcripts match the model cohort; available test predictions cover
4,687 transcripts at c50, 4,679 at c70 and 4,668 at c90.

The saved predictions were generated only by models for which the transcript
was in the test set, never the training or validation set. This is often called
**out-of-fold (OOF) prediction**. Across the ten random partitions, predictions
from eligible test runs were averaged separately for each of the 78 outputs.
The median across those outputs is distributed in `figure_data/figure6/predictions/`.
These files contain two prediction columns (mRNA-only and mRNA+T5u), not raw PTR
measurements. Generating new predictions from trained models is outside the
supported workflow; recalculating the figure from these predictions is supported.

## Calculations

The main scatterplots compare median observed PTR across available tissues with
median predicted TE. Tissue-specific correlations compare observed PTR in each
tissue with the same median TE prediction, not a matched output head.
Confidence intervals for paired correlation differences sample whole protein
clusters with replacement 5,000 times and take the 2.5th/97.5th percentiles.
The default random seed is 20260826. Expected results are in
`figure_data/figure6/statistics/`; generated tables are in `outputs/figure6/`.
The default figure prefix is `outputs/figures/FigPTR`.

Use `--predictions`, `--results-dir`, `--output-prefix`, `--seed` and `--bootstrap`
only when intentionally changing the calculation or output location. Changing
seed or resampling count will change the confidence intervals.

The original Table EV3 ZIP, extracted TSV and `data/raw/eraslan2019/` directory
are excluded by `.gitignore`. Users obtain them from the source article and
follow its terms. Neither source measurements nor pretrained weights are
redistributed in this repository.
