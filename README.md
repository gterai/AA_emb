# AA_emb

AA_emb is a research toolkit for predicting translation efficiency (TE) across
78 human cell and tissue types using mRNA sequence features and protein language
model embeddings. It supports mRNA-only, protein-only and combined models, with
fixed benchmarks for protein sequence similarity and held-out biological functions.

## Reproducibility scope

This repository provides two complementary workflows:

- **Recreate figures:** redraw the main result figures (Figures 3–6) from
  reference evaluation outputs. Figure 6 additionally requires user-downloaded
  Table EV3. Figures 1–2 are explanatory diagrams, not numerical analyses.
- **Retrain the proposed models:** run the primary model comparison, protein
  similarity benchmarks and functional holdouts through `evaluation/eval_multiemb.py`
  using a locally prepared `input.pkl.gz` and the published fixed partitions.

The repository does not provide an end-to-end rerun of every analysis in the
paper. External-model training, random/shuffled-encoder controls, mouse transfer
and the complete supplementary analysis suite are outside the supported training
workflows. Their inclusion in reference figures does not imply training support.
See [the figure and analysis map](benchmarks/FIGURES.md).

## Choose a workflow

| Goal | Start here |
| --- | --- |
| Recreate benchmark statistics and figures without training | [Recreate paper figures](#recreate-figures-from-reference-results-no-gpu-required) |
| Generate embeddings and prepare model inputs | [Prepare the input](#prepare-the-input) |
| Train and evaluate a model | [Train a model](#train-a-model) |
| Run primary, c50/c70/c90 or functional benchmarks | [Benchmark guide](benchmarks/README.md) |
| Check data integrity and run workflow tests | [Testing guide](benchmarks/TESTING.md) |

Pretrained weights, generated embeddings, trained checkpoints and raw source
data are not distributed. Users obtain the source data and pretrained models,
and generate embeddings locally using the scripts below.

## Installation

Clone this repository and enter its directory:

```bash
git clone https://github.com/gterai/AA_emb_private AA_emb
cd AA_emb
```

For model input preparation and training, the existing pipeline provides a
Linux x86_64 environment using Python 3.10.13 and CUDA 12.8:

```bash
pip install -r requirements.txt
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r benchmarks/requirements.txt
```

Use an appropriate PyTorch build for other platforms. The full embedding pipeline
uses large protein language models; a GPU with at least 48 GB memory is recommended.
CPU embedding generation and full training can be very slow. Requirements depend
on the chosen model, sequence lengths and batch size.

## Prepare the input

1. Download Supplementary Table 1 from the [source study in Nature Biotechnology](https://www.nature.com/articles/s41587-025-02712-x#Sec21).
2. Place `41587_2025_2712_MOESM3_ESM.xlsx` in `data/raw/`.
3. Generate protein features and construct the training input:

```bash
bash scripts/1_embed.sh
bash scripts/2_prep.sh
```

The resulting file is `data/processed/input.pkl.gz`. It contains transcript IDs,
encoded mRNA inputs, TE values and masks, protein features, and the ordered list
of 78 output tissues. Treat it as a locally generated file; it is ignored by Git.

The embedding pipeline supports:

| Representation | Input key |
| --- | --- |
| ProtT5 UniRef50 | `emb_T5u` |
| ProtT5 BFD | `emb_T5b` |
| ESM-2 650M / 3B | `emb_esm2` / `emb_esm2L` |
| Ankh base / Ankh3 XL | `emb_ank` / `emb_ank3` |
| Amino acid composition (20 features) | `emb_aacom` |
| Dipeptide composition (400 features) | `emb_dipep` |

## Train a model

For a single run with the default split and seed:

```bash
# mRNA-only
python evaluation/eval_multiemb.py data/processed/input.pkl.gz

# mRNA + T5u
python evaluation/eval_multiemb.py data/processed/input.pkl.gz --emb_name emb_T5u

# T5u-only (zero the mRNA representation)
python evaluation/eval_multiemb.py data/processed/input.pkl.gz --emb_name emb_T5u --abl_type m
```

The default evaluator saves `model_CNN.pth` and `class.txt` in the working
directory. Supply distinct `--model_fname` and `--out_class_fname` values for
separate runs. Use `--metrics_tsv path/to/metrics.tsv` to write the tissue name,
Pearson correlation, Spearman correlation and observed test sample count.
See `python evaluation/eval_multiemb.py --help` for all options.

To compare against the reference results, use the fixed benchmark workflows
rather than the evaluator's default split:

```bash
python benchmarks/scripts/run_benchmarks.py primary --device cuda
python benchmarks/scripts/run_benchmarks.py protein --device cuda
python benchmarks/scripts/run_benchmarks.py function --device cuda
```

The [benchmark guide](benchmarks/README.md) explains the partitions, model
conditions, run selection, statistical comparisons and output files.

## Repository layout

| Directory | Contents |
| --- | --- |
| `embedding/` | Protein embedding and composition feature generation |
| `preprocessing/` | Source data processing and input assembly |
| `evaluation/` | Model definitions, training and evaluation |
| `scripts/` | Embedding and input preparation entry points |
| `benchmarks/` | Fixed partitions, reference metrics, benchmark runners and analysis |
| `data/` | Locations for locally obtained source data and generated inputs |

## Recreate figures from reference results (no GPU required)

For statistics and plotting only, install the lightweight dependencies below;
PyTorch, pretrained models and the source dataset are not required.

Run these commands from the repository root to recreate
the paper's primary comparison (`FigMain`, Figure 3), robustness (`FigRob`,
Figure 4), and recent-model comparison (`FigModel`, Figure 5) figures,
including their Spearman counterparts, from the included reference metrics and
control summaries:

```bash
pip install -r benchmarks/requirements.txt
python benchmarks/scripts/validate_benchmarks.py
python benchmarks/scripts/plot_main_comparison.py
python benchmarks/scripts/plot_recent_models.py
python benchmarks/scripts/summarize_protein_cluster_baseline_t5u.py benchmarks/reference_metrics/protein_cluster_baseline benchmarks/results/protein_cluster_baseline
python benchmarks/scripts/summarize_go_slim_holdout.py benchmarks/reference_metrics/go_slim_holdout benchmarks/results/go_slim_holdout
python benchmarks/scripts/plot_benchmarks.py --results benchmarks/results --controls-results benchmarks/reference_results --output benchmarks/results/figures
```

The output includes `FigMain`, `SFigMain_spearman`, `FigRob`,
`SFigRob_spearman`, `FigModel` and `SFigModel_spearman` as PDF, PNG and SVG files in
`benchmarks/results/figures/`, using the paper's panel layout, axis ranges,
category order, colors and error bars. `FigMain` compares 15 model conditions
across HEK293T, HeLa, HepG2 and muscle tissue; its mean/SD and paired-test tables
are written to `benchmarks/results/main_comparison/`.

The `FigRob` panels show:

- **(a)** mRNA-only versus mRNA+T5u at c90/c70/c50.
- **(b)** The fraction of native T5u gain recovered by length, composition,
  random-encoder and shuffled-sequence controls at c50. The length control uses
  `log1p(length)` standardized with the training subset mean and sample SD.
- **(c)** Paired performance gains in the six functional holdouts.

The control panel uses included aggregate TSVs; it requires no embeddings or
checkpoints. These commands redraw Figures 3–5 from reference results without retraining.
The command below recreates Figure 6 after Table EV3 is downloaded. See the [figure guide](benchmarks/FIGURES.md)
for input provenance, export options and font requirements.

### External validation figure: FigPTR (Table EV3 required)

Download Table EV3 from [Eraslan et al. (2019)](https://doi.org/10.15252/msb.20188513)
into `data/raw/eraslan2019/`, then run:

```bash
python benchmarks/scripts/plot_ptr_validation.py --table-ev3 data/raw/eraslan2019/44320_2019_BFMSB188513_MOESM5_ESM.zip
```

This recreates `FigPTR.pdf` and its PNG/SVG versions. It recalculates correlations
and 5,000 cluster-bootstrap replicates per threshold from the user-supplied PTR
measurements and included prediction medians. Table EV3 is not distributed and
is excluded by `.gitignore`. See [the PTR guide](benchmarks/PTR.md) for the
source download, extracted-TSV option, inputs and outputs.

## License and model usage

The repository code is provided under the [MIT license](LICENSE). Pretrained
models and upstream datasets retain their own terms; consult their distribution
pages before use. The [license notice](LICENSE) includes information about Ankh.
Model weights and embeddings are not redistributed by this repository.
