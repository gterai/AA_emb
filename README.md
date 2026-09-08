# AA_emb

AA_emb is a research toolkit for predicting translation efficiency (TE) across
78 human cell and tissue types using mRNA sequence features and protein language
model embeddings. It supports mRNA-only, protein-only and combined models, with
fixed benchmarks for protein sequence similarity and held-out biological functions.

## Choose a workflow

| Goal | Start here |
| --- | --- |
| Recreate benchmark statistics and figures without training | [Quick example](#quick-example-no-gpu-required) |
| Generate embeddings and prepare model inputs | [Prepare the input](#prepare-the-input) |
| Train and evaluate a model | [Train a model](#train-a-model) |
| Compare models on c50/c70/c90 or functional holdouts | [Benchmark guide](benchmarks/README.md) |
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

For statistics and plotting only, install the lightweight dependencies below;
PyTorch, pretrained models and the source dataset are not required.

## Quick example (no GPU required)

Run these commands from the repository root to recreate summary tables and
figures from the included per-tissue benchmark metrics:

```bash
pip install -r benchmarks/requirements.txt
python benchmarks/scripts/validate_benchmarks.py
python benchmarks/scripts/summarize_protein_cluster_baseline_t5u.py benchmarks/reference_metrics/protein_cluster_baseline benchmarks/results/protein_cluster_baseline
python benchmarks/scripts/summarize_go_slim_holdout.py benchmarks/reference_metrics/go_slim_holdout benchmarks/results/go_slim_holdout
python benchmarks/scripts/plot_benchmarks.py --results benchmarks/results --output benchmarks/results/figures
```

The output includes Pearson and Spearman comparison figures as PNG/PDF files in
`benchmarks/results/figures/`. This example reproduces statistics and plots from
existing measurements; it does not retrain models or regenerate predictions.

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

## License and model usage

The repository code is provided under the [MIT license](LICENSE). Pretrained
models and upstream datasets retain their own terms; consult their distribution
pages before use. The [license notice](LICENSE) includes information about Ankh.
Model weights and embeddings are not redistributed by this repository.
