# Train and evaluate models

Run from the repository root after [creating the input](../prepare_input/README.md).
`run_paper.py` calls `eval_multiemb.py` with the paper's fixed partitions and
settings. It saves results in `training/runs/` and refuses to overwrite a run.
Training progress and errors are displayed live in the terminal and also saved
to `stdout.txt` and `stderr.txt` in each run directory. Input validation is shown
before training; loading the full input file can take time.

## Quick training example

This example checks that `eval_multiemb.py` runs by training and evaluating an
mRNA+T5u model for **one epoch on 500 transcripts**. It requires an **NVIDIA GPU
with CUDA support**, a CUDA-enabled PyTorch installation, and the prepared
`data/processed/input.pkl.gz`. Run from the repository root:

```bash
mkdir -p outputs/training_example
python training/eval_multiemb.py data/processed/input.pkl.gz \
  --emb_name emb_T5u --device cuda \
  --max_data 500 --epoch 1 --s_bat 8 --seed 0 \
  --model_fname "" \
  --out_class_fname outputs/training_example/class.txt \
  --metrics_tsv outputs/training_example/metrics.tsv
```

The command prints training progress and writes the data split (`class.txt`) and
per-tissue evaluation results (`metrics.tsv`) to `outputs/training_example/`.
It does not save model weights. This small run checks operation; its results are
not comparable to the paper's full experiments. The full input file is loaded
before selecting 500 transcripts, so sufficient system RAM is still needed.

## Paper experiments

| Command | Models and partitions | Runs |
| --- | --- | --- |
| `python training/run_paper.py primary --device cuda` | Figure 3: 15 model conditions; ten mRNA c80 partitions | 150 |
| `python training/run_paper.py protein --device cuda` | mRNA-only, mRNA+T5u and T5u-only; c50/c70/c90 | 90 |
| `python training/run_paper.py function --device cuda` | mRNA-only and mRNA+T5u; six held-out functions | 120 |

Start with one run, or use `--dry-run` to see the commands without training:

```bash
python training/run_paper.py primary --conditions mrna_t5u --seeds 0 --device cuda
python training/run_paper.py protein --groups c50 --conditions mrna_only mrna_t5u --seeds 0 --dry-run
```

The primary experiment uses `partitions/mrna_c80/`. Protein benchmarks use
`partitions/protein_c50/`, `protein_c70/`, `protein_c90/`; the function benchmark
uses `partitions/function_holdout/`. See [partition details](../partitions/README.md).

## Options and outputs

- `--input`: input file, default `data/processed/input.pkl.gz`.
- `--conditions`: selected model keys; use `--help` or `--dry-run` to list them.
- `--groups`: `c80` for primary, `c50 c70 c90` for protein, or GO category IDs for function.
- `--seeds`: selected values from 0 to 9; default all ten.
- `--device`: `cuda`, `cpu`, `mps` or `auto`; full CPU training is slow.
- `--epochs`: default 100; shorter runs are only for development.
- `--output`: a fresh output directory.
- `--save-checkpoints`: save model weights locally; default off.
- `--dry-run`: print the commands without loading input or training.

Each run saves `command.json`, `stdout.txt`, `stderr.txt`, `class_used.txt` and
`metrics.tsv`. Primary outputs are under `training/runs/main_comparison/`;
protein/function outputs are under `training/runs/protein_cluster_baseline/`
and `training/runs/go_slim_holdout/`. Generated files are ignored by Git.

Training uses Adam (learning rate 1e-4), batch size 100, 100 epochs, masked MAE
and selection by the lowest validation MAE. Protein embeddings remain fixed.
To match the primary reference runs, its mRNA-only condition uses
`--emb_name emb_T5u --abl_type p`, which zeros the projected protein branch.
The protein/function mRNA-only conditions omit `--emb_name`. Protein-only
conditions use `--abl_type m`, which zeros the mRNA representation.

## Recreate the paper figures

To recreate Figure 4 from the paper's precomputed results, run:

```bash
python figures/plot_figure4.py
```

This creates the paper's three-panel figure without retraining. It uses the
results in `figure_data/figure4/`, not your own training outputs.
See [the figure guide](../figures/README.md) for Figures 3–6.

These workflows cover the proposed models. External baseline training,
random/shuffled-encoder controls, prediction generation for Figure 6, mouse
transfer and the full supplementary analysis suite are outside this scope.
Hardware/software differences can change training results; bitwise identical
retraining is not promised. Earlier unused evaluator variants are in `no_use/`.
