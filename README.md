# AA_emb

Predict translation efficiency across 78 human cell and tissue types using
mRNA features and protein language model features.

## Where to start

| What you want to do | Directory |
| --- | --- |
| Create `input.pkl.gz` | [prepare_input/](prepare_input/README.md) |
| Train and evaluate a model | [training/](training/README.md) |
| Use the same train/test divisions as the paper | [partitions/](partitions/README.md) |
| Recreate the paper figures | [figures/](figures/README.md) |
| Find the precomputed results used for each figure | [figure_data/](figure_data/README.md) |

The repository supports retraining the proposed models and recreating the main
result figures (Figures 3–6). It does not rerun every analysis in the paper.
Figures 1–2 are explanatory diagrams. Pretrained weights, generated embeddings,
trained checkpoints and original datasets are not distributed.

Clone the repository and run the commands below from its root directory:

```bash
git clone https://github.com/gterai/AA_emb_private AA_emb
cd AA_emb
```

## 1. Create the input

Download the source dataset and install the training dependencies as described
in [prepare_input/README.md](prepare_input/README.md), then run:

```bash
bash prepare_input/1_embed.sh
bash prepare_input/2_prepare.sh
```

The output is `data/processed/input.pkl.gz`. This step generates the model
features locally and can require substantial GPU time and memory.

## 2. Train and evaluate

With `input.pkl.gz` prepared, run one paper condition:

```bash
python training/run_paper.py primary --conditions mrna_t5u --seeds 0 --device cuda
```

The runner uses the paper's partition files in `partitions/`. Results are saved
to `training/runs/`. See [training/README.md](training/README.md) for the complete
model comparison, c50/c70/c90 benchmarks, and held-out function/Pfam evaluation.

## Recreate figures from reference results (no GPU required)

You can recreate the figures without preparing `input.pkl.gz` or training models.
Install the plotting dependencies and use the precomputed data in `figure_data/`:

```bash
pip install -r figures/requirements.txt
python figures/plot_figure3.py
python figures/plot_figure4.py
python figures/plot_figure5.py
```

For Figure 6, first download Table EV3 as explained in
[figures/README.md](figures/README.md#figure-6), then run:

```bash
python figures/plot_figure6.py --table-ev3 data/raw/eraslan2019/44320_2019_BFMSB188513_MOESM5_ESM.zip
```

The figures are saved as PDF, PNG and SVG in `outputs/figures/`. The scripts use
the statistical procedures described in the paper, including Holm correction
for Figure 3. Figure 6 uses the paper's precomputed prediction values and
recalculates correlations and confidence intervals; it does not retrain models.

## License and model usage

The repository code is provided under the [MIT license](LICENSE). Pretrained
models and upstream datasets retain their own terms; consult their distribution
pages before use. The [license notice](LICENSE) includes information about Ankh.
Model weights and embeddings are not redistributed by this repository.
