# Create input.pkl.gz

Run commands from the repository root. You can skip this entire step if you
only want to recreate paper figures from precomputed results.

## Install dependencies

The reference input/training environment uses Linux x86_64, Python 3.10.13 and
CUDA 12.8:

```bash
pip install -r requirements.txt
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r figures/requirements.txt
```

Use an appropriate PyTorch build on other platforms. The full feature pipeline
uses large protein models; a GPU with at least 48 GB memory is recommended.
Memory requirements depend on the model and sequence lengths.

## Download the source data

Download Supplementary Table 1 from the
[source study](https://www.nature.com/articles/s41587-025-02712-x#Sec21) and place
`41587_2025_2712_MOESM3_ESM.xlsx` in `data/raw/`.

```bash
mkdir -p data/raw
bash prepare_input/1_embed.sh
bash prepare_input/2_prepare.sh
```

`1_embed.sh` generates protein features in `data/intermediate/` using the code
in `embedding/`. `2_prepare.sh` combines those features with the mRNA inputs and
TE measurements using `preprocessing/`. The result is
`data/processed/input.pkl.gz`, containing transcript IDs, encoded mRNA inputs,
TE values, missing-value masks, protein features and 78 ordered tissue labels.
Generated files are ignored by Git.

| Protein representation | Input key |
| --- | --- |
| ProtT5 UniRef50 / BFD | `emb_T5u` / `emb_T5b` |
| ESM-2 650M / 3B | `emb_esm2` / `emb_esm2L` |
| Ankh base / Ankh3 XL | `emb_ank` / `emb_ank3` |
| Amino acid / dipeptide composition | `emb_aacom` / `emb_dipep` |

Optional validation: `python tests/check_data.py --input data/processed/input.pkl.gz`.
Only load pickle files you created or otherwise trust. `input_reference.json`
records the reference schema and optional exact feature hashes.
