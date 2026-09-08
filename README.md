# AA_emb

## Revision: protein-cluster and functional holdout evaluation

See [revision/README.md](revision/README.md) to train on the published c50/c70/c90
and leave-one-function-out splits using the existing `input.pkl.gz`. The revision
includes fixed partitions, per-tissue reference metrics, aggregation and plotting
code. Embeddings and model checkpoints are not distributed.

A research-oriented Python toolkit for generating amino acid sequence embeddings
and evaluating translation efficiency (TE) prediction models.

---

# GPU Memory Requirement

This system relies on large-scale protein language models, which require substantial GPU memory to run efficiently.
We strongly recommend using a GPU with at least 48 GB of memory. Using GPUs with smaller memory may lead to out-of-memory errors.

---

# How to install
This package has been tested in a **Linux environment** running on an **Intel64 (x86_64)** architecture with **Python 3.10.13** and **CUDA 12.8**.


## Installation Instructions
To install package, please follow these steps:
```
git clone https://github.com/gterai/AA_emb_private AA_emb # Clone the private repository
cd AA_emb                                  # Navigate to the RNAgg directory
pip install -r requirements.txt            # Install the required dependencies
```

Then, install PyTorch with CUDA 12.8 support:
```
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 
```


## Using Different Environments
If you plan to use a different operating system, Python version, or CUDA version, you may need to install appropriate package versions that are compatible with your environment. Below is a list of key dependencies required for this package :
```
numpy
pandas
scipy
openpyxl
torch
transformers
protobuf
tiktoken
sentencepiece
```
Using a GPU is strongly recommended and is **effectively required**, as running the program on a CPU results in prohibitively slow performance.

# Overview

This repository provides a complete pipeline for:

1. Generating **protein sequence embeddings** using pretrained protein language models
2. Generating **amino acid composition features** (20-dimensional frequency vectors)
3. Generating **di-peptide composition features** (400-dimensional frequency vectors)
4. Constructing machine-learning-ready input datasets
5. Evaluating translation efficiency prediction performance, including **ablation studies**

The codebase was developed for systematic evaluation of how amino acid embeddings
contribute to translation efficiency prediction across multiple tissues and cell types.

---

## Supported Protein Language Models

- **Ankh** (ankh-base, ankh3-xl)
- **ESM2** (esm2_t33_650m_ur50d, esm2_t36_3b_ur50d)
- **ProtT5** (prot_t5_xl_uniref50, prot_t5_xl_bfd)
- **AA composition** (20-dimensional normalized amino acid frequencies)
- **Di-peptide composition** (400-dimensional normalized adjacent amino acid frequencies)

Each model is used to generate sequence-level embeddings via mean pooling
over residue-level representations.

Note on the license of Ankh:
- Ankh is licensed under CC BY-NC-SA 4.0 and may not be used for commercial purposes.　Users are responsible for ensuring compliance with the license terms.

---
# Pipeline

### External data (not included)
Please download **Supplementary Table 1** from the following paper:

Zeing et al., *Nature Biotechnology* (2025)  
https://www.nature.com/articles/s41587-025-02712-x#Sec21

After downloading, place the file:
```
41587_2025_2712_MOESM3_ESM.xlsx
```
into the following directory:
```
AA_emb/data/raw/
```

### Generate protein embeddings
The following command generates six protein embedding files, one amino acid composition feature file, and one di-peptide composition feature file. Each file contains sequence-level features derived from amino acid sequences translated from the mRNA sequences provided in the Excel file above.
```
cd AA_emb
bash scripts/1_embed.sh
```

After the script finishes, the generated embedding files will be located in:
```
AA_emb/data/intermediate/
```
Notes
- GPU is **strongly recommended**, as embedding generation on CPU is extremely slow.
- The execution of 1_embed.sh is time-consuming and typically requires approximately **6–12 hours**, depending on the GPU configuration.
- The protein language model commands in 1_embed.sh are independent and can be parallelized across multiple GPUs if needed. The amino acid composition and di-peptide composition feature generation are lightweight and CPU-friendly.
### Generate input data

The following command generates the input file used for model training and evaluation.
The file contains translation efficiency (TE) values and mRNA sequences represented in a one-hot–like format, as described in the main paper.
```
bash scripts/2_prep.sh
```
After the script finishes, the generated file **input.pkl.gz** will be located in:
```
AA_emb/data/processed/
```

### Learn and evaluate
The following command performs training and evaluation of the translation efficiency (TE) prediction model by combining mRNA features with embeddings generated by the prot_t5_xl_bfd model.
```
python evaluation/eval_multiemb.py data/processed/input.pkl.gz --emb_name emb_T5b
```

You can change the embedding type by modifying the --emb_name option. For example, to use embeddings generated by the esm2_t33_650m_ur50d model, run the following command:
```
python evaluation/eval_multiemb.py data/processed/input.pkl.gz --emb_name emb_esm2
```
To use amino acid composition features instead, run:
```
python evaluation/eval_multiemb.py data/processed/input.pkl.gz --emb_name emb_aacom
```
To use di-peptide composition features instead, run:
```
python evaluation/eval_multiemb.py data/processed/input.pkl.gz --emb_name emb_dipep
```
For additional options, see the help message:
```
python evaluation/eval_multiemb.py --help
```


### Ablation studies
The script eval_multiemb.py also allows you to perform ablation studies.

**•	Remove mRNA features:**
```
python evaluation/eval_multiemb.py data/processed/input.pkl.gz --emb_name emb_esm2 --abl_type m
```
**•	Remove protein embedding features:**
```
python evaluation/eval_multiemb.py data/processed/input.pkl.gz --emb_name emb_esm2 --abl_type p
```
**•	Remove part of the mRNA features:**
```
python evaluation/eval_multiemb.py data/processed/input.pkl.gz --emb_name emb_esm2 --abl_type v --abl_dim 0 1 2 3
```
The command above masks dimensions 0–3 of the mRNA feature vectors.
Each mRNA feature is represented as a sequence of 6-dimensional vectors
(see Figure 1 in the main text). Hence, this operation removes nucleotide
information from the mRNA features.


### Output data format

The evaluation script writes comment lines describing the selected epoch and
aggregate test performance, followed by a three-column table:

```text
tissue	pearson	spearman
TE_108T	...	...
TE_12T	...	...
```

Use `--metrics_tsv path/to/metrics.tsv` for a standalone machine-readable table
with `tissue`, `pearson`, `spearman`, and `n_test` (the tissue-specific number of
observed test targets). `--input_class_fname` accepts a fixed split file and
preserves the order within each subset. See [revision/README.md](revision/README.md)
for the c50/c70/c90 and leave-one-function-out workflows.
