# AA_emb

A research-oriented Python toolkit for generating amino acid sequence embeddings
and evaluating translation efficiency (TE) prediction models.

---

## Overview

This repository provides a complete pipeline for:

1. Generating **protein sequence embeddings** using pretrained protein language models
2. Constructing machine-learning-ready input datasets
3. Evaluating translation efficiency prediction performance, including **ablation studies**

The codebase was developed for systematic evaluation of how amino acid embeddings
contribute to translation efficiency prediction across multiple tissues and cell types.

---

## Supported Protein Language Models

- **Ankh** (ankh-base, ankh3-xl)
- **ESM2**
- **ProtT5**

Each model is used to generate sequence-level embeddings via mean pooling
over residue-level representations.

---

## Repository Structure

### `embedding/`
Scripts for generating amino acid sequence embeddings.

- `get_emb_ankh.py`  
- `get_emb_esm2.py`  
- `get_emb_protT5.py`  

These scripts output sequence-level embeddings from protein sequences.

- `codon.py`  
  Utility functions required by embedding and preprocessing scripts.

---

### `data/`

- `raw/`  
  Contains the supplemental material from RiboNN:  
  `41587_2025_2712_MOESM3_ESM.xlsx`

- `intermediate/`  
  Intermediate datasets generated from the Excel file.

- `processed/`  
  Final datasets with protein embeddings, used for model training and evaluation.

---

### `preprocessing/`

- `024makeData.py`  
  Converts the RiboNN supplemental Excel file into intermediate datasets.

- `510addEmb_all.py`  
  Adds protein embeddings to intermediate datasets and produces final input data.

---

### `models/`

Neural network architectures used for translation efficiency prediction.

- `MyCNN_emb.py`   (single embedding)
- `MyCNN_2emb.py`  (two embeddings)
- `MyCNN_3emb.py`  (three embeddings)

---

### `evaluation/`

Scripts for evaluating translation efficiency prediction performance.

- `100eval.py`  
  Evaluation using a **single embedding** with extensive ablation options.

- `101eval_2emb.py`  
  Evaluation using **two combined embeddings**  
  (uses `MyCNN_2emb.py`)

- `102eval_3emb.py`  
  Evaluation using **three combined embeddings**  
  (uses `MyCNN_3emb.py`)

All evaluation scripts support cross-validation and statistical analysis.

---

## Typical Workflow

```text
1. 024makeData.py
   └─ create intermediate datasets from RiboNN supplemental material

2. get_emb_*.py
   └─ generate protein embeddings

3. 510addEmb_all.py
   └─ add embeddings to datasets

4. 100eval.py / 101eval_2emb.py / 102eval_3emb.py
   └─ evaluate translation efficiency prediction
