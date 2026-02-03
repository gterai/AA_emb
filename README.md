# AA_emb

A research-oriented Python toolkit for generating amino acid sequence embeddings
and evaluating translation efficiency (TE) prediction models.

---

# How to install
This package has been tested in a **Linux environment** running on an **Intel64 (x86_64)** architecture with **Python 3.10.13** and **CUDA 12.8**.


## Installation Instructions
To install package, please follow these steps:
```
git clone https://github.com/gterai/AA_emb # Clone the repository
cd AA_emb                                  # Navigate to the RNAgg directory
pip install -r requirements.txt            # Install the required dependencies
```

## Using Different Environments
If you plan to use a different operating system, Python version, or CUDA version, you may need to install appropriate package versions that are compatible with your environment. Below is a list of key dependencies required for this package :
```
numpy
pandas
openpyxl
torch
transformers
protobuf
tiktoken
sentencepiece
```
Using a GPU is strongly recommended and is **effectively required**, as running the program on a CPU results in prohibitively slow performance.

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
## Pipeline

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

The following command creates six files each of which contains protein embedding data of amino acid sequence translated from mRNA sequence in the above excel file.
```
cd AA_emb
bash scripts/1_embed.sh
```
This create protein embedding data 
