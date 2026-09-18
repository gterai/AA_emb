#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw data/intermediate data/processed
python prepare_input/embedding/get_embedding.py data/raw/41587_2025_2712_MOESM3_ESM.xlsx data/intermediate/sid2aacom.pkl --model aacom
python prepare_input/embedding/get_embedding.py data/raw/41587_2025_2712_MOESM3_ESM.xlsx data/intermediate/sid2dipep.pkl --model dipep
python prepare_input/embedding/get_embedding.py data/raw/41587_2025_2712_MOESM3_ESM.xlsx data/intermediate/sid2esm2.pkl --model esm2
python prepare_input/embedding/get_embedding.py data/raw/41587_2025_2712_MOESM3_ESM.xlsx data/intermediate/sid2esm2L.pkl --model esm2L
python prepare_input/embedding/get_embedding.py data/raw/41587_2025_2712_MOESM3_ESM.xlsx data/intermediate/sid2ank.pkl --model ank
python prepare_input/embedding/get_embedding.py data/raw/41587_2025_2712_MOESM3_ESM.xlsx data/intermediate/sid2ank3.pkl --model ank3
python prepare_input/embedding/get_embedding.py data/raw/41587_2025_2712_MOESM3_ESM.xlsx data/intermediate/sid2T5b.pkl --model T5b
python prepare_input/embedding/get_embedding.py data/raw/41587_2025_2712_MOESM3_ESM.xlsx data/intermediate/sid2T5u.pkl --model T5u
