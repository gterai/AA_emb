python embedding/get_emb_esm2.py data/raw/41587_2025_2712_MOESM3_ESM.xlsx data/intermediate/sid2esm2.pkl --mtype esm2
python embedding/get_emb_esm2.py data/raw/41587_2025_2712_MOESM3_ESM.xlsx data/intermediate/sid2esm2L.pkl --mtype esm2L
python embedding/get_emb_ankh.py data/raw/41587_2025_2712_MOESM3_ESM.xlsx data/intermediate/sid2ank.pkl --mtype ank
python embedding/get_emb_ankh.py data/raw/41587_2025_2712_MOESM3_ESM.xlsx data/intermediate/sid2ank3.pkl --mtype ank3
python embedding/get_emb_protT5.py data/raw/41587_2025_2712_MOESM3_ESM.xlsx data/intermediate/sid2T5b.pkl --mtype T5b
python embedding/get_emb_protT5.py data/raw/41587_2025_2712_MOESM3_ESM.xlsx data/intermediate/sid2T5u.pkl --mtype T5u
