python preprocessing/makeBaseData.py data/raw/41587_2025_2712_MOESM3_ESM.xlsx data/intermediate/base.pkl

python preprocessing/addEmb.py data/intermediate/base.pkl data/processed/input.pkl --emb_pkl data/intermediate/sid2T5b.pkl data/intermediate/sid2T5u.pkl data/intermediate/sid2ank.pkl data/intermediate/sid2ank3.pkl data/intermediate/sid2esm2.pkl data/intermediate/sid2esm2L.pkl --emb_name emb_T5b emb_T5u emb_ank emb_ank3 emb_esm2 emb_esm2L
