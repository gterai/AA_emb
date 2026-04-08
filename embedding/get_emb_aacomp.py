# -*- coding: utf-8 -*-

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.append(os.environ['HOME'] + "/pyscript")
import codon as codon_lib


AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AA_ORDER)}
codon2aa = codon_lib.set_codon2aa()


def compute_aa_composition(seq: str):
    composition = np.zeros(len(AA_ORDER), dtype=np.float32)
    if not seq:
        return composition

    for aa in seq:
        if aa not in AA_TO_IDX:
            raise ValueError(f"Unknown amino acid: {aa}")
        composition[AA_TO_IDX[aa]] += 1.0

    composition /= len(seq)
    return composition


def main(args: dict):
    df = pd.read_excel(args.xlsx, sheet_name="Human")

    sid2emb = {}
    for idx in df.index:
        sid = df.loc[idx]["tx_id"]
        seq = df.loc[idx]["tx_sequence"]
        len_dict = {
            '5utr': df.loc[idx]['utr5_size'],
            'cds': df.loc[idx]['cds_size'],
            '3utr': df.loc[idx]['utr3_size'],
        }

        seq_5utr = seq[:len_dict['5utr']]
        seq_cds = seq[len_dict['5utr']:len_dict['5utr'] + len_dict['cds']]
        seq_3utr = seq[len_dict['5utr'] + len_dict['cds']:]

        aa = codon_lib.translate(seq_cds, codon2aa)
        aa = aa[:-1]  # 終止コドンは除く

        if '*' in aa:
            print(f"AA seq of {sid} contains '*'", file=sys.stderr)
            continue

        sid2emb[sid] = compute_aa_composition(aa)

    with open(args.out_pkl, 'wb') as f:
        pickle.dump(sid2emb, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('xlsx', help='input excel file')
    parser.add_argument('out_pkl', help='output file name')
    args = parser.parse_args()

    main(args)
