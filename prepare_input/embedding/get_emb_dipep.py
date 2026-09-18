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
DIPEPTIDES = [aa1 + aa2 for aa1 in AA_ORDER for aa2 in AA_ORDER]
DIPEP_TO_IDX = {dipep: idx for idx, dipep in enumerate(DIPEPTIDES)}
codon2aa = codon_lib.set_codon2aa()


def compute_dipeptide_composition(seq: str):
    composition = np.zeros(len(DIPEPTIDES), dtype=np.float32)
    if len(seq) < 2:
        return composition

    for i in range(len(seq) - 1):
        dipep = seq[i:i + 2]
        if dipep not in DIPEP_TO_IDX:
            raise ValueError(f"Unknown dipeptide: {dipep}")
        composition[DIPEP_TO_IDX[dipep]] += 1.0

    composition /= (len(seq) - 1)
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

        sid2emb[sid] = compute_dipeptide_composition(aa)

    with open(args.out_pkl, 'wb') as f:
        pickle.dump(sid2emb, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('xlsx', help='input excel file')
    parser.add_argument('out_pkl', help='output file name')
    args = parser.parse_args()

    main(args)
