# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import pandas as pd
import gzip
import pickle

MAX_PRE_LEN = 1000  # 開始コドン前の長さ
MAX_SUF_LEN = 10000 # 開始コドン後の長さ
MAX_MRNA_LEN = MAX_PRE_LEN + MAX_SUF_LEN

def main(args: dict):
    
    df = pd.read_excel(args.xlsx, sheet_name="Human")

    TE_cols = df.columns[8:86]

    enst2feat = {}

    n_genes = 0
    for idx in df.index:
        enst = df.loc[idx]["tx_id"]
        seq = df.loc[idx]["tx_sequence"]
        len_dict = {'5utr':df.loc[idx]['utr5_size'], 'cds':df.loc[idx]['cds_size'], '3utr':df.loc[idx]['utr3_size']}

        l_pre = len_dict['5utr']
        l_suf = len_dict['cds'] + len_dict['3utr']

        if l_pre > MAX_PRE_LEN:
            print(f"5'UTR is {l_pre} nt")
            continue
        if l_suf > MAX_SUF_LEN:
            print(f"ORF + 3'UTR is {l_suf} nt")
            continue
            
        TE_ar = df.loc[idx][TE_cols].to_numpy().astype(float)
        y_mask_ar = np.isnan(TE_ar).astype(int)
        
        oht, padseq, x_mask_ar, len_dict_pad = one_hot_aligned_encode(seq, len_dict)
        pad_seq_5utr = padseq[:MAX_PRE_LEN]
        
        enst2feat[enst] = {
            "oht"    :oht,
            "x_mask" :x_mask_ar,
            "TE"     :TE_ar,
            "y_mask" :y_mask_ar,
            #"padseq" :padseq,
            #"len_inf":len_dict_pad, # padding後の長さ
            #"parts_len":len_dict, # padding前の長さ
        }

        n_genes += 1
        if n_genes % 1000 == 0:
            print(f"Process: {n_genes}")
        
    f = gzip.open(args.out_pkl,'wb')
    pickle.dump([enst2feat, TE_cols], f)
    f.close

    print(f"Information of {n_genes} genes is stored.")
    
def mask_pad(n):
    if n == 'X':
        return 1
    else:
        return 0


def one_hot_aligned_encode(sequence:str, d:dict):

    len_5utr, len_cds, len_3utr = d['5utr'], d['cds'], d['3utr']

    # 塩基ごとのワンホットエンコーディングを定義
    mapping = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'X': [0, 0, 0, 0]
    }

    pre_pad_len = MAX_PRE_LEN - len_5utr
    for i in range(pre_pad_len):
        sequence = "X" + sequence
    suf_pad_len = MAX_SUF_LEN - (len_cds + len_3utr)
    for i in range(suf_pad_len):
        sequence = sequence + "X"
        
    # MAX_PRE_LENの前にATGがあったら１をつける
    atg_mark = [0 for i in range(len(sequence))]
    for i in range(MAX_PRE_LEN-1):
        if sequence[i:i+3] == "ATG":
            atg_mark[i] = 1
            
    # 塩基配列をワンホットベクトルに変換
    one_hot = np.array([mapping[base] for base in sequence])

    # cds位置を逆算する
    cds_fm = pre_pad_len + len_5utr # 0-based
    cds_to = pre_pad_len +len_5utr + len_cds - 1 # 0-based
    cds_len = cds_to - cds_fm + 1 # read length
    cds_mark = []
    for i,base in enumerate(sequence):
        if i < cds_fm or i > cds_to: # cds外
            cds_mark.append(0)
        else: # cds内
            r_pos = i - cds_fm
            if r_pos % 3 == 0:
                cds_mark.append(1)
            else:
                cds_mark.append(0)

    cds_mark = np.array([cds_mark]).reshape(len(sequence), 1)
    atg_mark = np.array([atg_mark]).reshape(len(sequence), 1)
    one_hot = np.hstack([one_hot, cds_mark])
    one_hot = np.hstack([one_hot, atg_mark])
    
    x_mask = np.array([mask_pad(n) for n in sequence])

    # テスト出力
    # ワンホットエンコーディングの表示
    if(0):
        for i in range(len(one_hot)):
            print(i, one_hot[i])
        exit(0)

    # わかりにくい
    # cds_fmは0-basedなので、これで長さを表す
    # cds_toは0-basedなので、-1しないといけない
    return one_hot, sequence, x_mask, {'5utr':cds_fm, 'cds':cds_len, '3utr':len(sequence) - cds_to - 1}


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('xlsx', help='excel file')
    parser.add_argument('out_pkl', help='output pickle file')
    args = parser.parse_args()

    main(args)
    
