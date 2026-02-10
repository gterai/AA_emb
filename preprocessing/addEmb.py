# -*- coding: utf-8 -*-

import os
import sys
import argparse
sys.path.append(os.environ['HOME'] + "/pyscript")
import basic
import numpy as np
import pandas as pd
import pickle
import gzip

def main(args: dict):
    
    f = gzip.open(args.ft_pkl, 'rb')
    (sid2ft_org, TE_cols) = pickle.load(f)
    f.close()

    sid2emb_dict_list = []
    for emb_file in args.emb_pkl:
        f = open(emb_file, 'rb')
        sid2tmp = pickle.load(f)
        f.close()

        sid2emb_dict_list.append(sid2tmp)
    
    #sid2emb[sid][emb_name]という形にする

    sid2emb = {}
    for sid in sid2ft_org:
        if sid not in sid2emb_dict_list[0]: # すべてのsid2emb_*.pickleには同じ遺伝子セットのemb情報が入っていると仮定
            continue
        
        sid2emb[sid] = {}
        for i, emb_name in enumerate(args.emb_name):
            sid2emb[sid][emb_name] = sid2emb_dict_list[i][sid]
            
    n_genes = 0
    sid2ft = {}
    for sid in sid2emb:
        sid2ft[sid] = sid2ft_org[sid]
        for emb_name in sid2emb[sid].keys():
            sid2ft[sid][emb_name] = sid2emb[sid][emb_name]
        
        n_genes += 1

    print(f"Information of {n_genes} genes was stored.", file=sys.stderr)
    f = gzip.open(args.out_pkl, 'wb')
    pickle.dump((sid2ft, TE_cols), f)
    f.close()

        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('ft_pkl', help='input pickle file')
    parser.add_argument('out_pkl', help='output file name')
    parser.add_argument('--emb_pkl', type=str, nargs='*', required=True,
                        help='List of embedding pickle files')
    parser.add_argument('--emb_name', type=str, nargs='*', required=True,
                        help='List of embedding names')
    args = parser.parse_args()

    main(args)
    
