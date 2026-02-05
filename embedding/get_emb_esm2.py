# -*- coding: utf-8 -*-

import os
import sys
import argparse
sys.path.append(os.environ['HOME'] + "/pyscript")
#import basic
import numpy as np
import pandas as pd
import pickle
#import gzip
import codon as codon_lib
#from collections import Counter
#from itertools import product

import torch

from transformers import AutoTokenizer, EsmModel

codon2aa = codon_lib.set_codon2aa()

def main(args: dict):

    df = pd.read_excel(args.xlsx, sheet_name="Human")
    
    sid_list = []
    input_list = []
    for idx in df.index:
        
        sid = df.loc[idx]["tx_id"]
        seq = df.loc[idx]["tx_sequence"]
        len_dict = {'5utr':df.loc[idx]['utr5_size'], 'cds':df.loc[idx]['cds_size'], '3utr':df.loc[idx]['utr3_size']}

        
        seq_5utr = seq[:len_dict['5utr']]
        seq_cds  = seq[len_dict['5utr']:len_dict['5utr']+len_dict['cds']]
        seq_3utr = seq[:len_dict['5utr']+len_dict['cds']:]
        
        aa = codon_lib.translate(seq_cds, codon2aa)
        aa = aa[:-1]  # 終始コドンは除く

        if '*' in aa:
            print(f"AA seq of {sid} contains '*'" ,file=sys.stderr)
            continue
        
        sid_list.append(sid)
        input_list.append(aa)

        #if len(sid_list) == 10:
        #    break
        
    # GPU があれば cuda, なければ cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device,file=sys.stderr)
    
    if args.mtype == "esm2":
        MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
    elif args.mtype == "esm2L":
        MODEL_NAME = "facebook/esm2_t36_3B_UR50D"
    else:
        print(f"Unknown model type ({args.mtype})", file=sys.stderr)
        exit(1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME)

    ## special tokenの確認
    #print(tokenizer.all_special_tokens)
    #print(tokenizer.all_special_ids)
    ##exit(0)

    ## 通常トークンの確認
    #vocab = tokenizer.get_vocab()
    #id2tok = {tid: tok for tok, tid in vocab.items()}
    #for tid in range(50):
    #    print(tid, id2tok[tid])
    #exit(0)
    
    
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"number of param.", params, file=sys.stderr)
    
    model.eval()
    model.to(device)
    
    protein_sequences = [seq for seq in input_list] # アミノ酸配列は文字列で渡す
    
    all_mean_embeddings = []
    
    # =========================
    # batch 処理
    # =========================
    BATCH_SIZE=2
    for i in range(0, len(protein_sequences), BATCH_SIZE):
        print(f"progress:{i}/{len(protein_sequences)} {args.mtype}",file=sys.stderr)
        
        batch_seqs = protein_sequences[i : i + BATCH_SIZE]
        #print(batch_seqs)
        #exit(0)
        inputs = tokenizer(
            batch_seqs, 
            add_special_tokens=True, 
            padding=True, 
            return_tensors="pt",
        )
        # GPUへ
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        # (B, L, D)
        hidden_states = outputs.last_hidden_state
        
        # =========================
        # mean pooling（重要部分）
        # =========================
        mask = inputs["attention_mask"].unsqueeze(-1)  # (B, L, 1)
        
        # ===== 特殊トークン除外 =====
        # ESM2: <cls> = 0番目, <eos> = 最後
        mask[:, 0, :] = 0           # CLS
        
        # EOS（token id で直接検出）
        eos_id = tokenizer.eos_token_id
        eos_positions = inputs["input_ids"] == eos_id  # (B, L) bool
        mask[eos_positions.unsqueeze(-1)] = 0


        check_mask_vs_sequence_length(batch_seqs, inputs, mask, tokenizer)

        
        # mean pooling
        hidden_states = hidden_states * mask
        mean_embeddings = hidden_states.sum(dim=1) / mask.sum(dim=1)
    
        all_mean_embeddings.append(mean_embeddings.cpu())

    # (N, hidden_dim)
    all_mean_embeddings = torch.cat(all_mean_embeddings, dim=0)
    embeddings = all_mean_embeddings.detach().numpy() 
    #print(embeddings.shape,file=sys.stderr)
    
    sid2emb = {}
    for i, sid in enumerate(sid_list):
        sid2emb[sid] = embeddings[i]
        
    f = open(args.out_pkl, 'wb')
    pickle.dump(sid2emb, f)
    f.close()

def check_mask_vs_sequence_length(seqs, inputs, mask, tokenizer):
    """
    seqs: list[str]        元のアミノ酸配列
    inputs: tokenizer出力
    mask: attention_mask after CLS/EOS removal (B,L or B,L,1)
    tokenizer: HF tokenizer
    """

    # mask を (B, L) に統一
    if mask.dim() == 3:
        mask_2d = mask.squeeze(-1)
    else:
        mask_2d = mask

    input_ids = inputs["input_ids"]

    for i, seq in enumerate(seqs):
        aa_len = len(seq)
        mask_len = int(mask_2d[i].sum().item())

        # tokenized length（参考情報）
        token_ids = input_ids[i]
        token_len = int((token_ids != tokenizer.pad_token_id).sum().item())

        #print(
        #    f"[{i:03d}] "
        #    f"AA length = {aa_len}, "
        #    f"mask ones = {mask_len}, "
        #    f"tokenized length (incl. special) = {token_len}"
        #)

        if aa_len != mask_len:
            print(
                f"  MISMATCH detected! "
                f"(AA={aa_len}, mask={mask_len})"
            )
            exit(0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('xlsx', help='input excel file')
    parser.add_argument('out_pkl', help='output file name')
    parser.add_argument('--mtype', required=True, choices=["esm2","esm2L"], help='model_type')
    args = parser.parse_args()

    main(args)
    
