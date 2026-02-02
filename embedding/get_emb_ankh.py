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
import codon as codon_lib
#from collections import Counter
#from itertools import product

import torch

from transformers import T5Tokenizer, T5EncoderModel

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
    print("Using device:", device)

    if args.mtype == "ank":
        MODEL_NAME = "ElnaggarLab/ankh-base"
    elif args.mtype == "ank3":
        MODEL_NAME = "ElnaggarLab/ankh3-xl"
    else:
        print(f"Unknown model type ({args.mtype})", file=sys.stderr)
        exit(1)

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5EncoderModel.from_pretrained(MODEL_NAME)
    
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"number of param.", params, file=sys.stderr)
    #exit(0)
    
    model.eval()
    model.to(device)
    
    #protein_sequences = ["[NLU]" + seq for seq in input_list] # [NLU]を加える
    protein_sequences = [seq for seq in input_list] 
    all_mean_embeddings = []
    
    # =========================
    # batch 処理
    # =========================
    BATCH_SIZE=2
    for i in range(0, len(protein_sequences), BATCH_SIZE):
        print(f"progress:{i}")
        
        batch_seqs     = protein_sequences[i : i + BATCH_SIZE]
        batch_seqs_NLU = ["[NLU]" + seq for seq in batch_seqs] # [NLU]を加える
        
        # --- tokenize ---
        inputs = tokenizer(
            batch_seqs,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
            is_split_into_words=False,
        )
        # GPUへ
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

        # (B, L, D)
        hidden_states = outputs.last_hidden_state
        
        # --- mask作成 ---
        mask = inputs["attention_mask"].unsqueeze(-1)  # (B, L, 1)
        
        # --- <eos> を除外 ---
        eos_id = tokenizer.eos_token_id
        eos_positions = inputs["input_ids"] == eos_id   # (B, L)
        mask[eos_positions.unsqueeze(-1)] = 0

        # 配列長とmask長の比較
        check_mask_vs_sequence_length(batch_seqs, inputs, mask, tokenizer)
        
        # --- mean pooling ---
        hidden_states = hidden_states * mask
        mean_embeddings = hidden_states.sum(dim=1) / mask.sum(dim=1)
        
        all_mean_embeddings.append(mean_embeddings.cpu())
        
        #print(i)
        
    # (N, hidden_dim)
    all_mean_embeddings = torch.cat(all_mean_embeddings, dim=0)
    embeddings = all_mean_embeddings.detach().numpy()
    print(embeddings.shape)
    
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

        if aa_len + 1 != mask_len: # +1は[NLU]の分
            print(
                f"ERROR: MISMATCH detected!"
                f"(AA={aa_len}, mask={mask_len})"
            )
            exit(0)
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('xlsx', help='input excel file')
    parser.add_argument('out_pkl', help='output file name')
    parser.add_argument('--mtype', required=True, choices=["ank","ank3"], help='model_type')
    args = parser.parse_args()

    main(args)
    
