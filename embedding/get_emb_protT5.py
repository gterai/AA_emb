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

import torch
from transformers import T5EncoderModel, T5Tokenizer

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
        
#        if len(aa) >= 3001:
#            print(f"AA seq of {sid} is too long and hence truncated" ,file=sys.stderr)
#            aa = aa[:3001]
        
        sid_list.append(sid)
        input_list.append(aa)

    # GPU があれば cuda, なければ cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # モデル名
    if args.mtype == "T5b":
        MODEL_NAME = "Rostlab/prot_t5_xl_bfd"
    elif args.mtype == "T5u":
        MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
    else:
        print(f"Unknown model type ({args.mtype})", file=sys.stderr)
        exit(1)
        
    # トークナイザとモデルをロード
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(MODEL_NAME)

    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"number of param.", params, file=sys.stderr)
    #exit(0)
    
    model = model.to(device)
    model.eval()
    
    BATCH_SIZE = 2
    
    emb_list = []  # embedding を溜める
    with torch.no_grad():
        for i in range(0, len(input_list), BATCH_SIZE):
            print(f"progress:{i}")
            batch = input_list[i:i+BATCH_SIZE]
            
            # Amino acid sequence を "A E C D ..." 形式へ変換
            batch_spaced = [" ".join(list(seq)) for seq in batch]

            # Tokenize
            tokens = tokenizer(
                batch_spaced,
                add_special_tokens=True,
                padding=True,
                return_tensors="pt"
            )
            
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            
            # モデルに通す
            with torch.no_grad():
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = output.last_hidden_state  # (B, L, D)
                #print(hidden.shape)
                
            eos_id = tokenizer.eos_token_id
            
            # padding を除いて平均 (sequence-level embedding)
            for j in range(hidden.size(0)):
                #mask = attention_mask[j].unsqueeze(-1)  # (L, 1)

                # (L,)
                mask = attention_mask[j].clone()
                
                # --- <eos> を除外 ---
                eos_positions = (input_ids[j] == eos_id)
                mask[eos_positions] = 0
                
                # (L, 1)
                mask = mask.unsqueeze(-1)
                
                check_mask_vs_sequence_length(batch[j], mask) # " "を入れる前の配列を渡している
                
                emb = (hidden[j] * mask).sum(dim=0) / mask.sum()
                emb_list.append(emb.cpu())
                
    embeddings = torch.stack(emb_list)  # (batch, hidden_dim)
    print(embeddings.shape)  # 例: torch.Size([2, 1024])
    embeddings = embeddings.cpu().detach().numpy() # これをしないとサイズが馬鹿でかくなる。なぜだろう？

    sid2emb = {}
    for i, sid in enumerate(sid_list):
        sid2emb[sid] = embeddings[i]
        
    f = open(args.out_pkl, 'wb')
    pickle.dump(sid2emb, f)
    f.close()

def check_mask_vs_sequence_length(seq: str, mask: torch.Tensor):
    """
    Check whether the number of valid tokens in mask
    matches the amino acid sequence length.

    Parameters
    ----------
    seq : str
        Original amino acid sequence
    mask : torch.Tensor
        Mask after excluding special tokens (shape: [L] or [L, 1])
    """
    # mask を (L,) に正規化
    if mask.dim() == 2:
        mask = mask.squeeze(-1)
    aa_len = len(seq)
    mask_len = int(mask.sum().item())
    
    if aa_len != mask_len:
        raise ValueError(
            f"AA length = {aa_len}, mask ones = {mask_len}"
        )

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('xlsx', help='input excel file')
    parser.add_argument('out_pkl', help='output file name')
    parser.add_argument('--mtype', required=True, choices=["T5b","T5u"], help='model_type')
    args = parser.parse_args()

    main(args)
    
