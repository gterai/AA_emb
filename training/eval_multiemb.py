# -*- coding: utf-8 -*-

import os
import sys
import argparse
import random
import gzip
import copy
import datetime
import pickle
import difflib
import re
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

sys.path.append(os.environ['HOME'] + "/pyscript")
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn.functional as F

from scipy.stats import pearsonr, spearmanr
import MyCNN_multiemb


train_loss_list = []
val_loss_list = []
test_loss_list = []


def read_class_file(fname):
    tra_set = []
    val_set = []
    tes_set = []

    with open(fname) as f:
        for line in f:
            line = line.replace('\n', '')
            sid, stype = line.split()

            if stype == "train":
                tra_set.append(sid)
            elif stype == "validation":
                val_set.append(sid)
            elif stype == "test":
                tes_set.append(sid)
            else:
                print("Unknown type (line)")
                exit(1)

    return tra_set, val_set, tes_set


def read_cd_hit_clstr(fname):
    sid_to_cluster = {}
    current_cluster = None

    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">Cluster "):
                current_cluster = line[1:]
                continue

            match = re.search(r">([^\.]+(?:\.[^\.]+)?)\.\.\.", line)
            if match is None:
                raise RuntimeError(f"Failed to parse CD-HIT .clstr line: {line}")
            sid = match.group(1)
            sid_to_cluster[sid] = current_cluster

    return sid_to_cluster


def _sid_at(dataset, index): # random_splitではDatasetではなくSubsetが生成される。これに対応している。
    if isinstance(dataset, Subset):
        return _sid_at(dataset.dataset, dataset.indices[index])
    return dataset.sid[index]


def dataset_sid_list(dataset):
    return [_sid_at(dataset, i) for i in range(len(dataset))]


def make_class_file(d_tra, d_val, d_tes, fname):
    tra_list = dataset_sid_list(d_tra)
    val_list = dataset_sid_list(d_val)
    tes_list = dataset_sid_list(d_tes)

    with open(fname, 'w') as fout:
        for g in tra_list:
            print(g, "train", file=fout)
        for g in val_list:
            print(g, "validation", file=fout)
        for g in tes_list:
            print(g, "test", file=fout)


def print_per_tissue_metrics(tissues, pearson_values, spearman_values, file=sys.stdout):
    """Write test-set correlations for each tissue as a labeled TSV block."""
    print("tissue", "pearson", "spearman", sep="\t", file=file)
    for tissue, pearson, spearman in zip(
        tissues, pearson_values, spearman_values, strict=True
    ):
        print(tissue, pearson, spearman, sep="\t", file=file)


def check_emb_dim(sid2ft: dict, emb_name: str):
    sid_list = list(sid2ft.keys())
    first_item = sid2ft[sid_list[0]]

    if emb_name not in first_item:
        emb_keys = sorted(key for key in first_item.keys() if key.startswith('emb_'))
        suggestions = difflib.get_close_matches(emb_name, emb_keys, n=3) # 親切に似たemb名を出力してエラー終了してくれる
        print(f"Embedding key not found: {emb_name}", file=sys.stderr)
        if suggestions:
            print(f"Did you mean: {', '.join(suggestions)}", file=sys.stderr)
        if emb_keys:
            print(f"Available embedding keys: {', '.join(emb_keys)}", file=sys.stderr)
        exit(1)

    emb_dim = len(first_item[emb_name])
    for sid in sid_list[1:]:
        if emb_dim != len(sid2ft[sid][emb_name]):
            print("Dimension of embedding is different", file=sys.stderr)
            exit(0)

    return emb_dim


def check_same_len(sid2ft: dict):
    sid_list = list(sid2ft.keys())
    length = len(sid2ft[sid_list[0]]['oht'])

    for sid in sid_list:
        ft_len = len(sid2ft[sid]['oht'])
        if ft_len != length:
            print(
                f"The length of feature {ft_len} is different from the indicated length {length}",
                file=sys.stderr,
            )
            exit(0)
    return length


def check_y(sid2ft: dict):
    sid_list = list(sid2ft.keys())

    y_len = len(sid2ft[sid_list[0]]['y_mask'])
    for sid in sid_list[1:]:
        if y_len != len(sid2ft[sid]['y_mask']):
            print("Number of task is different", file=sys.stderr)
            exit(0)
        y_len = len(sid2ft[sid]['y_mask'])

    return y_len


def limit_dataset_size(sid2ft: dict, max_data: int):
    if max_data is None or max_data >= len(sid2ft):
        return sid2ft

    sid_list = list(sid2ft.keys())
    random.shuffle(sid_list)
    selected_sid = sid_list[:max_data]
    return {sid: sid2ft[sid] for sid in selected_sid}


def split_sid2ft_by_cluster(sid2ft, sid_to_cluster, split_sizes, seed):
    # split_sizes: [n_train, n_val, n_test]
    missing_sids = [sid for sid in sid2ft if sid not in sid_to_cluster] # sid_to_clusterに存在しなければエラー終了
    if missing_sids:
        raise RuntimeError(
            f"{len(missing_sids)} IDs are missing from the CD-HIT cluster file. "
            f"Example: {', '.join(missing_sids[:5])}"
        )

    cluster_to_sids = {}
    for sid in sid2ft:
        cluster_id = sid_to_cluster[sid]
        cluster_to_sids.setdefault(cluster_id, []).append(sid)
        #cluster_id というキーがまだ無ければ [] を入れる
        #そのリストに sid を追加する
    cluster_ids = list(cluster_to_sids.keys())
    rng = random.Random(seed)
    rng.shuffle(cluster_ids)

    target_sizes = list(split_sizes)
    split_sid_lists = [[] for _ in target_sizes]
    split_counts = [0 for _ in target_sizes]

    for cluster_id in cluster_ids: # なるべく同じサイズになるようにする
        sids = cluster_to_sids[cluster_id]
        remaining = [target - count for target, count in zip(target_sizes, split_counts)]
        best_split_idx = max(
            range(len(target_sizes)),
            key=lambda idx: (remaining[idx], -split_counts[idx]),
        )
        #ここが肝
        #まず remaining[idx] が大きい split を優先
        #つまり「まだ足りていない数が多い split」に入れる
        #同点なら -split_counts[idx] が大きい方
        #これは split_counts[idx] が小さい方を優先、という意味です
        #つまり今の時点で小さい split に寄せる
        #要するに、
        #目標に対して一番不足している split
        #同じなら現在サイズがより小さい split
        #にクラスタを入れます。
        split_sid_lists[best_split_idx].extend(sids)
        split_counts[best_split_idx] += len(sids)

    split_dicts = []
    for sid_list in split_sid_lists:
        split_dicts.append({sid: sid2ft[sid] for sid in sid_list})

    return split_dicts, split_counts, len(cluster_ids)


class Dataset:
    def __init__(self, sid2ft, emb_names):
        self.emb_names = emb_names

        tmp_data = []
        tmp_embs = [[] for _ in emb_names]
        tmp_target = []
        tmp_y_mask = []
        tmp_sid = list(sid2ft.keys())

        p = 0
        for sid in tmp_sid:
            tmp_data.append(sid2ft[sid]['oht'])
            for idx, emb_name in enumerate(self.emb_names):
                tmp_embs[idx].append(sid2ft[sid][emb_name])
            tmp_target.append(sid2ft[sid]['TE'])
            tmp_y_mask.append(sid2ft[sid]['y_mask'])
            p += 1
            if p % 1000 == 0:
                print(f"process:{p}", file=sys.stderr)

        self.data = torch.tensor(np.array(tmp_data), dtype=torch.float32)
        self.data = torch.transpose(self.data, 1, 2)
        self.embeddings = [
            torch.tensor(np.array(values), dtype=torch.float32) for values in tmp_embs
        ]
        self.target = torch.tensor(np.array(tmp_target), dtype=torch.float32)
        self.y_mask = torch.tensor(np.array(tmp_y_mask), dtype=torch.float32)
        self.sid = tmp_sid

    def __getitem__(self, index):
        return (
            self.data[index],
            *[emb[index] for emb in self.embeddings], # ここでマルチembeddingに対応している!
            self.target[index],
            self.y_mask[index],
            self.sid[index],
        )

    def __len__(self):
        return len(self.data)


def split_dataset(dataset, seed, split_sizes):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.random_split(dataset, split_sizes, generator=generator)


def load_model(num_embeddings):
    if not 0 <= num_embeddings <= 3:
        print("Only 0 to 3 embeddings are supported by the current model set.", file=sys.stderr)
        exit(1)
    return MyCNN_multiemb.CNN_GRU_multiemb


def forward_pass(dataloader, model, n_task, device, use_mse_loss, mode, *, optimizer=None):
    n_data = len(dataloader.dataset)

    loss_mean = 0.0
    pred = None
    obs = None
    y_mask = None

    if mode == 'Train':
        model.train()
    else:
        model.eval()
    model.gru.flatten_parameters()

    for batch in dataloader:
        *model_inputs, t, y_m, _sid = batch
        s = model_inputs[0].shape
        model_inputs = [tensor.to(device) for tensor in model_inputs]
        t = t.to(device)
        y_m = y_m.to(device)
        t = torch.nan_to_num(t, nan=0.0)

        y = model(*model_inputs)

        if use_mse_loss:
            loss = F.mse_loss(y, t, reduction='none')
        else:
            loss = F.l1_loss(y, t, reduction='none')
        masked_loss = loss * (1 - y_m)
        mean_masked_loss = masked_loss.sum() / (1 - y_m).sum()

        if mode == "Train":
            optimizer.zero_grad()
            mean_masked_loss.backward()
            optimizer.step()

        loss_mean += mean_masked_loss.item() * s[0] / n_data

        batch_pred = y.detach().cpu().numpy()
        batch_obs = t.detach().cpu().numpy()
        batch_y_mask = y_m.detach().cpu().numpy()
        pred = batch_pred if pred is None else np.concatenate((pred, batch_pred), axis=0)
        obs = batch_obs if obs is None else np.concatenate((obs, batch_obs), axis=0)
        y_mask = (
            batch_y_mask
            if y_mask is None
            else np.concatenate((y_mask, batch_y_mask), axis=0)
        )

    y_mask = (y_mask == 0).transpose()
    pred = pred.transpose()
    obs = obs.transpose()

    g_test_obs = []
    g_test_pred = []
    cor_list = []
    rho_list = []
    for i in range(n_task):
        cor_list.append(pearsonr(obs[i][y_mask[i]], pred[i][y_mask[i]])[0])
        rho_list.append(spearmanr(obs[i][y_mask[i]], pred[i][y_mask[i]])[0])
        g_test_obs.append(obs[i][y_mask[i]])
        g_test_pred.append(pred[i][y_mask[i]])
    g_test_obs_flat = np.concatenate(g_test_obs)
    g_test_pred_flat = np.concatenate(g_test_pred)
    g_cor = pearsonr(g_test_obs_flat, g_test_pred_flat)[0]
    g_rho = spearmanr(g_test_obs_flat, g_test_pred_flat)[0]

    task_valid_counts = y_mask.sum(axis=1)
    sample_valid_counts = y_mask.sum(axis=0)
    sample_obs_mean = (obs * y_mask).sum(axis=0) / sample_valid_counts
    sample_pred_mean = (pred * y_mask).sum(axis=0) / sample_valid_counts
    mean_te_cor = pearsonr(sample_obs_mean, sample_pred_mean)[0]

    return loss_mean, cor_list, rho_list, g_cor, g_rho, mean_te_cor, task_valid_counts


def train(dataloader, model, n_task, optimizer, device, use_mse_loss):
    loss_mean, cor_list, rho_list, g_cor, g_rho, mean_te_cor, task_valid_counts = forward_pass(
        dataloader,
        model,
        n_task,
        device,
        use_mse_loss,
        "Train",
        optimizer=optimizer,
    )
    train_loss_list.append(loss_mean)
    return loss_mean, cor_list, rho_list, g_cor, g_rho, mean_te_cor, task_valid_counts


def val(dataloader, model, n_task, device, use_mse_loss):
    loss_mean, cor_list, rho_list, g_cor, g_rho, mean_te_cor, task_valid_counts = forward_pass(
        dataloader,
        model,
        n_task,
        device,
        use_mse_loss,
        "Val",
    )
    val_loss_list.append(loss_mean)
    return loss_mean, cor_list, rho_list, g_cor, g_rho, mean_te_cor, task_valid_counts


def standardize_protein_length(sid2ft, train_ids, val_ids, test_ids, audit_path):
    """Replace the length control in memory; fit log1p mean/sample SD on train only.

    Exact, versioned transcript IDs are required. No targets or existing length
    embeddings are used. Validation completes before any feature is replaced.
    """
    groups = [list(train_ids), list(val_ids), list(test_ids)]
    all_ids = [sid for group in groups for sid in group]
    if any(not group for group in groups):
        raise ValueError("Protein-length standardization requires three nonempty splits")
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate or overlapping transcript IDs in splits")
    if not set(all_ids).issubset(sid2ft):
        raise ValueError("Split IDs absent from input features")
    if len(train_ids) < 2:
        raise ValueError("At least two training transcripts are required")

    lengths = {}
    with open(audit_path, newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            if row['passes_final_cohort'] != 'True':
                continue
            sid = row['transcript_id_original']
            if sid in lengths:
                raise ValueError(f"Duplicate transcript in audit: {sid}")
            length = int(row['protein_length'])
            if length <= 0:
                raise ValueError(f"Nonpositive protein length: {sid}")
            lengths[sid] = length
    missing = sorted(set(all_ids) - set(lengths))
    if missing:
        raise ValueError(f"Split IDs absent from retained audit cohort: {missing[:5]}")

    train_order = sorted(train_ids)
    train_values = [math.log1p(lengths[sid]) for sid in train_order]
    mean = statistics.fmean(train_values)
    sd = statistics.stdev(train_values)
    if not math.isfinite(sd) or sd <= 0:
        raise ValueError("Training log protein lengths have zero or invalid SD")
    transformed = {
        sid: np.asarray([(math.log1p(lengths[sid]) - mean) / sd], dtype=np.float32)
        for sid in all_ids
    }
    metadata = {
        'feature_name': 'emb_protein_length',
        'fit_subset': 'train',
        'source': 'raw protein_length from transcript audit; existing embedding ignored',
        'formula': '(log1p(protein_length_aa) - training_mean) / training_sample_sd',
        'log1p_length_mean': mean,
        'log1p_length_sample_sd': sd,
        'std_ddof': 1,
        'n_train': len(train_ids), 'n_validation': len(val_ids), 'n_test': len(test_ids),
        'training_ids_sha256': hashlib.sha256(
            ('\n'.join(train_order) + '\n').encode()).hexdigest(),
        'transcript_audit': str(Path(audit_path).resolve()),
        'transcript_audit_sha256': hashlib.sha256(Path(audit_path).read_bytes()).hexdigest(),
    }
    for sid, value in transformed.items():
        sid2ft[sid]['emb_protein_length'] = value
    return metadata


def save_model(best_model, cnn_settings, n_task, model_path, preprocessing=None):
    save_model_dict = {
        'model_state_dict': best_model.state_dict(),
        'cnn_settings': cnn_settings,
        'n_task': n_task,
    }
    if preprocessing is not None:
        save_model_dict['preprocessing'] = preprocessing
    torch.save(save_model_dict, model_path)


def main(args):
    length_audit = getattr(args, 'protein_length_audit', None)
    if length_audit:
        if args.emb_name != ['emb_protein_length'] or not args.input_class_fname:
            raise ValueError('--protein-length-audit requires --emb_name emb_protein_length '
                             'and --input_class_fname (fixed matched splits)')
        if not args.model_fname:
            raise ValueError('--protein-length-audit requires --model_fname to preserve the scaler')
        for output in [args.model_fname, args.model_fname + '.preprocessing.json']:
            if Path(output).exists():
                raise FileExistsError(f'Use a new run directory; output already exists: {output}')
    preprocessing = None
    if not 0 <= len(args.emb_name) <= 3:
        print("--emb_name accepts 0 to 3 values.", file=sys.stderr)
        exit(1)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("mps" if torch.backends.mps.is_available() else device)
    print(f"device={device}", file=sys.stderr)

    with gzip.open(args.ft_gz, 'rb') as f:
        sid2ft, TE_cols = pickle.load(f)

    sid2ft = limit_dataset_size(sid2ft, args.max_data)
    if args.max_data is not None:
        print(f"Number of data after filtering: {len(sid2ft)}", file=sys.stderr)

    length = check_same_len(sid2ft)
    n_task = check_y(sid2ft)
    print("Number of task:", n_task, file=sys.stderr)

    emb_dims = [check_emb_dim(sid2ft, emb_name) for emb_name in args.emb_name]
    for idx, emb_dim in enumerate(emb_dims, start=1):
        print(f"Dimension of embedding {idx}:", emb_dim, file=sys.stderr)

    n_train = int(len(sid2ft) * 0.6)
    n_val = int(len(sid2ft) * 0.2)
    n_test = len(sid2ft) - n_train - n_val
    print(n_train, n_val, n_test, file=sys.stderr)

    # データのsplit方法は３つある
    if args.input_class_fname: # (1)テキストで指定された時
        train_set, val_set, test_set = read_class_file(args.input_class_fname)
        if length_audit:
            scaler = standardize_protein_length(
                sid2ft, train_set, val_set, test_set, length_audit)
            scaler['seed'] = args.seed
            scaler['class_file_sha256'] = hashlib.sha256(
                Path(args.input_class_fname).read_bytes()).hexdigest()
            preprocessing = {'protein_length': scaler}
            model_path = Path(args.model_fname)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path = Path(str(model_path) + '.preprocessing.json')
            with metadata_path.open('x') as handle:
                json.dump(preprocessing, handle, indent=2, allow_nan=False)
                handle.write('\n')
            print('Protein length standardized using training set only: '
                  f"n={scaler['n_train']} mean={scaler['log1p_length_mean']:.12g} "
                  f"sample_sd={scaler['log1p_length_sample_sd']:.12g}", file=sys.stderr)
        sid2ft_train = {k: sid2ft[k] for k in train_set}
        sid2ft_val = {k: sid2ft[k] for k in val_set}
        sid2ft_test = {k: sid2ft[k] for k in test_set}
        d_train = Dataset(sid2ft_train, args.emb_name)
        d_val = Dataset(sid2ft_val, args.emb_name)
        d_test = Dataset(sid2ft_test, args.emb_name)
    elif args.cd_hit_clstr: # (2)クラスター単位⭐️
        sid_to_cluster = read_cd_hit_clstr(args.cd_hit_clstr)
        split_dicts, split_counts, n_clusters = split_sid2ft_by_cluster(
            sid2ft,
            sid_to_cluster,
            [n_train, n_val, n_test],
            args.seed,
        )
        sid2ft_train, sid2ft_val, sid2ft_test = split_dicts
        print(f"Number of clusters: {n_clusters}", file=sys.stderr)
        print(
            "Cluster-based split sizes:",
            split_counts[0],
            split_counts[1],
            split_counts[2],
            file=sys.stderr,
        )
        d_train = Dataset(sid2ft_train, args.emb_name)
        d_val = Dataset(sid2ft_val, args.emb_name)
        d_test = Dataset(sid2ft_test, args.emb_name)
    else: # ランダム分割
        d = Dataset(sid2ft, args.emb_name)
        # この処理ではDatasetではなくSubsetができる
        d_train, d_val, d_test = split_dataset(d, args.seed, [n_train, n_val, n_test])

    if args.out_class_fname:
        make_class_file(d_train, d_val, d_test, args.out_class_fname)

    train_dataloader = DataLoader(d_train, batch_size=args.s_bat, shuffle=True, generator=g) # DataLoaderにはDatasetもSubsetも渡せる
    val_dataloader = DataLoader(d_val, batch_size=args.s_bat, shuffle=False, generator=g)    # __len__()と__getitem__()があれば渡せる
    test_dataloader = DataLoader(d_test, batch_size=args.s_bat, shuffle=False, generator=g)

    cnn_settings = {}
    cnn_settings['in_ch'] = [args.in_dim] + [64 for _ in range(10)]
    cnn_settings['out_ch'] = [64 for _ in range(11)]
    cnn_settings['ker'] = [5 for _ in range(11)]
    cnn_settings['pad'] = [0 for _ in range(11)]
    cnn_settings['stp'] = [1 for _ in range(11)]
    cnn_settings['dil'] = [1 for _ in range(11)]

    model_class = load_model(len(args.emb_name))
    model = model_class(
        cnn_settings,
        emb_dims,
        n_task,
        length,
        device=device,
        a_type=args.abl_type,
        a_dim=args.abl_dim,
    ).to(device)

    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print("number of param.", params, file=sys.stderr)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_model = None
    best_val_loss = 1e100
    best_val_epoch = -1

    for epoch in range(args.epoch):
        print(f"Epoch={epoch}", file=sys.stderr)

        loss_mean, cor_list, rho_list, g_cor, g_rho, mean_te_cor, _ = train(
            train_dataloader, model, n_task, optimizer, device, args.mse_loss
        )
        print(
            f"train: loss={loss_mean}",
            np.mean(g_cor),
            np.mean(g_rho),
            f"mean_te_cor={mean_te_cor}",
            file=sys.stderr,
        )

        loss_mean, cor_list, rho_list, g_cor, g_rho, mean_te_cor, _ = val(
            val_dataloader, model, n_task, device, args.mse_loss
        )
        print(
            f"valid: loss={loss_mean}",
            np.mean(g_cor),
            np.mean(g_rho),
            f"mean_te_cor={mean_te_cor}",
            file=sys.stderr,
        )

        if loss_mean < best_val_loss:
            best_val_loss = loss_mean
            best_model = copy.deepcopy(model)
            best_val_epoch = epoch
            if args.model_fname:
                print(f"New model is saved at epoch{epoch}", file=sys.stderr)
                save_model(best_model, cnn_settings, n_task, args.model_fname, preprocessing)

    if best_model is None:
        best_model = copy.deepcopy(model)

    loss_mean, cor_list, rho_list, g_cor, g_rho, mean_te_cor, task_valid_counts = forward_pass(
        test_dataloader, best_model, n_task, device, args.mse_loss, "Test"
    )
    print("#best_val_epoch:", best_val_epoch)
    print(f"#test: loss={loss_mean}", np.mean(g_cor), np.mean(g_rho), f"mean_te_cor={mean_te_cor}")
    print(f"#test_mean_te_cor: {mean_te_cor}")
    print("#best_val_epoch:", best_val_epoch, file=sys.stderr)
    print(
        f"#test: loss={loss_mean}",
        np.mean(g_cor),
        np.mean(g_rho),
        f"mean_te_cor={mean_te_cor}",
        file=sys.stderr,
    )
    print(f"#test_mean_te_cor: {mean_te_cor}", file=sys.stderr)

    print_per_tissue_metrics(TE_cols, cor_list, rho_list)

    if args.metrics_tsv:
        metrics_path = Path(args.metrics_tsv)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open('w', newline='') as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=['tissue', 'pearson', 'spearman', 'n_test'],
                delimiter='\t',
            )
            writer.writeheader()
            for tissue, cor, rho, count in zip(
                TE_cols, cor_list, rho_list, task_valid_counts
            ):
                writer.writerow({
                    'tissue': tissue,
                    'pearson': cor,
                    'spearman': rho,
                    'n_test': int(count),
                })


if __name__ == '__main__':
    dt_start = datetime.datetime.now()
    print(f'Started:{dt_start}', file=sys.stderr)

    parser = argparse.ArgumentParser()
    parser.add_argument('ft_gz', help='ft.pickle.gz file')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--in_dim', type=int, default=6, help='dimension of each vector in input vector sequence')
    parser.add_argument('--epoch', help='maximum epoch', default=100, type=int)
    parser.add_argument('--mse_loss', action='store_true', help='use MSE loss')
    #parser.add_argument('--mlp', action='store_true', help='use CNN-MLP model')
    parser.add_argument('--s_bat', type=int, default=100, help='batch size')
    parser.add_argument('--out_class_fname', help='classification file name', default="class.txt", type=str)
    parser.add_argument('--input_class_fname', help='classification file name', type=str)
    parser.add_argument('--cd_hit_clstr', help='CD-HIT .clstr file for cluster-aware split', type=str)
    parser.add_argument('--model_fname', help='model file name', default="model_CNN.pth", type=str)
    parser.add_argument('--abl_type', help='abl_type', choices=['v', 'm', 'p'], type=str)
    parser.add_argument('--abl_dim', nargs='*', help='abl_dim', type=int)
    parser.add_argument(
        '--emb_name',
        nargs='*',
        default=[],
        type=str,
        help='zero to three embedding names; omit values for the mRNA-only model',
    )
    parser.add_argument('--max_data', type=int, help='limit the number of samples before splitting')
    parser.add_argument('--seed', default=42, help='random seed', type=int)
    parser.add_argument('--metrics_tsv', help='write per-tissue test metrics as TSV', type=str)
    parser.add_argument('--protein-length-audit', type=str,
                        help='read raw lengths from this transcript audit TSV; fit log1p '
                             'standardization on training IDs only (length control with fixed splits)')

    args = parser.parse_args()

    main(args)

    dt_end = datetime.datetime.now()
    print(f'Finished:{dt_end}', file=sys.stderr)
    print("Running time is", dt_end - dt_start, file=sys.stderr)
