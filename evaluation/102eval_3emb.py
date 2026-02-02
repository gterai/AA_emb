# -*- coding: utf-8 -*-

import os
import sys
import argparse
sys.path.append(os.environ['HOME'] + "/pyscript")
import basic
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from scipy.stats import pearsonr, spearmanr
import datetime

import MyCNN_3emb
import random
import gzip
import copy

train_loss_list = []
val_loss_list = []
test_loss_list = []

#train_cor_list = []
#train_rho_list = []
#train_mean_cor_list = []
#train_mean_rho_list = []
#val_cor_list = []
#val_rho_list = []
#val_mean_cor_list = []
#val_mean_rho_list = []
#test_cor_list = []
#test_rho_list = []
#test_mean_cor_list = []
#test_mean_rho_list = []

def read_class_file(fname):
    tra_set = set()
    val_set = set()
    tes_set = set()

    with open(fname) as f:
        for line in f:
            line = line.replace('\n','')
            sid, stype = line.split()

            if stype == "train":
                tra_set.add(sid)
            elif stype == "validation":
                val_set.add(sid)
            elif stype == "test":
                tes_set.add(sid)
            else:
                print("Unknown type (line)")
                exit(1)
    f.close()

    return tra_set, val_set, tes_set

def make_class_file(d_tra, d_val, d_tes, fname):

    tra_list = [d_tra[i][2] for i in range(len(d_tra))]
    val_list = [d_val[i][2] for i in range(len(d_val))]
    tes_list = [d_tes[i][2] for i in range(len(d_tes))]

    fout = open(fname, 'w')
    for g in tra_list:
        print(g, "train", file=fout)
    for g in val_list:
        print(g, "validation", file=fout)
    for g in tes_list:
        print(g, "test", file=fout)
    fout.close()

def checkEmbDim(sid2ft:dict, emb_name:str):
    # embedding次元のチェック
    
    sid_list = list(sid2ft.keys())
    
    emb_dim = len(sid2ft[sid_list[0]][emb_name])
    for sid in sid_list[1:]:

        if emb_dim != len(sid2ft[sid][emb_name]):
            print(f"Dimension of embedding is different", file=sys.stderr)
            exit(0)

    return emb_dim

def main(args: dict):

    # seedの固定
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    #np.random.seed(0) # 外しても再現する
    #torch.cuda.manual_seed_all(0) # 外しても再現する
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    # cuDNN
    torch.backends.cudnn.deterministic = True # 結果の再現に必要
    torch.backends.cudnn.benchmark = False # 結果の再現に必要
    
    #torch.autograd.set_detect_anomaly(True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("mps" if torch.backends.mps.is_available() else device) # for macOS
    print(f"device={device}", file=sys.stderr)
    
    f = gzip.open(args.ft_gz,'rb')
    (sid2ft, TE_cols) = pickle.load(f)
    f.close

    """
    ## テスト用にデータ数を減らす
    sid_list = list(sid2ft.keys())
    random.shuffle(sid_list)
    tmp_d = {}
    for sid in sid_list[:1000]:
        tmp_d[sid] = sid2ft[sid]
    sid2ft = tmp_d
    """

    # 長さのチェック(同じ長さになるはず)
    L = checkSameLen(sid2ft)
    
    # マルチ数のチェック
    n_task = checkY(sid2ft)
    print(f"Number of task:", n_task, file=sys.stderr)
    
    # embedding次元のチェック
    emb_dim_1 = checkEmbDim(sid2ft, args.emb_name_1)
    emb_dim_2 = checkEmbDim(sid2ft, args.emb_name_2)
    emb_dim_3 = checkEmbDim(sid2ft, args.emb_name_3)
    print(f"Dimension of embedding 1:", emb_dim_1, file=sys.stderr)
    print(f"Dimension of embedding 2:", emb_dim_2, file=sys.stderr)
    print(f"Dimension of embedding 3:", emb_dim_3, file=sys.stderr)
    #exit(0)
    
    # train : validation: test = 60%: 20%: 20%
    n_train = int(len(sid2ft) * 0.6)
    n_val = int(len(sid2ft) * 0.2)
    n_test = len(sid2ft) - n_train - n_val
    print(n_train, n_val, n_test, file=sys.stderr)
    # data loaderの作成

    if args.input_class_fname:
        train_set, val_set, test_set = read_class_file(args.input_class_fname)
        sid2ft_train = {k: sid2ft[k] for k in train_set}
        sid2ft_val   = {k: sid2ft[k] for k in val_set}
        sid2ft_test  = {k: sid2ft[k] for k in test_set}
        d_train = Dataset(sid2ft_train, args.emb_name_1, args.emb_name_2, args.emb_name_3)
        d_val   = Dataset(sid2ft_val, args.emb_name_1, args.emb_name_2, args.emb_name_3)
        d_test  = Dataset(sid2ft_test, args.emb_name_1, args.emb_name_2, args.emb_name_3)
    else:
        d = Dataset(sid2ft, args.emb_name_1, args.emb_name_2, args.emb_name_3)
        d_train, d_val, d_test = torch.utils.data.random_split(d, [n_train, n_val, n_test], generator=g)

    if args.out_class_fname:
        make_class_file(d_train, d_val, d_test, args.out_class_fname)
        
    train_dataloader = DataLoader(d_train, batch_size=args.s_bat, shuffle=True, generator=g)
    val_dataloader = DataLoader(d_val, batch_size=args.s_bat, shuffle=False, generator=g)
    test_dataloader = DataLoader(d_test, batch_size=args.s_bat, shuffle=False, generator=g)

    # CNNの設定
    ###
    cnn_settings = {}
    cnn_settings['in_ch']      = [args.in_dim] + [64 for i in range(10)]
    cnn_settings['out_ch']     = [64 for i in range(11)]
    cnn_settings['ker']        = [5  for i in range(11)]
    cnn_settings['pad']        = [0  for i in range(11)]
    cnn_settings['stp']        = [1  for i in range(11)]
    cnn_settings['dil']        = [1  for i in range(11)]
    
    #else:
    model = MyCNN_3emb.CNN_GRU_3emb(cnn_settings, emb_dim_1, emb_dim_2, emb_dim_3, n_task, L, device=device,
                                  a_type=args.abl_type, a_dim=args.abl_dim).to(device)  
    # パラメータカウント
    if(1):
        params = 0
        for p in model.parameters():
            if p.requires_grad:
                params += p.numel()
        print(f"number of param.", params, file=sys.stderr)  
        #exit(0)


    optimizer = optim.Adam(model.parameters(), lr=args.lr) #

    best_model = ''
    best_val_loss = 1e100
    best_val_epoch = -1
    
    #global best_val_epoch
    #global best_val_loss
    #global best_model
    
    # main loop
    for epoch in range(args.epoch):
        print(f"Epoch={epoch}", file=sys.stderr)
        
        # train
        loss_mean, cor_list, rho_list, g_cor, g_rho = train(train_dataloader, model, n_task, optimizer, device)
        print(f"train: loss={loss_mean}", np.mean(g_cor), np.mean(g_rho), file=sys.stderr)

        # validation
        loss_mean, cor_list, rho_list, g_cor, g_rho = val(val_dataloader, model, n_task, device)
        print(f"valid: loss={loss_mean}", np.mean(g_cor), np.mean(g_rho), file=sys.stderr)
        #print(g_cor)

        if loss_mean < best_val_loss:
            best_val_loss = loss_mean
            best_model = copy.deepcopy(model)
            best_val_epoch = epoch
            if args.model_fname:
                print(f"New model is saved at epoch{epoch}", file=sys.stderr)
                save_model(best_model, cnn_settings, n_task, args.model_fname)

    # test
    loss_mean, cor_list, rho_list, g_cor, g_rho, = forward_pass(test_dataloader, best_model, n_task, device, "Test")
    print(f"#best_val_epoch:", best_val_epoch)
    print(f"#test: loss={loss_mean}", np.mean(g_cor), np.mean(g_rho))
    print(f"#best_val_epoch:", best_val_epoch, file=sys.stderr)
    print(f"#test: loss={loss_mean}", np.mean(g_cor), np.mean(g_rho), file=sys.stderr)
    
    for i,cor in enumerate(cor_list):
        print(TE_cols[i], cor)
    
    
def drawLossAndCor():
    plt.plot(train_loss_list, label="train")
    #if args.train_only:
    plt.plot(test_loss_list, label="test")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig(f"{args.pref}TF_loss.png")
    plt.close()

    plt.plot(train_mean_cor_list, label="train")
    plt.xlabel("epoch")
    plt.ylabel("pearson correlation")
    #if args.train_only:
    plt.plot(test_mean_cor_list, label="test")
    plt.legend()
    plt.savefig(f"{args.pref}TF_cor.png")
    plt.close()

    plt.plot(train_mean_rho_list, label="train")
    #if args.train_only:
    plt.plot(test_mean_rho_list, label="test")
    plt.xlabel("epoch")
    plt.ylabel("spearman correlation")
    plt.legend()
    plt.savefig(f"{args.pref}TF_rho.png")
    plt.close()

    #if args.train_only:
    max_mean_cor = max(test_mean_cor_list)
    max_mean_rho = max(test_mean_rho_list)
    print("max mean cor=", max_mean_cor)
    print("max mean rho=", max_mean_rho)
    print("epoch of best test loss=", best_test_epoch)
    print("cor of best test loss=", test_mean_cor_list[best_test_epoch])
    print("rho of best test loss=", test_mean_rho_list[best_test_epoch])

    
def forward_pass(dataloader, model, n_task, device, mode, *, optimizer=None):
    n_data = len(dataloader.dataset)
        
    sid_exec_list = []
    loss_mean = 0
    if 'pred' in locals():
        del pred
    if 'obs' in locals():
        del obs
    if 'y_mask' in locals():
        del y_mask

    if mode == 'Train':
        model.train()
    else:
        model.eval()
    model.gru.flatten_parameters() 

    for (x, x2, x3, x4, t, y_m, n) in dataloader:
        s = x.shape # batch x max_seq_len x oht(15 in this case)
        x = x.to(device) # batch x 配列長 x 次元
        x2 = x2.to(device) # batch x 配列長 x 次元
        x3 = x3.to(device) # batch x 配列長 x 次元
        x4 = x4.to(device) # batch x 配列長 x 次元
        t = t.to(device)
        sid_exec_list += n
        y_m = y_m.to(device)
        t = torch.nan_to_num(t, nan=0.0)

        y = model(x, x2, x3, x4)

        if args.mse_loss:
            loss = F.mse_loss(y,t, reduction='none')            
        else:
            loss = F.l1_loss(y,t, reduction='none')
        masked_loss = loss * (1 - y_m) # maskが1で実装されているので、0/1を入れ替える
        #print(masked_loss)
        #exit(0)
        #print(masked_loss.shape)
        #print(masked_loss.sum())
        #print(y_m.sum())
        #exit(0)
        mean_masked_loss = masked_loss.sum() / (1-y_m).sum()
        #mean_masked_loss = masked_loss.mean() # 全てが重み１
        
        if mode == "Train":
            optimizer.zero_grad()
            mean_masked_loss.backward()
            optimizer.step()
            #scheduler.step()  # ステップごとにschedulerを更新

        loss_mean += mean_masked_loss.item() * s[0]/n_data # ここで重み付きの平均をとってみる。

        if 'pred' not in locals():
            pred = y.detach().cpu().numpy()
        else:
            pred = np.concatenate((pred, y.detach().cpu().numpy()), axis=0)
        if 'obs' not in locals():
            obs = t.detach().cpu().numpy()
        else:
            obs = np.concatenate((obs, t.detach().cpu().numpy()), axis=0)
        if 'y_mask' not in locals():
            y_mask = y_m.detach().cpu().numpy()
        else:
            y_mask = np.concatenate((y_mask, y_m.detach().cpu().numpy()), axis=0)
            
    y_mask = (y_mask == 0)
    y_mask = y_mask.transpose()
    pred = pred.transpose()
    obs  = obs.transpose()

    g_test_obs = []
    g_test_pred = []
    cor_list = []
    rho_list = []
    for i in range(n_task):
        cor_list.append (pearsonr(obs[i][y_mask[i]], pred[i][y_mask[i]])[0])
        rho_list.append (spearmanr(obs[i][y_mask[i]], pred[i][y_mask[i]])[0])
        g_test_obs.append(obs[i][y_mask[i]])
        g_test_pred.append(pred[i][y_mask[i]])
    g_test_obs_flat = np.concatenate(g_test_obs)
    g_test_pred_flat = np.concatenate(g_test_pred)
    g_cor = pearsonr(g_test_obs_flat, g_test_pred_flat)[0]
    g_rho = spearmanr(g_test_obs_flat, g_test_pred_flat)[0]

    #if mode == "Test":
    #    return loss_mean, g_cor, g_rho, pred, obs, sid_exec_list
    #else:
    #    return loss_mean, g_cor, g_rho
    return loss_mean, cor_list, rho_list, g_cor, g_rho, 

def train(dataloader, model, n_task, optimizer, device):

    loss_mean, cor_list, rho_list, g_cor, g_rho = forward_pass(dataloader, model, n_task, device, "Train", optimizer=optimizer)

    train_loss_list.append(loss_mean)
    #train_cor_list.append(cor)
    #train_mean_cor_list.append(np.mean(cor))
    #train_rho_list.append(rho)
    #train_mean_rho_list.append(np.mean(rho))
    
    return loss_mean, cor_list, rho_list, g_cor, g_rho

def val(dataloader, model, n_task, device):

    loss_mean, cor_list, rho_list, g_cor, g_rho = forward_pass(dataloader, model, n_task, device, "Val")
    
    val_loss_list.append(loss_mean)
    #val_cor_list.append(cor)
    #val_mean_cor_list.append(np.mean(cor))
    #val_rho_list.append(rho)
    #val_mean_rho_list.append(np.mean(rho))

    #if loss_mean < best_val_loss:
    #    best_val_loss = loss_mean
    #    best_model = model
    #    best_val_epoch = epoch
    #    if args.model_fname:
    #        print(f"New model is saved at epoch{epoch}")
    #        save_model(best_model,cnn_settings, n_task, args.model_fname)

    return loss_mean, cor_list, rho_list, g_cor, g_rho

    
class Dataset:
    def __init__(self, sid2ft, emb_name_1, emb_name_2, emb_name_3):
        self.emb_name_1 = emb_name_1
        self.emb_name_2 = emb_name_2
        self.emb_name_3 = emb_name_3
        
        _tmp_data = []
        _tmp_data2 = []
        _tmp_data3 = []
        _tmp_data4 = []
        _tmp_x_mask = []
        _tmp_target = []
        _tmp_y_mask = []
        _tmp_sid = list(sid2ft.keys())
        
        #for sid in sid2ft.keys():
        #    print(sid, sid2ft[sid]['TE'].shape)
        #    exit(0)

        p = 0
        for sid in _tmp_sid:
            _tmp_data.append(sid2ft[sid]['oht']) # 説明変数
            _tmp_data2.append(sid2ft[sid][self.emb_name_1]) # 説明変数
            _tmp_data3.append(sid2ft[sid][self.emb_name_2]) # 説明変数
            _tmp_data4.append(sid2ft[sid][self.emb_name_3]) # 説明変数
            _tmp_x_mask.append(sid2ft[sid]['x_mask']) # paddingマスク
            _tmp_target.append(sid2ft[sid]['TE'])  # 目的変数
            _tmp_y_mask.append(sid2ft[sid]['y_mask']) # paddingマスク
            p += 1 
            if p % 1000 == 0: print(f"process:{p}", file=sys.stderr)
    
        self.data   = torch.tensor(np.array(_tmp_data), dtype=torch.float32)
        self.data   = torch.transpose(self.data, 1, 2) # 何度のtransposeするのは嫌なので、ここで一度だけ行う。
        self.data2  = torch.tensor(np.array(_tmp_data2), dtype=torch.float32)
        self.data3  = torch.tensor(np.array(_tmp_data3), dtype=torch.float32)
        self.data4  = torch.tensor(np.array(_tmp_data4), dtype=torch.float32)
        #self.x_mask = torch.tensor(np.array(_tmp_x_mask), dtype=torch.float32)
        self.target = torch.tensor(np.array(_tmp_target), dtype=torch.float32)
        self.y_mask = torch.tensor(np.array(_tmp_y_mask), dtype=torch.float32)
        self.sid = _tmp_sid
        #print(self.target.shape)
        #exit(0)
    def __getitem__(self, index):
        return self.data[index], self.data2[index], self.data3[index], self.data4[index], self.target[index], self.y_mask[index], self.sid[index]
    
    def __len__(self):
        return len(self.data)

    
def checkSameLen(sid2ft:dict):
    sid_list = list(sid2ft.keys())
    #print(list(sid2ft[sid_list[0]].keys()))
    #exit(0)
    L = len(sid2ft[sid_list[0]]['oht'])
    
    for sid in sid_list:
        ft_len = len(sid2ft[sid]['oht'])
        if ft_len != L:
            print(f"The length of feature {ft_len} is different from the indicated length {L}", file=sys.stderr)
            exit(0)
    return L

def checkY(sid2ft:dict):
    sid_list = list(sid2ft.keys())
    
    y_len = len(sid2ft[sid_list[0]]['y_mask'])
    for sid in sid_list[1:]:

        if y_len != len(sid2ft[sid]['y_mask']):
            print(f"Number of task is different", file=sys.stderr)
            exit(0)

        y_len = len(sid2ft[sid]['y_mask'])
    
    return y_len

def readType(fname: str):
    d = {}
    with open(fname) as f:
        for line in f:
            line = line.replace('\n','')
            items = line.split()
            gid, type = items[0], items[1]
            d[gid] = type
    return d

def save_model(best_model, cnn_settings, n_task, model_path):
    #save_model(best_model,cnn_settings, n_task, L, args.model_fname)
    save_model_dict = {
        'model_state_dict':best_model.state_dict(),
        'cnn_settings':cnn_settings,
        'n_task':n_task,
    }
    torch.save(save_model_dict, model_path)


# 空のデータセットを定義
class EmptyDataset(Dataset):
    def __len__(self):
        return 0  # データがない
    def __getitem__(self, idx):
        raise IndexError("This dataset is empty.")  # 取得できるデータがない


if __name__ == '__main__':

    dt_start = datetime.datetime.now()
    print(f'Started:{dt_start}', file=sys.stderr)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('ft_gz', help='ft.pickle.gz file')
    #parser.add_argument('--pref', default='', help='prefix for output files')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--in_dim', type=int, default=6, help='dimension of each vector in input vector sequence')
    parser.add_argument('--epoch', help='maximum epoch', default=100, type=int)
    parser.add_argument('--mse_loss', action='store_true', help='use MAE loss')
    parser.add_argument('--mlp', action='store_true', help='use CNN-MLP model')
    parser.add_argument('--s_bat', type=int, default=100, help='batch size')
    parser.add_argument('--out_class_fname', help='classification file name', default="class.txt", type=str)
    parser.add_argument('--input_class_fname', help='classification file name', type=str)
    parser.add_argument('--model_fname', help='model file name', default="model_CNN.pth", type=str)
    parser.add_argument('--abl_type', help='abl_type', choices=['v','m','p'],type=str)
    parser.add_argument('--abl_dim', nargs='*' , help='abl_dim', type=int)
    parser.add_argument('--emb_name_1', help='embedding name 1',
                        choices=['emb_esm2', 'emb_esm2L', 'emb_T5u', 'emb_T5b', 'emb_ank', 'emb_ank3'],
                        required=True,
                        type=str)
    parser.add_argument('--emb_name_2', help='embedding name 2',
                        choices=['emb_esm2', 'emb_esm2L', 'emb_T5u', 'emb_T5b', 'emb_ank', 'emb_ank3'],
                        required=True,
                        type=str)
    parser.add_argument('--emb_name_3', help='embedding name 3',
                        choices=['emb_esm2', 'emb_esm2L', 'emb_T5u', 'emb_T5b', 'emb_ank', 'emb_ank3'],
                        required=True,
                        type=str)
    parser.add_argument('--seed', default=42 , help='random seed', type=int)
    
    args = parser.parse_args()
    
    main(args)
    
    dt_end = datetime.datetime.now()
    print(f'Finished:{dt_end}', file=sys.stderr)

    run_time = dt_end-dt_start
    print("Running time is", run_time, file=sys.stderr)
    
