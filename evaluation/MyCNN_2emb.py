import sys
import os
import argparse
sys.path.append(os.environ['HOME'] + "/pyscript")
import basic
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ① 転置専用のモジュール
class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

    
class CNN_GRU_2emb(nn.Module):
    def __init__(self, settings:dict, emb_dim1:int, emb_dim2:int, n_task:int, L: int, device='cpu', *, a_type=None, a_dim=None): # seq_lenは使わないが、対称性のために受け取る
        super().__init__()
        self.device=device
        s = settings

        self.a_type = a_type
        self.a_dim = a_dim
        
        # CNNの最後の出力ベクトルの長さを求める
        cnn_out_len = self.calcCnnInterLen(L, s)
        print(cnn_out_len, file=sys.stderr)
        self.init_layer = nn.Conv1d(s['in_ch'][0], s['out_ch'][0], s['ker'][0], padding=s['pad'][0], stride=s['stp'][0], dilation=s['dil'][0])

        other_layers = []
        for i in range(1,len(s['in_ch'])):
            other_layers.append(nn.Conv1d(s['in_ch'][i], s['out_ch'][i], s['ker'][i], padding=s['pad'][i], stride=s['stp'][i], dilation=s['dil'][i]))
            other_layers.append(nn.ReLU())
            other_layers.append(Transpose(1,2))
            other_layers.append(nn.LayerNorm(s['out_ch'][i]))
            other_layers.append(Transpose(1,2))
            other_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            other_layers.append(nn.Dropout(0.2))
        self.other_layers = nn.Sequential(*other_layers)
        self.gru = nn.GRU(input_size = s['out_ch'][-1], # CNN最終層のチャネル数
                          hidden_size = s['out_ch'][-1], # CNN最終層のチャネル数
                          num_layers=3,
                          dropout=0.2,
                          batch_first = True)
        #self.fc1 = nn.Linear(s['out_ch'][-1] + emb_dim, 1024)
        #self.LN1 = nn.LayerNorm(1024)
        #self.fc2 = nn.Linear(1024, 128)
        #self.LN2 = nn.LayerNorm(128)
        #self.fc3 = nn.Linear(128, n_task)
        #self.drop = nn.Dropout(0.3)

        #self.FC = nn.Linear(s['out_ch'][-1] + emb_dim, 1)
        self.FC64_1 = nn.Linear(emb_dim1, s['out_ch'][-1])
        self.FC64_2 = nn.Linear(emb_dim2, s['out_ch'][-1])
        #self.FC = nn.Linear(s['out_ch'][-1]*2, n_task)
        self.FC = nn.Linear(s['out_ch'][-1]*3, n_task)
        
    def calcCnnInterLen(self, seq_len:int, s:dict): # このアーキテクチャにしか使えない計算プログラム！
        for i in range(len(s['in_ch'])):
            #print(i)
            seq_len = int(((seq_len + 2 * s['pad'][i] - s['dil'][i] * (s['ker'][i] - 1) - 1)/s['stp'][i]) + 1)
            if i != 0: # 最初のレイヤーは、maxpoolingを行わない。
                seq_len = int((seq_len - (2 - 1) - 1)/2 + 1) # kernel size=2, step=2のmaxpooling # この計算がわかっていない。。。
            #print(i, seq_len)
        return seq_len * s['out_ch'][-1] # 配列長 x 最後レイヤのチャネル数

    def forward(self, x, x2, x3):
        
        #if self.a_type == 'v': # oht-likeの各次元のablation
        #    x[:, self.a_dim, :] = 0 # 特定のdimentionを0 mask
              
        s = x.shape
        x = self.init_layer(x)
        x = self.other_layers(x)
        x = torch.transpose(x, 1, 2)
        x = self.gru(x)

        # 一番最後の隠れ状態を取る
        # for GRU
        s_out = x[1][-1].shape # layer x batch x dim
        x = x[1][-1].view(s_out[0], s_out[1])
        # 隠れ状態の平均を取る
        #x = torch.mean(x[0], dim=1)
        #print(x.shape)
        #exit(0)
        
        # 平均ベクトルにしてみる
        #print(x[0].shape, x[1].shape) # x[0]にすべての隠れ状態が入っている。
        #exit(0)
        #mean = torch.mean(x[0]
        #print(x.shape)
        #print(x2.shape)
        #exit(0)
        x2 = self.FC64_1(x2)
        x2 = torch.relu(x2)
        
        x3 = self.FC64_2(x3)
        x3 = torch.relu(x3)
        
        #if self.a_type == 'm':
        #    x  = torch.zeros_like(x)  # ablation of mRNA information
        #elif self.a_type == 'p':
        #    x2 = torch.zeros_like(x2) # ablation of protein embedding
            
        xx = torch.cat((x, x2, x3), dim=1)
        return self.FC(xx)
    
