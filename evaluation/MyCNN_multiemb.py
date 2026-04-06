import sys
import os

sys.path.append(os.environ['HOME'] + "/pyscript")

import torch
import torch.nn as nn


class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class CNN_GRU_multiemb(nn.Module):
    def __init__(self, settings: dict, emb_dims, n_task: int, L: int, device='cpu', *, a_type=None, a_dim=None):
        super().__init__()
        self.device = device
        self.a_type = a_type
        self.a_dim = a_dim
        self.emb_dims = list(emb_dims)
        s = settings

        cnn_out_len = self.calcCnnInterLen(L, s)
        print(cnn_out_len, file=sys.stderr)
        self.init_layer = nn.Conv1d(
            s['in_ch'][0],
            s['out_ch'][0],
            s['ker'][0],
            padding=s['pad'][0],
            stride=s['stp'][0],
            dilation=s['dil'][0],
        )

        other_layers = []
        for i in range(1, len(s['in_ch'])):
            other_layers.append(
                nn.Conv1d(
                    s['in_ch'][i],
                    s['out_ch'][i],
                    s['ker'][i],
                    padding=s['pad'][i],
                    stride=s['stp'][i],
                    dilation=s['dil'][i],
                )
            )
            other_layers.append(nn.ReLU())
            other_layers.append(Transpose(1, 2))
            other_layers.append(nn.LayerNorm(s['out_ch'][i]))
            other_layers.append(Transpose(1, 2))
            other_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            other_layers.append(nn.Dropout(0.2))
        self.other_layers = nn.Sequential(*other_layers)
        self.gru = nn.GRU(
            input_size=s['out_ch'][-1],
            hidden_size=s['out_ch'][-1],
            num_layers=3,
            dropout=0.2,
            batch_first=True,
        )
        self.embedding_layers = nn.ModuleList(
            [nn.Linear(emb_dim, s['out_ch'][-1]) for emb_dim in self.emb_dims] # Python の list だと PyTorch が「モデルの層」として認識せず、parameters() に入らない
        )
        self.FC = nn.Linear(s['out_ch'][-1] * (1 + len(self.emb_dims)), n_task)

    def calcCnnInterLen(self, seq_len: int, s: dict):
        for i in range(len(s['in_ch'])):
            seq_len = int(((seq_len + 2 * s['pad'][i] - s['dil'][i] * (s['ker'][i] - 1) - 1) / s['stp'][i]) + 1)
            if i != 0:
                seq_len = int((seq_len - (2 - 1) - 1) / 2 + 1)
        return seq_len * s['out_ch'][-1]

    def forward(self, x, *embeddings): # 可変長のembeddingをtuppleとして受け取る
        if len(embeddings) != len(self.embedding_layers):
            raise ValueError(
                f"Expected {len(self.embedding_layers)} embeddings, got {len(embeddings)}"
            )

        if self.a_type == 'v':
            x[:, self.a_dim, :] = 0

        x = self.init_layer(x)
        x = self.other_layers(x)
        x = torch.transpose(x, 1, 2)
        x = self.gru(x)

        s_out = x[1][-1].shape
        x = x[1][-1].view(s_out[0], s_out[1])

        projected_embeddings = []
        for layer, embedding in zip(self.embedding_layers, embeddings):
            projected_embeddings.append(torch.relu(layer(embedding)))

        if self.a_type == 'm':
            x = torch.zeros_like(x)
        elif self.a_type == 'p' and projected_embeddings:
            projected_embeddings[0] = torch.zeros_like(projected_embeddings[0])

        xx = torch.cat((x, *projected_embeddings), dim=1) # リストの中身を展開する
        return self.FC(xx)


class CNN_GRU_emb(CNN_GRU_multiemb): # 後方互換
    def __init__(self, settings: dict, emb_dim: int, n_task: int, L: int, device='cpu', *, a_type=None, a_dim=None):
        super().__init__(settings, [emb_dim], n_task, L, device=device, a_type=a_type, a_dim=a_dim)


class CNN_GRU_2emb(CNN_GRU_multiemb): # 後方互換
    def __init__(self, settings: dict, emb_dim1: int, emb_dim2: int, n_task: int, L: int, device='cpu', *, a_type=None, a_dim=None):
        super().__init__(settings, [emb_dim1, emb_dim2], n_task, L, device=device, a_type=a_type, a_dim=a_dim)


class CNN_GRU_3emb(CNN_GRU_multiemb): # 後方互換
    def __init__(self, settings: dict, emb_dim1: int, emb_dim2: int, emb_dim3: int, n_task: int, L: int, device='cpu', *, a_type=None, a_dim=None):
        super().__init__(settings, [emb_dim1, emb_dim2, emb_dim3], n_task, L, device=device, a_type=a_type, a_dim=a_dim)
