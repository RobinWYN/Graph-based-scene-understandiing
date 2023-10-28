import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sub_graph import MLP


class Linear(nn.Module):
    def __init__(self, n_in, n_out, hidden_size=64):
        super(Linear,self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(n_in, n_out, bias=False)

    def forward(self,x):
        out = self.linear(x)
        out = self.norm(out)
        out = self.act(out)
        return out


class PredNet(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()

        self.pred = nn.Sequential(
            MLP(2 * hidden_size, hidden_size, hidden_size),
            nn.Linear(hidden_size, 60)
        )
        self.cls = nn.Sequential(
            MLP(2 * hidden_size, hidden_size, hidden_size),
            nn.Linear(hidden_size, 1)
        )


    def forward(self, actors):

        reg = self.pred(actors)
        reg = reg.view(-1, 6, 30, 2) #[batch, 6, 30, 2]

        cls_pred = self.cls(actors)  #[batch, 6, 1]

        cls_pred, sort_idcs = cls_pred.sort(1, descending=True)
        row_idcs = torch.arange(len(sort_idcs)).long()
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        sort_idcs = sort_idcs.view(-1)
        reg = reg[row_idcs, sort_idcs].view(-1, 6, 30, 2)

        return reg, cls_pred


class BehaviorClsNet(nn.Module):
    def __init__(self, input_size=64, hidden_size=32, num_layer:int=4):
        super().__init__()
        self.scale = nn.Linear(input_size, hidden_size)

        self.cls = nn.Sequential(
            MLP(hidden_size, hidden_size, hidden_size),
            nn.Linear(hidden_size, 3),
            nn.Softmax(dim=-1)
        )
        self.layers = nn.Sequential()
        for i in range(num_layer):
            self.layers.add_module('mlp'+str(i), MLP(hidden_size, hidden_size, hidden_size))

        self.linear = nn.Linear(hidden_size, 3)
        self.pred = nn.Softmax(dim=-1)

    def forward(self, actors):
        inner = self.scale(actors)
        inner = self.layers(inner)
        inner = self.linear(inner)
        cls_pred = self.pred(inner)
        #cls_pred = self.cls(actors)  #[batch, 3]
        return cls_pred