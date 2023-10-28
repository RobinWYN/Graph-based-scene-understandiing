import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn

from models.utils import init_weights


class SubGraph(nn.Module):
    """
    Subgraph that computes all vectors in a polyline, and get a polyline-level feature
    """

    def __init__(self, tp, in_channels, num_subgraph_layres=2, hidden_unit=64):
        super(SubGraph, self).__init__()
        self.hidden_unit = hidden_unit
        self.in_channels = in_channels
        self.tp = tp

        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layres):
            self.layer_seq.add_module(
                f'glp_{i}', MLP(in_channels, hidden_unit, hidden_unit))
            in_channels = hidden_unit * 2

        self.linear = nn.Linear(hidden_unit * 2, hidden_unit)
        self.norm = nn.LayerNorm(hidden_unit)

        self.apply(init_weights)
        
    def forward(self, feats, mask):

        if self.tp == 'lane':
            mask = mask.unsqueeze(-1).repeat(1, 1, 19)

        for _, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                feats = layer(feats) #[batch, num, len, dim]

                agg_data, _ = torch.max(feats, dim=2) #[batch, num, dim]
                agg_data = agg_data.unsqueeze(2).repeat(1, 1, feats.shape[2], 1) #[batch, num, len, dim]

                feats = torch.cat([feats, agg_data], dim=3) #[batch, num, len, dim*2]

        feats = self.linear(feats) #[batch, num, len, dim]
        agg_data, _ = torch.max(feats, dim=2) #[batch, num, dim]

        agg_data = self.norm(agg_data)
        indices = (mask[:, :, -1] == False)
        agg_data[indices] = 0 
        return agg_data

class MLP(nn.Module):
    def __init__(self, in_channel, out_channel, hidden=64):
        super(MLP, self).__init__()
        act_layer = nn.ELU
        norm_layer = nn.LayerNorm

        # insert the layers
        self.linear1 = nn.Linear(in_channel, hidden, bias=True)
        self.linear2 = nn.Linear(hidden, out_channel, bias=True)

        self.norm1 = norm_layer(hidden)
        self.norm2 = norm_layer(out_channel)

        self.act1 = act_layer(inplace=True)
        self.act2 = act_layer(inplace=True)

        self.shortcut = None
        if in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channel, out_channel, bias=True),
                norm_layer(out_channel)
            )

        self.apply(init_weights)

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.linear2(out)
        out = self.norm2(out)

        if self.shortcut:
            out += self.shortcut(x)
        else:
            out += x
        return self.act2(out)