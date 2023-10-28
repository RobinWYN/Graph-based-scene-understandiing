import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn

from math import gcd

class ActorNet(nn.Module):
    """
    Actor feature extractor with Conv1D
    """
    def __init__(self, n_in, hidden_size):
        super(ActorNet, self).__init__()

        self.n_in = n_in
        self.hidden_size = hidden_size
        n_out = [hidden_size // 4, hidden_size // 2, hidden_size]
        blocks = [Res1d, Res1d, Res1d]
        num_blocks = [2, 2, 2]

        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i]))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2))

            for _ in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i]))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]

        self.groups = nn.ModuleList(groups)

        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], hidden_size, act=False))
        self.lateral = nn.ModuleList(lateral)

        self.output = Res1d(hidden_size, hidden_size)


    def forward(self, actors, mask):
        out = actors.transpose(1, 2)

        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)

        l_out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            if i == 0:
                l_out = F.interpolate(l_out, scale_factor=1.9, mode="linear", align_corners=False)
            else:
                l_out = F.interpolate(l_out, scale_factor=2, mode="linear", align_corners=False)
            l_out += self.lateral[i](outputs[i])
        
        feat = l_out[:, :, -1] 
        feat = feat.reshape(-1, 15, self.hidden_size) #[batch, 15, hidden]

        indices = (mask[:, :, -1] == False)
        feat[indices] = 0
        return feat

class SubGraph(nn.Module):
    """
    Subgraph that computes all vectors in a polyline, and get a polyline-level feature
    """

    def __init__(self, in_channels, num_subgraph_layres=3, hidden_unit=128):
        super(SubGraph, self).__init__()
        self.hidden_unit = hidden_unit
        self.in_channels = in_channels

        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layres):
            self.layer_seq.add_module(
                f'glp_{i}', MLP(in_channels, hidden_unit, hidden_unit))
            in_channels = hidden_unit * 2

        self.linear = nn.Linear(hidden_unit * 2, hidden_unit)
        self.norm = nn.LayerNorm(hidden_unit)
        self.linear.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
        
    def forward(self, feats, mask):

        if self.in_channels == 8:
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
    def __init__(self, in_channel, out_channel, hidden=128):
        super(MLP, self).__init__()
        act_layer = nn.ReLU
        norm_layer = nn.LayerNorm

        # insert the layers
        self.linear1 = nn.Linear(in_channel, hidden, bias=True)
        self.linear1.apply(self._init_weights)
        self.linear2 = nn.Linear(hidden, out_channel, bias=True)
        self.linear2.apply(self._init_weights)

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

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

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

class L_Norm(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.norm = nn.LayerNorm(n_out)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)

class Conv1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, act=True):
        super().__init__()

        self.conv = nn.Conv1d(
            n_in, n_out, kernel_size=kernel_size, 
            padding=(int(kernel_size) - 1) // 2, stride=stride, bias=False
        )
        self.conv.apply(self._init_weights)

        self.norm = L_Norm(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.act = act

    @staticmethod
    def _init_weights(m):
        torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out

class Res1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, act=True):
        super().__init__()
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv1.apply(self._init_weights)
        self.conv2.apply(self._init_weights)

        self.relu = nn.ReLU(inplace = True)
        self.bn1 = L_Norm(n_out)
        self.bn2 = L_Norm(n_out)

        if stride != 1 or n_out != n_in:
            self.downsample = nn.Sequential(
                nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                L_Norm(n_out)
            )
        else:
            self.downsample = None

        self.act = act

    @staticmethod
    def _init_weights(m):
        torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out = out + x
        if self.act:
            out = self.relu(out)
        return out
