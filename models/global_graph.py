import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sub_graph import MLP

class GCNBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.W_nbr = nn.Linear(hidden_size, hidden_size)
        self.W_pred = nn.Linear(hidden_size, hidden_size)
        self.W_succ = nn.Linear(hidden_size, hidden_size)
        self.W_pred_2 = nn.Linear(hidden_size, hidden_size)
        self.W_succ_2 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_size)

        self.output = nn.Linear(hidden_size, hidden_size)
        self.output.apply(self._init_weights)

    def forward(self, lane_feat, nbr, pred, succ, mask):

        pred_2 = torch.matmul(pred, pred)
        succ_2 = torch.matmul(succ, succ)
        pred_2[pred_2 > 1] = 1
        succ_2[succ_2 > 1] = 1

        next_feat = self.W(lane_feat) + self.W_nbr(torch.matmul(nbr, lane_feat)) + \
                    self.W_pred(torch.matmul(pred, lane_feat)) + \
                    self.W_succ(torch.matmul(succ, lane_feat)) + \
                    self.W_pred_2(torch.matmul(pred_2, lane_feat)) + \
                    self.W_succ_2(torch.matmul(succ_2, lane_feat))

        out = self.output(self.act(self.norm(next_feat)))
        out = self.act(self.norm(out)) + lane_feat

        output = self.act(out) #[batch, num_lane, hidden]
        return output

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

class GCNModule(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super().__init__()
        self.net_list = nn.ModuleList([])
        self.norm = nn.LayerNorm(hidden_size)

        for _ in range(num_layers):
            self.net_list.append(GCNBlock(hidden_size))

    def forward(self, lane_feat, nbr_mat, pred_mat, succ_mat, mask):
        
        for layer in self.net_list:
            lane_feat = layer(lane_feat, nbr_mat, pred_mat, succ_mat, mask) #[batch, num, hidden]
            lane_feat = self.norm(lane_feat)

        return lane_feat


class AttentionLayer(nn.Module):
    def __init__(self, q_channels, k_channels, num_unit, head):
        super().__init__()
        self.Q = nn.Linear(q_channels, num_unit)
        self.K = nn.Linear(k_channels, num_unit)
        self.V = nn.Linear(k_channels, num_unit)
        self.num_unit = num_unit
        self.head = head
        self.scale_factor_d = 1 + int(np.sqrt(k_channels))
        self.Q.apply(self._init_weights)
        self.K.apply(self._init_weights)
        self.V.apply(self._init_weights)

    def forward(self, query, key, value, mask, split=False):
        querys = self.Q(query)
        keys = self.K(key)
        values = self.V(value) #[batch, num, num_unit]
 
        split_size = self.num_unit // self.head
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, batch, num, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  
 
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, batch, num_q, num_k]
        scores = scores / self.scale_factor_d

        mask = mask.unsqueeze(1).unsqueeze(0).repeat(self.head, 1, querys.shape[2], 1)
        #[batch, num] -> [h, batch, num_q, num_k]
        scores = scores.masked_fill(~mask, -np.inf)
        scores = F.softmax(scores, dim=3)
 
        out = torch.matmul(scores, values) 
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)


class AttentionNet(nn.Module):
    """
    update agent features by lane features
    get agent interactions
    """
    def __init__(self, q_channels, k_channels, out_channels, head=8):
        super().__init__()

        self.layer = AttentionLayer(q_channels, k_channels, out_channels, head)
        self.norm = nn.LayerNorm(out_channels)
        #self.fc = MLP(out_channels, out_channels, out_channels)
        self.fc = nn.Linear(out_channels, out_channels)
        self.downsample = None
        if q_channels != out_channels:
            self.downsample = nn.Linear(q_channels, out_channels)

    def forward(self, query, key, value, mask):

        output = self.layer(query, key, value, mask)
        if self.downsample:
            query = self.downsample(query)
        output = self.norm(self.fc(output) + query)
        return output
