import torch
import torch.nn as nn
import numpy as np

from models.sub_graph import SubGraph
from models.sub_graph import MLP
from models.global_graph import AttentionNet
from models.decoder import PredNet, BehaviorClsNet


class VectorNetBackbone(nn.Module):
    """
    hierarchical GNN with trajectory prediction MLP
    """

    def __init__(self,
                 device,
                 in_chn_agent=6,
                 in_chn_lane=8,
                 num_subgraph_layers=3,
                 hidden_size=64,):
        super(VectorNetBackbone, self).__init__()
        # some params
        self.hidden_size = hidden_size
        self.device = device

        # subgraph feature extractor
        self.subgraph_lane = SubGraph('lane', in_chn_lane, num_subgraph_layers, hidden_size)
        self.subgraph_agent = SubGraph('agent', in_chn_agent, num_subgraph_layers, hidden_size)

        #lane_lane
        self.ll_interaction = AttentionNet(hidden_size, hidden_size, hidden_size)

        #lane_agent
        self.la_interaction = AttentionNet(hidden_size, hidden_size, hidden_size)

        # agent_lane
        self.al_interation = AttentionNet(hidden_size, hidden_size, hidden_size)

        # agent_agent
        #self.aa_interaction = AA_Attention(hidden_size, hidden_size, hidden_size)
        self.aa_interaction = AttentionNet(hidden_size, hidden_size, hidden_size)
        
        #self.aa_fc = MLP(hidden_size * 2, hidden_size, hidden_size)
        self.loc_embed = MLP(4, hidden_size, hidden_size)
        self.token_agent = nn.Parameter(torch.Tensor(15, 1))
        self.token_global = nn.Parameter(torch.Tensor(55, 1))
        nn.init.normal_(self.token_agent, mean=0., std=.02)
        nn.init.normal_(self.token_global, mean=0., std=.02)

        # ego_lane
        self.agent_lane = AttentionNet(hidden_size, hidden_size, hidden_size)
        self.multi_head = nn.Linear(hidden_size, hidden_size * 6)
        #self.dec = PredNet()
        self.dec = BehaviorClsNet()


    def forward(self, data):

        lane_feat = data["lane_feat"].to(self.device)
        traj_feat = data["traj_feat"].to(self.device)
        lane_mask = data["lane_mask"].to(self.device)
        traj_mask = data["traj_mask"].to(self.device)
        #traj_loc = data["traj_loc"].to(self.device)
        #traj_field = data["traj_field"].to(self.device)
        agent_mask = (traj_mask[:, :, -1] == True)

        sub_graph_lane = self.subgraph_lane(lane_feat, lane_mask)
      
        sub_graph_agent = self.subgraph_agent(traj_feat, traj_mask)
        
        graph_feat = self.ll_interaction(sub_graph_lane, sub_graph_lane, sub_graph_lane, lane_mask)
        graph_agent_feat = self.la_interaction(graph_feat, sub_graph_agent, sub_graph_agent, agent_mask)
    
        graph_feat = (graph_agent_feat + graph_feat + sub_graph_lane) / 3
        #[batch, num, hidden]
        
        agent_lane = self.al_interation(sub_graph_agent, graph_feat, graph_feat, lane_mask)
        #agent_feat = (agent_lane + sub_graph_agent + self.loc_embed(traj_loc)) / 3
        agent_feat = (agent_lane + sub_graph_agent) / 2
        #token_agent = self.token_agent.unsqueeze(0).repeat(len(lane_mask), 1, 1)
        #agent_feat = torch.cat([agent_feat, token_agent], dim=-1) #[batch, num, hidden + 1]
        target_feat = self.aa_interaction(agent_feat, agent_feat, agent_feat, agent_mask) 
        #[batch, num, hidden]

        global_feat = torch.cat(
            [target_feat, graph_feat],
            dim=1
        )
        global_mask = torch.cat(
            [agent_mask, lane_mask],
            dim=1
        )
        #token_global = self.token_global.unsqueeze(0).repeat(len(lane_mask), 1, 1)
        #global_feat = torch.cat(
        #    [global_feat, token_global],
        #    dim=-1
        #)
        target_query = target_feat[:, 0].unsqueeze(1)
    
        target_feat_lane = self.agent_lane(target_query, global_feat, global_feat, global_mask)
        #[batch, 1, hidden]
        """
        target_feat_lane = self.multi_head(target_feat_lane).reshape(-1, 6, 64)
        #[batch, 6, hidden]
        pred_traj_feat = torch.cat([target_query.repeat(1, 6, 1), target_feat_lane], dim=-1).squeeze(1)
        #[batch, 6, hidden * 2]
        """
        pred_traj_feat = target_feat_lane.squeeze(1)
        #[batch, hidden]
       
        dec_traj = self.dec(pred_traj_feat)
        return dec_traj  #[batch, 6, 30, 2]