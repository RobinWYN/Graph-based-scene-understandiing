import torch
import torch.nn as nn

from models.sub_graph import SubGraph
from models.sub_graph import ActorNet
from models.sub_graph import MLP
from models.global_graph import GCNModule
from models.global_graph import AttentionNet


class VectorNetBackbone(nn.Module):
    """
    hierarchical GNN with trajectory prediction MLP
    """

    def __init__(self,
                 device,
                 in_chn_agent=6,
                 in_chn_lane=8,
                 num_subgraph_layers=3,
                 hidden_size=128,):
        super(VectorNetBackbone, self).__init__()
        # some params
        self.hidden_size = hidden_size
        self.device = device

        # subgraph feature extractor
        self.subgraph_lane = SubGraph(in_chn_lane, num_subgraph_layers, hidden_size)
        #self.subgraph_lane = ActorNet(in_chn_lane)
        self.subgraph_agent = SubGraph(in_chn_agent, num_subgraph_layers, hidden_size)
        #self.subgraph_agent = ActorNet(in_chn_agent)

        #GCN
        #self.gcn = GCNModule(num_layers=3, hidden_size=hidden_size)

        # agent_lane
        self.al_interation = AttentionNet(hidden_size, hidden_size, hidden_size)

        # agent_agent
        self.aa_interaction = AttentionNet(hidden_size + 1, hidden_size + 1, hidden_size)
        self.aa_fc = MLP(hidden_size * 2, hidden_size, hidden_size)
        self.token_agent = nn.Parameter(torch.Tensor(15, 1))
        nn.init.normal_(self.token_agent, mean=0., std=.02)

        # ego_lane
        self.agent_lane = AttentionNet(hidden_size, hidden_size, hidden_size)
        self.dec1 = MLP(hidden_size, hidden_size, hidden_size)
        self.dec2 = nn.Linear(hidden_size, 60)


    def forward(self, data):

        lane_feat = data["lane_feat"].to(self.device)
        traj_feat = data["traj_feat"].to(self.device)
        lane_mask = data["lane_mask"].to(self.device)
        traj_mask = data["traj_mask"].to(self.device)
        nbr_mat = data["nbr_mat"].to(self.device)
        pred_mat = data["pred_mat"].to(self.device)
        succ_mat = data["succ_mat"].to(self.device)

        sub_graph_lane = self.subgraph_lane(lane_feat, lane_mask)
        sub_graph_agent = self.subgraph_agent(traj_feat, traj_mask)
        #sub_graph_agent = self.subgraph_agent(traj_feat.reshape(-1, 19, 6), traj_mask)
        #graph_feat = self.gcn(sub_graph_lane, nbr_mat, pred_mat, succ_mat, lane_mask)
        graph_feat = sub_graph_lane 
        #[batch, num, hidden]

        agent_lane = self.al_interation(sub_graph_agent, graph_feat, graph_feat, lane_mask)


        agent_feat = torch.cat([agent_lane, sub_graph_agent], dim=-1)
        agent_feat = self.aa_fc(agent_feat)
        token_agent = self.token_agent.unsqueeze(0).repeat(len(sub_graph_agent), 1, 1)
        agent_feat = torch.cat([agent_feat, token_agent], dim=-1) #[batch, num, hidden + 1]

        agent_mask = (traj_mask[:, :, -1] == True)
        target_feat = self.aa_interaction(agent_feat[:, 0].unsqueeze(1), agent_feat, agent_feat, agent_mask) 
        #[batch, 1, hidden]

        target_feat = self.agent_lane(target_feat, graph_feat, graph_feat, lane_mask)
        #[batch, 1, hidden]

        target_feat = target_feat.squeeze(1)
        #[batch, hidden]


        dec_traj = self.dec1(target_feat)
        dec_traj = self.dec2(dec_traj)

        pred_traj = dec_traj.view((-1, 30, 2)).cumsum(1)

        return pred_traj  #[batch, 30, 2]
