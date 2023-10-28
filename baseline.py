import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import init_weights


from models.decoder import PredNet, BehaviorClsNet
from models.sub_graph import SubGraph


class LSTM(nn.Module):
        """
        LSTM baseline
        """
        def __init__(self, device, in_chn_agent=6, in_chn_lane=8, hidden_size=64, num_subgraph_layers=2, lstm_layers=2, dropout=0.3):
                super(LSTM, self).__init__()
                self.device = device
                # subgraph feature extractor
                self.subgraph_lane = SubGraph('lane', in_chn_lane, num_subgraph_layers, hidden_size) 
                self.subgraph_agent = SubGraph('agent', in_chn_agent, num_subgraph_layers, hidden_size)
                
                self.agent_lstm = nn.LSTM(input_size=hidden_size, 
                                          hidden_size=hidden_size, 
                                          num_layers=lstm_layers, 
                                          batch_first=True, 
                                          dropout=dropout)
                #self.decoder = PredNet()
                self.decoder = BehaviorClsNet()
                self.linear = nn.Linear(hidden_size, hidden_size)
                
                self.apply(init_weights)
                
        
        def forward(self, data):
                lane_feat = data["lane_feat"].to(self.device)
                traj_feat = data["traj_feat"].to(self.device)
                lane_mask = data["lane_mask"].to(self.device)
                traj_mask = data["traj_mask"].to(self.device)
                
                sub_graph_agent = self.subgraph_agent(traj_feat, traj_mask) #[64,15,64]
                # agent_mask = (traj_mask[:, :, -1] == True)
                agent_feat, _ = self.agent_lstm(sub_graph_agent) #[64,15,64]
                target_feat = self.linear(agent_feat)
                
                target_feat = target_feat[:,-1,:] #[batch, hidden]
                pred_cls = self.decoder(target_feat)
                return pred_cls



class LSTM_interact(nn.Module):
        """
        LSTM baseline with lane-agent interaction
        attention mechanic included
        """
        def __init__(self, device, in_chn_agent=6, in_chn_lane=8, hidden_size=64, num_subgraph_layers=1, lstm_layers=2, dropout=0.3, num_head=4):
                super(LSTM_interact, self).__init__()
                self.device = device
                # subgraph feature extractor
                self.subgraph_lane = SubGraph(in_chn_lane, num_subgraph_layers, hidden_size) 
                self.subgraph_agent = SubGraph(in_chn_agent, num_subgraph_layers, hidden_size)
                
                self.agent_lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True, dropout=dropout)
                self.lane_l
                stm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True, dropout=dropout)
                
                self.al_interaction = AttentionLayer(hidden_size, hidden_size, hidden_size, num_head)
                self.aa_interaction = AttentionLayer(hidden_size+1, hidden_size+1, hidden_size, num_head)
                self.linear = nn.Linear(hidden_size, hidden_size)
                self.decoder = PredNet()
                
                self.token = nn.Parameter(torch.Tensor(15, 1))
                
                self.apply(init_weights)
                nn.init.normal_(self.token, mean=0., std=.02)
        
        def forward(self, data):
                lane_feat = data["lane_feat"].to(self.device)
                traj_feat = data["traj_feat"].to(self.device)
                lane_mask = data["lane_mask"].to(self.device)
                traj_mask = data["traj_mask"].to(self.device)
                
                sub_graph_agent = self.subgraph_agent(traj_feat, traj_mask) #[64,15,64]
                sub_graph_lane = self.subgraph_lane(lane_feat, lane_mask) #[64,60,64]
                agent_mask = (traj_mask[:, :, -1] == True)
                
                agent_feat, _ = self.agent_lstm(sub_graph_agent) #[64,15,64]
                lane_feat, _ = self.lane_lstm(sub_graph_lane) #[64,40,64]
                
                agent_feat = self.al_interaction(agent_feat, lane_feat, lane_feat, lane_mask) #(64,15,64)
                token = self.token.unsqueeze(0).repeat(len(sub_graph_agent), 1, 1)
                agent_feat = torch.cat([agent_feat, token], dim=-1) #[batch, num, hidden + 1]
                target_feat = self.aa_interaction(agent_feat[:, 0].unsqueeze(1), agent_feat, agent_feat, agent_mask)
                # from backbone
                target_feat = target_feat.squeeze(1) #[batch, hidden]
                
                pred_traj, pred_cls = self.decoder(target_feat)
                return pred_traj, pred_cls


    
class AttentionLayer(nn.Module):
        def __init__(self, num_q, num_k, embed_size, num_head):
                """
                k v are the same
                """
                super(AttentionLayer, self).__init__()
                self.head = num_head
                self.embed = embed_size
                self.split = embed_size // num_head
                self.Q = nn.Linear(num_q, embed_size)
                self.K = nn.Linear(num_k, embed_size)
                self.V = nn.Linear(num_k, embed_size)
                self.fc = nn.Linear(embed_size, embed_size)
                self.dk = 1 + int(np.sqrt(embed_size))
                
                self.apply(init_weights)
                
                      
        def forward(self, query, key, value, mask=None):
                Querys = torch.stack(torch.split(self.Q(query), self.split, dim=-1), dim = 0)
                Keys = torch.stack(torch.split(self.K(key), self.split, dim=-1), dim = 0)
                Values = torch.stack(torch.split(self.V(value), self.split, dim=-1), dim = 0)
                      
                energy = torch.matmul(Querys, Keys.transpose(-2,-1))
                if mask is not None:
                        mask = mask.unsqueeze(1).unsqueeze(0).repeat(self.head, 1, Querys.shape[2], 1)
                        #[batch, num] -> [h, batch, num_q, num_k]
                        energy = energy.masked_fill(~mask, -np.inf)
                energy = energy / self.dk
                scores = F.softmax(energy, dim=-1)
                      
                output = torch.matmul(scores, Values)
                output = torch.cat(torch.split(output, 1, dim=0), dim=-1).squeeze(0) #[N, T_q, embed_size]
                output = self.fc(output)

                return output
           
           
           
class TransformerBlock(nn.Module):
        def __init__(self, num_q, num_k, embed_size, num_head, expansion, dropout):
                super(TransformerBlock, self).__init__()
                self.selfattention = AttentionLayer(num_q, num_k, embed_size, num_head)
                self.feedforward = nn.Sequential(
                        nn.Linear(embed_size, embed_size * expansion),
                        nn.ReLU(),
                        nn.Linear(embed_size * expansion, embed_size)
                )
                self.norm1 = nn.LayerNorm(embed_size)
                self.norm2 = nn.LayerNorm(embed_size)
                self.dropout = nn.Dropout(dropout)
                
                self.apply(init_weights)
                
                      
        def forward(self, query, key, value, mask):
                attention = self.selfattention(query, key, value, mask)
                norm = self.dropout(self.norm1(attention + query)) #here's the prob
                forward = self.feedforward(norm)
                out = self.dropout(self.norm2(forward + attention))

                return out
           
           
           
class Encoder(nn.Module):
        def __init__(self, num_input_f, num_head, num_layers, device, expansion, dropout):
                super(Encoder, self).__init__()
                self.device = device
                      
                self.layers = nn.ModuleList(
                        [TransformerBlock(num_input_f, num_input_f, num_input_f, num_head, expansion, dropout)
                        for _ in range(num_layers)]
                )
                
                      
        def forward(self, input, mask):
                out = input
                for layer in self.layers:
                        out = layer(out, out, out, mask)
                                 
                return out



class DecoderBlock(nn.Module):
        def __init__(self, num_input_f, num_output_f, num_head, expansion, dropout):
                super(DecoderBlock, self).__init__()
                self.maskedattention = AttentionLayer(num_output_f, num_output_f, num_output_f, num_head)
                self.transform = TransformerBlock(num_output_f, num_input_f, num_input_f, num_head, expansion, dropout)
                self.dropout = nn.Dropout(dropout)
                self.norm = nn.LayerNorm(num_output_f)
                
                      
        def forward(self, output, input_enc, src_mask, mask):
                attention = self.maskedattention(output, output, output, src_mask)
                query = self.dropout(self.norm(attention + output))
                out = self.transform(query, input_enc, input_enc, mask)
                      
                return out
           


class Decoder(nn.Module):
        def __init__(self, num_input_f, num_output_f, num_head, num_layers, device, expansion, dropout):
                super(Decoder, self).__init__()
                self.device = device
                self.linear = nn.Linear(num_output_f, num_input_f)
                
                self.layers = nn.ModuleList(
                        [DecoderBlock(num_input_f, num_output_f, num_head, expansion, dropout)
                        for _ in range(num_layers)]
                )
                self.apply(init_weights)
                
                     
        def forward(self, output, input_enc, src_mask, mask):
                out = self.linear(output)
                for layer in self.layers:
                        out = layer(out, input_enc, src_mask, mask)   
                        
                return out
        


class Transformer(nn.Module):
        def __init__(self, hyperparams, device, num_subgraph_layers=2, hidden_size=64, 
                     in_chn_agent=6, in_chn_lane=8, enc_layers=2, dec_layers=2, num_head=4, expansion=8, dropout=0):
                """
                num_input_f: input feature dimension
                num_output_f: output feature dimension
                hidden_size: intended feature dimension
                num_head: multi-head
                """
                super(Transformer, self).__init__()
                self.hidden = hidden_size
                self.device = device
                
                self.subgraph_lane = SubGraph('lane', in_chn_lane, num_subgraph_layers, hidden_size)
                self.subgraph_agent = SubGraph('agent', in_chn_agent, num_subgraph_layers, hidden_size)
                
                self.token = nn.Parameter(torch.Tensor(40, 1))
                nn.init.normal_(self.token, mean=0., std=.02)
                
                self.Transformer_enc = Encoder(hidden_size, num_head, enc_layers, device, expansion, dropout)
                self.Transformer_dec = Decoder(hidden_size, hidden_size, num_head, dec_layers, device, expansion, dropout)
                self.linear = nn.Linear(hidden_size, hidden_size)
                
                self.aa_interaction = AttentionLayer(hidden_size+1, hidden_size+1, hidden_size, num_head)
                
                #self.decoder = PredNet(hidden_size)
                self.decoder = BehaviorClsNet()
                
                self.apply(init_weights)
                
                      
        def forward(self, data):
                """
                Transformer output: 'target_feat' shaping (batch_size,hidden_size)
                output shall be futher sent into PredNet
                """
                # data list
                lane_feat = data["lane_feat"].to(self.device) #[64, 40, 19, 8]
                traj_feat = data["traj_feat"].to(self.device) #[64, 15, 19, 6]
                lane_mask = data["lane_mask"].to(self.device) #[64, 40] 
                traj_mask = data["traj_mask"].to(self.device) #[64, 15, 19]
                nbr_mat = data["nbr_mat"].to(self.device) #[64, 40, 40]
                pred_mat = data["pred_mat"].to(self.device) #[64, 40, 40]
                succ_mat = data["succ_mat"].to(self.device) #[64, 40, 40]
                agent_mask = (traj_mask[:, :, -1] == True)
        
                sub_graph_lane = self.subgraph_lane(lane_feat, lane_mask) #[64,40,64]
                sub_graph_agent = self.subgraph_agent(traj_feat, traj_mask) #[64,15,64]
                
                out_enc = self.Transformer_enc(sub_graph_agent, agent_mask) #[64,15,64]
                out_dec = self.Transformer_dec(sub_graph_lane, out_enc, lane_mask, agent_mask) 
                agent_feat = self.linear(out_dec) #[64,40,64]
                # print("out:", target_feat.size())
                # agent_feat = agent_feat[:,0].unsqueeze(1)
                
                token = self.token.unsqueeze(0).repeat(len(sub_graph_agent), 1, 1)
                agent_feat = torch.cat([agent_feat, token], dim=-1) #[batch, num, hidden + 1]
                target_feat = self.aa_interaction(agent_feat[:, 0].unsqueeze(1), agent_feat, agent_feat, lane_mask)
                # from backbone

                target_feat = target_feat.squeeze(1) #[batch, hidden]
                pred_cls = self.decoder(target_feat)
                
                return pred_cls
        
        