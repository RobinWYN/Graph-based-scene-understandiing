'''
author: WYN and LHT
version: 1.2
date: 2023.4.18
funtion: translate the HighD csv into pkl files
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from tqdm import tqdm
import torch
from torch.utils.data import Sampler, DataLoader
import pathlib
import math

from utils.demo import track_plot

DIVIDE = 1
CURRENT = 50
DISCARD = 75
RAWPATH = 'D:\HighD_processed_tracks'
TARGETFRAMERATE = 10
DATAFRAMERATE = 25

def track_plot(path, data):
    lane_feat = data["lane_feat"]
    traj_feat = data["traj_feat"]
    traj_mask = data["traj_mask"]
    
    # lane plot
    for index in range(lane_feat.shape[1]):
        lane_start = lane_feat[:, index, 0:2]
        lane_end = lane_feat[:, index, 2:4]
        
        for lane_seg in range(lane_feat.shape[0]):
            plt.plot([lane_start[lane_seg, 0], lane_end[lane_seg, 0]], [lane_start[lane_seg, 1], lane_end[lane_seg, 1]], color='r')
            plt.scatter([lane_start[lane_seg, 0], lane_end[lane_seg, 0]], [lane_start[lane_seg, 1], lane_end[lane_seg, 1]], s=10, color='r')
    
    # trajactory plot
    for index in range(traj_feat.shape[1]):
        traj_start = traj_feat[:, index, 0:2]
        traj_end = traj_feat[:, index, 2:4]
        
        for traj_seg in range(traj_feat.shape[0]):
            if traj_mask[traj_seg][index]:
                # print(traj_feat[traj_seg][index])
                if traj_feat[traj_seg][index][5] == 1:
                    plt.plot([traj_start[traj_seg, 0], traj_end[traj_seg, 0]], [traj_start[traj_seg, 1], traj_end[traj_seg, 1]], color='b')
                    plt.scatter([traj_start[traj_seg, 0], traj_end[traj_seg, 0]], [traj_start[traj_seg, 1], traj_end[traj_seg, 1]], s=10, color='b')
                else:
                    plt.plot([traj_start[traj_seg, 0], traj_end[traj_seg, 0]], [traj_start[traj_seg, 1], traj_end[traj_seg, 1]], color='g')
                    plt.scatter([traj_start[traj_seg, 0], traj_end[traj_seg, 0]], [traj_start[traj_seg, 1], traj_end[traj_seg, 1]], s=10, color='g')
    
    plt.show()

def preprocess_HighD(raw_path, ground_truth, radius=50, max_num_agent=15, max_num_lane=40):
    
    df = pd.read_csv(raw_path).iloc[(DIVIDE-1):] #truncation
    max_n = df.shape[0]
    #print(df.columns.tolist())
    ego_df = df[["x","y","xVelocity", "yVelocity"]].iloc
    
    origin_x = ego_df[CURRENT-1]['x']
    origin_y = ego_df[CURRENT-1]['y']
    origin = torch.tensor([origin_x, origin_y], dtype=torch.float)
    
    agent_heading_vector = origin - torch.tensor([ego_df[CURRENT-2]['x'], ego_df[CURRENT-2]['y']], dtype=torch.float)
    theta = torch.atan2(agent_heading_vector[1], agent_heading_vector[0])
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]])
    #print("rotate_mat", rotate_mat)

    ground_truth = torch.tensor([ground_truth])
    # interp_f = interp1d(ground_truth[:,0], ground_truth[:, 1], kind='linear')
    # interp_x = np.linspace(ground_truth[CURRENT, 0], ground_truth[max_n-CURRENT-1, 0], 30)
    # interp_y = interp_f(interp_x)
    # gt = torch.from_numpy(np.stack([interp_x, interp_y], axis=-1)).float()
    # gt = torch.matmul(gt - origin, rotate_mat)
    
    
    # initialization
    traj_feats = torch.zeros(max_num_agent, 19, 6, dtype=torch.float)
    traj_mask = torch.zeros(max_num_agent, 19, dtype=torch.bool)
    lane_feat = torch.zeros(max_num_lane, 19, 8, dtype=torch.float)
    lane_mask = torch.ones(max_num_lane, dtype=torch.bool)
    
     
    # get ego agent features
    x = torch.zeros(20, 2, dtype=torch.float)
    x_mask = torch.zeros(20, dtype=torch.int)
    
    xy = torch.from_numpy(np.stack([df['x'].values, df['y'].values], axis=-1)).float()

    framerate = DATAFRAMERATE/TARGETFRAMERATE

    interp_i = [i for i in range(max_n)]
    interp_xf = interp1d(interp_i, xy[:, 0], kind='linear')
    interp_yf = interp1d(interp_i, xy[:, 1], kind='linear')
    target_i = np.linspace(CURRENT-19*framerate, CURRENT, 20)
    interp_x = interp_xf(target_i)
    interp_y = interp_yf(target_i)
    #xy = interp_arc(20, xy[:, 0], xy[:, 1])
    xy = torch.from_numpy(np.stack([interp_x, interp_y], axis=-1)).float()
    
    node_steps = np.arange(20)
    x[node_steps] = torch.matmul(xy - origin, rotate_mat)
    x_mask[node_steps] = 1
        
    traj_feats[0] = torch.cat(
        [x[:19], 
         x[1:], 
         torch.arange(19,0,-1).reshape(-1, 1), 
         0 * torch.ones(19, 1)], # idx * torch.ones(19, 1)]
        axis=-1
    )
    
    traj_mask[0] = x_mask[:19] & x_mask[1:]
    traj_feats[0] = torch.where(
        traj_mask[0].unsqueeze(-1),
        traj_feats[0],
        torch.zeros(19, 6)
    )

    # get other agents feature
    vehsPos = []
    for i in range(6,34,3):
        num = 1
        if i != 9 and i !=12 and abs(df.iloc[CURRENT][i])>0.5 :  # skip targetpreceding and targetfollowing and vehicle exist
            # num = num + 1
            
            agent_x = df.iloc[CURRENT][i-2] + df.iloc[CURRENT]['x']
            agent_y = df.iloc[CURRENT][i-1] + df.iloc[CURRENT]['y']
            dist2ego = math.sqrt((agent_x-origin_x)**2+(agent_y-origin_y)**2)
            vehPos = [agent_x,agent_y,dist2ego,i]
            vehsPos.append(vehPos)
    vehsPos = np.array(vehsPos)
    if vehsPos.size !=0:
        vehsPos = vehsPos[vehsPos[:,2].argsort()]
    index = 0
    for veh in vehsPos:
        index = index + 1
        veh_x = torch.zeros(20, 2, dtype=torch.float)
        veh_mask = torch.zeros(20, dtype=torch.int)
        
        # a = df.iloc[:,int(veh[3])-2].values
        a = np.stack([df.iloc[:,int(veh[3])-2].values, df.iloc[:,int(veh[3])-1].values], axis=-1)
        veh_xy = torch.from_numpy(np.stack([df.iloc[:,int(veh[3])-2].values, df.iloc[:,int(veh[3])-1].values], axis=-1)).float() \
            + torch.from_numpy(np.stack([df['x'].values, df['y'].values], axis=-1)).float()
        veh_xy_np = veh_xy.numpy()
        veh_interp_i = [i for i in range(max_n)]
        veh_interp_xf = interp1d(veh_interp_i, veh_xy[:, 0], kind='linear')
        veh_interp_yf = interp1d(veh_interp_i, veh_xy[:, 1], kind='linear')
        veh_target_i = np.linspace(CURRENT-19*framerate, CURRENT, 20)
        veh_interp_x = veh_interp_xf(veh_target_i)
        veh_interp_y = veh_interp_yf(veh_target_i)
        #xy = interp_arc(20, xy[:, 0], xy[:, 1])
        veh_xy = torch.from_numpy(np.stack([veh_interp_x, veh_interp_y], axis=-1)).float()
        
        node_steps = np.arange(20)
        veh_x[node_steps] = torch.matmul(veh_xy - origin, rotate_mat)
        veh_mask[node_steps] = 1
            
        traj_feats[index] = torch.cat(
            [veh_x[:19], 
            veh_x[1:], 
            torch.arange(19,0,-1).reshape(-1, 1), 
            1 * torch.ones(19, 1)], # idx * torch.ones(19, 1)]
            axis=-1
        )
        
        traj_mask[index] = veh_mask[:19] & veh_mask[1:]
        
        traj_feats[index] = torch.where(
            traj_mask[index].unsqueeze(-1),
            traj_feats[index],
            torch.zeros(19, 6)
        )
    
    # get lane feature
    # ego-lane center refers to time 0
    ego_lane_center = ego_df[0]['y']
    related_num = 4
    
    lanes = torch.from_numpy(np.ones((20, related_num))).float()
    if ego_df[CURRENT-1]['y'] >= 0 :
        related_lane = torch.tensor([ego_lane_center+2, ego_lane_center-2, ego_lane_center+6, ego_lane_center-6],
                                    dtype=torch.float)
    else:
        related_lane = torch.tensor([ego_lane_center-2, ego_lane_center+2, ego_lane_center-6, ego_lane_center+6],
                                    dtype=torch.float)
    lanes = lanes * related_lane
    
    for idx in range(related_num):
        lane_y = lanes[:,idx].reshape(-1,1)
        lane_x = np.linspace(ego_df[0]['x'], ego_df[max_n-CURRENT-1]['x'], 20).reshape(-1,1)
        
        lane_centerline = torch.from_numpy(np.stack([lane_x, lane_y], axis=-1)).float()
        lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat).squeeze(-2)

        is_intersection = 0
        turn_direction = 0
        traffic_control = 0

        lane_feat[idx] = torch.cat(
            [lane_centerline[:-1],
             lane_centerline[1:],
             is_intersection * torch.zeros(19, 1),
             turn_direction * torch.zeros(19, 1),
             traffic_control * torch.zeros(19, 1),
             idx * torch.ones(19, 1)],
            dim=1
        )
    
    
    # unnecessities
    nbr_mat = torch.from_numpy(np.zeros((40,40)))
    pred_mat = torch.from_numpy(np.zeros((40,40)))
    succ_mat = torch.from_numpy(np.zeros((40,40)))

    
    return {
        'traj_feat': traj_feats,
        'traj_mask': traj_mask,
        'lane_feat': lane_feat,
        'lane_mask': lane_mask,
        'nbr_mat': nbr_mat,
        'pred_mat': pred_mat,
        'succ_mat': succ_mat,
        'ground_truth': ground_truth,
        'rotate_mat': rotate_mat
    }

    
    
def Get_PreData(raw_paths):
    path = pathlib.Path.cwd().parent.joinpath(raw_paths)
    data = []

    for idx, raw_path in tqdm(enumerate(path.iterdir())):
        # print(raw_path.name)
        
        gt = 0
        # print(raw_path.name)
        if "left" in raw_path.name:
            gt = 1
        elif "right" in raw_path.name:
            gt = 2
        event = preprocess_HighD(raw_path, gt, radius=50, max_num_agent=15, max_num_lane=40)
        print(event['lane_feat'][0][1])
        track_plot(RAWPATH,event)
        data.append(event)
        # print(data)
        # if idx % 100 == 99:
        #     with open('D:\highD_pkl_1-51\HighD_VIFGNN_' + str(idx // 100) + '.pkl', 'wb') as f:
        #         torch.save(data, f)
        #     data = []



def test():
    #raw_path = '/home/linhaotian/coding/project2/processed_tracks'
    #Get_PreData(raw_path)
    raw_path = '/Users/mac/Desktop/Intro_DL/project2/processed_tracks/left_set01_veh0048.csv'
    data = preprocess_HighD(raw_path, 1, radius=50, max_num_agent=15, max_num_lane=40)
    #track_plot(raw_path, data) # checking correctness
    return data

    
if __name__ == "__main__":
    Get_PreData(RAWPATH)
    #test()