import os
import pathlib 
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.interpolate import interp_arc

from tqdm import tqdm


def preprocess(am, raw_path, radius, max_num_agent, max_num_lane):
    df = pd.read_csv(raw_path)
    # filter out actors that are unseen during the historical time steps
    timestamps = list(np.sort(df['TIMESTAMP'].unique()))
    curr_timestamp = timestamps[19]
    curr_df = df[df['TIMESTAMP'] == curr_timestamp]

    agent_df = df[df['OBJECT_TYPE'] == 'AGENT'].iloc
    city = df['CITY_NAME'].values[0]

    # make the scene centered at AV
    origin_x = agent_df[19]['X']
    origin_y = agent_df[19]['Y']
    origin = torch.tensor([origin_x, origin_y], dtype=torch.float)
    agent_heading_vector = origin - torch.tensor([agent_df[18]['X'], agent_df[18]['Y']], dtype=torch.float)
    theta = torch.atan2(agent_heading_vector[1], agent_heading_vector[0])
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]])

    agent_x = df[df['OBJECT_TYPE'] == 'AGENT']['X'].values.reshape(-1, 1)
    agent_y = df[df['OBJECT_TYPE'] == 'AGENT']['Y'].values.reshape(-1, 1)
    ground_truth = torch.from_numpy(np.concatenate([agent_x[20:], agent_y[20:]], axis=1)).float()
    gt = torch.matmul(ground_truth - origin, rotate_mat)

    # initialization
    x = torch.zeros(max_num_agent, 20, 2, dtype=torch.float)
    x_mask = torch.zeros(max_num_agent, 20, dtype=torch.bool)
    traj_feats = torch.zeros(max_num_agent, 19, 8, dtype=torch.float)
    traj_mask = torch.zeros(max_num_agent, 19, dtype=torch.bool)
    traj_loc = torch.zeros(max_num_agent, 4) #[x, y, sin, cos]
    traj_field = torch.zeros(max_num_agent)
    #get line distance and agent distance at current time, then sort them

    curr_x = (curr_df['X'].values - origin_x).reshape(-1, 1)
    curr_y = (curr_df['Y'].values - origin_y).reshape(-1, 1)
    dis = np.linalg.norm(np.concatenate([curr_x, curr_y], axis=1), axis=1)
    curr_df.insert(loc=6, column='DIS', value=dis)
    curr_df = curr_df.sort_values(by=['DIS'])
    curr_df = curr_df[curr_df['DIS'] < radius]
    len_curr = len(curr_df)
    curr_df = curr_df.iloc

    # get agent features

    agent_idx = 0
    for idx in range(len_curr):
        if agent_idx == max_num_agent:
            break

        track_id = curr_df[idx]['TRACK_ID']
        actor_df = df[df['TRACK_ID'] == track_id]
        actor_df = actor_df[actor_df['TIMESTAMP'] <= curr_timestamp]
        node_steps = [timestamps.index(timestamp) for timestamp in actor_df['TIMESTAMP']]

        if len(node_steps) <= 2:
            continue

        if node_steps[-2] != 18:
            continue

        xy = torch.from_numpy(np.stack([actor_df['X'].values, actor_df['Y'].values], axis=-1)).float()
        actor_origin = xy[-1]
        actor_head = actor_origin - xy[-2]
        actor_theta = torch.atan2(actor_head[1], actor_head[0])
        actor_rotate = torch.tensor([[torch.cos(actor_theta), -torch.sin(actor_theta)],
                                     [torch.sin(actor_theta), torch.cos(actor_theta)]])

        x[agent_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)
        x_mask[agent_idx, node_steps] = 1
                
        traj_feats[agent_idx] = torch.cat(
            [x[agent_idx, :19], 
             x[agent_idx, 1:],
             (x[agent_idx, 1:, 0] - x[agent_idx, :19, 0]).unsqueeze(-1) / 0.1,
             (x[agent_idx, 1:, 1] - x[agent_idx, :19, 1]).unsqueeze(-1) / 0.1,
             torch.arange(19).reshape(-1, 1), 
             agent_idx * torch.ones(19, 1)],
            axis=-1
        )
        traj_mask[agent_idx] = x_mask[agent_idx, :19] & x_mask[agent_idx, 1:]

        traj_feats[agent_idx] = torch.where(
            traj_mask[agent_idx].unsqueeze(-1),
            traj_feats[agent_idx],
            torch.zeros(19, 8)
        )

        traj_head = torch.matmul(actor_head, rotate_mat)
        traj_theta = torch.atan2(traj_head[1], traj_head[0])
        traj_loc[agent_idx] = torch.cat(
            [actor_origin - origin,
             torch.tensor([torch.sin(traj_theta), torch.cos(traj_theta)])]
        )

        #safety field

        if agent_idx != 0:
            agent_v = torch.norm(actor_head / 0.1)
            agent_origin = torch.matmul(origin - actor_origin, actor_rotate)
            agent_cos = torch.cos(torch.atan2(agent_origin[1], agent_origin[0]))
            traj_field[agent_idx] = torch.exp(agent_v * agent_cos) / (curr_df[idx]['DIS'] ** 2)

        agent_idx += 1
        
    '''
    for idx in range(len_curr):
        if idx == max_num_agent:
            break

        track_id = curr_df[idx]['TRACK_ID']
        actor_df = df[df['TRACK_ID'] == track_id]
        actor_df = actor_df[actor_df['TIMESTAMP'] <= curr_timestamp]

        node_steps = [timestamps.index(timestamp) for timestamp in actor_df['TIMESTAMP']]
        xy = torch.from_numpy(np.stack([actor_df['X'].values, actor_df['Y'].values], axis=-1)).float()
        x[idx, node_steps] = torch.matmul(xy - origin, rotate_mat)
        x_mask[idx, node_steps] = 1
        
        traj_feats[idx] = torch.cat(
            [x[idx, :19], 
             x[idx, 1:], 
             torch.arange(19).reshape(-1, 1), 
             idx * torch.ones(19, 1)],
            axis=-1
        )
        traj_mask[idx] = x_mask[idx, :19] & x_mask[idx, 1:]

        traj_feats[idx] = torch.where(
            traj_mask[idx].unsqueeze(-1),
            traj_feats[idx],
            torch.zeros(19, 6)
        )
    '''
    #get lane features

    lane_ids = np.array(am.get_lane_ids_in_xy_bbox(origin_x, origin_y, city, radius))
    lane_dis = []

    for lane_id in lane_ids:
        lane_diff = am.get_lane_segment_centerline(lane_id, city)[:, :2] - np.array([origin_x, origin_y])
        lane_mindis = np.min(np.linalg.norm(lane_diff, axis=1))
        lane_dis.append(lane_mindis)

    lane_dis = np.array(lane_dis)
    lane_ids = lane_ids[lane_dis < radius]
    lane_dis = lane_dis[lane_dis < radius]
    lane_ids = lane_ids[np.argsort(lane_dis)]
    lane_feat = torch.zeros(max_num_lane, 19, 8)
    lane_mask = torch.zeros(max_num_lane, dtype=torch.bool)

    for idx, lane_id in enumerate(lane_ids):
        if idx == max_num_lane:
            lane_ids = lane_ids[:idx]
            break

        lane_mask[idx] = 1
        lane_centerline = am.get_lane_segment_centerline(lane_id, city)[:, : 2]
        lane_centerline = interp_arc(20, lane_centerline[:, 0], lane_centerline[:, 1])
        lane_centerline = torch.from_numpy(lane_centerline).float()
        lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat)

        is_intersection = am.lane_is_in_intersection(lane_id, city)
        turn_direction = am.get_lane_turn_direction(lane_id, city)
        traffic_control = am.lane_has_traffic_control_measure(lane_id, city)
        if turn_direction == 'NONE':
            turn_direction = 0
        elif turn_direction == 'LEFT':
            turn_direction = 1
        elif turn_direction == 'RIGHT':
            turn_direction = 2

        lane_feat[idx] = torch.cat(
            [lane_centerline[:-1],
             lane_centerline[1:],
             is_intersection * torch.ones(19, 1),
             turn_direction * torch.ones(19, 1),
             traffic_control * torch.ones(19, 1),
             idx * torch.ones(19, 1)],
            dim=1
        )

    return {
        'traj_feat': traj_feats,
        'traj_mask': traj_mask,
        'traj_loc': traj_loc,
        'traj_field': traj_field,
        'lane_feat': lane_feat,
        'lane_mask': lane_mask,
        'ground_truth': gt
    }


def Get_PreData(am, raw_paths, split):
    base_dir = '../argoverse1/arg_data/'
    path = pathlib.Path.cwd().parent.joinpath(raw_paths)
    data = []

    for idx, raw_path in tqdm(enumerate(path.iterdir())):
        data.append(preprocess(am, raw_path, radius=50, max_num_agent=15, max_num_lane=40))

        if idx % 100 == 99:
            pathlib.Path(base_dir + split).mkdir(parents=True, exist_ok=True)
            with open(base_dir + split + str(idx // 100) + '.pkl','wb') as f:
                torch.save(data, f)
            data = []


if __name__ == "__main__":
    raw_paths = '../argoverse1/train/data'
    #raw_paths = '/argoverse1'
    am = ArgoverseMap()
    Get_PreData(am, raw_paths, 'train')
    