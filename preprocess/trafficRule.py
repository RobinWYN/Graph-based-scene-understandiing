# from curses import raw
# from multiprocessing.context import SpawnProcess
# from pickle import FRAME
# from pickletools import long1
# from pyexpat import model
import matplotlib.pyplot as plt
import pandas as pd
# import pathlib 
import numpy as np
# from tqdm import tqdm
import torch
import math


def laneProbibit(raw_data):
    nbr_vec = raw_data['nbr_mat'][0,:].numpy().tolist()
    succ_vec = raw_data['succ_mat'][0,:].numpy().tolist()
    curr_lane_y = raw_data['lane_feat'][0,0,1].numpy()
    lane_probit = torch.tensor([0,1,0])
    for i in range(len(nbr_vec)):
        if nbr_vec[i] > 0.01:
            lane_y = raw_data['lane_feat'][i,0,1].numpy()
            if lane_y > curr_lane_y:
                lane_probit[0] = torch.tensor(1)
            else:
                lane_probit[2] = torch.tensor(1)
    if np.amax(succ_vec) < 1:
        lane_probit[1] = torch.tensor(0)
    return lane_probit


def speeLimit(raw_data):
    v_max = 100
    v_min = 0
    speed_limit = torch.tensor([v_max,v_min])
    return speed_limit

def trafficRule(raw_data):
    return torch.cat([laneProbibit(raw_data),speeLimit(raw_data)],dim=0)

if __name__ == "__main__":
    sample = torch.load('G:/datasets/argo1/train_data_1/train46.pkl')
    data = sample[10]
    model_input = trafficRule(data)
    print(model_input)