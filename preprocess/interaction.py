# from curses import raw
# from multiprocessing.context import SpawnProcess
# from pickle import FRAME
# from pickletools import long1
# from pyexpat import model
# from socket import RDS_CMSG_RDMA_UPDATE
# from tkinter import N
import matplotlib.pyplot as pyplot
import pandas as pd
import pathlib 
import numpy as np
from tqdm import tqdm
import torch
import math
from SafetyField import *
from tqdm import tqdm, trange

HISTORYFRAME = 19
FUTUREFRAME = 20
DIFFINTERVAL = 5 #the interval of solving veloclities and accelerations by differentiation
FRAMERATE = 10
LONGREGIONINTER = 10
LANEWIDTH = 3.5
LONGFIELD = 50
LATERFIELD = 25
BLOCKSIZE = 0.5

def drivingSafetyField(raw_data):
    field_intensity = torch.zeros([3,3],dtype=torch.float).numpy()
    my_safetyf = SafetyField()
    my_config = config_reader().read()
    # print(my_config)
    my_config.adaptor.dataInput(raw_data)
    data = my_config.adaptor.get_data()
    safety_Field = my_safetyf.generate(data,my_config)
    my_config.reprocessor.run(safety_Field)
    safety_field = safety_Field.field
    
    pyplot.imshow(safety_field)
    pyplot.show()
    # print("max energy: ",np.max(safety_field))
    # visualization().display(safety_field, data, my_config.Resolution, my_config.RealSize)
    for i in np.arange(LONGFIELD-0.5*LONGREGIONINTER,LONGFIELD+2.5*LONGREGIONINTER,BLOCKSIZE):
        for j in np.arange(LATERFIELD-1.5*LANEWIDTH,LATERFIELD+1.5*LANEWIDTH,BLOCKSIZE):
            m = int(i // BLOCKSIZE)
            n = int(j // BLOCKSIZE)
            a = max(int(-((i+ 0.5*LONGREGIONINTER-LONGFIELD) // LONGREGIONINTER) + 2 ), 0)
            a = min(a,2)
            b = max(int(-((j + 0.5*LANEWIDTH-LATERFIELD) // LANEWIDTH )+1),0)
            b = min(b,2)
            if m > 201 or n > 101:
                print('error')
            field_intensity[a][b] = field_intensity[a][b] + safety_field[m][n]

    block_num = LANEWIDTH/BLOCKSIZE * LONGREGIONINTER/BLOCKSIZE
    for i in range(3):
        for j in range(3):
            field_intensity[i][j] = field_intensity[i][j] / block_num

    field_intensity = field_intensity.reshape([1,9])
    field_intensity = torch.from_numpy(field_intensity)
    field_intensity = torch.squeeze(field_intensity)
        
    return field_intensity


def roadRightConflict(raw_data):
    pos_now = torch.tensor([0.0,0.0])
    pos_past = raw_data['traj_feat'][0,HISTORYFRAME-DIFFINTERVAL,:2]
    time_interval = torch.tensor([1/FRAMERATE*DIFFINTERVAL,1/FRAMERATE*DIFFINTERVAL])
    ego_vel = ((pos_now-pos_past)/time_interval).numpy().tolist()
    pos_now_ego = pos_now.numpy().tolist()

    avail = raw_data['traj_mask'][1:,HISTORYFRAME-1]
    avail = torch.squeeze(avail)
    dim0 = len(avail.numpy().tolist())
    right_conflict = torch.zeros([3,3],dtype=torch.float)
    for i in range(dim0):
        if avail[i]:
            pos_now_surr = raw_data['traj_feat'][i+1,HISTORYFRAME-1,:2]
            pos_past_surr = raw_data['traj_feat'][i+1,HISTORYFRAME-DIFFINTERVAL-1,:2]
            time_interval = torch.tensor([1/FRAMERATE*DIFFINTERVAL,1/FRAMERATE*DIFFINTERVAL])
            surr_vel = ((pos_now_surr-pos_past_surr)/time_interval).numpy().tolist()
            pos_now_surr = pos_now_surr.numpy().tolist()

            for j in range(1,FUTUREFRAME,3):
                pos_pred_ego = pos_now_ego + [k*1/FRAMERATE for k in ego_vel]
                pos_pred_surr = pos_now_surr + [m*1/FRAMERATE for m in surr_vel]
                relative_dist = [0,0]
                relative_dist[0] = pos_pred_surr[0] - pos_pred_ego[0]
                relative_dist[1] = pos_pred_surr[1] - pos_pred_ego[1]
                if max(relative_dist) < LANEWIDTH and min(relative_dist) > -LANEWIDTH:
                    a = max(int(-((pos_pred_ego[0]+ 0.5*LONGREGIONINTER) // LONGREGIONINTER) + 2) , 0)
                    a = min(a,2)
                    b = max(int(-((pos_pred_ego[1] + 0.5*LANEWIDTH) // LANEWIDTH )+1),0)
                    b = min(b,2)
                    right_conflict[a,b] = torch.tensor(FUTUREFRAME-j) # road conflicet indicator: pred time domain - the time of crash
        else:
            break
    right_conflict = right_conflict.view(-1)

    return right_conflict

def interaction(raw_data):
    return torch.cat([drivingSafetyField(raw_data),roadRightConflict(raw_data)],dim=0)

if __name__ == "__main__":
    sample = torch.load('G:/datasets/argo1/train_data_1/train46.pkl')
    data = sample[40]
    # model_input = physicalFeature(data)
    # print("ego history: ",egoHistory(data))
    print("DSF: ",drivingSafetyField(data))
    print("road conflict: ",roadRightConflict(data))
    print("overall: ", interaction(data))
    # print(model_input)