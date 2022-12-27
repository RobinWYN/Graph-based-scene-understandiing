# from argoverse.map_representation.map_api import ArgoverseMap
# from argoverse.utils.interpolate import interp_arc
# from curses import raw
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

HISTORYFRAME = 19
FUTUREFRAME = 20
DIFFINTERVAL = 5 #the interval of solving veloclities and accelerations by differentiation
FRAMERATE = 10
LONGREGIONINTER = 10
LANEWIDTH = 3.5

def egoHistory(raw_data):
    his_x = torch.squeeze(raw_data['traj_feat'][0,:,:1])
    his_y = torch.squeeze(raw_data['traj_feat'][0,:,1:2])
    his = torch.cat([his_x,his_y], dim = 0)
    return his

def egoVelocity(raw_data):
    pos_now = torch.tensor([0.0,0.0])
    pos_past = raw_data['traj_feat'][0,HISTORYFRAME-DIFFINTERVAL,:2]
    time_interval = torch.tensor([1/FRAMERATE*DIFFINTERVAL,1/FRAMERATE*DIFFINTERVAL])
    velocity = (pos_now-pos_past)/time_interval
    return velocity


def egoAcce(raw_data):
    pos_now = torch.tensor([0.0,0.0])
    pos_past = raw_data['traj_feat'][0,HISTORYFRAME-DIFFINTERVAL,:2]
    time_interval = torch.tensor([1/FRAMERATE*DIFFINTERVAL,1/FRAMERATE*DIFFINTERVAL])
    vel_now = (pos_now-pos_past)/time_interval
    pos_ppast = raw_data['traj_feat'][0,HISTORYFRAME-DIFFINTERVAL*2,:2]
    vel_past = (pos_past-pos_ppast)/time_interval
    accel = (vel_now-vel_past)/time_interval
    return accel

def surrPosition(raw_data):
    avail = raw_data['traj_mask'][1:,HISTORYFRAME-1]
    avail = torch.squeeze(avail)
    dim0 = len(avail.numpy().tolist())
    surr_pos = torch.zeros([4,3,2],dtype=torch.float)
    for i in range(dim0):
        pos = raw_data['traj_feat'][i+1,HISTORYFRAME-1,:2].numpy().tolist()
        if avail[i] and pos[0] < 2.5*LONGREGIONINTER and pos[0] > -1.5*LONGREGIONINTER and abs(pos[1]) < LANEWIDTH :
            a = int(-((pos[0]+ 0.5*LONGREGIONINTER) // LONGREGIONINTER) + 2)
            b = int(-((pos[1] + 0.5*LANEWIDTH) // LANEWIDTH) +1)
            surr_pos[a,b] = raw_data['traj_feat'][i+1,HISTORYFRAME-1,:2]
        else:
            break
    # print("origin: ", surr_pos)
    surr_pos = surr_pos.view(-1)
    return surr_pos

def surrVelocity(raw_data):
    avail = raw_data['traj_mask'][1:,HISTORYFRAME-1]
    avail = torch.squeeze(avail)
    dim0 = len(avail.numpy().tolist())
    surr_vel = torch.zeros([4,3,2],dtype=torch.float)
    for i in range(dim0):
        pos = raw_data['traj_feat'][i+1,HISTORYFRAME-1,:2].numpy().tolist()
        if avail[i] and pos[0] < 2.5*LONGREGIONINTER and pos[0] > -1.5*LONGREGIONINTER and abs(pos[1]) < LANEWIDTH :
            a = int(-((pos[0]+ 0.5*LONGREGIONINTER) // LONGREGIONINTER) + 2)
            b = int(-((pos[1] + 0.5*LANEWIDTH) // LANEWIDTH) +1)
            pos_now = raw_data['traj_feat'][i+1,HISTORYFRAME-1,:2]
            pos_past = raw_data['traj_feat'][i+1,HISTORYFRAME-DIFFINTERVAL-1,:2]
            time_interval = torch.tensor([1/FRAMERATE*DIFFINTERVAL,1/FRAMERATE*DIFFINTERVAL])
            velocity = (pos_now-pos_past)/time_interval
            surr_vel[a,b] = velocity
        else:
            break
    # print('origin: ',surr_vel)
    surr_vel = surr_vel.view(-1)
    return surr_vel

def surrType(raw_data):
    avail = raw_data['traj_mask'][1:,HISTORYFRAME-1]
    avail = torch.squeeze(avail)
    dim0 = len(avail.numpy().tolist())
    surr_type = torch.zeros([4,3],dtype=torch.float)
    for i in range(dim0):
        pos = raw_data['traj_feat'][i+1,HISTORYFRAME-1,:2].numpy().tolist()
        if avail[i] and pos[0] < 2.5*LONGREGIONINTER and pos[0] > -1.5*LONGREGIONINTER and abs(pos[1]) < LANEWIDTH :
            a = int(-((pos[0]+ 0.5*LONGREGIONINTER) // LONGREGIONINTER) + 2)
            b = int(-((pos[1] + 0.5*LANEWIDTH) // LANEWIDTH) +1)
            surr_type[a,b] = torch.tensor([1]) # todo: other types
        else:
            break
    surr_type = surr_type.view(-1)
    return surr_type

def targetOrient(raw_data):
    current_pos = raw_data['ground_truth'][0,:]
    target_pose = raw_data['ground_truth'][FUTUREFRAME-1,:]
    relative = (target_pose-current_pos).numpy().tolist()
    if abs(relative[0]) < 0.1:
        if relative[1] > 0:
            orientation = torch.tensor([math.pi/2])
        else:
            orientation = torch.tensor([-math.pi/2])
    else:
        if relative[0] > 0:
            orientation = torch.tensor([math.atan(relative[1]/relative[0])])
        elif relative[1] > 0:
            orientation = torch.tensor([math.atan(relative[1]/relative[0])+math.pi])
        else:
            orientation = torch.tensor([math.atan(relative[1]/relative[0])-math.pi])
    return orientation

def physicalFeature(raw_data):
    return torch.cat([egoHistory(raw_data),egoVelocity(raw_data),\
        egoAcce(raw_data),surrPosition(raw_data),surrVelocity(raw_data),\
            surrType(raw_data),targetOrient(raw_data)],dim=0)

if __name__ == "__main__":
    sample = torch.load('G:/datasets/argo1/train_data_1/train46.pkl')
    data = sample[40]
    # model_input = physicalFeature(data)
    print("ego history: ",egoHistory(data))
    print("ego velocity: ",egoVelocity(data))
    print("ego acceleration: ",egoAcce(data))
    print("surrounding positions: ",surrPosition(data))
    print("surrounding type: ",surrType(data))
    print("surroundign velocity: ",surrVelocity(data))
    print("target orientation: ", targetOrient(data))
    # print(model_input)
