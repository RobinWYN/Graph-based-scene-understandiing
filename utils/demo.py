from matplotlib import pyplot as plt
import numpy as np

from models.backbone_test import VectorNetBackbone
from models.baselines import Transformer, LSTM, LSTM_interact


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
                plt.plot([traj_start[traj_seg, 0], traj_end[traj_seg, 0]], [traj_start[traj_seg, 1], traj_end[traj_seg, 1]], color='b')
                plt.scatter([traj_start[traj_seg, 0], traj_end[traj_seg, 0]], [traj_start[traj_seg, 1], traj_end[traj_seg, 1]], s=10, color='b')
    
    plt.show()


def result_plot(model_type: str = "VIF", trial_num: str = "04"):
    raw_path = "/home/xyn/training_codes/results/experiment_"
    assert(model_type != None), "please choose model for inspection!"
    path = raw_path + model_type + trial_num + '.npy'
    acc = np.load(path)
    x = np.arange(len(acc))
    plt.plot(x, acc)
    plt.show()
    

def param_visual(model, name: str = None):
    params = model.state_dict()
    #print(params)
    if name == None :
        pass
    elif name == 'aa_interaction':
        par = params[name + '.fc' + '.weight']
        print(params[name + '.fc' + '.weight'])
    
    plt.imshow(par)
    plt.tight_layout()
    plt.show()    
    
    
def loading(model_type: str = "transformer", trial_num: str = "04"):
    raw_path = "/home/xyn/training_codes/model_para/model_parameter_"
    assert(model_type != None), "please choose model for inspection!"
    path = raw_path + model_type + trial_num + '.pkl'

    if model_type == "VIF":
        model = VectorNetBackbone('cuda')
    elif model_type == "transformer":
        model =  Transformer('cuda')
    model.eval()
    param_visual(model, 'aa_interaction')
    


if __name__ == "__main__" :
    #loading()
    result_plot()