from matplotlib import pyplot as plt
import numpy as np


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


def training_plot(loss: np.ndarray, acc: np.ndarray, save_dir: str, show=False):
    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(np.arange(loss.shape[0]), loss)
    ax1.legend("training loss")

    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(np.arange(acc.shape[0]), acc)
    ax2.legend("eval acc")
    plt.savefig(save_dir + "/training_curves")

    if show:
        plt.show()

    plt.close()
    
