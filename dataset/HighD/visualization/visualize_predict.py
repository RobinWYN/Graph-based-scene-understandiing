import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "D:/Workspace/UncertaintySafeField/predict/outputs/HighD_outputs/SA-GRU/gaussian_HighD.pkl"
# DATA_PATH = "D:/Workspace/UncertaintySafeField/predict/outputs/HighD_outputs/Base/gaussian_HighD.pkl"
data = pd.read_pickle(DATA_PATH)

from IPython import embed
# 

# axes = plt.subplot(1, 1, 1)
# for id, traj in enumerate(data["position"]):
#     # if id > 10:
#     #     break
#     plt.plot(traj[:, 0], traj[:, 1])
#     plt.xlabel("x-axis(m)")
#     plt.ylabel("y-axis(m)")
#     plt.xlim(0, 350)
#     plt.ylim(-6, 6)
# plt.show()

axes = plt.subplot(1, 1, 1)
key_id = [0,24,49,74]
# for i in range(400,419):
# i = range(400,419)
X_global = data['X_global'][:,:,0:2]
gt_trajs = data['gt_trajs'][:,:,:]
pred_trajs = data['pred_trajs'][:,:,:]
K = pred_trajs.shape[2]
tiled_target_traj = np.tile(gt_trajs[:, :, None, :], (1, 1, K, 1))

# for id in range(20):
#     plt.plot(pred_trajs[key_id,id,0],pred_trajs[key_id,id,1],'-og')
best_num = 4
eval_results = {}
traj_DE = np.linalg.norm(pred_trajs - tiled_target_traj, axis=-1)
# traj_DE_lon = np.abs(pred_trajs[:,:,:,0]-tiled_target_traj[:,:,:,0]) # longitudinal
# traj_DE_lat = np.abs(pred_trajs[:,:,:,1]-tiled_target_traj[:,:,:,1]) # lateral
best_result = np.zeros((traj_DE.shape[0],traj_DE.shape[1]))
# best_result_lon = np.zeros((traj_DE.shape[0],traj_DE.shape[1]))
# best_result_lat = np.zeros((traj_DE.shape[0],traj_DE.shape[1]))
for sample in range(traj_DE.shape[0]):
    # best_in_sample = []
    for ts in range(traj_DE.shape[1]):
        DE = traj_DE[sample][ts]
        best_id = np.argpartition(DE, best_num)[:best_num]
        best_result[sample][ts] = np.mean(DE[best_id])
        # best_result_lon[sample][ts] = np.mean(traj_DE_lon[sample][ts][best_id])
        # best_result_lat[sample][ts] = np.mean(traj_DE_lat[sample][ts][best_id])
    #     best_in_sample.append(np.mean(DE[best_id]))
    # best_result.append(best_in_sample)

best4 = np.mean(best_result, axis=0)
eval_results["best4_1S"] = best4[24]
eval_results["best4_2S"] = best4[49]
eval_results["best4_3S"] = best4[74]



# traj_ADE = np.linalg.norm(pred_trajs - tiled_target_traj, axis=-1).mean(1)
# traj_FDE = np.linalg.norm(pred_trajs - tiled_target_traj, axis=-1)[:, -1]
# eval_results['ADE'] = np.min(traj_ADE, axis=1).mean()
# eval_results['FDE'] = np.min(traj_FDE, axis=1).mean()

# traj_lon_MAE = np.abs(pred_trajs[:,:,:,0]-tiled_target_traj[:,:,:,0]).mean(1) # longitudinal
# traj_lat_MAE = np.abs(pred_trajs[:,:,:,1]-tiled_target_traj[:,:,:,1]).mean(1) # lateral
# eval_results['MAE_LON'] = np.min(traj_lon_MAE, axis=1).mean()
# eval_results['MAE_LAT'] = np.min(traj_lat_MAE, axis=1).mean()
# eval_results['MAE_LON'] = traj_lon_MAE[np.argpartition(traj_lon_MAE, 3,axis=1)[:,:3]].mean()
# eval_results['MAE_LAT'] = traj_lat_MAE[np.argpartition(traj_lat_MAE, 3,axis=1)[:,:3]].mean()

# traj_lon_MAE = np.abs(pred_trajs[:,:,:,0]-tiled_target_traj[:,:,:,0]).mean(2) # longitudinal
# traj_lat_MAE = np.abs(pred_trajs[:,:,:,1]-tiled_target_traj[:,:,:,1]).mean(2) # lateral
# eval_results['per_step_MAE_LON'] = np.mean(traj_lon_MAE, axis=0)
# eval_results['per_step_MAE_LAT'] = np.mean(traj_lat_MAE, axis=0)

best = np.linalg.norm(pred_trajs - tiled_target_traj, axis=-1).min(axis=2).mean(axis=0)
# eval_results['DE_0_5s'] = eval_results['per_step_displacement_error'][12]
eval_results['best_1s'] = best[24]
eval_results['best_2s'] = best[49]
eval_results['best_3s'] = best[74]

# embed()
print(eval_results)
# plt.plot(X_global[:,0],X_global[:,1])
# plt.plot(gt_trajs[:,0],gt_trajs[:,1])
# plt.show()
