import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "D:/Workspace/UncertaintySafeField/highD/data/highd_changeLane/data.pickle"
data = pd.read_pickle(DATA_PATH)

axes = plt.subplot(1, 1, 1)
for id, traj in enumerate(data["position"]):
    # if id > 10:
    #     break
    plt.plot(traj[:, 0], traj[:, 1])
    plt.xlabel("x-axis(m)")
    plt.ylabel("y-axis(m)")
    plt.xlim(0, 350)
    plt.ylim(-6, 6)
plt.show()
