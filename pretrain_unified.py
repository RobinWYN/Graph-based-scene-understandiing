import os
import json
import time
import pathlib
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import trange
from typing import Any, Optional, List

from transformers import get_linear_schedule_with_warmup

# from backbone import VectorNetBackbone
from models.backbone_test import VectorNetBackbone
#from baseline import Transformer, LSTM, LSTM_interact
from models.baselines import Transformer, LSTM, LSTM_interact
from models.loss import PredLoss, ClsLoss, PreTrainLoss

from preprocess.preprocess_HighD import test
from argument_parser import args
from visualization.visualization import training_plot

from dataset.dataset import ArgDataArgo, ArgDataHighD, ArgDataV2

NUM_DATA_PACK = 133
NUM_DATA = 8000
DATA_DIR = "../highD/highD_pkl_1-51_large"

def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def train(
    hyperparams,
    model_dir,
    load_checkpoint=False,
    check_path: Optional[str] = None
):
    epoch = hyperparams["epochs"]
    lr = hyperparams["lr"]
    batch_size = hyperparams["batch_size"]
    schedule = hyperparams["schedule"]
    device = hyperparams["device"]

    ind = np.arange(NUM_DATA_PACK)
    np.random.shuffle(ind)

    raw_path_train = DATA_DIR + "/HighD_VIFGNN_"
    train_data = ArgDataArgo(raw_path_train, "train", ind)
    train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    raw_path_val = DATA_DIR + "/HighD_VIFGNN_"
    val_data = ArgDataArgo(raw_path_val, "val", ind)
    val_iter = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    """
    raw_path = '/home/xyn/highD_pkl_1-51/HighD_VIFGNN_'
    dataset = ArgDataV2(raw_path)
    train_iter, val_iter = torch.utils.data.random_split(dataset, [6500, 1500])
    """

    model = VectorNetBackbone(device).to(device)
    #model = Transformer(hyperparams, device).to(device)
    #model = LSTM(device).to(device)

    if load_checkpoint == True:
        checkpoint = torch.load(check_path + "model_parameter.pkl")
        model.load_state_dict(checkpoint)

    # loss = PredLoss()
    loss = PreTrainLoss(alpha=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    t_total = len(train_iter) * epoch

    if schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=t_total
        )
    elif schedule == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.65)
    elif schedule == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epoch, eta_min=0
        )
    # print(len(train_iter))
    # print(len(val_iter))
    loss_log = []

    for ith_epoch in trange(epoch):
        model.train()
        for idx, batch in enumerate(train_iter):
            optimizer.zero_grad()
            gt = batch["ground_truth"].to(device)
            o_h = np.zeros((len(gt), 3))
            o_h[np.arange(len(gt)), gt.cpu().reshape(1, -1)] = 1
            gt_onehot = torch.from_numpy(o_h).to(device)
            dec_cls = model(batch)
            los = loss(dec_cls.float(), gt_onehot.float())
        
            los.backward()
            optimizer.step()
            scheduler.step()

            if idx % 10 == 9:
                print("training loss", los)

            loss_log.append(los.detach().cpu())

        model.eval()
        with torch.no_grad():
            ade = 0
            fde = 0
            for batch in val_iter:
                gt = batch['ground_truth'].to(device)
                pred_traj = model(batch)
                ade += torch.norm(pred_traj - gt, p=2, dim=-1).mean(dim=-1).sum()
                fde += torch.norm(pred_traj[:, -1] - gt[:, -1], p=2, dim=-1).sum()

            print('epoch:', ith_epoch)
            print('ADE:', ade / len(val_data))
            print('FDE', fde / len(val_data))

    loss_log = np.array(loss_log)
    
    torch.save(
        model.state_dict(),
        model_dir + f"/model_parameter_{epoch}.pkl",
    )


if __name__ == "__main__":
    
    if torch.cuda.is_available():
        args.device = f"cuda"
    else:
        args.device = f"cpu"

    if not os.path.exists(args.conf):
        hyperparams={}
        print(f"Config json at {args.conf} not found!")
        #raise ValueError(f"Config json at {args.conf} not found!")
    else:
        with open(args.conf, "r", encoding="utf-8") as conf_json:
            hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams.update(
        {k: v for k, v in vars(args).items() if v is not None}
    )  # here set map_encoding to False

    # set training log
    model_dir_folder = hyperparams["tag"] + time.strftime("-%d_%b_%Y_%H_%M_%S", time.localtime())
    model_dir = os.path.join(hyperparams["log_dir"], model_dir_folder)
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(model_dir, "config.json"), 'w') as config_json:
        json.dump(hyperparams, config_json)
    print("model dir:", model_dir)
    
    train(hyperparams, model_dir, load_checkpoint=False)
