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
from models.loss import PredLoss, ClsLoss

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
    load_checkpoint=False
):
    epoch = hyperparams["epochs"]
    lr = hyperparams["lr"]
    batch_size = hyperparams["batch_size"]
    schedule = hyperparams["schedule"]
    device = hyperparams["device"]

    ind = np.arange(NUM_DATA_PACK)
    np.random.shuffle(ind)

    check_path = "./data/"

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
    loss = ClsLoss()
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
    accuracy = []

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
            conf_matrix = torch.zeros(3, 3).cpu()
            acc = 0
            for idx, batch in enumerate(val_iter):
                """
                if idx == len(train_iter) - 1:
                    break # can not exact divide here
                """
                gt = batch["ground_truth"].to(device)
                # print(gt)
                dec_cls = model(batch)
                pred_result = torch.argmax(dec_cls, axis=-1).reshape(-1, 1)

                # print("pred_result", pred_result)
                # print("acc", torch.where(gt==pred_result, 1., 0.))
                Pred = pred_result.cpu()
                Gt = gt.cpu()
                acc += torch.where(gt == pred_result, 1.0, 0.0).sum(dim=0).item()
                conf_matrix = confusion_matrix(Pred, Gt, conf_matrix)
                # conf_matrix = conf_matrix.cpu()
            conf_matrix = np.array(conf_matrix.cpu())  # 将混淆矩阵从gpu转到cpu再转到np
            corrects = conf_matrix.diagonal(offset=0)  # 抽取对角线的每种分类的识别正确个数
            per_kinds = conf_matrix.sum(axis=1)

            print("epoch:", ith_epoch)
            print("accuracy", acc / len(val_data))
            if ith_epoch == epoch - 1:
                print("confusion matrix:", conf_matrix)
            print(
                "accuracy for each kind:{0}".format(
                    [rate * 100 for rate in corrects / per_kinds]
                )
            )
            accuracy.append(acc / len(val_data))

    loss_log = np.array(loss_log)
    accuracy = np.array(accuracy)
    np.save(model_dir + "/accuracy_log.npy", accuracy)

    training_plot(loss_log.reshape(-1), accuracy, model_dir)

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
