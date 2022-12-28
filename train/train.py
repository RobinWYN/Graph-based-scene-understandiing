import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import trange

from transformers import get_linear_schedule_with_warmup
from backbone import VectorNetBackbone

class ArgData(Dataset):
    def __init__(self, raw_path, split):
        super().__init__()
        self.raw_path = raw_path
        self.data = []
        self.split = split

        if split == 'train':
            num = 2000
        else:
            num = 390

        for i in trange(num):

            path = self.raw_path + split + str(i) + '.pkl'
            with open(path, 'rb') as f:
                data = torch.load(f)

            self.data += data
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        s_data = self.data[index]
        return s_data


def train(epoch, lr, batch_size):

    raw_path_train = '../arg_data/train_data/'
    train_data = ArgData(raw_path_train, 'train')
    train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    raw_path_val = '../arg_data/val_data/'
    val_data = ArgData(raw_path_val, 'val')
    val_iter = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda:0")
    model = VectorNetBackbone(device).to(device)
    loss = nn.SmoothL1Loss(reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)

    t_total = len(train_iter) * epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

    for ith_epoch in trange(epoch):

        model.train()
        for idx, batch in enumerate(train_iter):

            optimizer.zero_grad()

            #print(batch['traj_feat'])
            gt = batch['ground_truth'].to(device)

            pred_traj = model(batch)
            los = loss(pred_traj.float(), gt.float())

            los.backward()
            optimizer.step()
            scheduler.step()

            if idx % 300 == 299:
                print('training loss', los)

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


epoch = 50
lr = 0.001
batch_size = 64
train(epoch, lr, batch_size)
