import os
import json
import time
import pathlib

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import trange
from typing import Any, Optional, List
import numpy as np

NUM_DATA_PACK = 133
NUM_DATA = 8000

class ArgDataHighD(Dataset):
    def __init__(self, raw_path, split, ind: np.ndarray = Any, train_num = 80):
        super().__init__()
        self.raw_path = raw_path
        self.data = []
        self.split = split

        if split == "train":
            for i in ind[:train_num]:
                path = self.raw_path + str(i) + ".pkl"
                with open(path, "rb") as f:
                    data = torch.load(f)
                self.data += data
            print("training data loaded successfully")

        else:
            for i in ind[train_num:]:
                path = self.raw_path + str(i) + ".pkl"
                with open(path, "rb") as f:
                    data = torch.load(f)
                self.data += data
            print("validation data loaded successfully")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        s_data = self.data[index]
        return s_data
    
    
class ArgDataArgo(Dataset):
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
        print(split + " data loaded successfully")
        
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        s_data = self.data[index]
        return s_data


class ArgDataV2(Dataset):
    def __init__(self, raw_path: str = Any):
        super().__init__()
        self.raw_path = raw_path
        self.data = []
        for i in range(NUM_DATA_PACK):
            path = self.raw_path + str(i) + ".pkl"
            with open(path, "rb") as f:
                data = torch.load(f)
            self.data += data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        s_data = self.data[index]
        return s_data