import os
import numpy as np
import torch
import random
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.functional import one_hot
from torchvision.transforms import ToTensor
from utils import *

class Kyu_Dan_Dataset(Dataset):
    def __init__(self, csv_path, transform=None, target_transform=None)->None:
        self.transform = transform
        self.target_transform = target_transform
        self.games = self.init(csv_path)
        self.datas, self.labels = None, None
        print(f"Total Games: {len(self.games)}")

    def init(self, csv_path):
        df = open(csv_path).read().splitlines()
        random.shuffle(df)
        return [i.split(',',1)[-1] for i in df]

    def load_data(self, idx_start, idx_end):
        print("---------- Prepare dataset from", idx_start, "to", idx_end, "----------\n")
        x = []
        y = []
        for game in tqdm(self.games[idx_start:idx_end]):
            moves_list = game.split(',')
            color = moves_list.pop(0)
            for count in range(0, len(moves_list)):
                if (color=="B" and count%2==0) or (color=="W" and count%2==1):
                    x.append(prepare_input(moves_list[:count]))
                    y.append(prepare_label(moves_list[count]))

        x = np.array(x).astype(np.float32)
        y = torch.LongTensor(y)

        y_one_hot = one_hot(y, num_classes=19*19)
        y_one_hot = y_one_hot.float()
        self.datas, self.labels = x, y_one_hot

    def clean(self):
        del self.datas, self.labels
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        x, y = self.datas[idx], self.labels[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

    def get_nums_games(self):
        return len(self.games)

class Playstyle_Dataset(Dataset):
    def __init__(self, csv_path, transform=None, target_transform=None)->None:
        self.transform = transform
        self.target_transform = target_transform
        self.games = self.init(csv_path)
        self.datas, self.labels = None, None
        print(f"Total Games: {len(self.games)}")

    def init(self, csv_path):
        df = open(csv_path).read().splitlines()
        random.shuffle(df)
        return [i.split(',',1)[-1] for i in df]

    def load_data(self, idx_start, idx_end):
        print("---------- Prepare dataset from", idx_start, "to", idx_end, "----------\n")
        x = []
        y = []
        for game in tqdm(self.games[idx_start:idx_end]):
            moves_list = game.split(',')
            style = int(moves_list.pop(0)) - 1
            color = moves_list[-1][0]
            for count in range(0, len(moves_list)):
                if (color=="B" and count%2==0) or (color=="W" and count%2==1):
                    x.append(prepare_input(moves_list[:count]))
                    y.append(style)

        x = np.array(x).astype(np.float32)
        y = torch.LongTensor(y)

        y_one_hot = one_hot(y, num_classes=3)
        y_one_hot = y_one_hot.float()
        self.datas, self.labels = x, y_one_hot

    def clean(self):
        del self.datas, self.labels
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        x, y = self.datas[idx], self.labels[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

    def get_nums_games(self):
        return len(self.games)

def get_GoDataLoader(dataset, valid_size=0.2, test_size = 0.1, batch_size=1024, shuffle=True):
    num_train = len(dataset)
    indices = list(range(num_train))
    if shuffle:
        np.random.shuffle(indices)
    valid_split = int(np.floor((valid_size) * num_train))
    test_split = int(np.floor((valid_size+test_size) * num_train))
    valid_idx, test_idx, train_idx = indices[:valid_split], indices[valid_split:test_split], indices[test_split:]

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(valid_idx))
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_idx))

    return train_loader, valid_loader, test_loader
