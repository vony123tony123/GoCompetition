import os
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.functional import one_hot

class GoDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None)->None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        chars = 'abcdefghijklmnopqrs'
        self.coordinates = {k:v for v,k in enumerate(chars)}
        self.chartonumbers = {k:v for k,v in enumerate(chars)}
        self.datas, self.labels = self.load_data(self.root)

    def load_data(self, root):
        df = open(root).read().splitlines()
        games = [i.split(',',2)[-1] for i in df]
        # Check how many samples can be obtained
        n_games = 0
        n_moves = 0
        for game in games:
            n_games += 1
            moves_list = game.split(',')
            for move in moves_list:
                n_moves += 1
        print(f"Total Games: {n_games}, Total Moves: {n_moves}")

        x = []
        y = []
        for game in games:
            moves_list = game.split(',')
            for count, move in enumerate(moves_list):
                x.append(self.prepare_input(moves_list[:count]))
                y.append(self.prepare_label(moves_list[count]))
        x = np.array(x)
        y = np.array(y)

        y_one_hot = one_hot(y, num_classes=19*19)
        x = torch.from_numpy(x)

        return x, y_one_hot
    
    def prepare_input(self, moves):
        x = np.zeros((19,19,4))
        for move in moves:
            color = move[0]
            column = self.coordinates[move[2]]
            row = self.coordinates[move[3]]
            if color == 'B':
                x[row,column,0] = 1
                x[row,column,2] = 1
            if color == 'W':
                x[row,column,1] = 1
                x[row,column,2] = 1
        if moves:
            last_move_column = self.coordinates[moves[-1][2]]
            last_move_row = self.coordinates[moves[-1][3]]
            x[last_move_column,last_move_row,3] = 1
        x[:,:,2] = np.where(x[:,:,2] == 0, 1, 0)
        return x

    def prepare_label(self, move):
        column = self.coordinates[move[2]]
        row = self.coordinates[move[3]]
        return column*19+row
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        x, y = self.datas[idx], self.labels[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

def get_GoDataLoader(root, valid_size=0.2, test_size = 0.1, batch_size=27558, shuffle=True):
    dataset = GoDataset(root)
    num_train = len(dataset)
    print(num_train)
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
