import os
import torch
import torch.utils.data
import torchvision.transforms as T
import numpy as np


class PDBDataset(torch.utils.data.Dataset):
    def __init__(self, inp_path, lab_path, train_scale=0.9, train=True):
        self.inp_path = inp_path
        self.lab_path = lab_path
        
        micros_tmp = torch.load(self.inp_path)
        labels_tmp = torch.load(self.lab_path)
        
        num = micros_tmp.shape[0]
        train_len = int(num*train_scale)
        
        if train:
            self.micros = micros_tmp[:train_len, :, :, :, :]
            self.labels = labels_tmp[:train_len, :, :, :, :]
        else:
            self.micros = micros_tmp[train_len:, :, :, :, :]
            self.labels = labels_tmp[train_len:, :, :, :, :]
    
    def __getitem__(self, idx):
        return self.micros[idx, :, :, :, :], self.labels[idx, :, :, :, :]

    def __len__(self):
        return len(self.micros)
