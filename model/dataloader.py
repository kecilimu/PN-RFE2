# -*- coding: utf-8 -*-
"""
@Time ： 2024/10/4 17:26
@Auth ： KECILIMU
@File ：dataloader.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
import time
import torch
from torch.utils.data import random_split
import pickle

from File_log import IOstream as ios
import utils
from File_log.file_create import torch_seed_set
from model.myDataset import myDataset


def data_gen(device):
    torch_seed_set()
    ori_data = ios.load_list_from_excel(utils.data_path)
    sample_number = len(ori_data)
    if device.type == 'cuda':
        ori_data = torch.FloatTensor(ori_data).to(device)
    train_size = int(sample_number * utils.train_size_ratio)
    val_size = sample_number - train_size
    data, label = ori_data[:, :-1], ori_data[:, -1]/1000
    dataset = myDataset(data, label)
    train_data, val_data = random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=utils.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=utils.batch_size_, shuffle=True)

    return train_loader, val_loader

def get_all_data(dataloder):
    all_data = None
    all_label = None
    for data, label in dataloder:
        if all_data is None:
            all_data = data
            all_label = label
        else:
            all_data = torch.cat((all_data, data), 0)
            all_label = torch.cat((all_label, label), 0)
    return all_data, all_label

def data_load_s(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data