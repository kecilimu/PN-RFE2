
"""
@Time ： 2024/10/4 21:04
@Auth ： KECILIMU
@File ：myDataset.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
# Name: mydataset
# Time: 2024/9/23
# Auther: "Kecilimu"
from torch.utils.data import Dataset


class myDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
