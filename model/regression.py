
"""
@Time ： 2024/10/4 22:03
@Auth ： KECILIMU
@File ：regression.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
import torch
import torch.nn as nn


class regression(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, output_dimension, drop_ratio):
        super(regression, self).__init__()
        self.fc = nn.Linear(input_dimension, hidden_dimension)
        self.fc2 = nn.Linear(hidden_dimension, output_dimension)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.drop = nn.Dropout(drop_ratio)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        out = self.drop(self.relu(self.fc(x)))
        out = self.fc2(out)
        return out