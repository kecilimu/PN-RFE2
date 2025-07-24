
"""
@Time ： 2024/10/4 22:03
@Auth ： KECILIMU
@File ：wid_network.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
import torch
import torch.nn as nn

class wid_ex(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, output_dimension, drop_ratio):
        super(wid_ex, self).__init__()
        self.ln = nn.LayerNorm(hidden_dimension)
        self.fc = nn.Linear(input_dimension, hidden_dimension)
        self.fc2 = nn.Linear(hidden_dimension, hidden_dimension)
        self.fc3 = nn.Linear(hidden_dimension, output_dimension)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.drop = nn.Dropout(drop_ratio)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)


    def forward(self, x):
        out = self.ln(self.relu(self.fc(x)))
        # out = self.relu(self.fc(x))
        out = self.relu(self.fc2(out))
        out = self.drop(self.relu(self.fc3(out)))
        return out