"""
@Time ： 2024/10/10 13:52
@Auth ： KECILIMU
@File ：deep_network.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
import torch
import torch.nn as nn

class deep_ex(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, output_dimension, drop_ratio):
        super(deep_ex, self).__init__()
        self.ln = nn.LayerNorm(hidden_dimension)
        self.fc = nn.Linear(input_dimension, hidden_dimension)
        self.fc2 = nn.Linear(hidden_dimension, hidden_dimension)
        self.fc3 = nn.Linear(hidden_dimension, hidden_dimension)
        self.fc4 = nn.Linear(hidden_dimension, hidden_dimension)
        self.fc5 = nn.Linear(hidden_dimension, hidden_dimension)
        self.fc6 = nn.Linear(hidden_dimension, hidden_dimension)
        self.fc7 = nn.Linear(hidden_dimension, output_dimension)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.drop = nn.Dropout(drop_ratio)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.fc4.weight)
        nn.init.kaiming_normal_(self.fc5.weight)
        nn.init.kaiming_normal_(self.fc6.weight)
        nn.init.kaiming_normal_(self.fc7.weight)

    def forward(self, x):
        ori_out = self.ln(self.relu(self.fc(x)))
        # ori_out = self.relu(self.fc(x))
        out = self.relu(self.fc2(ori_out))
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        sen_out = self.drop(self.relu(self.fc5(torch.add(out, ori_out))))
        out = self.relu(self.fc6(sen_out))
        out = self.drop(self.relu(self.fc7(out)))
        return out
