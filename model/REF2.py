
"""
@Time ： 2024/10/4 17:26
@Auth ： KECILIMU
@File ：REF2.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
import torch.nn as nn
from model.wid_network import wid_ex
from model.deep_network import deep_ex
from model.regression import regression

class REF2(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, output_dimension, drop_ratio):
        super(REF2, self).__init__()
        self.wex = wid_ex(input_dimension, hidden_dimension[0], hidden_dimension[1], drop_ratio)
        self.dex = deep_ex(hidden_dimension[1], hidden_dimension[2], hidden_dimension[3], drop_ratio)
        self.sl_dex = deep_ex(hidden_dimension[3], hidden_dimension[4], hidden_dimension[5], drop_ratio)
        self.re = regression(hidden_dimension[5], hidden_dimension[6], output_dimension, drop_ratio)

    def forward(self, x):
        out = self.wex(x)
        out_ex = self.dex(out)
        out_ex = self.sl_dex(out_ex)
        out = self.re(out_ex)
        return out, out_ex