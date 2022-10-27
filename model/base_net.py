# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 21:21:08 2022

@author: reoml
"""
import torch.nn as nn
# import sys
# sys.path.append("..")
# print(sys.path)
from model.Down_sample_blk import down_sample_blk
# import model.Down_sample_blk.Down_sample_blk as down_sample_blk
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)