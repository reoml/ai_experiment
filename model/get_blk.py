# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 21:20:45 2022

@author: reoml
"""
import torch.nn as nn
from model.base_net import base_net
# import model.base_net.base_net as base_net
from model.Down_sample_blk import down_sample_blk
# import model.Down_sample_blk.Down_sample_blk as down_sample_blk
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk