# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 21:29:27 2022

@author: reoml
"""
import torch
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)