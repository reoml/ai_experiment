# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 21:29:52 2022

@author: reoml
"""
import torch
from model.flatten_pred import flatten_pred


def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)