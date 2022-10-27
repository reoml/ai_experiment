# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 21:26:25 2022

@author: reoml
"""
from model.multibox_prior import multibox_prior
# import model.multibox_prior.multibox_prior as multibox_prior
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)