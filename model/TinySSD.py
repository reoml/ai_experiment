# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 21:13:27 2022

@author: reoml
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from model.get_blk import get_blk
import model.predictor as predictor
from model.blk_forward import blk_forward
from model.concat_preds import concat_preds

# sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
#          [0.88, 0.961]]
# ratios = [[1, 2, 0.5]] * 5
# num_anchors = len(sizes[0]) + len(ratios[0]) - 1


class TinySSD(nn.Module):
    def __init__(self, num_classes,sizes,ratios,num_anchors, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.sizes = sizes
        self.ratios = ratios
        self.num_anchors = num_anchors
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', predictor.cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', predictor.bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), self.sizes[i], self.ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds