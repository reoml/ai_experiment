# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 21:53:15 2022

@author: reoml
"""


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks,cls_loss,bbox_loss):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox