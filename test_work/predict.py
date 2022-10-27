# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 23:46:47 2022

@author: reoml
"""
import torch.nn.functional as F
from test_work.multibox_detection import multibox_detection
def Predict(X,net):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to('cpu'))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]
      