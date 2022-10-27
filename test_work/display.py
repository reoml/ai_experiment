# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 23:47:39 2022

@author: reoml
"""
import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from test_work.show_bboxes import show_bboxes
def display(img, output, threshold):
    fig = plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
