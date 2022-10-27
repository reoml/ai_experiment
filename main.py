# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 22:01:29 2022

@author: reoml
"""
import torch
from dataset.data_process import data_process 

from model.TinySSD import TinySSD

from test_work.Test import Test 
from train.train import Train
import torch.nn as nn

if __name__ == '__main__':
    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
              [0.88, 0.961]]
    ratios = [[1, 2, 0.5]] * 5
    num_anchors = len(sizes[0]) + len(ratios[0]) - 1
    num_classes = 1;
    batchsize = 32;
    istrain = True;
    num_epochs = 20;
    net_name = 'net_10.pkl';
    is_data_create = False;
    lr = 0.2;
    weight_decay=5e-4
    cls_loss = nn.CrossEntropyLoss(reduction='none')
    bbox_loss = nn.L1Loss(reduction='none')
    net = TinySSD(num_classes,sizes,ratios,num_anchors)
    net = net.to('cuda')
    train_iter = data_process(batchsize, istrain, is_data_create)
    Train(net,num_epochs,train_iter,cls_loss,bbox_loss,lr,weight_decay)
    Test(num_classes,sizes,ratios,num_anchors,net_name)



