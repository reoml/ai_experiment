# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 21:32:49 2022

@author: reoml
"""
import torch
import torch.nn as nn
import torchvision
from train.Accumulator import Accumulator
from train.multibox_target import multibox_target
from train.calc_loss import calc_loss
import train.Eval  as Eval

def Train(net,num_epochs,train_iter,cls_loss,bbox_loss,lr,weight_decay):
    # sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
    #          [0.88, 0.961]]
    # ratios = [[1, 2, 0.5]] * 5
    # num_anchors = len(sizes[0]) + len(ratios[0]) - 1
    # net = TinySSD(num_classes=1,sizes,ratios,num_anchors)
    # net = net.to('cuda')
    trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
    
    num_epochs = num_epochs #20
    for epoch in range(num_epochs):
        print('epoch: ', epoch)
        # 训练精确度的和，训练精确度的和中的示例数
        # 绝对误差的和，绝对误差的和中的示例数
        metric = Accumulator(4)
        net.train()
        for features, target in train_iter:
            trainer.zero_grad()
            X, Y = features.to('cuda'), target.to('cuda')
            # 生成多尺度的锚框，为每个锚框预测类别和偏移量
            anchors, cls_preds, bbox_preds = net(X)
            # 为每个锚框标注类别和偏移量
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
            # 根据类别和偏移量的预测和标注值计算损失函数
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                          bbox_masks,cls_loss,bbox_loss)
            l.mean().backward()
            trainer.step()
            metric.add(Eval.cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                        Eval.bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                        bbox_labels.numel())
        cls_err, bbox_mae = 1 - metric.data[0] / metric.data[1], metric.data[2] / metric.data[3]
        print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
    
    
        # 保存模型参数
        if epoch % 10 == 0:
            torch.save(net.state_dict(), 'net_' + str(epoch) + '.pkl')
     