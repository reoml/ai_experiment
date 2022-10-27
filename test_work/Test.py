# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 23:44:36 2022

@author: reoml
"""

from test_work.predict import Predict
from test_work.display import display
from model.TinySSD import TinySSD
import torch
import torchvision


def Test(num_classes,sizes,ratios,num_anchors,net_name):
    
    net = TinySSD(num_classes,sizes,ratios,num_anchors)
    net = net.to('cpu')
    
    # 加载模型参数
    net.load_state_dict(torch.load(net_name, map_location=torch.device('cpu')))
    
          
    name = 'test/R-C.jpg'
    X = torchvision.io.read_image(name).unsqueeze(0).float()
    img = X.squeeze(0).permute(1, 2, 0).long()
    
    output = Predict(X,net)    
    display(img, output.cpu(), threshold=0.6)