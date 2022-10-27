# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 17:05:34 2022

@author: reoml
"""

from torchvision import datasets
from  torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import torch
import torchvision
import numpy as np
import cv2
import dataset.create_train as create_train
def load_data(batch_size):
    train_iter = torch.utils.data.DataLoader(Dataset(is_train = True),batch_size,shuffle = True)
    return train_iter;
class Dataset(torch.utils.data.Dataset):
    def __init__(self, is_train):
        self.features, self.labels = read_data(is_train);
        print('read ' + str(len(self.features)) + (f' training examples' if is_train else f' val validation examples'));
        
    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])
        
    def __len__(self):
        return len(self.features)
def read_data(is_train = True):
    data_dir = 'detection_work/dataset/'
    #csv_fname = os.path.join(data_dir, 'sysu_train' if is_train else 'sysu_val', 'label.csv');
    csv_fname = os.path.join('sysu_train' if is_train else 'sysu_val', 'label.csv');
    csv_fname = csv_fname.replace('\\', "/")
    csv_data = pd.read_csv(csv_fname);
    csv_data = csv_data.set_index('img_name');
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        #images.append(torchvision.io.read_image(os.path.join(data_dir, 'sysu_train' if is_train else 'sysu_val','images',f'{img_name}')));
        images.append(torchvision.io.read_image(os.path.join('sysu_train' if is_train else 'sysu_val','images',f'{img_name}').replace('\\', "/")));
        targets.append(list(target))
    return images,torch.tensor(targets).unsqueeze(1) / 256;
def show_img(img_tensor):
    img_arr = img_tensor.numpy();
    #max_va  = img_arr.max();
    #print(max_va);
    #img_arr = img_arr * 255 / (max_va);
    img     = np.uint8(img_arr);
    img     = img.transpose(1, 2, 0);
    img     = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imshow('',img);
    cv2.waitKey(0)
    return;
def data_process(batchsizes,istrain,is_data_create):
    batch_size = batchsizes;
    is_train = istrain;
    if is_data_create:
        print('create_data')
        create_train.Create_train();
    train_iter = load_data(batch_size)
    print('get_train_iter')
    #print(os.path.join('data_dir/', 'sysu_train' if is_train else 'sysu_val', 'label.csv'))
    return train_iter;
        #in_arr =np.transpose(img,(1,2,0))
        #cvimg = cv2.cvtColor(np.uint8(in_arr*255))
    