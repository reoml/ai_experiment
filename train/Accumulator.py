# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 21:45:46 2022

@author: reoml
"""

class Accumulator:
    """
    在‘n’个变量上累加
    """
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[a+float(b) for a,b in zip(self.data,args)]
    def reset(self):
        self.data=[0.0]*len(self.data)
    def _getitem_(self,idx):
        return self.data[idx]