import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy

class CNN_32(nn.Module):
    """
        CNN
    """
    def __init__(self,  nc=1, ngf=32):
        super(CNN_32, self).__init__()
        self.conv_net = nn.Sequential(nn.ConvTranspose2d(nc,ngf,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ngf),
          nn.ReLU(),
          nn.ConvTranspose2d(ngf,ngf*2,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ngf*2),
          nn.ReLU(),
          nn.Conv2d(ngf*2,nc,kernel_size=4,stride=2,padding=1),
          nn.Tanh()
        )
    def forward(self, x, y=None):
        y_ = x.view(x.size(0), 1, 32, 32)
        y_ = self.conv_net(y_)
        return y_

class CNN_MAT_32(nn.Module):
    """
        CNN w/ Moran's Auxiliary Task
    """
    def __init__(self,  nc=1, ngf=32):
        super(CNN_MAT_32, self).__init__()
        self.conv_net = nn.Sequential(nn.ConvTranspose2d(nc,ngf,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ngf),
          nn.ReLU()
        )
        self.output_t1 = nn.Sequential(nn.ConvTranspose2d(ngf,ngf*2,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ngf*2),
          nn.ReLU(),
          nn.Conv2d(ngf*2,nc,kernel_size=4,stride=2,padding=1),
          nn.Tanh()
        )
        self.output_t2 = nn.Sequential(nn.ConvTranspose2d(ngf,ngf*2,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ngf*2),
          nn.ReLU(),
          nn.Conv2d(ngf*2,nc,kernel_size=4,stride=2,padding=1),
          nn.Tanh()
        )
    def forward(self, x, y=None):
        mi_x = batch_lw_tensor_local_moran(x.detach().cpu(),w_sparse_32)
        mi_x = mi_x.to(DEVICE)
        y_ = x.view(x.size(0), 1, 32, 32)
        mi_y_ = mi_x.view(mi_x.size(0), 1, 32, 32)
        y_ = self.conv_net(y_)
        mi_y_ = self.conv_net(mi_y_)
        y_ = self.output_t1(y_)
        mi_y_ = self.output_t2(mi_y_)
        return y_, mi_y_