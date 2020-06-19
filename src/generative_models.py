import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy

###
# VANILLA GAN
###

class Discriminator_VanillaGAN(nn.Module):
    """
        Simple Discriminator w/ MLP
    """
    def __init__(self, input_size=N*N, num_classes=1):
        super(Discriminator_VanillaGAN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, num_classes),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        y_ = x.view(x.size(0), -1)
        y_ = self.layer(y_)
        return y_

class Discriminator_VanillaGAN_MAT(nn.Module):
    """
        Simple Discriminator w/ MLP
    """
    def __init__(self, input_size=N*N, num_classes=1):
        super(Discriminator_VanillaGAN_MAT, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2)
        )
        self.output_t1 = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )
        self.output_t2 = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )
    def forward(self, x):
        if N==32:
          mi_x = batch_lw_tensor_local_moran(x.detach().cpu(),w_sparse_32)
        if N==64:
          mi_x = batch_lw_tensor_local_moran(x.detach().cpu(),w_sparse_64)
        mi_x = mi_x.to(DEVICE)
        y_ = x.view(x.size(0), -1)
        y_ = self.layer(y_)
        mi_x = mi_x.view(mi_x.size(0), -1)
        mi_y_ = self.layer(mi_x)
        y_ = self.output_t1(y_)
        mi_y_ = self.output_t2(mi_y_)
        return y_, mi_y_

class Discriminator_VanillaGAN_MRES_MAT(nn.Module):
    """
        Simple Discriminator w/ MLP
    """
    def __init__(self, input_size=N*N, num_classes=1):
        super(Discriminator_VanillaGAN_MRES_MAT, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2)
        )
        self.output_t1 = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )
        self.output_t2 = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )
        self.output_t3 = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )
        self.output_t4 = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )   
    def forward(self, x):
        x_d1 = downsample(x)
        x_d2 = downsample(x_d1)
        if N==32:
          mi_x = batch_lw_tensor_local_moran(x.detach().cpu(),w_sparse_32)
          mi_x_d1 = batch_lw_tensor_local_moran(x_d1.detach().cpu(),w_sparse_16)
          mi_x_d2 = batch_lw_tensor_local_moran(x_d2.detach().cpu(),w_sparse_8)
        if N==64:
          mi_x = batch_lw_tensor_local_moran(x.detach().cpu(),w_sparse_64)
          mi_x_d1 = batch_lw_tensor_local_moran(x_d1.detach().cpu(),w_sparse_32)
          mi_x_d2 = batch_lw_tensor_local_moran(x_d2.detach().cpu(),w_sparse_16)
        mi_x = mi_x.to(DEVICE)
        mi_x_d1 = mi_x_d1.to(DEVICE)
        mi_x_d2 = mi_x_d2.to(DEVICE)
        mi_x_d1 = nn.functional.interpolate(mi_x_d1,scale_factor=2,mode="nearest")
        mi_x_d2 = nn.functional.interpolate(mi_x_d2,scale_factor=4,mode="nearest")
        y_ = x.view(x.size(0), -1)
        y_ = self.layer(y_)
        mi_x = mi_x.view(mi_x.size(0), -1)
        mi_y_ = self.layer(mi_x)
        mi_x_d1 = mi_x_d1.view(mi_x_d1.size(0), -1)
        mi_y_d1 = self.layer(mi_x_d1)
        mi_x_d2 = mi_x_d2.view(mi_x_d2.size(0), -1)
        mi_y_d2 = self.layer(mi_x_d2)
        y_ = self.output_t1(y_)
        mi_y_ = self.output_t2(mi_y_)
        mi_y_d1 = self.output_t3(mi_y_d1)
        mi_y_d2 = self.output_t4(mi_y_d2)
        return y_, mi_y_, mi_y_d1, mi_y_d2

class Generator_VanillaGAN(nn.Module):
    """
        Simple Generator w/ MLP
    """
    def __init__(self, input_size=100, num_classes=N*N):
        super(Generator_VanillaGAN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, num_classes),
            nn.Tanh()
        )      
    def forward(self, x):
        y_ = self.layer(x)
        y_ = y_.view(x.size(0), 1, N, N)
        return y_

###
# DCGAN (also used for WGAN)
###

class Discriminator_DCGAN_32(nn.Module):
    """
        Convolutional Discriminator 
    """
    def __init__(self, in_channel=1, num_classes=1):
        super(Discriminator_DCGAN_32, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(4),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )    
    def forward(self, x, y=None):
        y_ = self.conv(x)
        y_ = y_.view(y_.size(0), -1)
        y_ = self.fc(y_)
        return y_

class Discriminator_DCGAN_MAT_32(nn.Module):
    """
        DeepConv Discriminator
    """
    def __init__(self, in_channel=1, num_classes=1):
        super(Discriminator_DCGAN_MAT_32, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.output_t1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(4)
        )
        self.output_t2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(4)
        )
        self.fc_t1 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.fc_t2 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    def forward(self, x, y=None):
        mi_x = batch_lw_tensor_local_moran(x.detach().cpu(),w_sparse_32)
        mi_x = mi_x.to(DEVICE)
        y_ = self.conv(x)
        mi_y_ = self.conv(mi_x)
        y_ = self.output_t1(y_)
        mi_y_ = self.output_t2(mi_y_)
        y_ = y_.view(y_.size(0), -1)
        mi_y_ = mi_y_.view(mi_y_.size(0), -1)
        y_ = self.fc_t1(y_)
        mi_y_ = self.fc_t2(mi_y_)
        return y_, mi_y_

class Discriminator_DCGAN_MRES_MAT_32(nn.Module):
    """
        DeepConv Discriminator
    """
    def __init__(self, in_channel=1, num_classes=1):
        super(Discriminator_DCGAN_MRES_MAT_32, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.output_t1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(4)
        )
        self.output_t2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(4)
        )
        self.output_t3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(4)
        )
        self.output_t4 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(4)
        )
        self.fc_t1 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.fc_t2 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.fc_t3 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.fc_t4 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    def forward(self, x, y=None):
        x_d1 = downsample(x)
        x_d2 = downsample(x_d1)
        mi_x = batch_lw_tensor_local_moran(x.detach().cpu(),w_sparse_32)
        mi_x_d1 = batch_lw_tensor_local_moran(x_d1.detach().cpu(),w_sparse_16)
        mi_x_d2 = batch_lw_tensor_local_moran(x_d2.detach().cpu(),w_sparse_8)
        mi_x = mi_x.to(DEVICE)
        mi_x_d1 = mi_x_d1.to(DEVICE)
        mi_x_d2 = mi_x_d2.to(DEVICE)
        mi_x_d1 = nn.functional.interpolate(mi_x_d1,scale_factor=2,mode="nearest")
        mi_x_d2 = nn.functional.interpolate(mi_x_d2,scale_factor=4,mode="nearest")
        y_ = self.conv(x)
        mi_y_ = self.conv(mi_x)
        mi_y_d1 = self.conv(mi_x_d1)
        mi_y_d2 = self.conv(mi_x_d2)
        y_ = self.output_t1(y_)
        mi_y_ = self.output_t2(mi_y_)
        mi_y_d1 = self.output_t3(mi_y_d1)
        mi_y_d2 = self.output_t4(mi_y_d2)
        y_ = y_.view(y_.size(0), -1)
        mi_y_ = mi_y_.view(mi_y_.size(0), -1)
        mi_y_d1 = mi_y_d1.view(mi_y_d1.size(0), -1)
        mi_y_d2 = mi_y_d2.view(mi_y_d2.size(0), -1)
        y_ = self.fc_t1(y_)
        mi_y_ = self.fc_t2(mi_y_)
        mi_y_d1 = self.fc_t3(mi_y_d1)
        mi_y_d2 = self.fc_t4(mi_y_d2)
        return y_, mi_y_, mi_y_d1, mi_y_d2


class Generator_DCGAN_32(nn.Module):
    """
        Convolutional Generator
    """
    def __init__(self, input_size=100, out_channel=1):
        super(Generator_DCGAN_32, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 4*4*512),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, out_channel, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )      
    def forward(self, x, y=None):
        x = x.view(x.size(0), -1)
        y_ = self.fc(x)
        y_ = y_.view(y_.size(0), 512, 4, 4)
        y_ = self.conv(y_)
        return y_

class Discriminator_DCGAN_64(nn.Module):
    """
        Convolutional Discriminator
    """
    def __init__(self, in_channel=1, num_classes=1):
        super(Discriminator_DCGAN_64, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
        )        
        self.fc = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    def forward(self, x, y=None):
        y_ = self.conv(x)
        y_ = y_.view(y_.size(0), -1)
        y_ = self.fc(y_)
        return y_

class Discriminator_DCGAN_MAT_64(nn.Module):
    """
        Convolutional Discriminator
    """
    def __init__(self, in_channel=1, num_classes=1):
        super(Discriminator_DCGAN_MAT_64, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.output_t1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
        )        
        self.output_t2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
        )          
        self.fc_t1 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.fc_t2 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    def forward(self, x, y=None):
        mi_x = batch_lw_tensor_local_moran(x.detach().cpu(),w_sparse_32)
        mi_x = mi_x.to(DEVICE)
        y_ = self.conv(x)
        mi_y_ = self.conv(mi_x)
        y_ = self.output_t1(y_)
        mi_y_ = self.output_t2(mi_y_)
        y_ = y_.view(y_.size(0), -1)
        mi_y_ = mi_y_.view(mi_y_.size(0), -1)
        y_ = self.fc_t1(y_)
        mi_y_ = self.fc_t2(mi_y_)
        return y_, mi_y_

class Discriminator_DCGAN_MRES_MAT_64(nn.Module):
    """
        Convolutional Discriminator
    """
    def __init__(self, in_channel=1, num_classes=1):
        super(Discriminator_DCGAN_MRES_MAT_64, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.output_t1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
        )        
        self.output_t2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
        )        
        self.output_t3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
        )        
        self.output_t4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
        )        
        self.fc_t1 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.fc_t2 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.fc_t3 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.fc_t4 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    def forward(self, x, y=None):
        x_d1 = downsample(x)
        x_d2 = downsample(x_d1)
        mi_x = batch_lw_tensor_local_moran(x.detach().cpu(),w_sparse_64)
        mi_x_d1 = batch_lw_tensor_local_moran(x_d1.detach().cpu(),w_sparse_32)
        mi_x_d2 = batch_lw_tensor_local_moran(x_d2.detach().cpu(),w_sparse_16)
        mi_x = mi_x.to(DEVICE)
        mi_x_d1 = mi_x_d1.to(DEVICE)
        mi_x_d2 = mi_x_d2.to(DEVICE)
        mi_x_d1 = nn.functional.interpolate(mi_x_d1,scale_factor=2,mode="nearest")
        mi_x_d2 = nn.functional.interpolate(mi_x_d2,scale_factor=4,mode="nearest")
        y_ = self.conv(x)
        mi_y_ = self.conv(mi_x)
        mi_y_d1 = self.conv(mi_x_d1)
        mi_y_d2 = self.conv(mi_x_d2)
        y_ = self.output_t1(y_)
        mi_y_ = self.output_t2(mi_y_)
        mi_y_d1 = self.output_t3(mi_y_d1)
        mi_y_d2 = self.output_t4(mi_y_d2)
        y_ = y_.view(y_.size(0), -1)
        mi_y_ = mi_y_.view(mi_y_.size(0), -1)
        mi_y_d1 = mi_y_d1.view(mi_y_d1.size(0), -1)
        mi_y_d2 = mi_y_d2.view(mi_y_d2.size(0), -1)
        y_ = self.fc_t1(y_)
        mi_y_ = self.fc_t2(mi_y_)
        mi_y_d1 = self.fc_t3(mi_y_d1)
        mi_y_d2 = self.fc_t4(mi_y_d2)
        return y_, mi_y_, mi_y_d1, mi_y_d2


class Generator_DCGAN_64(nn.Module):
    """
        Convolutional Generator
    """
    def __init__(self, out_channel=1, input_size=100):
        super(Generator_DCGAN_64, self).__init__()
        assert IMAGE_DIM[0] % 2**4 == 0, 'Should be divided 16'
        self.init_dim = (IMAGE_DIM[0] // 2**4, IMAGE_DIM[1] // 2**4)
        self.fc = nn.Sequential(
            nn.Linear(input_size, self.init_dim[0]*self.init_dim[1]*512),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, out_channel, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
    def forward(self, x, y=None):
        x = x.view(x.size(0), -1)
        y_ = self.fc(x)
        y_ = y_.view(y_.size(0), 512, self.init_dim[0], self.init_dim[1])
        y_ = self.conv(y_)
        return y_

###
# EDGAN
###

class Discriminator_EDGAN_32(nn.Module):
    """
        Convolutional Discriminator
    """
    def __init__(self,nc=1,ndf1=32):
        super(Discriminator_EDGAN_32,self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(nc,ndf1,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ndf1),
          nn.LeakyReLU(0.2,inplace=True),
          nn.Conv2d(ndf1,ndf1*2,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ndf1*2),
          nn.LeakyReLU(0.2,inplace=True),
          nn.Conv2d(ndf1*2,ndf1*4,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ndf1*4),
          nn.LeakyReLU(0.2,inplace=True)
        )
        self.output_t1 = nn.Sequential(nn.Conv2d(ndf1*4,1,kernel_size=4,stride=1,padding=0),
          nn.Sigmoid()
        )
    def forward(self, x, y=None):
        y_ = self.conv(x)
        y_ = self.output_t1(y_)
        y_ = y_.view(y_.size(0), -1)
        return y_

class Discriminator_EDGAN_MAT_32(nn.Module):
    """
        Convolutional Discriminator
    """
    def __init__(self,nc=1,ndf1=32):
        super(Discriminator_EDGAN_MAT_32,self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(nc,ndf1,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ndf1),
          nn.LeakyReLU(0.2,inplace=True),
          nn.Conv2d(ndf1,ndf1*2,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ndf1*2),
          nn.LeakyReLU(0.2,inplace=True),
          nn.Conv2d(ndf1*2,ndf1*4,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ndf1*4),
          nn.LeakyReLU(0.2,inplace=True),
        )
        self.output_t1 = nn.Sequential(nn.Conv2d(ndf1*4,1,kernel_size=4,stride=1,padding=0),
          nn.Sigmoid()
        )
        self.output_t2 = nn.Sequential(nn.Conv2d(ndf1*4,1,kernel_size=4,stride=1,padding=0),
          nn.Sigmoid()
        )
    def forward(self, x, y=None):
        mi_x = batch_lw_tensor_local_moran(x.detach().cpu(),w_sparse_32)
        mi_x = mi_x.to(DEVICE)
        y_ = self.conv(x)
        mi_y_ = self.conv(mi_x)
        y_ = self.output_t1(y_)
        mi_y_ = self.output_t2(mi_y_)
        y_ = y_.view(y_.size(0), -1)
        mi_y_ = mi_y_.view(mi_y_.size(0), -1)
        return y_, mi_y_

class Discriminator_EDGAN_MRES_MAT_32(nn.Module):
    """
        Convolutional Discriminator
    """
    def __init__(self,nc=1,ndf1=32):
        super(Discriminator_EDGAN_MRES_MAT_32,self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(nc,ndf1,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ndf1),
          nn.LeakyReLU(0.2,inplace=True),
          nn.Conv2d(ndf1,ndf1*2,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ndf1*2),
          nn.LeakyReLU(0.2,inplace=True),
          nn.Conv2d(ndf1*2,ndf1*4,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ndf1*4),
          nn.LeakyReLU(0.2,inplace=True)
        )
        self.output_t1 = nn.Sequential(nn.Conv2d(ndf1*4,1,kernel_size=4,stride=1,padding=0),
          nn.Sigmoid()
        )
        self.output_t2 = nn.Sequential(nn.Conv2d(ndf1*4,1,kernel_size=4,stride=1,padding=0),
          nn.Sigmoid()
        )
        self.output_t3 = nn.Sequential(nn.Conv2d(ndf1*4,1,kernel_size=4,stride=1,padding=0),
          nn.Sigmoid()
        )
        self.output_t4 = nn.Sequential(nn.Conv2d(ndf1*4,1,kernel_size=4,stride=1,padding=0),
          nn.Sigmoid()
        )
    def forward(self, x, y=None):
        x_d1 = downsample(x)
        x_d2 = downsample(x_d1)
        mi_x = batch_lw_tensor_local_moran(x.detach().cpu(),w_sparse_32)
        mi_x_d1 = batch_lw_tensor_local_moran(x_d1.detach().cpu(),w_sparse_16)
        mi_x_d2 = batch_lw_tensor_local_moran(x_d2.detach().cpu(),w_sparse_8)
        mi_x = mi_x.to(DEVICE)
        mi_x_d1 = mi_x_d1.to(DEVICE)
        mi_x_d2 = mi_x_d2.to(DEVICE)
        mi_x_d1 = nn.functional.interpolate(mi_x_d1,scale_factor=2,mode="nearest")
        mi_x_d2 = nn.functional.interpolate(mi_x_d2,scale_factor=4,mode="nearest")
        y_ = self.conv(x)
        mi_y_ = self.conv(mi_x)
        mi_y_d1 = self.conv(mi_x_d1)
        mi_y_d2 = self.conv(mi_x_d2)
        y_ = self.output_t1(y_)
        mi_y_ = self.output_t2(mi_y_)
        mi_y_d1 = self.output_t3(mi_y_d1)
        mi_y_d2 = self.output_t4(mi_y_d2)
        y_ = y_.view(y_.size(0), -1)
        mi_y_ = mi_y_.view(mi_y_.size(0), -1)
        mi_y_d1 = mi_y_d1.view(mi_y_d1.size(0), -1)
        mi_y_d2 = mi_y_d2.view(mi_y_d2.size(0), -1)
        return y_, mi_y_, mi_y_d1, mi_y_d2

class Generator_EDGAN_32(nn.Module):
    """
        Encoder-Decoder Generator
    """
    def __init__(self, input_size=100, nc=1, ngf=N):
        super(Generator_EDGAN_32, self).__init__()
        assert IMAGE_DIM[0] % 2**4 == 0, 'Should be divided 16'
        self.init_dim = (IMAGE_DIM[0] // 2**4, IMAGE_DIM[1] // 2**4)
        self.fc = nn.Sequential(
            nn.Linear(input_size, N*N),
            nn.ReLU(),
        )
        self.encoder = nn.Sequential(nn.Conv2d(nc,ngf,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ngf),
          nn.LeakyReLU(0.2,inplace=True),
          nn.Conv2d(ngf,ngf*2,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ngf*2),
          nn.LeakyReLU(0.2,inplace=True),
          nn.Conv2d(ngf*2,ngf*4,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ngf*4),
          nn.LeakyReLU(0.2,inplace=True)
        )
        self.decoder = nn.Sequential(nn.ConvTranspose2d(ngf*4,ngf*2,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ngf*2),
          nn.ReLU(),
          nn.ConvTranspose2d(ngf*2,ngf,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ngf),
          nn.ReLU(),
          nn.ConvTranspose2d(ngf,nc,kernel_size=4,stride=2,padding=1),
          nn.Tanh()
        )   
    def forward(self, x, y=None):
        x = x.view(x.size(0), -1)
        y_ = self.fc(x)
        y_ = y_.view(y_.size(0), 1, N, N)
        y_ = self.encoder(y_)
        y_ = self.decoder(y_)
        return y_

class Discriminator_EDGAN_64(nn.Module):
    """
        Convolutional Discriminator
    """
    def __init__(self,nc=1,ndf=N):
        super(Discriminator_EDGAN_64,self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=4,stride=2,padding=1),
          nn.LeakyReLU(0.2,inplace=True),
          nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ndf*2),
          nn.LeakyReLU(0.2,inplace=True),
          nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ndf*4),
          nn.LeakyReLU(0.2,inplace=True),
          nn.Conv2d(ndf*4,ndf*8,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ndf*8),
          nn.LeakyReLU(0.2,inplace=True),
          nn.Conv2d(ndf*8,1,kernel_size=4,stride=1,padding=0),
          nn.Sigmoid()
        )
    def forward(self, x, y=None):
        y_ = self.conv(x)
        y_ = y_.view(y_.size(0), -1)
        return y_
        
class Discriminator_EDGAN_MAT_64(nn.Module):
    """
        Convolutional Discriminator 
    """
    def __init__(self,nc=1,ndf=64):
        super(Discriminator_EDGAN_MAT_64,self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=4,stride=2,padding=1),
          nn.LeakyReLU(0.2,inplace=True),
          nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ndf*2),
          nn.LeakyReLU(0.2,inplace=True),
          nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ndf*4),
          nn.LeakyReLU(0.2,inplace=True),
          nn.Conv2d(ndf*4,ndf*8,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ndf*8),
          nn.LeakyReLU(0.2,inplace=True),
        )
        self.output_t1 = nn.Sequential(nn.Conv2d(ndf*8,1,kernel_size=4,stride=1,padding=0),
          nn.Sigmoid()
        )   
        self.output_t2 = nn.Sequential(nn.Conv2d(ndf*8,1,kernel_size=4,stride=1,padding=0),
          nn.Sigmoid()
        ) 
    def forward(self, x, y=None):
        mi_x = batch_lw_tensor_local_moran(x.detach().cpu(),w_sparse_64)
        mi_x = mi_x.to(DEVICE)
        y_ = self.conv(x)
        mi_y_ = self.conv(mi_x)
        y_ = self.output_t1(y_)
        mi_y_ = self.output_t2(mi_y_)
        y_ = y_.view(y_.size(0), -1)
        mi_y_ = mi_y_.view(mi_y_.size(0), -1)
        return y_, mi_y_

class Discriminator_EDGAN_MRES_MAT_64(nn.Module):
    """
        Convolutional Discriminator 
    """
    def __init__(self,nc=1,ndf=64):
        super(Discriminator_EDGAN_MRES_MAT_64,self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=4,stride=2,padding=1),
          nn.LeakyReLU(0.2,inplace=True),
          nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ndf*2),
          nn.LeakyReLU(0.2,inplace=True),
          nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ndf*4),
          nn.LeakyReLU(0.2,inplace=True),
          nn.Conv2d(ndf*4,ndf*8,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ndf*8),
          nn.LeakyReLU(0.2,inplace=True)
        )
        self.output_t1 = nn.Sequential(nn.Conv2d(ndf*8,1,kernel_size=4,stride=1,padding=0),
          nn.Sigmoid()
        )   
        self.output_t2 = nn.Sequential(nn.Conv2d(ndf*8,1,kernel_size=4,stride=1,padding=0),
          nn.Sigmoid()
        )   
        self.output_t3 = nn.Sequential(nn.Conv2d(ndf*8,1,kernel_size=4,stride=1,padding=0),
          nn.Sigmoid()
        )   
        self.output_t4 = nn.Sequential(nn.Conv2d(ndf*8,1,kernel_size=4,stride=1,padding=0),
          nn.Sigmoid()
        )   
    def forward(self, x, y=None):
        x_d1 = downsample(x)
        x_d2 = downsample(x_d1)
        mi_x = batch_lw_tensor_local_moran(x.detach().cpu(),w_sparse_64)
        mi_x_d1 = batch_lw_tensor_local_moran(x_d1.detach().cpu(),w_sparse_32)
        mi_x_d2 = batch_lw_tensor_local_moran(x_d2.detach().cpu(),w_sparse_16)
        mi_x = mi_x.to(DEVICE)
        mi_x_d1 = mi_x_d1.to(DEVICE)
        mi_x_d2 = mi_x_d2.to(DEVICE)
        mi_x_d1 = nn.functional.interpolate(mi_x_d1,scale_factor=2,mode="nearest")
        mi_x_d2 = nn.functional.interpolate(mi_x_d2,scale_factor=4,mode="nearest")
        y_ = self.conv(x)
        mi_y_ = self.conv(mi_x)
        mi_y_d1 = self.conv(mi_x_d1)
        mi_y_d2 = self.conv(mi_x_d2)
        y_ = self.output_t1(y_)
        mi_y_ = self.output_t2(mi_y_)
        mi_y_d1 = self.output_t3(mi_y_d1)
        mi_y_d2 = self.output_t4(mi_y_d2)
        y_ = y_.view(y_.size(0), -1)
        mi_y_ = mi_y_.view(mi_y_.size(0), -1)
        mi_y_d1 = mi_y_d1.view(mi_y_d1.size(0), -1)
        mi_y_d2 = mi_y_d2.view(mi_y_d2.size(0), -1)
        return y_, mi_y_, mi_y_d1_, mi_y_d2_

class Generator_EDGAN_64(nn.Module):
    """
        Encoder-Decoder Generator
    """
    def __init__(self, input_size=100, nc=1, ngf=N):
        super(Generator_EDGAN_64, self).__init__()
        assert IMAGE_DIM[0] % 2**4 == 0, 'Should be divided 16'
        self.init_dim = (IMAGE_DIM[0] // 2**4, IMAGE_DIM[1] // 2**4)
        self.fc = nn.Sequential(
            nn.Linear(input_size, N*N),
            nn.ReLU(),
        )
        self.encoder = nn.Sequential(nn.Conv2d(nc,ngf,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ngf),
          nn.LeakyReLU(0.2,inplace=True),
          nn.Conv2d(ngf,ngf*2,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ngf*2),
          nn.LeakyReLU(0.2,inplace=True),
          nn.Conv2d(ngf*2,ngf*4,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ngf*4),
          nn.LeakyReLU(0.2,inplace=True)
        )
        self.decoder = nn.Sequential(nn.ConvTranspose2d(ngf*4,ngf*2,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ngf*2),
          nn.ReLU(),
          nn.ConvTranspose2d(ngf*2,ngf,kernel_size=4,stride=2,padding=1),
          nn.BatchNorm2d(ngf),
          nn.ReLU(),
          nn.ConvTranspose2d(ngf,nc,kernel_size=4,stride=2,padding=1),
          nn.Tanh()
        )    
    def forward(self, x, y=None):
        x = x.view(x.size(0), -1)
        y_ = self.fc(x)
        y_ = y_.view(y_.size(0), 1, N, N)
        y_ = self.encoder(y_)
        y_ = self.decoder(y_)
        return y_