import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy
import pykeops
from pykeops.torch import LazyTensor


#Normalize data values. Default is [0,1] range; if min_val = -1, range is [-1,1]
def normal(tensor,min_val=-1):
  t_min = torch.min(tensor)
  t_max = torch.max(tensor)
  if t_min == 0 and t_max == 0:
    return torch.tensor(tensor)
  if min_val == -1:
    tensor_norm = 2 * ((tensor - t_min) / (t_max - t_min)) - 1
  if min_val== 0:
    tensor_norm = ((tensor - t_min) / (t_max - t_min))
  return torch.tensor(tensor_norm)

#Light-weight Local Moran's I for tensor data, requiring a sparse weight matrix input. 
#This can be used when there is no need to re-compute the weight matrix at each step
def lw_tensor_local_moran(y,w_sparse,na_to_zero=True,norm=True,norm_min_val=-1):
  y = y.reshape(-1)
  n = len(y)
  n_1 = n - 1
  z = y - y.mean()
  sy = y.std()
  z /= sy
  den = (z * z).sum()
  zl = torch.tensor(w_sparse * z)
  mi = n_1 * z * zl / den
  if na_to_zero==True:
    mi[torch.isnan(mi)] = 0
  if norm==True:
    mi = normal(mi,min_val=norm_min_val)
  return torch.tensor(mi)

#Batch version of lw_tensor_local_moran
def batch_lw_tensor_local_moran(y_batch,w_sparse,na_to_zero=True,norm=True,norm_min_val=-1):
  batch_size = y_batch.shape[0]
  N = y_batch.shape[3]
  mi_y_batch = torch.zeros(y_batch.shape)
  for i in range(batch_size):
    y = y_batch[i,:,:,:].reshape(N,N)
    y = y.reshape(-1)
    n = len(y)
    n_1 = n - 1
    z = y - y.mean()
    sy = y.std()
    z /= sy
    den = (z * z).sum()
    zl = torch.tensor(w_sparse * z)
    mi = n_1 * z * zl / den
    if na_to_zero==True:
      mi[torch.isnan(mi)] = 0
    if norm==True:
      mi = normal(mi,min_val=norm_min_val)
    mi_y_batch[i,0,:,:] = mi.reshape(N,N)
  return mi_y_batch    

#Maximum Mean Discrepancy score: measures discrepancy between two distributions (here real images x and fake images y)
def mmd(x,y,B,alpha=1):
  ###
  # Input:
  # x = tensor of shape [B, 1, IMG_DIM, IMG_DIM] (e.g. real images)
  # y = tensor of shape [B, 1, IMG_DIM, IMG_DIM] (e.g. fake images)
  # B = batch size (or size of samples to be compared); B(x) has to be B(y)
  # alpha = kernel parameter
  #
  # Output:
  # mmd score
  ###
  x = x.view(x.size(0), x.size(2) * x.size(3))
  y = y.view(y.size(0), y.size(2) * y.size(3))
  xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())
  rx = (xx.diag().unsqueeze(0).expand_as(xx))
  ry = (yy.diag().unsqueeze(0).expand_as(yy))
  K = torch.exp(- alpha * (rx.t() + rx - 2*xx))
  L = torch.exp(- alpha * (ry.t() + ry - 2*yy))
  P = torch.exp(- alpha * (rx.t() + ry - 2*zz))
  beta = (1./(B*(B-1)))
  gamma = (2./(B*B)) 
  mmd = beta * (torch.sum(K)-torch.trace(K)+torch.sum(L)-torch.trace(L)) - gamma * torch.sum(P)
  return mmd

#Classifier two-sample test (C2ST) with a kNN Classifier: trains a classifier to distinguish between real and fake data, evaluates on test set
def c2st(x_train,y_train,x_test,y_test,K=1):
  ###
  # Input:
  # x_train = tensor of shape [B, 1, IMG_DIM, IMG_DIM], training images
  # y_train = tensor of shape [B, 1], training labels (here "fake" or "real")
  # x_test = tensor of shape [B, 1, IMG_DIM, IMG_DIM], test images
  # y_test = tensor of shape [B, 1], test labels
  # k = number of neighbors to use in kNN classification
  #
  # Output:
  # error = error of the kNN classifier
  ###
  use_cuda = torch.cuda.is_available()
  N = x_train.shape[3]
  train_size = x_train.shape[0]
  test_size = x_test.shape[0]
  x_train = x_train.reshape(train_size,N*N)
  x_test = x_test.reshape(test_size,N*N)
  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)
  X_i = LazyTensor(x_test[:, None, :])  # test set
  X_j = LazyTensor(x_train[None, :, :])  # train set
  D_ij = ((X_i - X_j) ** 2).sum(-1)  # Symbolic matrix of squared L2 distances
  ind_knn = D_ij.argKmin(K, dim=1)  
  lab_knn = y_train[ind_knn]  
  y_knn, _ = lab_knn.mode()   # Compute the most likely label
  error = (y_knn != y_test).float().mean().item()
  if error > 0.5:
    error = 0.5
  pykeops.clean_pykeops()
  return error