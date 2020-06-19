import numpy as np
import esda
import pysal
import scipy

#Define image dimensions (assuming square shape)
N = 32
#Create weight matrix
w = pysal.lib.weights.lat2W(N,N,rook=False)
#Get sparse representations of the weight matrices
w_sparse = w
w_sparse.fit_transform = "r"
w_sparse = w_sparse.sparse
#Save sparse matrix
scipy.sparse.save_npz('w_sparse.npz', w_sparse)