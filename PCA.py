import math
import Image
import pandas as pd
import numpy as np
import scipy as sc
from scipy.optimize import fmin_bfgs
from sklearn import preprocessing


# Principal Component Analysis
# Reduce dimmensionality of an image in order to speed up
# the regression analysis

def pca(X):
	m, n = X.shape
	U = np.zeros(n)
	S = np.zeros(n)
	Sigma = np.cov(X.T)
	print 'Sigma shape should be nxn:', Sigma.shape
	print 'U will be m x ', np.min(Sigma.shape)
	U, S, V = np.linalg.svd(Sigma, full_matrices=False)
	return [U, S]

def shrink(example_norm, U, dimensions):
	Ureduce = U[:, 0:dimensions]
	Z = example_norm.dot(Ureduce)
	return Z

def inflate(shrunk_data, U, dimensions):
	Ureduce = U[:, 0:dimensions]
	X_rec = shrunk_data.dot(Ureduce.T)
	return X_rec

