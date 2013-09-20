import math
import pandas as pd
import numpy as np
from PCA import *


# __UTILITIES__
def sketchOCR(arrajo, cols):
	'''represent an OCR image from a numpy array'''
	# very brutish, does not differentiate grays
	# only uses the number of cols to know where to cut
	# must be well formated (theta0 is 1.0, dont send it)
	sol = ''
	count = 0
	for i in np.nditer(arrajo):
		#if i!=0:
		if i>=128:	# this way is clearer for greyscale
			sol += 'X'
		else:
			sol += ' '
		count +=1
		if count == cols:
			count = 0
			sol += '\n'
	return sol

def imageit(arrajo, name, side):
	'''create an image from the array values'''
	chorizo = (arrajo*255).round().astype(np.uint8)
	chorizo = chorizo.reshape((side, side))
	img = Image.fromarray(chorizo, 'L')
	img.save(name)
	pass

#__TESTS__
def test_2D_to_1D():
	'''Ex7 from Machine Learning Stanford Course.
	   Data consists of only 50 examples with 2 attributes each (x,y).
	   The idea is reduce the 2D to a single dimension'''
	tr = {}
	tr['X'] = np.array([[3.3816,4.5279,2.6557,2.7652,2.8466,3.8907,3.4758,5.9113,3.9289,4.5618,\
		4.5741,4.3717,4.1917,5.2441,2.8358,5.6353,4.6863,2.8505,5.1102,5.1826,5.7073,3.5797,5.6394,\
		4.2635,2.5365,3.2238,4.9295,5.7930,2.8168,3.8888,3.3432,5.8797,3.1039,5.3315,3.3754,4.7767,\
		2.6757,5.5003,1.7971,4.3225,4.4210,3.1793,3.0335,4.6093,2.9638,3.9718,1.1802,1.9190,3.9552,\
		5.1180],[3.3891,5.8542,4.4120,3.7154,4.1755,6.4884,3.6328,6.6808,5.0984,5.6233,5.3977,5.4612,\
		4.9547,4.6615,3.7680,6.3121,5.6652,4.6265,7.3632,4.6465,6.6810,4.8028,6.1204,4.6894,3.8845,\
		4.9426,5.9550,5.1084,4.8190,5.1004,5.8930,5.5214,3.8571,4.6807,4.5654,6.2544,3.7310,5.6795,\
		3.2475,5.1111,6.0256,4.4369,3.9788,5.8798,3.3002,5.4077,2.8787,5.0711,4.5053,6.0851]], dtype='float64').T
	tr['m'], tr['n'] = tr['X'].shape
	# scaling / normalization
	tr['xnorm'] = preprocessing.scale(tr['X'])
	assert tr['X'].shape == (50,2)
	assert tr['xnorm'].shape == tr['X'].shape
	assert np.allclose(np.sum(tr['xnorm'].mean(axis=0)), 0.0, atol=0.000001)
	# pca computation
	U = pca(tr['xnorm'])
	assert np.allclose(U, np.array([[-0.70710678, -0.70710678],[-0.70710678, 0.70710678]]))
	# reduction
	Z = shrink(tr['xnorm'], U, 1)
	assert np.allclose(Z[0], 1.49, atol=0.01) # diff from original => math accuracy when rescaling
	# inflation
	x_rec = inflate(np.array([1.49629135]), U, 1)
	#print 'sol', x_rec, tr['xnorm'][0,:]
	assert np.allclose(x_rec, np.array([-1.05803776, -1.05803776])) # close enough

def test_shrink_faces():
	'''Ex7 from Machine Learning Stanford Course.
	   Data consists of 5000 images of 32x32 => (5000, 1024)
	   The idea is to reduce from 1024 to 100 pixels'''
	# load data
	data = pd.read_csv('./tests/data/trainfaces.csv', header=None)
	tr = {}
	tr['X'] = data.ix[:,:].astype('float64')
	tr['m'], tr['n'] = tr['X'].shape
	tr['y'] = np.zeros(tr['n'])
	assert tr['X'].shape == (5000,1024)
	# scaling / normalization
	tr['xnorm'] = preprocessing.scale(tr['X'], with_std=True) # Needed std norm? #
	assert tr['xnorm'].shape == (5000,1024)
	assert np.allclose(np.sum(tr['xnorm'].mean(axis=0)), 0.0, atol=0.000001)
	imageit(tr['xnorm'][56,:], 'FACE_Orig.png', 32) # we choose example 56

	# pca computation
	U = pca(tr['xnorm'])
	assert U.shape == (1024,1024)
	assert np.allclose(U.dot(U.T), np.eye(1024))
	# reduction
	Z = shrink(tr['xnorm'], U, 100)
	assert Z.shape == (5000, 100)
	# inflation
	x_rec = inflate(Z, U, 100)
	imageit(x_rec[56,:], 'FACE_Mod.png', 32)
	assert x_rec.shape == (5000, 1024)
	#assert 1==2  # to see printouts and png the test must fail!


def test_OCR_numbers():
	'''Kaggle Number OCR data. 42.000 images 28x28 => (42.000, 784)'''
	# load data
	data = pd.read_csv('./examples/numberOCR/train.csv', header=0)
	tr = {}
	tr['X'] = data.ix[:, 1::].astype('float64')
	tr['y'] = data.ix[:, 0]
	tr['m'], tr['n'] = tr['X'].shape
	assert tr['X'].shape == (42000,784)
	# scale
	tr['xnorm'] = preprocessing.scale(tr['X'], with_std=False) # Needed std norm? #
	assert tr['xnorm'].shape == (42000,784)
	assert np.allclose(np.sum(tr['xnorm'].mean(axis=0)), 0.0, atol=0.000001)
	
	print '__________________ORIGINAL_____________________'
	stream_Original = np.array(tr['X'])[56, ::]
	print sketchOCR(stream_Original, 28)
	print 'GROUND TRUTH:', tr['y'][56]


	print '___________________', tr['xnorm'][56,:].size
	imageit(tr['xnorm'][56,:], 'OCR_Orig.png', 28) # we choose example 56
	
	# pca computation
	U = pca(tr['xnorm'])
	assert U.shape == (784, 784)
	assert np.allclose(U.dot(U.T), np.eye(784))
	# reduction
	Z = shrink(tr['xnorm'], U, 100)
	assert Z.shape == (42000, 100)	# the K must have sqrt
	# inflation
	x_rec = inflate(Z, U, 100)
	assert x_rec.shape == (42000, 784)
	print '__________________PCA_____________________'
	#stream_Original = np.array(tr['X'])[56, ::]
	print sketchOCR(x_rec[56,:], 28)
	bbb = (x_rec[56,:]*255).round().astype(np.uint8)
	imageit(bbb, 'OCR_Mod.png', 28)
	# assert 1==2  # to see printouts and png the test must fail!



