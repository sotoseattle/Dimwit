import math
import pandas as pd
import numpy as np
import scipy as sc
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_l_bfgs_b

import sys
sys.path.append( '../../.' ) # logysoft module address
import Logysoft as soft



def trainSoftModel(test=False):
    # load training data
    X_train = np.array(pd.read_csv('./data/ufldl/mnist_images_train.csv', header=None)).astype('float64')
    Y_train = np.array(pd.read_csv('./data/ufldl/mnist_labels_train.csv', header=None))
    assert X_train.shape[0] == Y_train.shape[0]
    
    # specify general parameters
    l = 1e-4                    # lambda, regularization parameter
    m, n = X_train.shape        # n = attributes per image, m = examples count
    numLabels = 10              # Number of possible values of Y (labels)

    #Before anything else we need to fix ufldl data, label 10 means 0
    for i in range(m):
        if Y_train[i,0] == 10:
            Y_train[i,0] = 0

    GT = soft.groundTruth(Y_train, numLabels)
    
    # various tests to make sure everything works
    if test:
        # test h (pretty hard coded for 10 x 784)
        a = np.ones((numLabels, n))
        b = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[1]]).astype('float64')/10
        tinit = a*b
        H = soft.h(tinit, X_train)
        assert np.allclose(H[0,:], np.array([3.14329571e-38, 1.53183141e-33, 7.46511842e-29, 3.63799780e-24, 1.77291602e-19, 8.64000303e-15, 4.21055773e-10, 2.05194331e-05, 9.99979480e-01, 3.14329571e-38]), rtol=1e-10)
        assert np.allclose(H[59999,:], np.array([3.13717129e-29, 1.14671660e-25, 4.19154339e-22, 1.53211666e-18, 5.60027948e-15, 2.04704584e-11, 7.48247778e-08, 2.73503761e-04, 9.99726421e-01, 3.13717129e-29]), rtol=1e-10)
        
        # test j
        sol_j = soft.j(tinit, X_train, GT, numLabels, l)
        print sol_j
        assert np.allclose(sol_j, 45.615423224, rtol=1e-10)
    
        # test v
        sol_v = soft.v(tinit, X_train, GT, numLabels, l)
        assert np.allclose(sol_v[0:20], np.array([1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 7.90849673203e-06, 1e-05, 1e-05, 1e-05, 1e-05, 9.80392156863e-06, 9.01960784314e-06]), rtol=1e-10)
    
        # check gradient with mini random sample
        x_check = np.random.rand(10,10)
        y_check = np.random.randint(1,11, (10,1))
        g_check = soft.groundTruth(y_check, numLabels)
        t_check = np.random.rand(10,10).astype('float64')
        g_theo = soft.v(t_check, x_check, g_check, 10, l)
        g_hand = soft.grad_by_hand(t_check, x_check, g_check, 10, l)
        diff = np.linalg.norm(g_hand - g_theo) / np.linalg.norm(g_hand + g_theo)
        assert diff <= 1e-9

    # find opt thetas with gradient descent
    tinit = 0.005* np.ones((numLabels, n))
    thetas = soft.optimizeThetas(tinit, X_train, GT, numLabels, l)
    np.savetxt('./data/ufldl/optimized_thetas.csv', np.array(thetas), delimiter=',')
    pass

def testSoftModel():
    X_test = np.array(pd.read_csv('./data/ufldl/mnist_images_test.csv', header=None))#.astype('float64')
    Y_test = np.array(pd.read_csv('./data/ufldl/mnist_labels_test.csv', header=None))
    print X_test.shape, Y_test.shape #####################################
    assert X_test.shape[0] == Y_test.shape[0]

    #Before anything else we need to fix the label 0 to mean 10
    m, n = X_test.shape
    for i in range(m):
        if Y_test[i,0] == 10:
            Y_test[i,0] = 0

    #Thetas = np.array(pd.read_csv('./data/ufldl/optimized_thetas.csv', header=None)).astype('float64')
    Thetas = np.array(pd.read_csv('./data/ufldl/optimized_thetas.csv', header=None)).astype('float64')
    Thetas = Thetas.reshape(10, -1)

    h = soft.h(Thetas, X_test)
    predictions = h.argmax(axis=1)
    #for i in range(20):
    #    print predictions[i], Y_test[i]        
    zeros_are_right = np.subtract(Y_test.T,predictions)
    misses = 1.0 * np.count_nonzero(zeros_are_right)
    acc = 1 - misses/len(predictions)
    print 'acc', acc


# uncomment which part to run
#trainSoftModel(False)
testSoftModel()

