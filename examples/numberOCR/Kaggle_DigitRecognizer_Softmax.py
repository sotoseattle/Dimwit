import math
import pandas as pd
import numpy as np
import scipy as sc
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_l_bfgs_b

import sys
sys.path.append( '../../.' ) # logysoft module address
import Logysoft as soft


class KaggleTrain(object):
    def __init__(self):
        data = pd.read_csv('./data/kaggle/train.csv', header=0)
        # train
        self.xt = np.array(data.ix[0:29999,1:]).astype('float64')
        self.yt = np.atleast_2d(data.ix[0:29999,0]).T
        # validate
        self.xv = np.array(data.ix[30000:35999,1:]).astype('float64')
        self.yv = np.atleast_2d(data.ix[30000:35999,0]).T
        # test
        self.xe = np.array(data.ix[36000:,1:]).astype('float64')
        self.ye = np.atleast_2d(data.ix[36000:,0]).T

        print 'train', self.xt.shape
        print 'valis', self.xv.shape
        print 'test ', self.xe.shape

        # specify general parameters
        self.l = 1e-4              # lambda, regularization parameter
        self.m, self.n = self.xt.shape  # n = attributes/img, m = n. examples
        self.labs = 10             # Number of possible values of Y (labels)
        self.GT = soft.groundTruth(self.yt, self.labs)

    def save_opt_thetas(self):
        tinit = 0.005* np.ones((self.labs, self.n))
        thetas = soft.optimizeThetas(tinit, self.xt, self.GT, self.labs, self.l)
        np.savetxt('./data/kaggle/opt_th.csv', np.array(thetas), delimiter=',')
        pass

    def check_accuracy(self):
        thetas = np.array(pd.read_csv('./data/kaggle/opt_th.csv', header=None)).astype('float64')
        thetas = thetas.reshape(10, -1)
        h = soft.h(thetas, self.xe)
        predictions = h.argmax(axis=1) # +1 for the 0-->10 positioning
        for i in range(20):
            print predictions[i], self.ye[i]        
        zeros_are_right = np.subtract(self.ye.T, predictions)
        misses = 1.0 * np.count_nonzero(zeros_are_right)
        acc = 1 - misses/len(predictions)
        print 'acc', acc
        pass

kt = KaggleTrain()
#kt.save_opt_thetas()
kt.check_accuracy()




def testSoftModel_submit():
    X_test = np.array(pd.read_csv('./data/kaggle/test.csv', header=0)).astype('float64')
    m, n = X_test.shape

    Thetas = np.array(pd.read_csv('./data/kaggle/optimized_thetas.csv', header=None)).astype('float64')
    h = soft.h(Thetas, X_test)

    predictions = np.zeros((m,2))
    for i in range(m):
        a = h[i,:].argmax() + 1
        predictions[i,:]=[i+1, a]

    print 'To submitt add header: ImageId,Label'
    np.savetxt('./data/kaggle/predictions.csv', predictions, fmt='%i,%i')


# uncomment which part to run
#trainSoftModel(False)
#testSoftModel()

