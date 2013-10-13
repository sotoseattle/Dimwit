import math
import pandas as pd
import numpy as np
import scipy as sc
from scipy.optimize import fmin_l_bfgs_b
from sklearn import preprocessing
import matplotlib.pyplot as plt


import sys
sys.path.append( '../../.' ) # logysoft module address
import Logysoft as soft
import PCA as pca


class Kaggle_OCR_Softmax(object):
    def __init__(self):
        # initialize general parameters
        self.L = 7                  # lambda, regularization parameter
        self.LABS = 10              # N. of possible values of Y (labels)
        self.N = 784                # N. of features per image (28x28)

        # load data files
        training_data = pd.read_csv('./data/kaggle/train.csv', header=0)
        testing_data = pd.read_csv('./data/kaggle/test.csv', header=0)
        x = np.array(training_data.ix[:, 1:]).astype('float64')
        y = np.atleast_2d(training_data.ix[:, 0]).T

        # training set
        self.xt = x[0:30000, :]
        self.yt = y[0:30000, :]
        self.gt = soft.groundTruth(self.yt, self.LABS)
        # evaluation set
        self.xe = x[30000:, :]
        self.ye = y[30000:, :]
        self.ge = soft.groundTruth(self.ye, self.LABS)
        # testing set
        self.x_test = np.array(testing_data.ix[:,:]).astype('float64')

    def choose_lambda(self):
        '''train with different regularization parameters and choose
           the one that minimizes the cost in the evaluation set.'''        
        tinit = 0.005* np.random.rand(self.LABS, self.N)

        # initialize some working vars
        #rango = np.array([1e-3, 1e-2, 1e-1, 1, 10, 100])
        rango = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
        Jt, Je = np.array([]), np.array([])
        bestC, bestL = 1e+10, 0.0

        # cycle through lambdas and choose the one with lowest cost
        for chosen_lambda in rango:
            t = soft.optimizeThetas(tinit, self.xt, self.gt, \
                numLabels=self.LABS, l=chosen_lambda, visual=True)

            cost_t = soft.j(t, self.xt, self.gt, self.LABS, chosen_lambda)
            cost_e = soft.j(t, self.xe, self.ge, self.LABS, chosen_lambda)

            Jt = np.append(Jt, cost_t)
            Je = np.append(Je, cost_e)

            print 'chosen_lambda:', chosen_lambda
            if cost_e < bestC:
                bestC = cost_e
                bestL = chosen_lambda
                print '_______________new best is', bestL, 'with cost_e', cost_e
        print "\n\nthe best lambda is", bestL

        # plot
        #line1 = plt.plot(np.log10(rango), Jt)
        #line2 = plt.plot(np.log10(rango), Je)
        line1 = plt.plot(rango, Jt)
        line2 = plt.plot(rango, Je)
        plt.setp(line1, linewidth=2.0, label='training', color='b', solid_joinstyle='round')
        plt.setp(line2, linewidth=2.0, label='training', color='r', solid_joinstyle='round')
        plt.xlabel('Lambda')
        plt.ylabel('J')
        plt.show()
        pass

    def check_accuracy(self):
        '''computes thetas on training set, saves them, and checks
           accuracy on evaluation set'''
        tinit = 0.005* np.random.rand(self.LABS, self.N)
        thetas = soft.optimizeThetas(tinit, self.xt, self.gt, self.LABS, self.L)
        thetas = thetas.reshape(self.LABS, -1)
        np.savetxt('./data/kaggle/optimized_thetas.csv', thetas, delimiter=',')
        h = soft.h(thetas, self.xe)
        predictions = h.argmax(axis=1)
        zeros_are_right = np.subtract(self.ye.T, predictions)
        misses = 1.0 * np.count_nonzero(zeros_are_right)
        acc = 1 - misses/len(predictions)
        print 'accuracy:', acc
        pass

    def learning_curves(self):
        tinit = 0.005* np.random.rand(self.LABS, self.N)
        m, n = self.xt.shape
        sample = np.array([3, 6, 9, 12, 15, 18, 21, 24, 27, 30])*1000
        Jt, Je = np.array([]), np.array([])
        
        for m in sample:
            my_t = soft.optimizeThetas(tinit, self.xt[0:m,:], self.gt[0:m,:], \
                numLabels=self.LABS, l=self.L, visual=False)
            
            Jt = np.append(Jt, soft.j(my_t, self.xt[0:m,:], self.gt[0:m,:], self.LABS, self.L))
            Je = np.append(Je, soft.j(my_t, self.xe, self.ge, self.LABS, self.L))

        # plot (m, Jtr) and (m, Jcv)
        line1 = plt.plot(sample, Jt)
        line2 = plt.plot(sample, Je)
        
        plt.setp(line1, linewidth=2.0, label='training', color='b', solid_joinstyle='round')
        plt.setp(line2, linewidth=2.0, label='training', color='r', solid_joinstyle='round')
        plt.xlabel('Number of Examples')
        plt.ylabel('Cost / Error')
        plt.show()
        pass

    def test_model_submit(self):
        # compute thetas on whole training set
        tinit = 0.005* np.random.rand(self.LABS, self.N)
        x = np.vstack([self.xt, self.xe])
        y = np.vstack([self.yt, self.ye])
        g = np.vstack([self.gt, self.ge])
        # find thetas and save them
        thetas = soft.optimizeThetas(tinit, x, g, self.LABS, self.L)
        thetas = thetas.reshape(self.LABS, -1)
        np.savetxt('./data/kaggle/submit_optimized_thetas.csv', thetas, delimiter=',')
        # compute predictions
        m, n = self.x_test.shape
        h = soft.h(thetas, self.x_test)
        predictions = np.zeros((m,2))
        for i in range(m):
            a = h[i,:].argmax()
            predictions[i,:]=[i+1, a]
        print 'To submitt add header: ImageId,Label'
        print predictions[0:10,:]
        np.savetxt('./data/kaggle/predictions.csv', predictions, fmt='%i,%i')
        pass


kt = Kaggle_OCR_Softmax()
kt.choose_lambda()
#kt.check_accuracy()
#kt.learning_curves()
#kt.test_model_submit()
