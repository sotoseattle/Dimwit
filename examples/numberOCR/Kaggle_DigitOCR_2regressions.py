import math
import pandas as pd
import numpy as np

import sys
sys.path.append( '../../.' ) # regression modules address
import Logysoft as soft
import Logysterical as logit

class Kaggle_OCR_2Regress(object):
    def __init__(self):
        self.LABS = 10              # N. of possible values of Y (labels)
        # load data files
        training_data = pd.read_csv('./data/kaggle/train.csv', header=0)
        testing_data = pd.read_csv('./data/kaggle/test.csv', header=0)
        self.xt = np.array(training_data.ix[:, 1:]).astype('float64')
        self.yt = np.atleast_2d(training_data.ix[:, 0]).T

        # evaluation set
        self.xe = self.xt[30000:, :]
        self.ye = self.yt[30000:, :]
        
        # testing set
        self.x_test = np.array(testing_data.ix[:,:]).astype('float64')

    def optimize_logit_for(self, pair):
        # extract the right images from the set
        [a,b] = [int(pair[0]), int(pair[1])]
        
        data = np.array(pd.read_csv('./data/kaggle/train.csv', header=0)).astype('float64')
        data_t = data[0:, :]
        
        data_a = data_t[data_t[:,0]==a]
        data_b = data_t[data_t[:,0]==b]
        data_ab = np.vstack([data_a, data_b])
        xt = data_ab[:, 1:]
        yt = data_ab[:, 0].astype('int8')
        yt = np.atleast_2d(yt).T
        print 'xt shape', xt.shape
    
        # Perform logistic regression
        xt2 = np.column_stack([np.ones((xt.shape[0],1)), xt])
        yt2 = yt.flatten()
        count = 0
        for i in range(yt2.size):
            if yt2[i]==a:
                yt2[i] = 1
                count +=1
            else:
                yt2[i] = 0    
        
        ini_thetas = 0.005*np.random.rand(xt2.shape[1],1)
        L = 1e+5
        opt_thetas = logit.optimizeThetas(ini_thetas, xt2, yt2, L, visual=False)
        return opt_thetas


    def check_accuracy(self):
        logit_thetas = {}

        soft_thetas = np.array(pd.read_csv('./data/kaggle/optimized_thetas.csv', header=None))
        soft_thetas = soft_thetas.reshape(self.LABS, -1)
        
        h = soft.h(soft_thetas, self.xe)
        m = h.shape[0]
        misses = 0.00
        count = 0.0
        for i in range(m):
            true_label = self.ye[i,0]
            [ml_1, ml_2] = h[i,:].argsort()[-2:][::-1] # 1st and 2nd model choices
            p1,p2 = h[i,:][ml_1], h[i,:][ml_2]
            
            right_order = True
            if ml_1 > ml_2:
                right_order = False
                s = `ml_2`+`ml_1`
            else:
                s = `ml_1`+`ml_2`
            
            if p1<0.99 and p2>0.01:

                if s not in logit_thetas:
                    count +=1
                    logit_thetas[s] = self.optimize_logit_for(s)

                l_t = logit_thetas[s]
                logix = np.hstack([1, self.xe[i,:]])

                p = logit.h(l_t, logix)
                if (p>0.5):
                    prediction = (ml_1 if right_order else ml_2)
                else:
                    prediction = (ml_2 if right_order else ml_1)
            else:
                prediction = ml_1
            
            #print prediction, true_label
            if prediction!=true_label:
                misses +=1.0
        
        print 'misses', misses
        print 'logit thetas searched', count
        acc = 1 - misses/m
        print 'accuracy:', acc
        pass

    def test_model_submit(self):
        logit_thetas = {}
        
        soft_thetas = np.array(pd.read_csv('./data/kaggle/submit_optimized_thetas.csv', header=None))
        soft_thetas = soft_thetas.reshape(self.LABS, -1)

        m, n = self.x_test.shape
        h = soft.h(soft_thetas, self.x_test)
        predictions = np.zeros((m,2))
        for i in range(m):
            [ml_1, ml_2] = h[i,:].argsort()[-2:][::-1] # 1st and 2nd model choices
            p1,p2 = h[i,:][ml_1], h[i,:][ml_2]
            right_order = True
            if ml_1 > ml_2:
                right_order = False
                s = `ml_2`+`ml_1`
            else:
                s = `ml_1`+`ml_2`
            
            if p1<0.99 and p2>0.01:
                if s not in logit_thetas:
                    logit_thetas[s] = self.optimize_logit_for(s)

                l_t = logit_thetas[s]
                logix = np.hstack([1, self.x_test[i,:]])

                p = logit.h(l_t, logix)
                if (p>0.5):
                    predictions[i,:] = ([i+1, ml_1] if right_order else [i+1, ml_2])
                else:
                    predictions[i,:] = ([i+1, ml_2] if right_order else [i+1, ml_1])
            else:
                predictions[i,:]=[i+1, ml_1]

        print 'To submitt add header: ImageId,Label'
        print predictions[0:10,:]
        np.savetxt('./data/kaggle/predictions_2steps.csv', predictions, fmt='%i,%i')
        pass


k2t = Kaggle_OCR_2Regress()
#k2t.check_accuracy()
k2t.test_model_submit()
