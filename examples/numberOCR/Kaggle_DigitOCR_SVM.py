import math
import random
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

import matplotlib.pyplot as plt


from scipy import misc
from scipy import ndimage

# load data files
train_data = pd.read_csv('./data/kaggle/train.csv', header=0) 
test_data = pd.read_csv('./data/kaggle/test.csv', header=0)

x = np.vstack([np.array(train_data.ix[:, 1:]).astype('float64'),
    np.array(test_data.ix[:,:]).astype('float64')])
y = np.array(train_data.ix[:, 0])

# re-scale the data
x = x/255
scaler = preprocessing.StandardScaler().fit(x)
x = scaler.transform(x)

# training set
xt = x[0:30000, :]
yt = y[0:30000]
# evaluation set
xe = x[30000:42000, :]
ye = y[30000:]
# testing set
x_test = x[42000:, :]





def choose_param(x, y):
    size_set = 5000
    #C_range = np.arange(5, 10, 0.2)
    #r_range = np.arange(0.20, 0.80, 0.02)
    C_range = [6.10, 6.15, 6.2, 6.25, 6.30]
    r_range = [0.47, 0.48. 0.49]

    param_grid = dict(coef0=r_range, C=C_range)
    cv = StratifiedKFold(y=y[0:size_set], n_folds=5)
    
    clf = svm.SVC(kernel='poly', degree=4, cache_size=1000)
    grid = GridSearchCV(clf, param_grid=param_grid, cv=cv, verbose=10)
    grid.fit(x[0:size_set,:], y[0:size_set])
    
    print("The best classifier is: ", grid.best_estimator_)
    score_dict = grid.grid_scores_
    
    # We extract just the scores
    scores = [x[1] for x in score_dict]
    scores = np.array(scores).reshape(len(C_range), len(r_range))
    
    # draw heatmap of accuracy as a function of gamma and C
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.spectral)
    plt.xlabel('r')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(r_range)), r_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    
    plt.show()
    pass


def twerk(x_old_set, y_old_set):
    m,n = x_old_set.shape
    x_new_set = np.zeros((7*m, n))
    y_new_set = np.zeros((7*m,))
    j = 0
    for i in range(m):
        # add the original one
        x_new_set[j] = x_old_set[i,:]
        y_new_set[j] = y_old_set[i]
        
        x = x_old_set[i,:].reshape(28,28)
        y = y_old_set[i]
        
        # move north 1 px
        j +=1
        x_new_set[j] = np.vstack([x[1:,:], x[27:,:]]).flatten()
        #x_new_set[j] = np.roll(x, 2*k[0], k[1]).flatten()
        y_new_set[j] = y
        # move south 1 px
        j +=1
        x_new_set[j] = np.vstack([x[0:1,:], x[0:27,:]]).flatten()
        y_new_set[j] = y
        # move left 1 px
        j +=1
        x_new_set[j] = np.hstack([x[:,1:], x[:,0:1]]).flatten()
        y_new_set[j] = y
        # move right 1 px
        j +=1
        x_new_set[j] = np.hstack([x[:,27:], x[:,0:27]]).flatten()
        y_new_set[j] = y
        # rotate 5 degrees
        j +=1
        x_new_set[j] = ndimage.rotate(x, 10, reshape=False).flatten()
        y_new_set[j] = y
        # rotate -5 degrees
        j +=1
        x_new_set[j] = ndimage.rotate(x, -10, reshape=False).flatten()
        y_new_set[j] = y
            
        j += 1
    return [x_new_set, y_new_set]

def check_eval():
    clf = svm.SVC(C=8.30, kernel='poly', degree=4, coef0=0.38, cache_size=200)
    clf.fit(xt, yt)
    x_support = clf.support_vectors_
    y_support = np.array(y[clf.support_])
    [x_new, y_new]= twerk(x_support, y_support)
    clf.fit(x_new, y_new)
    score = clf.score(xe, ye)
    print 'score', score
    
def prep_submit():
    clf = svm.SVC(C=8.30, kernel='poly', degree=4, coef0=0.38, cache_size=200)
    clf.fit(x[0:42000,:], y[0:42000])
    
    x_support = clf.support_vectors_
    y_support = np.array(y[clf.support_])
    [x_new, y_new]= twerk(x_support, y_support)
    clf.fit(x_new, y_new)
    
    a = clf.predict(x_test)
    b = np.arange(1,28001,1)
    predictions = np.vstack([b, a]).T
    print 'To submitt add header: ImageId,Label'
    np.savetxt('./data/kaggle/predictions_svm.csv', predictions, fmt='%i,%i')


#def visualize(x):
#    x = x.reshape(28,28)
#    plt.imshow(x)
#    plt.show()
#    pass
