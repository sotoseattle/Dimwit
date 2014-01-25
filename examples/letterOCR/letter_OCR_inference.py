import copy
import numpy as np
import sys
sys.path.append( '../../.' ) # logysoft module address

import Factor
import Rvar
import FactorOperations
import UndGraph
import BFS_Paths
import IndMarkovNet
import CliqueTree


# BASICS

alpha = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
k = len(alpha)
image_px = 128

def visualize(image):
    z = np.array(image)
    z = np.reshape(z, (16,8), order='F')
    plt.imshow(z)
    plt.show()
    pass

# THE SAMPLING DATA TO FEED THE BEAST
# there are 99 words inside allWords
# each allword[i] has 2 keys, gT and img
#   gt is a list of letters (as numbers)
#   img is a list of 128 cells arrays

file = open('data/allwords_imgs.csv')
stream_images = np.array(file.readline().split(',')).astype('int8')
stream_images = np.reshape(stream_images, (128,-1), order='F') # 128 x 691
file = open('data/allwords_gtruth.csv')
stream = np.array(file.readline().split(',')).astype('int8')

allWords = {}
w, counter = 0, 0
letters, images = [], []

for e in stream:
    if e==0:
        if len(letters)>0: #finish previous word
            allWords[w] = {}
            #print [alpha[x] for x in letters]
            allWords[w]['gT']  = letters
            allWords[w]['img'] = images
            w +=1
        # start new word
            letters = []
            images = []
    else:
        letters += [e-1]
        images += [stream_images[:, counter]]
        counter += 1

# Singleton Factor computations and its params 

def computeImageFactor(img, thetas):
    '''we use softmax regression h() method'''
    x = np.hstack([1., np.array(img)]) # 1. for the bias param
    H = np.append(x.dot(thetas), 0.)   # the 0 is to add the last letter
    H = H - np.amax(H)
    H = np.exp(H)
    suma = np.sum(H)
    H = np.divide(H, suma)
    return H

data = [float(line.strip()) for line in open('data/softmax_params_singleton.csv')]
theta = np.array(data[0:3200]).reshape(128, 25) # 128 x 25
bias = np.array(data[3200:]).reshape(1,-1)      # 1 x 25
theta = np.vstack([bias, theta])

def score_singletons():
    tot_chars, tot_words, chars_ok, words_ok = 0., 0., 0., 0.
    for i in range(len(allWords)):
        tot_words +=1.
        w = allWords[i]
        count = 0.
        num_letters = len(w['gT'])
        if num_letters==4:
            print i
        for e in range(num_letters):
            tot_chars +=1.
            img = w['img'][e]
            should_be = w['gT'][e]
            a = computeImageFactor(img, theta)
            what_is = np.argmax(a)
            if what_is==should_be:
                chars_ok +=1.
                count +=1.
        if count==num_letters:
            words_ok +=1.
    print 'words:', words_ok, (100.*words_ok/tot_words),'%'
    print 'letters:', chars_ok, (100.*chars_ok/tot_chars),'%'
    pass

def singletonFactor(var_char, img):
    f = Factor.Factor([var_char])
    f.values = computeImageFactor(img, theta)
    return f

# Pairwise and Triplet Factor

file = open('data/pairwiseModel.csv')
prob_pairs = np.array(file.readline().split(',')).astype('float64')
prob_pairs = prob_pairs + 1e-11
prob_pairs = np.reshape(prob_pairs, (26,26), order='F')

file = open('data/tripletModel.csv')
prob_triplets = np.ones((26,26,26))
for i in range(2000):
    [a,b,c, p] = file.readline().split(',')
    a,b,c = int(a), int(b), int(c)
    prob_triplets[a,b,c] = float(p)

def pairwiseFactor(var1, var2):
    f = Factor.Factor([var1, var2])
    f.values = prob_pairs
    return f

def tripletFactor(var1, var2, var3):
    f = Factor.Factor([var1, var2, var3])
    M = np.ones((26,26,26))
    f.values = prob_triplets
    return f

def similarity(img1, img2):
    meanSim = 0.283 # Avg sim score computed over held-out data.
    cosDist = np.dot(img1.T, img2) / (np.linalg.norm(img1) * np.linalg.norm(img2))
    diff = (cosDist - meanSim) ** 2;
    if (cosDist > meanSim):
        sim = 1 + 5*diff;
    else:
        sim = 1 / (1 + 5*diff);
    return sim

def image_simil_factor(var1, var2, sim):
    sol = np.ones(k) + np.diag(np.array([sim-1]*k))
    f = Factor.Factor([var1, var2])
    f.values = sol
    return f

######################## KAHUNA ########################

def MAP_Word(word):
    chars = len(word['gT'])
    vall = [None]*chars
    for i in range(chars):
        vall[i] = Rvar.Rvar(i, 26)
    f = []
    for i in range(chars):
        f.append(singletonFactor(vall[i], word['img'][i]))
    for i in range(chars-1):
        f.append(pairwiseFactor(vall[i], vall[i+1]))
    for i in range(chars-2):
        f.append(tripletFactor(vall[i], vall[i+1], vall[i+2]))

    ss = []
    for i in range(chars):
        for j in range(i+1, chars):
            ss.append([vall[i], vall[j], similarity(word['img'][i], word['img'][j])])
    
    ss = sorted(ss, key = lambda x: x[2])
    top1 = ss.pop()
    f.append(image_simil_factor(top1[0], top1[1], top1[2]))
    top2 = ss.pop()
    f.append(image_simil_factor(top2[0], top2[1], top2[2]))
    #alist.sort(key=lambda x: x.foo)
    #f1.append(image_simil_factor(vall[i], vall[j], word['img'][i], word['img'][j]))
    

    print 'number of factors', len(f)

    cc = CliqueTree.CliqueTree(f)
    #print cc
    cc.calibrate(isMax=True)
    
    # BEWARE I AM assuming that I get exact unambiguous marginals
    # which in the generality of problems does not have to happen
    # that is why the checking at bottom is important
    sol = []
    for vari in vall:
        for beta in cc.beta:
            if vari in beta.variables:
                fu = copy.copy(beta)
                for g in (set(beta.variables) - set([vari])):
                    fu = FactorOperations.max_marginalize(fu, g)
                maxi = np.max(fu.values)
                sol.append(list(fu.values).index(maxi))
                break
    return sol

#n = 0
#print [alpha[e] for e in MAP_Word(allWords[n])]
#print "\ntrue:", [alpha[e] for e in allWords[n]['gT']]

##############################

def score():
    tot_chars, tot_words, chars_ok, words_ok = 0., 0., 0., 0.
    for i in range(len(allWords)):
        
        tot_words +=1.
        w = allWords[i]
        count = 0.
        num_letters = len(w['gT'])
        map = MAP_Word(w)
        print 'word_', i, ':', ''.join([alpha[e] for e in map]), 'vs.', ''.join([alpha[e] for e in w['gT']])
        for e in range(num_letters):
            tot_chars +=1.
            if map[e]==w['gT'][e]:
                chars_ok +=1.
                count +=1.
        if count==num_letters:
            words_ok +=1.
    print 'words:', words_ok, (100.*words_ok/tot_words),'%'
    print 'letters:', chars_ok, (100.*chars_ok/tot_chars),'%'
    pass

score()

def inspect(word):
    chars = len(word['gT'])
    vall = [None]*chars
    for i in range(chars):
        vall[i] = Rvar.Rvar(i, 26)
    f = []
    for i in range(chars):
        f.append(singletonFactor(vall[i], word['img'][i]))
    for i in range(chars-1):
        f.append(pairwiseFactor(vall[i], vall[i+1]))
    for i in range(chars-2):
        f.append(tripletFactor(vall[i], vall[i+1], vall[i+2]))
    return CliqueTree.CliqueTree(f)
    
#cc = inspect(allWords[68])
#print cc

### CHECK CALIBRATION (IMPRESSIVE!)
# Check that the exact marginal over a var in adjacent nodes (beliefs) is the same
#for vari in v:
#    s = `vari.id` + ' : '
#    for beta in cc.beta:
#        if vari in beta.variables:
#            f = copy.copy(beta)
#            for g in (set(beta.variables) - set([vari])):
#                f = FactorOperations.max_marginalize(f, g)
#            maxi = np.max(f.values)
#            s += alpha[list(f.values).index(maxi)]
#    print s



#word = allWords[68]
#chars = len(word['gT'])
#vall = [None]*chars
#for i in range(chars):
#    vall[i] = Rvar.Rvar(i, 26)
#fu = image_simil_factor(vall[0], vall[1], word['img'][0], word['img'][1])
#print fu.variables
#print fu.values
#print np.diag(fu.values)


#print prob_pairs
#print np.diag(prob_pairs)

