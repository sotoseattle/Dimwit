# DIMWIT

### Data Inference Modeling with pIThon

Work in progress

## LOGISTIC REGRESSION

### SOFTMAX MULTIPLE CLASSIFIER

The key is the format in which we input the training data and the thetas.

Give k possible labels for classification and n features (i.e. 128 pixels in input image):

Thetas should be a numpy array with k rows and n+1 columns. For each row the first element is the bias theta (for x = 1), followed by the corresponding n theta parameters. In this manner for each possible label we have all the parameters in a row (including the bias).

A given input x should be formatted as a numpy array of n+1 elements. The first element a constant 1 (for the biased parameter)