import numpy as np
from numba import njit


# Attempting to follow MLPClassifier's naming convention
# To make it easier to test them against each other

################
# Cost functions
################

# From SKL's BaseMultilayerPerceptron._backprop
'''
# The calculation of delta[last] here works with following
# combinations of output activation and loss function:
# sigmoid and binary cross entropy, softmax and categorical cross
# entropy, and identity with squared loss
deltas[last] = activations[-1] - y
'''


#@njit
def cost_MSE(y, a):
    return np.mean((a - y) ** 2, axis=0, keepdims=True)


def d_cost_MSE(y, a):
#    print(y.shape, a.shape)
    #return (a-y)**2  # TODO: hmmm
    return (a - y) * 2  # TODO: look at above, pretty certain im not supposed to have *2 hmm
#    return np.mean((a - y) ** 2, axis=0, keepdims=True)


# check p. 195 ish av aurelion geron
@njit
def cost_LogLoss(y_data, y_model):
    pass


@njit
def d_cost_LogLoss(y_data, y_model):
    pass


# Cross-Entropy
@njit
def cost_CrossEntropy(y_data, y_model):
    pass


@njit
def d_cost_CrossEntropy(y_data, y_model):
    pass


######################
# Activation functions
######################

# Logistic sigmoid function
@njit
def sigmoid(z):
    return 1.0/(1 + np.exp(-z))


@njit
def d_sigmoid(z):
    # Could speed up by giving in a instead of z, but probably better to make NN code as general as possible
    a = sigmoid(z)
    return a*(1-a)


# ReLU
@njit
def relu(z):
    return z if z >= 0 else 0


@njit
def d_relu(z):
    return 1 if z >= 0 else 0


# Leaky ReLU
@njit
def leaky_relu(z):
    return z if z >= 0 else 0.01*z


@njit
def d_leaky_relu(z):
    return 1 if z >= 0 else 0.01


# Hyperbolic tangent
@njit
def tanh(z):
    return np.tanh(z)


@njit
def d_tanh(z):
    return 1 - tanh(z)**2


# Softmax
#@njit
def softmax(z):
    exp_term = np.exp(z)
    return exp_term / np.sum(exp_term, axis=1, keepdims=True)


@njit
def d_softmax(z):
    # since we use as output this will never be used
    # TODO: look closer at it
    return z


# No activation function
@njit
def identity(z):
    return z


@njit
def d_identity(z):
    return 1