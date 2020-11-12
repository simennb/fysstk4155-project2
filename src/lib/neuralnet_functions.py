import numpy as np
from numba import njit


# Attempting to follow MLPClassifier's naming convention
# To make it easier to test them against each other

################
# Cost functions
################

def cost_MSE(y, a):
    return np.mean((a - y) ** 2, axis=0) / 2


# check p. 195 ish av aurelion geron
# Log loss is the same as cross-entropy, right?
# Cross entropy = log loss in binary cases OK

# Cross-Entropy
def cost_CrossEntropy(y, a):
#    print(y.shape, a.shape)
#    res = - np.sum(y * np.log(a), axis=1)
#    print(res)
#    input()
#    return - np.sum(y * np.log(a), axis=1) #/ a.shape[0]  # ????
    res = - np.sum(y * np.log(a), axis=1)
    return np.mean(res)
#    return - np.sum(y * np.log(a), axis=1) / a.shape[0]  # ????


######################
# Activation functions
######################

# Logistic sigmoid function
@njit
def sigmoid(z):
    return 1.0/(1 + np.exp(-z))


@njit
def d_sigmoid(z):
    a = sigmoid(z)
    return a*(1-a)


# ReLU
@njit
def relu(z):
    return np.where(z >= 0, z, 0)


@njit
def d_relu(z):
    return np.where(z >= 0, 1, 0)


# Leaky ReLU
@njit
def leaky_relu(z):
    return np.where(z >= 0, z, 0.01*z)



@njit
def d_leaky_relu(z):
    return np.where(z >= 0, 1, 0.01)


# Hyperbolic tangent
@njit
def tanh(z):
    return np.tanh(z)


@njit
def d_tanh(z):
    return 1 - tanh(z)**2


# Softmax
def softmax(z):
    exp_term = np.exp(z)
    return exp_term / np.sum(exp_term, axis=1, keepdims=True)


@njit
def d_softmax(z):
    # since we use as output this will never be used
    # TODO: look closer at it
    # TODO: lecture notes week 41, slide 7
    # np.kron()
    return 0


# No activation function
@njit
def identity(z):
    return z


@njit
def d_identity(z):
    return 1
