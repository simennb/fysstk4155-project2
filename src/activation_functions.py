import numpy as np
from numba import njit


# Sigmoid function
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
