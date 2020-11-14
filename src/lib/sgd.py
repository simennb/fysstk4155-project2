import numpy as np
from numba import njit
from lib import functions as fun


def softmax(z):
    # More numerically stable softmax
    a = np.exp(z - np.max(z)) / np.sum(np.exp(z - np.max(z)))
    return a

# Attempting to speed up the process by using jit.
#@njit  # at some point, i did something that numba doesnt like, and had to remove jit
def gradient_descent_linreg(X, y, n_epochs, N_mb, m, theta, eta0, lmb=0.0, lr='constant', t0=1.0, t1=10):
    """
    An attempt to speed up the gradient descent using jit, as my first implementation is ~4 (or more depending on size)
    orders of magnitude slower than SGDRegressor.

    Now only ~0-1 orders of magnitude slower than SKL, which is neat.
    Forcing SKL to run all iterations (setting tol=None) brings the performance diff to roughly one order of magnitude.
    """
    for epoch in range(n_epochs):
        for i in range(N_mb):
            i_rand = np.random.randint(N_mb)
            xi = X[i_rand*m:i_rand*m + m]
            yi = y[i_rand*m:i_rand*m + m]

            gradients = 2 * xi.T @ ((xi @ theta) - yi) / m
            gradients += lmb*theta  # Ridge regularization

            if lr == 'constant':
                eta = eta0
            elif lr == 'optimal':
                eta = learning_schedule(epoch*N_mb + i, t0, t1)

            theta = theta - eta * gradients

        X, y = fun.shuffle_data(X, y)  # or put in GD-func?

    return theta


#@njit
def gradient_descent_logreg(X, y, n_epochs, N_mb, m, theta, eta0, lmb=0.0, lr='constant', t0=1.0, t1=10):
    """
    An attempt to speed up the gradient descent using jit, as my first implementation is ~4 (or more depending on size)
    orders of magnitude slower than SGDRegressor.

    Now only ~0-1 orders of magnitude slower than SKL, which is neat.
    Forcing SKL to run all iterations (setting tol=None) brings the performance diff to roughly one order of magnitude.
    """
    for epoch in range(n_epochs):
        for i in range(N_mb):
            i_rand = np.random.randint(N_mb)
            xi = X[i_rand*m:i_rand*m + m]
            yi = y[i_rand*m:i_rand*m + m]

            gradients = np.dot(xi.T, (softmax(xi @ theta) - yi))
            gradients += lmb*theta  # Ridge regularization

            if lr == 'constant':
                eta = eta0
            elif lr == 'optimal':
                eta = learning_schedule(epoch*N_mb + i, t0, t1)

            theta = theta - eta * gradients

        X, y = fun.shuffle_data(X, y)

    return theta


@njit
def learning_schedule(t, t0, t1):
    return t0 / (t + t1)

######################################################################


class LinRegSGD:
    def __init__(self, n_epochs, batch_size, eta0=0.1, learning_rate='constant'):
        """
        Performs linear regression SGD
        """
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._eta = eta0
        self._learning_rate = learning_rate

        self._lmb = 0.0
        self._t0 = 1.0
        self._t1 = 10.0

        self.theta = None

    def fit(self, X, y):

        N, p = X.shape
        m = self._batch_size
        N_mb = int(N/m)  # Number of mini-batches

        theta = np.random.randn(p)

        # Gradient descent
        theta = gradient_descent_linreg(X, y, self._n_epochs, N_mb, m, theta, self._eta, lmb=self._lmb,
                                        lr=self._learning_rate, t0=self._t0, t1=self._t1)

        self.theta = theta

    def predict(self, X):
        ytilde = X @ self.theta
        return ytilde

    def set_lambda(self, lmb):
        self._lmb = lmb

    def set_step_length(self, t0, t1):
        self._t0 = t0
        self._t1 = t1


class LogRegSGD:
    def __init__(self, n_epochs, batch_size, n_labels, eta0=0.1, learning_rate='constant'):
        """
        Performs logistic SGD
        """
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._n_labels = n_labels
        self._eta = eta0
        self._learning_rate = learning_rate

        self._lmb = 0.0
        self._t0 = 1.0
        self._t1 = 10.0

        self.theta = None

    def fit(self, X, y):
        N, p = X.shape
        m = self._batch_size
        N_mb = int(N / m)  # Number of mini-batches

        theta = np.random.randn(p, self._n_labels)

        # Gradient descent
        theta = gradient_descent_logreg(X, y, self._n_epochs, N_mb, m, theta, self._eta, lmb=self._lmb,
                                        lr=self._learning_rate, t0=self._t0, t1=self._t1)

        self.theta = theta

    def predict(self, X):
        ytilde = softmax(X @ self.theta)
        return ytilde

    def set_lambda(self, lmb):
        self._lmb = lmb

    def set_step_length(self, t0, t1):
        self._t0 = t0
        self._t1 = t1
