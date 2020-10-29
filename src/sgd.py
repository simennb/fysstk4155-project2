import numpy as np
from numba import njit
import functions as fun


# Attempting to speed up the process by using jit.
#@fun.timeit
@njit#(parallel=True)  # very slow with parallel, compilation takes forever, run itself is "only" ~36 seconds
def gradient_descent_linreg(X, y, n_epochs, N_mb, m, theta, eta0, penalty=None, lmb=0.0, lr='constant', t0=1.0, t1=10):#, seed):
    """
    An attempt to speed up the gradient descent using jit, as my first implementation is ~4 (or more depending on size)
    orders of magnitude slower than SGDRegressor.

    Now only ~0-1 orders of magnitude slower than SKL, which is neat.
    Forcing SKL to run all iterations (setting tol=None) brings the performance diff to roughly one order of magnitude.
    """
    j = 1.0
    for epoch in range(n_epochs):
        # TODO: shuffle in each epoch? no wait, that shuffled train and test each epoch?????
        for i in range(N_mb):
            i_rand = np.random.randint(N_mb)
            xi = X[i_rand*m:i_rand*m + m]
            yi = y[i_rand*m:i_rand*m + m]

#            gradients = 2 * xi.T @ ((xi @ theta) - yi)
            gradients = 2 * xi.T @ ((xi @ theta) - yi)
            if penalty == 'l2':
                gradients += lmb*theta  # Ridge
            elif penalty == 'l1':
                gradients -= lmb*np.sign(theta)  # Lasso, not sure about this

            if lr == 'constant':
                eta = eta0
            elif lr == 'optimal':
                # TODO: figure out how to deal with name, since not identical to SKL
                eta = learning_schedule(epoch*N_mb + i, t0, t1)
            elif lr == 'invscaling':
                eta = eta0 / np.power(j, 0.25)  # should be identical to SKL, looked at code
            theta = theta - eta * gradients
            j += 1

        X, y = fun.shuffle_data(X, y)  # or put in GD-func?
        # wont do anything for batches of size 1, but check for larger if it matters

    return theta

@njit
def learning_schedule(t, t0, t1):
    return t0 / (t + t1)

class LinRegSGD:
    def __init__(self, n_epochs, batch_size, penalty=None, eta0=0.1, learning_rate='constant'):
        """
        learning rate: string, takes either values 'constant' or 'invscaling'. Latter one chosen to match the similar
            learning schedule that SKL has, making it easier to run both. ÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆ
        """
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._penalty = penalty
        self._eta = eta0
        self._learning_rate = learning_rate

        self._lmb = None
        self._t0 = 1.0#None
        self._t1 = 10.0#None
        self._seed = None

        self.theta = None

    def fit(self, X, y):
        """
        TODO: currently i partition it into N_mb minibatches, so each batch is always the same indexes
        TODO: maybe make sure that we are shuffling at the correct time? there is no overlap between batches at least.
        """
#        print('Fit', X.shape)
        N, p = X.shape
        m = self._batch_size
        N_mb = int(N/m)  # Number of mini-batches
#        N_mb = self._n_minibatch
#        m = int(N/N_mb)  # number of elements in each minibatch

        theta = np.random.randn(p)#, 1)  # beta here is (p, 1), while in OLS its (p,) WHYYYYYYYYYYYYYYYYYYYYY
        # TODO: FIND UOT WHY

        # Gradient descent
        theta = gradient_descent_linreg(X, y, self._n_epochs, N_mb, m, theta, self._eta, penalty=self._penalty,
                                        lmb=self._lmb, lr=self._learning_rate, t0=self._t0, t1=self._t1)

        '''
        for epoch in range(self._n_epochs):
            for i in range(N_mb):
                i_rand = np.random.randint(N_mb)
                xi = X[i_rand*m:i_rand*m + m]
                yi = y[i_rand*m:i_rand*m + m]

                gradients = 2 * xi.T @ ((xi @ beta) - yi)
#                eta = self._learning_schedule(epoch * N_mb + i)
                eta = self._eta

                beta = beta - eta * gradients
        '''
        self.theta = theta

    def predict(self, X):
        ytilde = X @ self.theta
        return ytilde

    def set_lambda(self, lmb):
        self._lmb = lmb

    def set_step_length(self, t0, t1):
        self._t0 = t0
        self._t1 = t1

    def set_seed(self, seed):
        self._seed = seed


# HOLD ON
# THIS IS FROM NUMBA/JIT WEBPAGE
'''
@numba.jit(nopython=True, parallel=True)
def logistic_regression(Y, X, w, iterations):
    for i in range(iterations):
        w -= np.dot(((1.0 /
              (1.0 + np.exp(-Y * np.dot(X, w)))
              - 1.0) * Y), X)
    return w
'''

class LogRegSGD(LinRegSGD):
    pass

if __name__ == '__main__':
    from math import exp, sqrt
    from random import random, seed
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import SGDRegressor

    m = 100
    x = 2*np.random.rand(m,1)
    y = 4+3*x+np.random.randn(m,1)

    X = np.c_[np.ones((m,1)), x]
    theta_linreg = np.linalg.inv(X.T @ X) @ (X.T @ y)
    print("Own inversion")
    print(theta_linreg)
    sgdreg = SGDRegressor(max_iter = 50, penalty=None, eta0=0.1)
    sgdreg.fit(x,y.ravel())
    print("sgdreg from scikit")
    print(sgdreg.intercept_, sgdreg.coef_)


    theta = np.random.randn(2, 1)
    eta = 0.1
    Niterations = 1000


    for iter in range(Niterations):
        gradients = 2.0/m*X.T @ ((X @ theta)-y)
        theta -= eta*gradients
    print("theta from own gd")
    print(theta)

    xnew = np.array([[0],[2]])
    Xnew = np.c_[np.ones((2,1)), xnew]
    ypredict = Xnew.dot(theta)
    ypredict2 = Xnew.dot(theta_linreg)


    n_epochs = 50
    t0, t1 = 5, 50
    def learning_schedule(t):
        return t0/(t+t1)

    theta = np.random.randn(2,1)

    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2 * xi.T @ ((xi @ theta)-yi)
            eta = learning_schedule(epoch*m+i)
            theta = theta - eta*gradients
    print("theta from own sdg")
    print(theta)

    plt.plot(xnew, ypredict, "r-")
    plt.plot(xnew, ypredict2, "b-")
    plt.plot(x, y ,'ro')
    plt.axis([0,2.0,0, 15.0])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'Random numbers ')
    plt.show()