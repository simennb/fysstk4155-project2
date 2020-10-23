import numpy as np


class LinRegSGD:
    def __init__(self, n_epochs, n_minibatch, regularization=None, eta0=0.1):
        self._n_epochs = n_epochs
        self._n_minibatch = n_minibatch
        self._regularization = regularization
        self._eta = eta0

        self._lmb = None
        self._t0 = None
        self._t1 = None
        self._seed = None

        self.beta = None

    def fit(self, X, y):
        """
        TODO: currently i partition it into N_mb minibatches, so each batch is always the same indexes
        TODO: maybe make sure that we are shuffling at the correct time? there is no overlap between batches at least.
        """
        N, p = X.shape
        N_mb = self._n_minibatch
        m = int(N/N_mb)  # number of elements in each minibatch

        beta = np.random.randn(p, 1)

        for epoch in range(self._n_epochs):
            for i in range(N_mb):
                i_rand = np.random.randint(N_mb)
                xi = X[i_rand*m:i_rand*m + m]
                yi = y[i_rand*m:i_rand*m + m]

                gradients = 2 * xi.T @ ((xi @ beta) - yi)
#                eta = self._learning_schedule(epoch * N_mb + i)
                eta = self._eta

                beta = beta - eta * gradients

        self.beta = beta

    def predict(self, X):
        ytilde = X @ self.beta
        return ytilde

    def set_lambda(self, lmb):
        self._lmb = lmb

    def set_step_length(self, t0, t1):
        self._t0 = t0
        self._t1 = t1

    def set_seed(self, seed):
        self._seed = seed

    def _learning_schedule(self, t):
        return self._t0 / (t + self._t1)


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