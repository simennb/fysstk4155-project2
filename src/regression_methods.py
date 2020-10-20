import numpy as np
import functions as fun


class OrdinaryLeastSquares:
    def __init__(self, method=5):
        self.beta = None
        self.method = method  # for testing fit

    def fit(self, X, y):

        if self.method == 1:
            self.beta = fun.invert_SVD(X.T @ X) @ X.T @ y  # Method 1

        elif self.method == 2:
            self.beta = fun.SVDinv(X.T @ X) @ X.T @ y  # Method 2

        elif self.method == 3:
            self.beta = np.linalg.inv(X.T @ X) @ X.T @ y  # Method 3

        elif self.method == 4:
            self.beta = np.linalg.pinv(X.T @ X) @ X.T @ y  # Method 4
            # Very close to SKL accuracy wise

        elif self.method == 5:
            self.beta = np.linalg.pinv(X) @ y  # Method 5
            # For some reason yields better results than SKL and method 4

        return self.beta

    def predict(self, X):
        ytilde = X @ self.beta
        return ytilde

    def confidence_interval_beta(self, X, y, ytilde):
        N = X.shape[0]
        p = X.shape[1]

        sigma2 = 1/(N-p-1) * np.sum((y - ytilde)**2)
        var_beta = np.sqrt(np.diagonal(np.linalg.pinv(X.T @ X))) * sigma2

        conf = 2*np.sqrt(var_beta)  # 2 sigma is ~95.45% confidence intervals

        return conf


class RidgeRegression:
    def __init__(self):
        self.beta = None
        self.lmb = None

    def set_lambda(self, lmb):
        self.lmb = lmb

    def fit(self, X, y):
        I = np.identity(X.shape[1])
        self.beta = np.linalg.pinv(X.T @ X + self.lmb * I) @ X.T @ y

    def predict(self, X):
        ytilde = X @ self.beta
        return ytilde

    def confidence_interval_beta(self, X, y, ytilde):
        N = X.shape[0]
        p = X.shape[1]
        I = np.identity(X.shape[1])

        sigma2 = 1/(N-p-1) * np.sum((y - ytilde)**2)
        var_beta = np.sqrt(np.diagonal(np.linalg.pinv(X.T @ X + self.lmb*I) @ X.T @ X @ (np.linalg.pinv(X.T @ X + self.lmb*I)).T)) * sigma2

        conf = 2*np.sqrt(var_beta)

        return conf
