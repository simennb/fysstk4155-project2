import numpy as np
from sklearn.model_selection import KFold
from lib import functions as fun


class Bootstrap:
    def __init__(self, X_train, X_test, y_train, y_test, reg_obj):
        """
        :param X_train: train design matrix
        :param y_train: train data
        :param X_test: test design matrix
        :param y_test: test data
        :param reg_obj: object that has the functions fit() and predict(), for doing regression
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.reg = reg_obj

    def compute(self, N_bootstraps):
        """
        Computes the number of bootstraps that is sent in.
        :param N_bootstraps: number of bootstraps to perform
        :return: error, bias, variance, train_error
        """
        y_pred = np.zeros((self.y_test.shape[0], N_bootstraps))
        y_fit = np.zeros((self.y_train.shape[0], N_bootstraps))
        y_test = self.y_test.reshape(-1, 1)

        train_error = 0
        for i in range(N_bootstraps):
            X_new, y_new = self.resample(self.X_train, self.y_train)

            self.reg.fit(X_new, y_new)
            y_pred[:, i] = self.reg.predict(self.X_test)
            y_fit[:, i] = self.reg.predict(X_new)
            train_error += np.mean((y_new - y_fit[:, i])**2) / N_bootstraps

        error = np.mean(np.mean((y_test - y_pred) ** 2, axis=1, keepdims=True))
        bias = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True)) ** 2)
        variance = np.mean(np.var(y_pred, axis=1, keepdims=True))

        return error, bias, variance, train_error

    def resample(self, X, y):
        sample_ind = np.random.randint(0, len(X), len(X))

        X_new = (X[sample_ind]).copy()
        y_new = (y[sample_ind]).copy()

        return X_new, y_new


class CrossValidation:
    def __init__(self, X, y, reg_obj, stat=None):
        """
        :param X: Design matrix
        :param y: data
        :param reg_obj: object that has the functions fit() and predict(), for doing regression
        :param stat: list, contains functions to compute results with
        """
        self.X = X
        self.y = y
        self.reg = reg_obj
        if stat is None:
            self.stat = [fun.mean_squared_error]
        else:
            self.stat = stat

    def compute(self, K):
        """
        Computes K cross-validations.
        """
        error_train = np.zeros((K, len(self.stat)))
        error_test = np.zeros(error_train.shape)

        index = np.arange(len(self.y))
        np.random.shuffle(index)

        X = (self.X[index]).copy()
        y = (self.y[index]).copy()

        for i in range(K):
            X_train, X_test, y_train, y_test = self.split(X, y, K, i)

#            print('K = ', i)
#            print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

            self.reg.fit(X_train, y_train)
            y_fit = self.reg.predict(X_train)
            y_pred = self.reg.predict(X_test)

            for j in range(len(self.stat)):
                error_train[i, j] = self.stat[j](y_train, y_fit)
                error_test[i, j] = self.stat[j](y_test, y_pred)

        return np.mean(error_train, axis=0), np.mean(error_test, axis=0)

    def split(self, X, y, K, i):
        """
        Splits the X and y array into train and test data set based on K and i.
        There may be some index differences compared to how KFold.split() works
        but at the very least, every test set will be of the same length.
        If we don't shuffle beforehand there might be some issues with the last split
        since there may be one or two values that are separated from the rest of the training set.
        Depends on how divisable by K the data set is, I guess.
        """

        N = len(X)
        j = int(i*N/K)
        l = j + int(N/K)

#        print('Split #%d: N: %d, [%d : %d], %d' % (i, N, j, l, l-j))

        indices = np.arange(N)
        test_indices = indices[j:l]
        mask = np.ones(N, dtype=bool)
        mask[test_indices] = False
        train_indices = indices[mask]

        X_train = X[train_indices]
        X_test = X[test_indices]

        y_train = y[train_indices]
        y_test = y[test_indices]

        return X_train, X_test, y_train, y_test


class CrossValidationSKL:
    def __init__(self, X, y, reg_obj):
        """
        Class that behaves the same way as CrossValidation, except that it uses SKL's KFold for CV.
        """
        self.X = X
        self.y = y
        self.reg = reg_obj

    def compute(self, K):
        kfold = KFold(n_splits=K, random_state=0, shuffle=True)
        error_train = 0
        error_test = 0
        for train_inds, test_inds in kfold.split(self.X):
            X_train = self.X[train_inds]
            y_train = self.y[train_inds]

            X_test = self.X[test_inds]
            y_test = self.y[test_inds]

            beta = self.reg.fit(X_train, y_train)
            y_fit = self.reg.predict(X_train)
            y_predict = self.reg.predict(X_test)

            error_train += fun.mean_squared_error(y_train, y_fit) / K
            error_test += fun.mean_squared_error(y_test, y_predict) / K

        return error_train, error_test
