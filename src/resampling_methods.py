import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import KFold
import sys
import functions as fun


class Bootstrap:
    def __init__(self, X_train, X_test, y_train, y_test, reg_obj, stat):
        """
        :param X_init:
        :param y_init:
        :param X_test:
        :param y_test:
        :param reg_obj:
#        :param stat: function to compute some statistic
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.reg = reg_obj
        self.stat = stat

    def compute(self, N_bootstraps, N_samples=None, test=None):
        """
        :param N_bootstraps:
        :param N_samples:
        :return:
        """
        # For testing purposes since something seems wrong
        if test is not None:
            # TODO: fix differences in returns
            error_test, bias_test, variance_test = self.compute_test(N_bootstraps, N_samples)
            return error_test, bias_test, variance_test

        y_pred = np.zeros((self.y_test.shape[0], N_bootstraps))
        y_fit = np.zeros((self.y_train.shape[0], N_bootstraps))
        y_test = self.y_test.reshape(-1, 1)
        y_train = self.y_train.reshape(-1, 1)

        for i in range(N_bootstraps):
#            if i % 10 == 0:
#                print('Bootstrap number: %d' % i)
            X_new, y_new = self.resample(self.X_train, self.y_train)
#            X_new = fun.scale_X(X_new)
#            X_new, y_new = resample(self.X_train, self.y_train)#, n_samples=N_bs)
#            X_new = self.X_train
#            y_new = self.y_train

            self.reg.fit(X_new, y_new)
            y_pred[:, i] = self.reg.predict(self.X_test)
            y_fit[:, i] = self.reg.predict(self.X_train)

        error = np.mean(np.mean((y_test - y_pred) ** 2, axis=1, keepdims=True))
        bias = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True)) ** 2)
        variance = np.mean(np.var(y_pred, axis=1, keepdims=True))

        train_error = np.mean(np.mean((y_train - y_fit) ** 2, axis=1, keepdims=True))

        return error, bias, variance, train_error

    def compute_test(self, N_bootstraps, N_samples=None, test=None):
        """
        :param N_bootstraps:
        :param N_samples:
        :return:
        """
        y_pred = np.zeros((self.y_test.shape[0], N_bootstraps))
        y_test = self.y_test.reshape(-1, 1)
        statistic = np.zeros(N_bootstraps)
        #        bias = np.zeros(N_resamples)

        #        print(self.X_test[196, :])
        #        print('meow', self.X_test[0:5, 0:5])
        #        print('meow', self.X_test[190:200, -9:-5])

        #        self.X_train[196, :] = self.X_train[195, :]
        #        self.X_test[196, :] = self.X_test[195, :]
        #        y_test[196] = y_test[195]

        #        tot_unique = np.zeros(N_bootstraps)
        #        n_samp = np.zeros(N_bootstraps)
        #        self.X_test = fun.scale_X(self.X_test)  ## doees nothing + wa ?WW? wa aw+
        count_outliers = 0
        for i in range(N_bootstraps):
            if i % 10 == 0:
                print('Bootstrap number: %d' % i)
            X_new, y_new = self.resample(self.X_train, self.y_train)
            X_new = fun.scale_X(X_new)
            #            X_new, y_new = resample(self.X_train, self.y_train)#, n_samples=N_bs)
            #            tot_unique[i] = len(np.unique(y_new))
            #            n_samp[i] = len(X_new[:, -1])
            #            X_new = self.X_train
            #            y_new = self.y_train

            #            np.random.normal(0, 0.1, 100)

            self.reg.fit(X_new, y_new)
            y_pred[:, i] = self.reg.predict(self.X_test)

            #            print(np.max(y_pred))
            """
            The number of 'outliers', as in where predicted values values are very far from test data
            gets higher the higher polynomial degree, and causes the bias to shoot up for some reason

            Despite resampling, it appears to be the same indices that give the fucked up value
            In the n=1024, nbs=2, p=15, noise=0.05 case, index 196 has a y_pred of 19 and 25ish
            setting it to the value of 195 gives a max value of y_pred of 1.2ish
            So why the hell does one row get so incredibly messed up?????
            And why is it extremely wrong in both bootstraps despite having shuffled everything?

            Replacing X_test[196] with X_test[195] "removes" the outlier, while just changing y_test[196] does nothing 
            """

            # outliers = [196]
        #            print((y_pred[:, i] - y_test[:,0]).shape)
        #            sys.exit(1)
        # TODO: line below
        #            count_outliers += np.count_nonzero(np.abs((y_pred[:, i] - y_test[:, 0])) > 5)
        #            print(count_outliers)

        #            print('i=%d, j=%d, pred=%.5f, test=%.5f' % (i, j, y_pred[j, i], y_test[j]))
        #            print(y_pred[150:160, i], y_test[150:160])
        #            print(y_pred[-9, i], y_test[-9])

        #            print(np.max(y_new), np.max(self.y_train))
        #            statistic[i] = self.stat(y_pred[:, i], y_test)

        #        error[degree] = np.mean(np.mean((y_test - y_pred) ** 2, axis=1, keepdims=True))
        #        bias[degree] = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True)) ** 2)
        #        variance[degree] = np.mean(np.var(y_pred, axis=1, keepdims=True))

        #            print(len(np.unique(y_pred[:, i])))

        #        print('N_samples = %.2f , mean(unique) = %.2f  BS' % (np.mean(n_samp), np.mean(tot_unique)))
        print('Number of outliers = %d' % count_outliers)

        error = np.mean(np.mean((y_test - y_pred) ** 2, axis=1, keepdims=True))
        # error = np.mean(statistic)
        bias = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True)) ** 2)
        variance = np.mean(np.var(y_pred, axis=1, keepdims=True))

        a = np.mean(y_pred, axis=1, keepdims=True)
        b = np.argmax(a)
        print(y_pred[b, i], y_test[b])
        #        print(bias[i])
        #            bias[i] = np.mean((self.y_test - np.mean(y_pred)) ** 2)
        #        bias_ = np.mean(bias)

        return error, bias, variance

    def resample(self, X, y):
        sample_ind = np.random.randint(0, len(X), len(X))

        X_new = (X[sample_ind]).copy()
        y_new = (y[sample_ind]).copy()

        return X_new, y_new

'''
from numpy import *
from numpy.random import randint, randn
from time import time
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# Returns mean of bootstrap samples
def stat(data):
    return mean(data)

# Bootstrap algorithm
def bootstrap(data, statistic, R):
    t = zeros(R); n = len(data); inds = arange(n); t0 = time()
    # non-parametric bootstrap
    for i in range(R):
        t[i] = statistic(data[randint(0,n,n)])

    # analysis
    print("Runtime: %g sec" % (time()-t0)); print("Bootstrap Statistics :")
    print("original           bias      std. error")
    print("%8g %8g %14g %15g" % (statistic(data), std(data),mean(t),std(t)))
    return t


mu, sigma = 100, 15
datapoints = 10000
x = mu + sigma*random.randn(datapoints)
# bootstrap returns the data sample
t = bootstrap(x, stat, datapoints)
# the histogram of the bootstrapped  data
n, binsboot, patches = plt.hist(t, 50, normed=1, facecolor='red', alpha=0.75)

# add a 'best fit' line
y = mlab.normpdf( binsboot, mean(t), std(t))
lt = plt.plot(binsboot, y, 'r--', linewidth=1)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.axis([99.5, 100.6, 0, 3.0])
plt.grid(True)

plt.show()
'''


class CrossValidation:
    def __init__(self, X, y, reg_obj, stat):
        """
        :param X:
        :param y:
        :param reg_obj:
#        :param stat: function to compute some statistic
        """
        self.X = X
        self.y = y
        self.reg = reg_obj
        self.stat = stat

    def compute(self, K):
        error_train = np.zeros(K)
        error_test = np.zeros(K)

        # TODO: Don't shuffle
#        index = np.arange(len(self.y))
#        np.random.shuffle(index)
#        index = np.arange(len(self.X))
#        np.random.shuffle(index)
#        X = (self.X[index]).copy()
#        y = (self.y[index]).copy()

        X = self.X.copy()
        y = self.y.copy()

        for i in range(K):
            X_train, X_test, y_train, y_test = self.split(X, y, K, i)

            self.reg.fit(X_train, y_train)
            y_fit = self.reg.predict(X_train)
            y_pred = self.reg.predict(X_test)

            error_train[i] = np.mean((y_train - y_fit) ** 2)
            error_test[i] = np.mean((y_test - y_pred) ** 2)

        return np.mean(error_train), np.mean(error_test)

    def split(self, X, y, K, i):
        # TODO: Check behavior compared to SKL kfold.split
        # TODO: since not every number is neatly divisable with K

        N = len(X)
#        j = i*int(N/K)
#        l = j + int(N/K)
        j = int(i*N/K)  # TODO: check
        l = j + int(N/K)

#        print('Split #%d: N: %d, [%d : %d], %d' % (i, N, j, l, l-j))

        indices = np.arange(N)
        test_indices = indices[j:l]
        mask = np.ones(N, dtype=bool)
        mask[test_indices] = False
        train_indices = indices[mask]

        X_train = X[train_indices]
        X_test = X[test_indices]
#        print(X_train.shape, X_test.shape)
#        print(test_indices)
#        print('train ', X_train)
#        print('test ', X_test)

        y_train = y[train_indices]
        y_test = y[test_indices]

        return X_train, X_test, y_train, y_test

    def split_2(self, X, y, K):
        # TODO: Check behavior compared to SKL kfold.split
        # TODO: since not every number is neatly divisable with K
        N = len(X)
#        train_indices = np.zeros((N,), dtype=int)

        for i in range(K):

#        j = i*int(N/K)
#        l = j + int(N/K)
            j = int(i*N/K)  # TODO: check
            l = j + int(N/K)

#        print('Split #%d: N: %d, [%d : %d], %d' % (i, N, j, l, l-j))

            indices = np.arange(N)
            test_indices = indices[j:l]
            mask = np.ones(N, dtype=bool)
            mask[test_indices] = False
            train_indices = indices[mask]

        X_train = X[train_indices]
        X_test = X[test_indices]
#        print(X_train.shape, X_test.shape)
#        print(test_indices)
#        print('train ', X_train)
#        print('test ', X_test)

        y_train = y[train_indices]
        y_test = y[test_indices]

        return X_train, X_test, y_train, y_test



class CrossValidationSKL:
    def __init__(self, X, y, reg_obj):
        """
        :param X:
        :param y:
        :param reg_obj:
        """
        self.X = X
        self.y = y
        self.reg = reg_obj

    def compute(self, K):
        kfold = KFold(n_splits=K)
        #    kfold = KFold(n_splits=K, random_state=None, shuffle=True)
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
