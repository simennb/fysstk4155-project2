import functions as fun
import regression_methods as reg
import resampling_methods as res
import sgd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
import sys
from sklearn.linear_model import SGDRegressor
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


run_mode = 'a'
data = 'franke'

fig_path = '../figures/'
data_path = '../datafiles/'
write_path = '../datafiles/'

p = 20  # degree of polynomial for the task
scale = [True, False]  # first index is whether to subtract mean, second is to scale by std

test_size = 0.2

# Regression method
reg_str = 'OLS'
#reg_str = 'Ridge'
#reg_str = 'Lasso'  # probably not needed
#reg_str = 'SGD'
reg_str = 'SGD_SKL'

# Creating data set for the Franke function tasks
seed = 4155
n_franke = 32  # 529 points
N = n_franke**2  # Total number of samples n*2
noise = 0.05  # noise level

# Bootstrap and CV variables
N_bootstraps = 1#int(N / 2)  # number of resamples (ex. N/2, N/4)
K = 5

# Stochastic gradient descent parameters
N_epochs = 500  # Number of epochs in SGD
N_minibatch = 10  # Number of mini-batches
eta0 = 0.1  # Start training rate
learning_rate = 'meow'  # constant

# Benchmark settings
benchmark = False  # setting to True will adjust all relevant settings for all task
if benchmark is True:
    p = 5
    scale = [True, False]
    reg_str = 'SGD'  # set to SGD maybe since thats the point of the task
    n_franke = 23
    N = 529
    noise = 0.05
    N_bootstraps = 264
    K = 5
    N_epochs = 100
    N_minibatch = 10
    eta0 = 0.1


# Printing some information for logging purposes
fun.print_parameters_franke(seed, N, noise, p, scale, test_size)


# Randomly generated meshgrid
np.random.seed(seed)
x = np.sort(np.random.uniform(0.0, 1.0, n_franke))
y = np.sort(np.random.uniform(0.0, 1.0, n_franke))

x_mesh, y_mesh = np.meshgrid(x, y)
z_mesh = fun.franke_function(x_mesh, y_mesh)

# Adding normally distributed noise with strength noise
z_mesh = z_mesh + noise * np.random.randn(n_franke, n_franke)

# Raveling
x_ravel = np.ravel(x_mesh)
y_ravel = np.ravel(y_mesh)
z_ravel = np.ravel(z_mesh)

# Creating polynomial design matrix
X = fun.generate_polynomial(x_ravel, y_ravel, p)


########################################################################################################################
#@fun.timeit
@ignore_warnings(category=ConvergenceWarning)
def run_regression(X, z, reg_string, polydegree, lambdas, N_bs, K, test_size, scale, max_iter=50000):
    """
    Runs the selected regression methods for the input design matrix, p's, lambdas, and using
    the resampling methods as specified.
    While there may be several ways I could have done this more optimally, this function exists
    because a rather late attempt at restructuring the code in order to reduce the amount of duplicate
    lines of code regarding regression, that had just escalated out of control, making it extremely
    difficult to debug and finding whatever was causing all the issues.
    :param X: (N, p) array containing input design matrix
    :param z: (N, 1) array containing data points
    :param reg_string: string containing the name of the regression method to be used
    :param polydegree: list/range of the different p-values to be used
    :param lambdas: array of all the lambda values to be used
    :param N_bs: int, number of Bootstraps
    :param K: int, number of folds in the Cross-Validation
    :param test_size: float, size of the test partition [0.0, 1.0]
    :param scale: list determining if the scaling is only by the mean, the std or both [bool(mean), bool(std)]
    :param max_iter: maximum number of iterations for Lasso
    :return: a lot of arrays with the various results and different ways of representing the data
    """
    nlambdas = len(lambdas)  # number of lambdas
    p = polydegree[-1]  # the maximum p-value
    method = 4  # OLS method

    # Splitting into train and test, scaling the data
    X_train, X_test, z_train, z_test = fun.split_data(X, z, test_size=test_size)
    X_train_scaled = fun.scale_X(X_train, scale)
    X_test_scaled = fun.scale_X(X_test, scale)
    X_scaled = fun.scale_X(X, scale)

    # Bootstrap arrays
    bs_error_train = np.zeros((p, nlambdas))
    bs_error_test = np.zeros((p, nlambdas))
    bs_bias = np.zeros((p, nlambdas))
    bs_var = np.zeros((p, nlambdas))

    bs_error_train_opt = np.zeros((p, 2))
    bs_error_test_opt = np.zeros((p, 2))
    bs_bias_opt = np.zeros((p, 2))  # First index is min(MSE) lmb for each p, second at lmb that yields total lowest MSE
    bs_var_opt = np.zeros((p, 2))
    bs_lmb_opt = np.zeros(p)

    # Cross-validation arrays
    cv_error_train = np.zeros((p, nlambdas))
    cv_error_test = np.zeros((p, nlambdas))
    cv_error_train_opt = np.zeros((p, 2))
    cv_error_test_opt = np.zeros((p, 2))
    cv_lmb_opt = np.zeros(p)

    # Setting up regression object to be used for regression (Lasso is dealt with later)
    reg_obj = reg.OrdinaryLeastSquares(method)  # default
    if reg_string == 'SKL':
        reg_obj = skl.LinearRegression()  # Testing with scikit-learn OLS
    elif reg_string == 'Ridge':
        reg_obj = reg.RidgeRegression()
    elif reg_string == 'SGD':
        reg_obj = sgd.LinRegSGD(N_epochs, 100, eta0=0.1, learning_rate=learning_rate)

    # Looping over all polynomial degrees in the analysis
    for degree in polydegree:
        n_poly = fun.polynom_N_terms(degree)  # number of terms in the design matrix for the given degree
        print('p = %2d, np = %3d' % (degree, n_poly))

        # Setting up correct design matrices for the current degree
        X_train_bs = np.zeros((len(X_train_scaled), n_poly))
        X_test_bs = np.zeros((len(X_test_scaled), n_poly))
        X_cv = np.zeros((len(X_scaled), n_poly))

        # Filling the elements up to term n_poly
        X_train_bs[:, :] = X_train_scaled[:, 0:n_poly]
        X_test_bs[:, :] = X_test_scaled[:, 0:n_poly]
        X_cv[:, :] = X_scaled[:, 0:n_poly]

        # Looping over all the lambda values
        for i in range(nlambdas):
            lmb = lambdas[i]  # current lambda value

            # Printing out in order to gauge where we are
            if i % 10 == 0:
                print('i = %d, lmb= %.3e' % (i, lmb))

            # Updating the current lambda value for Ridge and Lasso
            if reg_string == 'Ridge':
                reg_obj.set_lambda(lmb)
            elif reg_string == 'Lasso':
                reg_obj = skl.Lasso(alpha=lmb, max_iter=max_iter, precompute=True, warm_start=True)
            elif reg_string == 'SGD_SKL':
#                reg_obj = SGDRegressor(max_iter=N_epochs, penalty=None, eta0=0.1)  # , learning_rate='constant')
                reg_obj = SGDRegressor(max_iter=N_epochs, penalty=None, eta0=0.1, learning_rate='constant')


            # Bootstrap
            BS = res.Bootstrap(X_train_bs, X_test_bs, z_train, z_test, reg_obj)
            error_, bias_, var_, trainE_ = BS.compute(N_bs)  # performing the Bootstrap
            bs_error_test[degree-1, i] = error_
            bs_bias[degree-1, i] = bias_
            bs_var[degree-1, i] = var_
            bs_error_train[degree-1, i] = trainE_

            # Cross validation
            CV = res.CrossValidation(X_cv, z, reg_obj)
            trainE, testE = CV.compute(K)  # performing the Cross-Validation
            cv_error_train[degree-1, i] = trainE
            cv_error_test[degree-1, i] = testE

        # Locating minimum MSE for each polynomial degree
        # Bootstrap
        index_bs = np.argmin(bs_error_test[degree - 1, :])
        bs_lmb_opt[degree - 1] = lambdas[index_bs]
        bs_error_train_opt[:, 0] = bs_error_train[:, index_bs]
        bs_error_test_opt[:, 0] = bs_error_test[:, index_bs]
        bs_bias_opt[:, 0] = bs_bias[:, index_bs]
        bs_var_opt[:, 0] = bs_var[:, index_bs]

        # Cross-validation
        index_cv = np.argmin(cv_error_test[degree - 1, :])
        cv_lmb_opt[degree - 1] = lambdas[index_cv]
        cv_error_train_opt[:, 0] = cv_error_train[:, index_cv]
        cv_error_test_opt[:, 0] = cv_error_test[:, index_cv]

    # Locate minimum MSE  to see how it depends on lambda
    bs_min = np.unravel_index(np.argmin(bs_error_test), bs_error_test.shape)
    cv_min = np.unravel_index(np.argmin(cv_error_test), cv_error_test.shape)
    bs_best = [polydegree[bs_min[0]], lambdas[bs_min[1]]]
    cv_best = [polydegree[cv_min[0]], lambdas[cv_min[1]]]

    # Bootstrap
    bs_error_train_opt[:, 1] = bs_error_train[:, bs_min[1]]
    bs_error_test_opt[:, 1] = bs_error_test[:, bs_min[1]]
    bs_bias_opt[:, 1] = bs_bias[:, bs_min[1]]
    bs_var_opt[:, 1] = bs_var[:, bs_min[1]]

    # Cross-validation
    cv_error_train_opt[:, 1] = cv_error_train[:, cv_min[1]]
    cv_error_test_opt[:, 1] = cv_error_test[:, cv_min[1]]

    # This return is extremely large, sadly, and should have been improved upon
    # this was just the fastest way of doing it when I had to restructure the code
    # so better planning in the future would be a better solution
    return (bs_error_train, bs_error_test, bs_bias, bs_var,
            bs_error_train_opt, bs_error_test_opt, bs_bias_opt, bs_var_opt, bs_lmb_opt,
            cv_error_train, cv_error_test, cv_error_train_opt, cv_error_test_opt, cv_lmb_opt,
            bs_min, bs_best, cv_min, cv_best)


    # Printing MSE and R2 score
#    fun.print_MSE_R2(z_test, z_pred, 'test', 'OLS')
#    fun.print_MSE_R2(z_train, z_fit, 'train', 'OLS')


########################################################################################################################
# But now makes every relevant Franke function plot for OLS

# Setting up to make sure things work
nlambdas = 1
lambdas = np.ones(nlambdas)

# Parameters for saving to file
save = 'N%d_pmax%d_nlamb%d_noise%.2f_seed%d' % (N, p, nlambdas, noise, seed)
save_bs = '%s_%s_%s_Nbs%d' % (save, reg_str, 'boot', N_bootstraps)
save_cv = '%s_%s_%s_k%d' % (save, reg_str, 'cv', K)

# Performing the regression
polydegree = np.arange(1, p + 1)
variables = run_regression(X, z_ravel, reg_str, polydegree, lambdas, N_bootstraps, K, test_size, scale)
# Unpacking variables
bs_error_train, bs_error_test = variables[0:2]
bs_bias, bs_var = variables[2:4]
bs_error_train_opt, bs_error_test_opt = variables[4:6]
bs_bias_opt, bs_var_opt, bs_lmb_opt = variables[6:9]
cv_error_train, cv_error_test = variables[9:11]
cv_error_train_opt, cv_error_test_opt, cv_lmb_opt = variables[11:14]
bs_min, bs_best, cv_min, cv_best = variables[14:18]

# Bootstrap plots
xlim = [1, 20]
ylim = [0.0, 0.02]
fun.plot_MSE_train_test(polydegree, bs_error_train_opt[:, 0], bs_error_test_opt[:, 0],
                        '%s, $N$=%d, $N_{bs}$=%d, noise=%.2f' % (reg_str, N, N_bootstraps, noise),
                        'train_test_%s' % save_bs, fig_path, run_mode,
                        resample='Bootstrap', xlim=xlim, ylim=ylim)

fun.plot_bias_variance(polydegree, bs_error_test_opt[:, 0], bs_bias_opt[:, 0], bs_var_opt[:, 0],
                       'Bootstrap, %s, $N$=%d, $N_{bs}$=%d, noise=%.2f' % (reg_str, N, N_bootstraps, noise),
                       '%s' % save_bs, fig_path, run_mode, xlim=xlim, ylim=ylim)

# Cross-validation plot
fun.plot_MSE_train_test(polydegree, cv_error_train_opt[:, 0], cv_error_test_opt[:, 0],
                        '%s, $N$=%d, $K$=%d, noise=%.2f' % (reg_str, N, K, noise),
                        'train_test_%s' % save_cv, fig_path, run_mode,
                        resample='CV', xlim=xlim, ylim=ylim)

# Write bootstrap to file
fun.save_to_file([bs_error_test_opt[:, 0], bs_bias_opt[:, 0], bs_var_opt[:, 0]],
                 ['bs_error_test', 'bs_bias', 'bs_var'],
                 write_path+'franke/bias_var_task_%s_%s.txt' % (run_mode, save_bs), benchmark)

# Write CV to file
fun.save_to_file([cv_error_test_opt[:, 0], cv_error_train[:, 0]], ['cv_error_test', 'cv_error_train'],
                 write_path+'franke/train_test_task_%s_%s.txt' % (run_mode, save_cv), benchmark)

plt.show()