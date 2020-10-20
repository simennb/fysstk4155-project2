import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.preprocessing import StandardScaler
from imageio import imread
import time
from numba import jit


###########################################################
def timeit(method):
    def timed(*args, **kw):
        t_start = time.time()
        result = method(*args, **kw)
        t_end = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((t_end - t_start))
        else:
            print('%r  %.4f s' % (method.__name__, t_end - t_start))
        return result
    return timed

###########################################################


def franke_function(x, y):
    """
    :param x:
    :param y:
    :return:
    """
    term1 = 0.75*np.exp(-0.25*(9*x-2)**2 - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-0.25*(9*x-7)**2 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def mean_squared_error(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


def calculate_R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)


def generate_polynomial(x, y, p):
    """
    :param x:
    :param y:
    :param p:
    :return:
    """
    l = polynom_N_terms(p)  # Number of terms in combined polynomial
    X = np.ones((len(x), l))

    j = 0  # Verified that yields correct results
    for i in range(1, p + 1):
        j = j + i - 1
        for k in range(i + 1):
            X[:, i + j + k] = x ** (i - k) * y ** k

    return X


def polynom_N_terms(p):
    """
    Returns the amount of terms the polynomial of degree p given by generate_polynomial
    :param p: polynomial degree
    :return:
    """
    return np.sum(np.arange(p+2))


def split_data(X, y, test_size=0.25):
    """
    :param X:
    :param y:
    :param test_size:
    :return:
    """
    N = len(y)
    i_split = int((1-test_size)*N)

#    X_train = np.zeros((i_split, X.shape[1]))
#    X_test = np.zeros((N-i_split, X_shape[1]))

    index = np.arange(N)
    np.random.shuffle(index)

    X = (X[index]).copy()
    y = (y[index]).copy()

    X_train = (X[0:i_split]).copy()
    X_test = (X[i_split:]).copy()

    y_train = (y[0:i_split]).copy()
    y_test = (y[i_split:]).copy()

    return X_train, X_test, y_train, y_test


def scale_X(X, scale=None):
    """
    Function for scaling X by subtracting the mean.
    Alternative to the skl StandardScaler to make sure intercept row is not set to 0
    :param X:
    :param scale:
    :return:
    """
    # TODO: check if scaling for test should be done with mean computed from train
#    print(X[0:5, 3])
    if scale is None:
        scale = [True, False]

    X_new = X.copy()
    std = np.std(X_new, axis=0, keepdims=True)
    if scale[0]:  # MEAN
        X_new[:, 1:] -= np.mean(X_new[:, 1:], axis=0, keepdims=True)
    if scale[1]:  # STD
        X_new[:, 1:] /= std[:, 1:]

    return X_new


def invert_SVD(X):
    """
    Computing the pseudo-inverse of X: X^-1 = V s^-1 UT
    :param X: input matrix
    :return: pseudo inverse of input matrix X
    """
    U, s, VT = np.linalg.svd(X)
    inv_sigma = np.zeros(len(s))
    inv_sigma[s != 0] = 1/s[s != 0]  # setting every non-zero element to 1/s[i]
    V = VT.T

    return V @ np.diag(inv_sigma) @ U.T


# TODO: remove, from mortens code
def SVDinv(A):
    ''' Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
    SVD is numerically more stable than the inversion algorithms provided by
    numpy and scipy.linalg at the cost of being slower.
    '''
    U, s, VT = np.linalg.svd(A)

    D = np.zeros((len(U),len(VT)))
    for i in range(0,len(VT)):
        D[i,i]=s[i]
    UT = np.transpose(U); V = np.transpose(VT); invD = np.linalg.inv(D)
    return np.matmul(V,np.matmul(invD,UT))


def read_terrain(filename, N, loc_start, norm=True):
    # Load the terrain
    terrain = imread(filename)

    # Normalize terrain data by dividing by maximum value
    if norm:
        terrain = terrain / np.amax(terrain)

    terrain = terrain[loc_start[1]:loc_start[1] + N, loc_start[0]:loc_start[0] + N]
    # Creates mesh of image pixels
    x = np.linspace(0, 1, np.shape(terrain)[0])
    y = np.linspace(0, 1, np.shape(terrain)[1])
    x_mesh, y_mesh = np.meshgrid(x, y)
    z_mesh = terrain

    return x_mesh, y_mesh, z_mesh

###########################################################
############# Plotting and printing functions #############
###########################################################


def print_MSE_R2(y_data, y_model, data_str, method):
    """
    :param y_data:
    :param y_model:
    :param data_str: 'train', 'test'
    :param method: 'OLS', 'RIDGE'
    :return:
    """
    MSE = mean_squared_error(y_data, y_model)
    R2 = calculate_R2(y_data, y_model)

    data_set = {'train': 'Training', 'test': 'Test'}
    print()  # Newline for readability
    print('%s MSE for %s: %.6f' % (data_set[data_str], method, MSE))
    print('%s R2 for %s: %.6f' % (data_set[data_str], method, R2))
    return


def print_parameters_franke(seed, N, noise, p, scale, test_size):
    print('Franke Function:')
    print('Seed =  %d \nN = %d \nnoise = %.4f \np = %d \nscale = %s \ntest_size = %.2f' %
          (seed, N, noise, p, str(scale), test_size))
    return


#def plot_MSE_train_test(polydegree, train_MSE, test_MSE, n_a, test_size, noise, fig_path, task, resample=None, fs=14):
def plot_MSE_train_test(polydegree, train_MSE, test_MSE, title_mod, save, fig_path, task, resample=None, fs=14):
    fig = plt.figure()
    plt.plot(polydegree, train_MSE, label='Train')
    plt.plot(polydegree, test_MSE, label='Test')
    plt.legend()
    plt.xlabel(r'Polynomial degree $p$', fontsize=fs)
    plt.ylabel('Mean squared error', fontsize=fs)
#    plt.yscale('log')
    plt.grid('on')
#    plt.tight_layout()
    if resample is not None:
        plt.title(r'Train test MSE, %s, %s' % (resample, title_mod), fontsize=fs)
#        plt.title('%s, N = %d, test size = %.2f, noise = %.2f' % (resample, n_a*n_a, test_size, noise), fontsize=fs)
#        plt.savefig(fig_path+'task_%s/MSE_train_test_n%d_%s.png' % (task, n_a*n_a, resample))
        plt.savefig(fig_path+'task_%s/MSE_train_test_%s_%s.png' % (task, resample, save))
    else:
        plt.title(r'Train test MSE, %s' % title_mod, fontsize=fs)
#        plt.title('N = %d, test size = %.2f, noise = %.2f' % (n_a*n_a, test_size, noise), fontsize=fs)
#        plt.savefig(fig_path+'task_%s/MSE_train_test_n%d.png' % (task, n_a*n_a))
        plt.savefig(fig_path+'task_%s/MSE_train_test_%s.png' % (task, save))


def plot_MSE_test_OLS_fit(polydegree, train_MSE, test_MSE, n_a, test_size, noise, OLSmethod, fs=14):
    fig = plt.figure()
    plt.plot(polydegree, train_MSE, label='Train')
    plt.plot(polydegree, test_MSE, label='Test')
    plt.legend()
    plt.xlabel('Polynomial degree', fontsize=fs)
    plt.ylabel('Mean squared error', fontsize=fs)
    plt.title('MSE for Ordinary Least Squares', fontsize=fs)
#    plt.yscale('log')
    plt.grid('on')
    plt.title('N = %d, test size = %.2f, noise = %.2f, method=%d' % (n_a*n_a, test_size, noise, OLSmethod))
#    plt.ylim([0, 0.025])
    plt.ylim([0.0, 0.02])
    plt.tight_layout()
    plt.savefig('../figures/MSE_train_test_method%d.png' % OLSmethod)


def plot_bias_variance(polydegree, error, bias, variance, title_mod, save, fig_path, task, fs=14):
    fig = plt.figure()
    plt.plot(polydegree, error, label='Error')
    plt.plot(polydegree, bias, label='Bias')
    plt.plot(polydegree, variance, label='Variance')
    plt.xlabel('Polynomial degree', fontsize=fs)
    plt.ylabel('Mean squared error', fontsize=fs)
    plt.title(r'%s' % title_mod, fontsize=fs)
    plt.grid('on')
    plt.legend()
    plt.ylim([0.0, 0.02])
#    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(fig_path+'task_%s/bias_variance_%s.png' % (task, save))


def plot_MSE_SIMPLE(polydegree, train_MSE, test_MSE, n_a, test_size, fs=14):
    fig = plt.figure()
    plt.plot(polydegree, train_MSE, label='Train')
    plt.plot(polydegree, test_MSE, label='Test')
    plt.xlabel('Polynomial degree', fontsize=fs)
    plt.ylabel('Mean squared error', fontsize=fs)
    plt.legend()
#    plt.yscale('log')
    plt.grid('on')
    plt.title('N = %d, test size = %.2f' % (n_a*n_a, test_size))#, noise))
#        plt.savefig(fig_path+'task_%s/MSE_train_test_n%d.png' % (task, n_a*n_a))
#    plt.ylim([0, 0.025])

    # TODO: add more sophisticated filename - figure out what variation we need plot wise


def plot_confidence_int(beta, conf, method, fig_path, task, fs=14):
    n = len(beta)
    fig = plt.figure()
    plt.errorbar(range(len(beta)), beta, conf, fmt=".", capsize=3, elinewidth=1, mew=1)
    plt.title(r'Confidence intervals for $\beta$, method=%s' % method, fontsize=fs)
    plt.xlabel(r'$i$', fontsize=fs)
    plt.ylabel(r'$\beta_i$', fontsize=fs)
    plt.grid('on')
    plt.tight_layout()
    plt.savefig(fig_path + 'task_%s/beta_conf_int_n%d_%s.png' % (task, n, method))


def plot_multiple_y(x, y, label, title, xlab, ylab, save, fig_path, task, fs=14):
    fig = plt.figure()
    for i in range(len(y)):
        plt.plot(x, y[i], label=label[i])
    plt.xlabel(xlab, fontsize=fs)
    plt.ylabel(ylab, fontsize=fs)
    plt.legend()
    plt.grid('on')
    plt.title(title, fontsize=fs)
    plt.tight_layout()
    plt.savefig(fig_path+'task_%s/%s.png' % (task, save))


def plot_degree_lambda(polydegree, lambdas, title, save, fig_path, task, fs=14):
    fig = plt.figure()
    plt.plot(polydegree, lambdas)
    plt.xlabel('Polynomial degree', fontsize=fs)
    plt.ylabel(r'$\lambda$', fontsize=fs)
    plt.title(r'%s' % title, fontsize=fs)
    plt.grid('on')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(fig_path+'task_%s/degree_lambdas_%s.png' % (task, save))


def plot_heatmap(x, y, z, zlab, title, save, fig_path, task, fs=14):
    fig, ax = plt.subplots()

    heatmap = ax.pcolor(z)
    cbar = plt.colorbar(heatmap, ax=ax)

    step = 2
    xticks = ['%1.2e' % x[i] for i in range(0, len(x), step)]
    yticks = ['%d' % y[i] for i in range(0, len(y), step)]

    ax.set_xticks(np.arange(0, z.shape[1], step) + 0.5, minor=False)
    ax.set_yticks(np.arange(0, z.shape[0], step) + 0.5, minor=False)
    ax.set_xticklabels(xticks, rotation=90, fontsize=10)
    ax.set_yticklabels(yticks, fontsize=10)

    cbar.ax.set_title(zlab, fontsize=fs)
    ax.set_xlabel(r'$\lambda$', fontsize=fs)
    ax.set_ylabel('Polynomial degree', fontsize=fs)
    ax.set_title(title, fontsize=fs)
    plt.tight_layout()
    plt.savefig(fig_path+'task_%s/heatmap_%s.png' % (task, save))
    # TODO: better filename


def plot_lambda_mse(lambdas, mse, title, save, fig_path, task, fs=14):
    plt.figure()
    plt.plot(lambdas, mse)
    plt.xlabel(r'$\lambda$', fontsize=fs)
    plt.ylabel('MSE', fontsize=fs)
    plt.title(title, fontsize=fs)
    plt.xscale('log')
    plt.grid('on')
    plt.tight_layout()
    plt.savefig(fig_path+'task_%s/lambda_mse_%s.png' % (task, save))


def plot_surf(x, y, z, xlab, ylab, zlab, title, save, fig_path, task, zlim=None, fs=14):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    plt.xlabel(r'%s' % xlab, fontsize=fs)
    plt.ylabel(r'%s' % ylab, fontsize=fs)
    plt.title(r'%s' % title, fontsize=fs)

    # Customize the z axis.
    if zlim is not None:
        ax.set_zlim(zlim[0], zlim[1])
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5, ax=ax)
    cbar.ax.set_title(r'%s' % zlab, fontsize=fs)
    plt.tight_layout()
    plt.savefig(fig_path+'task_%s/lambda_mse_%s.png' % (task, save))


def plot_terrain(z, title, save, fig_path, task, fs=14):
    plt.figure()
    plt.title(title, fontsize=fs)
    plt.imshow(z, cmap='gray')
    plt.xlabel('X', fontsize=fs)
    plt.ylabel('Y', fontsize=fs)
    plt.tight_layout()
    plt.savefig(fig_path+'task_%s/%s.png' % (task, save))


def save_to_file(array_list, column_names, filename):
    outfile = open(filename, 'w')

    n_cols = len(column_names)

    line = 'p '
    for j in range(n_cols):
        line += '%s ' % column_names[j]
    line += '\n'
    outfile.write(line)

    for i in range(len(array_list[0])):
        line = '%d ' % (i+1)
        for j in range(n_cols):
            line += '%2.4e ' % array_list[j][i]
        line += '\n'
        outfile.write(line)

    outfile.close()
    return


if __name__ == '__main__':
    # TODO: make it plot the franke function?
    terrain_data = '../datafiles/SRTM_data_Norway_3.tif'
    terrain = imread(terrain_data)
    plot_terrain(terrain, 'Terrain over Norway 3', 'entire_map', '../figures/', 'f', fs=10)

    plt.show()
