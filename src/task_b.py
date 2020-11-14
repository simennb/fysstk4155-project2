from lib import functions as fun, neural_network as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from perform_analysis import PerformAnalysis
from sklearn.neural_network import MLPRegressor


run_mode = 'b'
fig_path = '../figures/'
data_path = '../datafiles/franke/neuralnet/'

load_file = False  # if set to True, load files located there instead of performing analysis
#load_file = True

compare_skl = True  # whether or not the SKL loss curve is to be plotted

# Hyperparameters to loop over
################################
# The blocks below are some of the filename and parameter combinations that were used
# to create results in the report.
# This could probably have been done more optimal with parameter files or something like that.
filename = 'best_params'
n_epochs = [100]  # Number of epochs in SGD
batch_size = [50]  # Size of each mini-batch
eta0 = [1e-1]  # Start training rate
lambdas = [1e-3]  # Regularization
n_hidden_neurons = [25]  # Number of neurons in hidden layers
n_hidden_layers = [1]  # Number of hidden layers

'''
filename = 'test'
n_epochs = [10, 25, 50, 100]  # Number of epochs in SGD
batch_size = [1, 5, 10, 50]  # Size of each mini-batch
eta0 = [1e-1, 1e-2, 1e-3]  # Start training rate
lambdas = [0.0, 0.1, 0.01, 0.001]  # Regularization
n_hidden_neurons = [10, 25, 50]  # Number of neurons in hidden layers
n_hidden_layers = [1]  # Number of hidden layers
'''

# Neural net parameters
act_hidden = 'logistic'  # 'logistic', 'relu', 'tanh', 'leaky relu'
act_output = 'identity'
wb_init = 'random'  # 'random' or 'glorot'

# Stochastic gradient descent parameters
learning_rate = 'constant'  # 'optimal'

t0 = 1  # relevant if learning_rate set to 'optimal'
t1 = 10  # relevant if learning_rate set to 'optimal'

# Franke function data set parameters
seed = 4155
n_franke = 23  # 529 points
N = n_franke**2  # Total number of samples n*2
noise = 0.05  # noise level

# Test and scaling
scale = [True, False]  # first index is whether to subtract mean, second is to scale by std
test_size = 0.2

# Cross-validation
CV = True
K = 5

# String for folder and filenames
hyper_par_string = str(len(n_epochs)) + str(len(batch_size)) + str(len(eta0)) + \
                str(len(lambdas)) + str(len(n_hidden_neurons)) + str(len(n_hidden_layers))
data_path += '%s_%s/' % (filename, hyper_par_string)

# Printing some information for logging purposes
fun.print_parameters_franke(seed, N, noise, 0, scale, test_size)

# Create data set
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
z = z_ravel.reshape(-1, 1)

# Creating design matrix
X = np.zeros((x_ravel.shape[0], 2))
X[:, 0] = x_ravel
X[:, 1] = y_ravel

# Split into train and test, and scale data (only used by SKL)
X_train, X_test, z_train, z_test = fun.split_data(X, z, test_size=test_size)
X_train_scaled = fun.scale_X(X_train, scale)
X_test_scaled = fun.scale_X(X_test, scale)

# Only runs analysis if load_file=False, but since PerformAnalysis always saves to file, we load the files afterwards
# if analysis is performed, as a more robust interface with the class is missing due to time constraints.
if not load_file:
    analysis = PerformAnalysis('regression', 'neuralnet', learning_rate, data_path, filename, CV=CV, K=K, t0=t0, t1=t1)
    analysis.set_hyperparameters(n_epochs, batch_size, eta0, lambdas, n_hidden_neurons, n_hidden_layers)
    analysis.set_neural_net_params(act_hidden, act_output, wb_init)
    analysis.set_data(X, z, test_size=test_size, scale=scale)
    analysis.run()

if load_file or analysis.analysed:
    score = np.load(data_path+filename+'_score.npy')
    best_index = np.load(data_path+filename+'_best_index.npy')
    best_params = np.load(data_path+filename+'_best_params.npy')
    loss_curve_best = np.load(data_path+filename+'_loss_curve_best.npy')
i_, j_, k_, l_, m_, n_ = best_index

print('Best params:', best_params)


neural_net_SKL = MLPRegressor(hidden_layer_sizes=(int(best_params[4])), activation='logistic', solver='sgd',
                              alpha=best_params[3], batch_size=int(best_params[1]), learning_rate_init=best_params[2],
                              max_iter=int(best_params[0]), momentum=0.0, nesterovs_momentum=False)
neural_net_SKL.fit(X_train_scaled, z_train)

z_fit = neural_net_SKL.predict(X_train_scaled)
z_pred = neural_net_SKL.predict(X_test_scaled)

print('\nMLPRegressor:')
fun.print_MSE_R2(z_train, z_fit, 'train', 'NN')
fun.print_MSE_R2(z_test, z_pred, 'test', 'NN')

print(len(neural_net_SKL.loss_curve_))
loss_SKL = neural_net_SKL.loss_curve_


##########################################
################ PLOTTING ################
##########################################

# Results and figure saving could definitely be improved, and should have been considered more earlier in the process.
save = filename + '_lr_%s_Nhyp%s' % (learning_rate, hyper_par_string)
run_mode += '/%s' % filename
if not os.path.exists(fig_path + 'task_%s' % run_mode):
    os.mkdir(fig_path + 'task_%s' % run_mode)


fs = 16
indices = np.arange(loss_curve_best.shape[0])
plt.plot(indices, np.mean(loss_curve_best, axis=1), label='NeuralNet')
if compare_skl:
    plt.plot(range(len(loss_SKL)), loss_SKL, label='MLPRegressor')
plt.xlabel('epoch', fontsize=fs)
plt.ylabel('loss', fontsize=fs)
plt.title('Loss function over epoch', fontsize=fs)
plt.grid('on')
plt.legend()
plt.savefig(fig_path + 'task_%s/loss.png' % run_mode)

for i, metric in zip(range(2), ['MSE', 'R2']):
    # N_epochs vs batch_size
    fun.plot_heatmap(n_epochs, batch_size, score[:, :, k_, l_, m_, n_, i, 1].T,
                     'N epochs', 'batch size', metric, 'Test %s' % metric,
                     save+'_%s_%s' % (metric, 'n_epochs_bsize'), fig_path, run_mode, xt='int', yt='int')

    # N_epochs vs eta0
    fun.plot_heatmap(n_epochs, eta0, score[:, j_, :, l_, m_, n_, i, 1].T,
                     'N epochs', 'learning rate', metric, 'Test %s' % metric,
                     save+'_%s_%s' % (metric, 'n_epochs_eta0'), fig_path, run_mode, xt='int', yt='exp')

    # N_epochs vs lambdas
    fun.plot_heatmap(n_epochs, lambdas, score[:, j_, k_, :, m_, n_, i, 1].T,
                     'N epochs', 'lambda', metric, 'Test %s' % metric,
                     save+'_%s_%s' % (metric, 'n_epochs_lambdas'), fig_path, run_mode, xt='int', yt='exp')

    # batch_size vs lambdas
    fun.plot_heatmap(batch_size, lambdas, score[i_, :, k_, :, m_, n_, i, 1].T,
                     'batch size', 'lambda', metric, 'Test %s' % metric,
                     save+'_%s_%s' % (metric, 'bs_lambdas'), fig_path, run_mode, xt='int', yt='exp')

    # eta0 vs lambdas
    fun.plot_heatmap(eta0, lambdas, score[i_, j_, :, :, m_, n_, i, 1].T,
                     'learning rate', 'lambda', metric, 'Test %s' % metric,
                     save+'_%s_%s' % (metric, 'eta0_lambdas'), fig_path, run_mode, xt='exp', yt='exp')

    # NN specific heatmaps

    # N_epochs vs N_h_neurons
    fun.plot_heatmap(n_epochs, n_hidden_neurons, score[:, j_, k_, l_, :, n_, i, 1].T,
                     'N epochs', 'N hidden neurons', metric, 'Test %s' % metric,
                     save+'_%s_%s' % (metric, 'n_epochs_nhn'), fig_path, run_mode, xt='int', yt='int')

    # batch_size vs N_h_neurons
    fun.plot_heatmap(batch_size, n_hidden_neurons, score[i_, :, k_, l_, :, n_, i, 1].T,
                     'batch size', 'N hidden neurons', metric, 'Test %s' % metric,
                     save+'_%s_%s' % (metric, 'bs_nhn'), fig_path, run_mode, xt='int', yt='int')

    # eta0 vs N_h_neurons
    fun.plot_heatmap(eta0, n_hidden_neurons, score[i_, j_, :, l_, :, n_, i, 1].T,
                     'learning rate', 'N hidden neurons', metric, 'Test %s' % metric,
                     save+'_%s_%s' % (metric, 'eta0_nhn'), fig_path, run_mode, xt='exp', yt='int')

    # lambdas vs N_h_neurons
    fun.plot_heatmap(lambdas, n_hidden_neurons, score[i_, j_, k_, :, :, n_, i, 1].T,
                     'lambda', 'N hidden neurons', metric, 'Test %s' % metric,
                     save+'_%s_%s' % (metric, 'lambdas_nhn'), fig_path, run_mode, xt='exp', yt='int')

    ########################################################################################################
    if len(n_hidden_layers) > 1:  # No reason to plot unless we have more than one hidden layer
        # N_epochs vs N_h_layers
        fun.plot_heatmap(n_epochs, n_hidden_layers, score[:, j_, k_, l_, m_, :, i, 1].T,
                         'N epochs', 'N hidden layers', metric, 'Test %s' % metric,
                         save+'_%s_%s' % (metric, 'n_epochs_nhl'), fig_path, run_mode, xt='int', yt='int')

        # batch_size vs N_h_layers
        fun.plot_heatmap(batch_size, n_hidden_layers, score[i_, :, k_, l_, m_, :, i, 1].T,
                         'batch size', 'N hidden layers', metric, 'Test %s' % metric,
                         save+'_%s_%s' % (metric, 'bs_nhl'), fig_path, run_mode, xt='int', yt='int')

        # eta0 vs N_h_layers
        fun.plot_heatmap(eta0, n_hidden_layers, score[i_, j_, :, l_, m_, :, i, 1].T,
                         'learning rate', 'N hidden layers', metric, 'Test %s' % metric,
                         save+'_%s_%s' % (metric, 'eta0_nhl'), fig_path, run_mode, xt='exp', yt='int')

        # lambdas vs N_h_layers
        fun.plot_heatmap(lambdas, n_hidden_layers, score[i_, j_, k_, :, m_, :, i, 1].T,
                         'lambda', 'N hidden layers', metric, 'Test %s' % metric,
                         save+'_%s_%s' % (metric, 'lambdas_nhl'), fig_path, run_mode, xt='exp', yt='int')

        # lambdas vs N_h_layers
        fun.plot_heatmap(n_hidden_neurons, n_hidden_layers, score[i_, j_, k_, l_, :, :, i, 1].T,
                         'N hidden neurons', 'N hidden layers', metric, 'Test %s' % metric,
                         save+'_%s_%s' % (metric, 'nhn_nhl'), fig_path, run_mode, xt='int', yt='int')

plt.show()
