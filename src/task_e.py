from lib import functions as fun, neural_network as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn import datasets
from perform_analysis import PerformAnalysis
import os

run_mode = 'e'
fig_path = '../figures/'
data_path = '../datafiles/mnist/sgd/'

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
batch_size = [1]  # Size of each mini-batch
eta0 = [1e-3]  # Start training rate
lambdas = [1e-1]  # Regularization

'''
filename = 'test'
n_epochs = [10, 25, 50, 100]  # Number of epochs in SGD
batch_size = [1, 5, 10, 50]  # Size of each mini-batch
eta0 = [1e-1, 1e-2, 1e-3]  # Start training rate
lambdas = [0.0, 0.1, 0.01, 0.001]  # Regularization
'''

# Stochastic gradient descent parameters
learning_rate = 'constant'  # 'optimal'

t0 = 1
t1 = 10

# Test and scaling
scale = [True, False]  # first index is whether to subtract mean, second is to scale by std
test_size = 0.2

# Cross-validation
CV = True
K = 5

# String for folder and filenames
hyper_par_string = str(len(n_epochs)) + str(len(batch_size)) + str(len(eta0)) + str(len(lambdas))
data_path += '%s_%s/' % (filename, hyper_par_string)

# Set up the MNIST dataset
digits = datasets.load_digits()
inputs = digits.images
labels = digits.target

# Reshape input and labels into design matrix X and target vector y
n_inputs = len(inputs)
n_labels = 10
X = inputs.reshape(n_inputs, -1)
y = np.zeros((n_inputs, n_labels))
y[np.arange(n_inputs), labels] = 1

# Split into train and test, and scale data (only used by SKL)
X_train, X_test, y_train, y_test = fun.split_data(X, labels, test_size=test_size)
X_train_scaled = fun.scale_X(X_train, scale)
X_test_scaled = fun.scale_X(X_test, scale)

# Only runs analysis if load_file=False, but since PerformAnalysis always saves to file, we load the files afterwards
# if analysis is performed, as a more robust interface with the class is missing due to time constraints.
if not load_file:
    analysis = PerformAnalysis('classification', 'sgd', learning_rate, data_path, filename, CV=CV, K=K, t0=t0, t1=t1)
    analysis.set_hyperparameters(n_epochs, batch_size, eta0, lambdas)
    analysis.set_data(X, y, test_size=test_size, scale=scale)
    analysis.run()

if load_file or analysis.analysed:
    score = np.load(data_path+filename+'_score.npy')
    best_index = np.load(data_path+filename+'_best_index.npy')
    loss_curve_best = np.load(data_path+filename+'_loss_curve_best.npy')
i_, j_, k_, l_ = best_index
print(best_index)

sgd_skl = SGDClassifier(alpha=lambdas[l_], max_iter=n_epochs[i_])
sgd_skl.fit(X_train_scaled, y_train)

y_fit = sgd_skl.predict(X_train_scaled)
y_pred = sgd_skl.predict(X_test_scaled)

print('\nSGDClassifier:')
print('Train accuracy: ', sgd_skl.score(X_train_scaled, y_train))
print('Test accuracy: ', sgd_skl.score(X_test_scaled, y_test))


##########################################
################ PLOTTING ################
##########################################
# Results and figure saving could definitely be improved, and should have been considered more earlier in the process.
save = filename + '_lr_%s_Nhyp%s' % (learning_rate, hyper_par_string)
run_mode += '/%s' % filename
if not os.path.exists(fig_path + 'task_%s' % run_mode):
    os.mkdir(fig_path + 'task_%s' % run_mode)


for i, metric in zip(range(1), ['Accuracy']):  # easier to copy paste and reuse code from task_b
    # N_epochs vs batch_size
    fun.plot_heatmap(n_epochs, batch_size, score[:, :, k_, l_, i, 1].T,
                     'N epochs', 'batch size', metric, 'Test %s' % metric,
                     save+'_%s_%s' % (metric, 'n_epochs_bsize'), fig_path, run_mode, xt='int', yt='int')

    # N_epochs vs eta0
    fun.plot_heatmap(n_epochs, eta0, score[:, j_, :, l_, i, 1].T,
                     'N epochs', 'learning rate', metric, 'Test %s' % metric,
                     save+'_%s_%s' % (metric, 'n_epochs_eta0'), fig_path, run_mode, xt='int', yt='exp')

    # N_epochs vs lambdas
    fun.plot_heatmap(n_epochs, lambdas, score[:, j_, k_, :, i, 1].T,
                     'N epochs', 'lambda', metric, 'Test %s' % metric,
                     save+'_%s_%s' % (metric, 'n_epochs_lambdas'), fig_path, run_mode, xt='int', yt='exp')

    # batch_size vs lambdas
    fun.plot_heatmap(batch_size, lambdas, score[i_, :, k_, :, i, 1].T,
                     'batch size', 'lambda', metric, 'Test %s' % metric,
                     save+'_%s_%s' % (metric, 'bs_lambdas'), fig_path, run_mode, xt='int', yt='exp')

    # eta0 vs lambdas
    fun.plot_heatmap(eta0, lambdas, score[i_, j_, :, :, i, 1].T,
                     'learning rate', 'lambda', metric, 'Test %s' % metric,
                     save+'_%s_%s' % (metric, 'eta0_lambdas'), fig_path, run_mode, xt='exp', yt='exp')

plt.show()
