from lib import functions as fun, neural_network as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from perform_analysis import PerformAnalysis


run_mode = 'd'
fig_path = '../figures/'
data_path = '../datafiles/mnist/neuralnet/'

load_file = False  # if set to True, load files located there instead of performing analysis
#load_file = True

compare_skl = True  # whether or not the SKL loss curve is to be plotted

# Hyperparameters to loop over
################################
# The blocks below are some of the filename and parameter combinations that were used
# to create results in the report.
# This could probably have been done more optimal with parameter files or something like that.
filename = 'best_params'
n_epochs = [10]  # Number of epochs in SGD
batch_size = [5]  # Size of each mini-batch
eta0 = [1e-1]  # Start training rate
lambdas = [1e-2]  # Regularization
n_hidden_neurons = [10]  # Number of neurons in hidden layers
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
act_output = 'softmax'
wb_init = 'random'  # 'random' or 'glorot'

# Stochastic gradient descent parameters
learning_rate = 'constant'  # 'optimal'

t0 = 1  # relevant if learning_rate set to 'optimal'
t1 = 10  # relevant if learning_rate set to 'optimal'

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

# Plot a few randomly selected images from the dataset
indices = np.arange(n_inputs)
random_indices = np.random.choice(indices, size=5)
for i, image in enumerate(digits.images[random_indices]):
    plt.subplot(1, 5, i+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("Label: %d" % digits.target[random_indices[i]])
#    plt.savefig()

# Split into train and test, and scale data (only used by SKL)
X_train, X_test, y_train, y_test = fun.split_data(X, y, test_size=test_size)
X_train_scaled = fun.scale_X(X_train, scale)
X_test_scaled = fun.scale_X(X_test, scale)

# Only runs analysis if load_file=False, but since PerformAnalysis always saves to file, we load the files afterwards
# if analysis is performed, as a more robust interface with the class is missing due to time constraints.
if not load_file:
    analysis = PerformAnalysis('classification', 'neuralnet', learning_rate, data_path, filename, CV=CV, K=K, t0=t0, t1=t1)
    analysis.set_hyperparameters(n_epochs, batch_size, eta0, lambdas, n_hidden_neurons, n_hidden_layers)
    analysis.set_neural_net_params(act_hidden, act_output, wb_init)
    analysis.set_data(X, y, test_size=test_size, scale=scale)
    analysis.run()

if load_file or analysis.analysed:
    score = np.load(data_path+filename+'_score.npy')
    best_index = np.load(data_path+filename+'_best_index.npy')
    best_params = np.load(data_path+filename+'_best_params.npy')
    loss_curve_best = np.load(data_path+filename+'_loss_curve_best.npy')
i_, j_, k_, l_, m_, n_ = best_index

# Printing best indices since initial run had an error in determining best model (instead picking the worst model)
ii = np.argmax(score)
true_best_index = np.unravel_index(ii, score.shape)
print(np.unravel_index(ii, score.shape))
i_, j_, k_, l_, m_, n_ = true_best_index[0:-2]

# Using SKL for the same parameter settings
neural_net_SKL = MLPClassifier(hidden_layer_sizes=(int(best_params[4])), activation='logistic', solver='sgd',
                              alpha=best_params[3], batch_size=int(best_params[1]), learning_rate_init=best_params[2],
                              max_iter=int(best_params[0]), momentum=0.0, nesterovs_momentum=False)
neural_net_SKL.fit(X_train_scaled, y_train)

y_fit = neural_net_SKL.predict(X_train_scaled)
y_pred = neural_net_SKL.predict(X_test_scaled)

print('\nMLPClassifier:')
print('Train accuracy: ', fun.accuracy(y_train, y_fit))
print('Test accuracy: ', fun.accuracy(y_test, y_pred))

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
plt.figure()
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


for i, metric in zip(range(1), ['Accuracy']):  # easier to copy paste and reuse code from task_b
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
