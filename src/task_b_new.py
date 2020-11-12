from lib import functions as fun, neural_network as nn
import numpy as np
import matplotlib.pyplot as plt
from perform_analysis import PerformAnalysis
from sklearn.neural_network import MLPRegressor


run_mode = 'b'
data = 'franke'

fig_path = '../figures/'
data_path = '../datafiles/franke/neuralnet/'
load_file = False  # if set to True, load files located there instead of performing analysis

filename = 'test'

scale = [True, False]  # first index is whether to subtract mean, second is to scale by std
test_size = 0.2

# Hyperparameters to loop over
'''
n_epochs = 50  # Number of epochs in SGD
batch_size = 10  # Size of each mini-batch
eta0 = 0.1  # Start training rate
lambdas = np.array([0.0])  # Regularization
n_hidden_neurons = 50  # Number of neurons in hidden layers
n_hidden_layers = 1  # Number of hidden layers
'''

# TODO: maybe linspace/logspace
# I suspect this will take a while
'''
n_epochs = np.array([25, 50, 75, 100, 200], dtype=int)  # Number of epochs in SGD
batch_size = np.array([1, 5, 10, 50, 100], dtype=int)  # Size of each mini-batch
eta0 = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])  # Start training rate
lambdas = np.array([0.0, 1e-1, 1e-2, 1e-3, 1e-4])  # Regularization
n_hidden_neurons = np.array([25, 50, 75, 100, 200], dtype=int)  # Number of neurons in hidden layers
n_hidden_layers = np.array([1, 2, 3, 4, 5], dtype=int)  # Number of hidden layers
'''
'''
n_epochs = [25, 50, 75, 100, 200]  # Number of epochs in SGD
batch_size = [1, 5, 10, 50, 100]  # Size of each mini-batch
eta0 = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]  # Start training rate
lambdas = [0.0, 1e-1, 1e-2, 1e-3, 1e-4]  # Regularization
n_hidden_neurons = [25, 50, 75, 100, 200]  # Number of neurons in hidden layers
n_hidden_layers = [1, 2, 3, 4, 5]  # Number of hidden layers
'''
'''
n_epochs = [50]  # Number of epochs in SGD
batch_size = [1, 5, 10]  # Size of each mini-batch
eta0 = [1e-1]  # Start training rate
lambdas = [0.0]  # Regularization
n_hidden_neurons = [25, 50]  # Number of neurons in hidden layers
n_hidden_layers = [1, 2]  # Number of hidden layers
'''
n_epochs = [50]  # Number of epochs in SGD
batch_size = [1]  # Size of each mini-batch
eta0 = [1e-1]  # Start training rate
lambdas = [0.0]  # Regularization
n_hidden_neurons = [50]  # Number of neurons in hidden layers
n_hidden_layers = [1]  # Number of hidden layers


# String
hyper_par_string = str(len(n_epochs)) + str(len(batch_size)) + str(len(eta0)) + \
                str(len(lambdas)) + str(len(n_hidden_neurons)) + str(len(n_hidden_layers))
data_path += '%s/' % hyper_par_string

# Franke function data set parameters
seed = 4155
n_franke = 23  # 529 points
N = n_franke**2  # Total number of samples n*2
noise = 0.05  # noise level

# Cross-validation
CV = True
K = 5

# Neural net parameters
act_hidden = 'logistic'  # 'logistic', 'relu', 'tanh', 'leaky relu'
act_output = 'identity'
wb_init = 'random'  # 'random' or 'glorot' TODO: Maybe add zero?

# Stochastic gradient descent parameters
learning_rate = 'constant'  # 'optimal'
t0 = 1  # relevant if learning_rate set to 'optimal'
t1 = 5  # relevant if learning_rate set to 'optimal'

# TODO: Benchmark settings / Figure out
benchmark = False  # setting to True will adjust all relevant settings for all task
if benchmark is True:
    scale = [True, False]
    n_franke = 23
    N = 529
    noise = 0.05
    N_bootstraps = 264
    K = 5
    N_epochs = 100
    N_minibatch = 10
    eta0 = 0.1


# Printing some information for logging purposes
fun.print_parameters_franke(seed, N, noise, 0, scale, test_size)  # TODO: fix p dependence

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
print(z_ravel.shape)
z = z_ravel.reshape(-1, 1)

# Creating design matrix
X = np.zeros((x_ravel.shape[0], 2))  # TODO: maybe easier way
X[:, 0] = x_ravel
X[:, 1] = y_ravel

# Split into train and test, and scale data
X_train, X_test, z_train, z_test = fun.split_data(X, z, test_size=test_size)
X_train_scaled = fun.scale_X(X_train, scale)
X_test_scaled = fun.scale_X(X_test, scale)
X_scaled = fun.scale_X(X, scale)

#X_train_copy =X_train_scaled.copy()
#X_test_copy = X_test_scaled.copy()

# TODO #######################################################
# TODO #######################################################
# Only runs analysis if load_file=False, but since PerformAnalysis always saves to file, we load the files afterwards
# if analysis is performed, as a more robust interface with the class is missing due to time constraints.
if not load_file:
    analysis = PerformAnalysis('regression', 'neuralnet', learning_rate, data_path, filename, CV=CV, K=5, t0=t0, t1=t1)
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

print(loss_curve_best.shape)


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
#hyppar_string = str(len(n_epochs)) + str(len(batch_size)) + str(len(eta0)) + \
#                str(len(lambdas)) + str(len(n_hidden_neurons)) + str(len(n_hidden_layers))
save = '%s_N%d_noise%.2f_seed%d_lr_%s_Nhyp%s' % (data, N, noise, seed, learning_rate, hyper_par_string)

fs = 16
#N_loss = len(neural_net._loss)
#i_epochs = [int(i*N_loss/N_epochs) for i in range(N_epochs)]
indices = np.arange(loss_curve_best.shape[0])
for i in range(loss_curve_best.shape[1]):
    plt.plot(indices, loss_curve_best[:, i], label='fold=%d' % (i + 1))
plt.plot(range(len(loss_SKL)), loss_SKL, label='MLPRegressor')
#plt.plot(i_epochs, [0]*len(i_epochs), '+r', ms=9, label='epochs')
plt.xlabel('epoch', fontsize=fs)
plt.ylabel('loss', fontsize=fs)
plt.title('Loss function over epoch', fontsize=fs)
plt.grid('on')
plt.legend()

# score[i_, i_, k_, l_, m_, n_, 0, 1]
fun.plot_heatmap(n_epochs, batch_size, score[:, :, k_, l_, m_, n_, 0, 1],
                 'N_epochs', 'batch size', 'MSE', 'Test MSE', save, fig_path, run_mode)



#fun.plot_heatmap(n_epochs, batch_size, score[i_, j_, k_, l_, m_, n_],
#                 xlab, ylab, zlab, title, save, fig_path, task)


plt.show()

'''
# Create feed-forward neural net
neural_net = nn.NeuralNetwork(X_train_scaled, z_train, epochs=N_epochs, batch_size=batch_size, eta=eta0, lmb=lmb,
                              cost_function='MSE', learning_rate=learning_rate, t0=t0, t1=t1, gradient_scaling=1)
for i in range(len(neuron_layers)):
    neural_net.add_layer(neuron_layers[i], act_func_layers[i])

neural_net.initialize_weights_bias(wb_init='glorot')  # that performs worse, hmm

neural_net.fit()

z_fit = neural_net.predict(X_train_scaled)
z_pred = neural_net.predict(X_test_scaled)

print('\nNeuralNet:')
fun.print_MSE_R2(z_train, z_fit, 'train', 'NN')
fun.print_MSE_R2(z_test, z_pred, 'test', 'NN')

# Maybe keras?
neural_net_SKL = MLPRegressor(hidden_layer_sizes=(neuron_layers[0]), activation='logistic', solver='sgd',
                              alpha=lmb, batch_size=batch_size, learning_rate_init=eta0, max_iter=N_epochs,
                              momentum=0.0, nesterovs_momentum=False)
neural_net_SKL.fit(X_train_scaled, z_train)

z_fit = neural_net_SKL.predict(X_train_scaled)
z_pred = neural_net_SKL.predict(X_test_scaled)

print('\nMLPRegressor:')
fun.print_MSE_R2(z_train, z_fit, 'train', 'NN')
fun.print_MSE_R2(z_test, z_pred, 'test', 'NN')

print(len(neural_net_SKL.loss_curve_))
loss_SKL = neural_net_SKL.loss_curve_

#########################################
fs = 16
N_loss = len(neural_net._loss)
#i_epochs = [int(i*N_loss/N_epochs) for i in range(N_epochs)]
indices = np.arange(N_loss)
plt.plot(indices, neural_net._loss, label='NeuralNet')
plt.plot(range(len(loss_SKL)), loss_SKL, label='MLPRegressor')
#plt.plot(i_epochs, [0]*len(i_epochs), '+r', ms=9, label='epochs')
plt.xlabel('epoch', fontsize=fs)
plt.ylabel('loss', fontsize=fs)
plt.title('Loss function over epoch', fontsize=fs)
plt.grid('on')
plt.legend()
plt.show()
'''