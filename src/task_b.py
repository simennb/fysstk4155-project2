import functions as fun
import regression_methods as reg
import resampling_methods as res
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
import sys
import neural_network as nn


run_mode = 'b'
data = 'franke'

fig_path = '../figures/'
data_path = '../datafiles/'
write_path = '../datafiles/'

p = 15  # degree of polynomial for the task
scale = [True, False]  # first index is whether to subtract mean, second is to scale by std

test_size = 0.2

# Regression method
#reg_str = 'OLS'
#reg_str = 'Ridge'
#reg_str = 'Lasso'  # probably not needed
#reg_str = 'SGD'
#reg_str = 'SGD_SKL'

# Creating data set for the Franke function tasks
seed = 4155
n_franke = 23  # 529 points
N = n_franke**2  # Total number of samples n*2
noise = 0.05  # noise level

# Bootstrap and CV variables
N_bootstraps = 100#int(N / 2)  # number of resamples (ex. N/2, N/4)
K = 5

# TODO: remove redundancy
# Neural net parameters
N_hidden1 = 50
N_output = 1
act_hidden1 = 'sigmoid'
act_output = 'sigmoid'

neuron_layers = [50, 1]  # number of neurons in each layer, last is output layer
act_func_layers = ['sigmoid', '']#'sigmoid']

# Stochastic gradient descent parameters
N_epochs = 50  # Number of epochs in SGD
batch_size = 1  # size of each mini-batch
N_minibatch = int(N/batch_size)  # Number of mini-batches
eta0 = 0.1  # Start training rate
learning_rate = 'meow'  # constant

# Benchmark settings
benchmark = False  # setting to True will adjust all relevant settings for all task
if benchmark is True:
    p = 5
    scale = [True, False]
#    reg_str = 'SGD'  # set to SGD maybe since thats the point of the task
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

# Split into train and test, and scale data
X_train, X_test, z_train, z_test = fun.split_data(X, z_ravel, test_size=test_size)
X_train_scaled = fun.scale_X(X_train, scale)
X_test_scaled = fun.scale_X(X_test, scale)
#X_scaled = fun.scale_X(X, scale)


lmb = 0.0

# Create feed-forward neural net
neural_net = nn.NeuralNetwork(X_train, z_train, epochs=N_epochs, batch_size=batch_size, eta=eta0, lmb=lmb)
#neural_net.add_layer(N_hidden1, act_hidden1, )  # hidden layer 1
for i in range(len(neuron_layers)):
    neural_net.add_layer(neuron_layers[i], act_func_layers[i])

neural_net.fit()

z_fit = neural_net.predict(X_train_scaled)
z_pred = neural_net.predict(X_test_scaled)

print('\nNeuralNet:')
fun.print_MSE_R2(z_train, z_fit, 'train', 'NN')
fun.print_MSE_R2(z_test, z_pred, 'test', 'NN')

lecturenet = nn.LectureNetwork(X_train, z_train, 50, 1, N_epochs, batch_size, eta0, lmb)
lecturenet.train()

z_fit = lecturenet.predict_probabilities(X_train_scaled)
z_pred = lecturenet.predict_probabilities(X_test_scaled)

print('\nLectureNet:')
fun.print_MSE_R2(z_train, z_fit, 'train', 'NN')
fun.print_MSE_R2(z_test, z_pred, 'test', 'NN')
