from lib import functions as fun, neural_network as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from lib import resampling_methods as res

run_mode = 'b'
data = 'franke'

fig_path = '../figures/'
data_path = '../datafiles/'
write_path = '../datafiles/'

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

# Neural net parameters
n_hidden = 1
nodes_hidden = 50
act_hidden = 'logistic'
act_output = 'identity'

neuron_layers = [nodes_hidden] * n_hidden + [1]  # list of neurons in each layer (minus input layer)
act_func_layers = [act_hidden] * n_hidden + [act_output]

# Stochastic gradient descent parameters
N_epochs = 50  # Number of epochs in SGD
batch_size = 5  # size of each mini-batch
N_minibatch = int(N/batch_size)  # Number of mini-batches  # TODO: DOES NOT TAKE TRAIN TEST SPLIT INTO ACCOUNT
eta0 = 0.1  # Start training rate
#learning_rate = 'optimal'  # constant
learning_rate = 'constant'

t0 = 1
t1 = 5

# Benchmark settings
benchmark = False  # setting to True will adjust all relevant settings for all task
if benchmark is True:
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
fun.print_parameters_franke(seed, N, noise, 0, scale, test_size)  # TODO: fix p dependence


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
#print(z_ravel.shape)

# Creating design matrix
X = np.zeros((x_ravel.shape[0], 2))  # TODO: see if an easier way to do this with meshgrid?
X[:, 0] = x_ravel
X[:, 1] = y_ravel
#X = fun.generate_polynomial(x_ravel, y_ravel, p)

# Split into train and test, and scale data
X_train, X_test, z_train, z_test = fun.split_data(X, z, test_size=test_size)
X_train_scaled = fun.scale_X(X_train, scale)
X_test_scaled = fun.scale_X(X_test, scale)
X_scaled = fun.scale_X(X, scale)

lmb = 0.0#1
X_train_copy = X_train_scaled.copy()
X_test_copy = X_test_scaled.copy()

# Create feed-forward neural net
neural_net = nn.NeuralNetwork(X_train_scaled, z_train, epochs=N_epochs, batch_size=batch_size, eta=eta0, lmb=lmb,
                              cost_function='MSE', learning_rate=learning_rate, t0=t0, t1=t1, gradient_scaling=1,
                              wb_init='glorot')
for i in range(len(neuron_layers)):
    neural_net.add_layer(neuron_layers[i], act_func_layers[i])

#neural_net.initialize_weights_bias(wb_init='glorot')  # Gives very similar results to SKL

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

#neural_net.initialize_weights_bias(wb_init='glorot')  # Gives very similar results to SKL
CV = res.CrossValidation(X_scaled, z_ravel, neural_net, stat=[fun.mean_squared_error, fun.calculate_R2])
#CV = res.CrossValidation(X, y, neural_net_SKL, stat=[fun.mean_squared_error, fun.calculate_R2])
error_train, error_test = CV.compute(K=5)
print(error_train, error_test)
#[0.01610957 0.84249185] [0.01620809 0.64487109]

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
