from lib import functions as fun, neural_network as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from perform_analysis import PerformAnalysis

run_mode = 'd'
data = 'mnist'

fig_path = '../figures/'

fig_path = '../figures/'
data_path = '../datafiles/mnist/neuralnet/'
load_file = False  # if set to True, load files located there instead of performing analysis

filename = 'test'

scale = [True, False]  # first index is whether to subtract mean, second is to scale by std
test_size = 0.2

#save = 'N%d_pmax%d_nlamb%d_noise%.2f_seed%d' % (N, p, nlambdas, noise, seed)
#save_bs = '%s_%s_%s_Nbs%d' % (save, reg_str, 'boot', N_bootstraps)
#save_cv = '%s_%s_%s_k%d' % (save, reg_str, 'cv', K)
'''
n_epochs = [10]  # Number of epochs in SGD
batch_size = [1]  # Size of each mini-batch
eta0 = [1e-1]  # Start training rate
lambdas = [0.0, 0.1]  # Regularization
n_hidden_neurons = [10, 20]  # Number of neurons in hidden layers
n_hidden_layers = [1, 2, 3]  # Number of hidden layers
'''


n_epochs = [10, 25, 50, 100]  # Number of epochs in SGD
batch_size = [1, 5, 10, 50]  # Size of each mini-batch
eta0 = [1e-1, 1e-2, 1e-3]  # Start training rate
lambdas = [0.0, 0.1, 0.01, 0.001]  # Regularization
n_hidden_neurons = [10, 25, 50]  # Number of neurons in hidden layers
n_hidden_layers = [1]  # Number of hidden layers
# TODO: run with more hidden layers later together with neurons

# String
hyper_par_string = str(len(n_epochs)) + str(len(batch_size)) + str(len(eta0)) + \
                str(len(lambdas)) + str(len(n_hidden_neurons)) + str(len(n_hidden_layers))
data_path += '%s/' % hyper_par_string

# Cross-validation
CV = True
K = 5

# Neural net parameters
act_hidden = 'logistic'  # 'logistic', 'relu', 'tanh', 'leaky relu'
act_output = 'softmax'
wb_init = 'random'  # 'random' or 'glorot' TODO: Maybe add zero?

# Stochastic gradient descent parameters
learning_rate = 'constant'  # 'optimal'
t0 = 1  # relevant if learning_rate set to 'optimal'
t1 = 5  # relevant if learning_rate set to 'optimal'

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
# TODO: move into function
indices = np.arange(n_inputs)
random_indices = np.random.choice(indices, size=5)
for i, image in enumerate(digits.images[random_indices]):
    plt.subplot(1, 5, i+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("Label: %d" % digits.target[random_indices[i]])
#    plt.savefig()

# Split into train and test, and scale data
X_train, X_test, y_train, y_test = fun.split_data(X, y, test_size=test_size)
X_train_scaled = fun.scale_X(X_train, scale)
X_test_scaled = fun.scale_X(X_test, scale)
#X_scaled = fun.scale_X(X, scale)

# TODO #######################################################
# TODO #######################################################
# Only runs analysis if load_file=False, but since PerformAnalysis always saves to file, we load the files afterwards
# if analysis is performed, as a more robust interface with the class is missing due to time constraints.
if not load_file:
    analysis = PerformAnalysis('classification', 'neuralnet', learning_rate, data_path, filename, CV=CV, K=5, t0=t0, t1=t1)
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

#print(loss_curve_best.shape)


neural_net_SKL = MLPClassifier(hidden_layer_sizes=(int(best_params[4])), activation='logistic', solver='sgd',
                              alpha=best_params[3], batch_size=int(best_params[1]), learning_rate_init=best_params[2],
                              max_iter=int(best_params[0]), momentum=0.0, nesterovs_momentum=False)
neural_net_SKL.fit(X_train_scaled, y_train)

z_fit = neural_net_SKL.predict(X_train_scaled)
z_pred = neural_net_SKL.predict(X_test_scaled)

print('\nMLPClassifier:')
fun.print_MSE_R2(y_train, z_fit, 'train', 'NN')
fun.print_MSE_R2(y_test, z_pred, 'test', 'NN')

#print(len(neural_net_SKL.loss_curve_))
loss_SKL = neural_net_SKL.loss_curve_


##########################################
################ PLOTTING ################
##########################################
save = '%s_lr_%s_Nhyp%s' % (data, learning_rate, hyper_par_string)

fs = 16
plt.figure()
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
                 'N_epochs', 'batch size', 'Accuracy', 'Test accuracy', save, fig_path, run_mode)



#fun.plot_heatmap(n_epochs, batch_size, score[i_, j_, k_, l_, m_, n_],
#                 xlab, ylab, zlab, title, save, fig_path, task)


plt.show()





'''
lmb = 0.0  # TODO: move

# Create feed-forward neural net
neural_net = nn.NeuralNetwork(X_train, y_train, epochs=N_epochs, batch_size=batch_size, eta=eta0, lmb=lmb,
                              cost_function='MSE', learning_rate=learning_rate, t0=t0, t1=t1, gradient_scaling=1)
for i in range(len(neuron_layers)):
    neural_net.add_layer(neuron_layers[i], act_func_layers[i])

#neural_net.initialize_weights_bias(wb_init='glorot')  # that performs worse, NOT WITH CLASSIFICATION!

neural_net.fit()

y_fit = neural_net.predict(X_train)
y_pred = neural_net.predict(X_test)



def accuracy(y_data, y_model):
    n = len(y_data)
    t = np.argmax(y_model, axis=1)
    y = np.argmax(y_data, axis=1)
    res = np.sum(t == y)
    print(res, n)
    return res/n

print(accuracy(y_fit, y_train))
print(accuracy(y_pred, y_test))


##################################
neural_net_SKL = MLPClassifier(hidden_layer_sizes=(neuron_layers[0]), activation='logistic', solver='sgd',
                               alpha=lmb, batch_size=batch_size, learning_rate_init=eta0, max_iter=N_epochs)
#neural_net_SKL = MLPClassifier(activation='logistic', solver='sgd',
#                               alpha=lmb, batch_size=batch_size, learning_rate_init=1e-1, max_iter=200
#                               , learning_rate='invscaling')

neural_net_SKL.fit(X_train, y_train)

y_fit = neural_net_SKL.predict(X_train)
y_pred = neural_net_SKL.predict(X_test)
print('SKL')
print(accuracy(y_fit, y_train))
print(accuracy(y_pred, y_test))

print(neural_net_SKL.score(X_train, y_train))
print(neural_net_SKL.score(X_test, y_test))

loss_SKL = neural_net_SKL.loss_curve_

#########################################
plt.figure()
fs = 16
#N_loss = len(neural_net._loss)
#i_epochs = [int(i*N_loss/N_epochs) for i in range(N_epochs)]
#indices = np.arange(N_loss)
#plt.plot(indices, neural_net._loss, label='NeuralNet')
plt.plot(range(len(loss_SKL)), loss_SKL, label='MLPRegressor')
#plt.plot(i_epochs, [0]*len(i_epochs), '+r', ms=9, label='epochs')
plt.xlabel('epoch', fontsize=fs)
plt.ylabel('loss', fontsize=fs)
plt.title('Loss function over epoch', fontsize=fs)
plt.grid('on')
plt.legend()


plt.show()
'''

'''
###############################################
# Check if keras is better, since it has softmax. Not relevant for b, but maybe later
neural_net_SKL = MLPClassifier(hidden_layer_sizes=(neuron_layers[0]), activation='logistic', solver='sgd',
                               alpha=lmb, batch_size=batch_size, learning_rate_init=eta0, max_iter=N_epochs)
# neural_net_SKL.out_activation_ = 'softmax'  # looked at code, but this is set after calling train for the first time
# so would need to figure out how it determines it, and that looked like pain.
#  TODO: I think it uses softmax when the output is multiclass, and logistic when 1 output class
'''
