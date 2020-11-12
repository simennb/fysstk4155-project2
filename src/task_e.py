from lib import functions as fun, neural_network as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
import lib.sgd as sgd

run_mode = 'd'
data = 'mnist'

fig_path = '../figures/'
data_path = '../datafiles/'
write_path = '../datafiles/'

scale = [True, False]  # first index is whether to subtract mean, second is to scale by std
test_size = 0.2

#save = 'N%d_pmax%d_nlamb%d_noise%.2f_seed%d' % (N, p, nlambdas, noise, seed)
#save_bs = '%s_%s_%s_Nbs%d' % (save, reg_str, 'boot', N_bootstraps)
#save_cv = '%s_%s_%s_k%d' % (save, reg_str, 'cv', K)

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
#print(y)

# Stochastic gradient descent parameters
N_epochs = 1  # Number of epochs in SGD
batch_size = 1  # size of each mini-batch
eta0 = 0.001  # Start training rate
#learning_rate = 'optimal'  # constant
learning_rate = 'constant'

t0 = 1
t1 = 5

# Split into train and test, and scale data
X_train, X_test, y_train, y_test = fun.split_data(X, y, test_size=test_size)
X_train_scaled = fun.scale_X(X_train, scale)
X_test_scaled = fun.scale_X(X_test, scale)
#X_scaled = fun.scale_X(X, scale)
#print('AAAAAAAAAAAAAAAAAAAA', y_train[0:10])

lmb = 0.0  # TODO: move

logreg = sgd.LogRegSGD(n_epochs=N_epochs, n_labels=n_labels, batch_size=batch_size, eta0=eta0,
                       learning_rate=learning_rate)
logreg.set_step_length(t0=t0, t1=t1)
logreg.set_lambda(lmb)

logreg.fit(X_train_scaled, y_train)

y_fit = logreg.predict(X_train_scaled)
y_pred = logreg.predict(X_test_scaled)

#print('BBBBBBBBBBBBBBBBBBBBB', y_train[0:10])
#print('CCCCCCCCCCCCCCCCCCCCC', y_fit[0:10])


#print(y_fit)

print(fun.accuracy(y_fit, y_train))
print(fun.accuracy(y_pred, y_test))

'''

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
                               alpha=lmb, batch_size=batch_size, learning_rate_init=eta0, max_iter=N_epochs,#)
                               momentum=0.0, nesterovs_momentum=False)
#neural_net_SKL = MLPClassifier(activation='logistic', solver='sgd',
#                               alpha=lmb, batch_size=batch_size, learning_rate_init=1e-1, max_iter=200
#                               , learning_rate='invscaling')

neural_net_SKL.fit(X_train, y_train)

y_fit = neural_net_SKL.predict(X_train)
y_pred = neural_net_SKL.predict(X_test)
print('SKL')
print(accuracy(y_fit, y_train))
print(accuracy(y_pred, y_test))

#print(neural_net_SKL.score(X_train, y_train))
#print(neural_net_SKL.score(X_test, y_test))

loss_SKL = neural_net_SKL.loss_curve_

#########################################
plt.figure()
fs = 16
N_loss = len(neural_net._loss)
i_epochs = [int(i*N_loss/N_epochs) for i in range(N_epochs)]
indices = np.arange(N_loss)
plt.plot(indices, neural_net._loss, label='NeuralNet')
plt.plot(range(len(loss_SKL)), loss_SKL, label='MLPClassifier')
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