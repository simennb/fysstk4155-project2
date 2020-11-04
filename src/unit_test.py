import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
import neural_network as nn
import functions as fun
import sgd
import time


def test_SGD(x, y):
    X = np.c_[np.ones((m, 1)), x]
    beta_linreg = np.linalg.inv(X.T @ X) @ (X.T @ y)
    print("Linear inversion")
    print(beta_linreg)
    sgdreg = SGDRegressor(max_iter=N_epochs, penalty=None, eta0=eta0, tol=None, learning_rate=learning_rate)
#    sgdreg = SGDRegressor(max_iter=N_epochs, penalty=None, eta0=0.1, tol=None)
#    sgdreg = SGDRegressor(max_iter=N_epochs, penalty=None, eta0=0.1, tol=None, learning_rate='constant')
    ts_skl = time.time()
    sgdreg.fit(x, y.ravel())
    te_skl = time.time()
    print("sgdreg from scikit")
    print(sgdreg.intercept_, sgdreg.coef_)

    sgdreg_own = sgd.LinRegSGD(N_epochs, batch_size=batch_size, eta0=eta0, learning_rate=learning_rate)
    sgdreg_own.set_step_length(5, 50.0)
    sgdreg_own.fit(X, y.ravel())  # First time its run will also compile
    ts_own = time.time()
    sgdreg_own.fit(X, y.ravel())
    te_own = time.time()
    print("LinRegSGD")
    print(sgdreg_own.theta)

    t_skl = te_skl - ts_skl
    t_own = te_own - ts_own
    print('Time SKL: %.3e s' % t_skl)
    print('Time own: %.3e s' % t_own)

    try:
        print('Factor own/skl = %.3e' % (t_own/t_skl))
    except ZeroDivisionError:
        print('t_skl = 0, t_own = ', t_own)


def test_FFNN(x, y):
    N_hidden = 20
    x_train, x_test, y_train, y_test = fun.split_data(x, y)

    neural_net = nn.NeuralNetwork(x_train, y_train, epochs=N_epochs, batch_size=batch_size, eta=eta0, lmb=lmb,
                                  cost_function='MSE', learning_rate=learning_rate, t0=t0, t1=t1)
    neural_net.add_layer(N_hidden, 'logistic')
    neural_net.add_layer(1, 'identity')

    neural_net.fit()

    y_fit = neural_net.predict(x_train)
    y_pred = neural_net.predict(x_test)

    print('\nNeuralNet:')
    fun.print_MSE_R2(y_train, y_fit, 'train', 'NN')
    fun.print_MSE_R2(y_test, y_pred, 'test', 'NN')

    neural_net_SKL = MLPRegressor(hidden_layer_sizes=(N_hidden), activation='logistic', solver='sgd',
                                  alpha=lmb, batch_size=batch_size, learning_rate_init=eta0, max_iter=N_epochs,
                                  momentum=0.9)#, nesterovs_momentum=False)
    neural_net_SKL.fit(x_train, y_train)

    y_fit = neural_net_SKL.predict(x_train)
    y_pred = neural_net_SKL.predict(x_test)

    print('\nMLPRegressor:')
    fun.print_MSE_R2(y_train, y_fit, 'train', 'NN')
    fun.print_MSE_R2(y_test, y_pred, 'test', 'NN')


'''
m = 100
x = 2*np.random.rand(m,1)
y = 4+3*x+np.random.randn(m,1)

X = np.c_[np.ones((m,1)), x]
theta_linreg = np.linalg.inv(X.T @ X) @ (X.T @ y)

theta = np.random.randn(2,1)
eta = 0.1
Niterations = 1000


for iter in range(Niterations):
    gradients = 2.0/m*X.T @ ((X @ theta)-y)
    theta -= eta*gradients
print("theta from own gd")
print(theta)

xnew = np.array([[0],[2]])
Xnew = np.c_[np.ones((2,1)), xnew]
ypredict = Xnew.dot(theta)
ypredict2 = Xnew.dot(theta_linreg)


n_epochs = 50
t0, t1 = 5, 50
def learning_schedule(t):
    return t0/(t+t1)

theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T @ ((xi @ theta)-yi)
        eta = learning_schedule(epoch*m+i)
        theta = theta - eta*gradients
print("theta from own sdg")
print(theta)

plt.plot(xnew, ypredict, "r-")
plt.plot(xnew, ypredict2, "b-")
plt.plot(x, y ,'ro')
plt.axis([0,2.0,0, 15.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Random numbers ')
plt.show()
'''

if __name__ == '__main__':
    np.random.seed(100)
    m = 1000
    x = 2*np.random.rand(m, 1)
    y = 4+3*x+np.random.randn(m, 1)
    print(x.shape, y.shape)
    x = x.reshape(-1, 1)
    y = y.ravel()

    N_epochs = 50
    batch_size = 1
    learning_rate = 'constant'#'optimal'
    eta0 = 0.1
    t0 = 1.0
    t1 = 100.0
    lmb = 0.0

    print('STOCHASTIC GRADIENT DESCENT')
    test_SGD(x, y)

    print('NEURAL NETWORK')
    test_FFNN(x, y)
