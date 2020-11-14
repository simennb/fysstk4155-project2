import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from lib import sgd, functions as fun, neural_network as nn
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

    ts_own = time.time()
    neural_net.fit()
    te_own = time.time()

    y_fit = neural_net.predict(x_train)
    y_pred = neural_net.predict(x_test)

    print('\nNeuralNet:')
    fun.print_MSE_R2(y_train, y_fit, 'train', 'NN')
    fun.print_MSE_R2(y_test, y_pred, 'test', 'NN')

    neural_net_SKL = MLPRegressor(hidden_layer_sizes=(N_hidden), activation='logistic', solver='sgd',
                                  alpha=lmb, batch_size=batch_size, learning_rate_init=eta0, max_iter=N_epochs,
                                  momentum=0.0, nesterovs_momentum=False)
    ts_skl = time.time()
    neural_net_SKL.fit(x_train, y_train)
    te_skl = time.time()

    y_fit = neural_net_SKL.predict(x_train)
    y_pred = neural_net_SKL.predict(x_test)

    print('\nMLPRegressor:')
    fun.print_MSE_R2(y_train, y_fit, 'train', 'NN')
    fun.print_MSE_R2(y_test, y_pred, 'test', 'NN')

    # Timing
    t_skl = te_skl - ts_skl
    t_own = te_own - ts_own
    print('\nTime SKL: %.3e s' % t_skl)
    print('Time own: %.3e s' % t_own)
    try:
        print('Factor own/skl = %.3e' % (t_own/t_skl))
    except ZeroDivisionError:
        print('t_skl = 0, t_own = ', t_own)


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

    print('\n\n##############################\n\n')

    print('NEURAL NETWORK')
    test_FFNN(x, y)
