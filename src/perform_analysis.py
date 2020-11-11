from lib import functions as fun, neural_network as nn, resampling_methods as res
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
import time


class PerformAnalysis:
    """I dont know if this is a good idea
    This is a way of making the hyperparameter-search slightly easier
    """
    def __init__(self, mode, method, learning_rate, name, CV=True, K=5, SKL=0, t0=1, t1=5):
        self._mode = mode  # 'regression' or 'classication'
        self._method = method  # 'sgd' or 'neuralnet'
        self._learning_rate = learning_rate  # 'constant' or 'optimal' TODO: ?
        self._name = name  # identifier for the current run, to save to file
        self._CV = CV  # if to use Cross-Validation for the analysis
        self._K = K  # number of K-folds for CV
        self._SKL = SKL  # 0 = no, 1 = compare, 2 = only SKL TODO: HELP???!!?!?!??!?!?
        # TODO: or with SKL, we dont do it here, but rather manually call SKL
        # TODO: using default settings or the optimal parameters we found

        self._t0 = t0
        self._t1 = t1

        if self._mode == 'regression':
            self._stat = [fun.mean_squared_error, fun.calculate_R2]
            self._stat_name = ['MSE', 'R2']
        elif self._mode == 'classification':
            self._stat = [fun.accuracy]
            self._stat_name = ['Accuracy']

    def set_data(self, X, y, test_size=0.2, scale=[True, False]):
        # Figure out how to deal with potential CV
        # TODO: probably just give in full X/y + test_size
        self._X = X
        self._y = y

        if self._CV:
            pass
#            self._X = fun.scale_X(self._X, scale)
        else:
            self._X_train, self._X_test, self._y_train, self._y_test = fun.split_data(X, self._y, test_size=test_size)
            self._X_train = fun.scale_X(self._X_train, scale)
            self._X_test = fun.scale_X(self._X_test, scale)
        self._X = fun.scale_X(self._X, scale)

        # TODO: this wont necessarily work....
        self._n_inputs = self._X.shape[0]  # dunno, not used anywhere
        self._n_labels = self._y.shape[1]

    def set_hyperparameters(self, n_epochs, batch_sizes, eta0, lambdas,
                            n_h_neurons=None, n_h_layers=None):
        # TODO: make sure inputs are lists or array
        self._n_epochs = n_epochs
        self._batch_sizes = batch_sizes
        self._eta0 = eta0
        self._lambdas = lambdas
        self._n_h_neurons = n_h_neurons
        self._n_h_layers = n_h_layers

    def set_neural_net_params(self, activation_hidden, activation_output, wb_init):
        self._act_hidden = activation_hidden  # string
        self._act_output = activation_output  # string
        self._wb_init = wb_init  # 'random' or 'glorot'

        # Cost function
        if self._mode == 'regression':
            self._cf = 'MSE'
        elif self._mode == 'classification':
            self._cf = 'CE'

    def run(self):
        # Performs analysis with the specified method
        t_start = time.time()
        if self._method == 'sgd':
            score, best_index, loss_curve_best = self._run_sgd()
        elif self._method == 'neuralnet':
            score, best_index, loss_curve_best = self._run_nn()
        t_end = time.time()

        # TODO: save to file

        # Print optimal parameters and test results
        print('#######################')
        print('Optimal hyperparameters')
        print('n_epochs = %d' % self._n_epochs[best_index[0]])
        print('batch_size = %d' % self._batch_sizes[best_index[1]])
        print('eta0 = %.3e' % self._eta0[best_index[2]])
        print('lambda = %.3e' % self._lambdas[best_index[3]])
        if self._method == 'neuralnet':
            print('n_h_neurons = %d' % self._n_h_neurons[best_index[4]])
            print('n_h_layers = %d' % self._n_h_layers[best_index[5]])
        # Train test
        print('#######################')
        for o in range(len(self._stat)):
            print('%s:' % self._stat_name[o])
            print('train = %.3f  ,  test = %.3f' % (score[tuple(best_index + [o, 0])],
                                                    score[tuple(best_index + [o, 1])]))

        # Printing time usage
        t_tot = t_end - t_start
        t_h, t_m, t_s = fun.time2hms(t_tot)  # convert to hours/minutes/seconds
        print('#################')
        print('Analysis complete')
        print('Time elapsed: %.3d s' % t_tot)
        print('Time elapsed: %.3d hours, %.2d minutes, %.2d seconds' % (t_h, t_m, t_s))


    def _run_sgd(self):
        """
        Performs stochastic gradient descent (SGD) for all the hyperparameter combinations,
        computing train and test scores for the supplied statistics.
        """
        # In order to simplify some expressions
        n_epochs = self._n_epochs
        batch_sizes = self._batch_sizes
        eta0 = self._eta0
        lambdas = self._lambdas

        # Creating arrays for results
        score = np.zeros((len(n_epochs), len(batch_sizes), len(eta0),
                          len(lambdas),# len(n_h_neurons), len(n_h_layers),  # ))
                          len(self._stat), 2))  # TODO: last two are amount of stats + train/test

        loss_best = 1e10
        loss_curve_best = None
        # TODO: store best loss-curve for each hyperparameter maybe???
        # TODO: or just for the total best combination

        # Searching through all hyperparameter combinations
        for i in range(len(n_epochs)):
            for j in range(len(batch_sizes)):
                for k in range(len(eta0)):
                    for l in range(len(lambdas)):
                        # TODO: hmm
                        error_train = [1e10] * len(self._stat)
                        error_test = [1e10] * len(self._stat)

                        # TODO: PERFORM SGD

                        # Adding train and test error for all statistic functions
                        for o in range(len(self._stat)):
                            score[i, j, k, l, o, 0] = error_train[o][0]
                            score[i, j, k, l, o, 1] = error_test[o][1]

                        # Checking if test error is lower than the current best fit
                        loss_current = score[i, j, k, l, 0, 1]
                        if loss_current <= loss_best:
                            loss_best = loss_current
                            best_index = [i, j, k, l]

#                            loss_curve_best = np.array(neural_net._loss)
                            if self._CV:
                                loss_curve_best = loss_curve_best.reshape(n_epochs[i], self._K)

        return score, best_index, loss_curve_best

    def _run_nn(self):
        # In order to simplify some expressions
        n_epochs = self._n_epochs
        batch_sizes = self._batch_sizes
        eta0 = self._eta0
        lambdas = self._lambdas
        n_h_neurons = self._n_h_neurons
        n_h_layers = self._n_h_layers

        # Creating arrays for results
        score = np.zeros((len(n_epochs), len(batch_sizes), len(eta0),
                          len(lambdas), len(n_h_neurons), len(n_h_layers),  # ))
                          len(self._stat), 2))  # TODO: last two are amount of stats + train/test

        loss_best = 1e10
        loss_curve_best = None
        # TODO: store best loss-curve for each hyperparameter maybe???
        # TODO: or just for the total best combination

        # Searching through all hyperparameter combinations
        for i in range(len(n_epochs)):
            for j in range(len(batch_sizes)):
                for k in range(len(eta0)):
                    for l in range(len(lambdas)):
                        for m in range(len(n_h_neurons)):
                            for n in range(len(n_h_layers)):
                                # TODO: hmm
                                error_train = [1e10] * len(self._stat)
                                error_test = [1e10] * len(self._stat)

                                # Creating lists TODO s ahw da
                                neuron_layers = [n_h_neurons[m]] * n_h_layers[n] + [self._n_labels]
                                act_func_layers = [self._act_hidden] * n_h_layers[n] + [self._act_output]

                                # Create feed-forward neural net at current hyper-parameter combination
                                neural_net = nn.NeuralNetwork(self._X, self._y, epochs=n_epochs[i],
                                                              batch_size=batch_sizes[j], eta=eta0[k], lmb=lambdas[l],
                                                              cost_function=self._cf, learning_rate=self._learning_rate,
                                                              t0=self._t0, t1=self._t1, gradient_scaling=1,
                                                              wb_init=self._wb_init)
                                for layer_index in range(len(neuron_layers)):
                                    neural_net.add_layer(neuron_layers[layer_index], act_func_layers[layer_index])

                                # Perform the fit and prediction with either CV or not
                                if self._CV:
                                    print('meow')
                                    CV = res.CrossValidation(self._X, self._y, neural_net, stat=self._stat)
                                    error_train, error_test = CV.compute(K=self._K)
                                    print(error_train, error_test)

                                else:
                                    neural_net.fit(self._X_train, self._y_train)
                                    y_fit = neural_net.predict(self._X_train)
                                    y_pred = neural_net.predict(self._X_test)

                                    for o in range(len(self._stat)):
                                        error_train = self._stat[o](self._y_train, y_fit)
                                        error_test = self._stat[o](self._y_test, y_pred)

                                # Adding train and test error for all statistic functions
                                for o in range(len(self._stat)):
                                    score[i, j, k, l, m, n, o, 0] = error_train[o]
                                    score[i, j, k, l, m, n, o, 1] = error_test[o]

                                # Checking if test error is lower than the current best fit
                                loss_current = score[i, j, k, l, m, n, 0, 1]
                                if loss_current <= loss_best:
                                    loss_best = loss_current
                                    best_index = [i, j, k, l, m, n]

                                    loss_curve_best = np.array(neural_net._loss)
                                    if self._CV:
                                        loss_curve_best = loss_curve_best.reshape(n_epochs[i], self._K)

        return score, best_index, loss_curve_best

    def save_to_file(self):
        pass
