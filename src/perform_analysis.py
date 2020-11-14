from lib import functions as fun, neural_network as nn, resampling_methods as res, sgd as sgd
import numpy as np
import time
import os


class PerformAnalysis:
    """
    Grid search function for both SGD and FFNN, as an attempt
    to make it easier to perform the required analysis.

    Parameters
    ----------
    mode: str, which type of problem we are dealing with ('regression' or 'classification')
    method: str, method used for analysis ('sgd' or 'neuralnet')
    learning_rate: str, learning rate alternatives ('constant' or 'optimal')
    dir_path: str, base directory path for saving results
    filename: str, filename to add to results folder and filenames
    CV: bool, whether or not cross-validation is used, default=True
    K: int, number of folds for CV, default=5
    t0: int, for 'optimal' learning rate, default=1
    t1: int, for 'optimal' learning rate, default=10
    """
    def __init__(self, mode, method, learning_rate, dir_path, filename, CV=True, K=5, t0=1, t1=10):
        self._mode = mode
        self._method = method
        self._learning_rate = learning_rate
        self._dir_path = dir_path
        self._filename = filename
        self._CV = CV
        self._K = K
        self._t0 = t0
        self._t1 = t1

        if self._mode == 'regression':
            self._stat = [fun.mean_squared_error, fun.calculate_R2]
            self._stat_name = ['MSE', 'R2']
        elif self._mode == 'classification':
            self._stat = [fun.accuracy]
            self._stat_name = ['Accuracy']

        self.analysed = False

    def set_data(self, X, y, test_size=0.2, scale=[True, False]):
        """
        Setting up the data to be used for analysis.
        Scales (and splits if not using CV) the design matrix.
        """
        self._X = X
        self._y = y

        if self._CV:
            pass
        else:
            self._X_train, self._X_test, self._y_train, self._y_test = fun.split_data(X, self._y, test_size=test_size)
            self._X_train = fun.scale_X(self._X_train, scale)
            self._X_test = fun.scale_X(self._X_test, scale)
        self._X = fun.scale_X(self._X, scale)

        self._n_inputs = self._X.shape[0]  # not used anywhere
        self._n_labels = 1
        if self._mode == 'classification':
            self._n_labels = self._y.shape[1]

    def set_hyperparameters(self, n_epochs, batch_size, eta0, lambdas,
                            n_h_neurons=[None], n_h_layers=[None]):
        """
        Sets the hyperparameters to be used in analysis.

        Parameters
        ----------
        n_epochs: list or array, number of epochs to perform SGD over
        batch_size: list or array, size of each minibatch
        eta0: list or array, starting learning rate
        lambdas: list or array, size of L2 regularization
        n_h_neurons: list or array, number of neurons in the hidden layers, default=None since not used for plain SGD
        n_h_layers: list or array, number of hidden layers, default=None since not used for plain SGD
        """
        # TODO: make sure inputs are lists or array
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._eta0 = eta0
        self._lambdas = lambdas
        self._n_h_neurons = n_h_neurons
        self._n_h_layers = n_h_layers
        self._n_combinations = len(n_epochs) * len(batch_size) * len(eta0) * \
                               len(lambdas) * len(n_h_neurons) * len(n_h_layers)

    def set_neural_net_params(self, activation_hidden, activation_output, wb_init):
        """
        Sets the non-hyperparameter parameters for the feed-forward neural network

        Parameters
        ----------
        activation_hidden: string, name of activation function for hidden layers,
            can be ('logistic', 'relu', 'leaky relu', 'tanh', 'identity', 'softmax')
        activation_output: string, name of activation function for output layer,
            must be 'identity' if mode is regression, 'logistic' or 'softmax' if mode is classification
        wb_init: string, determines how weights and bias is initialized ('random' or 'glorot')
        """
        self._act_hidden = activation_hidden
        self._act_output = activation_output
        self._wb_init = wb_init

        # Cost function
        if self._mode == 'regression':
            self._cf = 'MSE'
        elif self._mode == 'classification':
            self._cf = 'CE'

    def run(self):
        # Performs analysis with the specified method
        t_start = time.time()
        if self._method == 'sgd':
            score, best_index, best_params, loss_curve_best = self._run_sgd()
        elif self._method == 'neuralnet':
            score, best_index, best_params, loss_curve_best = self._run_nn()
        t_end = time.time()

        # Save to file
        self.save_to_file(score, best_index, best_params, loss_curve_best)
        self.analysed = True

        # Print optimal parameters and test results
        print('#######################')
        print('Optimal hyperparameters')
        print('n_epochs = %d' % best_params[0])
        print('batch_size = %d' % best_params[1])
        print('eta0 = %.3e' % best_params[2])
        print('lambda = %.3e' % best_params[3])
        if self._method == 'neuralnet':
            print('n_h_neurons = %d' % best_params[4])
            print('n_h_layers = %d' % best_params[5])
        # Train test
        print('#######################')
        for o in range(len(self._stat)):
            print('%s:' % self._stat_name[o])
            print('train = %.3f  ,  test = %.3f' % (score[tuple(best_index + [o, 0])],
                                                    score[tuple(best_index + [o, 1])]))

        # Printing time usage
        t_tot = t_end - t_start
        t_h, t_m, t_s = fun.time2hms(t_tot)  # convert to hours/minutes/seconds
        print('#######################')
        print('Analysis complete')
        print('Time elapsed: %3d hours, %2d minutes, %2d seconds' % (t_h, t_m, t_s))

    def _run_sgd(self):
        """
        Performs stochastic gradient descent (SGD) for all the hyperparameter combinations,
        computing train and test scores for the supplied statistics.
        """
        # In order to simplify some expressions
        n_epochs = self._n_epochs
        batch_size = self._batch_size
        eta0 = self._eta0
        lambdas = self._lambdas

        # Creating arrays for results
        score = np.zeros((len(n_epochs), len(batch_size), len(eta0),
                          len(lambdas), len(self._stat), 2))  # Last two are amount of statistics + train/test

        loss_best = -1e10
        loss_curve_best = None  # not implemented for SGD, sadly :(

        iteration = 0
        percent_index = 0
        # Searching through all hyperparameter combinations
        for i in range(len(n_epochs)):
            for j in range(len(batch_size)):
                for k in range(len(eta0)):
                    for l in range(len(lambdas)):
                        # A "mostly" functional way to tell how far along the analysis is
                        if iteration % np.ceil(0.1 * self._n_combinations) == 0:
                            print('%3d percent complete. Combination %5d of %d.' % ((10 * percent_index),
                                                                                    iteration,
                                                                                    self._n_combinations))
                            percent_index += 1
                        iteration += 1

                        error_train = [1e10] * len(self._stat)
                        error_test = [1e10] * len(self._stat)

                        # Sets up the SGD model
                        if self._mode == 'regression':
                            sgd_obj = sgd.LinRegSGD(n_epochs[i], batch_size[j], eta0[k], self._learning_rate)
                        elif self._mode == 'classification':
                            sgd_obj = sgd.LogRegSGD(n_epochs[i], batch_size[j], self._n_labels,
                                                    eta0=eta0[k], learning_rate=self._learning_rate)
                        sgd_obj.set_lambda(lambdas[l])
                        sgd_obj.set_step_length(self._t0, self._t1)

                        # Perform the fit and prediction with either CV or not
                        if self._CV:
                            CV = res.CrossValidation(self._X, self._y, sgd_obj, stat=self._stat)
                            error_train, error_test = CV.compute(K=self._K)

                        else:
                            sgd_obj.fit(self._X_train, self._y_train)
                            y_fit = sgd_obj.predict(self._X_train)
                            y_pred = sgd_obj.predict(self._X_test)

                            for o in range(len(self._stat)):
                                error_train = self._stat[o](self._y_train, y_fit)
                                error_test = self._stat[o](self._y_test, y_pred)

                        # Adding train and test error for all statistic functions
                        for o in range(len(self._stat)):
                            score[i, j, k, l, o, 0] = error_train[o]
                            score[i, j, k, l, o, 1] = error_test[o]

                        # Checking if test error is lower than the current best fit
                        # Using R2 and Accuracy since both have score=1 as their goal
                        if self._mode == 'regression':
                            loss_current = score[i, j, k, l, 1, 1]  # R2 score
                        elif self._mode == 'classification':
                            loss_current = score[i, j, k, l, 0, 1]  # accuracy score

                        if loss_current > loss_best:
                            loss_best = loss_current
                            best_index = [i, j, k, l]
                            best_params = [n_epochs[i], batch_size[j], eta0[k], lambdas[l]]
                            loss_curve_best = np.array([0])  # To make sure things don't crash

        return score, best_index, best_params, loss_curve_best

    def _run_nn(self):
        # In order to simplify some expressions
        n_epochs = self._n_epochs
        batch_size = self._batch_size
        eta0 = self._eta0
        lambdas = self._lambdas
        n_h_neurons = self._n_h_neurons
        n_h_layers = self._n_h_layers

        # Creating arrays for results
        score = np.zeros((len(n_epochs), len(batch_size), len(eta0),
                          len(lambdas), len(n_h_neurons), len(n_h_layers),  # ))
                          len(self._stat), 2))  # Last two are amount of stats + train/test

        loss_best = -1e10
        loss_curve_best = None

        iteration = 0
        percent_index = 0
        # Searching through all hyperparameter combinations
        for i in range(len(n_epochs)):
            for j in range(len(batch_size)):
                for k in range(len(eta0)):
                    for l in range(len(lambdas)):
                        for m in range(len(n_h_neurons)):
                            for n in range(len(n_h_layers)):
                                # A "mostly" functional way to tell how far along the analysis is
                                if iteration % np.ceil(0.1 * self._n_combinations) == 0:
                                    print('%3d percent complete. Combination %5d of %d.' % ((10*percent_index),
                                                                                            iteration,
                                                                                            self._n_combinations))
                                    percent_index += 1
                                iteration += 1

                                error_train = [1e10] * len(self._stat)
                                error_test = [1e10] * len(self._stat)

                                # Creating lists of the make-up of the current neural network
                                neuron_layers = [n_h_neurons[m]] * n_h_layers[n] + [self._n_labels]
                                act_func_layers = [self._act_hidden] * n_h_layers[n] + [self._act_output]

                                # Create feed-forward neural net at current hyper-parameter combination
                                neural_net = nn.NeuralNetwork(self._X, self._y, epochs=n_epochs[i],
                                                              batch_size=batch_size[j], eta=eta0[k], lmb=lambdas[l],
                                                              cost_function=self._cf, learning_rate=self._learning_rate,
                                                              t0=self._t0, t1=self._t1, gradient_scaling=1,
                                                              wb_init=self._wb_init)
                                for layer_index in range(len(neuron_layers)):
                                    neural_net.add_layer(neuron_layers[layer_index], act_func_layers[layer_index])

                                # Perform the fit and prediction with either CV or not
                                if self._CV:
                                    CV = res.CrossValidation(self._X, self._y, neural_net, stat=self._stat)
                                    error_train, error_test = CV.compute(K=self._K)

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
                                # Using R2 and Accuracy since both have score=1 as their goal
                                if self._mode == 'regression':
                                    loss_current = score[i, j, k, l, m, n, 1, 1]  # R2 score
                                elif self._mode == 'classification':
                                    loss_current = score[i, j, k, l, m, n, 0, 1]  # accuracy score

                                if loss_current > loss_best:
                                    loss_best = loss_current
                                    best_index = [i, j, k, l, m, n]
                                    best_params = [n_epochs[i], batch_size[j], eta0[k],
                                                   lambdas[l], n_h_neurons[m], n_h_layers[n]]

                                    loss_curve_best = np.array(neural_net._loss)
                                    if self._CV:
                                        loss_curve_best = loss_curve_best.reshape(n_epochs[i], self._K, order='F')

        return score, best_index, best_params, loss_curve_best

    def save_to_file(self, score, best_index, best_params, loss_curve_best):
        # Makes sure directory exists
        if not os.path.exists(self._dir_path):
            os.mkdir(self._dir_path)

        filename = self._dir_path + self._filename

        # Saving arrays to binary .npy files
        np.save(filename + '_score', score)
        np.save(filename + '_best_index', best_index)
        if self._method == 'neuralnet':  # not sure why, but best_params gives weird problems with sgd
            np.save(filename + '_best_params', best_params)
        np.save(filename + '_loss_curve_best', loss_curve_best)

        # Save parameters
        np.save(filename + '_n_epochs', np.array(self._n_epochs, dtype=int))
        np.save(filename + '_batch_size', np.array(self._batch_size, dtype=int))
        np.save(filename + '_eta0', np.array(self._eta0))
        np.save(filename + '_lambdas', np.array(self._lambdas))
        if self._method == 'neuralnet':
            np.save(filename + '_n_h_neurons', np.array(self._n_h_neurons, dtype=int))
            np.save(filename + '_n_h_layers', np.array(self._n_h_layers, dtype=int))

        print('Results saved to directory: %s' % self._dir_path)
