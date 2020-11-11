import numpy as np
import sys
from lib import neuralnet_functions as nn_fun


# TODO 03/11: Could remove some redundancy with taking in X_data / y_data in init and self.train
# TODO: Could maybe make the train parameter optional?
class NeuralNetwork:
    """Multilayer Perceptron Model
    MEOW MEOW MEOW MEOW MEOW
    CAT CAT CAT CAT CAT

    Parameters
    ----------
    X_data: np.array
    y_data:
    epochs:
    batch_size:
    eta:
    lmb:
    cost_function: string, determines which cost function to use ('MSE' or 'CE')
    """
    def __init__(self, X_data, y_data, epochs, batch_size, eta, lmb, cost_function,
                 learning_rate='constant', t0=1.0, t1=10.0, gradient_scaling=0,
                 wb_init='random', bias_init=0.01):
#        np.random.seed(4155)

        self._X_data = X_data
        self._y_data = y_data
        self._epochs = epochs
        self._batch_size = batch_size
        self._lmb = lmb
        self._n_minibatch = len(self._y_data / self._batch_size)
        self._X_batch = None
        self._y_batch = None

        # Learning rate parameters
        self._learning_rate = learning_rate
        self._eta0 = eta
        self._eta = eta
        self._t0 = t0
        self._t1 = t1

        # Weight / bias initialization parameters
        self._wb_init = wb_init
        self._bias_init = bias_init
        # TODO: maybe add weight init?

        # Some of the variables necessary for feedforward/backprop
        self._n_layers = 0
        self._n_neurons = []
        self._weights = []
        self._bias = []
        self._a = []
        self._z = []
        self._output = None
        self._loss = []
        self._regularization = []  # TODO: figure out
        self._activation = []
        self._d_activation = []
        self._init_activation_functions()
        self._initialized = False
        self._weights_start = None
        self._bias_start = None

        # Set cost function
        self._cost_function = None
        self._set_cost_function(cost_function)

        # Create the input layer
        self._n_inputs = self._X_data.shape[0]
        self._n_features = self._X_data.shape[1]
        self.add_layer(self._n_features, 'identity')

        # Gradient scale factor
        self._set_gradient_scale(gradient_scaling)

    def add_layer(self, n_neurons, activation='identity', regularization=None):
        self._n_neurons.append(n_neurons)

        # Weights and bias, initialized later
        self._weights.append(None)
        self._bias.append(None)

        # For easier indexing
        self._a.append(None)
        self._z.append(None)

        # Add activation function and derivative to lists
        self._set_activation_function(activation)

        self._n_layers += 1

    def initialize_weights_bias(self, wb_init=None, bias_init=None):
        if wb_init is not None:
            self._wb_init = wb_init
        if bias_init is not None:
            self._bias_init = bias_init

        if not self._initialized:
            n_neurons = self._n_neurons

            # TODO: [0.02711087 0.59226773][0.02708315 0.58659946] WITHOUT
            # TODO: [0.02290557 0.65484501][0.02377248 0.63792649] WITH
            # TODO: So definitely a slight difference, and keeping them identical
            # TODO: makes the results more in line with the results from first run
            # Doing so that each fold (with CV) uses same initialization of weights and bias
            # So that the only thing that varies between them is the data set.
            # One could argue that doing so for all hyperparameter combinations would also
            # be TODO
            self._weights_start = [None] * self._n_layers
            self._bias_start = [None] * self._n_layers
            if self._wb_init == 'random':
                for i in range(1, self._n_layers):
                    self._weights_start[i] = np.random.randn(n_neurons[i - 1], n_neurons[i])
                    self._bias_start[i] = np.zeros(n_neurons[i]) + self._bias_init

            elif self._wb_init == 'glorot':
                # TODO: this is kinda weird, MLP has different indexing for things
                for i in range(1, self._n_layers):
                    factor = np.sqrt(6 / (n_neurons[i] + n_neurons[i - 1]))  # sqrt(2) for classification??????????
                    self._weights_start[i] = np.random.uniform(-factor, factor, (n_neurons[i - 1], n_neurons[i]))
                    self._bias_start[i] = np.random.uniform(-factor, factor, n_neurons[i])

        for i in range(1, self._n_layers):
            self._weights[i] = self._weights_start[i].copy()
            self._bias[i] = self._bias_start[i].copy()
        self._initialized = True

    def _feed_forward(self):
        for i in range(1, self._n_layers):
            self._z[i] = np.matmul(self._a[i-1], self._weights[i]) + self._bias[i]
            self._a[i] = self._activation[i](self._z[i])

        self._output = self._a[-1]  # TODO: ...yes?

    def _back_propagation(self):
        error = [None] * self._n_layers
        weights_gradient = [None] * self._n_layers
        bias_gradient = [None] * self._n_layers
        for i in range(self._n_layers - 1, 0, -1):
            if i == (self._n_layers - 1):
                # For output activation / cost function combinations of identity/MSE and softmax/CE this is correct.
                error[i] = self._a[i] - self._y_batch
            else:
                error[i] = np.matmul(error[i+1], self._weights[i+1].T) * self._d_activation[i](self._z[i])

            # Compute the gradients
            weights_gradient[i] = np.matmul(self._a[i-1].T, error[i])
            bias_gradient[i] = np.sum(error[i], axis=0)

            # L2 Regularization
            if self._lmb > 0.0:
                weights_gradient[i] += self._lmb * self._weights[i]

            # Scale the gradients
            weights_gradient[i] /= self._gradient_scale
            bias_gradient[i] /= self._gradient_scale
            # TODO: Do some testing

        # To avoid updating the weights and bias before all the gradients are calculated
        for i in range(self._n_layers - 1, 0, -1):
            self._weights[i] -= self._eta * weights_gradient[i]
            self._bias[i] -= self._eta * bias_gradient[i]

    def fit(self, X=None, y=None):
        # To make behavior more like SKL MLPRegressor/MLPClassifier
        if X is not None:
            self._X_data = X
            self._n_inputs = self._X_data.shape[0]
        if y is not None:
            self._y_data = y

        # Initialize weights and bias unless already done
#        if not self._initialized:
        self.initialize_weights_bias()

        # TODO: Wait, im confused
        # TODO: Do we do minibatches like in SGD or draw with replacement as done in the Lecture neural network?
        # Divide into mini batches and do SGD
        data_indices = np.arange(self._n_inputs)
        for i in range(self._epochs):
#            print('epoch = ', i)
            accumulated_loss = 0  # trying to do this as SKL does
            for j in range(self._n_minibatch):
                # pick datapoints with replacement
                batch_indices = np.random.choice(data_indices, size=self._batch_size, replace=False)

                self._X_batch = self._X_data[batch_indices]
                self._y_batch = (self._y_data[batch_indices])
                self._a[0] = self._X_batch#.copy()  # kinda superfluous to have both this and X_batch

#                print(i, j)
                self._feed_forward()
                accumulated_loss += self._compute_loss() * self._batch_size  # Batch loss
                if self._learning_rate == 'optimal':
                    self._eta = self._learning_schedule(self._epochs*self._n_minibatch + i)
                self._back_propagation()
#                self._loss.append(self._[0])  # to see how loss function goes over time
            self._loss.append(accumulated_loss / self._n_inputs)

    def predict(self, X):
        self._a[0] = X#.copy()
        self._feed_forward()
        return self._output

#    def score(self, X_test, y_test):
#        y_pred = self.predict(X_test)

    def _learning_schedule(self, t):
        return self._t0 / (t + self._t1)

    def _compute_loss(self):
        loss = self._cost_function(self._y_batch, self._a[-1])

        # L2 Regularization
        values = np.sum(np.array([np.dot(s.ravel(), s.ravel()) for s in self._weights[1:]]))
        loss += (0.5 * self._lmb) * values / self._batch_size

        return loss[0]

    def _set_data(self, X, y):
        pass

    def _set_gradient_scale(self, gradient_scaling):
        # TODO: Might be beneficial for testing purposes
        if gradient_scaling == 0:
            self._gradient_scale = self._batch_size * self._n_inputs
        elif gradient_scaling == 1:
            self._gradient_scale = self._batch_size  # i think this is the most similar to how SKL does it
        elif gradient_scaling == 2:
            self._gradient_scale = self._n_inputs
        else:
            self._gradient_scale = 1.0

    def _init_activation_functions(self):
        self._act_fun_dict = {
            'logistic': [nn_fun.sigmoid, nn_fun.d_sigmoid],
            'relu': [nn_fun.relu, nn_fun.d_relu],
            'leaky relu': [nn_fun.leaky_relu, nn_fun.d_leaky_relu],
            'tanh': [nn_fun.tanh, nn_fun.d_tanh],
            'softmax': [nn_fun.softmax, nn_fun.d_softmax],
            'identity': [nn_fun.identity, nn_fun.d_identity]
        }

    def _set_cost_function(self, cost_function):
        if cost_function == 'MSE':
            self._cost_function = nn_fun.cost_MSE
        elif cost_function == 'CE':
            self._cost_function = nn_fun.cost_CrossEntropy

    def _set_activation_function(self, activation):
        try:
            self._activation.append(self._act_fun_dict[activation][0])
            self._d_activation.append(self._act_fun_dict[activation][1])
        except KeyError:
            sys.exit('Activation function not found. Exiting.')


if __name__ == '__main__':
    print('meow')
