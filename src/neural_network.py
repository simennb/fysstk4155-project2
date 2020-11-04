import numpy as np
import sys
import activation_functions as act_fun


# TODO 03/11: Could remove some redundancy with taking in X_data / y_data in init and self.train
# TODO: Could maybe make the train parameter optional?
# TODO: CHECK WHAT KERAS DOES
class NeuralNetwork:
    """MEOW MEOW MEOW MEOW MEOW
    CAT CAT CAT CAT CAT

    Parameters
    ----------
    X_data: np.array
    """
    def __init__(self, X_data, y_data, epochs, batch_size, eta, lmb, t0=1.0, t1=10.0):
        np.random.seed(4155)

        self._X_data = X_data
        self._y_data = y_data
        self._epochs = epochs
        self._batch_size = batch_size
        self._lmb = lmb
        self._n_minibatch = len(self._y_data / self._batch_size)
        self._X_batch = None
        self._y_batch = None

        # Learning rate parameters
        self._eta = eta
        self._t0 = t0
        self._t1 = t1

        # lists etc for the neural network
        self._n_layers = 0
        self._n_neurons = []
        self._weights = []
        self._bias = []
        self._a = []
        self._z = []
        self._output = None
        self._regularization = []  # TODO: figure out
        self._activation = []
        self._d_activation = []
        self._init_activation_functions()

        # Create the input layer
        self._n_inputs = self._X_data.shape[0]
        self._n_features = self._X_data.shape[1]
        self.add_layer(self._n_features, 'identity')

    def add_layer(self, n_neurons, activation='identity', regularization=None, bias_init=0.01, weight_init='Random'):
        self._n_neurons.append(n_neurons)
        layer_index = len(self._n_neurons) - 1
        if len(self._weights) != 0:
            self._weights.append(np.random.randn(self._n_neurons[layer_index - 1], n_neurons))
            self._bias.append(np.zeros(n_neurons) + bias_init)
        else:
            self._weights.append(None)
            self._bias.append(None)

        # For easier indexing
        self._a.append(None)
        self._z.append(None)

        # Add activation function and derivative to lists
        self._add_activation_function(activation)

        self._n_layers += 1

    def _feed_forward(self):
#        print('_feed_forward')
        for i in range(1, self._n_layers):
            self._z[i] = np.matmul(self._a[i-1], self._weights[i]) + self._bias[i]
 #           self._z[i] = self._a[i-1] @ self._weights[i] + self._bias[i]  # identical hmm

            self._a[i] = self._activation[i](self._z[i])
#            print('layer=', i, self._a[i].shape)
        self._output = self._a[-1]  # TODO: ...yes?

    def _back_propagation(self):
#        activations = [X] + [None] * (len(layer_units) - 1)
#        deltas = [None] * (len(activations) - 1)
        error = [None] * self._n_layers  #list(range(self._n_layers))  # easier to do it this way
        weights_gradient = [None] * self._n_layers  #list(range(self._n_layers))
        bias_gradient = [None] * self._n_layers  #list(range(self._n_layers))
        for i in range(self._n_layers - 1, 0, -1):
            if i == (self._n_layers - 1):
                error[i] = self._a[i] - self._y_batch  # TODO: what?
#                error[i] = self._output - self._y_batch  # TODO: what?
            else:
                error[i] = np.matmul(error[i+1], self._weights[i+1].T) * self._d_activation[i](self._z[i])
#            print(i, self._a[i].shape, error[i].shape, self._y_batch.shape)
#            print('aa', self._a[i], 'bb', error[i], 'cc', self._y_batch)

            weights_gradient[i] = np.matmul(self._a[i-1].T, error[i]) / self._batch_size  # seems to work
            bias_gradient[i] = np.sum(error[i], axis=0) / self._batch_size

            if self._lmb > 0.0:
                weights_gradient[i] += self._lmb * self._weights[i]

        # To avoid updating the weights for the layers before all the gradients are calculated
        for i in range(self._n_layers - 1, 0, -1):
            self._weights[i] -= self._eta * weights_gradient[i]
            self._bias[i] -= self._eta * bias_gradient[i]

    def fit(self, X=None, y=None):
        # To make behavior more like SKL MLPRegressor/MLPClassifier
        if X is not None:
            self._X_data = X
        if y is not None:
            self._y_data = y

        # TODO: Wait, im confused
        # TODO: Do we do minibatches like in SGD or draw with replacement as done in the Lecture neural network?
        # Divide into mini batches and do SGD
        data_indices = np.arange(self._n_inputs)
        for i in range(self._epochs):
            for j in range(self._n_minibatch):
                # pick datapoints with replacement
                batch_indices = np.random.choice(data_indices, size=self._batch_size, replace=False)

                self._X_batch = self._X_data[batch_indices]
                self._y_batch = self._y_data[batch_indices].reshape(-1, 1)  # TODO: check resample, shapes are weird
                # TODO: yeah, reshaping changes shit when different batch sizes than 1
                self._a[0] = self._X_batch  # kinda superfluous to have both this and X_batch

                self._feed_forward()
                self._back_propagation()

    def predict(self, X):
        self._a[0] = X
        self._feed_forward()
        return self._output

    def _learning_schedule(self, t):
        return self._t0 / (t + self._t1)

    def _init_activation_functions(self):
        self._act_fun_dict = {
            'logistic': [act_fun.sigmoid, act_fun.d_sigmoid],
            'relu': [act_fun.relu, act_fun.d_relu],
            'leaky relu': [act_fun.leaky_relu, act_fun.d_leaky_relu],
            'tanh': [act_fun.tanh, act_fun.d_tanh],
            'softmax': [act_fun.softmax, act_fun.d_softmax],
            'identity': [act_fun.identity, act_fun.d_identity]
        }

    def _add_activation_function(self, activation):
        try:
            self._activation.append(self._act_fun_dict[activation][0])
            self._d_activation.append(self._act_fun_dict[activation][1])
        except KeyError:
            sys.exit('Activation function not found. Exiting.')


###########################################################
# From lecture notes week 41, slide 21
# temporary for easy comparison
class LectureNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_neurons=50,
            n_categories=10,
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.0):

        np.random.seed(4155)

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self):
        # feed-forward for training
        self.z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = act_fun.sigmoid(self.z_h)

        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias

#        exp_term = np.exp(self.z_o)
        self.probabilities = self.z_o#exp_term/ np.sum(exp_term, axis=1, keepdims=True)

    def feed_forward_out(self, X):
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = act_fun.sigmoid(z_h)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias

        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return z_o#probabilities

    def backpropagation(self):
        error_output = self.probabilities - self.Y_data
        error_hidden = np.matmul(error_output, self.output_weights.T) * act_fun.d_sigmoid(self.z_h)#* self.a_h * (1 - self.a_h)

        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self, X=None, y=None):
        data_indices = np.arange(self.n_inputs)
#        np.random.seed(4155)
        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints].reshape(-1, 1)

#                print(self.X_data.shape, self.Y_data.shape)

                self.feed_forward()
#                print(self.z_h, self.z_o)
#                print(self.a_h, self.probabilities)
                self.backpropagation()
#                sys.exit(1)

                if i == -1 and j % 50 == 0:
                    print('i=%d, j=%d' % (i, j))
                    print(chosen_datapoints)
                    print(self.probabilities)


if __name__ == '__main__':
    print('meow')
