import numpy as np
import activation_functions as act_fun


# TODO: 30/10, dimension mismatch in weights_gradient (at least, probably everywhere)
class NeuralNetwork:
    """MEOW MEOW MEOW MEOW MEOW
    CAT CAT CAT CAT CAT

    Parameters
    ----------
    X_data: np.array
    """
    def __init__(self, X_data, y_data, epochs, batch_size, eta, lmb, t0=1.0, t1=10.0):
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
        self._probabilities = None
        self._regularization = []  # TODO: figure out
        self._activation = []
        self._d_activation = []

        # Create the input layer
        self._n_inputs = self._X_data.shape[0]
        self._n_features = self._X_data.shape[1]
        self.add_layer(self._n_features, 'identity')

    def add_layer(self, n_neurons, activation, regularization=None, bias_init=0.01, weight_init='Random'):
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
        for i in range(1, self._n_layers):
            self._z[i] = np.matmul(self._a[i-1], self._weights[i]) + self._bias[i]
            self._a[i] = self._activation[i](self._z[i])

        self._probabilities = self._a[-1]  # TODO: ...yes?

    def _back_propagation(self):
        error = list(range(self._n_layers))  # easier to do it this way
        weights_gradient = list(range(self._n_layers))
        bias_gradient = list(range(self._n_layers))

        for i in range(self._n_layers - 1, 0, -1):
            if i == (self._n_layers - 1):
                # TODO: huh
                error[i] = self._probabilities - self._y_batch
                print(self._probabilities.shape)
                print(error[i].shape)
                print(self._y_batch.shape)
                print('meow')
            else:
                error[i] = np.matmul(error[i+1], self._weights[i+1].T) * self._d_activation[i](self._z[i])
                print('nyaa')

            print(self._a[i-1].shape)
            # TODO: check and compare dimensions with the Lecture notes NN code
            # TODO: cause atm there is something weird here
            # TODO: oh well

            weights_gradient[i] = np.matmul(self._a[i-1], error[i])
            bias_gradient[i] = np.sum(error[i], axis=0)

            if self._lmb > 0.0:
                weights_gradient[i] += self._lmb * self._weights[i]

            self._weights[i] -= self._eta * weights_gradient
            self._bias[i] -= self._eta * bias_gradient

    def fit(self):
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
                self._a[0] = self._X_batch  # kinda superfluous to have both this and X_batch

                self._feed_forward()
                self._back_propagation()

    def predict(self, X):
        self._a[0] = X
        self._feed_forward()
        return self._probabilities

    def _learning_schedule(self, t):
        return self._t0 / (t + self._t1)

    def _add_activation_function(self, activation):
        if activation == 'sigmoid':
            self._activation.append(act_fun.sigmoid)
            self._d_activation.append(act_fun.d_sigmoid)
        elif activation == 'relu':
            self._activation.append(act_fun.relu)
            self._d_activation.append(act_fun.d_relu)
        elif activation == 'leaky relu':
            self._activation.append(act_fun.leaky_relu)
            self._d_activation.append(act_fun.d_leaky_relu)
        elif activation == 'tanh':
            self._activation.append(act_fun.tanh)
            self._d_activation.append(act_fun.d_tanh)
        else:
            # if no activation function, not sure if this works
            self._activation.append(lambda z: z)
            self._d_activation.append(lambda z: 1)


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
        self.a_h = sigmoid(self.z_h)

        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias

        exp_term = np.exp(self.z_o)
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def feed_forward_out(self, X):
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = sigmoid(z_h)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias

        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities

    def backpropagation(self):
        error_output = self.probabilities - self.Y_data
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)

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

    def train(self):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()