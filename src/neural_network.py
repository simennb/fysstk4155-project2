import numpy as np
import activation_functions as act_fun


class NeuralNetwork:
    def __init__(self, X_data, y_data, epochs, batch_size, eta, lmb):
        self._X_data = X_data
        self._y_data = y_data
        self._epochs = epochs
        self._batch_size = batch_size
        self._eta = eta
        self._lmb = lmb

        # lists etc for the neural network
        self._n_layers = 0
        self._n_neurons = []
        self._weights = []
        self._bias = []
        self._a = []
        self._probabilities = None
        self._regularization = []
        self._activation = []

    def add_layer(self, n_neurons, activation, regularization, bias_init=0.01, weight_init='Random'):
        self._n_neurons.append(n_neurons)
        layer_index = len(self._n_neurons) - 1
        if len(self._weights) != 0:
            self._weights.append(np.random.randn(self._n_neurons[layer_index - 1], n_neurons))
            self._bias.append(np.zeros(n_neurons) + bias_init)
        else:
            self._weights.append(None)
            self._bias.append(None)

        # TODO: Make activation into a function to make adjustments independent of this
#        self._activation.append(self._get_activation_function(activation))
        if activation == 'sigmoid':
            self._activation.append(act_fun.sigmoid)
        elif activation == 'relu':
            self._activation.append(act_fun.relu)

#        self._error.append(None)
        self._n_layers += 1

    def _feed_forward(self):
        for i in range(1, self._n_layers):
            z = np.matmul(self.a[i-1], self._weights[i]) + self._bias[i]
            self.a[i] = self._activation[i](z)

        self._probabilities = self.a[i]  # something

        # feed-forward for training
#        self.z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
#        self.a_h = sigmoid(self.z_h)

#        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias

#        exp_term = np.exp(self.z_o)
#        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def _back_propagation(self):
        # TODO: Check if there is a smarter way to do this
        # lst = [None] * 10
        error = list(range(self._n_layers))  # easier to do it this way
        weights_gradient = list(range(self._n_layers))
        bias_gradient = list(range(self._n_layers))

        # TODO: see if i can put everything into the loop, IT SHOULD BE POSSIBLE
#        error[-1] = self._probabilities - self.Y_data
#        weights_gradient[-1] = np.matmul(self.a[-2].T, error[-1])
#        bias_gradient[-1] = np.sum(error[-1], axis=0)

        for i in range(self._n_layers, 0):
            if i == (self._n_layers - 1):
                error[i] = self._probabilities - self.Y_data
            else:
                error[i] = np.matmul(error[i+1], self._weights[i+1].T) * self.a[i] * (1 - self.a)  # TODO: check

            weights_gradient[i] = np.matmul(self.a[i-1], error[i])
            bias_gradient[i] = np.sum(error[i], axis=0)

            if self._lmb > 0.0:
                weights_gradient[i] += self.lmb * self._weights[i]

            self._weights[i] -= self.eta * weights_gradient
            self._bias[i] -= self.eta * bias_gradient

    def feed_forward_output(self):
        self._feed_forward()
        return self._probabilities

    def train(self):
        # Divide into mini batches and do SGD
        self._feed_forward()
        self._back_propagation()

    @staticmethod
    def _get_activation_function(activation):
        if activation == 'sigmoid':
            function = act_fun.sigmoid
        elif activation == 'relu':
            function = act_fun.relu

        return function


# From lecture notes week 41, slide 21
class NeuralNetwork:
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