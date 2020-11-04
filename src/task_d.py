import functions as fun
import regression_methods as reg
import resampling_methods as res
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
import sys
import neural_network as nn
from sklearn.neural_network import MLPClassifier

# Check if keras is better, since it has softmax. Not relevant for b, but maybe later
neural_net_SKL = MLPClassifier(hidden_layer_sizes=(neuron_layers[0]), activation='logistic', solver='sgd',
                               alpha=lmb, batch_size=batch_size, learning_rate_init=eta0, max_iter=N_epochs)
# neural_net_SKL.out_activation_ = 'softmax'  # looked at code, but this is set after calling train for the first time
# so would need to figure out how it determines it, and that looked like pain.
#  TODO: I think it uses softmax when the output is multiclass, and logistic when 1 output class
