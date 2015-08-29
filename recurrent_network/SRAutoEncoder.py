__author__ = 'eidonfiloi'

import numpy as np
import logging
from math import sqrt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class SRAutoEncoder(object):
    """Sparse Recurrent Autoencoder"""

    def __init__(self, config):
        self.name = config['name']
        self.num_inputs = config['num_inputs']
        self.sdr_size = config['sdr_size']
        self.activation_function = config['activation']
        self.threshold = 0.5 if self.activation_function == 'Sigmoid' else 0.0
        self.sparsity = config['sparsity']
        self.duty_cycle_decay = config['duty_cycle_decay']
        self.min_weight = config['min_weight']
        self.max_weight = config['max_weight']
        self.duty_cycle_decay = config['duty_cycle_decay']
        self.weights_lr = config['weights_lr']
        self.inhibition_lr = config['inhibition_lr']
        self.bias_lr = config['bias_lr']
        self.logger = logging.getLogger(self.__class__.__name__)

        self.initialize_hidden_state()

    def generate_node_output(self, inputs):
        sums = np.dot(self.weights, inputs.T) + self.biases

        if self.activation_function == "Sigmoid":
            self.activations = sigmoid(sums)
        elif self.activation_function == "Rectifier":
            self.activations = np.maximum(0.0, sums)
        elif self.activation_function == "Tanh":
            self.activations = np.tanh(sums)
        else:
            self.activations = np.maximum(0.0, sums)

        sdr = self.activations - np.dot(self.inhibition, self.activations)

        for i in range(0, self.activations.size):
            if sdr[i] > self.threshold:
                sdr[i] = 1.0
            else:
                sdr[i] = 0.0

        self.duty_cycles = (1.0 - self.duty_cycle_decay) * self.duty_cycles + self.duty_cycle_decay * sdr

        return sdr

    def learn(self, inputs, sdr):
        recon = self.reconstruct(sdr)

        error = inputs - recon
        mse = sqrt(np.mean(np.abs(error) ** 2, axis=0))
        if mse > error.shape[0]:
            self.weights_lr /= 10.0
            self.bias_lr /= 10.0
            self.inhibition_lr /= 10.0
            self.initialize_hidden_state()
        self.logger.info('{0}: error is {1}'.format(self.name, mse))

        if self.name == 'layer1-feedforward':
            self.logger.info('\n################# target sparsity: {0} - actual sparsity: {1}'.format(self.sparsity, np.mean(sdr)))
        for i in range(0, self.weights.shape[0]):
            lscf = self.sparsity - self.duty_cycles[i]

            learn = sdr[i]

            self.weights[i] += self.weights_lr * learn * error

            self.biases[i] += self.bias_lr * lscf

            self.inhibition[i] += self.inhibition_lr * lscf * self.activations[i] * self.activations

            self.inhibition[i] = np.maximum(0.0, self.inhibition[i])

            self.inhibition[i][i] = 0.0

        return mse

    def reconstruct(self, sdr):
        return np.dot(self.weights.T, sdr.T)

    def initialize_hidden_state(self):
        self.weights = np.random.rand(self.sdr_size, self.num_inputs) * (self.max_weight - self.min_weight) + self.min_weight
        self.biases = np.random.rand(self.sdr_size) * (self.max_weight - self.min_weight) + self.min_weight
        self.activations = np.zeros(self.sdr_size)
        self.inhibition = np.zeros((self.sdr_size, self.sdr_size))
        self.duty_cycles = np.zeros(self.sdr_size)
