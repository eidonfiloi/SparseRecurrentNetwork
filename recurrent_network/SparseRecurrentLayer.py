__author__ = 'eidonfiloi'

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class SparseRecurrentLayer(object):
    """Sparse Autoencoder"""

    def __init__(self,
                 name="SparseRecurrentLayer",
                 num_inputs=100,
                 sdr_size=50,
                 sparsity=0.2,
                 min_weight=1.0,
                 max_weight=-1.0,
                 duty_cycle_decay=0.02,
                 weights_lr=0.0005,
                 inhibition_lr=0.0001,
                 bias_lr=0.001
                ):
        self.name = name
        self.num_inputs = num_inputs
        self.sdr_size = sdr_size
        self.sparsity = sparsity
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.duty_cycle_decay = duty_cycle_decay
        self.weights_lr = weights_lr
        self.inhibition_lr = inhibition_lr
        self.bias_lr = bias_lr

        self.weights = np.random.rand(sdr_size, num_inputs) * (max_weight - min_weight) + min_weight
        self.biases = np.random.rand(sdr_size) * (max_weight - min_weight) + min_weight
        self.activations = np.zeros(sdr_size)
        self.inhibition = np.zeros((sdr_size, sdr_size))

    def generate(self, inputs):
        sums = np.dot(self.weights, inputs.T) + self.biases

        self.activations = np.maximum(0.0, sums)

        sdr = self.activations - np.dot(self.inhibition, self.activations)

        for i in range(0, self.activations.size):
            if sdr[i] > 0.0:
                sdr[i] = 1.0
            else:
                sdr[i] = 0.0

        return sdr

    def learn(self, inputs, sdr):
        recon = self.reconstruct(sdr)

        error = inputs - recon
        print '{0}: error is {1}'.format(self.name, np.mean(np.abs(error)**2, axis=0))

        for i in range(0, self.weights.shape[0]):
            lscf = self.sparsity - sdr[i]

            learn = sdr[i]

            self.weights[i] += self.weights_lr * learn * error

            self.biases[i] += self.bias_lr * lscf

            self.inhibition[i] += self.inhibition_lr * lscf * self.activations[i] * self.activations

            self.inhibition[i] = np.maximum(0.0, self.inhibition[i])

            self.inhibition[i][i] = 0.0

        return error

    def reconstruct(self, sdr):
        return np.dot(self.weights.T, sdr.T)
