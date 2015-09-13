__author__ = 'eidonfiloi'

import numpy as np
import logging
from math import sqrt
from scipy.special import expit
from copy import deepcopy
from utils.Utils import *
import abc


class Node(object):

    """ This class is an abstract base class for nodes """

    __metaclass__ = abc.ABCMeta

    def __init__(self, parameters):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.name = parameters['name']
        self.inputs_size = parameters['inputs_size']
        self.output_size = parameters['output_size']
        self.activation_function = parameters['activation_function']
        self.activation_threshold = parameters['activation_threshold']

        self.b = sqrt(6.0 / (self.inputs_size + self.output_size))
        self.min_weight = -self.b
        self.max_weight = self.b

        self.momentum = parameters['momentum']
        self.velocity = np.zeros((self.inputs_size, self.output_size)) if self.momentum is not None else None

        self.dropout_ratio = parameters['dropout_ratio']

        if self.dropout_ratio is not None:
            self.dropout = np.ones((self.inputs_size, self.output_size))
            for i in range(self.inputs_size):
                for j in range(self.output_size):
                    self.dropout[i][j] *= np.random.binomial(1, self.dropout_ratio)

        #self.local_activation_radius = np.round(parameters['local_activation_radius'] * self.inputs_size)
        # if self.local_activation_radius is not None:
        #     self.overlapping = self.local_activation_radius \
        #         - np.round((self.inputs_size / self.output_size))
        #     self.dropout = np.zeros((self.inputs_size, self.output_size))
        #     for i in range(self.inputs_size):
        #         start = i*(self.local_activation_radius - self.overlapping)
        #         end = start + self.local_activation_radius
        #         for j in range(self.dropout[i].size):
        #             if start <= j < end:
        #                 self.dropout[i][j] = 1.0

        self.weights_lr = parameters['weights_lr']
        self.bias_lr = parameters['bias_lr']

        self.weights = np.random.rand(self.inputs_size, self.output_size) * \
            (self.max_weight - self.min_weight) + self.min_weight
        self.biases = np.random.rand(self.output_size) * (self.max_weight - self.min_weight) + self.min_weight
        self.activations = np.zeros(self.output_size)
        self.sdr = np.ones(self.output_size)
        self.learning_rate_decay = parameters['learning_rate_decay']
        self.local_gain = np.ones((self.inputs_size, self.output_size))
        self.prev_local_gain = np.ones((self.inputs_size, self.output_size))

    @abc.abstractmethod
    def generate_node_output(self, inputs):
        """

        :param inputs:
        :return:
        """

    @abc.abstractmethod
    def backpropagate(self, inputs, delta):
        """
        
        :param delta: 
        :return:
        """


class FeedForwardNode(Node):

    """ A feedforward node capable of generating feedforward output and backpropagating errors """

    def __init__(self, parameters):
        super(FeedForwardNode, self).__init__(parameters)

        self.lifetime_sparsity = parameters['lifetime_sparsity']
        self.duty_cycle_decay = parameters['duty_cycle_decay']
        self.make_sparse = parameters['make_sparse']
        self.target_sparsity = parameters['target_sparsity']
        self.inhibition_lr = parameters['inhibition_lr']
        self.inhibition = np.zeros((self.output_size, self.output_size))
        self.duty_cycles = np.zeros(self.output_size)

    def generate_node_output(self, inputs):
        # if self.local_activation_radius is not None:
        #     modified_weights = np.multiply(self.dropout, self.weights)
        #     sums = np.dot(inputs.T, modified_weights).T + self.biases
        # else:
        #     sums = np.dot(inputs.T, self.weights).T + self.biases

        if self.dropout_ratio is not None:
            modified_weights = np.multiply(self.dropout, self.weights)
            sums = np.dot(inputs.T, modified_weights).T + self.biases
        else:
            sums = np.dot(inputs.T, self.weights).T + self.biases

        if self.activation_function == "Sigmoid":
            self.activations = expit(sums)
        elif self.activation_function == "Rectifier":
            self.activations = np.maximum(0.0, sums)
        elif self.activation_function == "Tanh":
            self.activations = np.tanh(sums)
        else:
            self.activations = sums

        output = self.activations
        if self.make_sparse:
            output = self.activations - np.dot(self.inhibition, self.activations)

            output[output >= self.activation_threshold] = 1.0
            output[output < self.activation_threshold] = 0.0

            self.duty_cycles = (1.0 - self.duty_cycle_decay) * self.duty_cycles + self.duty_cycle_decay * output
            self.sdr = output

        return output

    def backpropagate(self, inputs, delta):

        delta_ = delta * self.sdr

        delta_backpropagate = np.dot(self.weights, delta_) * Utils.derivative(inputs, self.activation_function)

        for i in range(0, self.weights.shape[0]):
            if self.momentum is not None:
                self.velocity[i] = self.momentum * self.velocity[i] + self.weights_lr * inputs[i] \
                    * (self.local_gain[i] * delta_)
                self.weights[i] -= self.velocity[i]
            else:
                self.weights[i] -= self.weights_lr * inputs[i] * (self.local_gain[i] * delta_)

            self.biases -= self.bias_lr * delta_

            # if self.learning_rate_decay is not None:
            #     for j in range(self.local_gain[i].size):
            #         if (self.prev_local_gain[i][j] * self.local_gain[i][j]) > 0.0:
            #             self.local_gain[i][j] = self.prev_local_gain[i][j] + self.learning_rate_decay
            #         else:
            #             self.local_gain[i][j] = self.prev_local_gain[i][j] * (1.0 - self.learning_rate_decay)

        if self.learning_rate_decay is not None:
            gradient_change = (np.multiply(self.prev_local_gain, self.local_gain) > 0.0).astype('int')

            gain_increase = np.multiply(gradient_change, self.prev_local_gain + self.learning_rate_decay * np.ones(self.prev_local_gain.shape))

            gradient_change = (np.multiply(self.prev_local_gain, self.local_gain) <= 0.0).astype('int')

            gain_decrease = np.multiply(gradient_change, self.prev_local_gain * (1.0 - self.learning_rate_decay))

            self.local_gain = gain_increase + gain_decrease

        return delta_backpropagate


class SRAutoEncoderNode(FeedForwardNode):

    """
    Sparse Recurrent Autoencoder Node, capable of generating feedforward output,
    backpropagating errors and reconstruction errors
    """

    def __init__(self, parameters):
        super(SRAutoEncoderNode, self).__init__(parameters)

        self.recon_bias_lr = parameters['recon_bias_lr']
        self.recon_biases = np.random.rand(self.inputs_size) * (self.max_weight - self.min_weight) + self.min_weight
        self.is_transpose_reconstruction = parameters['is_transpose_reconstruction']
        self.output_weights = self.weights.T if self.is_transpose_reconstruction \
            else np.random.rand(self.output_size, self.inputs_size) * (self.max_weight - self.min_weight) + self.min_weight
        self.output_local_gain = self.local_gain.T if self.is_transpose_reconstruction \
            else np.ones((self.output_size, self.inputs_size))
        self.prev_output_local_gain = self.prev_local_gain.T if self.is_transpose_reconstruction \
            else np.ones((self.output_size, self.inputs_size))

    def generate_node_output(self, inputs):
        sums = np.dot(inputs.T, self.weights).T + self.biases

        if self.activation_function == "Sigmoid":
            self.activations = expit(sums)
        elif self.activation_function == "Rectifier":
            self.activations = np.maximum(0.0, sums)
        elif self.activation_function == "Tanh":
            self.activations = np.tanh(sums)
        else:
            self.activations = sums

        output = self.activations
        if self.make_sparse:
            output = self.activations - np.dot(self.inhibition, self.activations)

            output[output >= self.activation_threshold] = 1.0
            output[output < self.activation_threshold] = 0.0

            self.duty_cycles = (1.0 - self.duty_cycle_decay) * self.duty_cycles + self.duty_cycle_decay * output

        return output

    def learn_reconstruction(self, output_target, hidden, input_target=None, backpropagate_hidden=True):

        recon = self.reconstruct(hidden)
        # if self.name == "layer1-recurrent":
        #     self.logger.info('################# recurrent\n'
        #                      'out_target:\n'
        #                      '{0}\n'
        #                      'hidden:\n'
        #                      '{1}'.format(output_target, hidden))

        error_diff = recon - output_target

        mse = sqrt(np.mean(np.abs(error_diff) ** 2, axis=0))
        self.logger.info('{0}: error is {1}'.format(self.name, mse))

        recon_delta = error_diff * Utils.derivative(recon, self.activation_function)

        for i in range(0, self.output_weights.shape[0]):
            self.output_weights[i] -= self.weights_lr * hidden[i] * (recon_delta * self.output_local_gain[i])
            self.recon_biases -= self.recon_bias_lr * recon_delta

        if self.learning_rate_decay is not None:
            gradient_change = (np.multiply(self.prev_output_local_gain, self.output_local_gain) > 0.0).astype('int')

            gain_increase = np.multiply(gradient_change, self.prev_output_local_gain + self.learning_rate_decay * np.ones(self.prev_output_local_gain.shape))

            gradient_change = (np.multiply(self.prev_output_local_gain, self.output_local_gain) <= 0.0).astype('int')

            gain_decrease = np.multiply(gradient_change, self.prev_output_local_gain * (1.0 - self.learning_rate_decay))

            self.local_gain = (gain_increase + gain_decrease).T

        if backpropagate_hidden:
            delta_hidden = np.dot(self.weights.T, recon_delta) * Utils.derivative(hidden, self.activation_function)

            if input_target is not None:
                self.backpropagate(input_target, delta_hidden)
            else:
                self.backpropagate(output_target, delta_hidden)

        if self.make_sparse:
            lifetime_sparsity_correction_factor = (np.array([self.lifetime_sparsity
                                                             in range(0, len(self.duty_cycles))]) - self.duty_cycles)
            for i in range(0, self.activations.shape[0]):
                self.inhibition[i] += self.inhibition_lr * lifetime_sparsity_correction_factor[i] * self.activations[i] \
                    * self.activations
                self.inhibition[i] = np.maximum(0.0, self.inhibition[i])
                self.inhibition[i][i] = 0.0

        return mse

    def reconstruct(self, hidden):
        reconstruct_activation = np.dot(hidden.T, self.output_weights).T + self.recon_biases
        if self.activation_function == "Sigmoid":
            return expit(reconstruct_activation)
        elif self.activation_function == "Rectifier":
            return np.maximum(0.0, reconstruct_activation)
        elif self.activation_function == "Tanh":
            return np.tanh(reconstruct_activation)
        else:
            return reconstruct_activation


class SRAutoEncoderOld(Node):
    """Sparse Recurrent Autoencoder"""

    def __init__(self, config):
        super(SRAutoEncoderOld, self).__init__()
        self.name = config['name']
        self.num_inputs = config['num_inputs']
        self.sdr_size = config['sdr_size']
        self.activation_function = config['activation']
        self.threshold = 0.5 if self.activation_function == 'Sigmoid' else 0.0
        self.lifetime_sparsity = config['lifetime_sparsity']
        self.duty_cycle_decay = config['duty_cycle_decay']
        self.min_weight = config['min_weight']
        self.max_weight = config['max_weight']
        self.dropout = config['dropout']
        self.zoom = config['zoom']
        self.sparsify = config['sparsify']
        self.target_sparsity = config['target_sparsity']
        self.duty_cycle_decay = config['duty_cycle_decay']
        self.weights_lr = config['weights_lr']
        self.inhibition_lr = config['inhibition_lr']
        self.output_bias_lr = config['output_bias_lr']
        self.hidden_bias_lr = config['hidden_bias_lr']

        self.weights = np.random.rand(self.num_inputs + 2 * self.zoom, self.sdr_size) * (
            self.max_weight - self.min_weight) + self.min_weight \
            if self.zoom is not None \
            else np.random.rand(self.num_inputs, self.sdr_size) * (self.max_weight - self.min_weight) + self.min_weight
        self.output_biases = np.random.rand(self.num_inputs + 2 * self.zoom) * (
            self.max_weight - self.min_weight) + self.min_weight \
            if self.zoom is not None \
            else np.random.rand(self.num_inputs) * (self.max_weight - self.min_weight) + self.min_weight
        self.hidden_biases = np.random.rand(self.sdr_size) * (self.max_weight - self.min_weight) + self.min_weight
        self.activations = np.zeros(self.sdr_size)
        self.inhibition = np.zeros((self.sdr_size, self.sdr_size))
        self.duty_cycles = np.zeros(self.sdr_size)

    def generate_node_output(self, inputs):
        inputs_ = deepcopy(inputs)
        if self.dropout is not None:
            probs = np.ones(self.num_inputs)
            for i in range(probs.size):
                inputs_[i] = np.random.binomial(1, self.dropout) * inputs[i]

        if self.zoom is not None and self.dropout is None:
            inputs_ = np.pad(inputs, (self.zoom, self.zoom), 'constant', constant_values=(0, 0))
            size = inputs.size
            middle = np.random.randint(self.zoom, size + self.zoom)
            end = middle + self.zoom
            start = middle - self.zoom
            for i in range(inputs_.size):
                if end < i or i < start:
                    inputs_[i] = 0

        sums = np.dot(inputs_.T, self.weights).T + self.hidden_biases

        if self.activation_function == "Sigmoid":
            self.activations = expit(sums)
        elif self.activation_function == "Rectifier":
            self.activations = np.maximum(0.0, sums)
        elif self.activation_function == "Tanh":
            self.activations = np.tanh(sums)
        else:
            self.activations = np.maximum(0.0, sums)

        sdr = self.activations
        if self.sparsify:
            sdr = self.activations - np.dot(self.inhibition, self.activations)

            for i in range(0, self.activations.size):
                if sdr[i] > self.threshold:
                    sdr[i] = 1.0
                else:
                    sdr[i] = 0.0

            self.duty_cycles = (1.0 - self.duty_cycle_decay) * self.duty_cycles + self.duty_cycle_decay * sdr
        actual_sparsity = np.mean(sdr)
        if actual_sparsity < self.target_sparsity * 0.5:
            probs = np.ones(sdr)
            for i in range(probs.size):
                probs[i] = np.random.binomial(1, 0.5)
            sdr = probs

        if actual_sparsity > self.target_sparsity * 1.5:
            pass

        return sdr

    def learn(self, inputs, sdr):

        recon = self.reconstruct(sdr)
        if self.zoom is not None:
            inputs_ = np.pad(np.zeros(inputs.size), (self.zoom, self.zoom), 'constant', constant_values=(0, 0))
        else:
            inputs_ = deepcopy(inputs)

        error = recon - inputs_

        mse = sqrt(np.mean(np.abs(error) ** 2, axis=0))
        self.logger.info('{0}: error is {1}'.format(self.name, mse))

        if mse > len(error):
            self.weights_lr /= 10.0
            self.output_bias_lr /= 10.0
            self.inhibition_lr /= 10.0
            self.reinitialize_hidden_state()
            self.logger.info('########################## network reinitialization ##########################')

        self.logger.info('{0} sparsity: {1}'.format(self.name, np.mean(sdr)))

        output_delta = error * Utils.sigmoid_derivative(recon) \
            if self.activation_function == "Sigmoid" \
            else (error * Utils.tanh_derivative(recon)
                  if self.activation_function == "Tanh"
                  else error * Utils.rectifier_derivative(recon))

        hidden_delta = np.dot(self.weights.T, output_delta) * Utils.sigmoid_derivative(self.activations) \
            if self.activation_function == "Sigmoid" \
            else (np.dot(self.weights.T, output_delta) * Utils.tanh_derivative(self.activations)
                  if self.activation_function == "Tanh"
                  else np.dot(self.weights.T, output_delta) * Utils.rectifier_derivative(self.activations))

        for i in range(0, self.weights.T.shape[0]):
            self.weights.T[i] -= self.weights_lr * inputs_[i] * output_delta
            self.output_biases -= self.output_bias_lr * output_delta

        for i in range(0, self.weights.shape[0]):
            self.weights[i] -= self.weights_lr * inputs_[i] * hidden_delta
            self.hidden_biases -= self.hidden_bias_lr * hidden_delta

        if self.sparsify:
            lifetime_sparsity_correction_factor = (np.array([self.lifetime_sparsity
                                                             for _ in
                                                             range(0, len(self.duty_cycles))]) - self.duty_cycles)
            for i in range(0, self.activations.shape[0]):
                self.inhibition[i] += self.inhibition_lr * lifetime_sparsity_correction_factor[i] * self.activations[
                    i] * self.activations
                self.inhibition[i] = np.maximum(0.0, self.inhibition[i])
                self.inhibition[i][i] = 0.0

        return mse

    def reconstruct(self, sdr):
        reconstruct_activation = np.dot(sdr.T, self.weights.T).T + self.output_biases
        if self.activation_function == "Sigmoid":
            return expit(reconstruct_activation)
        elif self.activation_function == "Rectifier":
            return np.maximum(0.0, reconstruct_activation)
        elif self.activation_function == "Tanh":
            return np.tanh(reconstruct_activation)
        else:
            return np.maximum(0.0, reconstruct_activation)

    def reinitialize_hidden_state(self):
        self.weights = np.random.rand(self.num_inputs + 2 * self.zoom, self.sdr_size) * (
            self.max_weight - self.min_weight) + self.min_weight \
            if self.zoom is not None \
            else np.random.rand(self.num_inputs, self.sdr_size) * (self.max_weight - self.min_weight) + self.min_weight
        self.output_biases = np.random.rand(self.num_inputs + 2 * self.zoom) * (
            self.max_weight - self.min_weight) + self.min_weight \
            if self.zoom is not None \
            else np.random.rand(self.num_inputs) * (self.max_weight - self.min_weight) + self.min_weight
        self.hidden_biases = np.random.rand(self.sdr_size) * (self.max_weight - self.min_weight) + self.min_weight
        self.activations = np.zeros(self.sdr_size)
        self.inhibition = np.zeros((self.sdr_size, self.sdr_size))
        self.duty_cycles = np.zeros(self.sdr_size)

    def backpropagate(self, inputs, delta):
        pass
