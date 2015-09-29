import logging
from math import sqrt
from copy import copy
import abc

from utils.Activations import *
from utils.Loss import *

__author__ = 'eidonfiloi'


class Node(object):

    """ This class is an abstract base class for nodes """

    __metaclass__ = abc.ABCMeta

    def __init__(self, parameters, serialized_object=None):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.parameters = parameters
        if serialized_object is not None:

            self.name = self.parameters['name']
            self.inputs_size = self.parameters['inputs_size']
            self.output_size = self.parameters['output_size']
            self.activation_function = self.parameters['activation_function']
            self.activation_threshold = self.parameters['activation_threshold']

            self.b = sqrt(6.0 / (self.inputs_size + self.output_size))
            self.min_weight = -self.b
            self.max_weight = self.b

            self.momentum = self.parameters['momentum']
            self.velocity = serialized_object['velocity']

            self.dropout_ratio = self.parameters['dropout_ratio']

            if self.dropout_ratio is not None:
                self.dropout = serialized_object['dropout']

            self.weights_lr = self.parameters['weights_lr']
            self.bias_lr = self.parameters['bias_lr']

            self.weights = serialized_object['weights']
            self.biases = serialized_object['biases']
            self.activations = np.zeros(self.output_size)
            self.sdr = np.ones(self.output_size)
            self.learning_rate_increase = self.parameters['learning_rate_increase']
            self.learning_rate_decrease = self.parameters['learning_rate_decrease']
            self.local_gain = serialized_object['local_gain']
            self.prev_local_gain = serialized_object['prev_local_gain']
        else:
            self.name = self.parameters['name']
            self.inputs_size = self.parameters['inputs_size']
            self.output_size = self.parameters['output_size']
            self.activation_function = self.parameters['activation_function']
            self.activation_threshold = self.parameters['activation_threshold']

            self.b = sqrt(6.0 / (self.inputs_size + self.output_size))
            self.min_weight = -self.b
            self.max_weight = self.b

            self.momentum = self.parameters['momentum']
            self.velocity = np.zeros((self.inputs_size, self.output_size)) if self.momentum is not None else None

            self.dropout_ratio = self.parameters['dropout_ratio']

            if self.dropout_ratio is not None:
                self.dropout = np.random.binomial(1, self.dropout_ratio, self.inputs_size)

            self.weights_lr = self.parameters['weights_lr']
            self.bias_lr = self.parameters['bias_lr']

            self.weights = np.random.rand(self.inputs_size, self.output_size) * \
                           (self.max_weight - self.min_weight) + self.min_weight
            self.biases = np.random.rand(self.output_size) * (self.max_weight - self.min_weight) + self.min_weight

            self.activations = np.zeros(self.output_size)
            self.sdr = np.ones(self.output_size)
            self.learning_rate_increase = self.parameters['learning_rate_increase']
            self.learning_rate_decrease = self.parameters['learning_rate_decrease']
            self.local_gain = np.ones((self.inputs_size, self.output_size))
            self.prev_local_gain = np.ones((self.inputs_size, self.output_size))
        self.prev_weight_derivative = np.zeros((self.inputs_size, self.output_size))
        self.weight_derivative = np.zeros((self.inputs_size, self.output_size))
        self.delta_weights = np.zeros((self.inputs_size, self.output_size))
        self.delta_biases = np.zeros(self.output_size)
        self.regularization = self.parameters['regularization']

    def serialize(self):

        serialized_object = dict()

        serialized_object['parameters'] = self.parameters
        serialized_object['weights'] = self.weights
        serialized_object['biases'] = self.biases
        serialized_object['velocity'] = self.velocity
        if self.dropout_ratio is not None:
            serialized_object['dropout'] = self.dropout
        serialized_object['local_gain'] = self.local_gain
        serialized_object['prev_local_gain'] = self.prev_local_gain
        
        return serialized_object

    def __getstate__(self):
        return self.serialize()
    
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

    def __init__(self, parameters, serialized_object=None):

        super(FeedForwardNode, self).__init__(parameters, serialized_object)
        self.parameters = parameters
        if serialized_object is not None:
            self.inhibition = serialized_object['inhibition']
            self.duty_cycles = serialized_object['duty_cycles']
        else:

            self.inhibition = np.zeros((self.output_size, self.output_size))
            self.duty_cycles = np.zeros(self.output_size)
        self.lifetime_sparsity = self.parameters['lifetime_sparsity']
        self.duty_cycle_decay = self.parameters['duty_cycle_decay']
        self.make_sparse = self.parameters['make_sparse']
        self.target_sparsity = self.parameters['target_sparsity']
        self.inhibition_lr = self.parameters['inhibition_lr']

    def serialize(self):
        serialized_object = super(FeedForwardNode, self).serialize()

        serialized_object['inhibition'] = self.inhibition
        serialized_object['duty_cycles'] = self.duty_cycles

        return serialized_object

    def __getstate__(self):
        return self.serialize()

    def generate_node_output(self, inputs):

        if self.dropout_ratio is not None:
            self.dropout = np.random.binomial(1, self.dropout_ratio, self.inputs_size)
            sums = np.dot(inputs.T*self.dropout.T, self.weights).T + self.biases
        else:
            sums = np.dot(inputs.T, self.weights).T + self.biases

        self.activations = Activations.activation(sums, self.activation_function)

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

        if self.dropout_ratio is not None:
            delta_backpropagate = (np.dot(self.weights, delta_) * Activations.derivative(inputs, self.activation_function)) * self.dropout
        else:
            delta_backpropagate = np.dot(self.weights, delta_) * Activations.derivative(inputs, self.activation_function)

        for i in range(0, self.weights.shape[0]):
            if self.momentum is not None:
                self.velocity[i] = self.momentum * self.velocity[i] + self.weights_lr * inputs[i] \
                    * (self.local_gain[i] * delta_)
                self.delta_weights[i] += self.velocity[i]
                # self.weights[i] -= self.velocity[i]
            else:
                # self.weights[i] -= self.weights_lr * inputs[i] * (self.local_gain[i] * delta_)
                self.delta_weights[i] += self.weights_lr * inputs[i] * (self.local_gain[i] * delta_)
        self.delta_biases += self.bias_lr * delta_

        if self.learning_rate_increase is not None:
            for i in range(0, self.weight_derivative.shape[0]):
                self.weight_derivative[i] = inputs[i] * delta_

            derivative_change = np.multiply(self.prev_weight_derivative, self.weight_derivative)
            gradient_change = (derivative_change > 0.0).astype('int')
            gain_increase = np.multiply(gradient_change, self.prev_local_gain + self.learning_rate_increase * np.ones(self.prev_local_gain.shape))
            gradient_change = (derivative_change <= 0.0).astype('int')
            gain_decrease = np.multiply(gradient_change, self.prev_local_gain * self.learning_rate_decrease)
            self.prev_local_gain = copy(self.local_gain)
            self.local_gain = gain_increase + gain_decrease
            self.prev_weight_derivative = copy(self.weight_derivative)

        return delta_backpropagate

    def update_weights(self, num_iter):

        self.weights -= ((1.0 / num_iter) * self.delta_weights + self.weights_lr * self.regularization * self.weights)
        self.biases -= (1.0 / num_iter) * self.delta_biases

        self.delta_weights = np.zeros(self.delta_weights.shape)
        self.delta_biases = np.zeros(self.delta_biases.shape)


class SRAutoEncoderNode(FeedForwardNode):

    """
    Sparse Recurrent Autoencoder Node, capable of generating feedforward output,
    backpropagating errors and reconstruction errors
    """

    def __init__(self, parameters, serialized_object=None):
        super(SRAutoEncoderNode, self).__init__(parameters, serialized_object)
        
        self.parameters = parameters

        if serialized_object is not None:
            self.recon_biases = serialized_object['recon_biases']
            self.output_weights = serialized_object['output_weights']
            self.output_local_gain = serialized_object['output_local_gain']
            self.prev_output_local_gain = serialized_object['prev_output_local_gain']
        else:
            self.recon_biases = np.random.rand(self.inputs_size) * (self.max_weight - self.min_weight) + self.min_weight
            self.is_transpose_reconstruction = self.parameters['is_transpose_reconstruction']
            self.output_weights = self.weights.T if self.is_transpose_reconstruction \
                else np.random.rand(self.output_size, self.inputs_size) * (self.max_weight - self.min_weight) + self.min_weight
            self.output_local_gain = self.local_gain.T if self.is_transpose_reconstruction \
                else np.ones((self.output_size, self.inputs_size))
            self.prev_output_local_gain = self.prev_local_gain.T if self.is_transpose_reconstruction \
                else np.ones((self.output_size, self.inputs_size))
        self.recon_bias_lr = self.parameters['recon_bias_lr']

    def serialize(self):

        serialized_object = super(SRAutoEncoderNode, self).serialize()

        serialized_object['recon_biases'] = self.recon_biases
        serialized_object['output_weights'] = self.output_weights
        serialized_object['output_local_gain'] = self.output_local_gain
        serialized_object['prev_output_local_gain'] = self.prev_output_local_gain

        return serialized_object

    def __getstate__(self):
        return self.serialize()

    def generate_node_output(self, inputs):
        return super(SRAutoEncoderNode, self).generate_node_output(inputs)

    def backpropagate(self, inputs, delta):
        return super(SRAutoEncoderNode, self).backpropagate(inputs, delta)

    def learn_reconstruction(self, output_target, hidden, input_target=None, backpropagate_hidden=True):

        recon = self.reconstruct(hidden)
        error_diff = recon - output_target

        mse = sqrt(np.mean(np.abs(error_diff) ** 2, axis=0))
        self.logger.info('{0}: error is {1}'.format(self.name, mse))

        if self.dropout_ratio is not None:
            recon_delta = (error_diff * Activations.derivative(recon, self.activation_function)) * self.dropout
        else:
            recon_delta = error_diff * Activations.derivative(recon, self.activation_function)

        for i in range(0, self.output_weights.shape[0]):
            self.output_weights[i] -= self.weights_lr * hidden[i] * (recon_delta * self.output_local_gain[i])
        self.recon_biases -= self.recon_bias_lr * recon_delta

        if self.learning_rate_increase is not None:
            gradient_change = (np.multiply(self.prev_output_local_gain, self.output_local_gain) > 0.0).astype('int')

            gain_increase = np.multiply(gradient_change, self.prev_output_local_gain + self.learning_rate_increase * np.ones(self.prev_output_local_gain.shape))

            gradient_change = (np.multiply(self.prev_output_local_gain, self.output_local_gain) <= 0.0).astype('int')

            gain_decrease = np.multiply(gradient_change, self.prev_output_local_gain * self.learning_rate_decrease)

            self.local_gain = (gain_increase + gain_decrease).T

        if backpropagate_hidden:
            delta_hidden = np.dot(self.weights.T, recon_delta) * Activations.derivative(hidden, self.activation_function)

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
        return Activations.activation(reconstruct_activation, self.activation_function)

