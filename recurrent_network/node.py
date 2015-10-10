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

    def __init__(self, parameters):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.parameters = parameters

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
        else:
            self.dropout = np.ones(self.inputs_size)

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
        self.delta_weights_mp = np.zeros((self.inputs_size, self.output_size))
        self.delta_biases = np.zeros(self.output_size)
        self.delta_biases_mp = np.zeros(self.output_size)
        self.regularization = self.parameters['regularization']

    def apply_parameters(self, parameters):
        self.parameters = parameters

        self.name = self.parameters['name']
        self.inputs_size = self.parameters['inputs_size']
        self.output_size = self.parameters['output_size']
        self.activation_function = self.parameters['activation_function']
        self.activation_threshold = self.parameters['activation_threshold']
        self.momentum = self.parameters['momentum']
        self.dropout_ratio = self.parameters['dropout_ratio']
        self.weights_lr = self.parameters['weights_lr']
        self.bias_lr = self.parameters['bias_lr']
        self.learning_rate_increase = self.parameters['learning_rate_increase']
        self.learning_rate_decrease = self.parameters['learning_rate_decrease']
        self.regularization = self.parameters['regularization']

    def init_internal_data(self):

        self.b = sqrt(6.0 / (self.inputs_size + self.output_size))
        self.min_weight = -self.b
        self.max_weight = self.b

        self.velocity = np.zeros((self.inputs_size, self.output_size)) if self.momentum is not None else None

        if self.dropout_ratio is not None:
            self.dropout = np.random.binomial(1, self.dropout_ratio, self.inputs_size)
        else:
            self.dropout = np.ones(self.inputs_size)

        self.weights = np.random.rand(self.inputs_size, self.output_size) * \
                       (self.max_weight - self.min_weight) + self.min_weight
        self.biases = np.random.rand(self.output_size) * (self.max_weight - self.min_weight) + self.min_weight

        self.activations = np.zeros(self.output_size)
        self.sdr = np.ones(self.output_size)
        self.local_gain = np.ones((self.inputs_size, self.output_size))
        self.prev_local_gain = np.ones((self.inputs_size, self.output_size))
        self.prev_weight_derivative = np.zeros((self.inputs_size, self.output_size))
        self.weight_derivative = np.zeros((self.inputs_size, self.output_size))
        self.delta_weights = np.zeros((self.inputs_size, self.output_size))
        self.delta_weights_mp = np.zeros((self.inputs_size, self.output_size))
        self.delta_biases = np.zeros(self.output_size)
        self.delta_biases_mp = np.zeros(self.output_size)

    def __getstate__(self):
        serialized_object = dict()

        serialized_object['parameters'] = self.parameters
        serialized_object['weights'] = self.weights
        serialized_object['biases'] = self.biases
        serialized_object['velocity'] = self.velocity
        serialized_object['dropout'] = self.dropout
        serialized_object['local_gain'] = self.local_gain
        serialized_object['prev_local_gain'] = self.prev_local_gain
        serialized_object['delta_weights_mp'] = self.delta_weights_mp
        serialized_object['delta_biases_mp'] = self.delta_biases_mp
        return serialized_object

    def __setstate__(self, dict):
        self.parameters = dict['parameters']

        self.apply_parameters(parameters=self.parameters)

        self.weights = dict['weights']
        self.biases = dict['biases']
        self.velocity = dict['velocity']
        self.dropout = dict['dropout']
        self.local_gain = dict['local_gain']
        self.prev_local_gain = dict['prev_local_gain']
        self.delta_weights_mp = dict['delta_weights_mp']
        self.delta_biases_mp = dict['delta_biases_mp']
    
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
        self.parameters = parameters
        self.inhibition = np.zeros((self.output_size, self.output_size))
        self.duty_cycles = np.zeros(self.output_size)
        self.lifetime_sparsity = self.parameters['lifetime_sparsity']
        self.duty_cycle_decay = self.parameters['duty_cycle_decay']
        self.make_sparse = self.parameters['make_sparse']
        self.target_sparsity = self.parameters['target_sparsity']
        self.inhibition_lr = self.parameters['inhibition_lr']

    def apply_parameters(self, parameters):
        self.parameters = parameters
        self.lifetime_sparsity = self.parameters['lifetime_sparsity']
        self.duty_cycle_decay = self.parameters['duty_cycle_decay']
        self.make_sparse = self.parameters['make_sparse']
        self.target_sparsity = self.parameters['target_sparsity']
        self.inhibition_lr = self.parameters['inhibition_lr']

    def __getstate__(self):
        parent_state = Node.__getstate__(self)
        parent_state['parameters'] = parent_state['parameters'].update(self.parameters)
        parent_state['inhibition'] = self.inhibition
        parent_state['duty_cycles'] = self.duty_cycles
        return parent_state

    def __setstate(self, dict):
        self.parameters = dict['parameters']
        self.apply_parameters(parameters=self.parameters)
        self.inhibition = dict['inhibition']
        self.duty_cycles = dict['duty_cycles']
        Node.__setstate__(self, dict)

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
            gain_increase = np.multiply(gradient_change, self.prev_local_gain + np.full(self.prev_local_gain.shape, self.learning_rate_increase))
            gradient_change = (derivative_change <= 0.0).astype('int')
            gain_decrease = np.multiply(gradient_change, self.learning_rate_decrease * self.prev_local_gain)
            self.prev_local_gain = copy(self.local_gain)
            self.local_gain = gain_increase + gain_decrease
            self.prev_weight_derivative = copy(self.weight_derivative)

        return delta_backpropagate

    def update_weights(self, num_iter):

        self.weights -= ((1.0 / num_iter) * self.delta_weights + self.weights_lr * self.regularization * self.weights)
        self.biases -= (1.0 / num_iter) * self.delta_biases

        self.delta_weights_mp += self.delta_weights
        self.delta_biases_mp += self.delta_biases_mp

        self.delta_weights = np.zeros(self.delta_weights.shape)
        self.delta_biases = np.zeros(self.delta_biases.shape)

    def collect_deltas(self):
        return [copy(self.delta_weights_mp), copy(self.delta_biases_mp)]

    def update_deltas(self):
        self.delta_weights_mp = np.zeros(self.delta_weights.shape)
        self.delta_biases_mp = np.zeros(self.delta_biases.shape)


class SRAutoEncoderNode(FeedForwardNode):

    """
    Sparse Recurrent Autoencoder Node, capable of generating feedforward output,
    backpropagating errors and reconstruction errors
    """

    def __init__(self, parameters):
        super(SRAutoEncoderNode, self).__init__(parameters)
        
        self.parameters = parameters
        self.recon_biases = np.random.rand(self.inputs_size) * (self.max_weight - self.min_weight) + self.min_weight
        self.is_transpose_reconstruction = self.parameters['is_transpose_reconstruction']
        self.output_weights = self.weights.T if self.is_transpose_reconstruction \
            else np.random.rand(self.output_size, self.inputs_size) * (self.max_weight - self.min_weight) + self.min_weight
        self.output_weight_derivative = np.zeros((self.output_size, self.inputs_size))
        self.prev_output_weight_derivative = np.zeros((self.output_size, self.inputs_size))
        self.output_local_gain = self.local_gain.T if self.is_transpose_reconstruction \
            else np.ones((self.output_size, self.inputs_size))
        self.prev_output_local_gain = self.prev_local_gain.T if self.is_transpose_reconstruction \
            else np.ones((self.output_size, self.inputs_size))
        self.recon_bias_lr = self.parameters['recon_bias_lr']

        self.delta_output_weights_mp = np.zeros((self.output_size, self.inputs_size))
        self.delta_recon_bias_mp = np.zeros(self.inputs_size)

    def __getstate__(self):
        parent_state = FeedForwardNode.__getstate__(self)
        parent_state['recon_biases'] = self.recon_biases
        parent_state['output_weights'] = self.output_weights
        parent_state['output_local_gain'] = self.output_local_gain
        parent_state['prev_output_local_gain'] = self.prev_output_local_gain
        return parent_state

    def __setstate(self, dict):

        self.recon_biases = dict['recon_biases']
        self.output_weights = dict['output_weights']
        self.output_local_gain = dict['output_local_gain']
        self.prev_output_local_gain = dict['prev_output_local_gain']

        FeedForwardNode.__setstate__(self, dict)

    def generate_node_output(self, inputs):
        return super(SRAutoEncoderNode, self).generate_node_output(inputs)

    def backpropagate(self, inputs, delta):
        return super(SRAutoEncoderNode, self).backpropagate(inputs, delta)

    def learn_reconstruction(self, output_target, hidden, input_target=None, backpropagate_hidden=True):

        recon = self.reconstruct(hidden)
        error_diff = recon - output_target

        mse = sqrt(np.mean(np.abs(error_diff) ** 2, axis=0))
        # self.logger.info('{0}: error is {1}'.format(self.name, mse))

        if self.dropout_ratio is not None:
            recon_delta = (error_diff * Activations.derivative(recon, self.activation_function)) * self.dropout
        else:
            recon_delta = error_diff * Activations.derivative(recon, self.activation_function)

        # hidden_tile = np.tile(hidden, (self.output_weights.shape[1], 1)).T
        # recon_delta_tile = np.tile(recon_delta, (self.output_weights.shape[1], 1)).T
        # delta_w = self.weights_lr * np.multiply(hidden_tile, np.multiply(recon_delta_tile, self.output_local_gain))
        delta_w = self.weights_lr * np.multiply(np.dot(np.matrix(hidden).T, np.matrix(recon_delta)), self.output_local_gain)
        self.delta_output_weights_mp += delta_w
        self.output_weights -= delta_w
        # for i in range(0, self.output_weights.shape[0]):
        #     delta_w = self.weights_lr * hidden[i] * (recon_delta * self.output_local_gain[i])
        #     self.delta_output_weights_mp[i] += delta_w
        #     self.output_weights[i] -= delta_w
        delta_b = self.recon_bias_lr * recon_delta
        self.recon_biases -= delta_b
        self.delta_recon_bias_mp += delta_b

        if self.learning_rate_increase is not None:
            self.output_weight_derivative = np.dot(np.matrix(hidden).T, np.matrix(recon_delta))
            # for i in range(0, self.output_weight_derivative.shape[0]):
            #     self.output_weight_derivative[i] = hidden[i] * recon_delta
            derivative_change = np.multiply(self.prev_output_weight_derivative, self.output_weight_derivative)
            gradient_change = (derivative_change > 0.0).astype('int')
            gain_increase = np.multiply(gradient_change, self.prev_output_local_gain + self.learning_rate_increase * np.ones(self.prev_output_local_gain.shape))
            gradient_change = (derivative_change <= 0.0).astype('int')
            gain_decrease = np.multiply(gradient_change, self.prev_output_local_gain * self.learning_rate_decrease)
            self.prev_output_local_gain = copy(self.output_local_gain)
            self.output_local_gain = gain_increase + gain_decrease
            self.prev_output_weight_derivative = copy(self.output_weight_derivative)

        if backpropagate_hidden:
            delta_hidden = np.dot(self.output_weights, recon_delta) * Activations.derivative(hidden, self.activation_function)

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

    def collect_deltas(self):
        return super(SRAutoEncoderNode, self).collect_deltas() + \
               [copy(self.delta_output_weights_mp), copy(self.delta_recon_bias_mp)]

    def update_deltas(self):
        super(SRAutoEncoderNode, self).update_deltas()
        self.delta_output_weights_mp = np.zeros(self.output_weights.shape)
        self.delta_recon_bias_mp = np.zeros(self.recon_biases.shape)

    def reconstruct(self, hidden):
        reconstruct_activation = np.dot(hidden.T, self.output_weights).T + self.recon_biases
        return Activations.activation(reconstruct_activation, self.activation_function)

