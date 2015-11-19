import logging
import time
import pickle

import matplotlib.pyplot as plt

from tensor_factorization.core_network.Layer import *
from utils.Loss import *

__author__ = 'ptoth'


class Network(object):

    """ This class is an abstract base class for deep learning networks """

    __metaclass__ = abc.ABCMeta

    def __init__(self, parameters):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.parameters = parameters
        self.verbose = self.parameters['verbose']
        self.layers = [Layer(layer_conf) for layer_conf in self.parameters['layers']]
        self.input_size = self.parameters['inputs_size']
        self.name = self.parameters['name']
        self.serialize_path = self.parameters['serialize_path']
        self.num_layers = len(self.layers)
        self.activation_function = self.parameters['activation_function']
        self.loss_function = self.parameters['loss_function']
        self.verbose = self.parameters['verbose']
        self.visualize_states = self.parameters['visualize_states']
        self.update_epochs = self.parameters['update_epochs']
        self.update_epochs_counter = 0
        self.curriculum_rate = self.parameters['curriculum_rate']

    def serialize(self, path=None, save=True):
        if path is None:
            path = '{0}/{1}.pickle'.format(self.serialize_path, self.name)
        layers_serialized = [layer.serialize() for layer in self.layers]
        serialized_object = {'layers': layers_serialized}
        if save:
            with open(path, 'wb') as f:
                pickle.dump(serialized_object, f)

        return serialized_object

    def __getstate__(self):
        return {'layers': self.layers}

    def __setstate__(self, dict):
        self.layers = dict['layers']
        self.logger = logging.getLogger(self.__class__.__name__)

    @abc.abstractmethod
    def run(self, inputs):
        """
        
        :param inputs: 
        :return:
        """


class SRNetwork(Network):

    """ Sparse Recurrent network"""
    
    def __init__(self, parameters):
        super(SRNetwork, self).__init__(parameters)

        self.layers = [SRLayer(layer_conf) for layer_conf in parameters['layers']]
        self.feedforward_errors = {layer.name: [] for layer in self.layers}
        self.recurrent_errors = {layer.name: [] for layer in self.layers}
        self.feedback_errors = {layer.name: [] for layer in self.layers}

        self.feedforward_outputs = {layer.name: [] for layer in self.layers}
        self.recurrent_outputs = {layer.name: [] for layer in self.layers}
        self.feedback_outputs = {layer.name: [] for layer in self.layers}

        self.feedforward_deltas = {layer.name: [] for layer in self.layers}
        self.recurrent_deltas = {layer.name: [] for layer in self.layers}
        self.feedback_deltas = {layer.name: [] for layer in self.layers}

        self.previous_prediction = np.zeros(self.layers[0].feedback_node.output_size)
        self.previous_inputs = np.zeros(self.input_size)

    def __getstate__(self):
        parent_state = Network.__getstate__(self)
        return parent_state

    def __setstate(self, dict):
        self.layers = dict['layers']
        self.feedforward_errors = {layer.name: [] for layer in self.layers}
        self.recurrent_errors = {layer.name: [] for layer in self.layers}
        self.feedback_errors = {layer.name: [] for layer in self.layers}

        self.feedforward_outputs = {layer.name: [] for layer in self.layers}
        self.recurrent_outputs = {layer.name: [] for layer in self.layers}
        self.feedback_outputs = {layer.name: [] for layer in self.layers}

        self.feedforward_deltas = {layer.name: [] for layer in self.layers}
        self.recurrent_deltas = {layer.name: [] for layer in self.layers}
        self.feedback_deltas = {layer.name: [] for layer in self.layers}

        self.previous_prediction = np.zeros(self.layers[0].feedback_node.output_size)
        self.previous_inputs = np.zeros(self.input_size)

    def run(self, inputs, target_out=None, learning_on=True):
        self.feedforward_errors = {layer.name: [] for layer in self.layers}
        self.recurrent_errors = {layer.name: [] for layer in self.layers}
        self.feedback_errors = {layer.name: [] for layer in self.layers}

        self.feedforward_outputs = {layer.name: [] for layer in self.layers}
        self.recurrent_outputs = {layer.name: [] for layer in self.layers}
        self.feedback_outputs = {layer.name: [] for layer in self.layers}

        self.feedforward_deltas = {layer.name: [] for layer in self.layers}
        self.recurrent_deltas = {layer.name: [] for layer in self.layers}
        self.feedback_deltas = {layer.name: [] for layer in self.layers}

        output_error = None
        if self.curriculum_rate is not None and np.random.binomial(1, self.curriculum_rate) == 1:
            prediction = self.feedforward_pass(np.concatenate((self.previous_inputs, self.previous_prediction)))
        else:
            prediction = self.feedforward_pass(np.concatenate((self.previous_inputs, inputs)))
        if learning_on:
            # output_error_delta = Loss.delta(self.previous_prediction, inputs, self.loss_function)
            target = inputs if target_out is None else target_out
            output_error = Loss.error(self.previous_prediction, target, self.loss_function)
            if self.verbose is not None and self.verbose > 0:
                self.logger.info('output error is {0}'.format(output_error))
            # delta_backpropagate = output_error_delta * Activations.derivative(self.previous_prediction,
            #                                                                   self.activation_function)
            delta_backpropagate = Loss.delta_backpropagate(self.previous_prediction,
                                                           target,
                                                           self.loss_function,
                                                           self.activation_function)
            self.backpropagate(delta_backpropagate)
        self.previous_prediction = deepcopy(prediction)
        self.previous_inputs = deepcopy(inputs)
        if self.visualize_states:
            self.visualize_hidden_states(self.feedforward_outputs, self.recurrent_outputs)
        for l in self.layers:
                l.cleanup_layer()

        if self.update_epochs_counter == self.update_epochs:
            for l in self.layers:
                l.update_layer_weights(self.update_epochs)
            self.update_epochs_counter = 0
        else:
            self.update_epochs_counter += 1

        return prediction, output_error

    def feedforward_pass(self, inputs, learning_on=True):
        current_input = inputs
        current_activation = inputs
        prediction = None

        for ind, layer in enumerate(self.layers):

            # ###################### feedforward pass ######################

            if ind < self.num_layers - 1:

                current_input, current_activation, error = layer.generate_feedforward(current_input, current_activation, learning_on)
                self.feedforward_errors[layer.name].append(error)
                self.feedforward_outputs[layer.name].append(current_input)

                current_input = np.concatenate((layer.prev_recurrent_output, current_input))
                current_activation = np.concatenate((layer.prev_recurrent_output_activations, current_activation))

                recurrent_output, error = layer.generate_recurrent(current_input, current_activation, learning_on)
                self.recurrent_errors[layer.name].append(error)
                self.recurrent_outputs[layer.name].append(recurrent_output)
            else:
                current_input, current_activation, error = layer.generate_feedforward(current_input, current_activation, learning_on)
                self.feedforward_errors[layer.name].append(error)
                self.feedforward_outputs[layer.name].append(current_input)

                current_input = np.concatenate((layer.prev_recurrent_output, current_input))
                current_activation = np.concatenate((layer.prev_recurrent_output_activations, current_activation))

                recurrent_output, error = layer.generate_recurrent(current_input, current_activation, learning_on)
                self.recurrent_errors[layer.name].append(error)
                self.recurrent_outputs[layer.name].append(recurrent_output)

                # ###################### feedback pass ######################

                for ind_back, layer_back in enumerate(reversed(self.layers)):
                    if ind_back == 0:
                        current_input, current_activation, error = layer_back.generate_feedback(layer_back.recurrent_output, layer_back.recurrent_output_activations, learning_on)
                        self.feedback_outputs[layer_back.name].append(current_input)
                        self.feedback_errors[layer_back.name].append(error)
                    elif ind_back == self.num_layers - 1:
                        prediction, current_activation, error = layer_back.generate_feedback(
                            np.concatenate([layer_back.recurrent_output, current_input]), np.concatenate([layer_back.recurrent_output_activations, current_activation]), learning_on)
                        self.feedback_errors[layer_back.name].append(error)
                        prediction = current_activation
                    else:
                        current_input, current_activation, error = layer_back.generate_feedback(
                            np.concatenate([layer_back.recurrent_output, current_input]),
                            np.concatenate([layer_back.recurrent_output_activations, current_activation]), learning_on)
                        self.feedback_outputs[layer_back.name].append(current_input)
                        self.feedback_errors[layer_back.name].append(error)

        return prediction

    def backpropagate(self, delta_backpropagate):
        if delta_backpropagate is not None:

            # ###################### backpropagation on feedback and recurrent ######################

            for ind_backprop, layer_backprop in enumerate(self.layers):
                delta_backpropagate = layer_backprop.backpropagate_feedback(delta_backpropagate)
                self.feedback_deltas[layer_backprop.name].append(delta_backpropagate)
                if ind_backprop == self.num_layers - 1:
                    delta_backpropagate = layer_backprop.backpropagate_recurrent(delta_backpropagate)
                    self.recurrent_deltas[layer_backprop.name].append(delta_backpropagate)

                    # ###################### backpropagation for feedforward ######################

                    for ind_backprop_back, layer_backprop_back in enumerate(reversed(self.layers)):
                        forward_delta = delta_backpropagate[layer_backprop_back.recurrent_node.output_size:]
                        delta_backpropagate = layer_backprop_back.backpropagate_feedforward(forward_delta)
                        self.feedforward_deltas[layer_backprop_back.name].append(delta_backpropagate)
                else:
                    rec_delta = delta_backpropagate[0:layer_backprop.recurrent_node.output_size]
                    back_delta = delta_backpropagate[layer_backprop.recurrent_node.output_size:]
                    rec_delta_backpropagate = layer_backprop.backpropagate_recurrent(rec_delta)
                    self.recurrent_deltas[layer_backprop.name].append(rec_delta_backpropagate)
                    delta_backpropagate = back_delta

    def collect_network_deltas(self):
        return [layer.collect_layer_deltas() for layer in self.layers]

    def visualize_hidden_states(self, feedforward_sdrs, recurrent_sdrs):
        plt.ion()
        for ind, k in enumerate(sorted(feedforward_sdrs)):
            v = feedforward_sdrs[k]
            if len(v) > 0:
                reshape_size = round(sqrt(v[0].shape[0]))
                ax0 = plt.subplot(self.num_layers, 3, 3*ind + 1)
                ax0.axis([1, reshape_size, 1, reshape_size])
                x, y = np.argwhere(v[0].reshape(reshape_size, reshape_size) > 0.5).T
                ax0.scatter(x, y, alpha=0.5, c='b', marker='s')
                ax0.set_title('forward: {0}'.format(ind + 1))
                ax0.set_xticks([])
                ax0.set_yticks([])

        for ind, k in enumerate(sorted(recurrent_sdrs)):
            v = recurrent_sdrs[k]
            if len(v) > 0:
                reshape_size = round(sqrt(v[0].shape[0]))
                ax1 = plt.subplot(self.num_layers, 3, 3*ind + 2)
                ax1.axis([1, reshape_size, 1, reshape_size])
                x, y = np.argwhere(v[0].reshape(reshape_size, reshape_size) > 0.5).T
                ax1.scatter(x, y, alpha=0.5, c='r', marker='s')
                ax1.set_title('recurrent: {0}'.format(ind + 1))
                ax1.set_xticks([])
                ax1.set_yticks([])
        plt.draw()
        time.sleep(0.05)
        plt.clf()


class SymmetricNetwork(SRNetwork):

    """ Symmetric inputs with dot product output """

    def __init__(self, parameters):
        super(SymmetricNetwork, self).__init__(parameters)

    def run(self, inputs, target_out=None, learning_on=True):
        self.feedforward_errors = {layer.name: [] for layer in self.layers}
        self.recurrent_errors = {layer.name: [] for layer in self.layers}
        self.feedback_errors = {layer.name: [] for layer in self.layers}

        self.feedforward_outputs = {layer.name: [] for layer in self.layers}
        self.recurrent_outputs = {layer.name: [] for layer in self.layers}
        self.feedback_outputs = {layer.name: [] for layer in self.layers}

        self.feedforward_deltas = {layer.name: [] for layer in self.layers}
        self.recurrent_deltas = {layer.name: [] for layer in self.layers}
        self.feedback_deltas = {layer.name: [] for layer in self.layers}

        output_error = None
        prediction = self.feedforward_pass(np.concatenate((self.previous_inputs, inputs)))
        if learning_on:
            target = inputs if target_out is None else target_out
            output_error = Loss.error(self.previous_prediction, target, self.loss_function)
            if self.verbose is not None and self.verbose > 0:
                self.logger.info('output error is {0}'.format(output_error))
            delta_backpropagate = Loss.delta_backpropagate(self.previous_prediction,
                                                           target,
                                                           self.loss_function,
                                                           self.activation_function)
            self.backpropagate(delta_backpropagate)
        self.previous_prediction = deepcopy(prediction)
        self.previous_inputs = deepcopy(inputs)
        if self.visualize_states:
            self.visualize_hidden_states(self.feedforward_outputs, self.recurrent_outputs)
        for l in self.layers:
                l.cleanup_layer()
        if self.update_epochs_counter == self.update_epochs:
            for l in self.layers:
                l.update_layer_weights(self.update_epochs)
            self.update_epochs_counter = 0
        else:
            self.update_epochs_counter += 1

        return prediction, output_error

    def feedforward_pass(self, inputs, learning_on=True):
        current_input = inputs
        current_activation = inputs
        prediction = None

        for ind, layer in enumerate(self.layers):

            # ###################### feedforward pass ######################

            if ind < self.num_layers - 1:

                current_input, current_activation, error = layer.generate_feedforward(current_input, current_activation, learning_on)
                self.feedforward_errors[layer.name].append(error)
                self.feedforward_outputs[layer.name].append(current_input)

                current_input = layer.prev_recurrent_output * current_input
                current_activation = layer.prev_recurrent_output_activations * current_activation

                recurrent_output, error = layer.generate_recurrent(current_input, current_activation, learning_on)
                self.recurrent_errors[layer.name].append(error)
                self.recurrent_outputs[layer.name].append(recurrent_output)
            else:
                current_input, current_activation, error = layer.generate_feedforward(current_input, current_activation, learning_on)
                self.feedforward_errors[layer.name].append(error)
                self.feedforward_outputs[layer.name].append(current_input)

                current_input = layer.prev_recurrent_output * current_input
                current_activation = layer.prev_recurrent_output_activations * current_activation

                recurrent_output, error = layer.generate_recurrent(current_input, current_activation, learning_on)
                self.recurrent_errors[layer.name].append(error)
                self.recurrent_outputs[layer.name].append(recurrent_output)

                # ###################### feedback pass ######################

                for ind_back, layer_back in enumerate(reversed(self.layers)):
                    if ind_back == 0:
                        current_input, current_activation, error = layer_back.generate_feedback(layer_back.recurrent_output, layer_back.recurrent_output_activations, learning_on)
                        self.feedback_outputs[layer_back.name].append(current_input)
                        self.feedback_errors[layer_back.name].append(error)
                    elif ind_back == self.num_layers - 1:
                        prediction, current_activation, error = layer_back.generate_feedback(
                            layer_back.recurrent_output*current_input,
                            layer_back.recurrent_output_activations*current_activation, learning_on)
                        self.feedback_errors[layer_back.name].append(error)
                        prediction = current_activation
                    else:
                        current_input, current_activation, error = layer_back.generate_feedback(
                            layer_back.recurrent_output*current_input,
                            layer_back.recurrent_output_activations*current_activation, learning_on)
                        self.feedback_outputs[layer_back.name].append(current_input)
                        self.feedback_errors[layer_back.name].append(error)

        return prediction

    def backpropagate(self, delta_backpropagate):
        if delta_backpropagate is not None:

            # ###################### backpropagation on feedback and recurrent ######################

            for ind_backprop, layer_backprop in enumerate(self.layers):
                delta_backpropagate = layer_backprop.backpropagate_feedback(delta_backpropagate)
                self.feedback_deltas[layer_backprop.name].append(delta_backpropagate)
                if ind_backprop == self.num_layers - 1:
                    delta_backpropagate = layer_backprop.backpropagate_recurrent(delta_backpropagate)
                    self.recurrent_deltas[layer_backprop.name].append(delta_backpropagate)

                    # ###################### backpropagation for feedforward ######################

                    for ind_backprop_back, layer_backprop_back in enumerate(reversed(self.layers)):
                        forward_delta = copy(delta_backpropagate)
                        delta_backpropagate = layer_backprop_back.backpropagate_feedforward(forward_delta)
                        self.feedforward_deltas[layer_backprop_back.name].append(delta_backpropagate)
                else:
                    rec_delta_backpropagate = layer_backprop.backpropagate_recurrent(delta_backpropagate)
                    self.recurrent_deltas[layer_backprop.name].append(rec_delta_backpropagate)
