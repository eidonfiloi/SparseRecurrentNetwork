__author__ = 'ptoth'

import abc

from Layer import *
import logging
import matplotlib.pyplot as plt
import time
from math import sqrt
from utils.Utils import *


class Network(object):

    """ This class is an abstract base class for deep learning networks """

    __metaclass__ = abc.ABCMeta

    def __init__(self, parameters):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.name = parameters['name']
        self.verbose = parameters['verbose']

    @abc.abstractmethod
    def run(self, inputs):
        """
        
        :param inputs: 
        :return:
        """


class SRNetwork(Network):

    """ Sparse Recurrent network"""
    
    def __init__(self, parameters):
        super(SRNetwork, self).__init__()

        self.layers = [SRLayer(layer_conf) for layer_conf in parameters['layers']]
        self.num_layers = len(self.layers)
        self.activation_function = parameters['activation_function']

    def run(self, inputs, learning_on=True):

        feedforward_errors = {layer.name: [] for layer in self.layers}
        recurrent_errors = {layer.name: [] for layer in self.layers}

        feedforward_outputs = {layer.name: [] for layer in self.layers}
        recurrent_outputs = {layer.name: [] for layer in self.layers}
        feedback_outputs = {layer.name: [] for layer in self.layers}

        feedforward_deltas = {layer.name: [] for layer in self.layers}
        recurrent_deltas = {layer.name: [] for layer in self.layers}
        feedback_deltas = {layer.name: [] for layer in self.layers}

        start_input = inputs
        current_input = inputs
        if self.num_layers == 1:
            layer = self.layers[0]
            current_input, error = layer.generate_feedforward(current_input)
            feedforward_errors[layer.name].append(error)
            feedforward_outputs[layer.name].append(current_input)

            current_input, error = layer.generate_recurrent(current_input)
            recurrent_errors[layer.name].append(error)
            recurrent_outputs[layer.name].append(current_input)

            prediction = layer.generate_feedback(current_input)
            feedback_outputs[layer.name].append(prediction)

            output_error = prediction - start_input

            output_mse = sqrt(np.mean(np.abs(output_error) ** 2, axis=0))

            delta_out = output_error * Utils.derivative(prediction, self.activation_function)

            delta_backpropagate = layer.backpropagate_feedback(delta_out)
            feedback_deltas[layer.name].append(delta_backpropagate)

            delta_backpropagate = layer.backpropagate_recurrent(delta_backpropagate)
            recurrent_deltas[layer.name].append(delta_backpropagate)

            delta_backpropagate = layer.backpropagate_feedforward(delta_backpropagate)
            feedforward_deltas[layer.name].append(delta_backpropagate)

            return prediction, output_mse

        prediction = None
        output_mse = None
        for ind, layer in enumerate(self.layers):
            if ind == self.num_layers - 1:

                # ###################### feedback pass ######################

                for ind_back, layer_back in enumerate(reversed(self.layers)):
                    if ind_back == 0:
                        current_input, error = layer_back.generate_recurrent(current_input)
                        recurrent_errors[layer_back.name].append(error)
                        recurrent_outputs[layer_back.name].append(current_input)

                        current_input = layer_back.generate_feedback(current_input)
                        feedback_outputs[layer_back.name].append(current_input)
                    elif ind_back == self.num_layers - 1:
                        prediction = layer_back.generate_feedback(
                            np.concatenate(layer_back.recurrent_output, current_input))
                        output_error = prediction - start_input
                        output_mse = sqrt(np.mean(np.abs(output_error) ** 2, axis=0))
                        delta_out = output_error * Utils.derivative(prediction, self.activation_function)
                        delta_backpropagate = delta_out

                        # ###################### backpropagation on feedback and recurrent ######################

                        for ind_backprop, layer_backprop in enumerate(self.layers):
                            delta_backpropagate = layer_backprop.backpropagate_feedback(delta_backpropagate)
                            feedback_deltas[layer_backprop.name].append(delta_backpropagate)
                            if ind_backprop == self.num_layers - 1:
                                delta_backpropagate = layer_backprop.backpropagate_recurrent(delta_backpropagate)
                                recurrent_deltas[layer_backprop.name].append(delta_backpropagate)

                                # ###################### backpropagation for feedforward ######################

                                for ind_backprop_back, layer_backprop_back in enumerate(reversed(self.layers)):
                                    delta_backpropagate = layer_backprop_back.backpropagate_feedforward(delta_backpropagate)
                                    feedforward_deltas[layer_backprop_back.name].append(delta_backpropagate)
                            else:
                                rec_delta = delta_backpropagate[0:layer_backprop.recurrent_node.output_size]
                                rec_delta_backpropagate = layer_backprop.backpropagate_recurrent(rec_delta)
                                recurrent_deltas[layer_backprop.name].append(rec_delta_backpropagate)
                    else:
                        current_input = layer_back.generate_feedback(
                            np.concatenate(layer_back.recurrent_output, current_input))
                        feedback_outputs[layer_back.name].append(current_input)

            # ###################### feedforward pass ######################
            else:
                current_input, error = layer.generate_feedforward(current_input)
                feedforward_errors[layer.name].append(error)
                feedforward_outputs[layer.name].append(current_input)

                current_input, error = layer.generate_recurrent(current_input)
                recurrent_errors[layer.name].append(error)
                recurrent_outputs[layer.name].append(current_input)

        return prediction, output_mse


class SRNetworkOld(Network):

    def __init__(self, config):
        super(SRNetworkOld, self).__init__()
        self.name = config['name']
        self.layers = [SRLayerOld(layer_conf) for layer_conf in config['layers']]
        self.num_layers = len(self.layers)
        self.verbose = config['verbose']
        self.visualize_grid_size = config['visualize_grid_size']

    def run(self, inputs, learning_on=True):

        feedforward_errors = {layer.name: [] for layer in self.layers}
        recurrent_errors = {layer.name: [] for layer in self.layers}
        feedback_errors = {layer.name: [] for layer in self.layers}

        feedforward_sdrs = {layer.name: [] for layer in self.layers}
        recurrent_sdrs = {layer.name: [] for layer in self.layers}
        feedback_sdrs = {layer.name: [] for layer in self.layers}

        current_input = inputs
        for ind, layer in enumerate(self.layers):
            current_input, forward_error = layer.generate_feedforward(current_input)
            if forward_error is not None:
                feedforward_errors[layer.name].append(forward_error)
            feedforward_sdrs[layer.name].append(current_input)
            if ind == self.num_layers - 1:
                for indx, layer_back in enumerate(reversed(self.layers)):
                    current_input, rec_error = layer_back.generate_recurrent(current_input, learning_on)
                    if rec_error is not None:
                        recurrent_errors[layer_back.name].append(rec_error)
                    recurrent_sdrs[layer_back.name].append(current_input)
                    if indx == self.num_layers - 1:
                        current_input, out_error = layer_back.generate_output(current_input)
                    else:
                        current_input, back_error = layer_back.generate_feedback(current_input, learning_on)
                        if back_error is not None:
                            feedback_errors[layer_back.name].append(back_error)
                        feedback_sdrs[layer_back.name].append(current_input)
                        #current_input = layer_back.generate_output(current_input)
        if self.verbose > 1:
            self.visualize_hidden_states(feedforward_sdrs, recurrent_sdrs, feedback_sdrs)
            #self.visualize_inhibition()
        return current_input, feedforward_errors, recurrent_errors, feedback_errors

    def visualize_hidden_states(self, feedforward_sdrs, recurrent_sdrs, feedback_sdrs):
        plt.ion()
        for ind, k in enumerate(sorted(feedforward_sdrs)):
            v = feedforward_sdrs[k]
            if len(v) > 0:
                reshape_size = round(sqrt(v[0].shape[0]))
                ax0 = plt.subplot(self.num_layers, 3, 3*ind + 1)
                ax0.axis([1, reshape_size, 1, reshape_size])
                x, y = np.argwhere(v[0].reshape(reshape_size, reshape_size) == 1).T
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
                x, y = np.argwhere(v[0].reshape(reshape_size, reshape_size) == 1).T
                ax1.scatter(x, y, alpha=0.5, c='r', marker='s')
                ax1.set_title('recurrent: {0}'.format(ind + 1))
                ax1.set_xticks([])
                ax1.set_yticks([])

        for ind, k in enumerate(sorted(feedback_sdrs)):
            v = feedback_sdrs[k]
            if len(v) > 0:
                reshape_size = round(sqrt(v[0].shape[0]))
                ax2 = plt.subplot(self.num_layers, 3, 3*ind + 3)
                ax2.axis([1,reshape_size, 1, reshape_size])
                x, y = np.argwhere(v[0].reshape(reshape_size, reshape_size) == 1).T
                ax2.scatter(x, y, alpha=0.5, c='g', marker='s')
                ax2.set_title('back: {0}'.format(ind + 1))
                ax2.set_xticks([])
                ax2.set_yticks([])
        plt.draw()
        time.sleep(0.05)
        plt.clf()

    def visualize_inhibition(self):
        plt.ion()
        for ind, k in enumerate(self.layers):
            m0 = k.feedforward_node.inhibition
            ax0 = plt.subplot(self.num_layers, 3, 3*ind + 1)
            ax0.pcolor(m0)
            ax0.axis([1, m0.shape[0], 1, m0.shape[0]])
            ax0.set_title('forward: {0}'.format(ind + 1))
            ax0.set_xticks([])
            ax0.set_yticks([])

            m1 = k.recurrent_node.inhibition
            plt.pcolor(m1)
            ax1 = plt.subplot(self.num_layers, 3, 3*ind + 2)
            ax1.axis([1, m1.shape[0], 1, m1.shape[0]])
            ax1.set_title('recurrent: {0}'.format(ind + 1))
            ax1.set_xticks([])
            ax1.set_yticks([])

            m2 = k.feedback_node.inhibition
            plt.pcolor(m2)
            ax2 = plt.subplot(self.num_layers, 3, 3*ind + 3)
            ax2.axis([1, m2.shape[0], 1, m2.shape[0]])
            ax2.set_title('back: {0}'.format(ind + 1))
            ax2.set_xticks([])
            ax2.set_yticks([])
        plt.draw()
        time.sleep(0.05)
        plt.clf()
