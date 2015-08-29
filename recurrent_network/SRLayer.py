__author__ = 'ptoth'

from SRAutoEncoder import *
from copy import deepcopy
import logging


class SRLayer(object):
    """

    """

    def __init__(self, config):
        self.name = config['name']
        self.feedforward_node = SRAutoEncoder(config['feedforward'])
        self.recurrent_node = SRAutoEncoder(config['recurrent'])
        self.feedback_node = SRAutoEncoder(config['feedback'])
        self.logger = logging.getLogger(self.__class__.__name__)

        self.feedforward_sdr = np.zeros(self.feedforward_node.activations.size)
        self.recurrent_sdr = np.zeros(self.recurrent_node.activations.size)
        self.feedback_sdr = np.zeros(self.feedback_node.activations.size)

        self.prev_recurrent_sdr = np.zeros(self.recurrent_node.activations.size)
        self.prev_feedback_sdr = np.zeros(self.feedback_node.activations.size)

        self.cur_layer_input = np.zeros(self.feedforward_node.num_inputs)

        self.prev_output = np.zeros
        self.cur_layer_hidden_output = np.zeros(self.feedforward_node.sdr_size)
        self.prev_layer_hidden_output = np.zeros(self.feedforward_node.sdr_size)

    def generate_feedforward(self, inputs):
        self.cur_layer_input = inputs
        self.feedforward_sdr = self.feedforward_node.generate_node_output(inputs)
        error = self.feedforward_node.learn(inputs, self.feedforward_sdr)
        return self.feedforward_sdr, error

    def generate_recurrent(self, inputs):
        next_prev_recurrent_sdr = self.recurrent_node.generate_node_output(self.feedforward_sdr)
        error = self.recurrent_node.learn(inputs, self.prev_recurrent_sdr)
        self.prev_recurrent_sdr = next_prev_recurrent_sdr
        return next_prev_recurrent_sdr, error

    def generate_feedback(self, inputs):
        self.feedback_sdr = self.feedback_node.generate_node_output(inputs)
        error = self.feedback_node.learn(self.cur_layer_input, self.prev_feedback_sdr)
        self.prev_feedback_sdr = self.feedback_sdr
        return self.feedback_sdr, error

    def generate_output(self, inputs):
        result = self.feedforward_node.reconstruct(inputs)
        self.feedforward_node.learn(self.cur_layer_input, self.prev_layer_hidden_output)
        self.prev_layer_hidden_output = inputs
        return result

