__author__ = 'ptoth'

from Node import *
import logging
import abc


class Layer(object):

    """ This class is an abstract base class for layers """

    __metaclass__ = abc.ABCMeta

    def __init__(self, parameters):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.name = parameters['name']


class SRLayer(Layer):

    """ Sparse Recurrent Layer contaning feedforward, recurrent and feedback nodes """

    def __init__(self, parameters):
        super(SRLayer, self).__init__(parameters)

        self.feedforward_node = SRAutoEncoderNode(parameters['feedforward']) if parameters['feedforward'] is not None \
            else None
        self.recurrent_node = SRAutoEncoderNode(parameters['recurrent']) if parameters['recurrent'] is not None \
            else None
        self.feedback_node = FeedForwardNode(parameters['feedback']) if parameters['feedback'] is not None \
            else None

        self.repeat_factor = parameters['repeat_factor']

        self.feedforward_output = None
        self.feedforward_output_activations = None
        self.feedforward_input = None
        self.recurrent_output = None
        self.recurrent_output_activations = None
        self.recurrent_input = None
        self.prev_recurrent_input = None
        self.prev_recurrent_output = None
        self.prev_feedback_input = None
        self.feedback_output = None
        self.feedback_input = None

    def generate_feedforward(self, inputs, activations):
        self.feedforward_input = activations
        self.feedforward_output = self.feedforward_node.generate_node_output(inputs)
        error = self.feedforward_node.learn_reconstruction(inputs, self.feedforward_output)
        if self.repeat_factor is not None:
            for i in range(self.repeat_factor - 1):
                self.feedforward_output = self.feedforward_node.generate_node_output(inputs)
                error = self.feedforward_node.learn_reconstruction(inputs, self.feedforward_output)
        self.feedforward_output_activations = self.feedforward_node.activations
        return self.feedforward_output, self.feedforward_node.activations, error

    def generate_recurrent(self, inputs, activations):
        self.recurrent_input = activations
        error = None
        if self.prev_recurrent_output is not None and self.feedforward_output is not None:
            error = self.recurrent_node.learn_reconstruction(self.feedforward_output, self.prev_recurrent_output)
        self.recurrent_output = self.recurrent_node.generate_node_output(inputs)
        self.prev_recurrent_output = self.recurrent_output
        self.recurrent_output_activations = self.recurrent_node.activations
        if self.prev_recurrent_input is None:
            self.prev_recurrent_input = self.recurrent_input
        return self.recurrent_output, error

    def generate_feedback(self, inputs, activations):
        self.feedback_input = activations
        self.feedback_output = self.feedback_node.generate_node_output(inputs)
        if self.prev_feedback_input is None:
            self.prev_feedback_input = self.feedback_input
        return self.feedback_output, self.feedback_node.activations

    def backpropagate_feedback(self, delta):
        self.logger.info('{0}-feedback mean delta: {1}'.format(self.name, np.mean(delta)))
        result = self.feedback_node.backpropagate(self.prev_feedback_input, delta)
        self.prev_feedback_input = self.feedback_input
        return result

    def backpropagate_recurrent(self, delta):
        self.logger.info('{0}-recurrent mean delta: {1}'.format(self.name, np.mean(delta)))
        result = self.recurrent_node.backpropagate(self.prev_recurrent_input, delta)
        self.prev_recurrent_input = self.recurrent_input
        return result

    def backpropagate_feedforward(self, delta):
        self.logger.info('{0}-feedforward mean delta: {1}'.format(self.name, np.mean(delta)))
        return self.feedforward_node.backpropagate(self.feedforward_input, delta)


class SRLayerOld(Layer):

    def __init__(self, parameters):
        super(SRLayerOld, self).__init__()

        self.feedforward_node = SRAutoEncoderOld(parameters['feedforward'])
        self.recurrent_node = SRAutoEncoderOld(parameters['recurrent'])
        self.feedback_node = SRAutoEncoderOld(parameters['feedback'])

        self.repeat_factor = parameters['repeat_factor']

        self.feedforward_sdr = np.zeros(self.feedforward_node.sdr_size)
        self.recurrent_sdr = np.zeros(self.recurrent_node.sdr_size)
        self.feedback_sdr = np.zeros(self.feedback_node.sdr_size)
        self.prev_recurrent_sdr = np.zeros(self.recurrent_node.sdr_size)
        self.prev_feedback_sdr = np.zeros(self.feedback_node.sdr_size)
        self.cur_layer_input = np.zeros(self.feedforward_node.num_inputs)
        self.prev_output = np.zeros
        self.cur_layer_hidden_output = np.zeros(self.feedforward_node.sdr_size)
        self.prev_layer_hidden_output = np.zeros(self.feedforward_node.sdr_size)

    def generate_feedforward(self, inputs):
        self.cur_layer_input = inputs
        error = None
        for i in range(self.repeat_factor):
            self.feedforward_sdr = self.feedforward_node.generate_node_output(inputs)
            error = self.feedforward_node.learn(inputs, self.feedforward_sdr)
        return self.feedforward_sdr, error

    def generate_recurrent(self, inputs, learning_on=True):
        next_prev_recurrent_sdr = None
        error = None
        for i in range(self.repeat_factor):
            next_prev_recurrent_sdr = self.recurrent_node.generate_node_output(self.feedforward_sdr)
            error = self.recurrent_node.learn(self.feedforward_sdr, self.prev_recurrent_sdr) \
                if learning_on else None
        self.prev_recurrent_sdr = next_prev_recurrent_sdr
        return next_prev_recurrent_sdr, error

    def generate_feedback(self, inputs, learning_on=True):
        error = None
        for i in range(self.repeat_factor):
            self.feedback_sdr = self.feedback_node.generate_node_output(inputs)
            error = self.feedback_node.learn(self.cur_layer_input, self.prev_feedback_sdr) \
                if learning_on else None
        self.prev_feedback_sdr = self.feedback_sdr
        return self.feedback_sdr, error

    def generate_output(self, inputs, learning_on=True):
        result = self.feedforward_node.reconstruct(inputs)
        output_error = self.feedforward_node.learn(self.cur_layer_input, self.prev_layer_hidden_output) \
            if learning_on else None
        self.prev_layer_hidden_output = inputs
        return result, output_error

