from copy import deepcopy
import abc
import logging
import importlib

from Node import *

__author__ = 'ptoth'


class Layer(object):

    """ This class is an abstract base class for layers """

    __metaclass__ = abc.ABCMeta

    def __init__(self, parameters):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.parameters = parameters
        self.name = self.parameters['name']
        self.verbose = self.parameters['verbose']

    def serialize(self):
        return {'parameters': self.parameters}


class SRLayer(Layer):

    """ Sparse Recurrent Layer containing feedforward, recurrent and feedback nodes """

    def __init__(self, parameters):
        super(SRLayer, self).__init__(parameters)

        module = importlib.import_module(Node.__module__)
        class_ = getattr(module, parameters['feedforward']['node_type'])
        self.feedforward_node = class_(parameters['feedforward']) \
            if parameters['feedforward'] is not None \
            else None
        class_ = getattr(module, parameters['recurrent']['node_type'])
        self.recurrent_node = class_(parameters['recurrent']) \
            if parameters['recurrent'] is not None \
            else None
        class_ = getattr(module, parameters['feedback']['node_type'])
        self.feedback_node = class_(parameters['feedback']) \
            if parameters['feedback'] is not None \
            else None

        self.repeat_factor = parameters['repeat_factor']

        self.feedforward_input = None
        self.feedforward_output = None
        self.feedforward_output_activations = None
        self.prev_feedforward_input = np.zeros(self.feedforward_node.inputs_size)
        self.prev_feedforward_output = np.zeros(self.feedforward_node.output_size)

        self.recurrent_input = None
        self.recurrent_output = None
        self.recurrent_output_activations = None
        self.prev_recurrent_input = np.random.rand(self.recurrent_node.inputs_size)
        self.prev_recurrent_output = np.random.rand(self.recurrent_node.activations.size)
        self.prev_recurrent_output_activations = np.random.rand(self.recurrent_node.activations.size)

        self.feedback_input = None
        self.feedback_output = None
        self.prev_feedback_input = np.zeros(self.feedback_node.inputs_size)
        self.prev_feedback_output = np.zeros(self.feedback_node.output_size)

    def serialize(self):

        serialized_object = super(SRLayer, self).serialize()

        serialized_object['feedforward_node'] = self.feedforward_node.serialize()
        serialized_object['recurrent_node'] = self.recurrent_node.serialize()
        serialized_object['feedback_node'] = self.feedback_node.serialize()

        return serialized_object

    def __getstate__(self):
        return {'feedforward_node': self.feedforward_node,
                'recurrent_node': self.recurrent_node,
                'feedback_node': self.feedback_node}

    def __setstate(self, dict):
        self.feedforward_node = dict['feedforward_node']
        self.recurrent_node = dict['recurrent_node']
        self.feedback_node = dict['feedback_node']

    def generate_feedforward(self, inputs, activations, learning_on=True):
        self.feedforward_input = activations
        for i in range(self.repeat_factor):
            self.feedforward_output = self.feedforward_node.generate_node_output(inputs)
            error = None
            if learning_on and self.parameters['feedforward']['node_type'] == "SRAutoEncoderNode":
                error = self.feedforward_node.learn_reconstruction(inputs,
                                                                   self.feedforward_output,
                                                                   backpropagate_hidden=True)
        self.feedforward_output_activations = self.feedforward_node.activations
        return self.feedforward_output, self.feedforward_node.activations, error

    def generate_recurrent(self, inputs, activations, learning_on=True):
        self.recurrent_input = activations
        error = None
        for i in range(self.repeat_factor):
            self.recurrent_output = self.recurrent_node.generate_node_output(inputs)
            if learning_on and self.parameters['recurrent']['node_type'] == "SRAutoEncoderNode":
                error = self.recurrent_node.learn_reconstruction(inputs,
                                                                 self.prev_recurrent_output,
                                                                 input_target=self.prev_recurrent_input,
                                                                 backpropagate_hidden=False)
        self.recurrent_output_activations = self.recurrent_node.activations
        return self.recurrent_output, error

    def generate_feedback(self, inputs, activations, learning_on=True):
        self.feedback_input = activations
        error = None
        for i in range(self.repeat_factor):
            self.feedback_output = self.feedback_node.generate_node_output(inputs)
            if learning_on and self.parameters['feedback']['node_type'] == "SRAutoEncoderNode":
                error = self.feedback_node.learn_reconstruction(inputs,
                                                                self.feedback_output,
                                                                backpropagate_hidden=True)
        return self.feedback_output, self.feedback_node.activations, error

    def backpropagate_feedback(self, delta):
        if self.verbose is not None and self.verbose > 0:
            self.logger.info('{0}-feedback mean delta: {1}'.format(self.name, np.mean(delta)))
        result = self.feedback_node.backpropagate(self.prev_feedback_input, delta)
        return result

    def backpropagate_recurrent(self, delta):
        if self.verbose is not None and self.verbose > 0:
            self.logger.info('{0}-recurrent mean delta: {1}'.format(self.name, np.mean(delta)))
        result = self.recurrent_node.backpropagate(self.prev_recurrent_input, delta)
        return result

    def backpropagate_feedforward(self, delta):
        if self.verbose is not None and self.verbose > 0:
            self.logger.info('{0}-feedforward mean delta: {1}'.format(self.name, np.mean(delta)))
        result_delta = self.feedforward_node.backpropagate(self.prev_feedforward_input, delta)
        return result_delta

    def cleanup_layer(self):

        self.prev_feedforward_input = deepcopy(self.feedforward_input)
        self.prev_feedforward_output = deepcopy(self.feedforward_output)

        self.prev_recurrent_output = deepcopy(self.recurrent_output)
        self.prev_recurrent_output_activations = deepcopy(self.recurrent_output_activations)
        self.prev_recurrent_input = deepcopy(self.recurrent_input)

        self.prev_feedback_input = deepcopy(self.feedback_input)
        self.prev_feedback_output = deepcopy(self.feedback_output)

    def update_layer_weights(self, num_iter):

        self.feedforward_node.update_weights(num_iter)
        self.recurrent_node.update_weights(num_iter)
        self.feedback_node.update_weights(num_iter)

    def collect_layer_deltas(self):
        feedforward_deltas = self.feedforward_node.collect_deltas()
        recurrent_deltas = self.recurrent_node.update_deltas()
        feedback_deltas = self.feedback_node.update_deltas()

        return [feedforward_deltas, recurrent_deltas, feedback_deltas]

    def update_layer_deltas(self):
        self.feedforward_node.update_deltas()
        self.recurrent_node.update_deltas()
        self.feedback_node.update_deltas()

