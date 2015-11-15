import abc
import tensorflow as tf
from Layer import *

__author__ = 'ptoth'


class Network(object):
    """

    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, parameters):
        self.parameters = parameters
        self.name = self.parameters['name']
        self.layers = [Layer(layer_conf) for layer_conf in self.parameters['layers']]
        self.loss_function = self.parameters['loss_function']


    @abc.abstractmethod
    def run(self, sess, inputs):
        """

        :param sess:
        :param inputs:
        :return:
        """


class SRNetwork(Network):
    """

    """

    def __init__(self, parameters):
        super(SRNetwork, self).__init__(parameters)
        self.layers = [SRLayer(layer_conf) for layer_conf in self.parameters['layers']]
        self.output_shape = self.parameters['output_shape']
        self.learning_rate = self.parameters['learning_rate']

        # example shape [None, 8820]
        self.output = tf.placeholder(tf.float32, shape=self.output_shape, name="{0}/output".format(self.name))
        self.target = tf.placeholder(tf.float32, shape=self.output_shape, name="{0}/target".format(self.name))

        # output LOSS
        self.output_loss = tf.reduce_mean(tf.square(self.output - self.target), name="{0}/output_loss".format(self.name))
        tf.scalar_summary(self.output_loss.op.name, self.output_loss)

        # TensorFlow OP for train output optimization
        self.output_train_op = tf.train\
            .GradientDescentOptimizer(self.learning_rate)\
            .minimize(self.output_loss)

    def run(self, sess, inputs, targets=None):

        current_input = None
        prev_input = None
        for sample in inputs:
            for layer in self.layers:
                layer_output = layer.run()

