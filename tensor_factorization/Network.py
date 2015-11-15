import abc
import tensorflow as tf

__author__ = 'ptoth'


class Network(object):
    """

    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, parameters):
        self.parameters = parameters

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

    def run(self, sess, inputs):
        pass