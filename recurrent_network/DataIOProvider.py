import abc
import logging
import numpy as np

__author__ = 'ptoth'


class DataIOProvider(object):
    """ This class is an abstract base class for handling network inputs """

    __metaclass__ = abc.ABCMeta

    def __init__(self, parameters):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.name = parameters['name']

    @abc.abstractmethod
    def prepare(self):
        """

        :return:
        """

    @abc.abstractmethod
    def post_process(self):
        """

        :return:
        """

    @abc.abstractmethod
    def next_input(self):
        """

        :param inputs:
        :return:
        """


class SoundInputProvider(DataIOProvider):
    """

    """

    def __init__(self, parameters):
        super(SoundInputProvider, self).__init__(parameters)
        self.input_file = parameters['input_file']
        self.x_train = np.load(self.input_file + '_x.npy')
        self.y_train = np.load(self.input_file + '_y.npy')
        self.x_mean = np.load(self.input_file + '_mean.npy')
        self.x_var = np.load(self.input_file + '_var.npy')

        self.num_train_examples = self.x_train.shape[0]
        self.num_time_steps = self.x_train.shape[1]
        self.num_freq_dims = self.x_train.shape[2]

        self.ind_train_examples = 0
        self.ind_time_steps = 0

    def prepare(self):
        pass

    def post_process(self):
        pass

    def next_input(self):
        self.ind_time_steps += 1
        if self.ind_train_examples < self.num_train_examples:
            if self.ind_time_steps < self.num_time_steps:
                return self.x_train[self.ind_train_examples][self.ind_time_steps]
            else:
                self.ind_train_examples += 1
                self.ind_train_examples += 1


