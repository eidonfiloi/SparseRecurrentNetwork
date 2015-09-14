import abc
import logging

__author__ = 'ptoth'


class InputProvider(object):
    """ This class is an abstract base class for handling network inputs """

    __metaclass__ = abc.ABCMeta

    def __init__(self, parameters):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.name = parameters['name']

    @abc.abstractmethod
    def next(self):
        """

        :param inputs:
        :return:
        """


class SoundInputProvider(InputProvider):
    """

    """
    def __init__(self, parameters):
        super(SoundInputProvider, self).__init__(parameters)

    def next(self):
        pass