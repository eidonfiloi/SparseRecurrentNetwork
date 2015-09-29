__author__ = 'ptoth'

import config.sr_network_configuration as base_config
import unittest
import numpy as np


class BaseTest(unittest.TestCase):

    def test_np(self):
        arr = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0])


        print np.matrix(arr2).T

        print np.dot(np.matrix(arr).T, np.matrix(arr2))

    def iterator_test(self):

        data = np.array([[[1, 2, 3], [4, 5, 6]], [[10, 20, 30], [40, 50, 60]]])
        print data.shape
        it = np.nditer(data, order='F')

        while not it.finished:
            print it
            it.iternext()

    def dropout_test(self):

        dropout_matrix = np.random.binomial(1, 0.5, 3)

        print dropout_matrix

