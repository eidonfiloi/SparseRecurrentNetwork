__author__ = 'ptoth'

import config.sr_network_configuration as base_config
import unittest
import numpy as np


class BaseTest(unittest.TestCase):

    def test_np(self):
        arr = np.array([1.0, 2.0, 3.0])

        print arr * (1.0 - arr)
