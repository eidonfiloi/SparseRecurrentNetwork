__author__ = 'ptoth'

import config.sr_network_configuration_test as base_config
import unittest
import numpy as np
from recurrent_network.Network import *


class SerializationTest(unittest.TestCase):

    def test_pickle(self):

        params = base_config.get_config()

        params['global']['epochs'] = 2

        network = SRNetwork(params['network'])

        constant = np.array([
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        ])

        inputs = constant

        for inp in inputs:
            output = network.run(inp)

        weights = network.layers[0].feedforward_node.weights

        with open('test_network_serialized.pickle', 'wb') as f:
            pickle.dump(network, f)

        # network.serialize('test_network_serialized.pickle')

        with open('test_network_serialized.pickle', "rb") as f:
            x = pickle.load(f)
            network_loaded = x

        print network_loaded

        self.assertTrue((weights[0] == network_loaded.layers[0].feedforward_node.weights[0]).all())



