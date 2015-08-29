__author__ = 'ptoth'

import config.sr_network_configuration_test as base_config
import unittest
from recurrent_network.SRNetwork import *
import numpy as np
import matplotlib.pyplot as plt


class SRNetworkTest(unittest.TestCase):

    def test_init(self):
        logging.basicConfig(level=logging.INFO)
        config = base_config.get_config()
        network = SRNetwork(config['network'])

        input_series = np.array([
            [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
            [-1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
            [-1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1],
            [-1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1],
            [-1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1]
        ])

        feedforward_errors = {layer['name']: [] for layer in config['network']['layers']}
        recurrent_errors = {layer['name']: [] for layer in config['network']['layers']}
        feedback_errors = {layer['name']: [] for layer in config['network']['layers']}

        prev_output = None
        for ep in range(config['global']['epochs']):
            for ind, inp in enumerate(input_series):
                output, forward_err, rec_err, back_err = network.run(inp)
                if prev_output is not None:
                    prev_output_bin = np.zeros(prev_output.size).astype('int')
                    for i in range(0, prev_output.size):
                        if prev_output[i] > 0.0:
                            prev_output_bin[i] = 1
                        else:
                            prev_output_bin[i] = -1
                    print '############### epoch: {0}\n' \
                          '############### sequence: {1}\n' \
                          '############### input: \n' \
                          '{2}\n' \
                          '############### prev_output_bin: \n' \
                          '{3}\n' \
                          '############### prev_output: \n' \
                          '{4}'.format(ep, ind, inp, prev_output_bin, prev_output)
                prev_output = output
                for key, v in forward_err.items():
                    if len(v) > 0:
                        feedforward_errors[key].append(v[0])
                for key, v in rec_err.items():
                    if len(v) > 0:
                        recurrent_errors[key].append(v[0])
                for key, v in back_err.items():
                    if len(v) > 0:
                        feedback_errors[key].append(v[0])

        plt.subplot(3, 1, 1)
        for k, v in feedforward_errors.items():
            if len(v) > 0:
                plt.plot(range(len(v)), v, label=k)
        plt.xlabel('epochs')
        plt.ylabel('feedforward_errors')
        # plt.title('feedforward_errors')
        plt.legend(loc=1)

        plt.subplot(3, 1, 2)
        for k, v in recurrent_errors.items():
            if len(v) > 0:
                plt.plot(range(len(v)), v, label=k)
        plt.xlabel('epochs')
        plt.ylabel('recurrent_errors')
        # plt.title('recurrent_errors')
        plt.legend(loc=1)

        plt.subplot(3, 1, 3)
        for k, v in feedback_errors.items():
            if len(v) > 0:
                plt.plot(range(len(v)), v, label=k)
        plt.xlabel('epochs')
        plt.ylabel('feedback_errors')
        # plt.title('feedback_errors')
        plt.legend(loc=1)

        plt.show()