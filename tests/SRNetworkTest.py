__author__ = 'ptoth'

import config.sr_network_configuration_test as base_config
import config.sym_network_config_test as base_config
import unittest
from recurrent_network.Network import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


class SRNetworkTest(unittest.TestCase):

    def test_init(self):
        logging.basicConfig(level=logging.INFO)
        config = base_config.get_config()
        # network = SRNetwork(config['network'])
        network = SymmetricNetwork(config['network'])

        long_sequence = np.array([
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        ])

        constant = np.array([
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        ])

        simple_sequence = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        ])

        input_series = simple_sequence #simple_sequence #long_sequence #constant

        sd_input_series = preprocessing.scale(input_series)

        feedforward_errors = {layer['name']: [] for layer in config['network']['layers']}
        recurrent_errors = {layer['name']: [] for layer in config['network']['layers']}
        feedback_errors = {layer['name']: [] for layer in config['network']['layers']}
        output_mse = []

        prev_output = None
        for ep in range(config['global']['epochs']):
            for ind, inp in enumerate(input_series):
                output, mse = network.run(inp)
                output_mse.append(mse)
                if prev_output is not None:
                    prev_output_bin = np.zeros(prev_output.size).astype('int')
                    for i in range(0, prev_output.size):
                        if prev_output[i] > 0.5:
                            prev_output_bin[i] = 1
                        else:
                            prev_output_bin[i] = 0
                    mod_input = np.zeros(inp.size).astype('int')
                    for i in range(0, inp.size):
                        if inp[i] >= 0.5:
                            mod_input[i] = 1

                    print '############### epoch: {0}\n' \
                          '############### sequence: {1}\n' \
                          '############### input: \n' \
                          '{2}\n' \
                          '############### prev_output_bin: \n' \
                          '{3}\n' \
                          '############### prev_output: \n' \
                          '{4}\n' \
                          'output_error: {5}'.format(ep, ind, mod_input, prev_output_bin, prev_output, mse)
                    plt.ion()
                    plt.axis([-1, 4, -1, 4])
                    x_r, y_r = np.argwhere(mod_input.reshape(4, 4) == 1).T
                    x_t, y_t = np.argwhere(prev_output_bin.reshape(4, 4) == 1).T
                    plt.scatter(x_r, y_r, alpha=0.5, c='r', marker='s', s=150)
                    plt.scatter(x_t, y_t, alpha=0.5, c='b', marker='o', s=130)
                    plt.draw()
                    time.sleep(0.1)
                    plt.clf()
                prev_output = output
                for key, v in network.feedforward_errors.items():
                    if len(v) > 0:
                        feedforward_errors[key].append(v[0])
                for key, v in network.recurrent_errors.items():
                    if len(v) > 0:
                        recurrent_errors[key].append(v[0])
                for key, v in network.feedback_errors.items():
                    if len(v) > 0:
                        feedback_errors[key].append(v[0])

        plt.ioff()
        plt.subplot(4, 1, 1)
        for k, v in feedforward_errors.items():
            if len(v) > 0:
                plt.plot(range(len(v)), v, label=k)
        plt.xlabel('epochs')
        plt.ylabel('feedforward_errors')
        # plt.title('feedforward_errors')
        plt.legend(loc=1)

        plt.subplot(4, 1, 2)
        for k, v in recurrent_errors.items():
            if len(v) > 0:
                plt.plot(range(len(v)), v, label=k)
        plt.xlabel('epochs')
        plt.ylabel('recurrent_errors')
        # plt.title('recurrent_errors')
        plt.legend(loc=1)

        plt.subplot(4, 1, 3)
        for k, v in feedback_errors.items():
            if len(v) > 0:
                plt.plot(range(len(v)), v, label=k)
        plt.xlabel('epochs')
        plt.ylabel('feedback_errors')
        # plt.title('recurrent_errors')
        plt.legend(loc=1)

        plt.subplot(4, 1, 4)
        plt.plot(range(len(output_mse)), output_mse, label="mse")
        plt.xlabel('epochs')
        plt.ylabel('output_mse')
        # plt.title('recurrent_errors')
        plt.legend(loc=1)

        plt.show()