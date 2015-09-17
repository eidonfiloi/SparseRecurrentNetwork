__author__ = 'ptoth'

import config.sr_network_configuration as base_config
import unittest
import numpy as np
from recurrent_network.Network import *


class SerializationTest(unittest.TestCase):

    def test_pickle(self):

        params = {}

        params['global'] = {
            'epochs': 200
        }

        verbose = 1
        activation_function = "Sigmoid"
        activation_threshold = 0.5
        min_w = -1.0
        max_w = 1.0
        lifetime_sparsity = 0.014
        duty_cycle_decay = 0.006
        w_lr = 0.05
        inh_lr = 0.05
        b_lr = 0.05
        r_b_lr = 0.05
        learning_rate_decay = 0.01
        dropout_ratio = None
        momentum = 0.9
        zoom = 0.4
        make_sparse = False
        target_sparsity = 0.1
        layer_repeat_factor = 10
        local_activation_radius = None
        is_transpose_reconstruction = True

        layer1 = {
            'name': "layer1",
            'repeat_factor': 1,
            'feedforward': {
                'name': "layer1-feedforward",
                'inputs_size': 2,
                'output_size': 2,
                'activation_function': activation_function,
                'activation_threshold': activation_threshold,
                'lifetime_sparsity': lifetime_sparsity,
                'min_weight': min_w,
                'max_weight': max_w,
                'dropout_ratio': dropout_ratio,
                'momentum': momentum,
                'local_activation_radius': local_activation_radius,
                'zoom': zoom,
                'make_sparse': make_sparse,
                'target_sparsity': target_sparsity,
                'duty_cycle_decay': duty_cycle_decay,
                'learning_rate_decay': learning_rate_decay,
                'is_transpose_reconstruction': is_transpose_reconstruction,
                'weights_lr': w_lr,
                'inhibition_lr': inh_lr,
                'bias_lr': b_lr,
                'recon_bias_lr': r_b_lr
            },
            'recurrent': {
                'name': "layer1-recurrent",
                'inputs_size': 2,
                'output_size': 2,
                'activation_function': activation_function,
                'activation_threshold': activation_threshold,
                'lifetime_sparsity': lifetime_sparsity,
                'min_weight': min_w,
                'max_weight': max_w,
                'dropout_ratio': dropout_ratio,
                'momentum': momentum,
                'local_activation_radius': local_activation_radius,
                'zoom': zoom,
                'make_sparse': make_sparse,
                'target_sparsity': target_sparsity,
                'duty_cycle_decay': duty_cycle_decay,
                'learning_rate_decay': learning_rate_decay,
                'is_transpose_reconstruction': False,
                'weights_lr': w_lr,
                'inhibition_lr': inh_lr,
                'bias_lr': b_lr,
                'recon_bias_lr': r_b_lr
            },
            'feedback': {
                'name': "layer1-feedback",
                'inputs_size': 4,
                'output_size': 2,
                'activation_function': activation_function,
                'activation_threshold': activation_threshold,
                'lifetime_sparsity': lifetime_sparsity,
                'min_weight': min_w,
                'max_weight': max_w,
                'dropout_ratio': dropout_ratio,
                'momentum': momentum,
                'local_activation_radius': local_activation_radius,
                'zoom': zoom,
                'make_sparse': make_sparse,
                'target_sparsity': target_sparsity,
                'duty_cycle_decay': duty_cycle_decay,
                'learning_rate_decay': learning_rate_decay,
                'is_transpose_reconstruction': is_transpose_reconstruction,
                'weights_lr': w_lr,
                'inhibition_lr': inh_lr,
                'bias_lr': b_lr,
                'recon_bias_lr': r_b_lr
            }
        }

        layer2 = {
            'name': "layer2",
            'repeat_factor': 1,
            'feedforward': {
                'name': "layer2-feedforward",
                'inputs_size': 4,
                'output_size': 2,
                'activation_function': activation_function,
                'activation_threshold': activation_threshold,
                'lifetime_sparsity': lifetime_sparsity,
                'min_weight': min_w,
                'max_weight': max_w,
                'dropout_ratio': dropout_ratio,
                'momentum': momentum,
                'local_activation_radius': local_activation_radius,
                'zoom': zoom,
                'make_sparse': make_sparse,
                'target_sparsity': target_sparsity,
                'duty_cycle_decay': duty_cycle_decay,
                'learning_rate_decay': learning_rate_decay,
                'is_transpose_reconstruction': is_transpose_reconstruction,
                'weights_lr': w_lr,
                'inhibition_lr': inh_lr,
                'bias_lr': b_lr,
                'recon_bias_lr': r_b_lr
            },
            'recurrent': {
                'name': "layer2-recurrent",
                'inputs_size': 2,
                'output_size': 2,
                'activation_function': activation_function,
                'activation_threshold': activation_threshold,
                'lifetime_sparsity': lifetime_sparsity,
                'min_weight': min_w,
                'max_weight': max_w,
                'dropout_ratio': dropout_ratio,
                'momentum': momentum,
                'local_activation_radius': local_activation_radius,
                'zoom': zoom,
                'make_sparse': make_sparse,
                'target_sparsity': target_sparsity,
                'duty_cycle_decay': duty_cycle_decay,
                'learning_rate_decay': learning_rate_decay,
                'is_transpose_reconstruction': False,
                'weights_lr': w_lr,
                'inhibition_lr': inh_lr,
                'bias_lr': b_lr,
                'recon_bias_lr': r_b_lr
            },
            'feedback': {
                'name': "layer2-feedback",
                'inputs_size': 2,
                'output_size': 2,
                'activation_function': activation_function,
                'activation_threshold': activation_threshold,
                'lifetime_sparsity': lifetime_sparsity,
                'min_weight': min_w,
                'max_weight': max_w,
                'dropout_ratio': dropout_ratio,
                'momentum': momentum,
                'local_activation_radius': local_activation_radius,
                'zoom': zoom,
                'make_sparse': make_sparse,
                'target_sparsity': target_sparsity,
                'duty_cycle_decay': duty_cycle_decay,
                'learning_rate_decay': learning_rate_decay,
                'is_transpose_reconstruction': is_transpose_reconstruction,
                'weights_lr': w_lr,
                'inhibition_lr': inh_lr,
                'bias_lr': b_lr,
                'recon_bias_lr': r_b_lr
            }
        }

        params['network'] = {
            'name': "test network",
            'verbose': verbose,
            'activation_function': activation_function,
            'visualize_states': False,
            'input': {

            },
            'layers': [
                layer1
                , layer2
            ],
            'output': {

            }
        }

        network = SRNetwork(params['network'])

        input = np.array([0, 1])

        output = network.run(input)

        weights = network.layers[0].feedforward_node.weights

        network.serialize('test_network_serialized.pickle')
        #pickle.dump(network, f)

        network_loaded = None

        with open('test_network_serialized.pickle', "rb") as f:
            network_loaded = pickle.load(f)

        print network_loaded.name

        self.assertTrue((weights[0] == network_loaded.layers[0].feedforward_node.weights[0]).all())



