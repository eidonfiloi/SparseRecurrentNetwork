__author__ = 'ptoth'


def get_config():

    params = {}

    params['global'] = {
        'epochs': 500
    }

    verbose = 1
    activation_function = "Rectifier"
    min_w = -1.0
    max_w = 1.0
    sparsity = 0.12
    duty_cycle_decay = 0.005
    w_lr = 0.001
    inh_lr = 0.001
    b_lr = 0.001

    layer1 = {
        'name': "layer1",
        'feedforward': {
            'name': "layer1-feedforward",
            'num_inputs': 20,
            'sdr_size': 8,
            'activation': activation_function,
            'sparsity': sparsity,
            'min_weight': min_w,
            'max_weight': max_w,
            'duty_cycle_decay': duty_cycle_decay,
            'weights_lr': w_lr,
            'inhibition_lr': inh_lr,
            'bias_lr': b_lr
        },
        'recurrent': {
            'name': "layer1-recurrent",
            'num_inputs': 8,
            'sdr_size': 8,
            'activation': activation_function,
            'sparsity': sparsity,
            'min_weight': min_w,
            'max_weight': max_w,
            'duty_cycle_decay': duty_cycle_decay,
            'weights_lr': w_lr,
            'inhibition_lr': inh_lr,
            'bias_lr': b_lr
        },
        'feedback': {
            'name': "layer1-feedback",
            'num_inputs': 8,
            'sdr_size': 8,
            'activation': activation_function,
            'sparsity': sparsity,
            'min_weight': min_w,
            'max_weight': max_w,
            'duty_cycle_decay': duty_cycle_decay,
            'weights_lr': w_lr,
            'inhibition_lr': inh_lr,
            'bias_lr': b_lr
        }
    }

    layer2 = {
        'name': "layer2",
        'feedforward': {
            'name': "layer2-feedforward",
            'num_inputs': 8,
            'sdr_size': 8,
            'activation': activation_function,
            'sparsity': sparsity,
            'min_weight': min_w,
            'max_weight': max_w,
            'duty_cycle_decay': duty_cycle_decay,
            'weights_lr': w_lr,
            'inhibition_lr': inh_lr,
            'bias_lr': b_lr
        },
        'recurrent': {
            'name': "layer2-recurrent",
            'num_inputs': 8,
            'sdr_size': 8,
            'activation': activation_function,
            'sparsity': sparsity,
            'min_weight': min_w,
            'max_weight': max_w,
            'duty_cycle_decay': duty_cycle_decay,
            'weights_lr': w_lr,
            'inhibition_lr': inh_lr,
            'bias_lr': b_lr
        },
        'feedback': {
            'name': "layer2-feedback",
            'num_inputs': 8,
            'sdr_size': 8,
            'activation': activation_function,
            'sparsity': sparsity,
            'min_weight': min_w,
            'max_weight': max_w,
            'duty_cycle_decay': duty_cycle_decay,
            'weights_lr': w_lr,
            'inhibition_lr': inh_lr,
            'bias_lr': b_lr
        }
    }

    params['network'] = {
        'name': "test network",
        'verbose': verbose,
        'visualize_grid_size': 4,
        'input': {

        },
        'layers': [
            layer1
            , layer2
        ],
        'output': {

        }
    }

    return params
