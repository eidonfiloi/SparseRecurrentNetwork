__author__ = 'ptoth'


def get_config():

    params = {}

    params['global'] = {
        'epochs': 10
    }

    verbose = 2
    activation_function = "Sigmoid"
    min_w = -1.0
    max_w = 1.0
    sparsity = 0.1
    duty_cycle_decay = 0.005
    w_lr = 0.005
    inh_lr = 0.001
    b_lr = 0.005

    layer1 = {
        'name': "layer1",
        'feedforward': {
            'name': "layer1-feedforward",
            'num_inputs': 44100,
            'sdr_size': 1024,
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
            'num_inputs': 1024,
            'sdr_size': 1024,
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
            'num_inputs': 1024,
            'sdr_size': 1024,
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
            'num_inputs': 1024,
            'sdr_size': 1024,
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
            'num_inputs': 1024,
            'sdr_size': 1024,
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
            'num_inputs': 1024,
            'sdr_size': 1024,
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

    layer3 = {
        'name': "layer3",
        'feedforward': {
            'name': "layer3-feedforward",
            'num_inputs': 1024,
            'sdr_size': 1024,
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
            'name': "layer3-recurrent",
            'num_inputs': 1024,
            'sdr_size': 1024,
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
            'name': "layer3-feedback",
            'num_inputs': 1024,
            'sdr_size': 1024,
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
        'name': "Bach network",
        'verbose': verbose,
        'visualize_grid_size': 32,
        'input': {

        },
        'layers': [
            layer1
            , layer2
            , layer3
        ],
        'output': {

        }
    }

    return params
