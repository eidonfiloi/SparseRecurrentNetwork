__author__ = 'ptoth'


def get_config():

    params = {}

    params['global'] = {
        'epochs': 10
    }

    verbose = 1
    activation_function = "Sigmoid"
    min_w = -1.0
    max_w = 1.0
    lifetime_sparsity = 0.012
    duty_cycle_decay = 0.005
    w_lr = 0.005
    inh_lr = 0.01
    b_lr = 0.005
    r_b_lr = 0.005
    dropout = None
    sparsify = True
    target_sparsity = 0.3
    zoom = None
    layer_repeat_factor = 10

    layer1 = {
        'name': "layer1",
        'repeat_factor': 100,
        'feedforward': {
            'name': "layer1-feedforward",
            'inputs_size': 9,
            'output_size': 8,
            'activation': activation_function,
            'lifetime_sparsity': lifetime_sparsity,
            'min_weight': min_w,
            'max_weight': max_w,
            'dropout': dropout,
            'zoom': zoom,
            'sparsify': True,
            'target_sparsity': target_sparsity,
            'duty_cycle_decay': duty_cycle_decay,
            'weights_lr': w_lr,
            'inhibition_lr': inh_lr,
            'bias_lr': b_lr,
            'recon_bias_lr': r_b_lr
        },
        'recurrent': {
            'name': "layer1-recurrent",
            'num_inputs': 8,
            'output': 8,
            'activation': activation_function,
            'lifetime_sparsity': lifetime_sparsity,
            'min_weight': min_w,
            'max_weight': max_w,
            'dropout': dropout,
            'zoom': zoom,
            'sparsify': False,
            'target_sparsity': target_sparsity,
            'duty_cycle_decay': duty_cycle_decay,
            'weights_lr': w_lr,
            'inhibition_lr': inh_lr,
            'bias_lr': b_lr,
            'recon_bias_lr': r_b_lr
        },
        'feedback': {
            'name': "layer1-feedback",
            'num_inputs': 8,
            'output_size': 8,
            'activation': activation_function,
            'lifetime_sparsity': lifetime_sparsity,
            'min_weight': min_w,
            'max_weight': max_w,
            'dropout': dropout,
            'zoom': zoom,
            'sparsify': False,
            'target_sparsity': target_sparsity,
            'duty_cycle_decay': duty_cycle_decay,
            'weights_lr': w_lr,
            'inhibition_lr': inh_lr,
            'bias_lr': b_lr,
            'recon_bias_lr': r_b_lr
        }
    }

    layer2 = {
        'name': "layer2",
        'repeat_factor': layer_repeat_factor,
        'feedforward': {
            'name': "layer2-feedforward",
            'num_inputs': 8,
            'output_size': 8,
            'activation': activation_function,
            'lifetime_sparsity': lifetime_sparsity,
            'min_weight': min_w,
            'max_weight': max_w,
            'dropout': dropout,
            'zoom': zoom,
            'sparsify': sparsify,
            'target_sparsity': target_sparsity,
            'duty_cycle_decay': duty_cycle_decay,
            'weights_lr': w_lr,
            'inhibition_lr': inh_lr,
            'bias_lr': b_lr,
            'recon_bias_lr': r_b_lr
        },
        'recurrent': {
            'name': "layer2-recurrent",
            'num_inputs': 8,
            'output_size': 8,
            'activation': activation_function,
            'lifetime_sparsity': lifetime_sparsity,
            'min_weight': min_w,
            'max_weight': max_w,
            'dropout': dropout,
            'zoom': zoom,
            'sparsify': sparsify,
            'target_sparsity': target_sparsity,
            'duty_cycle_decay': duty_cycle_decay,
            'weights_lr': w_lr,
            'inhibition_lr': inh_lr,
            'bias_lr': b_lr,
            'recon_bias_lr': r_b_lr
        },
        'feedback': {
            'name': "layer2-feedback",
            'num_inputs': 8,
            'output_size': 8,
            'activation': activation_function,
            'lifetime_sparsity': lifetime_sparsity,
            'min_weight': min_w,
            'max_weight': max_w,
            'dropout': dropout,
            'zoom': zoom,
            'sparsify': sparsify,
            'target_sparsity': target_sparsity,
            'duty_cycle_decay': duty_cycle_decay,
            'weights_lr': w_lr,
            'inhibition_lr': inh_lr,
            'bias_lr': b_lr,
            'recon_bias_lr': r_b_lr
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
