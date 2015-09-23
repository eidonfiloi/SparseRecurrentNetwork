__author__ = 'ptoth'


def get_config():

    params = {}

    params['global'] = {
        'epochs': 1
    }

    update_epochs = 2

    verbose = 1
    activation_function = "Sigmoid"
    loss_function = "CROSS_ENTROPY"
    activation_threshold = 0.5
    min_w = -1.0
    max_w = 1.0
    lifetime_sparsity = 0.014
    duty_cycle_decay = 0.006
    w_lr = 0.005
    inh_lr = 0.005
    b_lr = 0.005
    r_b_lr = 0.005
    learning_rate_increase = 0.001
    learning_rate_decrease = 0.995
    dropout_ratio = 0.2
    zoom = 0.4
    make_sparse = False
    target_sparsity = 0.1
    layer_repeat_factor = 1
    momentum = 0.5
    local_activation_radius = 0.2
    is_transpose_reconstruction = True
    regularization = 0.0

    layer1 = {
        'name': "layer1",
        'repeat_factor': layer_repeat_factor,
        'feedforward': {
            'name': "layer1-feedforward",
            'inputs_size': 512,
            'output_size': 256,
            'activation_function': activation_function,
            'activation_threshold': activation_threshold,
            'lifetime_sparsity': lifetime_sparsity,
            'min_weight': min_w,
            'max_weight': max_w,
            'dropout_ratio': 0.1,
            'momentum': momentum,
            'local_activation_radius': local_activation_radius,
            'zoom': zoom,
            'make_sparse': make_sparse,
            'target_sparsity': target_sparsity,
            'duty_cycle_decay': duty_cycle_decay,
            'learning_rate_increase': learning_rate_increase,
            'learning_rate_decrease': learning_rate_decrease,
            'is_transpose_reconstruction': is_transpose_reconstruction,
            'regularization': regularization,
            'weights_lr': w_lr,
            'inhibition_lr': inh_lr,
            'bias_lr': b_lr,
            'recon_bias_lr': r_b_lr
        },
        'recurrent': {
            'name': "layer1-recurrent",
            'inputs_size': 512,
            'output_size': 256,
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
            'learning_rate_increase': learning_rate_increase,
            'learning_rate_decrease': learning_rate_decrease,
            'is_transpose_reconstruction': False,
            'regularization': regularization,
            'weights_lr': w_lr,
            'inhibition_lr': inh_lr,
            'bias_lr': b_lr,
            'recon_bias_lr': r_b_lr
        },
        'feedback': {
            'name': "layer1-feedback",
            'inputs_size': 512,
            'output_size': 256,
            'activation_function': "SoftMax",
            'activation_threshold': activation_threshold,
            'lifetime_sparsity': lifetime_sparsity,
            'min_weight': min_w,
            'max_weight': max_w,
            'dropout_ratio': dropout_ratio,
            'momentum': momentum,
            'local_activation_radius': local_activation_radius,
            'zoom': zoom,
            'make_sparse': False,
            'target_sparsity': target_sparsity,
            'duty_cycle_decay': duty_cycle_decay,
            'learning_rate_increase': learning_rate_increase,
            'learning_rate_decrease': learning_rate_decrease,
            'is_transpose_reconstruction': False,
            'regularization': regularization,
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
            'inputs_size': 512,
            'output_size': 128,
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
            'learning_rate_increase': learning_rate_increase/2,
            'learning_rate_decrease': learning_rate_decrease,
            'is_transpose_reconstruction': is_transpose_reconstruction,
            'regularization': regularization,
            'weights_lr': w_lr/2,
            'inhibition_lr': inh_lr/2,
            'bias_lr': b_lr/2,
            'recon_bias_lr': r_b_lr/2
        },
        'recurrent': {
            'name': "layer2-recurrent",
            'inputs_size': 256,
            'output_size': 128,
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
            'learning_rate_increase': learning_rate_increase,
            'learning_rate_decrease': learning_rate_decrease,
            'is_transpose_reconstruction': False,
            'regularization': regularization,
            'weights_lr': w_lr,
            'inhibition_lr': inh_lr,
            'bias_lr': b_lr,
            'recon_bias_lr': r_b_lr
        },
        'feedback': {
            'name': "layer2-feedback",
            'inputs_size': 384,
            'output_size': 256,
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
            'learning_rate_increase': learning_rate_increase/2,
            'learning_rate_decrease': learning_rate_decrease,
            'is_transpose_reconstruction': is_transpose_reconstruction,
            'regularization': regularization,
            'weights_lr': w_lr/2,
            'inhibition_lr': inh_lr/2,
            'bias_lr': b_lr/2,
            'recon_bias_lr': r_b_lr/2
        }
    }

    layer3 = {
        'name': "layer3",
        'repeat_factor': 1,
        'feedforward': {
            'name': "layer3-feedforward",
            'inputs_size': 256,
            'output_size': 256,
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
            'learning_rate_increase': learning_rate_increase/5,
            'learning_rate_decrease': learning_rate_decrease,
            'is_transpose_reconstruction': is_transpose_reconstruction,
            'regularization': regularization,
            'weights_lr': w_lr/5,
            'inhibition_lr': inh_lr/5,
            'bias_lr': b_lr/5,
            'recon_bias_lr': r_b_lr/5
        },
        'recurrent': {
            'name': "layer3-recurrent",
            'inputs_size': 384,
            'output_size': 128,
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
            'learning_rate_increase': learning_rate_increase,
            'learning_rate_decrease': learning_rate_decrease,
            'is_transpose_reconstruction': False,
            'regularization': regularization,
            'weights_lr': w_lr,
            'inhibition_lr': inh_lr,
            'bias_lr': b_lr,
            'recon_bias_lr': r_b_lr
        },
        'feedback': {
            'name': "layer3-feedback",
            'inputs_size': 128,
            'output_size': 256,
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
            'learning_rate_increase': learning_rate_increase,
            'learning_rate_decrease': learning_rate_decrease/5,
            'is_transpose_reconstruction': is_transpose_reconstruction,
            'regularization': regularization,
            'weights_lr': w_lr/5,
            'inhibition_lr': inh_lr/5,
            'bias_lr': b_lr/5,
            'recon_bias_lr': r_b_lr/5
        }
    }

    params['network'] = {
        'name': "alpha_network",
        'verbose': verbose,
        'serialize': False,
        'serialize_path': 'serialized_models',
        'activation_function': "SoftMax",
        'loss_function': loss_function,
        'visualize_states': False,
        'update_epochs': update_epochs,
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
