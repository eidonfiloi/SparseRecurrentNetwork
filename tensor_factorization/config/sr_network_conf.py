__author__ = 'ptoth'


def get_config():

    params = dict()

    params['global'] = {
        'epochs': 1
    }

    learning_rate = 0.01

    layer1 = {
        'name': "layer1",
        'inputs_shape': 8820,
        'sp_hidden_shape': 1024,
        'rec_hidden_shape': 1024,
        'learning_rate': learning_rate
    }

    layer2 = {
        'name': "layer2",
        'inputs_shape': 1024,
        'sp_hidden_shape': 1024,
        'rec_hidden_shape': 1024,
        'learning_rate': learning_rate
    }

    params['network'] = {
        'name': "SRNetwork",
        'layers_output_shape': 1024,
        'output_shape': 8820,
        'learning_rate': learning_rate,
        'loss_function': 'MSE',
        'layers': [
            layer1
            , layer2
        ]

    }

    return params
