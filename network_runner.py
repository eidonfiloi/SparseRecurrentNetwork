__author__ = 'eidonfiloi'

import logging
import matplotlib.pyplot as plt
from recurrent_network.Network import *
import config.sr_network_configuration as base_config
from data_io.audio_data_utils import *

_LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

     # Load up the training data
    _LOGGER.info('Loading training data')
    input_file = 'data_prepared/test_bach26_freq4'
    # X_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
    X_train_freq = np.load(input_file + '_x.npy')
    # y_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
    y_train_freq = np.load(input_file + '_y.npy')
    # X_mean is a matrix of size (num_frequency_dims,) containing the mean for each frequency dimension
    X_mean_freq = np.load(input_file + '_mean.npy')
    # X_var is a matrix of size (num_frequency_dims,) containing the variance for each frequency dimension
    X_var_freq = np.load(input_file + '_var.npy')
    _LOGGER.info('Finished loading training data')

    config = base_config.get_config()

    network = SRNetwork(config['network'])

    input_sample = X_train_freq[2]
    max_value = np.max(input_sample)
    input_sample /= max_value
    input_sample = (input_sample + 1.0) / 2.0
    input_sample_y = y_train_freq[2]

    epochs = config['global']['epochs']

    output = []

    feedforward_errors = {layer['name']: [] for layer in config['network']['layers']}
    recurrent_errors = {layer['name']: [] for layer in config['network']['layers']}
    output_mse = []
    current_j = 0
    prev_j = -1
    for j in range(epochs):
        for i in range(input_sample.shape[0]):
            #for k in range(X_train.shape[0]):
            #for i in range(X_train.shape[1]):
            input_ = input_sample[i]#X_train[k][i]
            _LOGGER.info(
                '\n############## epoch {0}\n'
                '############## sequence {1}\n'
                '############## sample {2}\n '
                '############## input min is {3}, max is {4}'.format(j, 1, i, np.min(input_), np.max(input_)))
            if i == 0:
                network_output, mse = network.run(input_, learning_on=False)
            else:
                network_output, mse = network.run(input_, learning_on=True)
            for key, v in network.feedforward_errors.items():
                if len(v) > 0:
                    feedforward_errors[key].append(v[0])
            for key, v in network.recurrent_errors.items():
                if len(v) > 0:
                    recurrent_errors[key].append(v[0])
            _LOGGER.info('output length {0}\n{1}'.format(len(network_output), network_output[0:20]))
            network_output = 2.0*network_output - 1.0
            output.append(network_output * max_value)

    for i in xrange(len(output)):
        output[i] *= X_var_freq
        output[i] += X_mean_freq
    save_generated_example("bach_output.wav", output, useTimeDomain=False)

    plt.ioff()
    plt.subplot(2, 1, 1)
    for k, v in feedforward_errors.items():
        if len(v) > 0:
            plt.plot(range(len(v)), v, label=k)
    plt.xlabel('epochs')
    plt.ylabel('feedforward_errors')
    # plt.title('feedforward_errors')
    plt.legend(loc=1)

    plt.subplot(2, 1, 2)
    for k, v in recurrent_errors.items():
        if len(v) > 0:
            plt.plot(range(len(v)), v, label=k)
    plt.xlabel('epochs')
    plt.ylabel('recurrent_errors')
    # plt.title('recurrent_errors')
    plt.legend(loc=1)

    plt.show()
