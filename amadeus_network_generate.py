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
    input_file = 'data_prepared/bach_goldberg_aria_10'
    # X_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
    X_train_freq = np.load(input_file + '_x.npy')
    # X_mean is a matrix of size (num_frequency_dims,) containing the mean for each frequency dimension
    X_mean_freq = np.load(input_file + '_mean.npy')
    # X_var is a matrix of size (num_frequency_dims,) containing the variance for each frequency dimension
    X_var_freq = np.load(input_file + '_var.npy')

    config = base_config.get_config()

    network = SRNetwork(config['network'])

    input_sample = X_train_freq[0]
    max_value = np.max(input_sample)
    input_sample /= max_value
    input_sample = (input_sample + 1.0) / 2.0

    epochs = config['global']['epochs']
    num_generated_steps = 100
    network_output = None
    for i in range(input_sample.shape[0]):
        input_ = input_sample[i]
        network_output, mse = network.run(input_)

    output_gen = []
    network_output_gen = network_output
    for i in range(num_generated_steps):
        network_output_gen, mse = network.run(network_output_gen, learning_on=False)
        output_gen.append((2.0*network_output_gen - 1.0)*max_value)

    for i in xrange(len(output_gen)):
        output_gen[i] *= X_var_freq
        output_gen[i] += X_mean_freq
    save_generated_example("bach_goldberg_aria_output_gen.wav", output_gen, useTimeDomain=False)
