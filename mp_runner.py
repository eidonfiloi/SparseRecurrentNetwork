import logging
import multiprocessing as mp
from recurrent_network.Network import *
import config.sr_network_configuration as base_config
from data_io.audio_data_utils import *
import pickle
from copy import copy

__author__ = 'ptoth'


def network_runner(sequence, network):

    output_errors = []
    for sample in sequence:
        pred, error = network.run(sample)
        output_errors.append(error)

    return network.layers, output_errors


_LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

     # Load up the training data
    _LOGGER.info('Loading training data')
    input_file = 'data_prepared/bach_goldberg_aria_10'
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

    input_tensor = X_train_freq[0:1]
    max_value = np.max(input_tensor)
    input_tensor /= max_value
    input_tensor = (input_tensor + 1.0) / 2.0

    epochs = config['global']['epochs']

    output = []

    args = [(input_tensor[i][15:20], network) for i in range(input_tensor.shape[0])]

    pool = mp.Pool(processes=2)
    results = [pool.apply_async(network_runner, args=inp) for inp in args]
    output = [p.get() for p in results]

    print len(output)