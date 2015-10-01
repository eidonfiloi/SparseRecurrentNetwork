import logging
import multiprocessing as mp
from recurrent_network.Network import *
import config.sr_network_configuration as base_config
from data_io.audio_data_utils import *
import pickle
from copy import copy

__author__ = 'ptoth'


def network_runner(network, sequence):

    print "inside network runner"
    output_errors = []
    for sample in sequence:
        pred, error = network.run(sample)
        output_errors.append(error)

    print output_errors, network.layers
    return network, output_errors

result_list = []


def dummy(inp):
    return 2


def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    print result
    result_list.append(result)

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

    input_tensor = X_train_freq[0:2]
    max_value = np.max(input_tensor)
    input_tensor /= max_value
    input_tensor = (input_tensor + 1.0) / 2.0

    epochs = config['global']['epochs']

    output_ = []
    output = mp.Queue()

    args = [input_tensor[i][15:16] for i in range(input_tensor.shape[0])]

    pool = mp.Pool(processes=2)

    for inp in args:
        pool.apply_async(network_runner, args=(copy(network), inp), callback=log_result)

    pool.close()
    pool.join()
    print(result_list)

    # # Setup a list of processes that we want to run
    # processes = [mp.Process(target=network_runner, args=(network, inp)) for inp in args]
    #
    # # Run processes
    # for p in processes:
    #     p.start()
    #
    # # Exit the completed processes
    # for p in processes:
    #     p.join()
    #
    # # Get process results from the output queue
    # results = [output.get() for p in processes]
    #
    # for network, error in results:
    #     print error


