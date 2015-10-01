__author__ = 'eidonfiloi'

import logging
import matplotlib.pyplot as plt
from recurrent_network.Network import *
import config.sr_network_configuration as base_config
from data_io.audio_data_utils import *
import pickle
from copy import copy

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

    load_model = False

    if load_model:
        with open('serialized_models/aria_network_10ep.pickle', "rb") as f:
            network_data = pickle.load(f)
            network = SRNetwork(config['network'], network_data)
    else:
        network = SRNetwork(config['network'])

    # input_sample = X_train_freq[0]
    # output = []
    # for i in range(input_sample.shape[0]):
    #     output.append(input_sample[i])
    # for i in xrange(len(output)):
    #     output[i] *= X_var_freq
    #     output[i] += X_mean_freq
    # save_generated_example("bach_golberg_test.wav", output, useTimeDomain=False)

    input_sample = X_train_freq[1]
    max_value = np.max(input_sample)
    input_sample /= max_value
    input_sample = (input_sample + 1.0) / 2.0
    input_sample_y = y_train_freq[0]

    epochs = config['global']['epochs']

    output = []

    feedforward_errors = {layer['name']: [] for layer in config['network']['layers']}
    recurrent_errors = {layer['name']: [] for layer in config['network']['layers']}
    feedback_errors = {layer['name']: [] for layer in config['network']['layers']}
    output_mse = []
    current_j = 0
    prev_j = -1
    prev_output = None
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
            network_output, mse = network.run(input_)
            output_mse.append(mse)
            if prev_output is not None:
                    prev_output_bin = np.zeros(prev_output.size).astype('int')
                    for ind in range(0, prev_output.size):
                        if prev_output[ind] > 0.5:
                            prev_output_bin[ind] = 1
                        else:
                            prev_output_bin[ind] = 0
                    mod_input = np.zeros(input_.size).astype('int')
                    for ind in range(0, input_.size):
                        if input_[ind] > 0.5:
                            mod_input[ind] = 1

                    print '############### epoch: {0}\n' \
                          '############### sequence: {1}\n' \
                          '############### input: \n' \
                          '{2}\n' \
                          '############### prev_output_bin: \n' \
                          '{3}\n' \
                          '############### prev_output: \n' \
                          '{4}\n' \
                          'output_error: {5}'.format(j, i, input_, prev_output_bin, prev_output, mse)
                    plt.ion()
                    plt.axis([-1, 90, -1, 98])
                    x_r, y_r = np.argwhere(mod_input.reshape(90, 98) == 1).T
                    x_t, y_t = np.argwhere(prev_output_bin.reshape(90, 98) == 1).T
                    plt.scatter(x_r, y_r, alpha=0.5, c='r', marker='s', s=15)
                    plt.scatter(x_t, y_t, alpha=0.5, c='b', marker='o', s=13)
                    plt.draw()
                    time.sleep(0.1)
                    plt.clf()
            prev_output = copy(network_output)
            for key, v in network.feedforward_errors.items():
                if len(v) > 0:
                    feedforward_errors[key].append(v[0])
            for key, v in network.recurrent_errors.items():
                if len(v) > 0:
                    recurrent_errors[key].append(v[0])
            for key, v in network.feedback_errors.items():
                if len(v) > 0:
                    feedback_errors[key].append(v[0])
            _LOGGER.info('output length {0}\n{1}'.format(len(network_output), network_output[0:20]))
            network_output = 2.0*network_output - 1.0
            output.append(network_output * max_value)

    for i in xrange(len(output)):
        output[i] *= X_var_freq
        output[i] += X_mean_freq
    save_generated_example("bach_goldberg_aria_10.wav", output, useTimeDomain=False)

    if config['network']['serialize']:
        network.serialize()

    plt.ioff()
    plt.subplot(4, 1, 1)
    for k, v in feedforward_errors.items():
        if len(v) > 0:
            plt.plot(range(len(v)), v, label=k)
    plt.xlabel('epochs')
    plt.ylabel('feedforward_errors')
    # plt.title('feedforward_errors')
    plt.legend(loc=1)

    plt.subplot(4, 1, 2)
    for k, v in recurrent_errors.items():
        if len(v) > 0:
            plt.plot(range(len(v)), v, label=k)
    plt.xlabel('epochs')
    plt.ylabel('recurrent_errors')
    # plt.title('recurrent_errors')
    plt.legend(loc=1)

    plt.subplot(4, 1, 3)
    for k, v in feedback_errors.items():
        if len(v) > 0:
            plt.plot(range(len(v)), v, label=k)
    plt.xlabel('epochs')
    plt.ylabel('feedback_errors')
    # plt.title('recurrent_errors')
    plt.legend(loc=1)

    plt.subplot(4, 1, 4)
    plt.plot(range(len(output_mse)), output_mse, label="mse")
    plt.xlabel('epochs')
    plt.ylabel('output_mse')
    # plt.title('recurrent_errors')
    plt.legend(loc=1)

    plt.show()
