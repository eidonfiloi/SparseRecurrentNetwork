__author__ = 'eidonfiloi'

import logging
import matplotlib.pyplot as plt
from recurrent_network.Network import *
import config.forecast_network_configuration as base_config
from data_io.audio_data_utils import *
import pickle
import json
from copy import copy
import csv
import math

_LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

     # Load up the training data
    _LOGGER.info('Loading training data')

    forecast_data = dict()
    data_keys = None
    current_store = None
    # with open("data_io/forecast_data.csv", 'rb') as f:
    #     for idx, lines in enumerate(f):
    #         if idx > 0:
    #             line_arr = lines.split(',')
    #             side = line_arr[0]
    #             id = line_arr[1]
    #             store = line_arr[2]
    #             day = line_arr[3]
    #             data = np.array(line_arr[4:25], dtype='|S4').astype('float')
    #             if current_store is None or current_store != store:
    #                 forecast_data[store] = {
    #                     'train': [(id, day, data)],
    #                     'validation': [],
    #                     'test': [],
    #                 }
    #             else:
    #                 forecast_data[store][side].append((id, day, data))
    #             current_store = copy(store)
    # data_keys = forecast_data.keys()
    with open("data_io/forecast_data_scaled.pickle", 'rb') as f:
        forecast_data = pickle.load(f)
        data_keys = forecast_data.keys()
    _LOGGER.info('Finished loading training data')

    config = base_config.get_config()

    # network = SRNetwork(config['network'])

    sales_mean = 6.98716
    sales_sd = 3.537734
    sales_max = 10.635

    epochs = config['global']['epochs']

    preds = []
    targets = []

    feedforward_errors = {layer['name']: [] for layer in config['network']['layers']}
    recurrent_errors = {layer['name']: [] for layer in config['network']['layers']}
    feedback_errors = {layer['name']: [] for layer in config['network']['layers']}
    output_error = []
    prev_output = None
    kaggle_preds = []
    for k, v in forecast_data.items():
        network = SRNetwork(config['network'])
        for j in range(epochs):
            for id, day, dat in v['train']:
                _LOGGER.info(
                '\n############## epoch {0}\n'
                '############## store {1}\n'
                '############## day {2}\n '
                '############## data {3}'.format(j, k, day, dat))
                network_output, error = network.run(dat)
                output_error.append(error)
                if prev_output is not None:
                    _LOGGER.info('############### input: \n' \
                          '{0}\n' \
                          '############### prev_output: \n' \
                          '{1}\n' \
                          'inp: {2}\n' \
                          'prev_pred: {3}\n' \
                          'output_error: {4}'.format(dat[20], prev_output[20], dat, prev_output, error))
                    if j == epochs - 1:
                        preds.append(prev_output[20])
                        targets.append(dat[20])
                prev_output = copy(network_output)
                for key, vv in network.feedforward_errors.items():
                    if len(vv) > 0:
                        feedforward_errors[key].append(vv[0])
                for key, vv in network.recurrent_errors.items():
                    if len(vv) > 0:
                        recurrent_errors[key].append(vv[0])
                for key, vv in network.feedback_errors.items():
                    if len(vv) > 0:
                        feedback_errors[key].append(vv[0])
            for id, day, dat in v['validation']:
                _LOGGER.info(
                '\n############## epoch {0}\n'
                '############## store {1}\n'
                '############## day {2}\n '
                '############## data {3}'.format(j, k, day, dat))
                network_output, error = network.run(dat)
                output_error.append(error)
                if prev_output is not None:
                    _LOGGER.info('############### input: \n' \
                          '{0}\n' \
                          '############### prev_output: \n' \
                          '{1}\n' \
                          'inp: {2}\n' \
                          'prev_pred: {3}\n' \
                          'output_error: {4}'.format(dat[20], prev_output[20], dat, prev_output, error))
                    # if j == epochs - 1:
                    #     preds.append(prev_output[20])
                    #     targets.append(dat[20])
                prev_output = copy(network_output)
                for key, vv in network.feedforward_errors.items():
                    if len(vv) > 0:
                        feedforward_errors[key].append(vv[0])
                for key, vv in network.recurrent_errors.items():
                    if len(vv) > 0:
                        recurrent_errors[key].append(vv[0])
                for key, vv in network.feedback_errors.items():
                    if len(vv) > 0:
                        feedback_errors[key].append(vv[0])
        starting_dat = v['validation'][2][-1]
        network_output, error = network.run(starting_dat, learning_on=False)
        for id, day, dat in v['test']:
            dat[20] = network_output[20]
            kaggle_preds.append((id, math.exp(sales_max*network_output[20]) - 1.0))
            network_output, error = network.run(dat)

    with open('data_prepared/rossmann_full_single.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
        writer.writerow(['Id', 'Sales'])
        for ids, sales in kaggle_preds:
            writer.writerow([ids, sales])

    # preds_vs_target = {
    #     'preds': preds,
    #     'targets': targets
    # }
    #
    # with open("data_prepared/forecast_preds_target.json", 'wb') as fo:
    #     json.dump(preds_vs_target, fo)

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
    plt.plot(range(len(output_error)), output_error, label="mse")
    plt.xlabel('epochs')
    plt.ylabel('output_mse')
    # plt.title('recurrent_errors')
    plt.legend(loc=1)

    # plt.subplot(1, 1, 1)
    # plt.plot(range(len(preds)), preds, label="preds")
    # plt.plot(range(len(targets)), targets, label="targets")
    # plt.xlabel('epochs')
    # plt.ylabel('sales')
    # # plt.title('recurrent_errors')
    # plt.legend(loc=1)

    plt.show()
