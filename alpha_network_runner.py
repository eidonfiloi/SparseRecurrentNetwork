from copy import copy

import numpy as np

from recurrent_network.Network import *
import config.alpha_network_configuration as base_config

_LOGGER = logging.getLogger(__name__)

_CHARS_ONE_HOT = (0.05 / 255) * np.ones((256, 256))
np.fill_diagonal(_CHARS_ONE_HOT, 0.95) #np.eye(256, 256)


def one_hot_decode(arr):
    return np.argmax(arr)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    text_data_path = 'data_prepared/dickens_twocities.txt' #"data_prepared/text_test.txt" #"data_prepared/wiki_100k.txt"

    params = base_config.get_config()
    network = SRNetwork(params['network'])
    epochs = params['global']['epochs']

    feedforward_errors = {layer['name']: [] for layer in params['network']['layers']}
    recurrent_errors = {layer['name']: [] for layer in params['network']['layers']}
    feedback_errors = {layer['name']: [] for layer in params['network']['layers']}
    output_mse = []
    output_preds = []

    prev_pred = None
    for j in range(epochs):
        with open(text_data_path, 'rb') as f:
            for i, line in enumerate(f):
                if i < 5000:
                    print '######################### line {0}'.format(i)
                    arr = [ord(ch) for ch in list(line)]
                    output_arr = []
                    _LOGGER.info('input: {0}'.format(line))
                    for el in arr:
                        inp = _CHARS_ONE_HOT[el]
                        pred, mse = network.run(inp)
                        output_arr.append(chr(one_hot_decode(pred)))
                        output_mse.append(mse)
                        if prev_pred is not None:
                            prev_pred_int = one_hot_decode(prev_pred)
                            prev_pred_ch = chr(prev_pred_int)
                            prev_pred_vector = _CHARS_ONE_HOT[prev_pred_int]

                            print '############### epoch: {0}\n' \
                              '############### line: {1}\n' \
                              '############### input: \n' \
                              '{2}\n' \
                              '############### prev_pred: \n' \
                              '{3}\n' \
                              'output_error: {4}'.format(j, i, chr(el), prev_pred_ch, mse)
                            plt.ion()
                            plt.axis([-1, 16, -1, 16])
                            x_r, y_r = np.argwhere(inp.reshape(16, 16) >= 0.95).T
                            x_t, y_t = np.argwhere(prev_pred_vector.reshape(16, 16) >= 0.95).T
                            plt.scatter(x_r, y_r, alpha=0.5, c='r', marker='s', s=255)
                            plt.scatter(x_t, y_t, alpha=0.5, c='b', marker='o', s=245)
                            plt.draw()
                            time.sleep(0.1)
                            plt.clf()
                        prev_pred = copy(pred)
                    _LOGGER.info('################### input line: {0}'.format(line))
                    _LOGGER.info('################### output line: {0}'.format(''.join(output_arr)))
                    output_preds.append(''.join(output_arr))
                    for key, v in network.feedforward_errors.items():
                        if len(v) > 0:
                            feedforward_errors[key].append(v[0])
                    for key, v in network.recurrent_errors.items():
                        if len(v) > 0:
                            recurrent_errors[key].append(v[0])
                    for key, v in network.feedback_errors.items():
                        if len(v) > 0:
                            feedback_errors[key].append(v[0])
                else:
                    break

    for lin in output_preds:
        print lin

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
