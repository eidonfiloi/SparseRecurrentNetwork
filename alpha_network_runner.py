from copy import copy
import pickle
import nltk

import numpy as np

from recurrent_network.Network import *
import config.alpha_network_configuration as base_config

_LOGGER = logging.getLogger(__name__)

# np.eye(256, 256)
_CHARS_ONE_HOT = np.eye(256, 256) # 0.1*np.ones((256, 256))

# for i in range(_CHARS_ONE_HOT.shape[0]):
#     _CHARS_ONE_HOT[i][i:min(i+10, 255)] = 0.9

# (0.05 / 255) * np.ones((256, 256))
# np.fill_diagonal(_CHARS_ONE_HOT, 0.95)


def one_hot_decode(arr):
    return np.argmax(arr)


def binary_vector(tup):
    return np.array([1 if idx in list(tup) else 0 for idx in range(20)])


def get_closest_word(pred, sett):
    return sorted([(np.dot(pred, binary_vector(keyy)), keyy)
                   for keyy in sett], reverse=True)[0]

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    text_data_path = '/Users/ptoth/Downloads/OpenSubtitles2013.en' #"data_prepared/text_test.txt" #"data_prepared/wiki_100k.txt"

    params = base_config.get_config()
    network = SRNetwork(params['network'])
    epochs = params['global']['epochs']

    feedforward_errors = {layer['name']: [] for layer in params['network']['layers']}
    recurrent_errors = {layer['name']: [] for layer in params['network']['layers']}
    feedback_errors = {layer['name']: [] for layer in params['network']['layers']}
    output_mse = []
    output_preds = []

    test_sdr_dict = "data_prepared/opensub2013_dict_1m.pickle"
    sdr_dict = None
    with open(test_sdr_dict, 'rb') as f:
        sdr_dict = pickle.load(f)

    sdr_to_word = sdr_dict['sdr_to_word']
    word_to_sdr = sdr_dict['word_to_sdr']

    print "sdr to words size: ", len(sdr_to_word)

    sdr_eof = tuple(np.random.choice(20, 8, False))
    while sdr_eof in sdr_to_word:
        sdr_eof = tuple(np.random.choice(20, 8, False))

    sdr_to_word[sdr_eof] = u'ENDOFSENTENCE'
    word_to_sdr[u'ENDOFSENTENCE'] = sdr_eof

    prev_pred = None
    prev_line = None

    with open(text_data_path, 'rb') as f:
        for i, line in enumerate(f):
            if i % 2 == 0:
                prev_line = copy(line)
            else:
                for j in range(epochs):
                    print '######################### line {0}'.format(i)
                    prev_tokens = (prev_line + " ENDOFSENTENCE").split(" ")
                    tokens = (line + " ENDOFSENTENCE").split(" ")
                    output_arr = []
                    _LOGGER.info('input: {0}'.format(prev_line))
                    _LOGGER.info('input: {0}'.format(line))
                    for el in prev_tokens:
                        # inp = _CHARS_ONE_HOT[el]
                        inp = binary_vector(word_to_sdr[el.decode('utf-8')])
                        pred, error = network.run(inp)
                        output_mse.append(error)
                        if prev_pred is not None:
                            mod_prev_pred = np.zeros(prev_pred.size).astype('int')
                            for ind in range(0, prev_pred.size):
                                if prev_pred[ind] > 0.5:
                                    mod_prev_pred[ind] = 1
                            print '################ \n' \
                              'input: {0}\n' \
                              'prev_pred: {1}\n' \
                              '#################'.format(inp, mod_prev_pred)
                            # prev_pred_int = one_hot_decode(prev_pred)
                            # prev_pred_ch = chr(prev_pred_int)
                            # prev_pred_vector = _CHARS_ONE_HOT[prev_pred_int]
                            if j == epochs - 1:
                                closeness, prev_pred_closest_word_tup = get_closest_word(prev_pred, sdr_to_word.keys())
                                closest_word = sdr_to_word[prev_pred_closest_word_tup]#.decode('utf-8')
                                output_arr.append(closest_word)
                                print '############### epoch: {0}\n' \
                                  '############### line: {1}\n' \
                                  '############### input: \n' \
                                  '{2}\n' \
                                  '############### prev_pred: \n' \
                                  '{3}\n' \
                                  'output_error: {4}\n' \
                                  'closeness: {5}'.format(j, i, el, closest_word, error, closeness)
                            plt.ion()
                            plt.axis([-1, 5, -1, 4])
                            x_r, y_r = np.argwhere(inp.reshape(5, 4) >= 0.5).T
                            x_t, y_t = np.argwhere(prev_pred.reshape(5, 4) >= 0.5).T
                            plt.scatter(x_r, y_r, alpha=0.5, c='r', marker='s', s=255)
                            plt.scatter(x_t, y_t, alpha=0.5, c='b', marker='o', s=245)
                            plt.draw()
                            time.sleep(0.1)
                            plt.clf()
                        prev_pred = copy(pred)
                    _LOGGER.info('################### input line: {0}'.format(line))
                    _LOGGER.info('################### output line: {0}'.format(' '.join(output_arr)))
                    for el in tokens:
                        # inp = _CHARS_ONE_HOT[el]
                        inp = binary_vector(word_to_sdr[el.decode('utf-8')])
                        pred, error = network.run(inp)
                        output_mse.append(error)
                        if prev_pred is not None:
                            mod_prev_pred = np.zeros(prev_pred.size).astype('int')
                            for ind in range(0, prev_pred.size):
                                if prev_pred[ind] > 0.5:
                                    mod_prev_pred[ind] = 1
                            print '################ \n' \
                              'input: {0}\n' \
                              'prev_pred: {1}\n' \
                              '#################'.format(inp, mod_prev_pred)
                            # prev_pred_int = one_hot_decode(prev_pred)
                            # prev_pred_ch = chr(prev_pred_int)
                            # prev_pred_vector = _CHARS_ONE_HOT[prev_pred_int]
                            if j == epochs - 1:
                                closeness, prev_pred_closest_word_tup = get_closest_word(prev_pred, sdr_to_word.keys())
                                closest_word = sdr_to_word[prev_pred_closest_word_tup]#.decode('utf-8')
                                output_arr.append(closest_word)
                                print '############### epoch: {0}\n' \
                                  '############### line: {1}\n' \
                                  '############### input: \n' \
                                  '{2}\n' \
                                  '############### prev_pred: \n' \
                                  '{3}\n' \
                                  'output_error: {4}\n' \
                                  'closeness: {5}'.format(j, i, el, closest_word, error, closeness)
                            plt.ion()
                            plt.axis([-1, 5, -1, 4])
                            x_r, y_r = np.argwhere(inp.reshape(5, 4) >= 0.5).T
                            x_t, y_t = np.argwhere(prev_pred.reshape(5, 4) >= 0.5).T
                            plt.scatter(x_r, y_r, alpha=0.5, c='r', marker='s', s=255)
                            plt.scatter(x_t, y_t, alpha=0.5, c='b', marker='o', s=245)
                            plt.draw()
                            time.sleep(0.1)
                            plt.clf()
                        prev_pred = copy(pred)
                    _LOGGER.info('################### input line: {0}'.format(line))
                    _LOGGER.info('################### output line: {0}'.format(' '.join(output_arr)))
                    # output_preds.append(''.join(output_arr))
                    for key, v in network.feedforward_errors.items():
                        if len(v) > 0:
                            feedforward_errors[key].append(v[0])
                    for key, v in network.recurrent_errors.items():
                        if len(v) > 0:
                            recurrent_errors[key].append(v[0])
                    for key, v in network.feedback_errors.items():
                        if len(v) > 0:
                            feedback_errors[key].append(v[0])

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
