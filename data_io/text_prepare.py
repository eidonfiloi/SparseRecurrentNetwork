import numpy as np
import nltk
import random
import pickle

__author__ = 'ptoth'


if __name__ == '__main__':

    text_data_path = "/Users/ptoth/Downloads/OpenSubtitles2013.en"

    tokens = set("ENDOFSENTENCE".decode('utf-8'))
    with open(text_data_path, 'rb') as f:
            for i, line in enumerate(f):
                if i % 10000 == 0:
                    print "line ", i
                if i < 1000000:
                    for t in line.split(" "):
                        tokens.add(t.decode('utf-8'))
                else:
                    break
                    # for t in nltk.word_tokenize(line.decode('utf-8').strip()):
                    #     tokens.add(t)

    sdr_to_word = dict()
    word_to_sdr = dict()
    print "length of tokens: ", len(tokens)

    for idx, t in enumerate(tokens):
        if idx % 1000 == 1:
            print idx
        sdr = tuple(np.random.choice(20, 8, False))
        while (sdr in sdr_to_word):
            sdr = tuple(np.random.choice(20, 8, False))
        # print sdr
        # np_sdr = tuple([1 if idx in sdr else 0 for idx in range(16)])
        sdr_to_word[sdr] = t
        word_to_sdr[t] = sdr

    # print sdr_to_word
    # print word_to_sdr

    with open("../data_prepared/opensub2013_dict_1m.pickle", 'wb') as f:
        pickle.dump({
            'sdr_to_word': sdr_to_word,
            'word_to_sdr': word_to_sdr
        }, f)



