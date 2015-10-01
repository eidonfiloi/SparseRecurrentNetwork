import numpy as np
from copy import copy
import pickle

__author__ = 'ptoth'


if __name__ == '__main__':

    current_store = None
    data_dict = dict()
    with open("forecast_data.csv", 'rb') as f:
        for idx, lines in enumerate(f):
            if idx > 0:
                line_arr = lines.split(',')
                side = line_arr[0]
                id = line_arr[1]
                store = line_arr[2]
                day = line_arr[3]
                data = np.array(line_arr[4:25], dtype='|S4').astype('float')
                if current_store is None or current_store != store:
                    data_dict[store] = {
                        'train': [(id, day, data)],
                        'validation': [],
                        'test': [],
                    }
                else:
                    data_dict[store][side].append((id, day, data))
                current_store = copy(store)

    with open("forecast_data.pickle", 'wb') as f:
        pickle.dump(data_dict, f)


