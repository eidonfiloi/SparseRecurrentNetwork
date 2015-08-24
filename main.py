__author__ = 'eidonfiloi'

from sklearn import preprocessing
import numpy as np
from recurrent_network.SparseRecurrentLayer import SparseRecurrentLayer
import matplotlib.pyplot as plt
import time



if __name__ == "__main__":

    #Load up the training data
    print ('Loading training data')
    input_file = 'data_prepared/test_bach26'
    #X_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
    #y_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
    #X_mean is a matrix of size (num_frequency_dims,) containing the mean for each frequency dimension
    #X_var is a matrix of size (num_frequency_dims,) containing the variance for each frequency dimension
    X_train = np.load(input_file + '_x.npy')
    y_train = np.load(input_file + '_y.npy')
    X_mean = np.load(input_file + '_mean.npy')
    X_var = np.load(input_file + '_var.npy')
    print ('Finished loading training data')

    spatial_layer = SparseRecurrentLayer(name="SpatialLayer",
                                         num_inputs=22050,
                                         sdr_size=1024,
                                         sparsity=0.05,
                                         min_weight=1.0,
                                         max_weight=-1.0,
                                         duty_cycle_decay=0.02,
                                         weights_lr=0.0009,
                                         inhibition_lr=0.005,
                                         bias_lr=0.0005)
    recurrent_layer = SparseRecurrentLayer(name="RecurrentLayer",
                                           num_inputs=1024,
                                           sdr_size=1024,
                                           sparsity=0.001,
                                           min_weight=1.0,
                                           max_weight=-1.0,
                                           duty_cycle_decay=0.02,
                                           weights_lr=0.001,
                                           inhibition_lr=0.0001,
                                           bias_lr=0.001)


    input_sample_r = X_train[0]
    input_sample_y_r = y_train[0]

    input_sample = preprocessing.scale(input_sample_r)
    input_sample_y = preprocessing.scale(input_sample_y_r)

    prev_rec_sdr = np.zeros(1024)
    epochs = 100

    # plt.axis([1, 32, 1, 32])
    # plt.ion()
    # plt.show()


    spatial_errors = []
    recurrent_errors = []
    for j in range(epochs):


        for i in range(input_sample.shape[0]):
            input = input_sample[i]
            print 'input min is {0}, max is {1}'.format(np.min(input), np.max(input))
            sdr = spatial_layer.generate(input)
            spat_er = spatial_layer.learn(input, sdr)
            next_prev_rec_sdr = recurrent_layer.generate(sdr)
            rec_er = None
            if i > 0:
                rec_er = recurrent_layer.learn(sdr, prev_rec_sdr)
            prev_rec_sdr = next_prev_rec_sdr

            print 'sdr {0}\n{1}'.format(i, sdr[0:20])
            print 'prev {0}\n{1}'.format(i, prev_rec_sdr[0:20])
            if spat_er is not None:
                spatial_errors.append(np.mean(np.abs(spat_er)**2, axis=0))
            if rec_er is not None:
                recurrent_errors.append(np.mean(np.abs(rec_er)**2, axis=0))

            # x, y = np.argwhere(sdr.reshape(32, 32) == 1).T
            # x_, y_ = np.argwhere(prev_rec_sdr.reshape(32, 32) == 1).T
            # #plt.scatter(x, y, c='b')
            # plt.scatter(x_, y_, c='r', marker='s')
            # plt.draw()
            # time.sleep(0.05)
            # plt.cla()


    plt.subplot(2, 1, 1)
    plt.plot(range(len(spatial_errors)), spatial_errors, label="spatial errors", color='r')
    plt.xlabel('epochs')
    plt.ylabel('errors')
    plt.title('spatial')
    plt.legend(loc=2)

    plt.subplot(2,1,2)
    plt.plot(range(len(recurrent_errors)), recurrent_errors, label="recurrent errors", color='b')
    plt.xlabel('epochs')
    plt.ylabel('errors')
    plt.title('recurrent')
    plt.legend(loc=2)

    plt.show()
    print spatial_errors
    print recurrent_errors
