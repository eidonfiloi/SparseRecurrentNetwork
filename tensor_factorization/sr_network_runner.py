import logging
import numpy as np
import config.sr_network_conf as base_config
from core_network.Network import *


__author__ = 'ptoth'

_LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Load up the training data
    _LOGGER.info('Loading training data')
    input_file = '../data_prepared/bach_goldberg_aria_10'
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

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():

        network = SRNetwork(config['network'])
        # merge all summaries
        summary_op = tf.merge_all_summaries()

        # Add an op to initialize the variables.
        init_op = tf.initialize_all_variables()

        # launching the model
        with tf.Session() as sess:
            # Run the init operation.
            sess.run(init_op)
            summary_writer = tf.train.SummaryWriter('summary', graph_def=sess.graph_def)

            # Use the model
            for j in range(1):
                # for idx, sample in enumerate(X_train_freq):
                network.run(sess, X_train_freq, y_train_freq, summary_op, summary_writer)




