import tensorflow as tf
import logging
import numpy as np

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

    # Create two variables.
    encoder_weights = tf.Variable(tf.random_normal([8820, 200], stddev=0.35), name="encoder_weights")
    decoder_weights = tf.Variable(tf.random_normal([8820, 200], stddev=0.35), name="decoder_weights")
    encoder_biases = tf.Variable(tf.zeros([200]), name="encoder_biases")
    decoder_biases = tf.Variable(tf.zeros([200]), name="decoder_biases")

    # Add an op to initialize the variables.
    init_op = tf.initialize_all_variables()

    # launching the model
    with tf.Session() as sess:
      # Run the init operation.
      sess.run(init_op)

      # Use the model
      pass