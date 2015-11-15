import tensorflow as tf
import logging
import numpy as np

__author__ = 'ptoth'

_LOGGER = logging.getLogger(__name__)


def activation_summary(x):

    """
    Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
    x: Tensor
    Returns:
    nothing
    """
    tensor_name = x.op.name
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


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

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():

        # placeholders for input and target output
        input = tf.placeholder(tf.float32, shape=[None, 8820], name="input")
        target = tf.placeholder(tf.float32, shape=[None, 8820], name="target")

        with tf.variable_scope('layer1') as scope:

            # ##############################################################################################################
            #
            #           LAYER 1
            #
            # ##############################################################################################################

            # spatial autoencoder variables W,b
            sp_enc_w = tf.Variable(tf.random_normal([8820, 1024], stddev=0.35), name="sp_enc_w_{0}".format(scope.name))
            sp_enc_b = tf.Variable(tf.zeros([1024]), name="sp_enc_b_{0}".format(scope.name))
            sp_dec_w = tf.Variable(tf.random_normal([1024, 8820], stddev=0.35), name="sp_dec_w_{0}".format(scope.name))
            sp_dec_b = tf.Variable(tf.zeros([8820]), name="sp_dec_b_{0}".format(scope.name))

            # spatial hidden activation
            sp_hidden = tf.nn.sigmoid(tf.matmul(input, sp_enc_w) + sp_enc_b, name="sp_hidden_{0}".format(scope.name))
            activation_summary(sp_hidden)

            # previous spatial hidden activation
            sp_hidden_prev = tf.placeholder(tf.float32, shape=[None, 1024], name="sp_hidden_prev_{0}".format(scope.name))

            # spatial reconstruction
            sp_reconstruct = tf.nn.sigmoid(tf.matmul(sp_hidden, sp_dec_w) + sp_dec_b, name="sp_reconstruct_{0}".format(scope.name))
            activation_summary(sp_reconstruct)

            # spatial autoencoder reconstruction loss
            sp_reconstruct_loss = tf.reduce_mean(tf.square(sp_reconstruct - input), name="sp_reconstruct_loss_{0}".format(scope.name))
            tf.scalar_summary(sp_reconstruct_loss.op.name, sp_reconstruct_loss)

            # ############################
            # OP
            # ############################
            # spatial autoencoder gradient optimization over reconstruction loss
            sp_reconstruct_train_op1 = tf.train.GradientDescentOptimizer(0.01).minimize(sp_reconstruct_loss)

            # recurrent autoencoder variables W,b
            rec_enc_w = tf.Variable(tf.random_normal([1024, 512], stddev=0.35), name="rec_enc_w_{0}".format(scope.name))
            rec_enc_b = tf.Variable(tf.zeros([512]), name="rec_enc_b_{0}".format(scope.name))
            rec_dec_w = tf.Variable(tf.random_normal([512, 1024], stddev=0.35), name="rec_dec_w_{0}".format(scope.name))
            rec_dec_b = tf.Variable(tf.zeros([1024]), name="rec_dec_b_{0}".format(scope.name))

            # recurrent hidden activation
            rec_hidden = tf.nn.sigmoid(tf.matmul(sp_hidden_prev, rec_enc_w) + rec_enc_b, name="rec_hidden_{0}".format(scope.name))
            activation_summary(rec_hidden)

            # recurrent reconstruction
            rec_reconstruct = tf.nn.sigmoid(tf.matmul(rec_hidden, rec_dec_w) + rec_dec_b, name="rec_reconstruct_{0}".format(scope.name))
            activation_summary(rec_reconstruct)

            # recurrent autoencoder reconstruction loss
            rec_reconstruct_loss = tf.reduce_mean(tf.square(rec_reconstruct - sp_hidden), name="rec_reconstruct_loss_{0}".format(scope.name))
            tf.scalar_summary(rec_reconstruct_loss.op.name, rec_reconstruct_loss)

            # ############################
            # OP
            # ############################
            # recurrent autoencoder gradient optimization over reconstruction loss
            rec_reconstruct_train_op1 = tf.train.GradientDescentOptimizer(0.01).minimize(rec_reconstruct_loss)

            layer1_output = tf.nn.sigmoid(sp_hidden + rec_reconstruct, name="layer1_output")
            activation_summary(layer1_output)

        with tf.variable_scope('layer2') as scope:

            ###############################################################################################################
            #
            #           LAYER 2
            #
            ###############################################################################################################

            # spatial autoencoder variables W,b
            sp_enc_w = tf.Variable(tf.random_normal([1024, 1024], stddev=0.35), name="sp_enc_w_{0}".format(scope.name))
            sp_enc_b = tf.Variable(tf.zeros([1024]), name="sp_enc_b_{0}".format(scope.name))
            sp_dec_w = tf.Variable(tf.random_normal([1024, 8820], stddev=0.35), name="sp_dec_w_{0}".format(scope.name))
            sp_dec_b = tf.Variable(tf.zeros([8820]), name="sp_dec_b_{0}".format(scope.name))

            # spatial hidden activation
            sp_hidden = tf.nn.sigmoid(tf.matmul(input, sp_enc_w) + sp_enc_b, name="sp_hidden_{0}".format(scope.name))
            activation_summary(sp_hidden)

            # previous spatial hidden activation
            sp_hidden_prev = tf.placeholder(tf.float32, shape=[None, 1024], name="sp_hidden_prev_{0}".format(scope.name))

            # spatial reconstruction
            sp_reconstruct = tf.nn.sigmoid(tf.matmul(sp_hidden, sp_dec_w) + sp_dec_b, name="sp_reconstruct_{0}".format(scope.name))
            activation_summary(sp_reconstruct)

            # spatial autoencoder reconstruction loss
            sp_reconstruct_loss = tf.reduce_mean(tf.square(sp_reconstruct - input), name="sp_reconstruct_loss_{0}".format(scope.name))
            tf.scalar_summary(sp_reconstruct_loss.op.name, sp_reconstruct_loss)

            # spatial autoencoder gradient optimization over reconstruction loss
            sp_reconstruct_train_op2 = tf.train.GradientDescentOptimizer(0.01).minimize(sp_reconstruct_loss)

            # recurrent autoencoder variables W,b
            rec_enc_w = tf.Variable(tf.random_normal([1024, 512], stddev=0.35), name="rec_enc_w_{0}".format(scope.name))
            rec_enc_b = tf.Variable(tf.zeros([512]), name="rec_enc_b_{0}".format(scope.name))
            rec_dec_w = tf.Variable(tf.random_normal([512, 1024], stddev=0.35), name="rec_dec_w_{0}".format(scope.name))
            rec_dec_b = tf.Variable(tf.zeros([1024]), name="rec_dec_b_{0}".format(scope.name))

            # recurrent hidden activation
            rec_hidden = tf.nn.sigmoid(tf.matmul(sp_hidden_prev, rec_enc_w) + rec_enc_b, name="rec_hidden_{0}".format(scope.name))
            activation_summary(rec_hidden)

            # recurrent reconstruction
            rec_reconstruct = tf.nn.sigmoid(tf.matmul(rec_hidden, rec_dec_w) + rec_dec_b, name="rec_reconstruct_{0}".format(scope.name))
            activation_summary(rec_reconstruct)

            # recurrent autoencoder reconstruction loss
            rec_reconstruct_loss = tf.reduce_mean(tf.square(rec_reconstruct - sp_hidden), name="rec_reconstruct_loss_{0}".format(scope.name))
            tf.scalar_summary(rec_reconstruct_loss.op.name, rec_reconstruct_loss)

            # recurrent autoencoder gradient optimization over reconstruction loss
            rec_reconstruct_train_op2 = tf.train.GradientDescentOptimizer(0.01).minimize(rec_reconstruct_loss)

            layer2_output = tf.nn.sigmoid(sp_hidden + rec_reconstruct, name="layer2_output")
            activation_summary(layer2_output)

            output = tf.nn.sigmoid(tf.matmul(layer2_output, sp_dec_w) + sp_dec_b)
            activation_summary(output)

            output_loss = tf.reduce_mean(tf.square(output - target), name="output_loss")
            tf.scalar_summary(output_loss.op.name, output_loss)

            output_train_op = tf.train.GradientDescentOptimizer(0.01).minimize(output_loss)

        summary_op = tf.merge_all_summaries()

        # Add an op to initialize the variables.
        init_op = tf.initialize_all_variables()
        # launching the model
        with tf.Session() as sess:
            # Run the init operation.
            sess.run(init_op)
            summary_writer = tf.train.SummaryWriter('summary', graph_def=sess.graph_def)
            # Use the model
            for j in range(10):
                for i in range(X_train_freq.shape[0]):
                    feed_dict = {input: X_train_freq[i],
                                 target: y_train_freq[i]}

                    sess.run(sp_reconstruct_train_op1, feed_dict=feed_dict)
                    sess.run(rec_reconstruct_train_op1, feed_dict=feed_dict)
                    sess.run(sp_reconstruct_train_op2)
                    sess.run(rec_reconstruct_train_op2)
                    # _, loss = sess.run([train_op, mse_loss], feed_dict=feed_dict)

                    # assert not np.isnan(loss), 'Model diverged with loss = NaN'
                    #
                    # summary_str = sess.run(summary_op, feed_dict)
                    # summary_writer.add_summary(summary_str, i)
                    # print('{0} iteration, sample {1}, loss: {2}'.format(j, i, loss))


