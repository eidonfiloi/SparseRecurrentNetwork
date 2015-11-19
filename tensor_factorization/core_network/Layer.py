import tensorflow as tf
import abc
import numpy as np

from tensor_factorization.utils.SummaryHelpers import SummaryHelpers

__author__ = 'ptoth'


class Layer(object):
    """
        This class represents layers
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, parameters):
        self.parameters = parameters

    @abc.abstractmethod
    def run(self, sess, inputs, state):
        """

        :param sess:
        :param inputs:
        :return:
        """


class SRLayer(Layer):
    """

    """

    def __init__(self, parameters):
        super(SRLayer, self).__init__(parameters)
        self.name = parameters['name']
        self.inputs_shape = parameters['inputs_shape']
        self.sp_hidden_shape = parameters['sp_hidden_shape']
        self.rec_hidden_shape = parameters['rec_hidden_shape']
        self.learning_rate = parameters['learning_rate']

        with tf.variable_scope(self.name):

            # variables holding spatial and recurrent weights and biases
            self.sp_enc_w = tf.Variable(tf.random_normal([self.inputs_shape, self.sp_hidden_shape], stddev=0.35),
                                        name="sp_enc_w")
            self.sp_enc_b = tf.Variable(tf.random_normal([self.sp_hidden_shape], stddev=0.35),
                                        name="sp_enc_b")
            self.sp_dec_w = tf.Variable(tf.random_normal([self.sp_hidden_shape, self.inputs_shape], stddev=0.35),
                                        name="sp_dec_w")
            self.sp_dec_b = tf.Variable(tf.random_normal([self.inputs_shape], stddev=0.35),
                                        name="sp_dec_b")
            self.rec_enc_w = tf.Variable(tf.random_normal([self.sp_hidden_shape, self.rec_hidden_shape], stddev=0.35),
                                         name="rec_enc_w")
            self.rec_enc_b = tf.Variable(tf.random_normal([self.rec_hidden_shape], stddev=0.35),
                                         name="rec_enc_b")
            self.rec_dec_w = tf.Variable(tf.random_normal([self.rec_hidden_shape, self.sp_hidden_shape], stddev=0.35),
                                         name="rec_dec_w")
            self.rec_dec_b = tf.Variable(tf.random_normal([self.sp_hidden_shape], stddev=0.35),
                                         name="rec_dec_b")

    def run(self, sess, inputs, state):
        with tf.variable_scope(self.name):
            # tensorflow OP spatial hidden
            sp_hidden = tf.nn.sigmoid(tf.matmul(inputs, self.sp_enc_w) + self.sp_enc_b,
                                      name="sp_hidden")
            SummaryHelpers.activation_summary(sp_hidden)

            # tensorflow OP spatial reconstruction
            sp_reconstruct = tf.nn.sigmoid(tf.matmul(sp_hidden, self.sp_dec_w) + self.sp_dec_b,
                                           name="sp_reconstruct")
            SummaryHelpers.activation_summary(sp_reconstruct)

            # spatial autoencoder reconstruction loss
            sp_reconstruct_loss = tf.reduce_mean(tf.square(sp_reconstruct - inputs),
                                                 name="sp_reconstruct_loss")
            tf.scalar_summary(sp_reconstruct_loss.op.name, sp_reconstruct_loss)

            # spatial autoencoder gradient optimization over reconstruction loss
            sp_reconstruct_train_op = tf.train\
                .GradientDescentOptimizer(self.learning_rate)\
                .minimize(sp_reconstruct_loss)


            # tensorflow OP recurrent hidden activation
            rec_hidden = tf.nn.sigmoid(tf.matmul(state, self.rec_enc_w) + self.rec_enc_b,
                                       name="rec_hidden")
            SummaryHelpers.activation_summary(rec_hidden)

            # tensorflow OP recurrent reconstruction
            rec_reconstruct = tf.nn.sigmoid(tf.matmul(rec_hidden, self.rec_dec_w) + self.rec_dec_b,
                                            name="rec_reconstruct")
            SummaryHelpers.activation_summary(rec_reconstruct)

            # recurrent autoencoder reconstruction loss
            rec_reconstruct_loss = tf.reduce_mean(tf.square(rec_reconstruct - sp_hidden),
                                                  name="rec_reconstruct_loss")
            tf.scalar_summary(rec_reconstruct_loss.op.name, rec_reconstruct_loss)

            # recurrent autoencoder gradient optimization over reconstruction loss
            rec_reconstruct_train_op = tf.train\
                .GradientDescentOptimizer(self.learning_rate)\
                .minimize(rec_reconstruct_loss)

            sess.run(sp_reconstruct_train_op)
            with tf.control_dependencies([sp_reconstruct_train_op]):
                sess.run(rec_reconstruct_train_op)

            # compute layer output
            layer_output = tf.nn.sigmoid(sp_hidden + rec_reconstruct,
                                         name="layer_output")
            SummaryHelpers.activation_summary(layer_output)
            state = tf.identity(sp_hidden)
            return layer_output, state

    def initial_state(self, batch_size):
        return tf.zeros([batch_size, self.sp_hidden_shape])


        



