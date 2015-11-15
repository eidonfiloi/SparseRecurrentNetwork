import tensorflow as tf
import abc

from utils.SummaryHelpers import SummaryHelpers

__author__ = 'ptoth'


class Layer(object):
    """
        This class represents layers
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, parameters):
        self.parameters = parameters

    @abc.abstractmethod
    def run(self, sess, inputs):
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

        # placeholders for input and previous spatial hidden state
        self.layer_input = tf.placeholder(tf.float32,
                                          shape=self.inputs_shape,
                                          name='{0}/input'.format(self.name))
        self.sp_hidden_prev = tf.placeholder(tf.float32,
                                             shape=self.sp_hidden_shape,
                                             name="{0}/sp_hidden_prev".format(self.name))

        # variables holding spatial and recurrent weights and biases
        with tf.variable_scope(self.name):
            self.sp_enc_w = tf.get_variable("sp_enc_w",
                                            shape=[self.inputs_shape, self.sp_hidden_shape],
                                            initializer=(tf.random_normal([self.inputs_shape, self.sp_hidden_shape],
                                                                          stddev=0.35)))
            self.sp_enc_b = tf.get_variable("sp_enc_b",
                                            shape=[self.sp_hidden_shape],
                                            initializer=(tf.zeros([self.sp_hidden_shape])))
            self.sp_dec_w = tf.get_variable("sp_dec_w",
                                            shape=[self.sp_hidden_shape, self.inputs_shape],
                                            initializer=(tf.random_normal([self.sp_hidden_shape, self.inputs_shape],
                                                                          stddev=0.35)))
            self.sp_dec_b = tf.get_variable("sp_dec_b",
                                            shape=[self.inputs_shape],
                                            initializer=(tf.zeros([self.inputs_shape])))
            self.rec_enc_w = tf.get_variable("rec_enc_w",
                                             shape=[self.sp_hidden_shape, self.rec_hidden_shape],
                                             initializer=(tf.random_normal([self.sp_hidden_shape, self.rec_hidden_shape],
                                                                           stddev=0.35)))
            self.rec_enc_b = tf.get_variable("rec_enc_b",
                                             shape=[self.rec_hidden_shape],
                                             initializer=(tf.zeros([self.rec_hidden_shape])))
            self.rec_dec_w = tf.get_variable("rec_dec_w",
                                             shape=[self.rec_hidden_shape, self.sp_hidden_shape],
                                             initializer=(tf.random_normal([self.rec_hidden_shape, self.sp_hidden_shape],
                                                                           stddev=0.35)))
            self.rec_dec_b = tf.get_variable("rec_dec_b",
                                             shape=[self.sp_hidden_shape],
                                             initializer=(tf.zeros([self.sp_hidden_shape])))

            # tensorflow OP spatial hidden
            self.sp_hidden = tf.nn.sigmoid(tf.matmul(self.layer_input, self.sp_enc_w) + self.sp_enc_b,
                                           name="{0}/sp_hidden".format(self.name))
            SummaryHelpers.activation_summary(self.sp_hidden)

            # tensorflow OP spatial reconstruction
            self.sp_reconstruct = tf.nn.sigmoid(tf.matmul(self.sp_hidden, self.sp_dec_w) + self.sp_dec_b,
                                                name="/{0}sp_reconstruct".format(self.name))
            SummaryHelpers.activation_summary(self.sp_reconstruct)

            # spatial autoencoder reconstruction loss
            self.sp_reconstruct_loss = tf.reduce_mean(tf.square(self.sp_reconstruct - self.layer_input),
                                                      name="{0}/sp_reconstruct_loss".format(self.name))
            tf.scalar_summary(self.sp_reconstruct_loss.op.name, self.sp_reconstruct_loss)

            # spatial autoencoder gradient optimization over reconstruction loss
            self.sp_reconstruct_train_op = tf.train\
                .GradientDescentOptimizer(self.learning_rate)\
                .minimize(self.sp_reconstruct_loss)

            # tensorflow OP recurrent hidden activation
            self.rec_hidden = tf.nn.sigmoid(tf.matmul(self.sp_hidden_prev, self.rec_enc_w) + self.rec_enc_b,
                                            name="{0}/rec_hidden".format(self.name))
            SummaryHelpers.activation_summary(self.rec_hidden)

            # tensorflow OP recurrent reconstruction
            self.rec_reconstruct = tf.nn.sigmoid(tf.matmul(self.rec_hidden, self.rec_dec_w) + self.rec_dec_b,
                                                 name="{0}/rec_reconstruct".format(self.name))
            SummaryHelpers.activation_summary(self.rec_reconstruct)

            # recurrent autoencoder reconstruction loss
            self.rec_reconstruct_loss = tf.reduce_mean(tf.square(self.rec_reconstruct - self.sp_hidden),
                                                       name="{0}/rec_reconstruct_loss".format(self.name))
            tf.scalar_summary(self.rec_reconstruct_loss.op.name, self.rec_reconstruct_loss)

            # recurrent autoencoder gradient optimization over reconstruction loss
            self.rec_reconstruct_train_op = tf.train\
                .GradientDescentOptimizer(self.learning_rate)\
                .minimize(self.rec_reconstruct_loss)

    def run(self, sess, inputs):

        self.layer_input = inputs
        sess.run(self.sp_reconstruct_train_op)
        with sess.graph_def.control_dependencies([self.sp_reconstruct_train_op]):
            sess.run(self.rec_reconstruct_train_op)
        layer_output = tf.nn.sigmoid(self.sp_hidden + self.rec_reconstruct,
                                     name="{0}/layer_output".format(self.name))
        SummaryHelpers.activation_summary(layer_output)

        self.sp_hidden_prev = tf.identity(self.sp_hidden)
        return layer_output



