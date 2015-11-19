from Layer import *

__author__ = 'ptoth'


class Network(object):
    """

    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, parameters):
        self.parameters = parameters
        self.name = self.parameters['name']
        self.loss_function = self.parameters['loss_function']

    @abc.abstractmethod
    def run(self, sess, inputs):
        """

        :param sess:
        :param inputs:
        :return:
        """


class SRNetwork(Network):
    """

    """

    def __init__(self, parameters):
        super(SRNetwork, self).__init__(parameters)
        self.layers = [SRLayer(layer_conf) for layer_conf in self.parameters['layers']]
        self.layers_output_shape = self.parameters['layers_output_shape']
        self.output_shape = self.parameters['output_shape']
        self.learning_rate = self.parameters['learning_rate']

        with tf.variable_scope(self.name):

            self.out_w = tf.Variable(tf.random_normal([self.layers_output_shape, self.output_shape], stddev=0.35),
                                     name="out_w")

            self.out_b = tf.Variable(tf.random_normal([self.output_shape], stddev=0.35),
                                     name="out_b")

    def run(self, sess, inputs, targets=None, summary_op=None, summary_writer=None):

        with tf.variable_scope(self.name):
            current_input = None
            current_target = None
            layer_states = [layer.initial_state(inputs.shape[1]) for layer in self.layers]
            for idx, sample in enumerate(inputs):
                if targets is not None:
                    current_target = targets[idx]
                if current_target is not None:
                    current_input = sample.astype('float32')
                    new_states = []
                    for idxx, layer in enumerate(self.layers):
                        current_input, current_layer_state = layer.run(sess, current_input, layer_states[idxx])
                        new_states.append(current_layer_state)
                    layer_states = new_states
                    output = tf.nn.sigmoid(tf.matmul(current_input, self.out_w) + self.out_b,
                                           name="output")
                    SummaryHelpers.activation_summary(output)

                    # output LOSS
                    output_loss = tf.reduce_mean(tf.square(output - current_target),
                                                 name="output_loss")
                    tf.scalar_summary(output_loss.op.name, output_loss)

                    # TensorFlow OP for train output optimization
                    output_train_op = tf.train\
                        .GradientDescentOptimizer(self.learning_rate)\
                        .minimize(output_loss)
                    _, loss = sess.run([output_train_op, output_loss])
                    assert not np.isnan(loss), 'Model diverged with loss = NaN'
                    if summary_op is not None and summary_writer is not None:
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str)
                    print(loss)




