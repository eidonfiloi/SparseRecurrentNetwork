import tensorflow as tf

__author__ = 'ptoth'


class SummaryHelpers(object):
    """

    """
    @staticmethod
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
