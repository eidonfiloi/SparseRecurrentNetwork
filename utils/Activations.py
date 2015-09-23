from scipy.special import expit
import numpy as np
__author__ = 'ptoth'


class Activations(object):

    @staticmethod
    def sigmoid_derivative(x):
        return x*(1.0-x)

    @staticmethod
    def tanh_derivative(x):
        return 1.0 - x*x

    @staticmethod
    def rectifier_derivative(x):
        res = (x > 0.0).astype('int')
        return res

    @staticmethod
    def softplus_derivative(x):
        return expit(x)

    @staticmethod
    def softmax_derivative(x):
        return x*(1.0-x)

    @staticmethod
    def sigmoid_activation(x):
        return expit(x)

    @staticmethod
    def rectifier_activation(x):
        return np.maximum(0.0, x)

    @staticmethod
    def tanh_activation(x):
        return np.tanh(x)

    @staticmethod
    def softplus_activation(x):
        return np.log(1.0 + np.exp(x))

    @staticmethod
    def softmax_activation(x):
        vec = np.exp(x)
        return vec / np.sum(vec)

    @staticmethod
    def derivative(x, activation_function="Sigmoid"):
        if activation_function == "Sigmoid":
            return Activations.sigmoid_derivative(x)
        elif activation_function == "Rectifier":
            return Activations.rectifier_derivative(x)
        elif activation_function == "Tanh":
            return Activations.tanh_derivative(x)
        elif activation_function == "SoftPlus":
            return Activations.softplus_derivative(x)
        elif activation_function == "SoftMax":
            return Activations.softmax_derivative(x)
        else:
            return Activations.sigmoid_derivative(x)

    @staticmethod
    def activation(x, activation_function="Sigmoid"):
        if activation_function == "Sigmoid":
            return Activations.sigmoid_activation(x)
        elif activation_function == "Rectifier":
            return Activations.rectifier_activation(x)
        elif activation_function == "Tanh":
            return Activations.tanh_activation(x)
        elif activation_function == "SoftPlus":
            return Activations.softplus_activation(x)
        elif activation_function == "SoftMax":
            return Activations.softmax_activation(x)
        else:
            return Activations.sigmoid_activation(x)


