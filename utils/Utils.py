__author__ = 'ptoth'


class Utils(object):

    @staticmethod
    def sigmoid_derivative(x):
        return x*(1.0-x)

    @staticmethod
    def tanh_derivative(x):
        return (1.0 - x)*(1.0 + x)

    @staticmethod
    def rectifier_derivative(x):
        return x

    @staticmethod
    def derivative(x, activation_function="Sigmoid"):
        if activation_function == "Sigmoid":
            return Utils.sigmoid_derivative(x)
        elif activation_function == "Rectifier":
            return Utils.rectifier_derivative(x)
        elif activation_function == "Tanh":
            return Utils.tanh_derivative(x)
        else:
            return 1.0
