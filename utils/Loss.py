import numpy as np
from math import sqrt
from Activations import *

__author__ = 'ptoth'


class Loss(object):

    @staticmethod
    def delta(pred, target, loss_function="MSE"):
        if loss_function == "MSE":
            return pred - target
        elif loss_function == "CROSS_ENTROPY":
            return -1.0 * (target / pred)
        else:
            return pred - target

    @staticmethod
    def delta_backpropagate(pred, target, loss_function="MSE", activation_function="Sigmoid"):

        if loss_function == "CROSS_ENTROPY" and activation_function == "SoftMax":
            return pred - target
        else:
            return Loss.delta(pred, target, loss_function) * Activations.derivative(pred, activation_function)

    @staticmethod
    def error(pred, target, loss_function="MSE"):
        if loss_function == "MSE":
            return sqrt(np.mean(np.abs(Loss.delta(pred, target, loss_function)) ** 2, axis=0))
        elif loss_function == "CROSS_ENTROPY":
            return -np.sum(np.log(pred)*target)
        else:
            return sqrt(np.mean(np.abs(Loss.delta(pred, target, loss_function)) ** 2, axis=0))


