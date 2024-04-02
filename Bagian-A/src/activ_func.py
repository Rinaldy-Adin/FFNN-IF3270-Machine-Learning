from enum import Enum
import numpy as np
import math


class Activation_Function(Enum):
    """
    Activation Function Enums
    """

    LINEAR = 1
    RELU = 2
    SIGMOID = 3
    SOFTMAX = 4


def relu(v: float):
    return max(0, v)


def sigmoid(v: float):
    return 1 / (1 + math.exp(-v))

def softmax(v: float):
    e_v = np.exp(v - np.max(v)) 
    return e_v / e_v.sum(axis=0)


"""
Vectorizing activation functions so
functions can be easily applied to ndarrays
"""
reluVect = np.vectorize(relu)
sigmoidVect = np.vectorize(sigmoid)
softmaxVect = np.vectorize(softmax)
