from enum import Enum
import numpy as np
import math


class Activation_Function(Enum):
    """
    Activation Function Enums
    """

    LINEAR = "linear"
    RELU = "relu"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"


def reluElmt(v: float):
    return max(0.0, v)

def sigmoidElmt(v: float):
    return 1.0 / (1.0 + math.exp(-v))

def softmaxElmt(v: float, expSum: float):
    e_v = np.exp(v) 
    return e_v / expSum


"""
Vectorizing activation functions so
functions can be easily applied to ndarrays
"""
reluVect = np.vectorize(reluElmt)
sigmoidVect = np.vectorize(sigmoidElmt)
softmaxVect = np.vectorize(softmaxElmt)

def softmax(mat: np.ndarray):
    mat = mat.transpose()
    new_mat = np.empty((0,mat.shape[1]))
    for row_idx in range(0, mat.shape[0]):
        row = mat[row_idx]
        row_exp_sum = np.sum(np.exp(row))
        sm = softmaxVect(row, row_exp_sum)
        new_mat = np.append(new_mat, np.array([sm]), axis=0)
    return new_mat.transpose()