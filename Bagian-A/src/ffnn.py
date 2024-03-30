import numpy as np
from activ_func import Activation_Function, reluVect, sigmoidVect

"""
Layer object class
"""
class Layer:
    def __init__(self, w: np.ndarray[float], activ_func: Activation_Function) -> None:
        if (w.ndim != 2):
            raise RuntimeError("Layer initialized with non 2-dimensional array")

        self.w = w
        self.n_inputs = w.shape[0]
        self.n_neurons = w.shape[1]
        self.activ_func = activ_func


"""
Feed Forward Neural Network object class
"""
class FFNN:
    def __init__(self, input_size: int) -> None:
        self._input_size = input_size
        self._input: list[list[float]] = []
        self._layers: list[Layer] = []

    def addInput(self, newInput: list[float]):
        if self._input_size != len(newInput):
            raise RuntimeError("Added input that is different from input_size")

        self._input.append(newInput)

    def addLayer(self, newLayer: Layer):
        if len(self._input) == 0:
            raise RuntimeError("Input not defined before adding hidden layer")

        if (
            len(self._layers) == 0 and (len(self._input[0]) + 1) != newLayer.n_inputs
        ) or (len(self._layers) != 0 and (self._layers[-1].n_neurons + 1) != newLayer.n_inputs):
            raise RuntimeError(
                "Number of inputs in layer matrix does not match output from previous layer"
            )

        self._layers.append(newLayer)

    def run(self):
        """
        Works by iterating through each layer's weight matrix
        
        net = wT * x
        wT = current layer weight transposed
        x = current input/hidden layer results

        For each iteration, x will have bias (1.0) values appended 
        """
        current = np.array(self._input).transpose()
        bias = np.array([[1.0 for _ in self._input]])

        for layer in self._layers:
            current = np.concatenate((bias, current), axis=0)

            new_current = layer.w.transpose() @ current
            current = new_current
            
            if layer.activ_func == Activation_Function.RELU:
                current = reluVect(current)
            elif layer.activ_func == Activation_Function.SIGMOID:
                current = sigmoidVect(current)

        return current.transpose()
