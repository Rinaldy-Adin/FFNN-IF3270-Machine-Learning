import csv
from matplotlib import pyplot as plt
import numpy as np
import json
import pickle
import copy
from activ_func import Activation_Function, reluVect, sigmoidVect, softmax
from backprop_func import delta_linear_output, delta_relu_output, delta_sigmoid_output, delta_softmax_output, delta_linear_hidden, delta_relu_hidden, delta_sigmoid_hidden, delta_softmax_hidden

class Layer:
    def __init__(self, w: np.ndarray[float], activ_func: Activation_Function) -> None:
        if (w.ndim != 2):
            raise RuntimeError("Layer initialized with non 2-dimensional array")

        self.w = w
        self.n_inputs = w.shape[0]
        self.n_neurons = w.shape[1]
        self.activ_func = activ_func


class FFNN:
    def __init__(self, n_inputs: int, n_classes: int, learning_rate: float) -> None:
        self._n_inputs = n_inputs
        self._n_classes = n_classes

        self._targets: list[list[float]] = []
        self._input: list[list[float]] = []
        self._layers: list[Layer] = []

        self._current_output: np.ndarray = None

        self._learning_rate = learning_rate

    def get_output(self):
        return np.transpose(self._current_output).tolist()

    def addInput(self, newInput: list[float], target_output: list[float]):
        if self._n_inputs != len(newInput):
            raise RuntimeError("Added input with incorrect number of attributes")
        if self._n_classes != len(target_output):
            raise RuntimeError("Added target with incorrect number of classes")

        self._input.append(newInput)
        self._targets.append(target_output)

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

    def feed_forward(self):
        for cur_input in self._input:
            layer_inputs: list[list[float]] = []
            layer_nets: list[list[float]] = []

            current = np.transpose(np.array([cur_input]))
            bias = np.array([[1.0]])

            for _, layer in enumerate(self._layers):
                current = np.concatenate((bias, current), axis=0)
                layer_inputs.append(current.copy().transpose().tolist())

                new_current = np.transpose(layer.w) @ current
                current = new_current
                layer_nets.append(current.copy().transpose().tolist())

                if layer.activ_func == Activation_Function.RELU:
                    current = reluVect(current)
                elif layer.activ_func == Activation_Function.SIGMOID:
                    current = sigmoidVect(current)
                elif layer.activ_func == Activation_Function.SOFTMAX:
                    current = softmax(current)

            self._current_output = current
            self.backwards_propagation(layer_inputs, layer_nets)

    def update_w(self, layer_idx: int, delta: np.ndarray, inputs: list[float]):
        grad = delta * self._learning_rate

        # print("delta:")
        # print(delta)
        # print()

        # print("grad:")
        # print(grad)
        # print()

        self._layers[layer_idx].w += grad

    def backwards_propagation(self, layer_inputs: list[list[float]], layer_nets: list[list[float]]):
        # print("Inputs:")
        # print(layer_inputs)

        # print()
        # print("nets:")
        # print(layer_nets)
        # print()

        ds_delta: np.ndarray = None
        for idx, layer in enumerate(reversed(self._layers)):
            layer_idx = (-1-idx) % len(self._layers)
            nets = np.array(layer_nets[layer_idx]).transpose()

            if idx == 0:
                target = np.array(self._targets[layer_idx]).transpose()
                cur_layer_input = np.array(layer_inputs[layer_idx]).transpose()

                if layer.activ_func == Activation_Function.SOFTMAX:
                    ds_delta = delta_softmax_output(self._current_output, target, layer.n_inputs)
                elif layer.activ_func == Activation_Function.RELU:
                    ds_delta = delta_relu_output(self._current_output, target, nets, layer.n_inputs)
                elif layer.activ_func == Activation_Function.SIGMOID:
                    ds_delta = delta_sigmoid_output(self._current_output, target, layer.n_inputs)
                else:
                    ds_delta = delta_linear_output(self._current_output, target, cur_layer_input)

                # print("delta:")
                # print(ds_delta)
                # print()
            else:
                cur_delta = None
                layer_outputs = np.array(layer_inputs[layer_idx + 1][1:])

                if layer.activ_func == Activation_Function.SOFTMAX:
                    cur_delta = delta_softmax_hidden(layer_outputs, ds_delta, self._layers[layer_idx + 1].w, layer.n_inputs)
                elif layer.activ_func == Activation_Function.RELU:
                    cur_delta = delta_relu_hidden(nets, ds_delta, self._layers[layer_idx + 1].w, layer.n_inputs)
                elif layer.activ_func == Activation_Function.SIGMOID:
                    cur_delta = delta_sigmoid_hidden(layer_outputs, ds_delta, self._layers[layer_idx + 1].w, layer.n_inputs)
                else:
                    cur_delta = delta_linear_hidden(ds_delta, self._layers[layer_idx + 1].w, layer.n_inputs)

                self.update_w(layer_idx + 1, ds_delta, layer_inputs[layer_idx + 1])
                ds_delta = cur_delta

        self.update_w(0, ds_delta, layer_inputs[0])

# if __name__ == "__main__":
#     n_attr = 4
#     n_classes = 3
#     learning_rate = 0.2

#     fnaf = FFNN(n_attr, n_classes, learning_rate)

#     data = []

#     with open('../data/iris.csv', newline='') as f:
#         reader = csv.reader(f)
#         data = list(reader)[1:]

#     for row in data[:50]:
#         inputs = list(map(float, row[1:5]))

#         if row[5] == "Iris-setosa":
#             target = [1.0,0.0,0.0]
#         elif row[5] == "Iris-versicolor":
#             target = [0.0,1.0,0.0]
#         else:
#             target = [0.0,0.0,1.0]
        
#         fnaf.addInput(inputs, target)

#     w_hidden = np.random.uniform(-0.5, 0.5, size=(5, 4))
#     w_out = np.random.uniform(-0.5, 0.5, size=(5, 3))

#     layer_hidden = Layer(w_hidden, Activation_Function.RELU)
#     layer_out = Layer(w_out, Activation_Function.SIGMOID)

#     fnaf.addLayer(layer_hidden)
#     fnaf.addLayer(layer_out)

#     fnaf.feed_forward()

#     print(fnaf.get_output())

if __name__ == "__main__":
    with open('../models/linear_1.json', 'r') as f:
        json_data = json.load(f)

    # Extract data from JSON
    input_size = json_data['case']['model']['input_size']
    input_data = np.array(json_data['case']['input'])
    target_data = np.array(json_data['case']['target'])
    learning_rate = np.array(json_data['case']['learning_parameters']['learning_rate'])
    initial_weights = [np.array(layer) for layer in json_data['case']['initial_weights']]

    fnaf = FFNN(input_size, 3, learning_rate)

    for idx, input in enumerate(input_data):
        fnaf.addInput(input, target_data[idx])

    for w in initial_weights:
        fnaf.addLayer(Layer(w, Activation_Function.LINEAR))
    
    fnaf.feed_forward()
    
    for layer in fnaf._layers:
        print(layer.w)