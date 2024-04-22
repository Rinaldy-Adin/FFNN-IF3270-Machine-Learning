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
    def __init__(self, n_inputs: int, n_classes: int, learning_rate: float, batch_size: int, max_iter: int) -> None:
        self._n_inputs = n_inputs
        self._n_classes = n_classes
        self._batch_size = batch_size
        self._max_iter = max_iter

        self._targets: list[list[float]] = []
        self._input: list[list[float]] = []
        self._layers: list[Layer] = []
        self._batch_grad: list[np.ndarray[float]] = []

        self._current_output: np.ndarray = None

        self._learning_rate = learning_rate

    def init_batch_grad(self):
        self._batch_grad = [np.zeros(shape=(layer.n_inputs, layer.n_neurons)) for layer in self._layers]

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
        for iter in range(self._max_iter):
            self.init_batch_grad()
            for idx, cur_input in enumerate(self._input):
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

                target = self._targets[idx]
                self._current_output = current
                self.backwards_propagation(layer_inputs, layer_nets, target, iter, idx)

                if (idx + 1) % self._batch_size == 0 or idx + 1 == len(self._input):
                    self.update_weights()
                    # print(f"grads {iter}:")
                    # print(self._batch_grad[0])
                    # print()
                    self.init_batch_grad()

    def update_weights(self):
        for idx in range(len(self._layers)):
            # print(f"layer {idx}")
            # print(self._batch_grad[idx])
            # print()
            self._layers[idx].w += self._batch_grad[idx]

    def update_batch_grad(self, layer_idx: int, delta: np.ndarray, layer_input: np.ndarray):
        grad = layer_input * delta * self._learning_rate

        # print(f"layer {layer_idx}")
        # print(grad)
        # print()

        self._batch_grad[layer_idx] += grad

    def backwards_propagation(self, layer_inputs: list[list[float]], layer_nets: list[list[float]], target: list[float], iter: int, input_idx: int):
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
                target_mat = np.array(target).transpose()

                if layer.activ_func == Activation_Function.SOFTMAX:
                    ds_delta = delta_softmax_output(self._current_output, target_mat)
                elif layer.activ_func == Activation_Function.RELU:
                    ds_delta = delta_relu_output(self._current_output, target_mat, nets)
                elif layer.activ_func == Activation_Function.SIGMOID:
                    ds_delta = delta_sigmoid_output(self._current_output, target_mat)
                else:
                    ds_delta = delta_linear_output(self._current_output, target_mat)

                # print(self._layers[layer_idx].activ_func)
                # print(f"nets {input_idx} {layer_idx}:")
                # print(nets)
                # print()
                # print(f"target_mat {input_idx} {layer_idx}:")
                # print(target_mat)
                # print()

                # print(f"delta {input_idx} {layer_idx}:")
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
                    cur_delta = delta_sigmoid_hidden(layer_outputs, ds_delta, self._layers[layer_idx + 1].w)
                else:
                    cur_delta = delta_linear_hidden(ds_delta, self._layers[layer_idx + 1].w)

                # print("cur_delta")
                # print(cur_delta)
                # print()

                ds_layer_input = np.array(layer_inputs[layer_idx + 1]).transpose()
                self.update_batch_grad(layer_idx + 1, ds_delta,ds_layer_input)
                ds_delta = cur_delta

        ds_layer_input = np.array(layer_inputs[0]).transpose()
        self.update_batch_grad(0, ds_delta, ds_layer_input)

if __name__ == "__main__":
    with open('../models/mlp_sigmoid.json', 'r') as f:
        json_data = json.load(f)

    # Extract data from JSON
    input_size = json_data['case']['model']['input_size']
    input_data = np.array(json_data['case']['input'])
    target_data = np.array(json_data['case']['target'])
    learning_rate = json_data['case']['learning_parameters']['learning_rate']
    initial_weights = [np.array(layer) for layer in json_data['case']['initial_weights']]
    n_classes = json_data['case']['model']['layers'][-1]['number_of_neurons']
    batch_size = json_data['case']['learning_parameters']['batch_size']
    max_iter = json_data['case']['learning_parameters']['max_iteration']
    layer_config = json_data['case']['model']['layers']

    fnaf = FFNN(input_size, n_classes, learning_rate, batch_size, max_iter)

    for idx, input in enumerate(input_data):
        fnaf.addInput(input, target_data[idx])

    for idx, w in enumerate(initial_weights):
        if layer_config[idx]['activation_function'] == "linear":
            fnaf.addLayer(Layer(w, Activation_Function.LINEAR))
        elif layer_config[idx]['activation_function'] == "relu":
            fnaf.addLayer(Layer(w, Activation_Function.RELU))
        elif layer_config[idx]['activation_function'] == "sigmoid":
            fnaf.addLayer(Layer(w, Activation_Function.SIGMOID))
        else:
            fnaf.addLayer(Layer(w, Activation_Function.SOFTMAX))
    
    fnaf.feed_forward()
    
    for idx, layer in enumerate(fnaf._layers):
        print(f"layer {idx}")
        print(layer.w)
        print()