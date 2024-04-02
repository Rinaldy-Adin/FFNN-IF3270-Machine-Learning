from ffnn import FFNN, Layer
import numpy as np
from activ_func import Activation_Function

def difficult_test():
    """
    Example run
    """
    ffnn = FFNN(3)

    input_1 = [-1.0, 0.5, 0.8]
    ffnn.addInput(input_1)

    input_2 = [0.1, 0.2, 0.3]
    ffnn.addInput(input_2)

    layer_1_data = np.array(
        [
            [0.1, 0.2, 0.3, -1.2],
            [-0.5, 0.6, 0.7, 0.5],
            [0.9, 1.0, -1.1, -1.0],
            [1.3, 1.4, 1.5, 0.1],
        ]
    )
    layer_2_data = np.array(
        [
            [0.1, 0.1, 0.3],
            [-0.4, 0.5, 0.6],
            [0.7, 0.4, -0.9],
            [0.2, 0.3, 0.4],
            [-0.1, 0.2, 0.1],
        ]
    )
    layer_3_data = np.array([[0.1, 0.2], [-0.3, 0.4], [0.6, 0.1], [0.1, -0.4]])
    layer_4_data = np.array([[0.1], [-0.2], [0.3]])

    layer_1 = Layer(layer_1_data, Activation_Function.RELU)
    layer_2 = Layer(layer_2_data, Activation_Function.RELU)
    layer_3 = Layer(layer_3_data, Activation_Function.RELU)
    layer_4 = Layer(layer_4_data, Activation_Function.SIGMOID)

    ffnn.addLayer(layer_1)
    ffnn.addLayer(layer_2)
    ffnn.addLayer(layer_3)
    ffnn.addLayer(layer_4)

    print(ffnn.run())
    
    ffnn.draw_network()


difficult_test()
