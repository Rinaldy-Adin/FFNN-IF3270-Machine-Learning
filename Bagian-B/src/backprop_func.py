import numpy as np

def d_relu(v: float, net: float):
    return 0 if net < 0 else v

d_relu_vect = np.vectorize(d_relu)

def d_softmax(o: float, net: float):
    return -o if net == 1.0 else 1-0

d_softmax_vect = np.vectorize(d_softmax)

"""
All variables output, target, nets are in the form of
single column matrix
"""

def delta_linear_output(output: np.ndarray, target: np.ndarray, n_inputs: int):
    output_mat = np.tile(np.transpose(output), reps=(n_inputs, 1))
    target_mat = np.tile(np.transpose(target), reps=(n_inputs, 1))
    return target_mat - output_mat

def delta_relu_output(output: np.ndarray, target: np.ndarray, nets: np.ndarray, n_inputs: int):
    output_mat = np.tile(np.transpose(output), reps=(n_inputs, 1))
    target_mat = np.tile(np.transpose(target), reps=(n_inputs, 1))
    nets_mat = np.tile(np.transpose(nets), reps=(n_inputs, 1))

    return d_relu_vect(target_mat - output_mat, nets_mat)

def delta_sigmoid_output(output: np.ndarray, target: np.ndarray, n_inputs: int):
    output_mat = np.tile(np.transpose(output), reps=(n_inputs, 1))
    target_mat = np.tile(np.transpose(target), reps=(n_inputs, 1))
    
    return (target_mat - output_mat) * output_mat * (1 - output_mat)

def delta_softmax_output(output: np.ndarray, target: np.ndarray, n_inputs: int):
    output_mat = np.tile(np.transpose(output), reps=(n_inputs, 1))
    target_mat = np.tile(np.transpose(target), reps=(n_inputs, 1))

    return d_softmax_vect(output_mat, target_mat)

def delta_linear_hidden(
    ds_delta: np.ndarray, ds_w: np.ndarray, n_inputs: int
):
    sigma_vect = np.array([[np.sum(ds_delta[row_idx] * ds_w[row_idx]) for row_idx in range(ds_w.shape[0])]])

    return np.tile(sigma_vect, reps=(n_inputs, 1))

def delta_relu_hidden(
    nets: np.ndarray, ds_delta: np.ndarray, ds_w: np.ndarray, n_inputs: int
):
    nets_vect = np.transpose(nets)
    # TODO check if this handles bias correctly
    sigma_vect = np.array([[np.sum(ds_delta[row_idx] * ds_w[row_idx]) for row_idx in range(ds_w.shape[0])]])

    return np.tile(d_relu_vect(sigma_vect, nets_vect), reps=(n_inputs, 1))


def delta_sigmoid_hidden(
    output: np.ndarray, ds_delta: np.ndarray, ds_w: np.ndarray, n_inputs: int
):
    output_vect = np.transpose(output)
    output_vect = output_vect * (1 - output_vect)

    sigma_vect = np.array([[np.sum(ds_delta[row_idx] * ds_w[row_idx]) for row_idx in range(ds_w.shape[0])]])

    return np.tile(output_vect * sigma_vect, reps=(n_inputs, 1))

def delta_softmax_hidden(
    output: np.ndarray, ds_delta: np.ndarray, ds_w: np.ndarray, n_inputs: int
):
    output_deltas = []
    o_list: list[float] = np.transpose(output).tolist()[0]

    ds_sums: list[float] = [np.sum(ds_delta[j] * ds_w[j]) for j in range(len(ds_delta))]

    for i, oi in enumerate(o_list):
        o_sum = 0
        for j, oj in enumerate(o_list):
            if (i == j):
                do_dnet = oi * (1 - oj)
            else:
                do_dnet = -oi * oj
            o_sum += do_dnet * ds_sums[j]
        output_deltas.append(o_sum)
    
    return np.tile(np.array([output_deltas]), reps=(n_inputs, 1))
