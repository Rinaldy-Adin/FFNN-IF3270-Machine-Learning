import numpy as np

def d_relu(v: float, net: float):
    print(0 if net < 0 else v)
    return 0 if net < 0 else v

def d_relu_vect(v: np.ndarray, net: np.ndarray):
    res = np.empty_like(v)
    for row_idx, row in enumerate(v):
        for col_idx, val in enumerate(row):
            res[row_idx][col_idx] = 0 if net[row_idx][col_idx] < 0 else val
    return res

def d_softmax(o: float, net: float):
    return 1-o if net == 1.0 else -o

def d_softmax_vect(o: np.ndarray, target: np.ndarray):
    res = np.empty_like(o)
    for row_idx, row in enumerate(o):
        for col_idx, o_val in enumerate(row):
            res[row_idx][col_idx] = 1-o_val if target[col_idx] == 1.0 else -o_val
    return res

"""
All variables output, target, nets, layer_input are in the form of
single column matrix
"""

def delta_linear_output(output: np.ndarray, target: np.ndarray):
    output_mat = np.transpose(output)
    target_mat = np.transpose(target)

    return (target_mat - output_mat)

def delta_relu_output(output: np.ndarray, target: np.ndarray, nets: np.ndarray):
    output_mat = np.transpose(output)
    target_mat = np.transpose(target)
    nets_mat = np.transpose(nets)

    res = d_relu_vect(target_mat - output_mat, nets_mat)

    return res

def delta_sigmoid_output(output: np.ndarray, target: np.ndarray):
    output_mat = np.transpose(output)
    target_mat = np.transpose(target)
    
    return ((target_mat - output_mat) * output_mat * (1 - output_mat))

def delta_softmax_output(output: np.ndarray, target: np.ndarray):
    output_mat = np.transpose(output)
    target_mat = np.transpose(target)

    return d_softmax_vect(output_mat, target_mat)

def delta_linear_hidden(
    ds_delta: np.ndarray, ds_w: np.ndarray
):
    sigma_delta_w = ds_delta * ds_w
    sigma_list = np.sum(sigma_delta_w, axis=1).transpose()

    return np.array([sigma_list[1:]])

def delta_relu_hidden(
    nets: np.ndarray, ds_delta: np.ndarray, ds_w: np.ndarray, n_inputs: int
):
    sigma_delta_w = ds_delta * ds_w
    sigma_list = np.sum(sigma_delta_w, axis=1).transpose()
    sigma_nets = np.array([sigma_list[1:]])

    nets_mat = np.transpose(nets)
    res = d_relu_vect(sigma_nets, nets_mat)

    return res


# TODO fix
def delta_sigmoid_hidden(
    output: np.ndarray, ds_delta: np.ndarray, ds_w: np.ndarray, n_inputs: int
):
    output_vect = np.transpose(output)
    output_vect = output_vect * (1 - output_vect)

    sigma_vect = np.array([[np.sum(ds_delta[row_idx] * ds_w[row_idx]) for row_idx in range(ds_w.shape[0])]])

    return np.tile(output_vect * sigma_vect, reps=(n_inputs, 1))

# TODO fix
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
